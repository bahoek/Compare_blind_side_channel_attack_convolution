import numpy as np
import os
import time
import math
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# =========================================================================
# 📦 1. Global Adversarial Configuration & Environment Setup
# =========================================================================
BATCH_SIZE = 128  
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEST_FILE = r"./data/measured_test_precompiled.npz"
MODEL_PATH = r"./models/kyber_ASC.pth"
EPS_SAVE_PATH = r"./data/guessing_entropy_EFF.eps"

CANDIDATES = [-2, -1, 0, 1, 2] 

# =========================================================================
# 🧠 2. Modern 1D ASCAD Multi-Head Architecture (26-Axis Parallel Taxonomy)
# =========================================================================
class ASCAD(nn.Module):
    def __init__(self, num_targets=26): 
        super(ASCAD, self).__init__()
        self.num_targets = num_targets
        
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=65, stride=2, padding=32),  # 40000 -> 20000
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Conv1d(64, 128, kernel_size=33, stride=2, padding=16), # 20000 -> 10000
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Conv1d(128, 256, kernel_size=17, stride=2, padding=8),  # 10000 -> 5000
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
        )
        
        self.shared_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(5000 * 256, 1024), 
            nn.ReLU(),
            nn.Dropout(0.4)
        )
        
        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(1024, 256) # Emits 256 probabilities for exact numeric states
            ) for _ in range(num_targets)
        ])

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.shared_fc(x) 
        outputs = [clf(x) for clf in self.classifiers]
        return outputs


# =========================================================================
# 🚀 3. Execution Controller & Scholarly Visualization Pipeline
# =========================================================================
if __name__ == "__main__":
    print(f"🖥️ Execution Target Device: {DEVICE}")
    
    if not os.path.exists(TEST_FILE):
        print("⚠️ [Data Notice] Precompiled test matrix file not found. Please place 'measured_test_precompiled.npz'.")
        
    print(f"📂 [TEST SET] Loading Pre-compiled Evaluation Matrices...")
    test_data = np.load(TEST_FILE)
    traces = test_data['traces']
    labels = test_data['labels']      # Pre-compiled Matrix mapped across candidate spaces
    true_secret_keys = test_data['true_keys'] # Target ground truth vectors
    
    MAX_ATTACK_TRACES = len(traces) 
    
    model = ASCAD(num_targets=26).to(DEVICE)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("✅ 26-Axis Multi-Head Measured Model Weights Loaded Successfully.")
    model.eval()
    
    print(f"🔮 Initializing Multi-Head 1st-Stage Deep Learning Fast Scan...")
    pred_probabilities = [[] for _ in range(26)]
    
    with torch.no_grad():
        for i in tqdm(range(0, MAX_ATTACK_TRACES, 128), desc="⚡ Network Inference"):
            batch_x_np = traces[i:i+128]
            X_tensor = torch.from_numpy(batch_x_np).float()
            # Standard Z-Score normalization framework
            X_tensor = (X_tensor - X_tensor.mean(dim=1, keepdim=True)) / (X_tensor.std(dim=1, keepdim=True) + 1e-12)
            X_tensor = X_tensor.unsqueeze(1).to(DEVICE)
            
            outputs = model(X_tensor)
            for idx in range(26):
                probs = torch.softmax(outputs[idx], dim=1).cpu().numpy()
                pred_probabilities[idx].append(probs)
                
    for idx in range(26):
        pred_probabilities[idx] = np.vstack(pred_probabilities[idx])
        
    print("\n⚔️ Processing Bayesian Log-Likelihood Accumulator & Filtering Vector...")
    log_likelihoods = np.zeros((26, len(CANDIDATES)))
    ge_trends = np.zeros((MAX_ATTACK_TRACES, 26))
    sr_trends = np.zeros(MAX_ATTACK_TRACES)
    
    for t in tqdm(range(MAX_ATTACK_TRACES), desc="📈 Sequential Convergence"):
        for idx in range(26):
            true_key = true_secret_keys[t, idx]
            if true_key not in CANDIDATES:
                true_key = 0
            
            for c_idx, k_cand in enumerate(CANDIDATES):
                calc_upper = labels[t, idx, c_idx]
                prob_val = pred_probabilities[idx][t, calc_upper]
                log_likelihoods[idx, c_idx] += np.log(prob_val + 1e-15)
                
            sorted_indices = np.argsort(log_likelihoods[idx])[::-1]
            true_cand_idx = CANDIDATES.index(true_key)
            current_rank = np.where(sorted_indices == true_cand_idx)[0][0]
            ge_trends[t, idx] = current_rank
            
        sr_trends[t] = np.sum(ge_trends[t, :] == 0) / 26
        
    final_ge = np.mean(ge_trends[-1, :])
    final_sr = sr_trends[-1] * 100
    
    print(f"\n🏆 [Empirical Blind Attack Verification Performance Summary]")
    print(f"   -> Consolidated Framework Success Rate (SR) : {final_sr:.2f}%")
    print(f"   -> Mean Global Guessing Entropy (GE) Target Plane : {final_ge:.2f} (0-indexed)")
    
    # Generate High-Resolution Vector EPS Graphic Profile for Journal Submission
    print("🎨 Exporting High-Resolution Scholarly Standard Vector Graphics (.eps)...")
    mean_ge = np.mean(ge_trends, axis=1)
    
    upper_bound = np.percentile(ge_trends, 90, axis=1)
    lower_bound = np.percentile(ge_trends, 10, axis=1)
    
    x_axis = range(1, MAX_ATTACK_TRACES + 1)
    
    for idx in range(26):
        plt.plot(x_axis, ge_trends[:, idx], alpha=0.04, linewidth=0.5, color='gray')
        
    plt.fill_between(x_axis, lower_bound, upper_bound, color='royalblue', alpha=0.15, label='90% Axis Distribution')
    plt.plot(x_axis, mean_ge, label='Mean Guessing Entropy', color='crimson', linewidth=2.5)
    
    plt.xlabel('Number of Traces', fontsize=12)
    plt.ylabel('Guessing Entropy (0-indexed)', fontsize=12)
    plt.title('20K Traces Blind SCA Key Convergence Profile', fontsize=14, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.yticks(range(5)) 
    plt.ylim(-0.2, 4.2)
    plt.legend(fontsize=11, loc='upper right')
    
    eps_dir = os.path.dirname(EPS_SAVE_PATH)
    if not os.path.exists(eps_dir): os.makedirs(eps_dir, exist_ok=True)
    plt.savefig(EPS_SAVE_PATH, format='eps', bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"💾 Vector visualization saved successfully ➔ {EPS_SAVE_PATH}")