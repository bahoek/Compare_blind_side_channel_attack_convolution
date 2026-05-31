import numpy as np
import os
import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast

# =========================================================================
# 📦 1. Multi-Target Side-Channel Dataset Loader Interface (IP Protected)
# =========================================================================
class SCAPackageDataset(Dataset):
    """
    Custom Dataset class for loading pre-compiled multi-target cryptographic traces.
    
    NOTE: To protect proprietary research infrastructure, algorithmic zetas mapping,
    and hardware-specific NTT alignment/labeling emulators, this loader assumes 
    that all physical power traces and target 26-axis distinctive coefficient labels 
    have already been pre-calculated, synchronized, and compiled into a 
    standardized matrix format beforehand.
    """
    def __init__(self, npz_path, DEVICE):
        data = np.load(npz_path, mmap_mode='r')
        self.traces = torch.from_numpy(data['traces']).float().to(DEVICE)
        self.labels = torch.from_numpy(data['labels']).long().to(DEVICE)
        
    def __len__(self):
        return len(self.traces)

    def __getitem__(self, idx):
        trace = self.traces[idx]
        label = self.labels[idx] 
            
        return trace.unsqueeze(0), label


# =========================================================================
# 🧠 2. Modern 1D ASCAD Multi-Head Architecture (26-Axis Parallel Taxonomy)
# =========================================================================
class ASCAD(nn.Module):
    def __init__(self, num_targets=26): 
        super(ASCAD, self).__init__()
        self.num_targets = num_targets
        
        # Core Feature Extractor
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
        
        # Global Flatten Shared Space
        self.shared_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(5000 * 256, 1024), 
            nn.ReLU(),
            nn.Dropout(0.4)
        )
        
        # 26 Distinctive Sub-Bit Coefficient Classification Heads
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
        
        # Parallel classification routing
        outputs = [clf(x) for clf in self.classifiers]
        return outputs


# =========================================================================
# 🧪 3. Independent Quantitative Metric Evaluation Engine
# =========================================================================
def evaluate_metrics(model, dataloader, device):
    model.eval()
    ge_list = [[] for _ in range(26)]
    total_correct = 0
    total_elements = 0
    val_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            batch_size = batch_x.size(0)
            
            for idx in range(26):
                val_loss += criterion(outputs[idx], batch_y[:, idx]).item()
                sorted_indices = torch.argsort(outputs[idx], dim=1, descending=True)
                preds = torch.argmax(outputs[idx], dim=1)
                
                total_correct += (preds == batch_y[:, idx]).sum().item()
                total_elements += batch_size
                
                for i in range(batch_size):
                    rank = (sorted_indices[i] == batch_y[i, idx]).nonzero(as_tuple=True)[0].item()
                    ge_list[idx].append(rank)
                        
    final_ge_list = np.array([int(np.mean(ge_list[idx])) for idx in range(26)])
    avg_accuracy = (total_correct / total_elements) * 100
    final_val_loss = val_loss / (len(dataloader) * 26)
    return final_ge_list, avg_accuracy, final_val_loss


# =========================================================================
# 🚀 4. Accelerated Multi-Head Deep Learning Profiling Controller
# =========================================================================
if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 32
    EPOCHS = 100
    LR = 0.0001
    patience = 15
    counter = 0
    best_val_acc = 0.0
    
    # Repository directory settings
    MODEL_DIR = "./models"
    save_path = os.path.join(MODEL_DIR, "kyber_ASC_measured.pth")
    os.makedirs(MODEL_DIR, exist_ok=True)
        
    train_path = "./data/measured_train.npz"
    valid_path = "./data/measured_val.npz"
    
    if not (os.path.exists(train_path) and os.path.exists(valid_path)):
        print("⚠️ [Data Path Information] Please place pre-compiled 'measured_train.npz' and 'measured_val.npz' files.")
    
    # Initialize sanitised dataset interface
    train_dataset = SCAPackageDataset(train_path, DEVICE=DEVICE)
    val_dataset = SCAPackageDataset(valid_path, DEVICE=DEVICE)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = ASCAD(num_targets=26).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()
    
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-6)
    
    best_model_relative_time = 0.0
    global_start_time = time.time()
    
    print("🚀 Parallel Multi-Head Profiled Blind SCA Engine Initialized.")
    
    try:
        for epoch in range(EPOCHS):
            epoch_start = time.time()
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_elements = 0
            
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:03d}/{EPOCHS} [Train]")
            for batch_x, batch_y in train_pbar:
                batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                optimizer.zero_grad()
                
                with autocast():
                    outputs = model(batch_x)
                    total_loss = 0.0
                    for idx in range(26):
                        total_loss += criterion(outputs[idx], batch_y[:, idx])
                
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                
                train_loss += total_loss.item()
                batch_size = batch_x.size(0)
                for idx in range(26):
                    preds = torch.argmax(outputs[idx], dim=1)
                    train_correct += (preds == batch_y[:, idx]).sum().item()
                    train_elements += batch_size
                    
                train_pbar.set_postfix({"Batch Loss": f"{total_loss.item():.4f}"})
                
            current_train_loss = train_loss / len(train_loader)
            current_train_acc = (train_correct / train_elements) * 100
            
            epoch_ge_list, val_accuracy, current_val_loss = evaluate_metrics(model, val_loader, DEVICE)
            scheduler.step()
            
            epoch_duration = time.time() - epoch_start
            avg_ge = np.mean(epoch_ge_list)
            
            print(f"📊 Epoch [{epoch+1:03d}/{EPOCHS}] Evaluation Report ({epoch_duration:.1f}s)")
            print(f"   -> [Train] Loss: {current_train_loss:.4f} | Accuracy: {current_train_acc:.2f}%")
            print(f"   -> [Valid] Loss: {current_val_loss:.4f} | Accuracy: {val_accuracy:.2f}%")
            print(f"   -> [SCA]   Mean Guessing Entropy Plane: {avg_ge:.1f}")
            
            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
                counter = 0
                best_model_relative_time = time.time() - global_start_time
                torch.save(model.state_dict(), save_path)
                print(f"💾 [BEST] Model weights dynamically updated -> {save_path}")
            else:
                counter += 1
                
            if counter >= patience:
                print("\n🛑 Early stopping triggered due to validation accuracy stagnation.")
                break

    except KeyboardInterrupt:
        print("\n⚠️ Execution gracefully halted via user keyboard interrupt command.")

    total_elapsed_time = time.time() - global_start_time
    print("\n========================================================")
    print("🏆 final Execution Performance Evaluation Report")
    print("========================================================")
    print(f"📢 Optimal Target Validation Accuracy Obtained : {best_val_acc:.2f}%")
    print(f"⏱️ Total Training Framework Operational Period: {total_elapsed_time/60:.2f} min")
    print(f"🎯 Converged Golden Weight Timestamp Target   : {best_model_relative_time/60:.2f} min")
    print("========================================================")