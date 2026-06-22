import numpy as np
import os
from tqdm import tqdm
import time

import torch
import torch.nn as nn 
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast, GradScaler

# =========================================================================
# 📦 1. Side-Channel Analysis Dataset Loader Interface (IP Protected)
# =========================================================================
class SCAPackageDataset(Dataset):
    """
    Custom Dataset class for loading pre-compiled cryptographic power traces.
    
    NOTE: To protect proprietary research infrastructure and hardware-specific 
    alignment/labeling emulators, this loader assumes that all physical traces 
    and target 256-class direct coefficient labels have already been converted, 
    synchronized, and consolidated into a standardized matrix format beforehand.
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
# 🧠 2. Modern 1D Convolutional Neural Network Architecture (ASCAD-1D)
# =========================================================================
class ASCAD(nn.Module):
    def __init__(self, num_classes=256):
        super(ASCAD, self).__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=65, stride=2, padding=32, bias=False),  # Temporal resolution tuning
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Conv1d(64, 128, kernel_size=33, stride=2, padding=16, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Conv1d(128, 256, kernel_size=17, stride=2, padding=8, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
        )
        
        self.shared_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(6250 * 256, 1024), 
            nn.ReLU(),
            nn.Dropout(0.4)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.shared_fc(x)
        x = self.classifier(x)
        return x


# =========================================================================
# 🚀 3. Core Neural Network Optimization and Training Execution Loop
# =========================================================================
if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 32
    EPOCHS = 100
    LR = 0.0001
    
    # Standard repository directories
    MODEL_DIR = "./models"
    save_path = os.path.join(MODEL_DIR, "kyber_ASC_open.pth")
    os.makedirs(MODEL_DIR, exist_ok=True)
        
    train_path = "./data/train.npz"
    valid_path = "./data/val.npz"
    
    # Defensive data path check for reproducibility validation
    if not (os.path.exists(train_path) and os.path.exists(valid_path)):
        print("⚠️ [Data Path Information] Please configure and locate 'train.npz' and 'val.npz' within the local directory.")
    
    # Initialize sanitized dataset objects
    train = SCAPackageDataset(train_path, DEVICE=DEVICE)
    valid = SCAPackageDataset(valid_path, DEVICE=DEVICE)
    
    train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid, batch_size=BATCH_SIZE, shuffle=False)
    
    history = {'loss': [], 'val_loss': []}
    best_acc = 0.0
    patience = 15
    counter = 0
    best_epoch_idx = 0
    best_model_time = 0.0
    
    # ⚙️ Optimization Boundary Conditions for Low-Entropy Target Tracking
    min_delta = 0.5         # Minimum accuracy improvement required to clear early stopping patience
    target_threshold = 95.0 # Early termination accuracy ceiling
    
    scaler = GradScaler()
    start_time = time.time()

    model = ASCAD(num_classes=256).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    print(f"🚀 ASCAD Profiled SCA Benchmarking Module Initialized. (Target Backend: {DEVICE})")
    
    try:
        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0
            correct = 0
            total = 0
            for traces, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
                traces, labels = traces.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                
                with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                    outputs = model(traces)
                    loss = criterion(outputs, labels)
                    
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            train_acc = 100 * correct / total
            avg_train_loss = total_loss / len(train_loader)
            
            model.eval()
            running_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for traces, labels in valid_loader:
                    traces, labels = traces.to(DEVICE), labels.to(DEVICE)
                    
                    with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                        outputs = model(traces)
                        loss = criterion(outputs, labels)
                        
                    running_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    
            val_acc = 100 * correct / total
            scheduler.step(val_acc)
            val_epoch_loss = running_loss / len(valid_loader)

            history['loss'].append(avg_train_loss) 
            history['val_loss'].append(val_epoch_loss)
            
            print(f"📊 Epoch {epoch+1}: Loss: {val_epoch_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
            
            is_significantly_improved = val_acc >= (best_acc + min_delta)
            is_above_target = val_acc >= target_threshold

            # 💡 [CRITICAL PATHWAYS]: Target-driven early execution interceptor
            if is_above_target:
                if val_acc > best_acc:
                    best_acc = val_acc
                    torch.save(model.state_dict(), save_path)
                    best_epoch_idx = epoch + 1
                    best_model_time = time.time() - start_time
                
                print(f"\n✨ [🎯 TARGET REACHED] Validation accuracy has exceeded the designated {target_threshold}% limit.")
                print(f"💾 Optimized target weights compiled ➔ [Epoch {best_epoch_idx} / Checkpoint Time: {best_model_time // 60:.0f}m {best_model_time % 60:.0f}s]")
                print("🛑 Convergence criterion satisfied. Terminating the remaining pipeline sessions cleanly.")
                break 

            if is_significantly_improved:
                if val_acc > best_acc:
                    best_acc = val_acc
                    torch.save(model.state_dict(), save_path)
                    best_epoch_idx = epoch + 1
                    best_model_time = time.time() - start_time
                    print(f"💾 New Best Model Saved (Acc: {val_acc:.2f}%)")
                
                print(f"📈 [SIGNIFICANT UPDATE] Validation metric advanced by $\ge$ {min_delta}%. Resetting early stopping patience counters.")
                counter = 0 
            else:
                counter += 1
                print(f"⚠️ Convergence plateau detected. Aggregating internal patience metric ➔ [{counter}/{patience}]")

            if counter >= patience:
                print(f"🛑 Early stopping triggered at epoch {epoch+1} due to loss convergence boundaries.")
                break
                
    except KeyboardInterrupt:
        print("\n⚠️ Process interrupted gracefully by user command.")
    
    total_time = time.time() - start_time 
    print("-" * 50)
    print(f"🏆 Final Network Evaluation Report")
    print(f"1. Total Training Time         : {total_time // 60:.0f}m {total_time % 60:.0f}s")
    print(f"2. Best Performance Epoch      : Epoch {best_epoch_idx}")
    print(f"3. Maximum Peak Validation Acc : {best_acc:.2f}%")
    print("-" * 50)
