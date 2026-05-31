import numpy as np
import os
import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
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
# 🧠 2. Modern 1D ResNet18 Multi-Head Architecture (26-Axis Parallel)
# =========================================================================
class ResidualBlock1d(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock1d, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet18(nn.Module):
    def __init__(self, num_targets=26):
        super(ResNet18, self).__init__()
        self.in_channels = 64
        self.num_targets = num_targets
        
        # 40,000 차원 광역 전력 파형 초기 압축 스템
        self.stem = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=11, stride=2, padding=5, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=5, stride=5)
        )
        
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        
        # 26개 독립 헤드 선언 (각 축별 256클래스 확률 로짓 분출)
        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(0.4),
                nn.Linear(512, 256)
            ) for _ in range(num_targets)
        ])

    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(ResidualBlock1d(self.in_channels, out_channels, s))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.mean(dim=2) # GAP 연산 레이어 효과
        
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
    patience = 15
    counter = 0
    best_val_acc = 0.0
    LR = 0.0001
    
    # Repository directory settings
    MODEL_DIR = "./models"
    save_path = os.path.join(MODEL_DIR, "kyber_RES.pth")
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
    
    model = ResNet18(num_targets=26).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()
    
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-6)
    
    best_model_relative_time = 0.0
    global_start_time = time.time()
    
    print("🚀 Parallel Multi-Head 1D-ResNet18 Blind SCA Engine Initialized.")
    
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