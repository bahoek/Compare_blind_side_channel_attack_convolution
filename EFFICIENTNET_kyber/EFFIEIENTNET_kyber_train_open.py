import numpy as np
import os
from tqdm import tqdm
import time
import math

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
# 🧠 2. Modern 1D EfficientNet-B0 Backbone Architecture
# =========================================================================
class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
    
class SEBlock(nn.Module):
    def __init__(self, in_channels, r=4):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, in_channels // r, bias=False),
            SiLU(),
            nn.Linear(in_channels // r, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1)
        return x * y
    
class MBConv(nn.Module):
    def __init__(self, in_planes, out_planes, expand_ratio, kernel_size, stride, reduction=4):
        super(MBConv, self).__init__()
        self.use_residual = (in_planes == out_planes) and (stride == 1)
        hidden_dim = in_planes * expand_ratio
        self.expand = in_planes != hidden_dim

        layers = []
        if self.expand:
            layers += [
                nn.Conv1d(in_planes, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm1d(hidden_dim),
                SiLU()
            ]
        padding = (kernel_size - 1) // 2
        layers += [
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size, stride=stride, padding=padding, groups=hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            SiLU()
        ]

        layers += [SEBlock(hidden_dim, reduction)]

        layers += [
            nn.Conv1d(hidden_dim, out_planes, 1, bias=False),
            nn.BatchNorm1d(out_planes)
        ]

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)
        
class EfficientNetB0(nn.Module):
    def __init__(self, num_classes=256):
        super(EfficientNetB0, self).__init__()
        self.configs = [
            [1, 16, 1, 1, 3],
            [6, 24, 2, 2, 3],
            [6, 40, 2, 2, 5],
            [6, 80, 3, 2, 3],
            [6, 112, 3, 1, 5],
            [6, 192, 4, 2, 5],
            [6, 320, 1, 1, 3]
        ]
        
        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(32),
            SiLU(),
            nn.MaxPool1d(kernel_size=5, stride=5)
        )

        layers = []
        in_channels = 32
        for t, c, n, s, k in self.configs:
            for i in range(n):
                stride = s if i == 0 else 1
                layers.append(MBConv(in_channels, c, expand_ratio=t, kernel_size=k, stride=stride))
                in_channels = c
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Conv1d(in_channels, 1280, kernel_size=1, bias=False),
            nn.BatchNorm1d(1280),
            SiLU()
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes)
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = self.head(x)
        x = x.mean(dim=2) # Global Average Pooling (GAP)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()


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
    save_path = os.path.join(MODEL_DIR, "kyber_EFF_open.pth")
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
    min_delta = 0.5         
    target_threshold = 95.0
    
    scaler = GradScaler()
    start_time = time.time()
    
    model = EfficientNetB0(num_classes=256).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    print(f"🚀 EfficientNetB0 Profiled SCA Benchmarking Module Initialized. (Target Backend: {DEVICE})")
    
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

            if is_above_target:
                if val_acc > best_acc:
                    best_acc = val_acc
                    torch.save(model.state_dict(), save_path)
                    best_epoch_idx = epoch + 1
                    best_model_time = time.time() - start_time
                print(f"\n✨ [🎯 TARGET REACHED] Target Accuracy of {target_threshold}% Successfully Captured!")
                break 

            if is_significantly_improved:
                if val_acc > best_acc:
                    best_acc = val_acc
                    torch.save(model.state_dict(), save_path)
                    best_epoch_idx = epoch + 1
                    best_model_time = time.time() - start_time
                    print(f"💾 New Best Model Saved (Acc: {val_acc:.2f}%)")
                counter = 0 
            else:
                counter += 1

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