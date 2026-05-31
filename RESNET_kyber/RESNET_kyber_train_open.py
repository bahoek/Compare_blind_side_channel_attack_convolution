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
# 🧠 2. Modern 1D ResNet Backbone Architecture
# =========================================================================
class BasicBlock1d(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock1d, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels * self.expansion)
            )
            
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x) 
        return self.relu(out)

class ResNet(nn.Module):
    def __init__(self, block=BasicBlock1d, num_blocks=[2, 2, 2], num_classes=256):
        super(ResNet, self).__init__()
        self.in_channels = 16 
        self.conv1 = nn.Conv1d(1, 16, kernel_size=11, stride=2, padding=5, bias=False)
        self.bn1 = nn.BatchNorm1d(16)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(4) 
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(64, 128), 
            nn.ReLU(), 
            nn.Dropout(p=0.5), 
            nn.Linear(128, num_classes)
        )
        
    def _make_layer(self, block, out_channels, num_blocks, stride):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, num_blocks): 
            layers.append(block(self.in_channels, out_channels, stride=1))
        return nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


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
    save_path = os.path.join(MODEL_DIR, "kyber_RES_open.pth")
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

    model = ResNet(block=BasicBlock1d, num_blocks=[2, 2, 2], num_classes=256).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    print(f"🚀 1D-ResNet Profiled SCA Benchmarking Module Initialized. (Target Backend: {DEVICE})")
    
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