"""
Profiling on the public reference dataset — single-head Hamming-weight model.

A trace in this dataset covers one coefficient pair, so the model has a single
classification head predicting the Hamming weight of the 16-bit accumulator
word (17 classes).  Everything else — backbones, optimiser, schedule, early
stopping, seed protocol — matches train_measured.py, so the two datasets are
compared under identical training conditions.

    python train_open.py

Input : pre-compiled packages (see dataset.py for the format)
          data/open_train.npz
          data/open_val.npz
Output: models/{arch}_open_s{seed}.pth
        models/summary_{arch}_open.npz
"""

import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import SCAPackage, NUM_CLASSES
from models import build_model

# ----------------------------------------------------------------------
DATA_DIR = "./data"
SAVE_DIR = "./models"

ARCH  = "resnet"                # resnet | ascad | mobilenet | efficientnet
SEEDS = list(range(10))

BATCH     = 128
LR        = 1e-3                # MBConv-based backbones may need 3e-4
EPOCHS    = 100
PATIENCE  = 15
MIN_DELTA = 0.3                 # percentage points
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 5.0


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


@torch.no_grad()
def evaluate(model, loader, device):
    """Returns (mean loss, accuracy %, guessing entropy)."""
    model.eval()
    ce = nn.CrossEntropyLoss(reduction="sum")
    loss, correct, rank, seen = 0.0, 0, 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        y = y[:, 0]
        o = model(x)[0]
        loss += ce(o, y).item()
        correct += (o.argmax(1) == y).sum().item()
        true_score = o.gather(1, y[:, None])
        rank += ((o > true_score).sum(1) + 1).float().sum().item()
        seen += len(y)
    return loss / seen, correct / seen * 100, rank / seen


def run_seed(seed, train_set, val_set, device):
    set_seed(seed)
    train_loader = DataLoader(train_set, batch_size=BATCH, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=256, shuffle=False)

    model = build_model(ARCH, num_targets=1, input_len=train_set.trace_len).to(device)
    model.init_bias(train_set.labels_numpy())

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    best_acc, best_patience, stale, best_ge = 0.0, 0.0, 0, np.nan
    ckpt = os.path.join(SAVE_DIR, f"{ARCH}_open_s{seed}.pth")
    started = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running, batches, grad_norms = 0.0, 0, []
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            y = y[:, 0]
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                with torch.amp.autocast("cuda"):
                    loss = criterion(model(x)[0], y)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = criterion(model(x)[0], y)
                loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
            running += loss.item()
            batches += 1
            if np.isfinite(float(gn)):
                grad_norms.append(float(gn))

        val_loss, acc, ge = evaluate(model, val_loader, device)
        scheduler.step()
        print(f"  [s{seed} {epoch:03d}/{EPOCHS}] train {running / batches:.4f} "
              f"| val {val_loss:.4f} | acc {acc:.2f}% | GE {ge:.2f} "
              f"| grad {np.median(grad_norms):.3f}")

        if acc > best_acc:
            best_acc, best_ge = acc, ge
            torch.save({"model": model.state_dict(), "arch": ARCH, "seed": seed,
                        "num_targets": 1, "acc": acc, "ge": ge}, ckpt)
        if acc > best_patience + MIN_DELTA:
            best_patience, stale = acc, 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break

    minutes = (time.time() - started) / 60
    print(f"  [s{seed}] best {best_acc:.2f}%  GE {best_ge:.2f}  ({minutes:.1f} min)")
    return best_acc, best_ge, minutes


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_set = SCAPackage(os.path.join(DATA_DIR, "open_train.npz"), device)
    val_set = SCAPackage(os.path.join(DATA_DIR, "open_val.npz"), device)
    if train_set.num_targets != 1:
        raise ValueError(f"expected a single-target package, "
                         f"got {train_set.num_targets} targets")

    baseline = float(val_set.majority_class_accuracy()[0])
    print(f"device {device} | arch {ARCH} | {NUM_CLASSES} classes | seeds {SEEDS}")
    print(f"train: {train_set.summary()}")
    print(f"val  : {val_set.summary()}")
    print(f"majority-class baseline {baseline:.2f}%\n")

    accs, ges, minutes = [], [], []
    for seed in SEEDS:
        a, g, m = run_seed(seed, train_set, val_set, device)
        accs.append(a)
        ges.append(g)
        minutes.append(m)

    accs = np.array(accs)
    ges = np.array(ges)
    minutes = np.array(minutes)

    print("\n" + "=" * 52)
    print(f"{ARCH} | open | {len(SEEDS)} seeds")
    print("=" * 52)
    print(f"  accuracy : {accs.mean():.2f}% +- {accs.std():.2f}%  "
          f"(baseline {baseline:.2f}%)")
    print(f"  GE       : {ges.mean():.3f} +- {ges.std():.3f}")
    print(f"  per seed : " + ", ".join(f"{a:.1f}" for a in accs))

    np.savez(os.path.join(SAVE_DIR, f"summary_{ARCH}_open.npz"),
             acc=accs, ge=ges, baseline=baseline,
             seeds=np.array(SEEDS), minutes=minutes)
    print(f"\nsaved summary_{ARCH}_open.npz")


if __name__ == "__main__":
    main()
