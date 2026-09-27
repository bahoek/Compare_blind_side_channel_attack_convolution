"""
Profiling on the measured dataset (STM32F4) — multi-head Hamming-weight model.

One trace covers many coefficients of the polynomial product, so a single
trunk feeds 78 classification heads, one per target coefficient, each
predicting the Hamming weight of the 16-bit accumulator word (17 classes).

Run the four backbones by changing ARCH; ten seeds per backbone give the
mean and standard deviation reported in the paper.

    python train_measured.py

Input : pre-compiled packages (see dataset.py for the format)
          data/measured_train.npz
          data/measured_val.npz
Output: models/{arch}_measured_s{seed}.pth
        models/summary_{arch}_measured.npz

Training choices that matter
----------------------------
* Loss is the mean over heads, not the sum.  With a sum, gradient-norm
  clipping removes a share of the update that grows with the number of
  heads, so a model with more heads is handicapped by the clipping rather
  than by its capacity.
* Head bias starts at log(class prior) and head weights keep their default
  initialisation (see models.init_head_bias).
* Early stopping saves on any improvement but only resets its patience
  counter on an improvement larger than MIN_DELTA, so a long tail of
  fractional gains does not keep training alive.
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

ARCH        = "resnet"          # resnet | ascad | mobilenet | efficientnet
NUM_TARGETS = 78
SEEDS       = list(range(10))

BATCH     = 128
LR        = 3e-3                # MBConv-based backbones may need 3e-4
EPOCHS    = 100
PATIENCE  = 15
MIN_DELTA = 0.3                 # percentage points
WEIGHT_DECAY = 3e-3
GRAD_CLIP = 5.0


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


@torch.no_grad()
def evaluate(model, loader, device, num_targets):
    """Returns (mean loss, per-head accuracy %, per-head guessing entropy)."""
    model.eval()
    ce = nn.CrossEntropyLoss(reduction="sum")
    loss = 0.0
    correct = np.zeros(num_targets, np.int64)
    rank = np.zeros(num_targets, np.float64)
    seen = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        outs = model(x)
        seen += x.size(0)
        for i in range(num_targets):
            o = outs[i]
            loss += ce(o, y[:, i]).item()
            correct[i] += (o.argmax(1) == y[:, i]).sum().item()
            true_score = o.gather(1, y[:, i:i + 1])
            rank[i] += ((o > true_score).sum(1) + 1).float().sum().item()
    return loss / (seen * num_targets), correct / seen * 100, rank / seen


def run_seed(seed, train_set, val_set, device):
    set_seed(seed)
    train_loader = DataLoader(train_set, batch_size=BATCH, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=256, shuffle=False)

    model = build_model(ARCH, num_targets=NUM_TARGETS,
                        input_len=train_set.trace_len).to(device)
    model.init_bias(train_set.labels_numpy())

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    best_acc, best_patience, stale = 0.0, 0.0, 0
    best_head_acc, best_head_ge = None, None
    ckpt = os.path.join(SAVE_DIR, f"{ARCH}_measured_s{seed}.pth")
    started = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running, batches, grad_norms = 0.0, 0, []
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                with torch.amp.autocast("cuda"):
                    outs = model(x)
                    loss = sum(criterion(outs[i], y[:, i])
                               for i in range(NUM_TARGETS)) / NUM_TARGETS
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
            else:
                outs = model(x)
                loss = sum(criterion(outs[i], y[:, i])
                           for i in range(NUM_TARGETS)) / NUM_TARGETS
                loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
            running += loss.item()
            batches += 1
            if np.isfinite(float(gn)):
                grad_norms.append(float(gn))

        val_loss, head_acc, head_ge = evaluate(model, val_loader, device, NUM_TARGETS)
        scheduler.step()
        print(f"  [s{seed} {epoch:03d}/{EPOCHS}] train {running / batches:.4f} "
              f"| val {val_loss:.4f} | acc {head_acc.mean():.2f}% "
              f"| GE {head_ge.mean():.2f} | grad {np.median(grad_norms):.3f}")

        current = head_acc.mean()
        if current > best_acc:
            best_acc = current
            best_head_acc, best_head_ge = head_acc.copy(), head_ge.copy()
            torch.save({"model": model.state_dict(), "arch": ARCH, "seed": seed,
                        "num_targets": NUM_TARGETS, "acc": head_acc, "ge": head_ge},
                       ckpt)
        if current > best_patience + MIN_DELTA:
            best_patience, stale = current, 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break

    minutes = (time.time() - started) / 60
    print(f"  [s{seed}] best {best_acc:.2f}%  GE {best_head_ge.mean():.2f}  "
          f"({minutes:.1f} min)")
    return best_acc, best_head_acc, best_head_ge, minutes


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_set = SCAPackage(os.path.join(DATA_DIR, "measured_train.npz"), device)
    val_set = SCAPackage(os.path.join(DATA_DIR, "measured_val.npz"), device)
    if train_set.num_targets != NUM_TARGETS:
        raise ValueError(f"package has {train_set.num_targets} targets, "
                         f"NUM_TARGETS is {NUM_TARGETS}")

    baseline = val_set.majority_class_accuracy()
    print(f"device {device} | arch {ARCH} | {NUM_CLASSES} classes | seeds {SEEDS}")
    print(f"train: {train_set.summary()}")
    print(f"val  : {val_set.summary()}")
    print(f"majority-class baseline {baseline.mean():.2f}% | "
          f"prior cross-entropy {val_set.prior_entropy():.4f}\n")

    accs, ges, head_accs, minutes = [], [], [], []
    for seed in SEEDS:
        a, ha, hg, m = run_seed(seed, train_set, val_set, device)
        accs.append(a)
        ges.append(hg.mean())
        head_accs.append(ha)
        minutes.append(m)

    accs = np.array(accs)
    ges = np.array(ges)
    head_accs = np.stack(head_accs)
    minutes = np.array(minutes)

    print("\n" + "=" * 62)
    print(f"{ARCH} | measured | {NUM_TARGETS} targets | {len(SEEDS)} seeds")
    print("=" * 62)
    print(f"  accuracy : {accs.mean():.2f}% +- {accs.std():.2f}%  "
          f"(baseline {baseline.mean():.2f}%)")
    print(f"  GE       : {ges.mean():.3f} +- {ges.std():.3f}")
    print(f"  per seed : " + ", ".join(f"{a:.1f}" for a in accs))
    mean_head = head_accs.mean(0)
    order = np.argsort(-mean_head)
    print(f"  strongest coefficients: " +
          ", ".join(f"c{int(i)}={mean_head[i]:.1f}" for i in order[:3]))
    print(f"  weakest   coefficients: " +
          ", ".join(f"c{int(i)}={mean_head[i]:.1f}" for i in order[-3:]))

    np.savez(os.path.join(SAVE_DIR, f"summary_{ARCH}_measured.npz"),
             acc=accs, ge=ges, head_acc=head_accs, baseline=baseline,
             seeds=np.array(SEEDS), minutes=minutes)
    print(f"\nsaved summary_{ARCH}_measured.npz")


if __name__ == "__main__":
    main()
