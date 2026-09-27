"""
Evaluation on the measured dataset — Hamming-weight GE and SR per coefficient.

Metrics (single trace, attack package)
--------------------------------------
  GE(c) = mean rank of the correct Hamming-weight class among the 17 classes
          for coefficient c; 1 means the correct class is always ranked first
  SR(c) = top-1 accuracy for coefficient c

The Hamming-weight label changes from trace to trace, so there is no
accumulation over traces here: an accumulated score is only defined for a
target that stays fixed across traces, which is the secret key.  That
accumulation is what eval_key_recovery.py measures.  These two metrics are
the single-trace basis on which the four backbones are compared.

    python eval_measured.py

Input : data/measured_attack.npz
        models/{arch}_measured_s*.pth
Output: figures/ge_measured_{arch}.eps   (one per architecture)
        figures/ge_measured_{arch}.npz
"""

import glob
import os
import re
import time

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import SCAPackage, NUM_CLASSES
from models import build_model, load_checkpoint

# ----------------------------------------------------------------------
DATA_DIR  = "./data"
MODEL_DIR = "./models"
FIG_DIR   = "./figures"

ARCHS       = ["resnet", "ascad", "mobilenet", "efficientnet"]
NUM_TARGETS = 78
BATCH       = 256


@torch.no_grad()
def evaluate(model, dataset, device, num_targets):
    """Per-head (GE, SR) over the whole attack package."""
    n = len(dataset)
    ge = np.zeros(num_targets, np.float64)
    sr = np.zeros(num_targets, np.float64)
    for start in range(0, n, BATCH):
        stop = min(start + BATCH, n)
        x = dataset.traces[start:stop].unsqueeze(1).to(device)
        y = dataset.labels[start:stop].to(device)
        outs = model(x)
        for h in range(num_targets):
            o = outs[h]
            true_score = o.gather(1, y[:, h:h + 1])
            ge[h] += ((o > true_score).sum(1) + 1).float().sum().item()
            sr[h] += (o.argmax(1) == y[:, h]).sum().item()
    return ge / n, sr / n * 100


def plot_architecture(arch, seeds, ge_seed):
    """One figure per architecture: per-seed curves plus their mean."""
    plt.figure(figsize=(4.5, 3.5))
    cmap = plt.get_cmap("tab10")
    xs = np.arange(ge_seed.shape[1])
    for i, seed in enumerate(seeds):
        plt.plot(xs, ge_seed[i], color=cmap(i % 10), linewidth=0.8,
                 label=f"seed {seed}")
    plt.plot(xs, ge_seed.mean(0), color="k", linewidth=1.8, label="mean")
    plt.axhline(1, color="k", linestyle=":", linewidth=1)
    plt.xlabel("Coefficient Index", fontsize=11)
    plt.ylabel("Guessing Entropy", fontsize=11)
    plt.xticks(fontsize=9)
    plt.yticks(fontsize=9)
    plt.title(f"{arch.capitalize()} Hamming weight guessing entropy", fontsize=11)
    plt.grid(True, linestyle=":", color="0.85")
    plt.legend(fontsize=6, ncol=3, framealpha=1.0, edgecolor="black")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, f"ge_measured_{arch}.eps")
    plt.savefig(path, format="eps", bbox_inches="tight")
    plt.close()
    return path


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    attack = SCAPackage(os.path.join(DATA_DIR, "measured_attack.npz"), device)
    print(f"device {device}")
    print(f"attack: {attack.summary()}")
    print(f"majority-class baseline "
          f"{attack.majority_class_accuracy().mean():.2f}%")

    summary = {}
    for arch in ARCHS:
        pattern = os.path.join(MODEL_DIR, f"{arch}_measured_s*.pth")
        files = sorted(glob.glob(pattern),
                       key=lambda p: int(re.search(r"_s(\d+)\.pth", p).group(1)))
        if not files:
            print(f"\n[{arch}] no checkpoints matching {pattern}")
            continue
        print(f"\n[{arch}] {len(files)} seeds")

        seeds, ge_seed, sr_seed = [], [], []
        for path in files:
            seed = int(re.search(r"_s(\d+)\.pth", path).group(1))
            obj = torch.load(path, map_location=device, weights_only=False)
            model = build_model(arch, num_targets=NUM_TARGETS,
                                input_len=attack.trace_len).to(device)
            load_checkpoint(model, obj["model"])
            model.eval()

            started = time.time()
            ge, sr = evaluate(model, attack, device, NUM_TARGETS)
            seeds.append(seed)
            ge_seed.append(ge)
            sr_seed.append(sr)
            print(f"  s{seed}: GE {ge.mean():.3f}  SR {sr.mean():.2f}%  "
                  f"({time.time() - started:.0f}s)")

            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        ge_seed = np.stack(ge_seed)
        sr_seed = np.stack(sr_seed)
        summary[arch] = (ge_seed, sr_seed)

        print(f"  saved {plot_architecture(arch, seeds, ge_seed)}")
        np.savez(os.path.join(FIG_DIR, f"ge_measured_{arch}.npz"),
                 seeds=np.array(seeds), ge=ge_seed, sr=sr_seed)

    print("\n" + "=" * 56)
    print(f"{'architecture':<16s} {'SR (%)':>18s} {'GE':>16s}")
    print("-" * 56)
    for arch, (ge_seed, sr_seed) in summary.items():
        sr = sr_seed.mean(1)
        ge = ge_seed.mean(1)
        print(f"{arch:<16s} {sr.mean():>10.2f} +- {sr.std():<5.2f} "
              f"{ge.mean():>9.3f} +- {ge.std():.3f}")


if __name__ == "__main__":
    main()
