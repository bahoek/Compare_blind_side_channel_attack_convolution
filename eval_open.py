"""
Evaluation on the public reference dataset — Hamming-weight GE and SR.

Same metrics as eval_measured.py:

  GE = mean rank of the correct Hamming-weight class among the 17 classes
  SR = top-1 accuracy

This dataset has a single target, so instead of a coefficient axis the figure
resolves the metric by true Hamming-weight class (0..16).  That view also
answers the obvious question about a 17-class problem with a non-uniform
prior: whether performance comes from the crowded middle classes alone or
holds at the sparse extremes.  Per-class accuracy is stored alongside.

    python eval_open.py

Input : data/open_attack.npz
        models/{arch}_open_s*.pth
Output: figures/ge_open_{arch}.eps   (one per architecture)
        figures/ge_open_{arch}.npz
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

ARCHS = ["resnet", "ascad", "mobilenet", "efficientnet"]
BATCH = 128


@torch.no_grad()
def evaluate(model, dataset, device):
    """Overall (GE, SR) plus the same two metrics split by true class."""
    n = len(dataset)
    class_rank = np.zeros(NUM_CLASSES, np.float64)
    class_hit = np.zeros(NUM_CLASSES, np.int64)
    class_count = np.zeros(NUM_CLASSES, np.int64)

    for start in range(0, n, BATCH):
        stop = min(start + BATCH, n)
        x = dataset.traces[start:stop].unsqueeze(1).to(device)
        y = dataset.labels[start:stop, 0].to(device)
        o = model(x)[0]
        true_score = o.gather(1, y[:, None])
        rank = ((o > true_score).sum(1) + 1).float().cpu().numpy()
        hit = (o.argmax(1) == y).cpu().numpy().astype(np.int64)
        cls = y.cpu().numpy()
        np.add.at(class_rank, cls, rank)
        np.add.at(class_hit, cls, hit)
        np.add.at(class_count, cls, 1)

    denom = np.maximum(class_count, 1)
    return (class_rank.sum() / n,
            class_hit.sum() / n * 100,
            np.where(class_count > 0, class_rank / denom, np.nan),
            np.where(class_count > 0, class_hit / denom * 100, np.nan))


def plot_architecture(arch, seeds, class_ge):
    """One figure per architecture: per-seed curves plus their mean."""
    plt.figure(figsize=(4.5, 3.5))
    cmap = plt.get_cmap("tab10")
    xs = np.arange(NUM_CLASSES)
    for i, seed in enumerate(seeds):
        plt.plot(xs, class_ge[i], color=cmap(i % 10), linewidth=0.8,
                 marker="o", markersize=2.5, label=f"seed {seed}")
    plt.plot(xs, np.nanmean(class_ge, 0), color="k", linewidth=1.8, label="mean")
    plt.axhline(1, color="k", linestyle=":", linewidth=1)
    plt.xlabel("True Hamming Weight Class", fontsize=11)
    plt.ylabel("Guessing Entropy", fontsize=11)
    plt.xticks(xs[::2], fontsize=9)
    plt.yticks(fontsize=9)
    plt.title(f"{arch.capitalize()} Hamming weight guessing entropy", fontsize=11)
    plt.grid(True, linestyle=":", color="0.85")
    plt.legend(fontsize=6, ncol=3, framealpha=1.0, edgecolor="black")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, f"ge_open_{arch}.eps")
    plt.savefig(path, format="eps", bbox_inches="tight")
    plt.close()
    return path


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    attack = SCAPackage(os.path.join(DATA_DIR, "open_attack.npz"), device)
    print(f"device {device}")
    print(f"attack: {attack.summary()}")
    print(f"majority-class baseline "
          f"{attack.majority_class_accuracy()[0]:.2f}%")

    summary = {}
    for arch in ARCHS:
        pattern = os.path.join(MODEL_DIR, f"{arch}_open_s*.pth")
        files = sorted(glob.glob(pattern),
                       key=lambda p: int(re.search(r"_s(\d+)\.pth", p).group(1)))
        if not files:
            print(f"\n[{arch}] no checkpoints matching {pattern}")
            continue
        print(f"\n[{arch}] {len(files)} seeds")

        seeds, ges, srs, class_ges, class_accs = [], [], [], [], []
        for path in files:
            seed = int(re.search(r"_s(\d+)\.pth", path).group(1))
            obj = torch.load(path, map_location=device, weights_only=False)
            model = build_model(arch, num_targets=1,
                                input_len=attack.trace_len).to(device)
            load_checkpoint(model, obj["model"])
            model.eval()

            started = time.time()
            ge, sr, class_ge, class_acc = evaluate(model, attack, device)
            seeds.append(seed)
            ges.append(ge)
            srs.append(sr)
            class_ges.append(class_ge)
            class_accs.append(class_acc)
            print(f"  s{seed}: GE {ge:.3f}  SR {sr:.2f}%  "
                  f"({time.time() - started:.0f}s)")

            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        ges = np.array(ges)
        srs = np.array(srs)
        class_ges = np.stack(class_ges)
        class_accs = np.stack(class_accs)
        summary[arch] = (ges, srs)

        print(f"  saved {plot_architecture(arch, seeds, class_ges)}")
        np.savez(os.path.join(FIG_DIR, f"ge_open_{arch}.npz"),
                 seeds=np.array(seeds), ge=ges, sr=srs,
                 class_ge=class_ges, class_acc=class_accs)

    print("\n" + "=" * 56)
    print(f"{'architecture':<16s} {'SR (%)':>18s} {'GE':>16s}")
    print("-" * 56)
    for arch, (ges, srs) in summary.items():
        print(f"{arch:<16s} {srs.mean():>10.2f} +- {srs.std():<5.2f} "
              f"{ges.mean():>9.3f} +- {ges.std():.3f}")


if __name__ == "__main__":
    main()
