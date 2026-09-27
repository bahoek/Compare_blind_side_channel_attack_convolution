"""
Key recovery from Hamming-weight predictions — guessing entropy over traces.

What this measures
------------------
The accumulator word written by one base multiplication depends on a pair of
secret coefficients, so the key hypothesis is a pair (k0, k1) drawn from a
q x q grid of 3329^2 = 1.11e7 candidates.  For a candidate pair, the word it
would produce on a given trace is a sum of two partial products, each of which
depends on one coefficient of the pair only:

    word(k0, k1) = (A[k0] + B[k1]) mod 2^16

Because the two terms separate, the whole grid for one trace is the outer sum
of two length-q vectors, and the model's log-softmax over Hamming-weight
classes turns it into a score grid in one gather.  The secret stays fixed
across traces, so the log-likelihoods accumulate, and the rank of the correct
pair falls as traces are added.  This is the step the Hamming-weight
benchmark (eval_measured.py, eval_open.py) does not cover: it shows that the
recovered leakage translates into the secret key itself.

  GE(n) = rank of the correct pair among 1.11e7 candidates after n traces,
          averaged over N_PAIRS pairs and N_REPEAT accumulation orders

Note on the threat model.  Recovering the Hamming weight uses the trace only.
Turning those predictions into key hypotheses additionally uses the
ciphertexts, which are public values transmitted in the clear during key
encapsulation and therefore available to any eavesdropper; the shared secret
is never used.  The targeted intermediate is a function of both the key and
the ciphertext, so ciphertext knowledge is structurally required for this
stage.

Hypothesis package
------------------
The partial-product tables are produced by the same in-house toolchain as the
trace packages (see dataset.py) and shipped as data/key_hypotheses.npz:

  mul_tab   uint16, (M, M)   product-word table, M = 2q - 1.
                             mul_tab[u + offset, v + offset] is the 16-bit
                             word the device holds for operands u, v, stored
                             as its unsigned encoding.
  offset    int              index offset, q - 1
  q         int              3329
  head      int64,  (P,)     model head (coefficient index) for each pair
  row_a     int32,  (P, N)   mul_tab row of the first partial product, per trace
  row_b     int32,  (P, N)   mul_tab row of the second partial product
  row_tw    int32,  (P,)     mul_tab row of the twiddle applied to the second
                             partial product, or -1 when none applies
  true_pair int64,  (P, 2)   correct (k0, k1), used for scoring only

Performance
-----------
A[k0] and B[k1] are gathered for all traces at once, the Hamming-weight lookup
and the log-softmax lookup are fused into one 65536-entry table per trace, and
traces are accumulated CHUNK at a time so each chunk costs two kernels rather
than dozens.  The final checkpoint uses the whole attack set, where the
accumulated score does not depend on order, so it is computed once and shared
across the repetitions instead of being recomputed for each.

    python eval_key_recovery.py

Input : data/measured_attack.npz, data/key_hypotheses.npz
        models/{arch}_measured_s*.pth
Output: figures/key_recovery_{arch}.eps, figures/key_recovery_{arch}.npz
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

ARCH        = "resnet"          # resnet | ascad | mobilenet | efficientnet
NUM_TARGETS = 78

N_PAIRS   = 39                  # coefficient pairs evaluated per seed
N_TRACES  = [1, 2, 5, 10, 20, 50, 100, 200, 500,
             1000, 2000, 5000, 10000, 20000]
N_REPEAT  = 10                  # accumulation orders averaged per point
CHUNK     = 64                  # traces scored per kernel launch
ATTACK_N  = 20000
USE_COMPILE = True              # fuse add / mask / gather / sum

HW16 = np.array([bin(v).count("1") for v in range(65536)], dtype=np.int64)


# ---------------------------------------------------------------- scoring
def _chunk_score(a_chunk, b_chunk, lut):
    """a_chunk, b_chunk: (T, q) int32; lut: (T, 65536) float32 -> (q*q,)."""
    t = a_chunk.size(0)
    idx = ((a_chunk[:, :, None].long() + b_chunk[:, None, :].long()) & 0xFFFF)
    return lut.gather(1, idx.view(t, -1)).sum(0)


if USE_COMPILE:
    try:
        _chunk_score = torch.compile(_chunk_score, dynamic=True)
    except Exception as exc:                                  # pragma: no cover
        print(f"torch.compile unavailable, running eager: {exc}")


def _accumulate(score, indices, a_all, b_all, logp, hw_lut, device):
    for start in range(0, len(indices), CHUNK):
        ids = torch.as_tensor(indices[start:start + CHUNK].astype(np.int64),
                              device=device)
        lut = logp[ids][:, hw_lut].contiguous()
        score += _chunk_score(a_all[ids], b_all[ids], lut)


def pair_curve(logp, a_np, b_np, true_flat, q, device, rng):
    """Guessing-entropy curve for one coefficient pair."""
    n_all = len(logp)
    a_all = torch.from_numpy(a_np).to(device)
    b_all = torch.from_numpy(b_np).to(device)
    lp = torch.from_numpy(logp).to(device)
    hw_lut = torch.from_numpy(HW16).to(device)

    points = [n for n in N_TRACES if n <= n_all]
    full_last = points[-1] == n_all
    stepped = points[:-1] if full_last else points

    ranks = np.zeros((N_REPEAT, len(points)), np.float64)
    for rep in range(N_REPEAT):
        order = rng.permutation(n_all)
        score = torch.zeros(q * q, device=device, dtype=torch.float32)
        cursor = 0
        for j, n in enumerate(stepped):
            _accumulate(score, order[cursor:n], a_all, b_all, lp, hw_lut, device)
            cursor = n
            ranks[rep, j] = int((score > score[true_flat]).sum().item()) + 1

    if full_last:
        score = torch.zeros(q * q, device=device, dtype=torch.float32)
        _accumulate(score, np.arange(n_all), a_all, b_all, lp, hw_lut, device)
        ranks[:, -1] = int((score > score[true_flat]).sum().item()) + 1

    return np.array(points), ranks.mean(0)


# ---------------------------------------------------- hypothesis package
class HypothesisPackage:
    def __init__(self, path, n_traces):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"hypothesis package not found: {path}\n"
                f"See the format description at the top of this file.")
        d = np.load(path, mmap_mode="r")
        self.mul_tab = d["mul_tab"]
        self.offset = int(d["offset"])
        self.q = int(d["q"])
        self.head = np.asarray(d["head"], np.int64)
        self.row_a = np.asarray(d["row_a"], np.int64)[:, :n_traces]
        self.row_b = np.asarray(d["row_b"], np.int64)[:, :n_traces]
        self.row_tw = np.asarray(d["row_tw"], np.int64)
        self.true_pair = np.asarray(d["true_pair"], np.int64)
        self.n_pairs = len(self.head)
        self._cols = np.arange(self.q, dtype=np.int64) + self.offset

    @staticmethod
    def _signed(word):
        w = word.astype(np.int64)
        return np.where(w >= 32768, w - 65536, w)

    def partial_products(self, p):
        """(A, B) for pair p as (n_traces, q) int32 arrays of 16-bit words."""
        a = self.mul_tab[self.row_a[p][:, None], self._cols[None, :]]
        b = self.mul_tab[self.row_b[p][:, None], self._cols[None, :]]
        tw = int(self.row_tw[p])
        if tw >= 0:
            b = self.mul_tab[tw, self._signed(b) + self.offset]
        return (np.asarray(a, np.int64) & 0xFFFF).astype(np.int32), \
               (np.asarray(b, np.int64) & 0xFFFF).astype(np.int32)

    def true_index(self, p):
        k0, k1 = self.true_pair[p]
        return int(k0) * self.q + int(k1)


# ---------------------------------------------------------------- driver
def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {device} | arch {ARCH} | chunk {CHUNK}")

    attack = SCAPackage(os.path.join(DATA_DIR, "measured_attack.npz"))
    n = min(ATTACK_N, len(attack))
    print(f"attack: {attack.summary()}  (using {n})")

    hyp = HypothesisPackage(os.path.join(DATA_DIR, "key_hypotheses.npz"), n)
    n_pairs = min(N_PAIRS, hyp.n_pairs)
    print(f"candidates per pair: {hyp.q}^2 = {hyp.q ** 2 / 1e6:.2f}e6, "
          f"random-guess baseline {hyp.q ** 2 // 2:,}")

    pattern = os.path.join(MODEL_DIR, f"{ARCH}_measured_s*.pth")
    files = sorted(glob.glob(pattern),
                   key=lambda p: int(re.search(r"_s(\d+)\.pth", p).group(1)))
    if not files:
        print(f"no checkpoints matching {pattern}")
        return
    print(f"{len(files)} seeds\n")

    traces = attack.traces[:n]
    curves = []
    for path in files:
        seed = int(re.search(r"_s(\d+)\.pth", path).group(1))
        obj = torch.load(path, map_location=device, weights_only=False)
        accuracy = float(np.mean(obj["acc"])) if "acc" in obj else np.nan

        model = build_model(ARCH, num_targets=NUM_TARGETS,
                            input_len=attack.trace_len).to(device)
        load_checkpoint(model, obj["model"])
        model.eval()

        heads = [int(hyp.head[p]) for p in range(n_pairs)]
        logps = {h: np.empty((n, NUM_CLASSES), np.float32) for h in heads}
        with torch.no_grad():
            for start in range(0, n, 256):
                xb = traces[start:start + 256].unsqueeze(1).to(device)
                outs = model(xb)
                for h in heads:
                    logps[h][start:start + xb.size(0)] = \
                        torch.log_softmax(outs[h], 1).cpu().numpy()
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        rng = np.random.default_rng(1)
        per_pair, started = [], time.time()
        for p in range(n_pairs):
            a_np, b_np = hyp.partial_products(p)
            points, ge = pair_curve(logps[int(hyp.head[p])], a_np, b_np,
                                    hyp.true_index(p), hyp.q, device, rng)
            per_pair.append(ge)
            if p == 0:
                elapsed = time.time() - started
                print(f"  s{seed}: 1 pair {elapsed:.0f}s -> "
                      f"{n_pairs} pairs about {elapsed * n_pairs / 60:.0f} min")

        ge_mean = np.mean(per_pair, axis=0)
        curves.append((seed, points, ge_mean, accuracy))
        reached = points[ge_mean <= 1.0]
        print(f"  s{seed}: acc {accuracy:.1f}%  GE(1)={ge_mean[0]:,.0f}  "
              f"GE({points[-1]})={ge_mean[-1]:,.0f}  "
              f"first GE=1 at n={reached[0] if len(reached) else 'not reached'}")

        np.savez(os.path.join(FIG_DIR, f"key_recovery_{ARCH}_partial.npz"),
                 n_traces=np.array(curves[0][1]),
                 seeds=np.array([c[0] for c in curves]),
                 ge=np.array([c[2] for c in curves]),
                 accuracy=np.array([c[3] for c in curves]))

    # ------------------------------------------------------------- figure
    plt.figure(figsize=(4.5, 3.5))
    cmap = plt.get_cmap("tab10")
    for i, (seed, points, ge, _) in enumerate(curves):
        plt.plot(points, ge, color=cmap(i % 10), linewidth=1.3,
                 marker="o", markersize=3, label=f"seed {seed}")
    plt.axhline(1, color="k", linestyle=":", linewidth=1)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Number of Traces", fontsize=11)
    plt.ylabel("Guessing Entropy", fontsize=11)
    plt.xticks(fontsize=9)
    plt.yticks(fontsize=9)
    plt.title(f"{ARCH.capitalize()} key pair recovery", fontsize=11)
    plt.grid(True, linestyle=":", color="0.85")
    plt.legend(fontsize=7, ncol=2, framealpha=1.0, edgecolor="black")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, f"key_recovery_{ARCH}.eps")
    plt.savefig(path, format="eps", bbox_inches="tight")
    plt.close()
    print(f"\nsaved {path}")

    np.savez(os.path.join(FIG_DIR, f"key_recovery_{ARCH}.npz"),
             n_traces=np.array(curves[0][1]),
             seeds=np.array([c[0] for c in curves]),
             ge=np.array([c[2] for c in curves]),
             accuracy=np.array([c[3] for c in curves]))
    print(f"saved key_recovery_{ARCH}.npz")


if __name__ == "__main__":
    main()
