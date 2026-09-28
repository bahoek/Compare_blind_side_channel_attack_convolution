# Blind Side-Channel Analysis of Software-based CRYSTALS-Kyber

Code for the paper **"Benchmarking Modern Deep Learning Architectures for Blind Side-Channel Attacks on Software-based Kyber"** (submitted to IEEE Embedded Systems Letters).

This repository contains the four benchmarked 1D-CNN architectures, the training scripts, and the evaluation scripts used in the paper.

---

## Contents

| File | Purpose |
|---|---|
| `models.py` | The four architectures (Modified ASCAD, ResNet-18, MobileNet V2, EfficientNet-B0), in multi-head and single-head form |
| `dataset.py` | `.npz` loader and the expected array format |
| `train_measured.py` | Training on the locally measured dataset (78 heads) |
| `train_open.py` | Training on the public dataset (1 head) |
| `eval_measured.py` | Single-trace Hamming-weight GE / SR, local dataset |
| `eval_open.py` | Single-trace Hamming-weight GE / SR, open dataset |
| `eval_key_recovery.py` | Pairwise key-recovery GE over the full candidate space |

All results in the paper use 10 random seeds per configuration, except the non-convergent EfficientNet-B0 runs on the local dataset, which used 5-7 seeds at each of four learning rates.

## Profiling target

The classification target is the 17-class Hamming weight (HW 0-16) of the accumulated BaseMul output. The local dataset profiles 78 coefficients concurrently; the open dataset profiles 2.

Training configuration (Table I of the paper): Adam, lr 1e-4, batch 32, 100 epochs, patience 15, early stopping at 0.5% improvement, hard stop at 95% validation accuracy. Two documented exceptions apply on the local dataset: MobileNet V2 uses learning rate 1e-3 and batch size 128, and EfficientNet-B0 was swept over four learning rates (1e-4 to 3e-3).

---

## Input format

The scripts read `.npz` files whose traces are already cropped to the region of interest **and already normalized**. They do not normalize internally.

| Array | Shape | Notes |
|---|---|---|
| `traces` | `(N, L)` | `L` = 40,000 (local) or 50,000 (open) |
| `labels` | `(N, 78)` local, `(N,)` open | Hamming weight, values 0-16 |

`eval_key_recovery.py` additionally reads a `key_hypotheses.npz` holding the precomputed candidate table and the per-trace row indices.

### Normalization

The released traces are raw. Both datasets in the paper are Z-score standardized **per sample position**, with mean and standard deviation computed on the **profiling (training) split only** and applied unchanged to all three splits. Apply this before training or evaluation:

```python
import numpy as np

Xtr = np.load("kyber_train_crop.npz")["traces"].astype(np.float32)
mu  = Xtr.mean(axis=0)
sd  = Xtr.std(axis=0)
sd[sd == 0] = 1.0

Xtr = (Xtr - mu) / sd          # reuse the same mu/sd for valid and test
```

Do not recompute `mu`/`sd` on the validation or test split, and do not normalize per trace. Either change alters the reported numbers.

The acquisition, alignment, and labeling pipeline is internal to our laboratory and is not part of this release. Section III of the paper describes the measurement setup and the leakage model in full; the open dataset of Rezaeezade et al. is distributed with the inputs needed to derive its labels, so the open-dataset results can be reproduced end to end from public material.

## Dataset

The locally measured traces are available at:

**https://doi.org/10.5281/zenodo.22961832**

Traces were collected on an STM32F415 (CW308 UFO board) running the unprotected reference implementation of Kyber-768 (adapted from PQClean), captured with a ChipWhisperer-Husky Plus at 25 dB LNA gain. Each trace uses a fresh uniformly random 1088-byte ciphertext. The region of interest spans 40,000 samples around the BaseMul operation in the NTT domain, stored as raw unscaled signals; normalize them as described above before use.

| File | Traces |
|---|---|
| `kyber_train_crop.npz` | 64,000 |
| `kyber_valid_crop.npz` | 16,000 |
| `kyber_test_crop.npz` | 20,000 |

### Reassembling the files

Each file is split into ~300 MB parts on Zenodo. After downloading all parts, concatenate them in order:

```bash
cat kyber_train_crop.npz.part* > kyber_train_crop.npz
cat kyber_valid_crop.npz.part* > kyber_valid_crop.npz
cat kyber_test_crop.npz.part*  > kyber_test_crop.npz
```

The shell expands `part*` in lexicographic order, which matches the split order.

---

## Citation

If you use this code or the dataset, please cite the dataset record:

```
S.-W. Bae, "Blind SCA on Software-based CRYSTALS-Kyber: Power Trace Dataset,"
Zenodo, 2026. doi: 10.5281/zenodo.22961832
```
