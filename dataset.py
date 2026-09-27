"""
Pre-compiled side-channel package loader.

NOTE
----
Trace acquisition, trigger alignment, NTT/basemul index mapping and label
derivation are performed by a separate in-house toolchain that is not part of
this release, in line with institutional policy on measurement infrastructure.
This loader therefore assumes that the power traces have already been
acquired, aligned, amplitude-normalised and paired with their target labels,
and compiled into a standardised matrix package.  The leakage model, the
labelling target and the normalisation procedure are specified in the paper.

Package format (.npz)
---------------------
  traces : float32, (N, L)
           Amplitude-normalised traces.  Normalisation is per sample position
           (axis 0) using the statistics of the profiling set, applied
           identically to the profiling, validation and attack packages.
           L = 40000 for the measured package, 50000 for the public
           reference package.

  labels : int64, (N, T) multi-target, or (N,) single-target
           Hamming weight of the 16-bit accumulator word, values in 0..16.
           T = 78 for the measured package (one column per coefficient),
           single-target for the public reference package.

Loading the whole package onto the accelerator keeps the input pipeline out of
the training loop; pass `device=None` to keep it in host memory instead.
"""

import os

import numpy as np
import torch
from torch.utils.data import Dataset

NUM_CLASSES = 17


class SCAPackage(Dataset):
    def __init__(self, npz_path, device=None):
        if not os.path.exists(npz_path):
            raise FileNotFoundError(
                f"package not found: {npz_path}\n"
                f"Place the pre-compiled .npz packages in the data directory; "
                f"see the format description in dataset.py.")
        data = np.load(npz_path, mmap_mode="r")
        traces = np.asarray(data["traces"], dtype=np.float32)
        labels = np.asarray(data["labels"], dtype=np.int64)
        if labels.ndim == 1:
            labels = labels[:, None]
        if len(traces) != len(labels):
            raise ValueError(f"{npz_path}: {len(traces)} traces vs "
                             f"{len(labels)} labels")

        self.traces = torch.from_numpy(traces)
        self.labels = torch.from_numpy(labels)
        if device is not None:
            self.traces = self.traces.to(device)
            self.labels = self.labels.to(device)

        self.trace_len = self.traces.shape[1]
        self.num_targets = self.labels.shape[1]

    def __len__(self):
        return len(self.traces)

    def __getitem__(self, idx):
        return self.traces[idx].unsqueeze(0), self.labels[idx]

    def labels_numpy(self):
        return self.labels.cpu().numpy()

    def majority_class_accuracy(self):
        """Accuracy of always predicting the most frequent class, per target."""
        Y = self.labels_numpy()
        return np.array([np.bincount(Y[:, i], minlength=NUM_CLASSES).max() / len(Y) * 100
                         for i in range(self.num_targets)])

    def prior_entropy(self):
        """Cross-entropy of the class prior, averaged over targets."""
        Y = self.labels_numpy()
        p = np.stack([np.bincount(Y[:, i], minlength=NUM_CLASSES) / len(Y)
                      for i in range(self.num_targets)])
        return float(np.mean(-(p * np.log(p + 1e-12)).sum(1)))

    def summary(self):
        return (f"{len(self)} traces x {self.trace_len} samples, "
                f"{self.num_targets} target(s), "
                f"HW range {int(self.labels.min())}..{int(self.labels.max())}")
