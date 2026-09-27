"""
1D CNN architectures for profiled side-channel analysis.

Four backbones adapted to 1-D power traces, each available in two forms:

  * multi-head  (num_targets > 1) : one classification head per target
                                    coefficient, sharing a single trunk.
                                    Used for the measured STM32F4 dataset,
                                    where a single trace covers many
                                    coefficients of the polynomial product.
  * single-head (num_targets = 1) : one classification head.
                                    Used for the public reference dataset,
                                    where a trace covers one coefficient pair.

All heads predict the Hamming weight of the 16-bit accumulator word
(17 classes, HW 0..16).

Head design notes
-----------------
* Each head is BatchNorm -> Dropout -> Linear.  The BatchNorm normalises the
  trunk feature scale; without it the gradient reaching the trunk is too small
  for the deeper backbones to train on this data.
* `init_bias` sets the final Linear bias to log(class prior) and leaves the
  weights at their default initialisation.  Shrinking the final weights (a
  common default of normal_(0, 0.01)) scales dL/dh by the same factor and
  stalls trunk learning; starting from the prior achieves the intended
  "neutral start" without that side effect.
* ASCAD keeps the original flatten + shared fully-connected stage rather than
  global average pooling: the leakage of different coefficients sits at
  different time offsets, and pooling over time removes that positional
  information.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_CLASSES = 17                 # Hamming weight of a 16-bit word: 0..16
MOBILENET_STEM_POOL = True       # keep stem downsampling identical to ResNet


# ----------------------------------------------------------------------
# shared head helpers
# ----------------------------------------------------------------------
def make_heads(feat_dim, num_targets, dropout=0.4):
    return nn.ModuleList([
        nn.Sequential(nn.BatchNorm1d(feat_dim),
                      nn.Dropout(dropout),
                      nn.Linear(feat_dim, NUM_CLASSES))
        for _ in range(num_targets)])


@torch.no_grad()
def init_head_bias(linear, labels_1d):
    """Final-layer bias <- log(class prior).  Weights untouched."""
    cnt = np.bincount(np.asarray(labels_1d), minlength=NUM_CLASSES).astype(np.float64)
    prior = (cnt + 1e-6) / cnt.sum()
    linear.bias.copy_(torch.from_numpy(np.log(prior)).float())


class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


# ----------------------------------------------------------------------
# ResNet-18 (1D)
# ----------------------------------------------------------------------
class ResidualBlock1d(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(cout)
        self.conv2 = nn.Conv1d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm1d(cout)
        self.sc = nn.Sequential()
        if stride != 1 or cin != cout:
            self.sc = nn.Sequential(nn.Conv1d(cin, cout, 1, stride, bias=False),
                                    nn.BatchNorm1d(cout))

    def forward(self, x):
        o = F.relu(self.bn1(self.conv1(x)))
        o = self.bn2(self.conv2(o))
        return F.relu(o + self.sc(x))


class ResNet18(nn.Module):
    def __init__(self, num_targets=1, input_len=None):
        super().__init__()
        self.num_targets = num_targets
        self.in_ch = 64
        self.stem = nn.Sequential(
            nn.Conv1d(1, 64, 11, 2, 5, bias=False), nn.BatchNorm1d(64),
            nn.ReLU(), nn.MaxPool1d(5, 5))
        self.layer1 = self._mk(64, 2, 1)
        self.layer2 = self._mk(128, 2, 2)
        self.layer3 = self._mk(256, 2, 2)
        self.layer4 = self._mk(512, 2, 2)
        self.classifiers = make_heads(512, num_targets)

    def _mk(self, cout, n, stride):
        layers = []
        for s in [stride] + [1] * (n - 1):
            layers.append(ResidualBlock1d(self.in_ch, cout, s))
            self.in_ch = cout
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = x.mean(dim=2)
        return [c(x) for c in self.classifiers]

    @torch.no_grad()
    def init_bias(self, Y):
        _init_all(self.classifiers, Y)


# ----------------------------------------------------------------------
# ASCAD-style CNN (flatten + shared FC, VGG lineage)
# ----------------------------------------------------------------------
class ASCAD(nn.Module):
    def __init__(self, num_targets=1, input_len=40000):
        super().__init__()
        self.num_targets = num_targets
        self.feat = nn.Sequential(
            nn.Conv1d(1, 64, 65, 2, 32), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
            nn.Conv1d(64, 128, 33, 2, 16), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Conv1d(128, 256, 17, 2, 8), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.4))
        with torch.no_grad():
            fdim = self.feat(torch.zeros(1, 1, input_len)).flatten(1).shape[1]
        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fdim, 1024), nn.BatchNorm1d(1024), nn.ReLU(), nn.Dropout(0.4))
        self.classifiers = nn.ModuleList([
            nn.Sequential(nn.Linear(1024, 1024), nn.BatchNorm1d(1024), nn.ReLU(),
                          nn.Dropout(0.3), nn.Linear(1024, NUM_CLASSES))
            for _ in range(num_targets)])

    def forward(self, x):
        f = self.shared(self.feat(x))
        return [c(f) for c in self.classifiers]

    @torch.no_grad()
    def init_bias(self, Y):
        _init_all(self.classifiers, Y)


# ----------------------------------------------------------------------
# MobileNetV2 (1D)
# ----------------------------------------------------------------------
class DWSep1d(nn.Module):
    def __init__(self, cin, cout, k, s, p):
        super().__init__()
        self.dw = nn.Conv1d(cin, cin, k, s, p, groups=cin, bias=False)
        self.pw = nn.Conv1d(cin, cout, 1, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm1d(cin)
        self.bn2 = nn.BatchNorm1d(cout)

    def forward(self, x):
        x = F.relu6(self.bn1(self.dw(x)))
        return F.relu6(self.bn2(self.pw(x)))


class InvRes1d(nn.Module):
    def __init__(self, cin, cout, stride, t):
        super().__init__()
        hid = int(cin * t)
        self.use_res = (stride == 1 and cin == cout)
        layers = []
        if t != 1:
            layers += [nn.Conv1d(cin, hid, 1, 1, 0, bias=False),
                       nn.BatchNorm1d(hid), nn.ReLU6()]
        layers.append(DWSep1d(hid, cout, 3, stride, 1))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.conv(x) if self.use_res else self.conv(x)


class MobileNetV2(nn.Module):
    CFG = [[1, 16, 1, 1], [6, 24, 2, 2], [6, 32, 3, 2], [6, 64, 4, 2],
           [6, 96, 3, 1], [6, 160, 3, 2], [6, 320, 1, 1]]

    def __init__(self, num_targets=1, input_len=None):
        super().__init__()
        self.num_targets = num_targets
        stem = [nn.Conv1d(1, 32, 11, 2, 5, bias=False), nn.BatchNorm1d(32), nn.ReLU6()]
        if MOBILENET_STEM_POOL:
            stem.append(nn.MaxPool1d(5, 5))
        self.stem = nn.Sequential(*stem)
        layers, cin = [], 32
        for t, c, n, s in self.CFG:
            for i in range(n):
                layers.append(InvRes1d(cin, c, s if i == 0 else 1, t))
                cin = c
        self.features = nn.Sequential(*layers)
        self.proj = nn.Sequential(nn.Conv1d(cin, 1280, 1, 1, 0, bias=False),
                                  nn.BatchNorm1d(1280), nn.ReLU6())
        self.classifiers = make_heads(1280, num_targets)

    def forward(self, x):
        x = self.proj(self.features(self.stem(x))).mean(dim=2)
        return [c(x) for c in self.classifiers]

    @torch.no_grad()
    def init_bias(self, Y):
        _init_all(self.classifiers, Y)


# ----------------------------------------------------------------------
# EfficientNet-B0 (1D)
# ----------------------------------------------------------------------
class SEBlock(nn.Module):
    def __init__(self, ch, r=4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(nn.Linear(ch, ch // r, bias=False), SiLU(),
                                nn.Linear(ch // r, ch, bias=False), nn.Sigmoid())

    def forward(self, x):
        s = self.fc(self.pool(x).squeeze(-1)).unsqueeze(-1)
        return x * s


class MBConv(nn.Module):
    def __init__(self, cin, cout, t, k, stride, r=4):
        super().__init__()
        hid = cin * t
        self.use_res = (stride == 1 and cin == cout)
        layers = []
        if t != 1:
            layers += [nn.Conv1d(cin, hid, 1, bias=False), nn.BatchNorm1d(hid), SiLU()]
        layers += [nn.Conv1d(hid, hid, k, stride, k // 2, groups=hid, bias=False),
                   nn.BatchNorm1d(hid), SiLU(), SEBlock(hid, r),
                   nn.Conv1d(hid, cout, 1, bias=False), nn.BatchNorm1d(cout)]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.conv(x) if self.use_res else self.conv(x)


class EfficientNetB0(nn.Module):
    CFG = [[1, 16, 1, 1, 3], [6, 24, 2, 2, 3], [6, 40, 2, 2, 5], [6, 80, 3, 2, 3],
           [6, 112, 3, 1, 5], [6, 192, 4, 2, 5], [6, 320, 1, 1, 3]]

    def __init__(self, num_targets=1, input_len=None):
        super().__init__()
        self.num_targets = num_targets
        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, 11, 2, 5, bias=False), nn.BatchNorm1d(32),
            SiLU(), nn.MaxPool1d(5, 5))
        layers, cin = [], 32
        for t, c, n, s, k in self.CFG:
            for i in range(n):
                layers.append(MBConv(cin, c, t, k, s if i == 0 else 1))
                cin = c
        self.features = nn.Sequential(*layers)
        self.proj = nn.Sequential(nn.Conv1d(cin, 1280, 1, bias=False),
                                  nn.BatchNorm1d(1280), SiLU())
        self.classifiers = make_heads(1280, num_targets)

    def forward(self, x):
        x = self.proj(self.features(self.stem(x))).mean(dim=2)
        return [c(x) for c in self.classifiers]

    @torch.no_grad()
    def init_bias(self, Y):
        _init_all(self.classifiers, Y)


# ----------------------------------------------------------------------
def _init_all(classifiers, Y):
    """Y: (N,) for a single head, (N, num_targets) for multi-head."""
    Y = np.asarray(Y)
    if Y.ndim == 1:
        Y = Y[:, None]
    for i, clf in enumerate(classifiers):
        init_head_bias(clf[-1], Y[:, i])


ARCHITECTURES = {"resnet": ResNet18, "ascad": ASCAD,
                 "mobilenet": MobileNetV2, "efficientnet": EfficientNetB0}


def build_model(arch, num_targets=1, input_len=40000):
    if arch not in ARCHITECTURES:
        raise KeyError(f"unknown architecture '{arch}'; "
                       f"choose from {sorted(ARCHITECTURES)}")
    return ARCHITECTURES[arch](num_targets=num_targets, input_len=input_len)


def load_checkpoint(model, state_dict):
    """Accept checkpoints saved under either shortcut-attribute name."""
    renamed = {k.replace(".shortcut.", ".sc."): v for k, v in state_dict.items()}
    model.load_state_dict(renamed)
    return model
