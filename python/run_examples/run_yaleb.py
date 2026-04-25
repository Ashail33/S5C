"""Run S5C on the Extended Yale B dataset.

Mirrors code/run_examples/run_yaleb.m.
"""

from __future__ import annotations

import os
import sys

import numpy as np
from scipy.io import loadmat

# Allow running this file directly from the run_examples directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from s5c import run_s5c  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.normpath(os.path.join(HERE, "..", "..", "data", "YaleBCrop025.mat"))


def main():
    mat = loadmat(DATA_PATH)
    Y0 = mat["Y"]  # shape (p, n_per_subject, L)

    L = 38
    n_per = 64

    # Stack the L subjects column-wise: Y has shape (p, n_per * L).
    Y = np.concatenate([Y0[:, :, i] for i in range(L)], axis=1)

    # Ground-truth labels: 1..L, each repeated n_per times.
    A0 = np.repeat(np.arange(1, L + 1), n_per)

    CE, ET = run_s5c(Y, A0, L, 20 * L)

    print("CE =", CE)
    print("ET =", ET)


if __name__ == "__main__":
    main()
