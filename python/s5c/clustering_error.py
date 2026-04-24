"""Clustering error via the Hungarian algorithm.

Equivalent of code/utils/clustering_error.m, but uses
``scipy.optimize.linear_sum_assignment`` instead of the bundled MATLAB
Hungarian implementation.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def clustering_error(label: np.ndarray, orig_label: np.ndarray) -> float:
    """Misclassification rate after the optimal label permutation.

    Both inputs are 1-D integer arrays.  Labels need not start at 1; any
    relabelling is handled internally.
    """
    label = np.asarray(label).ravel()
    orig_label = np.asarray(orig_label).ravel()
    assert label.size == orig_label.size
    N = label.size

    orig_classes = np.unique(orig_label)
    pred_classes = np.unique(label)
    L = max(orig_classes.size, pred_classes.size)

    orig_idx = {c: i for i, c in enumerate(orig_classes)}
    pred_idx = {c: i for i, c in enumerate(pred_classes)}

    # Cost matrix: -count of (orig=i, pred=j) pairs.  We then maximize
    # agreement by minimizing the negative count.
    cost = np.zeros((L, L), dtype=np.int64)
    for o, p in zip(orig_label, label):
        cost[orig_idx[o], pred_idx[p]] -= 1

    row_ind, col_ind = linear_sum_assignment(cost)

    # Build a map: predicted-class -> matched original-class.
    pred_to_orig = {}
    for r, c in zip(row_ind, col_ind):
        if r < orig_classes.size and c < pred_classes.size:
            pred_to_orig[pred_classes[c]] = orig_classes[r]

    # Predictions not matched to any real class get a sentinel that won't
    # equal any orig label.
    sentinel = orig_classes.max() + 1
    relabelled = np.array([pred_to_orig.get(p, sentinel) for p in label])

    return float(np.sum(orig_label != relabelled) / N)
