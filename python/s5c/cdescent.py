"""Coordinate-descent inner loop for the LASSO sub-problem.

Direct port of code/representation_learning/cdescentCycleC.c. Numba is
used in place of the MATLAB MEX file so we get C-like speed without a
build step.
"""

from __future__ import annotations

import numpy as np

try:
    from numba import njit
except ImportError:  # pragma: no cover - fall back to pure python if numba absent
    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]

        def _wrap(fn):
            return fn

        return _wrap


@njit(cache=True, fastmath=True)
def cdescent_cycle(X, r, w, idx, norms, threshold):
    """One cyclic sweep of coordinate descent over columns listed in ``idx``.

    Mirrors cdescentCycleC.c. Updates ``w`` and the residual ``r`` in place
    and also returns them so the call site reads naturally.

    Parameters
    ----------
    X : (m, p) float64
        Column-subset matrix (X_S in the paper).
    r : (m,) float64
        Current residual r = x_i - X_S w. Updated in place.
    w : (p,) float64
        Current weight vector. Updated in place.
    idx : (k,) int64
        Zero-based column indices to sweep over.
    norms : (p,) float64
        Squared column norms of X.
    threshold : float
        Soft-thresholding parameter (lambda).
    """
    m = X.shape[0]
    for jj in range(idx.shape[0]):
        j = idx[jj]
        wj_old = w[j]

        # gj = r + wj_old * X[:, j]
        # wj_new (pre-threshold) = X[:, j] . gj
        s = 0.0
        for i in range(m):
            s += X[i, j] * (r[i] + wj_old * X[i, j])

        if s >= 0.0:
            wj_new = (s - threshold) / norms[j] if (s - threshold) > 0.0 else 0.0
        else:
            wj_new = (s + threshold) / norms[j] if (-s - threshold) > 0.0 else 0.0

        delta = wj_new - wj_old
        if delta != 0.0:
            for i in range(m):
                r[i] -= delta * X[i, j]
            w[j] = wj_new

    return w, r
