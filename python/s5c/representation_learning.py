"""Selective Sampling-based Scalable Sparse Subspace Clustering (S5C).

Faithful Python port of code/representation_learning/representation_learning_S5C.m
and code/representation_learning/mylasso.m.

Reference:
    Shin Matsushima and Maria Brbic, "Selective Sampling-based Scalable
    Sparse Subspace Clustering", NeurIPS 2019.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy import sparse

from .cdescent import cdescent_cycle


@dataclass
class _Stats:
    reltol: float
    Lambda: float
    normsSt: np.ndarray  # (num_subsamples,)
    XSt: np.ndarray      # (m, num_subsamples)
    W: np.ndarray        # (num_subsamples, n) -- LASSO coefficients
    R: np.ndarray        # (m, n) residual = X - X*C
    time_for_CD: float = 0.0
    time_for_fit: float = 0.0


def representation_learning_s5c(
    X: np.ndarray,
    lam: float,
    num_subsamples: int,
    reltol: float = 1e-3,
    num_I_t: int = 1,
    seed: int = 1234,
    verbose: bool = False,
):
    """Learn the SSC representation using selective sub-sampling.

    Parameters
    ----------
    X : (m, n) ndarray
        Data matrix; each column is one data point.
    lam : float
        Sparsity hyper-parameter (lambda in the paper).
    num_subsamples : int
        Target number of subsamples to select.
    reltol : float
        Relative tolerance for the inner coordinate-descent loop.
    num_I_t : int
        Number of random queries used to approximate the subgradient at
        each selection step.
    seed : int
        Random seed.
    verbose : bool
        If True, print per-phase timing.

    Returns
    -------
    C : scipy.sparse.csr_matrix, shape (n, n)
        Sparse representation matrix.
    S_t : (num_St,) ndarray of int64
        Indices (0-based) of the selected subsamples.
    """
    rng = np.random.default_rng(seed)
    X = np.ascontiguousarray(X, dtype=np.float64)
    m, n = X.shape

    if n < num_subsamples:
        warnings.warn(
            "number of subsamples is larger than the number of datapoints; "
            "clipping to n"
        )
        num_subsamples = n

    # rand_idx is the (potentially padded) random ordering of column ids
    # used to draw I_t at each iteration.  Length must be max_t * num_I_t.
    max_t = num_subsamples * 2
    num_rand_idx = max_t * num_I_t
    if n >= num_rand_idx:
        rand_idx = rng.permutation(n)[:num_rand_idx]
    else:
        chunks = [rng.permutation(n) for _ in range(int(np.ceil(num_rand_idx / n)))]
        rand_idx = np.concatenate(chunks)[:num_rand_idx]

    S_t = np.zeros(num_subsamples, dtype=np.int64)
    num_St = 0

    norms_all = np.einsum("ij,ij->j", X, X)  # squared column norms

    stats = _Stats(
        reltol=reltol,
        Lambda=lam,
        normsSt=np.zeros(num_subsamples, dtype=np.float64),
        XSt=np.zeros((m, num_subsamples), dtype=np.float64),
        W=np.zeros((num_subsamples, n), dtype=np.float64),
        R=X.copy(),  # residual starts as X (since C = 0)
    )

    # ------------------------------------------------------------------
    # Phase 1: greedily select subsamples by approximated subgradient
    # ------------------------------------------------------------------
    t_sel = time.perf_counter()
    for t in range(max_t):
        I_t = rand_idx[t * num_I_t:(t + 1) * num_I_t]

        if num_St != 0:
            S_t_active = S_t[:num_St]
            for i in I_t:
                _mylasso(X, S_t_active, stats, int(i))

        # grad_L: (num_I_t, n).  Soft-threshold and mask out indices that
        # are already in S_t or that are themselves the queried columns.
        grad_L = stats.R[:, I_t].T @ X
        grad = np.minimum(grad_L + lam, np.maximum(0.0, grad_L - lam))
        if num_St != 0:
            grad[:, S_t[:num_St]] = 0.0
        grad[:, I_t] = 0.0

        scores = np.einsum("ij,ij->j", grad, grad)
        dSt = int(scores.argmax())
        if scores[dSt] != 0.0:
            S_t[num_St] = dSt
            stats.normsSt[num_St] = norms_all[dSt]
            stats.XSt[:, num_St] = X[:, dSt]
            num_St += 1
            if num_St == num_subsamples:
                break

    # Trim to the number we actually selected.
    S_t = S_t[:num_St]
    stats.normsSt = stats.normsSt[:num_St]
    stats.XSt = stats.XSt[:, :num_St]
    sel_time = time.perf_counter() - t_sel

    # ------------------------------------------------------------------
    # Phase 2: solve LASSO for every column with the final S_t
    # ------------------------------------------------------------------
    t_lasso = time.perf_counter()
    for i in range(n):
        _mylasso(X, S_t, stats, i)
    lasso_time = time.perf_counter() - t_lasso

    # ------------------------------------------------------------------
    # Phase 3: assemble sparse C with S_t rows
    # ------------------------------------------------------------------
    t_set = time.perf_counter()
    W_active = stats.W[:num_St, :]                 # (num_St, n)
    rows = np.tile(S_t, n)                          # column-major flatten of W_active
    cols = np.repeat(np.arange(n, dtype=np.int64), num_St)
    data = W_active.flatten(order="F")
    C = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
    set_time = time.perf_counter() - t_set

    if verbose:
        print(
            f"S5C timings (s):\n"
            f"  1. selection      : {sel_time:.3f}\n"
            f"  2. final lasso    : {lasso_time:.3f}\n"
            f"  3. fit (in lasso) : {stats.time_for_fit:.3f}\n"
            f"  4. CD  (in fit)   : {stats.time_for_CD:.3f}\n"
            f"  5. set C          : {set_time:.3f}"
        )

    return C, S_t


# ----------------------------------------------------------------------
# Inner LASSO solve for a single column i, using the warm-started
# coefficients stored in ``stats.W[:|S|, i]``.  Direct port of mylasso.m
# and lassoFit.
# ----------------------------------------------------------------------
def _mylasso(X: np.ndarray, S: np.ndarray, stats: _Stats, i: int) -> None:
    w_size = S.shape[0]
    XS = stats.XSt[:, :w_size]
    norms_S = stats.normsSt[:w_size]
    w_vec = stats.W[:w_size, i].copy()
    r = stats.R[:, i].copy()

    # diag_col: position(s) of i in S, so we can skip the self-coordinate
    # during CD (the LASSO is constrained to w_j = 0 for j == i).
    diag_col = np.flatnonzero(S == i)

    t_fit = time.perf_counter()
    w_vec, r, t_cd = _lasso_fit(
        diag_col=diag_col,
        X=XS,
        w=w_vec,
        r=r,
        threshold=stats.Lambda,
        reltol=stats.reltol,
        norms=norms_S,
    )
    fit_elapsed = time.perf_counter() - t_fit

    stats.R[:, i] = r
    stats.W[:w_size, i] = w_vec
    stats.time_for_CD += t_cd
    stats.time_for_fit += fit_elapsed


def _lasso_fit(diag_col, X, w, r, threshold, reltol, norms):
    """Active-set + KKT-violator coordinate descent (mirrors lassoFit)."""
    p = w.shape[0]
    if p == 0:
        return w, r, 0.0

    time_for_CD = 0.0
    active = w != 0.0
    w_old = w.copy()

    while True:
        idx = _setdiff_active(active, diag_col)
        t0 = time.perf_counter()
        cdescent_cycle(X, r, w, idx, norms, threshold)
        time_for_CD += time.perf_counter() - t0
        active = w != 0.0

        if _converged(w, w_old, reltol):
            # Active set stable -> sweep all predictors looking for KKT
            # violators.  If none are found we're done.
            w_old = w.copy()
            potentially_active = np.abs(r @ X) > threshold
            if potentially_active.any():
                new_active = active | potentially_active
                idx = _setdiff_active(new_active, diag_col)
                t0 = time.perf_counter()
                cdescent_cycle(X, r, w, idx, norms, threshold)
                time_for_CD += time.perf_counter() - t0
                new_active = w != 0.0
            else:
                new_active = active

            if np.array_equal(new_active, active):
                break
            active = new_active

            if _converged(w, w_old, reltol):
                break

        w_old = w.copy()

    return w, r, time_for_CD


def _setdiff_active(active_mask: np.ndarray, exclude: np.ndarray) -> np.ndarray:
    idx = np.flatnonzero(active_mask)
    if exclude.size:
        idx = np.setdiff1d(idx, exclude, assume_unique=True)
    return idx.astype(np.int64, copy=False)


def _converged(w: np.ndarray, w_old: np.ndarray, reltol: float) -> bool:
    return np.max(np.abs(w - w_old) / (1.0 + np.abs(w_old))) < reltol
