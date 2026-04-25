"""Top-level S5C driver.

Equivalent of code/run_S5C.m: sweeps lambda over 2^-1 .. 2^-10, runs the
selection + spectral pipeline at each value, and returns the per-lambda
clustering errors and elapsed times.
"""

from __future__ import annotations

import time

import numpy as np

from .clustering_error import clustering_error
from .representation_learning import representation_learning_s5c
from .spectral_clustering import spectral_clustering_s5c


def _normalize_columns(Y: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(Y, axis=0)
    norms[norms == 0] = 1.0
    return Y / norms


def run_s5c(Y: np.ndarray, A0: np.ndarray, L: int, num_subsamples: int,
            *, verbose: bool = False):
    """Run the full S5C pipeline across the standard lambda grid.

    Parameters
    ----------
    Y : (m, N) ndarray
        Data matrix; columns are points.
    A0 : (N,) ndarray
        Ground-truth labels.
    L : int
        Number of clusters.
    num_subsamples : int
        Number of subsamples to select per run.

    Returns
    -------
    clustering_errors : ndarray of length 10
    elapsed_times     : ndarray of length 10
    """
    print("Running S5C..")
    Y_n = _normalize_columns(np.asarray(Y, dtype=np.float64))

    clustering_errors = np.zeros(10)
    elapsed_times = np.zeros(10)

    for k, plambda in enumerate(range(1, 11)):
        lam = 2.0 ** (-plambda)
        print(f"lambda = 2^-{plambda}")

        t0 = time.perf_counter()
        C, _ = representation_learning_s5c(Y_n, lam, num_subsamples,
                                           verbose=verbose)
        t_repr = time.perf_counter() - t0

        W = np.abs(C) + np.abs(C).T

        t0 = time.perf_counter()
        A = spectral_clustering_s5c(W, L, verbose=verbose)
        t_spec = time.perf_counter() - t0

        elapsed_times[k] = t_repr + t_spec
        clustering_errors[k] = clustering_error(A, A0)

    return clustering_errors, elapsed_times
