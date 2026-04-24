"""Spectral clustering for S5C.

Port of code/spectral_clustering/spectral_clustering_S5C.m and
code/spectral_clustering/orth_iter.m.  Uses orthogonal iteration on
``lambda_max * I - L_sym`` (largest eigenvectors there correspond to the
smallest of the normalized Laplacian).
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.cluster import KMeans


def orth_iter(A, K: int, err_tol: float = 1e-3, max_iter: int = 100, seed: int = 1):
    """Block power iteration for the K largest eigenvectors of ``A``.

    Mirrors orth_iter.m: random init, QR after each multiply, Frobenius
    convergence test scaled by sqrt(N*K).
    """
    rng = np.random.default_rng(seed)
    N = A.shape[0]

    Q, _ = np.linalg.qr(rng.random((N, K)))
    Q_prev = Q.copy()

    for _ in range(max_iter):
        Z = A @ Q
        Q, _ = np.linalg.qr(Z)
        if np.linalg.norm(Q - Q_prev, ord="fro") / np.sqrt(N * K) < err_tol:
            break
        Q_prev = Q

    return Q


def spectral_clustering_s5c(
    W,
    K: int,
    err_tol: float = 1e-5,
    max_iter: int = 100,
    seed: int = 1234,
    verbose: bool = False,
):
    """Spectral clustering on an affinity matrix W.

    Parameters
    ----------
    W : (N, N) sparse or dense matrix
        Symmetric non-negative affinity matrix.
    K : int
        Number of clusters.

    Returns
    -------
    A : (N,) ndarray of int
        Cluster assignments in 1..K (matches the MATLAB convention).
    """
    if not sparse.issparse(W):
        W = sparse.csr_matrix(W)

    N_old = W.shape[0]
    deg = np.asarray(W.sum(axis=1)).ravel()
    connected = deg != 0

    W_c = W[connected][:, connected]
    deg_c = np.asarray(W_c.sum(axis=1)).ravel()
    N = W_c.shape[0]

    # Normalized Laplacian L_sym = I - D^{-1/2} W D^{-1/2}
    inv_sqrt = 1.0 / np.sqrt(deg_c)
    DN = sparse.diags(inv_sqrt)
    Lsym = sparse.eye(N) - DN @ W_c @ DN

    # Largest eigenvectors of (lambda_max * I - L_sym) == smallest of L_sym
    lambda_max = 2.0
    Lmax = lambda_max * sparse.eye(N) - Lsym

    U = orth_iter(Lmax, K, err_tol=err_tol, max_iter=max_iter, seed=1)

    # Row-normalize (the spectral-embedding step that follows the random
    # walk normalization in Ng-Jordan-Weiss).
    row_norms = np.linalg.norm(U, axis=1, keepdims=True)
    row_norms[row_norms == 0] = 1.0
    U = U / row_norms

    rng = np.random.default_rng(seed)
    km = KMeans(n_clusters=K, n_init=20, max_iter=200, random_state=int(rng.integers(2**31 - 1)))
    labels_c = km.fit_predict(U) + 1  # 1-indexed to match MATLAB

    if labels_c.size != N_old:
        # Re-insert isolated nodes with random labels (matches MATLAB).
        labels = rng.integers(1, K + 1, size=N_old)
        labels[connected] = labels_c
    else:
        labels = labels_c

    return labels
