"""K-way clustering by recursive modularity bisection cast as a QUBO.

Each bisection step solves an Ising problem of size N over {-1, +1}:

    H(s) = - sum_{i,j} B[i,j] s_i s_j

where ``B[i,j] = W[i,j] - deg_i deg_j / (2m)`` is the modularity matrix
(Newman, 2006).  Maximising modularity in two communities is equivalent
to minimising H, which has a direct Ising representation and runs
unmodified on adiabatic / quantum-annealing hardware.

We recurse on the resulting partitions until we have ``K`` groups.
Recursive 2-way needs only N spins per call (not N*K), which is the
only formulation that fits on present-day annealers for non-trivial N.

The default backend is the classical ``dwave.samplers`` simulated
annealer (no hardware required).  Pass ``use_dwave=True`` to dispatch to
the actual D-Wave QPU through ``EmbeddingComposite`` -- you'll need
``dwave-system`` installed and a configured API token.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse


def _modularity_matrix(W: np.ndarray) -> np.ndarray:
    """Newman's modularity matrix B = W - dd^T / (2m)."""
    deg = W.sum(axis=1)
    two_m = deg.sum()
    if two_m <= 0:
        return np.zeros_like(W)
    return W - np.outer(deg, deg) / two_m


def _bisect(B: np.ndarray, sampler, num_reads: int, seed: int) -> np.ndarray:
    """Solve the 2-community Ising problem and return s in {-1, +1}^N."""
    import dimod

    N = B.shape[0]

    # Energy = - sum_{i,j} B[i,j] s_i s_j
    #        = -2 sum_{i<j} B[i,j] s_i s_j  + const  (B symmetric, s_i^2 = 1)
    # In dimod ordering: J_{ij} for i<j is the coefficient of s_i s_j.
    # Drop the factor of 2 -- doesn't change argmin.
    iu = np.triu_indices(N, k=1)
    j_vals = -B[iu]
    nz = j_vals != 0.0
    J = {(int(i), int(j)): float(v)
         for i, j, v in zip(iu[0][nz], iu[1][nz], j_vals[nz])}
    h = {i: 0.0 for i in range(N)}

    bqm = dimod.BinaryQuadraticModel(h, J, vartype="SPIN")

    kwargs = {"num_reads": num_reads}
    # Only the classical samplers expose `seed`; the QPU sampler does
    # not, so pass it conditionally.
    try:
        sampleset = sampler.sample(bqm, seed=seed, **kwargs)
    except TypeError:
        sampleset = sampler.sample(bqm, **kwargs)

    best = sampleset.first.sample
    return np.array([best[i] for i in range(N)], dtype=np.int8)


def _make_sampler(use_dwave: bool):
    if use_dwave:
        from dwave.system import DWaveSampler, EmbeddingComposite
        return EmbeddingComposite(DWaveSampler())
    from dwave.samplers import SimulatedAnnealingSampler
    return SimulatedAnnealingSampler()


def spectral_clustering_qubo(
    W,
    K: int,
    *,
    use_dwave: bool = False,
    num_reads: int = 100,
    seed: int = 1,
):
    """Cluster the affinity matrix ``W`` into K groups via QUBO bisection.

    Parameters
    ----------
    W : (N, N) ndarray or scipy sparse matrix
        Symmetric non-negative affinity matrix produced by S5C
        (``W = |C| + |C|.T``).
    K : int
        Number of clusters.
    use_dwave : bool
        If True, dispatch to ``EmbeddingComposite(DWaveSampler())``.
        Requires ``dwave-system`` and a valid API token / endpoint.
        Default False uses ``dwave.samplers.SimulatedAnnealingSampler``.
    num_reads : int
        Number of independent reads / annealing runs per bisection.
    seed : int
        Seed for the classical sampler (ignored on hardware).

    Returns
    -------
    labels : (N,) ndarray of int
        Cluster assignments in 1..K (matches the MATLAB convention used
        by ``spectral_clustering_s5c``).
    """
    if sparse.issparse(W):
        W = W.toarray()
    W = np.asarray(W, dtype=np.float64)
    N = W.shape[0]

    # Drop isolated nodes (rows summing to zero) and reinsert later with
    # random labels, the same way ``spectral_clustering_s5c`` does.
    deg = W.sum(axis=1)
    connected = deg != 0
    if not connected.all():
        idx_conn = np.flatnonzero(connected)
        W_conn = W[np.ix_(idx_conn, idx_conn)]
    else:
        idx_conn = np.arange(N)
        W_conn = W

    sampler = _make_sampler(use_dwave)

    # Iterative bisection: split the largest group until we have K groups.
    groups = [np.arange(W_conn.shape[0])]
    while len(groups) < K:
        groups.sort(key=lambda g: -len(g))
        idx = groups.pop(0)

        if len(idx) <= 1:
            groups.append(idx)
            break

        W_sub = W_conn[np.ix_(idx, idx)]
        B = _modularity_matrix(W_sub)
        s = _bisect(B, sampler, num_reads=num_reads, seed=seed)

        left = idx[s > 0]
        right = idx[s < 0]
        if left.size == 0 or right.size == 0:
            # Annealer collapsed to a single side; cannot split further.
            groups.append(idx)
            break

        groups.append(left)
        groups.append(right)

    labels_conn = np.zeros(W_conn.shape[0], dtype=np.int64)
    for k, idx in enumerate(groups, start=1):
        labels_conn[idx] = k

    if connected.all():
        labels = labels_conn
    else:
        rng = np.random.default_rng(seed)
        labels = rng.integers(1, K + 1, size=N)
        labels[idx_conn] = labels_conn

    return labels
