"""
Exact, fast fidelity quantum kernel for ZZ-type feature maps (pure NumPy).

Why this is valid
-----------------
Qiskit's ZZFeatureMap is  [H^n  ->  diagonal phase U(x)]  repeated `reps` times.
(P(2 x_i) on each qubit, and CX - P(2(pi-x_i)(pi-x_j)) - CX on each entangled pair;
CX-P-CX is diagonal: it adds phase theta when bit_i XOR bit_j = 1.)
So the state of one sample is a 2^n vector we can build directly, and the
FidelityQuantumKernel entry  |<psi(x)|psi(y)>|^2  is one matrix product.
This reproduces the same quantity Qiskit's statevector-based kernel computes,
exactly (no shot noise), in seconds instead of hours.
tests/test_qkernel.py checks it against an independent gate-by-gate simulator.
"""
import numpy as np

CHUNK = 8192


def _bits(n):
    idx = np.arange(2 ** n)
    return ((idx[None, :] >> np.arange(n)[:, None]) & 1).astype(np.float64)  # (n, 2^n)


def _hadamard_all(n):
    H = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0)
    W = np.array([[1.0]])
    for _ in range(n):
        W = np.kron(W, H)
    return W


def entangler_pairs(n, entanglement="linear"):
    if entanglement == "linear":
        return [(i, i + 1) for i in range(n - 1)]
    if entanglement == "circular":
        return [(i, i + 1) for i in range(n - 1)] + ([(n - 1, 0)] if n > 2 else [])
    if entanglement == "full":
        return [(i, j) for i in range(n) for j in range(i + 1, n)]
    raise ValueError(f"unknown entanglement {entanglement!r}")


def zz_statevectors(X, reps=2, entanglement="linear", bandwidth=1.0):
    """Statevectors (N, 2^n) of ZZFeatureMap(feature_dimension=n, reps, entanglement)
    applied to rows of X (angles). `bandwidth` rescales the data first (x -> bw * x)."""
    X = np.asarray(X, dtype=np.float64) * float(bandwidth)
    N, n = X.shape
    D = 2 ** n
    B = _bits(n)
    W = _hadamard_all(n)
    pairs = entangler_pairs(n, entanglement)
    parity = {p: B[p[0]] + B[p[1]] - 2.0 * B[p[0]] * B[p[1]] for p in pairs}
    out = np.empty((N, D), dtype=np.complex128)
    for s in range(0, N, CHUNK):
        x = X[s:s + CHUNK]
        phase = 2.0 * x @ B
        for (i, j) in pairs:
            coef = 2.0 * (np.pi - x[:, i]) * (np.pi - x[:, j])
            phase += coef[:, None] * parity[(i, j)][None, :]
        U = np.exp(1j * phase)
        psi = U / np.sqrt(D)               # H^n |0...0> is uniform, then phase layer 1
        for _ in range(1, reps):
            psi = (psi @ W) * U            # H^n then phase layer
        out[s:s + CHUNK] = psi
    return out


def fidelity_kernel(A, B=None):
    """K_ij = |<A_i|B_j>|^2 for statevector matrices A (N,D), B (M,D)."""
    B = A if B is None else B
    G = A @ B.conj().T
    return G.real ** 2 + G.imag ** 2


def kernel_diagnostics(K):
    """Concentration diagnostics: if off-diagonal entries pile up near one value
    (variance ~ 0) the kernel carries little information (kernel concentration)."""
    n = K.shape[0]
    off = K[~np.eye(n, dtype=bool)]
    return {"offdiag_mean": float(off.mean()), "offdiag_var": float(off.var()),
            "offdiag_min": float(off.min()), "offdiag_max": float(off.max())}
