"""Validate the fast kernel against an independent gate-by-gate simulator.
Run:  python tests/test_qkernel.py      (also works under pytest)"""
import sys, os, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from qkernel import zz_statevectors, fidelity_kernel, entangler_pairs

H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)


def _axis(n, q):            # qubit q (bit q of basis index) <-> tensor axis n-1-q
    return n - 1 - q


def apply_1q(psi, U, q, n):
    t = psi.reshape((2,) * n)
    t = np.moveaxis(np.tensordot(U, t, axes=([1], [_axis(n, q)])), 0, _axis(n, q))
    return t.reshape(-1)


def apply_cx(psi, c, t_, n):
    idx = np.arange(2 ** n)
    ctrl = (idx >> c) & 1
    perm = np.where(ctrl == 1, idx ^ (1 << t_), idx)
    out = np.empty_like(psi)
    out[perm] = psi
    return out


def gate_level_state(x, reps, ent, n):
    """Literal circuit: per rep -> H on all, P(2x_i) on all, then for each pair
    CX(i,j), P(2(pi-x_i)(pi-x_j)) on j, CX(i,j)  (Qiskit ZZFeatureMap definition)."""
    psi = np.zeros(2 ** n, dtype=complex); psi[0] = 1
    for _ in range(reps):
        for q in range(n):
            psi = apply_1q(psi, H, q, n)
        for q in range(n):
            psi = apply_1q(psi, np.diag([1, np.exp(1j * 2 * x[q])]), q, n)
        for (i, j) in entangler_pairs(n, ent):
            psi = apply_cx(psi, i, j, n)
            psi = apply_1q(psi, np.diag([1, np.exp(1j * 2 * (np.pi - x[i]) * (np.pi - x[j]))]), j, n)
            psi = apply_cx(psi, i, j, n)
    return psi


def test_matches_gate_level():
    rng = np.random.default_rng(0)
    for n in (2, 3, 4):
        for reps in (1, 2, 3):
            for ent in ("linear", "full", "circular"):
                X = rng.uniform(0, np.pi, size=(5, n))
                fast = zz_statevectors(X, reps, ent)
                for a in range(5):
                    ref = gate_level_state(X[a], reps, ent, n)
                    # equal up to global phase -> compare fidelity with itself and cross-fidelities
                    assert abs(abs(np.vdot(ref, fast[a])) - 1) < 1e-10, (n, reps, ent)
                Kf = fidelity_kernel(fast)
                R = np.array([gate_level_state(x, reps, ent, n) for x in X])
                Kr = fidelity_kernel(R)
                assert np.allclose(Kf, Kr, atol=1e-10), (n, reps, ent)


def test_kernel_properties():
    rng = np.random.default_rng(1)
    X = rng.uniform(0, np.pi, size=(40, 4))
    K = fidelity_kernel(zz_statevectors(X, 2, "linear"))
    assert np.allclose(np.diag(K), 1) and np.allclose(K, K.T)
    assert K.min() >= -1e-12 and K.max() <= 1 + 1e-12
    assert np.linalg.eigvalsh(K).min() > -1e-9          # PSD
    K0 = fidelity_kernel(zz_statevectors(X, 2, "linear", bandwidth=1e-6))
    assert np.allclose(K0, 1, atol=1e-6)                 # bw->0 : constant kernel


def test_qiskit_crosscheck_optional():
    try:
        from qiskit.circuit.library import ZZFeatureMap
        from qiskit.quantum_info import Statevector
    except Exception:
        print("  (qiskit not installed - optional cross-check skipped)")
        return
    rng = np.random.default_rng(2)
    X = rng.uniform(0, np.pi, size=(6, 4))
    fm = ZZFeatureMap(4, reps=2, entanglement="linear")
    ref = np.array([Statevector(fm.assign_parameters(x)).data for x in X])
    assert np.allclose(fidelity_kernel(zz_statevectors(X, 2, "linear")), fidelity_kernel(ref), atol=1e-10)


def test_speed():
    X = np.random.default_rng(3).uniform(0, np.pi, size=(2000, 4))
    t = time.time()
    psi = zz_statevectors(X, 2, "linear")
    K = fidelity_kernel(psi)
    dt = time.time() - t
    print(f"  2000x2000 kernel (4 qubits, reps=2): {dt:.2f}s")
    assert dt < 30


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
