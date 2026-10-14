"""Model families. Every family exposes:  name, rep, configs(), fit(cfg, Xtr, ctx) -> model,
and model.score(X, tag) -> anomaly score (higher = more anomalous).
rep selects the input representation: 'ang' (angles in [0,pi]), 'pca', or 'full' (29-d scaled)."""
import itertools
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from sklearn.metrics.pairwise import laplacian_kernel
from qkernel import zz_statevectors, fidelity_kernel, kernel_diagnostics

SCORE_CHUNK = 4096


class _PrecomputedKernelModel:
    native_threshold = 0.0          # OCSVM's own decision boundary (score > 0 => anomaly)

    def __init__(self, svm, fam, key, Xtr_or_psi):
        self.svm, self.fam, self.key, self.ref = svm, fam, key, Xtr_or_psi

    def score(self, X, tag):
        Y = self.fam._embed(self.key, tag, X)
        out = np.empty(len(Y))
        for s in range(0, len(Y), SCORE_CHUNK):
            K = self.fam._kernel(Y[s:s + SCORE_CHUNK], self.ref, self.key)
            out[s:s + SCORE_CHUNK] = -self.svm.decision_function(K)
        return out


class QuantumOCSVM:
    name, rep = "Quantum-OCSVM (fidelity kernel)", "ang"

    def __init__(self, reps=(1, 2, 3), ents=("linear", "full"), bws=(0.1, 0.25, 0.5, 1.0),
                 nus=(0.02, 0.05, 0.10, 0.15, 0.20)):
        self.reps, self.ents, self.bws, self.nus = reps, ents, bws, nus
        self._emb_key = None; self._emb = {}

    def configs(self):
        return [dict(reps=r, ent=e, bw=b, nu=v) for r, e, b in
                itertools.product(self.reps, self.ents, self.bws) for v in self.nus]

    def _embed(self, key, tag, X):
        if self._emb_key != key:
            self._emb_key, self._emb = key, {}
        if tag not in self._emb:
            self._emb[tag] = zz_statevectors(X, reps=key[0], entanglement=key[1], bandwidth=key[2])
        return self._emb[tag]

    @staticmethod
    def _kernel(A, B, key):
        return fidelity_kernel(A, B)

    def fit(self, cfg, Xtr, ctx):
        key = (cfg["reps"], cfg["ent"], cfg["bw"])
        if ctx.get("q_key") != key:
            psi = zz_statevectors(Xtr, reps=key[0], entanglement=key[1], bandwidth=key[2])
            ctx["q_key"], ctx["q_psi"], ctx["q_K"] = key, psi, fidelity_kernel(psi)
        svm = OneClassSVM(kernel="precomputed", nu=cfg["nu"]).fit(ctx["q_K"])
        m = _PrecomputedKernelModel(svm, self, key, ctx["q_psi"])
        m.diag = kernel_diagnostics(ctx["q_K"])
        return m


class LaplacianOCSVM:
    name, rep = "Laplacian-OCSVM", "pca"

    def __init__(self, gammas=(0.05, 0.1, 0.25, 0.5, 1.0), nus=(0.02, 0.05, 0.10, 0.15, 0.20)):
        self.gammas, self.nus = gammas, nus

    def configs(self):
        return [dict(gamma=g, nu=v) for g in self.gammas for v in self.nus]

    def _embed(self, key, tag, X):
        return X

    @staticmethod
    def _kernel(A, B, key):          # key = effective gamma (stored per model, not on self)
        return laplacian_kernel(A, B, gamma=key)

    def fit(self, cfg, Xtr, ctx):
        g = cfg["gamma"] / Xtr.shape[1]
        K = laplacian_kernel(Xtr, Xtr, gamma=g)
        svm = OneClassSVM(kernel="precomputed", nu=cfg["nu"]).fit(K)
        return _PrecomputedKernelModel(svm, self, g, Xtr)


class RBFOCSVM:
    name, rep = "RBF-OCSVM (tuned nu & gamma)", "pca"

    def __init__(self, gammas=None, nus=(0.02, 0.05, 0.10, 0.15, 0.20)):
        self.gammas = np.geomspace(0.03, 30, 24) if gammas is None else gammas   # 24 gammas x 5 nus = 120 = quantum grid size (3 reps x 2 ent x 4 bw x 5 nu)
        self.nus = nus

    def configs(self):
        return [dict(gmult=float(g), nu=v) for g in self.gammas for v in self.nus]

    def fit(self, cfg, Xtr, ctx):
        base = 1.0 / (Xtr.shape[1] * Xtr.var())          # sklearn gamma='scale'
        svm = OneClassSVM(kernel="rbf", gamma=cfg["gmult"] * base, nu=cfg["nu"]).fit(Xtr)
        return _Wrap(lambda X, tag: -svm.decision_function(X), native=0.0)


class _Wrap:
    def __init__(self, fn, native=None):
        self.fn, self.native_threshold = fn, native

    def score(self, X, tag):
        return self.fn(X, tag)


class IForest:
    def __init__(self, rep, label, trees=(100, 300), max_samples=(64, 256, 1.0)):
        self.rep, self.name = rep, label
        self.trees, self.ms = trees, max_samples

    def configs(self):
        return [dict(trees=t, ms=m) for t in self.trees for m in self.ms]

    def fit(self, cfg, Xtr, ctx):
        ms = cfg["ms"] if cfg["ms"] == 1.0 else min(int(cfg["ms"]), len(Xtr))
        m = IsolationForest(n_estimators=cfg["trees"], max_samples=ms, random_state=ctx["seed"]).fit(Xtr)
        return _Wrap(lambda X, tag: -m.score_samples(X))


class LOF:
    name, rep = "LocalOutlierFactor (novelty)", "pca"

    def __init__(self, ks=(5, 10, 20, 50)):
        self.ks = ks

    def configs(self):
        return [dict(k=k) for k in self.ks]

    def fit(self, cfg, Xtr, ctx):
        m = LocalOutlierFactor(n_neighbors=min(cfg["k"], len(Xtr) - 1), novelty=True).fit(Xtr)
        return _Wrap(lambda X, tag: -m.score_samples(X))


class PCARecon:
    name, rep = "PCA reconstruction error (29-d)", "full"

    def __init__(self, comps=(1, 2, 4, 8, 12)):
        self.comps = comps

    def configs(self):
        return [dict(c=c) for c in self.comps]

    def fit(self, cfg, Xtr, ctx):
        p = PCA(cfg["c"], random_state=0).fit(Xtr)
        return _Wrap(lambda X, tag: ((X - p.inverse_transform(p.transform(X))) ** 2).sum(axis=1))


class Autoencoder:
    name, rep = "Autoencoder (29-d, MLP)", "full"

    def __init__(self, bottlenecks=(4, 8), alphas=(1e-4, 1e-2)):
        self.b, self.a = bottlenecks, alphas

    def configs(self):
        return [dict(b=b, alpha=a) for b in self.b for a in self.a]

    def fit(self, cfg, Xtr, ctx):
        m = MLPRegressor(hidden_layer_sizes=(32, cfg["b"], 32), alpha=cfg["alpha"], max_iter=300,
                         random_state=ctx["seed"]).fit(Xtr, Xtr)
        return _Wrap(lambda X, tag: ((X - m.predict(X)) ** 2).mean(axis=1))


def build_families(quick=False):
    if quick:
        return [QuantumOCSVM(reps=(1, 2), ents=("linear",), bws=(0.25,), nus=(0.05, 0.15)),
                RBFOCSVM(gammas=np.geomspace(0.1, 10, 4), nus=(0.05, 0.15)),
                LaplacianOCSVM(gammas=(0.25, 1.0), nus=(0.05, 0.15)),
                IForest("pca", "IsolationForest (same 4-d PCA)", trees=(100,), max_samples=(256,)),
                IForest("full", "IsolationForest (29-d)", trees=(100,), max_samples=(256,)),
                LOF(ks=(10,)), PCARecon(comps=(2, 8)), Autoencoder(bottlenecks=(4,), alphas=(1e-3,))]
    return [QuantumOCSVM(), RBFOCSVM(), LaplacianOCSVM(),
            IForest("pca", "IsolationForest (same 4-d PCA)"), IForest("full", "IsolationForest (29-d)"),
            LOF(), PCARecon(), Autoencoder()]
