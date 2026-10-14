"""Data loading, 3-way chronological split, leakage-safe preprocessing."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount"]


def make_synthetic(n=30000, fraud_rate=0.01, seed=0):
    """Small stand-in with the ULB schema, only for smoke-testing the pipeline."""
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0, 172800, n))
    A = rng.normal(size=(28, 28)) * 0.3 + np.eye(28)
    X = rng.normal(size=(n, 28)) @ A
    y = (rng.random(n) < fraud_rate).astype(int)
    shift = np.zeros(28); shift[:6] = [2.5, -2, 1.5, -2, 1.5, 1.0]
    X[y == 1] = X[y == 1] * 1.3 + shift
    df = pd.DataFrame(X, columns=FEATURES[:28])
    df["Amount"] = rng.lognormal(3, 1.2, n)
    df.insert(0, "Time", t)
    df["Class"] = y
    return df


@dataclass
class Part:
    X: np.ndarray      # (n, 29) with log1p(Amount), unscaled
    y: np.ndarray
    t: np.ndarray


@dataclass
class Dataset:
    train: Part
    val: Part
    test: Part
    info: dict


def load_and_split(path=None, synthetic=False, fracs=(0.6, 0.2, 0.2), drop_overlap=False):
    """Sort by Time; first 60% = train, next 20% = validation, last 20% = test.
    Hyper-parameters are chosen on validation; test is touched once per model."""
    df = make_synthetic() if synthetic else pd.read_csv(path)
    df = df.sort_values("Time", kind="stable").reset_index(drop=True)
    df["Amount"] = np.log1p(df["Amount"])
    n = len(df)
    a, b = int(n * fracs[0]), int(n * (fracs[0] + fracs[1]))
    h = pd.util.hash_pandas_object(df[FEATURES], index=False).to_numpy()
    train_hashes = set(h[:a])
    overlap = np.array([x in train_hashes for x in h])
    overlap[:a] = False
    info = {"n_rows": n, "n_fraud": int(df["Class"].sum()), "fraud_rate": float(df["Class"].mean())}
    parts = {}
    for name, sl in (("train", slice(0, a)), ("val", slice(a, b)), ("test", slice(b, n))):
        keep = np.ones(len(df), dtype=bool)
        keep[:] = False; keep[sl] = True
        if drop_overlap and name != "train":
            keep &= ~overlap
        sub = df[keep]
        parts[name] = Part(sub[FEATURES].to_numpy(float), sub["Class"].to_numpy(int), sub["Time"].to_numpy(float))
        info[f"{name}_rows"] = len(sub)
        info[f"{name}_fraud"] = int(sub["Class"].sum())
        info[f"{name}_feature_dupes_of_train"] = int(overlap[keep].sum()) if not drop_overlap else 0
    info["drop_overlap"] = drop_overlap
    return Dataset(parts["train"], parts["val"], parts["test"], info)


class Preprocessor:
    """StandardScaler -> PCA(kmax) fitted ONLY on training-period normal transactions.
    PCA components are nested, so PCA(k) = first k columns of PCA(kmax).
    'ang' = first k components min-max scaled to [0, pi] using robust percentiles
    from train normals, then clipped (test values outside the range are clipped)."""

    def __init__(self, kmax, lo_pct=0.5, hi_pct=99.5):
        self.kmax, self.lo_pct, self.hi_pct = kmax, lo_pct, hi_pct

    def fit(self, Xnorm):
        self.sc = StandardScaler().fit(Xnorm)
        Z = self.sc.transform(Xnorm)
        self.pca = PCA(self.kmax, random_state=0).fit(Z)
        P = self.pca.transform(Z)
        self.lo = np.percentile(P, self.lo_pct, axis=0)
        self.hi = np.percentile(P, self.hi_pct, axis=0)
        self.explained = self.pca.explained_variance_ratio_
        return self

    def transform(self, X, k):
        Z = self.sc.transform(X)
        P = self.pca.transform(Z)[:, :k]
        raw = (P - self.lo[:k]) / (self.hi[:k] - self.lo[:k])
        ang = np.clip(raw, 0, 1) * np.pi
        clip_rate = float(((raw < 0) | (raw > 1)).any(axis=1).mean())
        return {"full": Z, "pca": P, "ang": ang, "clip_rate": clip_rate}
