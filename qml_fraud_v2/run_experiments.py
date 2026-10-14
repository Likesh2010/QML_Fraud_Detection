"""
Main experiment driver.

Protocol (all selection on validation; test scored once per model per cell):
  1. Chronological split  train (first 60%) | validation (next 20%) | test (last 20%).
  2. Scaler + PCA fitted on train-period normals only.
  3. For each seed: sample `n_train` normal training transactions.
  4. For every model family: fit every config in its grid, pick the config with the best
     VALIDATION average precision (threshold-free, so nu is not confused with the threshold).
  5. Operating points are set on validation normals (FPR = 0.1%, 1%, 5% and a cost-optimal
     point) and applied unchanged to the untouched test period.
  6. Test period is evaluated at its natural fraud prevalence (no enrichment).
Results: <out>/metrics.jsonl (one record per cell x model), <out>/scores_*.npz (test scores).
"""
import argparse, json, time
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.ensemble import HistGradientBoostingClassifier
from data import load_and_split, Preprocessor
from models import build_families
from metrics import threshold_at_fpr, point_metrics, cost_optimal_threshold

FPRS = (0.001, 0.01, 0.05)
SUP_NAME = "Supervised HGB (upper bound, uses fraud labels)"


def evaluate(sv, st, yv, yt, prevalence, cost_ratio, native):
    rec = {"val_ap": float(average_precision_score(yv, sv)),
           "test_auc": float(roc_auc_score(yt, st)), "test_ap": float(average_precision_score(yt, st))}
    thrs = {}
    for f in FPRS:
        thr = threshold_at_fpr(sv[yv == 0], f)
        thrs[f"fpr{f}"] = thr
        rec[f"at_fpr_{f}"] = point_metrics(st, yt, thr)
    thr, cost = cost_optimal_threshold(sv, yv, prevalence, cost_ratio)
    thrs["cost"] = thr
    rec["cost_opt"] = {**point_metrics(st, yt, thr), "val_cost_per_10k": cost}
    if native is not None:                      # the SVM's own nu-defined boundary
        thrs["native"] = native
        rec["native"] = point_metrics(st, yt, native)
    rec["thresholds"] = thrs
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="creditcard.csv")
    ap.add_argument("--synthetic", action="store_true", help="smoke-test on synthetic data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--n-train", type=int, nargs="+", default=[1000, 2000])
    ap.add_argument("--pca-dims", type=int, nargs="+", default=[4])
    ap.add_argument("--cost-ratio", type=float, default=50.0, help="cost of a missed fraud / cost of a false alarm")
    ap.add_argument("--drop-overlap", action="store_true", help="drop val/test rows whose features duplicate a train row")
    ap.add_argument("--quick", action="store_true", help="tiny grids for smoke tests")
    a = ap.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    D = load_and_split(a.data, synthetic=a.synthetic, drop_overlap=a.drop_overlap)
    print("DATA:", json.dumps(D.info))
    (out / "data_info.json").write_text(json.dumps(D.info, indent=1))
    np.save(out / "y_test.npy", D.test.y)
    yv, yt = D.val.y, D.test.y
    prevalence = D.info["fraud_rate"]

    kmax = max(a.pca_dims)
    prep = Preprocessor(kmax).fit(D.train.X[D.train.y == 0])
    (out / "pca_explained.json").write_text(json.dumps(prep.explained.tolist()))
    print("PCA explained variance ratio:", np.round(prep.explained, 3), "cum:", round(prep.explained.sum(), 3))

    mpath = out / "metrics.jsonl"
    done = set()
    if mpath.exists():
        for line in mpath.read_text().splitlines():
            r = json.loads(line); done.add((r["k"], r["n_train"], r["seed"]))

    # supervised upper bound: trained once on the whole training period WITH labels
    full_tr = prep.transform(D.train.X, 2)["full"]
    sup = HistGradientBoostingClassifier(max_iter=200, early_stopping=False, random_state=0).fit(full_tr, D.train.y)
    sup_v = sup.predict_proba(prep.transform(D.val.X, 2)["full"])[:, 1]
    sup_t = sup.predict_proba(prep.transform(D.test.X, 2)["full"])[:, 1]

    fams = build_families(a.quick)
    print("grid sizes:", {f.name: len(f.configs()) for f in fams})
    norm_idx = np.where(D.train.y == 0)[0]
    for k in a.pca_dims:
        Rtr_all = prep.transform(D.train.X[norm_idx], k)
        Rv, Rt = prep.transform(D.val.X, k), prep.transform(D.test.X, k)
        print(f"[k={k}] clip rate val={Rv['clip_rate']:.3f} test={Rt['clip_rate']:.3f}")
        for n in a.n_train:
            for seed in range(a.seed_start if hasattr(a, "seed_start") else 0, a.seeds):
                if (k, n, seed) in done:
                    continue
                rng = np.random.default_rng(1000 + seed)
                idx = rng.choice(len(norm_idx), size=n, replace=False)
                Rtr = {r: Rtr_all[r][idx] for r in ("ang", "pca", "full")}
                ctx = {"seed": seed}
                scores = {}
                records = []
                for fam in fams:
                    t0 = time.time(); best = None
                    for cfg in fam.configs():
                        m = fam.fit(cfg, Rtr[fam.rep], ctx)
                        sv = m.score(Rv[fam.rep], "val")
                        apv = average_precision_score(yv, sv)
                        if best is None or apv > best[0]:
                            best = (apv, cfg, m, sv)
                    apv, cfg, m, sv = best
                    st = m.score(Rt[fam.rep], "test")
                    rec = evaluate(sv, st, yv, yt, prevalence, a.cost_ratio, getattr(m, "native_threshold", None))
                    rec.update(k=k, n_train=n, seed=seed, family=fam.name, cfg=cfg, n_configs=len(fam.configs()),
                               seconds=round(time.time() - t0, 1))
                    if hasattr(m, "diag"):
                        rec["kernel_diag"] = m.diag
                    records.append(rec); scores[fam.name] = st.astype(np.float32)
                    print(f"  k={k} n={n} seed={seed} {fam.name[:34]:34s} valAP={apv:.3f} testAUC={rec['test_auc']:.3f} "
                          f"testAP={rec['test_ap']:.3f} rec@1%FPR={rec['at_fpr_0.01']['recall']:.2f} ({rec['seconds']}s)")
                rec = evaluate(sup_v, sup_t, yv, yt, prevalence, a.cost_ratio, None)
                rec.update(k=k, n_train=n, seed=seed, family=SUP_NAME, cfg={}, n_configs=1, seconds=0)
                records.append(rec); scores[SUP_NAME] = sup_t.astype(np.float32)
                np.savez_compressed(out / f"scores_k{k}_n{n}_s{seed}.npz", **{f"f{i}": v for i, v in enumerate(scores.values())},
                                    names=np.array(list(scores.keys())))
                with open(mpath, "a") as fh:
                    for r in records:
                        fh.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
