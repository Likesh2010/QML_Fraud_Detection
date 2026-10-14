"""Aggregate results -> tables, paired statistics, figures, and an auto-filled draft of the
abstract / conclusion whose claims follow from the statistics (not from the hoped-for result).
Usage: python report.py --results results [--k 4] [--n 2000]"""
import argparse, json, collections
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, precision_recall_curve
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

Q = "Quantum-OCSVM (fidelity kernel)"
RBF = "RBF-OCSVM (tuned nu & gamma)"
OCSVMS = {Q, RBF, "Laplacian-OCSVM"}
SUP = "Supervised HGB (upper bound, uses fraud labels)"


def load(res):
    rows, raw = [], []
    for line in (res / "metrics.jsonl").read_text().splitlines():
        r = json.loads(line); raw.append(r)
        rows.append(dict(family=r["family"], k=r["k"], n=r["n_train"], seed=r["seed"], auc=r["test_auc"], ap=r["test_ap"],
                         val_ap=r["val_ap"], rec01=r["at_fpr_0.001"]["recall"], rec1=r["at_fpr_0.01"]["recall"],
                         rec5=r["at_fpr_0.05"]["recall"], prec1=r["at_fpr_0.01"]["precision"],
                         alerts1=r["at_fpr_0.01"]["alerts_per_10k"], fpr1=r["at_fpr_0.01"]["fpr"],
                         cost_rec=r["cost_opt"]["recall"], cost_alerts=r["cost_opt"]["alerts_per_10k"],
                         nat_f1=r.get("native", {}).get("f1", np.nan), nat_rec=r.get("native", {}).get("recall", np.nan),
                         nat_fpr=r.get("native", {}).get("fpr", np.nan), n_configs=r["n_configs"]))
    return pd.DataFrame(rows), raw


def ms(x, pct=False, d=3):
    x = np.asarray(x, float)
    if np.all(np.isnan(x)): return "-"
    m, s = np.nanmean(x), np.nanstd(x, ddof=1) if len(x) > 1 else 0.0
    return f"{100*m:.1f} ± {100*s:.1f}" if pct else f"{m:.{d}f} ± {s:.{d}f}"


def md(df):
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    out += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(out)


def main_table(df):
    rows = []
    for fam, g in df.groupby("family", sort=False):
        rows.append({"Model": fam, "ROC-AUC": ms(g.auc), "PR-AUC (AP)": ms(g.ap), "Recall@0.1%FPR (%)": ms(g.rec01, True),
                     "Recall@1%FPR (%)": ms(g.rec1, True), "Precision@1%FPR (%)": ms(g.prec1, True),
                     "Alerts / 10k @1%FPR (TP+FP)": f"{g.alerts1.mean():.0f}",
                     "Native-nu F1 (%)": ms(g.nat_f1, True)})
    return pd.DataFrame(rows)


def paired_ci(a, b):
    d = np.asarray(a) - np.asarray(b)
    if len(d) < 2: return d.mean(), np.nan, np.nan
    h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return d.mean(), d.mean() - h, d.mean() + h


def load_scores(res, k, n, seed):
    z = np.load(res / f"scores_k{k}_n{n}_s{seed}.npz", allow_pickle=True)
    return {nm: z[f"f{i}"] for i, nm in enumerate(z["names"])}


def bootstrap_diff(sa, sb, ta, tb, y, B=200, seed=0):
    """Paired, stratified bootstrap over TEST rows for (AP diff, recall@fixed-threshold diff)."""
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    ap_d, rc_d = [], []
    for _ in range(B):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        ap_d.append(average_precision_score(y[idx], sa[idx]) - average_precision_score(y[idx], sb[idx]))
        p = idx[y[idx] == 1]
        rc_d.append((sa[p] > ta).mean() - (sb[p] > tb).mean())
    q = lambda v: (np.percentile(v, 2.5), np.percentile(v, 97.5))
    return q(ap_d), q(rc_d)


def mcnemar_frauds(sa, sb, ta, tb, y):
    p = y == 1
    ca, cb = sa[p] > ta, sb[p] > tb
    b, c = int((ca & ~cb).sum()), int((~ca & cb).sum())
    return b, c, (stats.binomtest(b, b + c, 0.5).pvalue if b + c else 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results"); ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--n", type=int, default=None); ap.add_argument("--boot", type=int, default=200)
    a = ap.parse_args()
    res = Path(a.results); df, raw = load(res)
    info = json.loads((res / "data_info.json").read_text())
    k = a.k; n = a.n or int(df[df.k == k].n.max())
    P = df[(df.k == k) & (df.n == n)]
    seeds = sorted(P.seed.unique()); S = len(seeds)
    y = np.load(res / "y_test.npy")
    L = [f"# Results (k={k} PCA dims/qubits, n_train={n}, {S} seeds)\n",
         f"Test period: {info['test_rows']} transactions, {info['test_fraud']} frauds "
         f"(natural prevalence {100*info['test_fraud']/info['test_rows']:.3f}%). "
         f"Validation: {info['val_rows']} rows, {info['val_fraud']} frauds. Mean ± std across seeds (training-sample resampling).\n",
         "## Table 1 - main comparison (hyper-parameters chosen on validation; thresholds set on validation)\n", md(main_table(P)), ""]

    # paired comparisons
    occ = [f for f in P.family.unique() if f not in (Q, SUP)]
    best_other = max([f for f in occ if f != RBF], key=lambda f: P[P.family == f].ap.mean()) if len(occ) > 1 else None
    comps = [RBF] + ([best_other] if best_other else [])
    L += ["## Table 2 - paired comparison, Quantum minus comparator\n"]
    rows, verdict = [], {}
    per_seed = {s: load_scores(res, k, n, s) for s in seeds}
    recs = {(r["family"], r["seed"]): r for r in raw if r["k"] == k and r["n_train"] == n}
    for comp in comps:
        gq, gc = P[P.family == Q].sort_values("seed"), P[P.family == comp].sort_values("seed")
        for met, lab in (("ap", "PR-AUC"), ("auc", "ROC-AUC"), ("rec1", "Recall@1%FPR"), ("rec01", "Recall@0.1%FPR")):
            m, lo, hi = paired_ci(gq[met], gc[met])
            rows.append({"Comparator": comp, "Metric": lab, "Mean diff": f"{m:+.3f}", "95% CI over seeds": f"[{lo:+.3f}, {hi:+.3f}]",
                         "Seeds Q>C": f"{int((gq[met].values > gc[met].values).sum())}/{S}"})
            verdict[(comp, met)] = (m, lo, hi)
        bo, mc = [], []
        for s in seeds:
            ta, tb = recs[(Q, s)]["thresholds"]["fpr0.01"], recs[(comp, s)]["thresholds"]["fpr0.01"]
            (alo, ahi), (rlo, rhi) = bootstrap_diff(per_seed[s][Q], per_seed[s][comp], ta, tb, y, a.boot, seed=s)
            bo.append((alo > 0, ahi < 0, rlo > 0, rhi < 0))
            mc.append(mcnemar_frauds(per_seed[s][Q], per_seed[s][comp], ta, tb, y)[2] < 0.05)
        bo = np.array(bo)
        rows.append({"Comparator": comp, "Metric": "test-set bootstrap (per seed, 95%)",
                     "Mean diff": "-", "95% CI over seeds": "-",
                     "Seeds Q>C": f"AP: Q sig. better {bo[:,0].sum()}/{S}, worse {bo[:,1].sum()}/{S}; Recall@1%: better {bo[:,2].sum()}/{S}, worse {bo[:,3].sum()}/{S}; McNemar p<.05 in {sum(mc)}/{S}"})
    L += [md(pd.DataFrame(rows)), "",
          "CIs over seeds reflect training-sample variability only (all seeds share one test set); the bootstrap rows capture test-set sampling noise.\n"]

    # scaling, pca-dim, config stability
    A = df[df.k == k]
    L += ["## Table 3 - training-size scaling, PR-AUC (mean ± std)\n",
          md(A.pivot_table(index="family", columns="n", values="ap", aggfunc=lambda x: ms(x)).reset_index()), ""]
    B_ = df[df.n == n]
    if B_.k.nunique() > 1:
        L += ["## Table 4 - PCA-dimension / qubit-count ablation, PR-AUC\n",
              md(B_[B_.family.isin([Q, RBF])].pivot_table(index="family", columns="k", values="ap", aggfunc=lambda x: ms(x)).reset_index()), ""]
    cfgs = collections.Counter(json.dumps(r["cfg"], sort_keys=True) for (f, s), r in recs.items() if f == Q)
    L += ["## Table 5 - quantum configurations selected on validation (stability across seeds)\n"] + [f"- `{c}`: {v}x" for c, v in cfgs.most_common()]
    diag = [r["kernel_diag"] for (f, s), r in recs.items() if f == Q]
    L += ["", f"Selected-kernel concentration (mean over seeds): off-diagonal mean {np.mean([d['offdiag_mean'] for d in diag]):.3f}, "
              f"off-diagonal variance {np.mean([d['offdiag_var'] for d in diag]):.4f} (variance near 0 = concentrated, uninformative kernel).\n"]
    nat = P[P.family.isin(OCSVMS)].groupby("family").agg(nat_rec=("nat_rec", "mean"), nat_fpr=("nat_fpr", "mean"), nat_f1=("nat_f1", "mean")).round(3).reset_index()
    L += ["## Table 6 - OCSVM native nu-boundary (what the original report evaluated), natural prevalence\n", md(nat), ""]

    # figures
    fams = list(P.family.unique())
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    for a_, met, ttl in zip(ax, ("ap", "auc", "rec1"), ("PR-AUC", "ROC-AUC", "Recall @ 1% FPR")):
        m_ = [P[P.family == f][met].mean() for f in fams]; s_ = [P[P.family == f][met].std(ddof=1) for f in fams]
        a_.barh(range(len(fams)), m_, xerr=s_, color=["tab:blue" if f == Q else "tab:gray" for f in fams]); a_.set_yticks(range(len(fams)))
        a_.set_yticklabels([f[:28] for f in fams] if met == "ap" else [""] * len(fams), fontsize=8); a_.set_title(ttl); a_.invert_yaxis()
    plt.tight_layout(); plt.savefig(res / "fig_main_comparison.png", dpi=150); plt.close()
    fig = plt.figure(figsize=(7, 4.5))
    for f in fams:
        g = A[A.family == f].groupby("n").ap.agg(["mean", "std"]).reset_index()
        plt.errorbar(g.n, g["mean"], yerr=g["std"], marker="o", capsize=3, label=f[:30], lw=2.5 if f == Q else 1)
    plt.xscale("log"); plt.xlabel("normal training samples"); plt.ylabel("test PR-AUC"); plt.legend(fontsize=7); plt.tight_layout()
    plt.savefig(res / "fig_scaling.png", dpi=150); plt.close()
    s0 = seeds[0]; plt.figure(figsize=(6, 4.5))
    for f in (Q, RBF, best_other):
        if f:
            pr, rc, _ = precision_recall_curve(y, per_seed[s0][f]); plt.plot(rc, pr, label=f[:30])
    plt.xlabel("recall"); plt.ylabel("precision (natural prevalence)"); plt.yscale("log"); plt.legend(fontsize=7); plt.tight_layout()
    plt.savefig(res / "fig_pr_curves.png", dpi=150); plt.close()

    # auto-filled draft
    g = lambda f, m_: P[P.family == f][m_].mean()
    def word(comp, met):
        m_, lo, hi = verdict[(comp, met)]
        return "higher" if lo > 0 else "lower" if hi < 0 else "statistically indistinguishable"
    names = {"ap": "PR-AUC", "auc": "ROC-AUC", "rec1": "recall at 1% FPR"}
    claims = {met: word(RBF, met) for met in names}
    hi_ = [names[m_] for m_, w in claims.items() if w == "higher"]
    lo_ = [names[m_] for m_, w in claims.items() if w == "lower"]
    eq_ = [names[m_] for m_, w in claims.items() if w == "statistically indistinguishable"]
    parts = ([f"higher {', '.join(hi_)}" ] if hi_ else []) + ([f"lower {', '.join(lo_)}"] if lo_ else []) + \
            ([f"no significant difference in {', '.join(eq_)}"] if eq_ else [])
    headline = ("Compared with a tuned RBF One-Class SVM under the same tuning budget, the quantum-kernel detector showed "
                + "; ".join(parts) + " (95% CIs over seeds, Table 2)"
                + ("; no quantum advantage was demonstrated." if not hi_ else "."))
    nq, nr = int(P[P.family == Q].n_configs.iloc[0]), int(P[P.family == RBF].n_configs.iloc[0])
    grid_txt = f"{nq} quantum vs {nr} RBF configurations" + (" (equal budgets)" if nq == nr else " (UNEQUAL - rerun without --quick)")
    draft = f"""# Draft abstract and conclusion (auto-filled from `{res}`; edit wording, but do not change what the numbers support)

## Abstract
Credit-card fraud detection is an extremely imbalanced anomaly-detection problem ({info['n_fraud']} frauds in {info['n_rows']:,} ULB transactions, {100*info['fraud_rate']:.3f}%). We study whether a fidelity quantum kernel improves a One-Class SVM trained only on normal transactions. Transactions are split chronologically into train/validation/test periods; scaling and PCA ({k} components, encoded on {k} qubits) are fitted on training-period normals only. Hyper-parameters of every model (ZZ feature-map repetitions, entanglement, bandwidth and nu for the quantum model; gamma and nu for RBF) are chosen on the validation period with comparable search budgets ({grid_txt}), and operating thresholds are fixed on validation, so nu is not conflated with the decision threshold. The test period ({info['test_fraud']} frauds, natural prevalence) is evaluated once per model, over {S} training-sample seeds. Quantum kernels are computed exactly from {2**k}-dimensional statevectors, validated against a gate-level simulator. At n={n} training transactions the quantum model reached PR-AUC {ms(P[P.family==Q].ap)} and recall@1%FPR {ms(P[P.family==Q].rec1, True)}% versus {ms(P[P.family==RBF].ap)} and {ms(P[P.family==RBF].rec1, True)}% for the tuned RBF baseline. {headline} At a 1% false-positive budget the quantum model raises about {g(Q,'alerts1'):.0f} alerts per 10,000 transactions, of which precision is {100*g(Q,'prec1'):.1f}%, illustrating the false-alarm burden at realistic prevalence.

## Conclusion
A quantum-kernel One-Class SVM was implemented end-to-end and evaluated under a leakage-safe chronological protocol with matched tuning, multiple seeds and natural-prevalence testing. {headline} Against the strongest additional classical baseline ({best_other}) it achieved PR-AUC {ms(P[P.family==Q].ap)} versus {ms(P[P.family==best_other].ap) if best_other else '-'}. The supervised reference (which uses fraud labels) reached PR-AUC {ms(P[P.family==SUP].ap)}, showing the headroom that one-class methods leave. Results are limited to {k}-qubit noiseless simulation (which is classically simulable), one dataset, and {info['test_fraud']} test frauds, so absolute differences carry the uncertainty reported in Tables 1-2. Future work: hardware/noise studies, larger qubit counts, other datasets, and cost-calibrated thresholds.
"""
    (res / "RESULTS.md").write_text("\n".join(L)); (res / "abstract_conclusion_draft.md").write_text(draft)
    print("\n".join(L)); print("\n" + "=" * 80 + "\n" + draft)


if __name__ == "__main__":
    main()
