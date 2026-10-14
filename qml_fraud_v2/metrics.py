"""Fraud-oriented metrics: fixed-FPR operating points, alerts per 10k, expected cost."""
import numpy as np


def threshold_at_fpr(neg_scores, fpr):
    return float(np.quantile(neg_scores, 1.0 - fpr))


def point_metrics(scores, y, thr):
    pred = scores > thr
    pos, neg = y == 1, y == 0
    tp = int((pred & pos).sum()); fp = int((pred & neg).sum())
    fn = int((~pred & pos).sum()); tn = int((~pred & neg).sum())
    P, N = tp + fn, fp + tn
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / P if P else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "recall": rec,
            "fpr": fp / N if N else float("nan"), "precision": prec, "f1": f1,
            "alerts_per_10k": (tp + fp) / (P + N) * 1e4}


def cost_optimal_threshold(val_scores, val_y, prevalence, cost_ratio, n_grid=2000):
    """Threshold minimising expected cost per transaction at the NATURAL prevalence:
    cost = cost_ratio * FNR * pi + FPR * (1 - pi)   (a false alarm costs 1 unit,
    a missed fraud costs `cost_ratio` units). TPR/FPR come from validation."""
    neg = np.sort(val_scores[val_y == 0]); pos = np.sort(val_scores[val_y == 1])
    cand = np.unique(np.quantile(val_scores, np.linspace(0.5, 1.0, n_grid)))
    fpr = 1 - np.searchsorted(neg, cand, side="right") / len(neg)
    tpr = 1 - np.searchsorted(pos, cand, side="right") / len(pos)
    cost = cost_ratio * (1 - tpr) * prevalence + fpr * (1 - prevalence)
    i = int(cost.argmin())
    return float(cand[i]), float(cost[i] * 1e4)
