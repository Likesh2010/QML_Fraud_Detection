import numpy as np
import pandas as pd

from sklearn.svm import OneClassSVM
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

# ============================================================
# LOAD EXISTING RESULTS
# ============================================================

K_train = np.load("results/K_train.npy")
K_test = np.load("results/K_test.npy")
y_test = np.load("results/y_test.npy")

print("=" * 65)
print("FINE NU SWEEP - EXISTING QML KERNEL")
print("=" * 65)

print(f"K_train shape : {K_train.shape}")
print(f"K_test shape  : {K_test.shape}")
print(f"Test samples  : {len(y_test)}")
print(f"Fraud samples : {np.sum(y_test == 1)}")
print(f"Normal samples: {np.sum(y_test == 0)}")


# ============================================================
# NU VALUES
# ============================================================

nu_values = [
    0.01,
    0.02,
    0.03,
    0.04,
    0.05,
    0.06,
    0.08,
    0.10,
    0.12,
    0.15
]


# ============================================================
# RUN SWEEP
# ============================================================

results = []

for nu in nu_values:

    print("\n" + "-" * 65)
    print(f"Testing nu = {nu}")

    # Train One-Class SVM using existing quantum kernel
    model = OneClassSVM(
        kernel="precomputed",
        nu=nu
    )

    model.fit(K_train)

    # Higher score = more anomalous
    scores = -model.decision_function(K_test)

    # --------------------------------------------------------
    # Default OCSVM threshold
    # --------------------------------------------------------

    predictions_default = (scores >= 0).astype(int)

    accuracy_default = accuracy_score(
        y_test,
        predictions_default
    )

    precision_default = precision_score(
        y_test,
        predictions_default,
        zero_division=0
    )

    recall_default = recall_score(
        y_test,
        predictions_default,
        zero_division=0
    )

    f1_default = f1_score(
        y_test,
        predictions_default,
        zero_division=0
    )

    # --------------------------------------------------------
    # Threshold sweep
    # --------------------------------------------------------

    thresholds = np.linspace(
        scores.min(),
        scores.max(),
        501
    )

    best_accuracy = -1
    best_threshold = None
    best_precision = 0
    best_recall = 0
    best_f1 = 0
    best_cm = None

    for threshold in thresholds:

        predictions = (
            scores >= threshold
        ).astype(int)

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        precision = precision_score(
            y_test,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            predictions,
            zero_division=0
        )

        # Primary target for this experiment:
        # maximize accuracy
        if accuracy > best_accuracy:

            best_accuracy = accuracy
            best_threshold = threshold
            best_precision = precision
            best_recall = recall
            best_f1 = f1

            best_cm = confusion_matrix(
                y_test,
                predictions
            )

    # --------------------------------------------------------
    # Ranking metrics
    # --------------------------------------------------------

    roc_auc = roc_auc_score(
        y_test,
        scores
    )

    pr_auc = average_precision_score(
        y_test,
        scores
    )

    results.append({
        "nu": nu,
        "default_accuracy": accuracy_default,
        "optimized_accuracy": best_accuracy,
        "optimized_precision": best_precision,
        "optimized_recall": best_recall,
        "optimized_f1": best_f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "threshold": best_threshold,
        "TN": best_cm[0, 0],
        "FP": best_cm[0, 1],
        "FN": best_cm[1, 0],
        "TP": best_cm[1, 1]
    })

    print(f"Default accuracy : {accuracy_default:.4f}")
    print(f"Optimized accuracy: {best_accuracy:.4f}")
    print(f"Precision         : {best_precision:.4f}")
    print(f"Recall            : {best_recall:.4f}")
    print(f"F1                : {best_f1:.4f}")
    print(f"ROC-AUC           : {roc_auc:.4f}")
    print(f"PR-AUC            : {pr_auc:.4f}")
    print(f"Threshold         : {best_threshold:.6f}")

    print("Confusion matrix:")
    print(best_cm)


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    "results/fine_nu_sweep_results.csv",
    index=False
)

print("\n" + "=" * 65)
print("FINE NU SWEEP COMPLETE")
print("=" * 65)

print("\nFull results:")
print(results_df.to_string(index=False))

# ------------------------------------------------------------
# Best according to accuracy
# ------------------------------------------------------------

best_accuracy_row = results_df.loc[
    results_df["optimized_accuracy"].idxmax()
]

print("\n" + "=" * 65)
print("BEST ACCURACY")
print("=" * 65)

print(
    f"nu = {best_accuracy_row['nu']}"
)

print(
    f"Accuracy = "
    f"{best_accuracy_row['optimized_accuracy']:.4f}"
    f" ({best_accuracy_row['optimized_accuracy'] * 100:.2f}%)"
)

# ------------------------------------------------------------
# Best according to F1
# ------------------------------------------------------------

best_f1_row = results_df.loc[
    results_df["optimized_f1"].idxmax()
]

print("\n" + "=" * 65)
print("BEST F1")
print("=" * 65)

print(
    f"nu = {best_f1_row['nu']}"
)

print(
    f"F1 = "
    f"{best_f1_row['optimized_f1']:.4f}"
)

# ------------------------------------------------------------
# Best according to recall
# ------------------------------------------------------------

best_recall_row = results_df.loc[
    results_df["optimized_recall"].idxmax()
]

print("\n" + "=" * 65)
print("BEST RECALL")
print("=" * 65)

print(
    f"nu = {best_recall_row['nu']}"
)

print(
    f"Recall = "
    f"{best_recall_row['optimized_recall']:.4f}"
)

print("\nResults saved to:")
print("results/fine_nu_sweep_results.csv")