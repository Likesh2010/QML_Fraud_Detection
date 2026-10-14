import numpy as np

from sklearn.svm import OneClassSVM
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
    average_precision_score
)


# ============================================================
# LOAD SAVED QUANTUM KERNELS
# ============================================================

print("=" * 70)
print("QML ONE-CLASS SVM - NU TUNING")
print("=" * 70)

print("\nLoading saved quantum kernels...")

K_train = np.load("results/K_train.npy")
K_test = np.load("results/K_test.npy")
y_test = np.load("results/y_test.npy")

print("K_train shape:", K_train.shape)
print("K_test shape :", K_test.shape)
print("Test labels  :", y_test.shape)

print("\nQuantum kernels loaded successfully.")
print("No quantum kernel calculation is required.")


# ============================================================
# NU VALUES TO TEST
# ============================================================

nu_values = [
    0.01,
    0.03,
    0.05,
    0.10,
    0.15,
    0.20
]


# ============================================================
# STORE RESULTS
# ============================================================

results = []


# ============================================================
# TEST EACH NU VALUE
# ============================================================

print("\n" + "=" * 70)
print("TESTING DIFFERENT NU VALUES")
print("=" * 70)

for nu in nu_values:

    print("\n" + "-" * 70)
    print(f"Testing nu = {nu}")
    print("-" * 70)

    # --------------------------------------------------------
    # Create One-Class SVM
    # --------------------------------------------------------

    ocsvm = OneClassSVM(
        kernel="precomputed",
        nu=nu
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    ocsvm.fit(K_train)

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    predictions = ocsvm.predict(K_test)

    # One-Class SVM:
    # +1 = normal
    # -1 = anomaly
    #
    # Dataset:
    # 0 = normal
    # 1 = fraud

    y_pred = np.where(
        predictions == -1,
        1,
        0
    )

    # --------------------------------------------------------
    # Decision scores
    # --------------------------------------------------------

    decision_scores = ocsvm.decision_function(K_test)

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        -decision_scores
    )

    pr_auc = average_precision_score(
        y_test,
        -decision_scores
    )

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    # --------------------------------------------------------
    # Store
    # --------------------------------------------------------

    results.append({
        "nu": nu,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc
    })

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")
    print(f"PR-AUC   : {pr_auc:.4f}")

    print("\nConfusion Matrix:")
    print(cm)


# ============================================================
# CONVERT RESULTS TO NUMPY/PANDAS TABLE
# ============================================================

import pandas as pd

results_df = pd.DataFrame(results)


# ============================================================
# DISPLAY FINAL COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("NU TUNING RESULTS")
print("=" * 70)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# SAVE RESULTS
# ============================================================

output_path = "results/nu_tuning_results.csv"

results_df.to_csv(
    output_path,
    index=False
)

print("\nResults saved to:")
print(output_path)


# ============================================================
# BEST VALUES BY DIFFERENT METRICS
# ============================================================

print("\n" + "=" * 70)
print("METRIC COMPARISON")
print("=" * 70)

best_f1 = results_df.loc[
    results_df["f1"].idxmax()
]

best_recall = results_df.loc[
    results_df["recall"].idxmax()
]

best_pr_auc = results_df.loc[
    results_df["pr_auc"].idxmax()
]

print("\nHighest F1-score:")
print(
    f"nu = {best_f1['nu']:.2f}, "
    f"F1 = {best_f1['f1']:.4f}"
)

print("\nHighest Recall:")
print(
    f"nu = {best_recall['nu']:.2f}, "
    f"Recall = {best_recall['recall']:.4f}"
)

print("\nHighest PR-AUC:")
print(
    f"nu = {best_pr_auc['nu']:.2f}, "
    f"PR-AUC = {best_pr_auc['pr_auc']:.4f}"
)


# ============================================================
# COMPLETED
# ============================================================

print("\n" + "=" * 70)
print("NU TUNING COMPLETED")
print("=" * 70)

print("\nSaved:")
print("results/nu_tuning_results.csv")