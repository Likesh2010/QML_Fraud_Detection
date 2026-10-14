import os
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
# 1000-SAMPLE NU FINE-TUNING
# Uses precomputed quantum kernels
# ============================================================

K_TRAIN_PATH = "results/final_1000/K_train.npy"
K_TEST_PATH = "results/final_1000/K_test.npy"
Y_TEST_PATH = "results/final_1000/y_test.npy"

OUTPUT_DIR = "results/final_1000/nu_tuning"
os.makedirs(OUTPUT_DIR, exist_ok=True)

NU_VALUES = [
    0.01,
    0.03,
    0.05,
    0.08,
    0.10,
    0.12,
    0.15,
    0.20,
    0.25
]

print("=" * 60)
print("1000-SAMPLE NU FINE-TUNING")
print("=" * 60)

# ------------------------------------------------------------
# Load precomputed kernels
# ------------------------------------------------------------

print("\nLoading precomputed quantum kernels...")

K_train = np.load(K_TRAIN_PATH)
K_test = np.load(K_TEST_PATH)
y_test = np.load(Y_TEST_PATH)

print("K_train shape:", K_train.shape)
print("K_test shape :", K_test.shape)
print("y_test shape :", y_test.shape)

# ------------------------------------------------------------
# Fine-tuning
# ------------------------------------------------------------

results = []

for nu in NU_VALUES:

    print("\n" + "-" * 60)
    print(f"Testing nu = {nu}")
    print("-" * 60)

    model = OneClassSVM(
        kernel="precomputed",
        nu=nu
    )

    # Train using quantum kernel
    model.fit(K_train)

    # Raw predictions:
    # +1 = normal
    # -1 = anomaly
    raw_predictions = model.predict(K_test)

    # Convert:
    # 0 = normal
    # 1 = fraud/anomaly
    predictions = np.where(raw_predictions == -1, 1, 0)

    # Decision function:
    # Higher value = more normal
    decision_scores = model.decision_function(K_test)

    # For anomaly ranking:
    anomaly_scores = -decision_scores

    # Metrics
    accuracy = accuracy_score(y_test, predictions)
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

    roc_auc = roc_auc_score(
        y_test,
        anomaly_scores
    )

    pr_auc = average_precision_score(
        y_test,
        anomaly_scores
    )

    cm = confusion_matrix(
        y_test,
        predictions
    )

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}")
    print(f"PR-AUC    : {pr_auc:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    results.append({
        "nu": nu,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "TN": cm[0, 0],
        "FP": cm[0, 1],
        "FN": cm[1, 0],
        "TP": cm[1, 1]
    })


# ------------------------------------------------------------
# Save results
# ------------------------------------------------------------

results_df = pd.DataFrame(results)

results_path = os.path.join(
    OUTPUT_DIR,
    "nu_tuning_results.csv"
)

results_df.to_csv(
    results_path,
    index=False
)

# ------------------------------------------------------------
# Find best values
# ------------------------------------------------------------

best_f1 = results_df.loc[
    results_df["f1"].idxmax()
]

best_recall = results_df.loc[
    results_df["recall"].idxmax()
]

best_pr_auc = results_df.loc[
    results_df["pr_auc"].idxmax()
]

best_roc_auc = results_df.loc[
    results_df["roc_auc"].idxmax()
]

print("\n")
print("=" * 60)
print("NU FINE-TUNING SUMMARY")
print("=" * 60)

print("\nAll Results:")
print(
    results_df[
        [
            "nu",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "pr_auc"
        ]
    ].to_string(index=False)
)

print("\nBest F1:")
print(
    f"nu = {best_f1['nu']:.2f}, "
    f"F1 = {best_f1['f1']:.4f}"
)

print("\nBest Recall:")
print(
    f"nu = {best_recall['nu']:.2f}, "
    f"Recall = {best_recall['recall']:.4f}"
)

print("\nBest ROC-AUC:")
print(
    f"nu = {best_roc_auc['nu']:.2f}, "
    f"ROC-AUC = {best_roc_auc['roc_auc']:.4f}"
)

print("\nBest PR-AUC:")
print(
    f"nu = {best_pr_auc['nu']:.2f}, "
    f"PR-AUC = {best_pr_auc['pr_auc']:.4f}"
)

print("\nResults saved to:")
print(results_path)

print("\nExperiment complete.")