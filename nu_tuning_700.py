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
# 700-SAMPLE NU TUNING
# ============================================================

K_TRAIN_PATH = "results/training_size_experiment/700/K_train.npy"
K_TEST_PATH = "results/training_size_experiment/700/K_test.npy"
Y_TEST_PATH = "results/training_size_experiment/700/y_test.npy"

OUTPUT_DIR = "results/training_size_experiment/700"

NU_VALUES = [0.01, 0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20]

print("=" * 60)
print("700-SAMPLE NU TUNING")
print("=" * 60)

# ------------------------------------------------------------
# Load existing quantum kernels
# ------------------------------------------------------------

print("\nLoading existing quantum kernels...")

K_train = np.load(K_TRAIN_PATH)
K_test = np.load(K_TEST_PATH)
y_test = np.load(Y_TEST_PATH)

print("K_train shape:", K_train.shape)
print("K_test shape :", K_test.shape)
print("y_test shape :", y_test.shape)

results = []

# ------------------------------------------------------------
# Test different nu values
# ------------------------------------------------------------

for nu in NU_VALUES:

    print("\n" + "-" * 60)
    print(f"Testing nu = {nu}")
    print("-" * 60)

    model = OneClassSVM(
        kernel="precomputed",
        nu=nu
    )

    model.fit(K_train)

    raw_predictions = model.predict(K_test)

    # OCSVM:
    # +1 = normal
    # -1 = anomaly
    predictions = np.where(raw_predictions == -1, 1, 0)

    decision_scores = model.decision_function(K_test)

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
        -decision_scores
    )

    pr_auc = average_precision_score(
        y_test,
        -decision_scores
    )

    cm = confusion_matrix(y_test, predictions)

    tn, fp, fn, tp = cm.ravel()

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
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp
    })

# ------------------------------------------------------------
# Save results
# ------------------------------------------------------------

results_df = pd.DataFrame(results)

output_file = f"{OUTPUT_DIR}/nu_tuning_700_results.csv"

results_df.to_csv(
    output_file,
    index=False
)

print("\n" + "=" * 60)
print("FINAL NU TUNING RESULTS")
print("=" * 60)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print("\nHighest accuracy:")
best_accuracy = results_df.loc[
    results_df["accuracy"].idxmax()
]

print(best_accuracy)

print("\nHighest F1:")
best_f1 = results_df.loc[
    results_df["f1"].idxmax()
]

print(best_f1)

print("\nHighest ROC-AUC:")
best_roc = results_df.loc[
    results_df["roc_auc"].idxmax()
]

print(best_roc)

print("\nHighest PR-AUC:")
best_pr = results_df.loc[
    results_df["pr_auc"].idxmax()
]

print(best_pr)

print("\nResults saved to:")
print(output_file)

print("\nExperiment complete.")