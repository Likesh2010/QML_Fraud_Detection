import os
import numpy as np

from sklearn.svm import OneClassSVM
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# FINAL QML MODEL
# ============================================================

NU = 0.15

K_TRAIN_PATH = "results/reps1_linear_K_train.npy"
K_TEST_PATH = "results/reps1_linear_K_test.npy"
Y_TEST_PATH = "results/y_test.npy"

RESULTS_DIR = "results"


print("=" * 70)
print("FINAL QML ANOMALY DETECTION MODEL")
print("=" * 70)

print("\nConfiguration:")
print("Feature map       : ZZ")
print("Qubits            : 4")
print("Repetitions       : 1")
print("Entanglement      : linear")
print("Kernel            : Fidelity Quantum Kernel")
print("Detector          : One-Class SVM")
print(f"Nu                : {NU}")


# ============================================================
# LOAD SAVED QUANTUM KERNELS
# ============================================================

print("\n" + "=" * 70)
print("LOADING QUANTUM KERNEL MATRICES")
print("=" * 70)

if not os.path.exists(K_TRAIN_PATH):
    raise FileNotFoundError(
        f"Training kernel not found: {K_TRAIN_PATH}"
    )

if not os.path.exists(K_TEST_PATH):
    raise FileNotFoundError(
        f"Test kernel not found: {K_TEST_PATH}"
    )

if not os.path.exists(Y_TEST_PATH):
    raise FileNotFoundError(
        f"Test labels not found: {Y_TEST_PATH}"
    )


K_train = np.load(K_TRAIN_PATH)
K_test = np.load(K_TEST_PATH)
y_test = np.load(Y_TEST_PATH)


print(f"K_train shape : {K_train.shape}")
print(f"K_test shape  : {K_test.shape}")
print(f"y_test shape  : {y_test.shape}")


# ============================================================
# TRAIN ONE-CLASS SVM
# ============================================================

print("\n" + "=" * 70)
print("TRAINING ONE-CLASS SVM")
print("=" * 70)

model = OneClassSVM(
    kernel="precomputed",
    nu=NU
)

model.fit(K_train)

print("One-Class SVM training completed.")


# ============================================================
# PREDICTION
# ============================================================

print("\n" + "=" * 70)
print("GENERATING PREDICTIONS")
print("=" * 70)

raw_predictions = model.predict(K_test)

# One-Class SVM:
# +1 = normal
# -1 = anomaly
#
# Convert to:
#  0 = normal
#  1 = fraud/anomaly

predictions = np.where(raw_predictions == -1, 1, 0)

decision_scores = -model.decision_function(K_test)


# ============================================================
# METRICS
# ============================================================

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
    decision_scores
)

pr_auc = average_precision_score(
    y_test,
    decision_scores
)

cm = confusion_matrix(
    y_test,
    predictions
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 70)
print("FINAL QML RESULTS")
print("=" * 70)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")
print(f"PR-AUC    : {pr_auc:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        target_names=["Normal", "Fraud"],
        zero_division=0
    )
)


# ============================================================
# SAVE FINAL RESULTS
# ============================================================

print("\n" + "=" * 70)
print("SAVING FINAL QML RESULTS")
print("=" * 70)

np.save(
    os.path.join(RESULTS_DIR, "final_qml_predictions.npy"),
    predictions
)

np.save(
    os.path.join(RESULTS_DIR, "final_qml_decision_scores.npy"),
    decision_scores
)

np.save(
    os.path.join(RESULTS_DIR, "final_qml_confusion_matrix.npy"),
    cm
)

print("Saved:")
print(" - final_qml_predictions.npy")
print(" - final_qml_decision_scores.npy")
print(" - final_qml_confusion_matrix.npy")

print("\n" + "=" * 70)
print("FINAL QML MODEL COMPLETE")
print("=" * 70)