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
# CLASSICAL ONE-CLASS SVM BASELINE
# ============================================================

NU = 0.15

RESULTS_DIR = "results"

print("=" * 70)
print("CLASSICAL ONE-CLASS SVM BASELINE")
print("=" * 70)

print("\nConfiguration:")
print("Representation : 4D PCA representation")
print("Kernel         : RBF")
print("Detector       : One-Class SVM")
print(f"Nu             : {NU}")


# ============================================================
# LOAD SAME PREPROCESSED DATA USED BY QML
# ============================================================

print("\n" + "=" * 70)
print("LOADING PREPROCESSED DATA")
print("=" * 70)

# Execute preprocessing.py so that:
# X_train, X_test and y_test are created.
#
# This guarantees that the classical model uses
# exactly the same data preparation as the QML model.

exec(open("preprocessing.py").read())

print("\nData loaded successfully.")

print(f"Training data shape : {X_train.shape}")
print(f"Testing data shape  : {X_test.shape}")
print(f"Test labels shape   : {y_test.shape}")

print(f"\nNormal training samples : {len(X_train)}")
print(f"Test samples             : {len(X_test)}")
print(f"Fraud test samples       : {np.sum(y_test == 1)}")
print(f"Normal test samples      : {np.sum(y_test == 0)}")


# ============================================================
# TRAIN CLASSICAL ONE-CLASS SVM
# ============================================================

print("\n" + "=" * 70)
print("TRAINING CLASSICAL ONE-CLASS SVM")
print("=" * 70)

model = OneClassSVM(
    kernel="rbf",
    nu=NU
)

model.fit(X_train)

print("One-Class SVM training completed.")


# ============================================================
# GENERATE PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("GENERATING PREDICTIONS")
print("=" * 70)

raw_predictions = model.predict(X_test)

# One-Class SVM output:
#
# +1 = normal
# -1 = anomaly
#
# Convert to:
#
# 0 = normal
# 1 = anomaly/fraud

predictions = np.where(
    raw_predictions == -1,
    1,
    0
)

# Larger score = more anomalous
decision_scores = -model.decision_function(X_test)

print("Predictions generated.")


# ============================================================
# CALCULATE METRICS
# ============================================================

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
print("CLASSICAL OCSVM RESULTS")
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
# SAVE RESULTS
# ============================================================

print("\n" + "=" * 70)
print("SAVING CLASSICAL RESULTS")
print("=" * 70)

os.makedirs(RESULTS_DIR, exist_ok=True)

np.save(
    os.path.join(
        RESULTS_DIR,
        "classical_predictions.npy"
    ),
    predictions
)

np.save(
    os.path.join(
        RESULTS_DIR,
        "classical_decision_scores.npy"
    ),
    decision_scores
)

np.save(
    os.path.join(
        RESULTS_DIR,
        "classical_confusion_matrix.npy"
    ),
    cm
)

print("Saved:")
print(" - results/classical_predictions.npy")
print(" - results/classical_decision_scores.npy")
print(" - results/classical_confusion_matrix.npy")


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("CLASSICAL MODEL COMPLETE")
print("=" * 70)