import os
import numpy as np

from sklearn.svm import OneClassSVM
from sklearn.metrics import confusion_matrix, classification_report

from preprocessing import X_train, X_test, y_test
from quantum_kernel import quantum_kernel


# ============================================================
# QML ANOMALY DETECTION - ONE-CLASS SVM
# ============================================================

print("=" * 70)
print("QML ANOMALY DETECTION - ONE-CLASS SVM")
print("=" * 70)


# ============================================================
# USE FULL PREPROCESSED DATA
# ============================================================

X_train_full = X_train
X_test_full = X_test
y_test_full = y_test

print("\nTraining samples:", len(X_train_full))
print("Testing samples :", len(X_test_full))


# ============================================================
# CREATE RESULTS DIRECTORY
# ============================================================

os.makedirs("results", exist_ok=True)

print("\nResults directory ready.")


# ============================================================
# TRAINING QUANTUM KERNEL
# ============================================================

K_train_path = "results/K_train.npy"

print("\n" + "=" * 70)
print("TRAINING QUANTUM KERNEL")
print("=" * 70)

if os.path.exists(K_train_path):

    print("\nSaved training kernel found.")
    print("Loading K_train...")

    K_train = np.load(K_train_path)

else:

    print("\nNo saved training kernel found.")
    print("Computing training quantum kernel...")
    print("This may take several minutes.")

    K_train = quantum_kernel.evaluate(
        x_vec=X_train_full
    )

    print("\nTraining quantum kernel calculation completed.")

    np.save(
        K_train_path,
        K_train
    )

    print("Training kernel saved to:")
    print(K_train_path)


print("\nTraining kernel shape:", K_train.shape)


# ============================================================
# TEST QUANTUM KERNEL
# ============================================================

K_test_path = "results/K_test.npy"

print("\n" + "=" * 70)
print("TEST QUANTUM KERNEL")
print("=" * 70)

if os.path.exists(K_test_path):

    print("\nSaved test kernel found.")
    print("Loading K_test...")

    K_test = np.load(K_test_path)

else:

    print("\nNo saved test kernel found.")
    print("Computing test quantum kernel...")
    print("This may take several minutes.")

    K_test = quantum_kernel.evaluate(
        x_vec=X_test_full,
        y_vec=X_train_full
    )

    print("\nTest quantum kernel calculation completed.")

    np.save(
        K_test_path,
        K_test
    )

    print("Test kernel saved to:")
    print(K_test_path)


print("\nTest kernel shape:", K_test.shape)


# ============================================================
# SAVE TEST LABELS
# ============================================================

y_test_path = "results/y_test.npy"

np.save(
    y_test_path,
    y_test_full
)

print("\nTest labels saved to:")
print(y_test_path)


# ============================================================
# TRAIN ONE-CLASS SVM
# ============================================================

print("\n" + "=" * 70)
print("TRAINING ONE-CLASS SVM")
print("=" * 70)

ocsvm = OneClassSVM(
    kernel="precomputed",
    nu=0.10
)

ocsvm.fit(K_train)

print("\nOne-Class SVM trained successfully.")


# ============================================================
# PREDICTION
# ============================================================

print("\n" + "=" * 70)
print("PREDICTING ANOMALIES")
print("=" * 70)

predictions = ocsvm.predict(
    K_test
)

decision_scores = ocsvm.decision_function(
    K_test
)


# ============================================================
# CONVERT ONE-CLASS SVM OUTPUT
# ============================================================

# One-Class SVM:
#
# +1 = normal
# -1 = anomaly
#
# Our dataset:
#
# 0 = normal
# 1 = fraud/anomaly

y_pred = np.where(
    predictions == -1,
    1,
    0
)


print("\nPrediction completed.")


# ============================================================
# SAVE PREDICTIONS
# ============================================================

predictions_path = "results/predictions.npy"
scores_path = "results/decision_scores.npy"

np.save(
    predictions_path,
    y_pred
)

np.save(
    scores_path,
    decision_scores
)

print("\nPredictions saved to:")
print(predictions_path)

print("\nDecision scores saved to:")
print(scores_path)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)


# ------------------------------------------------------------
# Confusion Matrix
# ------------------------------------------------------------

cm = confusion_matrix(
    y_test_full,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)


# ------------------------------------------------------------
# Classification Report
# ------------------------------------------------------------

print("\nClassification Report:")

print(
    classification_report(
        y_test_full,
        y_pred,
        zero_division=0
    )
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("QML BASELINE COMPLETED")
print("=" * 70)

print("\nSaved files:")

print("1. results/K_train.npy")
print("2. results/K_test.npy")
print("3. results/y_test.npy")
print("4. results/predictions.npy")
print("5. results/decision_scores.npy")

print("\nQuantum kernel matrices will be reused")
print("in the next experiments without recalculating them.")

print("\nDone.")