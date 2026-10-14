import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
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

from qiskit.circuit.library import zz_feature_map
from qiskit_machine_learning.kernels import FidelityQuantumKernel


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "data/creditcard.csv"
RESULTS_DIR = "results"

N_QUBITS = 4
REPS = 1
ENTANGLEMENT = "linear"

N_TRAIN = 500

# Approximately the original dataset fraud prevalence (~0.17%)
N_FRAUD = 10
N_NORMAL = 5780

NU = 0.15
RANDOM_STATE = 42

os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# 1. LOAD DATA
# ============================================================

print("\n" + "=" * 60)
print("REALISTIC-PREVALENCE EVALUATION")
print("=" * 60)

df = pd.read_csv(DATA_PATH)

print(f"\nFull dataset shape: {df.shape}")

# Sort chronologically
df = df.sort_values("Time").reset_index(drop=True)

# Log-transform Amount
df["Amount"] = np.log1p(df["Amount"])

FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount"]


# ============================================================
# 2. SAME TIME-BASED SPLIT AS MAIN EXPERIMENT
# ============================================================

TRAIN_RATIO = 0.70

split_index = int(len(df) * TRAIN_RATIO)

train_df = df.iloc[:split_index].copy()
test_df = df.iloc[split_index:].copy()

print("\nTime-based split:")
print(f"Training rows: {len(train_df)}")
print(f"Test rows:     {len(test_df)}")


# ============================================================
# 3. NORMAL-ONLY TRAINING DATA
# ============================================================

normal_train_df = train_df[
    train_df["Class"] == 0
].copy()

train_sample = normal_train_df.sample(
    n=N_TRAIN,
    random_state=RANDOM_STATE
)

X_train_raw = train_sample[FEATURES].values


# ============================================================
# 4. CREATE PREVALENCE-MATCHED TEST SET
# ============================================================

normal_test_df = test_df[
    test_df["Class"] == 0
].copy()

fraud_test_df = test_df[
    test_df["Class"] == 1
].copy()

print("\nAvailable chronological test data:")
print(f"Normal transactions: {len(normal_test_df)}")
print(f"Fraud transactions:  {len(fraud_test_df)}")

# Select a reproducible subset
fraud_sample = fraud_test_df.sample(
    n=N_FRAUD,
    random_state=RANDOM_STATE
)

normal_sample = normal_test_df.sample(
    n=N_NORMAL,
    random_state=RANDOM_STATE
)

realistic_test_df = pd.concat(
    [normal_sample, fraud_sample],
    axis=0
)

# Shuffle test set
realistic_test_df = realistic_test_df.sample(
    frac=1,
    random_state=RANDOM_STATE
).reset_index(drop=True)

X_test_raw = realistic_test_df[FEATURES].values
y_test = realistic_test_df["Class"].values


# ============================================================
# 5. TEST PREVALENCE
# ============================================================

fraud_count = np.sum(y_test == 1)
normal_count = np.sum(y_test == 0)

fraud_rate = fraud_count / len(y_test)

print("\nRealistic test set:")
print(f"Normal: {normal_count}")
print(f"Fraud:  {fraud_count}")
print(f"Total:  {len(y_test)}")
print(f"Fraud prevalence: {fraud_rate * 100:.4f}%")


# ============================================================
# 6. STANDARDIZATION
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)


# ============================================================
# 7. PCA
# ============================================================

pca = PCA(n_components=N_QUBITS)

X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print("\nPCA explained variance:")
for i, value in enumerate(
    pca.explained_variance_ratio_,
    start=1
):
    print(f"PC{i}: {value:.6f}")

print(
    f"Total explained variance: "
    f"{pca.explained_variance_ratio_.sum():.6f}"
)


# ============================================================
# 8. SCALE TO QUANTUM ANGLES [0, PI]
# ============================================================

angle_scaler = MinMaxScaler(
    feature_range=(0, np.pi)
)

X_train = angle_scaler.fit_transform(X_train_pca)

X_test = angle_scaler.transform(X_test_pca)

X_test = np.clip(
    X_test,
    0,
    np.pi
)


# ============================================================
# 9. QUANTUM FEATURE MAP
# ============================================================

feature_map = zz_feature_map(
    feature_dimension=N_QUBITS,
    reps=REPS,
    entanglement=ENTANGLEMENT
)

print("\nQuantum configuration:")
print(f"Qubits:        {N_QUBITS}")
print(f"Feature map:   ZZ")
print(f"Reps:          {REPS}")
print(f"Entanglement:  {ENTANGLEMENT}")
print(f"Depth:         {feature_map.depth()}")


# ============================================================
# 10. QUANTUM KERNEL
# ============================================================

quantum_kernel = FidelityQuantumKernel(
    feature_map=feature_map
)

print("\nCalculating realistic test quantum kernel...")
print(
    f"Kernel dimensions: "
    f"{len(X_test)} x {len(X_train)}"
)

K_test_realistic = quantum_kernel.evaluate(
    x_vec=X_test,
    y_vec=X_train
)

print(
    f"Quantum test kernel shape: "
    f"{K_test_realistic.shape}"
)


# ============================================================
# 11. LOAD FINAL TRAINING KERNEL
# ============================================================

train_kernel_path = (
    "results/reps1_linear_K_train.npy"
)

if os.path.exists(train_kernel_path):

    print(
        "\nLoading existing final QML training kernel..."
    )

    K_train = np.load(
        train_kernel_path
    )

else:

    print(
        "\nTraining kernel not found."
    )

    print(
        "Calculating training quantum kernel..."
    )

    K_train = quantum_kernel.evaluate(
        x_vec=X_train,
        y_vec=X_train
    )


# ============================================================
# 12. QML ONE-CLASS SVM
# ============================================================

print("\nTraining QML One-Class SVM...")

qml_model = OneClassSVM(
    kernel="precomputed",
    nu=NU
)

qml_model.fit(K_train)


# ============================================================
# 13. QML PREDICTION
# ============================================================

qml_raw_predictions = qml_model.predict(
    K_test_realistic
)

qml_decision_scores = qml_model.decision_function(
    K_test_realistic
)

# One-Class SVM:
# +1 = normal
# -1 = anomaly

qml_predictions = np.where(
    qml_raw_predictions == -1,
    1,
    0
)


# ============================================================
# 14. QML METRICS
# ============================================================

qml_accuracy = accuracy_score(
    y_test,
    qml_predictions
)

qml_precision = precision_score(
    y_test,
    qml_predictions,
    zero_division=0
)

qml_recall = recall_score(
    y_test,
    qml_predictions,
    zero_division=0
)

qml_f1 = f1_score(
    y_test,
    qml_predictions,
    zero_division=0
)

# Convert anomaly score direction so that
# larger value means more likely fraud
qml_anomaly_score = -qml_decision_scores

qml_roc_auc = roc_auc_score(
    y_test,
    qml_anomaly_score
)

qml_pr_auc = average_precision_score(
    y_test,
    qml_anomaly_score
)

qml_cm = confusion_matrix(
    y_test,
    qml_predictions
)


# ============================================================
# 15. CLASSICAL RBF ONE-CLASS SVM
# ============================================================

print("\nTraining classical RBF One-Class SVM...")

classical_model = OneClassSVM(
    kernel="rbf",
    nu=NU
)

classical_model.fit(
    X_train_scaled
)

classical_raw_predictions = classical_model.predict(
    X_test_scaled
)

classical_decision_scores = (
    classical_model.decision_function(
        X_test_scaled
    )
)

classical_predictions = np.where(
    classical_raw_predictions == -1,
    1,
    0
)

classical_anomaly_score = (
    -classical_decision_scores
)


# ============================================================
# 16. CLASSICAL METRICS
# ============================================================

classical_accuracy = accuracy_score(
    y_test,
    classical_predictions
)

classical_precision = precision_score(
    y_test,
    classical_predictions,
    zero_division=0
)

classical_recall = recall_score(
    y_test,
    classical_predictions,
    zero_division=0
)

classical_f1 = f1_score(
    y_test,
    classical_predictions,
    zero_division=0
)

classical_roc_auc = roc_auc_score(
    y_test,
    classical_anomaly_score
)

classical_pr_auc = average_precision_score(
    y_test,
    classical_anomaly_score
)

classical_cm = confusion_matrix(
    y_test,
    classical_predictions
)


# ============================================================
# 17. PRINT RESULTS
# ============================================================

print("\n" + "=" * 60)
print("QML RESULTS — PREVALENCE-MATCHED TEST")
print("=" * 60)

print(f"Accuracy  : {qml_accuracy:.4f}")
print(f"Precision : {qml_precision:.4f}")
print(f"Recall    : {qml_recall:.4f}")
print(f"F1 Score  : {qml_f1:.4f}")
print(f"ROC-AUC   : {qml_roc_auc:.4f}")
print(f"PR-AUC    : {qml_pr_auc:.4f}")

print("\nQML Confusion Matrix:")
print(qml_cm)

print("\nQML Classification Report:")
print(
    classification_report(
        y_test,
        qml_predictions,
        target_names=["Normal", "Fraud"],
        zero_division=0
    )
)


print("\n" + "=" * 60)
print("CLASSICAL RESULTS — PREVALENCE-MATCHED TEST")
print("=" * 60)

print(f"Accuracy  : {classical_accuracy:.4f}")
print(f"Precision : {classical_precision:.4f}")
print(f"Recall    : {classical_recall:.4f}")
print(f"F1 Score  : {classical_f1:.4f}")
print(f"ROC-AUC   : {classical_roc_auc:.4f}")
print(f"PR-AUC    : {classical_pr_auc:.4f}")

print("\nClassical Confusion Matrix:")
print(classical_cm)

print("\nClassical Classification Report:")
print(
    classification_report(
        y_test,
        classical_predictions,
        target_names=["Normal", "Fraud"],
        zero_division=0
    )
)


# ============================================================
# 18. COMPARISON
# ============================================================

print("\n" + "=" * 60)
print("QML vs CLASSICAL")
print("=" * 60)

comparison = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC",
        "PR-AUC"
    ],
    "QML": [
        qml_accuracy,
        qml_precision,
        qml_recall,
        qml_f1,
        qml_roc_auc,
        qml_pr_auc
    ],
    "Classical": [
        classical_accuracy,
        classical_precision,
        classical_recall,
        classical_f1,
        classical_roc_auc,
        classical_pr_auc
    ]
})

comparison["Difference_QML_minus_Classical"] = (
    comparison["QML"]
    - comparison["Classical"]
)

print("\n")
print(
    comparison.to_string(
        index=False
    )
)


# ============================================================
# 19. SAVE RESULTS
# ============================================================

np.save(
    "results/realistic_y_test.npy",
    y_test
)

np.save(
    "results/realistic_qml_predictions.npy",
    qml_predictions
)

np.save(
    "results/realistic_qml_decision_scores.npy",
    qml_anomaly_score
)

np.save(
    "results/realistic_qml_confusion_matrix.npy",
    qml_cm
)

np.save(
    "results/realistic_classical_predictions.npy",
    classical_predictions
)

np.save(
    "results/realistic_classical_decision_scores.npy",
    classical_anomaly_score
)

np.save(
    "results/realistic_classical_confusion_matrix.npy",
    classical_cm
)

comparison.to_csv(
    "results/realistic_comparison.csv",
    index=False
)

realistic_test_df.to_csv(
    "results/realistic_test_set.csv",
    index=False
)


print("\nResults saved to:")
print("results/realistic_y_test.npy")
print("results/realistic_qml_predictions.npy")
print("results/realistic_qml_decision_scores.npy")
print("results/realistic_qml_confusion_matrix.npy")
print("results/realistic_classical_predictions.npy")
print("results/realistic_classical_decision_scores.npy")
print("results/realistic_classical_confusion_matrix.npy")
print("results/realistic_comparison.csv")
print("results/realistic_test_set.csv")

print("\nEvaluation complete.")