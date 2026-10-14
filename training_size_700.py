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
)

from qiskit.circuit.library import zz_feature_map
from qiskit_machine_learning.kernels import FidelityQuantumKernel


# ============================================================
# SETTINGS
# ============================================================

DATA_PATH = "data/creditcard.csv"

N_QUBITS = 4
N_TRAIN = 700
N_TEST_NORMAL = 500

NU = 0.08
RANDOM_STATE = 42

RESULT_DIR = "results/training_size_experiment/700"
os.makedirs(RESULT_DIR, exist_ok=True)


# ============================================================
# 1. LOAD DATA
# ============================================================

print("=" * 60)
print("700-SAMPLE TRAINING SIZE EXPERIMENT")
print("=" * 60)

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)


# ============================================================
# 2. SORT BY TIME
# ============================================================

df = df.sort_values("Time").reset_index(drop=True)

print("\nTime sorted.")


# ============================================================
# 3. LOG TRANSFORM AMOUNT
# ============================================================

df["Amount"] = np.log1p(df["Amount"])


# ============================================================
# 4. FEATURES
# ============================================================

FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount"]

print("Number of features:", len(FEATURES))


# ============================================================
# 5. CHRONOLOGICAL TRAIN / TEST SPLIT
# ============================================================

split_index = int(len(df) * 0.70)

train_df = df.iloc[:split_index].copy()
test_df = df.iloc[split_index:].copy()

print("\nChronological split:")
print("Training rows:", len(train_df))
print("Test rows:", len(test_df))


# ============================================================
# 6. NORMAL-ONLY TRAINING DATA
# ============================================================

normal_train_df = train_df[
    train_df["Class"] == 0
].copy()

print("\nAvailable normal training transactions:",
      len(normal_train_df))


# Select 700 normal transactions
train_df = normal_train_df.sample(
    n=N_TRAIN,
    random_state=RANDOM_STATE
)


# ============================================================
# 7. TEST DATA
# ============================================================

normal_test_df = test_df[
    test_df["Class"] == 0
].copy()

fraud_test_df = test_df[
    test_df["Class"] == 1
].copy()


normal_test_df = normal_test_df.sample(
    n=min(N_TEST_NORMAL, len(normal_test_df)),
    random_state=RANDOM_STATE
)


test_df = pd.concat(
    [normal_test_df, fraud_test_df],
    axis=0
)

test_df = test_df.sample(
    frac=1,
    random_state=RANDOM_STATE
).reset_index(drop=True)


print("\nTest composition:")
print("Normal:", (test_df["Class"] == 0).sum())
print("Fraud :", (test_df["Class"] == 1).sum())
print("Total :", len(test_df))


# ============================================================
# 8. RAW FEATURES
# ============================================================

X_train_raw = train_df[FEATURES].values
X_test_raw = test_df[FEATURES].values

y_test = test_df["Class"].values


print("\nRaw training shape:", X_train_raw.shape)
print("Raw testing shape :", X_test_raw.shape)


# ============================================================
# 9. STANDARDIZATION
# ============================================================

print("\nFitting StandardScaler...")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)


# ============================================================
# 10. PCA
# ============================================================

print("Fitting PCA...")

pca = PCA(n_components=N_QUBITS)

X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

explained = pca.explained_variance_ratio_

print("\nPCA explained variance:")

for i, value in enumerate(explained, start=1):
    print(f"PC{i}: {value:.6f}")

print(
    "Total:",
    f"{explained.sum():.6f}"
)


# ============================================================
# 11. SCALE TO QUANTUM ANGLES
# ============================================================

print("\nScaling features to [0, pi]...")

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
# 12. QUANTUM FEATURE MAP
# ============================================================

print("\nCreating ZZ feature map...")

feature_map = zz_feature_map(
    feature_dimension=N_QUBITS,
    reps=1,
    entanglement="linear"
)

print("Qubits       :", N_QUBITS)
print("Repetitions  : 1")
print("Entanglement : linear")
print("Circuit depth:", feature_map.depth())


# ============================================================
# 13. QUANTUM KERNEL
# ============================================================

print("\nCreating Fidelity Quantum Kernel...")

quantum_kernel = FidelityQuantumKernel(
    feature_map=feature_map
)


# ============================================================
# 14. TRAINING KERNEL
# ============================================================

train_kernel_path = os.path.join(
    RESULT_DIR,
    "K_train.npy"
)

test_kernel_path = os.path.join(
    RESULT_DIR,
    "K_test.npy"
)


if os.path.exists(train_kernel_path):

    print("\nLoading cached training kernel...")

    K_train = np.load(train_kernel_path)

else:

    print("\nComputing training quantum kernel...")
    print(
        f"Expected kernel size: "
        f"{N_TRAIN} x {N_TRAIN}"
    )

    K_train = quantum_kernel.evaluate(
        x_vec=X_train
    )

    np.save(
        train_kernel_path,
        K_train
    )

    print("Training kernel saved.")


# ============================================================
# 15. TEST KERNEL
# ============================================================

if os.path.exists(test_kernel_path):

    print("\nLoading cached test kernel...")

    K_test = np.load(test_kernel_path)

else:

    print("\nComputing test quantum kernel...")
    print(
        f"Expected kernel size: "
        f"{len(X_test)} x {N_TRAIN}"
    )

    K_test = quantum_kernel.evaluate(
        x_vec=X_test,
        y_vec=X_train
    )

    np.save(
        test_kernel_path,
        K_test
    )

    print("Test kernel saved.")


print("\nKernel shapes:")
print("K_train:", K_train.shape)
print("K_test :", K_test.shape)


# ============================================================
# 16. ONE-CLASS SVM
# ============================================================

print("\nTraining One-Class SVM...")

model = OneClassSVM(
    kernel="precomputed",
    nu=NU
)

model.fit(K_train)


# ============================================================
# 17. PREDICTION
# ============================================================

print("Generating predictions...")

pred_raw = model.predict(K_test)

decision_scores = model.decision_function(K_test)


# Convert:
# +1 = normal -> 0
# -1 = anomaly -> 1

predictions = np.where(
    pred_raw == -1,
    1,
    0
)


# ============================================================
# 18. METRICS
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
    -decision_scores
)

pr_auc = average_precision_score(
    y_test,
    -decision_scores
)

cm = confusion_matrix(
    y_test,
    predictions
)


# ============================================================
# 19. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 60)
print("700-SAMPLE RESULTS")
print("=" * 60)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")
print(f"PR-AUC    : {pr_auc:.4f}")

print("\nConfusion Matrix:")
print(cm)


# ============================================================
# 20. SAVE RESULTS
# ============================================================

np.save(
    os.path.join(
        RESULT_DIR,
        "predictions.npy"
    ),
    predictions
)

np.save(
    os.path.join(
        RESULT_DIR,
        "decision_scores.npy"
    ),
    decision_scores
)

np.save(
    os.path.join(
        RESULT_DIR,
        "confusion_matrix.npy"
    ),
    cm
)

np.save(
    os.path.join(
        RESULT_DIR,
        "y_test.npy"
    ),
    y_test
)


results = pd.DataFrame([{
    "training_size": N_TRAIN,
    "nu": NU,
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "roc_auc": roc_auc,
    "pr_auc": pr_auc
}])

results.to_csv(
    os.path.join(
        RESULT_DIR,
        "results.csv"
    ),
    index=False
)

print("\nResults saved to:")
print(RESULT_DIR)

print("\nExperiment complete.")