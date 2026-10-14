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

# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "data/creditcard.csv"
RESULTS_DIR = "results/training_size_experiment/700_classical"

N_QUBITS = 4
N_TRAIN = 700
N_TEST_NORMAL = 500
TRAIN_RATIO = 0.70
NU = 0.08
RANDOM_STATE = 42

FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount"]

os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("CLASSICAL ONE-CLASS SVM — 700 TRAINING SAMPLES")
print("=" * 60)

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")

df = df.sort_values("Time").reset_index(drop=True)
df["Amount"] = np.log1p(df["Amount"])

# ============================================================
# TIME-BASED SPLIT
# ============================================================

split_index = int(len(df) * TRAIN_RATIO)

train_df = df.iloc[:split_index].copy()
test_df = df.iloc[split_index:].copy()

normal_train_df = train_df[
    train_df["Class"] == 0
].copy()

print(f"Chronological training rows: {len(train_df)}")
print(f"Chronological test rows: {len(test_df)}")
print(f"Available normal training rows: {len(normal_train_df)}")

# ============================================================
# SELECT 700 NORMAL TRAINING SAMPLES
# ============================================================

train_selected = normal_train_df.sample(
    n=N_TRAIN,
    random_state=RANDOM_STATE
)

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

evaluation_df = pd.concat(
    [normal_test_df, fraud_test_df],
    axis=0
).sample(
    frac=1,
    random_state=RANDOM_STATE
).reset_index(drop=True)

print(f"Selected normal training samples: {len(train_selected)}")
print(f"Test normal samples: {len(normal_test_df)}")
print(f"Test fraud samples: {len(fraud_test_df)}")
print(f"Total test samples: {len(evaluation_df)}")

# ============================================================
# PREPROCESSING
# Same 29 -> StandardScaler -> PCA(4) pipeline
# used for the 700-sample QML experiment.
# ============================================================

X_train_raw = train_selected[FEATURES].values
X_test_raw = evaluation_df[FEATURES].values

y_test = evaluation_df["Class"].values

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

pca = PCA(n_components=N_QUBITS)

X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(
    "PCA explained variance:",
    pca.explained_variance_ratio_
)

print(
    "Total PCA explained variance:",
    pca.explained_variance_ratio_.sum()
)

# ============================================================
# CLASSICAL REPRESENTATION
# Use the same 4-dimensional PCA representation as QML.
# No quantum angle scaling is required for RBF SVM.
# ============================================================

X_train_classical = X_train_pca
X_test_classical = X_test_pca

# ============================================================
# CLASSICAL ONE-CLASS SVM
# ============================================================

print()
print("Training classical RBF One-Class SVM...")
print(f"Training samples: {len(X_train_classical)}")
print(f"Features: {X_train_classical.shape[1]}")
print(f"Kernel: RBF")
print(f"Nu: {NU}")

model = OneClassSVM(
    kernel="rbf",
    nu=NU
)

model.fit(X_train_classical)

# ============================================================
# PREDICTION
# ============================================================

raw_predictions = model.predict(
    X_test_classical
)

decision_scores = model.decision_function(
    X_test_classical
)

# sklearn:
# +1 = normal
# -1 = anomaly
predictions = np.where(
    raw_predictions == -1,
    1,
    0
)

# For ROC/PR, larger score should mean more anomalous.
anomaly_scores = -decision_scores

# ============================================================
# METRICS
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

print()
print("=" * 60)
print("CLASSICAL 700-SAMPLE RESULTS")
print("=" * 60)

print(f"Accuracy : {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Precision: {precision:.4f} ({precision * 100:.2f}%)")
print(f"Recall   : {recall:.4f} ({recall * 100:.2f}%)")
print(f"F1 Score : {f1:.4f} ({f1 * 100:.2f}%)")
print(f"ROC-AUC  : {roc_auc:.4f} ({roc_auc * 100:.2f}%)")
print(f"PR-AUC   : {pr_auc:.4f} ({pr_auc * 100:.2f}%)")

print()
print("Confusion Matrix:")
print(cm)

# ============================================================
# SAVE RESULTS
# ============================================================

np.save(
    os.path.join(
        RESULTS_DIR,
        "predictions.npy"
    ),
    predictions
)

np.save(
    os.path.join(
        RESULTS_DIR,
        "decision_scores.npy"
    ),
    anomaly_scores
)

np.save(
    os.path.join(
        RESULTS_DIR,
        "y_test.npy"
    ),
    y_test
)

np.save(
    os.path.join(
        RESULTS_DIR,
        "confusion_matrix.npy"
    ),
    cm
)

metrics_df = pd.DataFrame(
    {
        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "ROC-AUC",
            "PR-AUC",
        ],
        "Value": [
            accuracy,
            precision,
            recall,
            f1,
            roc_auc,
            pr_auc,
        ],
    }
)

metrics_df.to_csv(
    os.path.join(
        RESULTS_DIR,
        "metrics.csv"
    ),
    index=False
)

print()
print(f"Results saved to: {RESULTS_DIR}")
print("Done.")
