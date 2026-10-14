import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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
# 1000-SAMPLE CLASSICAL ONE-CLASS SVM
# ============================================================

DATA_PATH = "data/creditcard.csv"
OUTPUT_DIR = "results/final_1000"

N_TRAIN = 1000
N_TEST_NORMAL = 500
NU = 0.08
N_COMPONENTS = 4
RANDOM_STATE = 42

FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("1000-SAMPLE CLASSICAL ONE-CLASS SVM")
print("=" * 60)

# ------------------------------------------------------------
# Load dataset
# ------------------------------------------------------------

df = pd.read_csv(DATA_PATH)

print("\nDataset shape:", df.shape)

# ------------------------------------------------------------
# Chronological split
# ------------------------------------------------------------

train_df = df[df["Time"] <= 132928].copy()
test_df = df[df["Time"] >= 132929].copy()

print("Chronological train:", train_df.shape)
print("Chronological test :", test_df.shape)

# ------------------------------------------------------------
# Normal-only training
# ------------------------------------------------------------

normal_train = train_df[train_df["Class"] == 0].copy()

train_sample = normal_train.sample(
    n=N_TRAIN,
    random_state=RANDOM_STATE
)

# ------------------------------------------------------------
# Test set: 500 normal + all fraud
# ------------------------------------------------------------

test_normal = test_df[test_df["Class"] == 0].sample(
    n=N_TEST_NORMAL,
    random_state=RANDOM_STATE
)

test_fraud = test_df[test_df["Class"] == 1].copy()

test_sample = pd.concat(
    [test_normal, test_fraud],
    axis=0
).sample(
    frac=1,
    random_state=RANDOM_STATE
)

print("\nTraining normal samples:", len(train_sample))
print("Testing normal samples :", len(test_normal))
print("Testing fraud samples  :", len(test_fraud))
print("Total test samples     :", len(test_sample))

# ------------------------------------------------------------
# Prepare features
# ------------------------------------------------------------

X_train = train_sample[FEATURES].copy()
X_test = test_sample[FEATURES].copy()

y_test = test_sample["Class"].values

# Log transform Amount
X_train["Amount"] = np.log1p(X_train["Amount"])
X_test["Amount"] = np.log1p(X_test["Amount"])

# ------------------------------------------------------------
# Standardization
# ------------------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------------------------------------------------
# PCA: 29 -> 4
# ------------------------------------------------------------

pca = PCA(
    n_components=N_COMPONENTS,
    random_state=RANDOM_STATE
)

X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print("\nPCA")
print("Original features:", X_train.shape[1])
print("PCA components   :", N_COMPONENTS)
print(
    "Explained variance:",
    pca.explained_variance_ratio_.sum()
)

# ------------------------------------------------------------
# Classical One-Class SVM
# ------------------------------------------------------------

print("\nTraining classical One-Class SVM...")

model = OneClassSVM(
    kernel="rbf",
    nu=NU,
    gamma="scale"
)

model.fit(X_train_pca)

# ------------------------------------------------------------
# Prediction
# ------------------------------------------------------------

raw_predictions = model.predict(X_test_pca)

# OCSVM:
# +1 = normal
# -1 = anomaly
predictions = np.where(
    raw_predictions == -1,
    1,
    0
)

decision_scores = model.decision_function(X_test_pca)

# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------

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

tn, fp, fn, tp = cm.ravel()

# ------------------------------------------------------------
# Display results
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("CLASSICAL 1000 RESULTS")
print("=" * 60)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")
print(f"PR-AUC    : {pr_auc:.4f}")

print("\nConfusion Matrix:")
print(cm)

# ------------------------------------------------------------
# Save results
# ------------------------------------------------------------

results = pd.DataFrame([{
    "training_size": N_TRAIN,
    "nu": NU,
    "kernel": "RBF",
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
}])

output_file = f"{OUTPUT_DIR}/classical_1000_results.csv"

results.to_csv(
    output_file,
    index=False
)

np.save(
    f"{OUTPUT_DIR}/classical_1000_predictions.npy",
    predictions
)

np.save(
    f"{OUTPUT_DIR}/classical_1000_decision_scores.npy",
    decision_scores
)

np.save(
    f"{OUTPUT_DIR}/classical_1000_confusion_matrix.npy",
    cm
)

print("\nResults saved to:")
print(output_file)

print("\nExperiment complete.")