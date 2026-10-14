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

    confusion_matrix

)



from qiskit.circuit.library import zz_feature_map

from qiskit_machine_learning.kernels import FidelityQuantumKernel





# ============================================================

# CONFIGURATION

# ============================================================



DATA_PATH = "data/creditcard.csv"



OUTPUT_DIR = "results/final_1000"

os.makedirs(OUTPUT_DIR, exist_ok=True)



N_QUBITS = 4

REPS = 2

ENTANGLEMENT = "linear"



NU = 0.08



TRAIN_RATIO = 0.70



N_TRAIN = 1000

N_TEST_NORMAL = 500



RANDOM_STATE = 42



FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount"]





# ============================================================

# HEADER

# ============================================================



print("=" * 60)

print("1000-SAMPLE TRAINING + REPS=2 EXPERIMENT")

print("=" * 60)





# ============================================================

# LOAD DATASET

# ============================================================



print("\nLoading dataset...")



df = pd.read_csv(DATA_PATH)



print("Dataset shape:", df.shape)





# ============================================================

# SORT BY TIME

# ============================================================



df = df.sort_values("Time").reset_index(drop=True)



print("\nTime sorted.")





# ============================================================

# LOG TRANSFORM AMOUNT

# ============================================================



df["Amount"] = np.log1p(df["Amount"])



print("Amount transformed using log1p.")



print("Number of features:", len(FEATURES))





# ============================================================

# CHRONOLOGICAL TRAIN / TEST SPLIT

# ============================================================



split_index = int(len(df) * TRAIN_RATIO)



train_df = df.iloc[:split_index].copy()

test_df = df.iloc[split_index:].copy()



print("\nChronological split:")

print("Training rows:", len(train_df))

print("Test rows:", len(test_df))





# ============================================================

# NORMAL TRAINING DATA

# ============================================================



normal_train_df = train_df[

    train_df["Class"] == 0

].copy()



print(

    "\nAvailable normal training transactions:",

    len(normal_train_df)

)



train_df = normal_train_df.sample(

    n=N_TRAIN,

    random_state=RANDOM_STATE

)





# ============================================================

# TEST DATA

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

)



print("\nTest composition:")

print("Normal:", len(normal_test_df))

print("Fraud :", len(fraud_test_df))

print("Total :", len(test_df))





# ============================================================

# RAW FEATURES

# ============================================================



X_train_raw = train_df[FEATURES].values

X_test_raw = test_df[FEATURES].values



y_test = test_df["Class"].values



print("\nRaw training shape:", X_train_raw.shape)

print("Raw testing shape :", X_test_raw.shape)





# ============================================================

# STANDARD SCALING

# ============================================================



print("\nFitting StandardScaler...")



scaler = StandardScaler()



X_train_scaled = scaler.fit_transform(

    X_train_raw

)



X_test_scaled = scaler.transform(

    X_test_raw

)





# ============================================================

# PCA

# ============================================================



print("Fitting PCA...")



pca = PCA(

    n_components=N_QUBITS

)



X_train_pca = pca.fit_transform(

    X_train_scaled

)



X_test_pca = pca.transform(

    X_test_scaled

)



print("\nPCA explained variance:")



for i, variance in enumerate(

    pca.explained_variance_ratio_,

    start=1

):

    print(

        f"PC{i}: {variance:.6f}"

    )



print(

    "Total:",

    f"{pca.explained_variance_ratio_.sum():.6f}"

)





# ============================================================

# SCALE TO QUANTUM ANGLES

# ============================================================



print("\nScaling features to [0, pi]...")



angle_scaler = MinMaxScaler(

    feature_range=(0, np.pi)

)



X_train = angle_scaler.fit_transform(

    X_train_pca

)



X_test = angle_scaler.transform(

    X_test_pca

)



X_test = np.clip(

    X_test,

    0,

    np.pi

)





# ============================================================

# QUANTUM FEATURE MAP

# ============================================================



print("\nCreating ZZ feature map...")



feature_map = zz_feature_map(

    feature_dimension=N_QUBITS,

    reps=REPS,

    entanglement=ENTANGLEMENT

)



print("Qubits       :", N_QUBITS)

print("Repetitions  :", REPS)

print("Entanglement :", ENTANGLEMENT)

print("Circuit depth:", feature_map.depth())





# ============================================================

# QUANTUM KERNEL

# ============================================================



print("\nCreating Fidelity Quantum Kernel...")



quantum_kernel = FidelityQuantumKernel(

    feature_map=feature_map

)





# ============================================================

# TRAINING KERNEL

# ============================================================



print("\nComputing training quantum kernel...")



print(

    "Expected kernel size:",

    f"{N_TRAIN} x {N_TRAIN}"

)



K_train = quantum_kernel.evaluate(

    X_train

)



np.save(

    f"{OUTPUT_DIR}/K_train.npy",

    K_train

)



print("Training kernel saved.")





# ============================================================

# TEST KERNEL

# ============================================================



print("\nComputing test quantum kernel...")



print(

    "Expected kernel size:",

    f"{len(X_test)} x {N_TRAIN}"

)



K_test = quantum_kernel.evaluate(

    X_test,

    X_train

)



np.save(

    f"{OUTPUT_DIR}/K_test.npy",

    K_test

)



print("Test kernel saved.")





# ============================================================

# KERNEL SHAPES

# ============================================================



print("\nKernel shapes:")



print(

    "K_train:",

    K_train.shape

)



print(

    "K_test :",

    K_test.shape

)





# ============================================================

# SAVE TEST LABELS

# ============================================================



np.save(

    f"{OUTPUT_DIR}/y_test.npy",

    y_test

)





# ============================================================

# ONE-CLASS SVM

# ============================================================



print("\nTraining One-Class SVM...")



model = OneClassSVM(

    kernel="precomputed",

    nu=NU

)



model.fit(K_train)





# ============================================================

# PREDICTIONS

# ============================================================



print("Generating predictions...")



raw_predictions = model.predict(K_test)



# OCSVM:

# +1 = normal

# -1 = anomaly



predictions = np.where(

    raw_predictions == -1,

    1,

    0

)



decision_scores = model.decision_function(

    K_test

)





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

# RESULTS

# ============================================================



print("\n")

print("=" * 60)

print("1000-SAMPLE REPS=2 RESULTS")

print("=" * 60)



print(

    f"Accuracy  : {accuracy:.4f}"

)



print(

    f"Precision : {precision:.4f}"

)



print(

    f"Recall    : {recall:.4f}"

)



print(

    f"F1 Score  : {f1:.4f}"

)



print(

    f"ROC-AUC   : {roc_auc:.4f}"

)



print(

    f"PR-AUC    : {pr_auc:.4f}"

)



print("\nConfusion Matrix:")



print(cm)





# ============================================================

# SAVE RESULTS

# ============================================================



np.save(

    f"{OUTPUT_DIR}/predictions.npy",

    predictions

)



np.save(

    f"{OUTPUT_DIR}/decision_scores.npy",

    decision_scores

)



np.save(

    f"{OUTPUT_DIR}/confusion_matrix.npy",

    cm

)



results = pd.DataFrame([{

    "training_samples": N_TRAIN,

    "qubits": N_QUBITS,

    "reps": REPS,

    "entanglement": ENTANGLEMENT,

    "nu": NU,

    "accuracy": accuracy,

    "precision": precision,

    "recall": recall,

    "f1": f1,

    "roc_auc": roc_auc,

    "pr_auc": pr_auc

}])



results.to_csv(

    f"{OUTPUT_DIR}/results.csv",

    index=False

)





# ============================================================

# FINAL

# ============================================================



print("\nResults saved to:")

print(OUTPUT_DIR)



print("\nExperiment complete.")