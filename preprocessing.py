import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA


# ============================================================
# QML ANOMALY DETECTION
# DATA PREPROCESSING PIPELINE
# ============================================================

DATA_PATH = "data/creditcard.csv"

# Initial experiment configuration
N_QUBITS = 4
TRAIN_RATIO = 0.70

N_TRAIN = 500
N_TEST_NORMAL = 500

RANDOM_STATE = 42


# ------------------------------------------------------------
# 1. Load dataset
# ------------------------------------------------------------

print("=" * 70)
print("LOADING DATASET")
print("=" * 70)

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")


# ------------------------------------------------------------
# 2. Sort by time
# ------------------------------------------------------------

df = df.sort_values("Time").reset_index(drop=True)

print("\nDataset sorted by Time.")


# ------------------------------------------------------------
# 3. Log-transform Amount
# ------------------------------------------------------------

df["Amount"] = np.log1p(df["Amount"])

print("Applied log1p transformation to Amount.")


# ------------------------------------------------------------
# 4. Define model features
# ------------------------------------------------------------

FEATURES = [
    f"V{i}"
    for i in range(1, 29)
] + ["Amount"]

print("\nModel features:")
print(FEATURES)

print(f"\nNumber of original features: {len(FEATURES)}")


# ------------------------------------------------------------
# 5. Time-based train/test split
# ------------------------------------------------------------

split_index = int(
    len(df) * TRAIN_RATIO
)

train_df = df.iloc[:split_index].copy()
test_df = df.iloc[split_index:].copy()

print("\n" + "=" * 70)
print("TIME-BASED SPLIT")
print("=" * 70)

print(
    f"Training rows : {len(train_df):,}"
)

print(
    f"Testing rows  : {len(test_df):,}"
)

print(
    f"Training time range: "
    f"{train_df['Time'].min()} → {train_df['Time'].max()}"
)

print(
    f"Testing time range: "
    f"{test_df['Time'].min()} → {test_df['Time'].max()}"
)


# ------------------------------------------------------------
# 6. Training data = NORMAL ONLY
# ------------------------------------------------------------

normal_train_df = train_df[
    train_df["Class"] == 0
].copy()

print("\n" + "=" * 70)
print("NORMAL TRAINING DATA")
print("=" * 70)

print(
    f"Available normal training rows: "
    f"{len(normal_train_df):,}"
)

print(
    f"Fraud rows in training data: "
    f"{(train_df['Class'] == 1).sum():,}"
)


# ------------------------------------------------------------
# 7. Select training sample
# ------------------------------------------------------------

train_df = normal_train_df.sample(
    n=N_TRAIN,
    random_state=RANDOM_STATE
)

print(
    f"\nSelected {len(train_df)} normal "
    f"transactions for QML training."
)


# ------------------------------------------------------------
# 8. Build test set
# ------------------------------------------------------------

normal_test_df = test_df[
    test_df["Class"] == 0
].copy()

fraud_test_df = test_df[
    test_df["Class"] == 1
].copy()


normal_test_df = normal_test_df.sample(
    n=min(
        N_TEST_NORMAL,
        len(normal_test_df)
    ),
    random_state=RANDOM_STATE
)


test_df = pd.concat(
    [
        normal_test_df,
        fraud_test_df
    ],
    axis=0
)

# Shuffle test rows
test_df = test_df.sample(
    frac=1,
    random_state=RANDOM_STATE
)


print("\n" + "=" * 70)
print("TEST DATA")
print("=" * 70)

print(
    f"Normal test samples : "
    f"{len(normal_test_df):,}"
)

print(
    f"Fraud test samples  : "
    f"{len(fraud_test_df):,}"
)

print(
    f"Total test samples  : "
    f"{len(test_df):,}"
)


# ------------------------------------------------------------
# 9. Extract raw features
# ------------------------------------------------------------

X_train_raw = train_df[
    FEATURES
].values

X_test_raw = test_df[
    FEATURES
].values

y_test = test_df[
    "Class"
].values


print("\nRaw training shape:", X_train_raw.shape)
print("Raw testing shape :", X_test_raw.shape)


# ------------------------------------------------------------
# 10. StandardScaler
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STANDARD SCALING")
print("=" * 70)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train_raw
)

X_test_scaled = scaler.transform(
    X_test_raw
)

print(
    "Scaler fitted ONLY on training data."
)

print(
    "Scaled training shape:",
    X_train_scaled.shape
)

print(
    "Scaled testing shape:",
    X_test_scaled.shape
)


# ------------------------------------------------------------
# 11. PCA
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("PCA")
print("=" * 70)

pca = PCA(
    n_components=N_QUBITS
)

X_train_pca = pca.fit_transform(
    X_train_scaled
)

X_test_pca = pca.transform(
    X_test_scaled
)

print(
    f"PCA components: {N_QUBITS}"
)

print(
    "PCA training shape:",
    X_train_pca.shape
)

print(
    "PCA testing shape:",
    X_test_pca.shape
)


print("\nExplained variance ratio:")

for i, variance in enumerate(
    pca.explained_variance_ratio_,
    start=1
):

    print(
        f"PC{i}: {variance:.6f}"
    )


print(
    f"\nTotal explained variance: "
    f"{pca.explained_variance_ratio_.sum():.6f}"
)


# ------------------------------------------------------------
# 12. Scale PCA features to quantum angles
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("QUANTUM ANGLE SCALING")
print("=" * 70)

angle_scaler = MinMaxScaler(
    feature_range=(0, np.pi)
)

X_train = angle_scaler.fit_transform(
    X_train_pca
)

X_test = angle_scaler.transform(
    X_test_pca
)

# Protect against numerical values slightly outside range
X_test = np.clip(
    X_test,
    0,
    np.pi
)


print(
    "Angle range: [0, π]"
)

print(
    "Quantum training shape:",
    X_train.shape
)

print(
    "Quantum testing shape:",
    X_test.shape
)

print(
    "\nTraining angle range:"
)

print(
    f"Min: {X_train.min():.6f}"
)

print(
    f"Max: {X_train.max():.6f}"
)

print(
    "\nTesting angle range:"
)

print(
    f"Min: {X_test.min():.6f}"
)

print(
    f"Max: {X_test.max():.6f}"
)


# ------------------------------------------------------------
# 13. Final summary
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("PREPROCESSING COMPLETE")
print("=" * 70)

print(
    f"Original features : {len(FEATURES)}"
)

print(
    f"Quantum features  : {N_QUBITS}"
)

print(
    f"Training samples  : {len(X_train):,}"
)

print(
    f"Testing samples   : {len(X_test):,}"
)

print(
    f"Fraud in test     : {y_test.sum():,}"
)

print(
    f"Normal in test    : {(y_test == 0).sum():,}"
)

print("\nReady for quantum kernel.")