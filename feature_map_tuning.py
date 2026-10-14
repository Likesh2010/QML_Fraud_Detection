import os
import numpy as np
import pandas as pd

from pathlib import Path
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

from preprocessing import X_train, X_test, y_test


# ============================================================
# CONFIGURATION
# ============================================================

RESULTS_DIR = Path("results")
CACHE_DIR = RESULTS_DIR / "feature_map_cache"

RESULTS_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

# Same nu used for the architecture comparison.
# We previously found nu=0.15 to be a useful development candidate.
NU = 0.15

# To prevent losing a large computation if interrupted.
BLOCK_SIZE = 100


# Feature-map configurations to test
CONFIGS = [
    {
        "name": "reps1_linear",
        "reps": 1,
        "entanglement": "linear"
    },
    {
        "name": "reps2_linear",
        "reps": 2,
        "entanglement": "linear"
    },
    {
        "name": "reps2_full",
        "reps": 2,
        "entanglement": "full"
    }
]


# ============================================================
# BLOCK-WISE QUANTUM KERNEL COMPUTATION
# ============================================================

def compute_kernel_in_blocks(
    quantum_kernel,
    X,
    Y,
    prefix,
    block_size=100
):
    """
    Compute a quantum kernel matrix in blocks.

    Each completed block is saved to disk so that an interrupted
    computation can continue without losing completed blocks.
    """

    blocks = []

    total_blocks = int(np.ceil(len(X) / block_size))

    print(f"\nTotal blocks: {total_blocks}")

    for block_number, start in enumerate(
        range(0, len(X), block_size),
        start=1
    ):

        end = min(start + block_size, len(X))

        cache_file = CACHE_DIR / (
            f"{prefix}_block_{block_number}.npy"
        )

        print(
            f"\nBlock {block_number}/{total_blocks} "
            f"-> rows {start}:{end}"
        )

        # ----------------------------------------------------
        # Load cached block if it already exists
        # ----------------------------------------------------

        if cache_file.exists():

            print("Loading cached block...")

            block = np.load(cache_file)

        else:

            print("Computing quantum kernel block...")

            block = quantum_kernel.evaluate(
                x_vec=X[start:end],
                y_vec=Y
            )

            np.save(cache_file, block)

            print("Block saved.")

        blocks.append(block)

    kernel_matrix = np.vstack(blocks)

    return kernel_matrix


# ============================================================
# METRIC CALCULATION
# ============================================================

def evaluate_model(
    y_true,
    predictions,
    decision_scores
):

    # Convert OCSVM output:
    #
    # +1 = normal
    # -1 = anomaly
    #
    # Convert to:
    #
    # 0 = normal
    # 1 = fraud

    y_pred = np.where(
        predictions == -1,
        1,
        0
    )

    # OCSVM decision score:
    #
    # higher = more normal
    # lower  = more anomalous
    #
    # Therefore negate it for anomaly ranking.

    anomaly_scores = -decision_scores

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_true,
        anomaly_scores
    )

    pr_auc = average_precision_score(
        y_true,
        anomaly_scores
    )

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": cm
    }


# ============================================================
# MAIN EXPERIMENT
# ============================================================

all_results = []

print("=" * 75)
print("QML FEATURE-MAP TUNING")
print("=" * 75)

print(f"\nTraining samples : {len(X_train)}")
print(f"Test samples     : {len(X_test)}")
print(f"Number of qubits : {X_train.shape[1]}")
print(f"OCSVM nu         : {NU}")

print("\nConfigurations:")
for config in CONFIGS:
    print(
        f"  {config['name']} -> "
        f"reps={config['reps']}, "
        f"entanglement={config['entanglement']}"
    )


# ============================================================
# LOOP THROUGH FEATURE MAPS
# ============================================================

for config in CONFIGS:

    name = config["name"]
    reps = config["reps"]
    entanglement = config["entanglement"]

    print("\n")
    print("=" * 75)
    print(f"TESTING FEATURE MAP: {name}")
    print("=" * 75)

    print(f"Repetitions : {reps}")
    print(f"Entanglement: {entanglement}")

    # --------------------------------------------------------
    # Special case:
    # Current baseline already has these kernel matrices.
    # --------------------------------------------------------

    if name == "reps2_linear":

        train_kernel_file = RESULTS_DIR / "K_train.npy"
        test_kernel_file = RESULTS_DIR / "K_test.npy"

        if train_kernel_file.exists() and test_kernel_file.exists():

            print("\nUsing existing baseline kernel matrices.")

            K_train = np.load(train_kernel_file)
            K_test = np.load(test_kernel_file)

        else:

            print("\nBaseline kernel files not found.")
            print("Computing them from scratch...")

            feature_map = zz_feature_map(
                feature_dimension=X_train.shape[1],
                reps=reps,
                entanglement=entanglement
            )

            quantum_kernel = FidelityQuantumKernel(
                feature_map=feature_map
            )

            K_train = compute_kernel_in_blocks(
                quantum_kernel,
                X_train,
                X_train,
                name + "_train",
                BLOCK_SIZE
            )

            K_test = compute_kernel_in_blocks(
                quantum_kernel,
                X_test,
                X_train,
                name + "_test",
                BLOCK_SIZE
            )

    else:

        # ----------------------------------------------------
        # Create feature map
        # ----------------------------------------------------

        print("\nCreating ZZ feature map...")

        feature_map = zz_feature_map(
            feature_dimension=X_train.shape[1],
            reps=reps,
            entanglement=entanglement
        )

        print("Feature map created.")

        print(f"Depth      : {feature_map.depth()}")
        print(f"Parameters : {feature_map.num_parameters}")

        # ----------------------------------------------------
        # Create quantum kernel
        # ----------------------------------------------------

        print("\nCreating Fidelity Quantum Kernel...")

        quantum_kernel = FidelityQuantumKernel(
            feature_map=feature_map
        )

        print("Quantum kernel created.")

        # ----------------------------------------------------
        # Training kernel
        # ----------------------------------------------------

        print("\nComputing TRAINING quantum kernel...")

        K_train = compute_kernel_in_blocks(
            quantum_kernel,
            X_train,
            X_train,
            name + "_train",
            BLOCK_SIZE
        )

        train_kernel_file = (
            RESULTS_DIR /
            f"{name}_K_train.npy"
        )

        np.save(
            train_kernel_file,
            K_train
        )

        print(
            f"Training kernel saved: "
            f"{train_kernel_file}"
        )

        # ----------------------------------------------------
        # Test kernel
        # ----------------------------------------------------

        print("\nComputing TEST quantum kernel...")

        K_test = compute_kernel_in_blocks(
            quantum_kernel,
            X_test,
            X_train,
            name + "_test",
            BLOCK_SIZE
        )

        test_kernel_file = (
            RESULTS_DIR /
            f"{name}_K_test.npy"
        )

        np.save(
            test_kernel_file,
            K_test
        )

        print(
            f"Test kernel saved: "
            f"{test_kernel_file}"
        )

    # ========================================================
    # TRAIN ONE-CLASS SVM
    # ========================================================

    print("\nTraining One-Class SVM...")

    model = OneClassSVM(
        kernel="precomputed",
        nu=NU
    )

    model.fit(K_train)

    print("Model trained.")

    # ========================================================
    # PREDICTIONS
    # ========================================================

    print("\nGenerating predictions...")

    predictions = model.predict(K_test)

    decision_scores = model.decision_function(
        K_test
    )

    # ========================================================
    # EVALUATION
    # ========================================================

    metrics = evaluate_model(
        y_test,
        predictions,
        decision_scores
    )

    cm = metrics["confusion_matrix"]

    print("\n" + "-" * 75)
    print(f"RESULTS: {name}")
    print("-" * 75)

    print(
        f"Accuracy  : {metrics['accuracy']:.4f}"
    )

    print(
        f"Precision : {metrics['precision']:.4f}"
    )

    print(
        f"Recall    : {metrics['recall']:.4f}"
    )

    print(
        f"F1 Score  : {metrics['f1']:.4f}"
    )

    print(
        f"ROC-AUC   : {metrics['roc_auc']:.4f}"
    )

    print(
        f"PR-AUC    : {metrics['pr_auc']:.4f}"
    )

    print("\nConfusion Matrix:")

    print(cm)

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    all_results.append({
        "feature_map": name,
        "reps": reps,
        "entanglement": entanglement,
        "nu": NU,
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "roc_auc": metrics["roc_auc"],
        "pr_auc": metrics["pr_auc"],
        "TN": cm[0, 0],
        "FP": cm[0, 1],
        "FN": cm[1, 0],
        "TP": cm[1, 1]
    })


# ============================================================
# FINAL COMPARISON TABLE
# ============================================================

results_df = pd.DataFrame(
    all_results
)

output_file = (
    RESULTS_DIR /
    "feature_map_tuning_results.csv"
)

results_df.to_csv(
    output_file,
    index=False
)

print("\n")
print("=" * 75)
print("FINAL FEATURE-MAP COMPARISON")
print("=" * 75)

print(
    results_df[
        [
            "feature_map",
            "reps",
            "entanglement",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "pr_auc"
        ]
    ].to_string(index=False)
)

print("\nResults saved to:")
print(output_file)

print("\nFeature-map tuning completed.")