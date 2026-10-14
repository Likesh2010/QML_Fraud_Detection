import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

RESULTS_DIR = "results"
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

os.makedirs(PLOTS_DIR, exist_ok=True)


# ============================================================
# 1. ν TUNING — ν VS F1 SCORE
# ============================================================

print("\n[1/6] Generating ν tuning graph...")

nu_file = os.path.join(
    RESULTS_DIR,
    "nu_tuning_results.csv"
)

nu_df = pd.read_csv(nu_file)

print("ν tuning results:")
print(nu_df)

plt.figure(figsize=(8, 5))

plt.plot(
    nu_df["nu"],
    nu_df["f1"],
    marker="o",
    linewidth=2
)

plt.xlabel("ν")
plt.ylabel("F1 Score")
plt.title("One-Class SVM ν Tuning")

plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(PLOTS_DIR, "01_nu_vs_f1.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 2. FEATURE MAP TUNING
# ============================================================

print("\n[2/6] Generating feature-map tuning graph...")

feature_file = os.path.join(
    RESULTS_DIR,
    "feature_map_tuning_results.csv"
)

feature_df = pd.read_csv(feature_file)

print("Feature-map tuning results:")
print(feature_df)

plt.figure(figsize=(8, 5))

plt.bar(
    feature_df["feature_map"],
    feature_df["f1"]
)

plt.xlabel("Feature Map Configuration")
plt.ylabel("F1 Score")
plt.title("Quantum Feature-Map Configuration Tuning")

plt.ylim(0, 0.75)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(PLOTS_DIR, "02_feature_map_vs_f1.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# HELPER FUNCTION FOR DEVELOPMENT METRICS
# ============================================================

def calculate_metrics(y_true, predictions, decision_scores):
    """
    Calculate the metrics needed for comparison.
    """

    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        average_precision_score
    )

    # OCSVM decision function:
    # higher = more normal
    # Therefore negate it so higher = more anomalous.
    anomaly_scores = -decision_scores

    metrics = {
        "Accuracy": accuracy_score(
            y_true,
            predictions
        ),

        "Precision": precision_score(
            y_true,
            predictions,
            zero_division=0
        ),

        "Recall": recall_score(
            y_true,
            predictions,
            zero_division=0
        ),

        "F1": f1_score(
            y_true,
            predictions,
            zero_division=0
        ),

        "ROC-AUC": roc_auc_score(
            y_true,
            anomaly_scores
        ),

        "PR-AUC": average_precision_score(
            y_true,
            anomaly_scores
        )
    }

    return metrics


# ============================================================
# 3. DEVELOPMENT QML VS CLASSICAL
# ============================================================

print("\n[3/6] Generating development comparison graph...")

y_test = np.load(
    os.path.join(
        RESULTS_DIR,
        "y_test.npy"
    )
)

qml_predictions = np.load(
    os.path.join(
        RESULTS_DIR,
        "final_qml_predictions.npy"
    )
)

qml_scores = np.load(
    os.path.join(
        RESULTS_DIR,
        "final_qml_decision_scores.npy"
    )

)

classical_predictions = np.load(
    os.path.join(
        RESULTS_DIR,
        "classical_predictions.npy"
    )
)

classical_scores = np.load(
    os.path.join(
        RESULTS_DIR,
        "classical_decision_scores.npy"
    )
)


qml_metrics = calculate_metrics(
    y_test,
    qml_predictions,
    qml_scores
)

classical_metrics = calculate_metrics(
    y_test,
    classical_predictions,
    classical_scores
)

metrics = list(qml_metrics.keys())

qml_values = [
    qml_metrics[m]
    for m in metrics
]

classical_values = [
    classical_metrics[m]
    for m in metrics
]


x = np.arange(len(metrics))
width = 0.35

plt.figure(figsize=(11, 6))

plt.bar(
    x - width / 2,
    qml_values,
    width,
    label="QML"
)

plt.bar(
    x + width / 2,
    classical_values,
    width,
    label="Classical RBF"
)

plt.xticks(
    x,
    metrics,
    rotation=20
)

plt.ylabel("Score")
plt.title(
    "QML vs Classical — Enriched Development Test"
)

plt.ylim(0, 1)

plt.legend()

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        PLOTS_DIR,
        "03_development_qml_vs_classical.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 4. REALISTIC PREVALENCE — DETECTION METRICS
# ============================================================

print("\n[4/6] Generating realistic detection metrics graph...")

realistic_file = os.path.join(
    RESULTS_DIR,
    "realistic_comparison.csv"
)

realistic_df = pd.read_csv(
    realistic_file
)

print("Realistic comparison:")
print(realistic_df)

# Detection metrics
detection_metrics = [
    "Accuracy",
    "Recall",
    "F1 Score"
]

detection_df = realistic_df[
    realistic_df["Metric"].isin(detection_metrics)
].copy()

x = np.arange(len(detection_metrics))
width = 0.35

plt.figure(figsize=(9, 5))

plt.bar(
    x - width / 2,
    detection_df["QML"],
    width,
    label="QML"
)

plt.bar(
    x + width / 2,
    detection_df["Classical"],
    width,
    label="Classical RBF"
)

plt.xticks(
    x,
    detection_metrics
)

plt.ylabel("Score")
plt.title(
    "Realistic-Prevalence Evaluation — Detection Metrics"
)

plt.ylim(0, 1)

plt.legend()

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        PLOTS_DIR,
        "04_realistic_detection_metrics.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 5. REALISTIC PREVALENCE — RANKING METRICS
# ============================================================

print("\n[5/6] Generating realistic ranking metrics graph...")

ranking_metrics = [
    "Precision",
    "ROC-AUC",
    "PR-AUC"
]

ranking_df = realistic_df[
    realistic_df["Metric"].isin(ranking_metrics)
].copy()

x = np.arange(len(ranking_metrics))
width = 0.35

plt.figure(figsize=(9, 5))

plt.bar(
    x - width / 2,
    ranking_df["QML"],
    width,
    label="QML"
)

plt.bar(
    x + width / 2,
    ranking_df["Classical"],
    width,
    label="Classical RBF"
)

plt.xticks(
    x,
    ranking_metrics
)

plt.ylabel("Score")
plt.title(
    "Realistic-Prevalence Evaluation — Ranking Metrics"
)

plt.ylim(0, 1)

plt.legend()

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        PLOTS_DIR,
        "05_realistic_ranking_metrics.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 6. REALISTIC CONFUSION MATRICES
# ============================================================

print("\n[6/6] Generating confusion matrices...")

qml_cm = np.load(
    os.path.join(
        RESULTS_DIR,
        "realistic_qml_confusion_matrix.npy"
    )
)

classical_cm = np.load(
    os.path.join(
        RESULTS_DIR,
        "realistic_classical_confusion_matrix.npy"
    )
)


fig, axes = plt.subplots(
    1,
    2,
    figsize=(11, 5)
)

confusion_matrices = [
    (qml_cm, "QML"),
    (classical_cm, "Classical RBF")
]


for ax, (cm, title) in zip(
    axes,
    confusion_matrices
):

    ax.imshow(cm)

    ax.set_title(title)

    ax.set_xlabel(
        "Predicted Label"
    )

    ax.set_ylabel(
        "Actual Label"
    )

    ax.set_xticks(
        [0, 1],
        ["Normal", "Fraud"]
    )

    ax.set_yticks(
        [0, 1],
        ["Normal", "Fraud"]
    )

    for i in range(2):
        for j in range(2):

            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                fontsize=13
            )


fig.suptitle(
    "Realistic-Prevalence Confusion Matrices"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        PLOTS_DIR,
        "06_realistic_confusion_matrices.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n" + "=" * 60)
print("GRAPH GENERATION COMPLETE")
print("=" * 60)

print(f"\nGraphs saved in:")
print(os.path.abspath(PLOTS_DIR))

print("\nGenerated files:")

for filename in sorted(
    os.listdir(PLOTS_DIR)
):

    print(
        f"  ✓ {filename}"
    )

print("\nNo QML model was retrained.")
print("All graphs were generated from existing experiment results.")