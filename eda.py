import pandas as pd


# ============================================================
# QML ANOMALY DETECTION
# Exploratory Data Analysis
# ============================================================

DATA_PATH = "data/creditcard.csv"


# ------------------------------------------------------------
# 1. Load dataset
# ------------------------------------------------------------

df = pd.read_csv(DATA_PATH)

print("=" * 70)
print("DATASET OVERVIEW")
print("=" * 70)

print(f"Rows    : {df.shape[0]:,}")
print(f"Columns : {df.shape[1]}")


# ------------------------------------------------------------
# 2. Column information
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("COLUMNS")
print("=" * 70)

print(df.columns.tolist())


# ------------------------------------------------------------
# 3. Data types
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DATA TYPES")
print("=" * 70)

print(df.dtypes)


# ------------------------------------------------------------
# 4. Missing values
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)

missing = df.isnull().sum()

print(missing[missing > 0])

if missing.sum() == 0:
    print("No missing values found.")


# ------------------------------------------------------------
# 5. Duplicate rows
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DUPLICATES")
print("=" * 70)

duplicates = df.duplicated().sum()

print(f"Duplicate rows: {duplicates:,}")


# ------------------------------------------------------------
# 6. Class distribution
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("CLASS DISTRIBUTION")
print("=" * 70)

class_counts = df["Class"].value_counts()

print(class_counts)

normal_count = class_counts.get(0, 0)
fraud_count = class_counts.get(1, 0)

print(f"\nNormal transactions : {normal_count:,}")
print(f"Fraud transactions  : {fraud_count:,}")

print(
    f"Fraud percentage    : "
    f"{fraud_count / len(df) * 100:.4f}%"
)


# ------------------------------------------------------------
# 7. Amount statistics
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("AMOUNT STATISTICS")
print("=" * 70)

print(df["Amount"].describe())


# ------------------------------------------------------------
# 8. Time statistics
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("TIME STATISTICS")
print("=" * 70)

print(df["Time"].describe())


# ------------------------------------------------------------
# 9. Feature statistics
# ------------------------------------------------------------

features = [
    f"V{i}"
    for i in range(1, 29)
] + ["Amount"]

print("\n" + "=" * 70)
print("FEATURE STATISTICS")
print("=" * 70)

print(
    df[features]
    .describe()
    .T
)


# ------------------------------------------------------------
# 10. Amount by class
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("AMOUNT BY CLASS")
print("=" * 70)

print(
    df.groupby("Class")["Amount"]
    .agg(
        [
            "count",
            "mean",
            "median",
            "min",
            "max"
        ]
    )
)


# ------------------------------------------------------------
# 11. Time by class
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("TIME BY CLASS")
print("=" * 70)

print(
    df.groupby("Class")["Time"]
    .agg(
        [
            "count",
            "mean",
            "median",
            "min",
            "max"
        ]
    )
)


# ============================================================
# 12. DUPLICATE ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("DUPLICATE ANALYSIS")
print("=" * 70)

# Include every row belonging to a duplicate group
duplicates_df = df[
    df.duplicated(keep=False)
].copy()

print(
    f"Rows involved in duplicate groups: "
    f"{len(duplicates_df):,}"
)


# ------------------------------------------------------------
# Check duplicate groups for conflicting labels
# ------------------------------------------------------------

feature_columns = [
    col for col in df.columns
    if col != "Class"
]

duplicate_label_check = (
    duplicates_df
    .groupby(feature_columns)["Class"]
    .nunique()
)

conflicting_duplicates = (
    duplicate_label_check[
        duplicate_label_check > 1
    ]
)

print(
    f"Duplicate groups with conflicting labels: "
    f"{len(conflicting_duplicates)}"
)


# ------------------------------------------------------------
# Class distribution among duplicate rows
# ------------------------------------------------------------

print("\nClass distribution among duplicate rows:")

print(
    duplicates_df["Class"]
    .value_counts()
)


# ------------------------------------------------------------
# Number of duplicate groups
# ------------------------------------------------------------

print(
    "\nNumber of duplicate groups:",
    len(duplicate_label_check)
)


print("\nEDA completed successfully.")