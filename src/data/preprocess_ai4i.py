
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "ai4i" / "ai4i2020.csv"
OUTPUT_DIR = ROOT / "data" / "processed"

TARGET = "Machine failure"
ID_COLUMNS = ["UDI", "Product ID"]
FAILURE_MODE_COLUMNS = ["TWF", "HDF", "PWF", "OSF", "RNF"]
RANDOM_STATE = 42
VALIDATION_SIZE = 0.20


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    # Basic data-quality checks
    if df.isnull().values.any():
        raise ValueError("AI4I dataset contains missing values.")

    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' was not found.")

    if not set(FAILURE_MODE_COLUMNS).issubset(df.columns):
        raise ValueError("One or more failure-mode columns are missing.")

    print("Original dataset shape:", df.shape)
    print("\nTarget counts:")
    print(df[TARGET].value_counts().sort_index())

    # Keep a cleaned copy for EDA, excluding identifiers.
    # Failure-mode flags remain here for descriptive analysis only.
    cleaned = df.drop(columns=ID_COLUMNS).copy()

    # Encode machine Type using one-hot encoding.
    cleaned = pd.get_dummies(
        cleaned,
        columns=["Type"],
        prefix="Type",
        dtype=int
    )

    cleaned.to_csv(OUTPUT_DIR / "ai4i_cleaned.csv", index=False)

    # Build model-ready inputs without target-leaking failure-mode flags.
    feature_columns = [
        col for col in cleaned.columns
        if col != TARGET and col not in FAILURE_MODE_COLUMNS
    ]

    X = cleaned[feature_columns]
    y = cleaned[TARGET]

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # Save inputs and labels together for convenient downstream use.
    train = X_train.copy()
    train[TARGET] = y_train

    validation = X_val.copy()
    validation[TARGET] = y_val

    train.to_csv(OUTPUT_DIR / "ai4i_train.csv", index=False)
    validation.to_csv(
        OUTPUT_DIR / "ai4i_validation.csv",
        index=False
    )

    # Save feature names and split metadata for reproducibility.
    metadata = {
        "dataset": "AI4I 2020 Predictive Maintenance Dataset",
        "random_state": RANDOM_STATE,
        "validation_size": VALIDATION_SIZE,
        "target": TARGET,
        "features": feature_columns,
        "excluded_identifiers": ID_COLUMNS,
        "excluded_failure_mode_columns": FAILURE_MODE_COLUMNS,
        "train_rows": len(train),
        "validation_rows": len(validation),
    }

    import json

    with open(OUTPUT_DIR / "ai4i_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\nPreprocessing complete.")
    print("Cleaned dataset:", cleaned.shape)
    print("Training shape:", train.shape)
    print("Validation shape:", validation.shape)
    print("\nTraining target distribution:")
    print(y_train.value_counts().sort_index())
    print("\nValidation target distribution:")
    print(y_val.value_counts().sort_index())
    print("\nFeatures used:")
    print(feature_columns)
    print("\nSaved files to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
