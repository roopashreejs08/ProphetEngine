
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from load_cmapss import load_cmapss_file, columns

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
RUL_CAP = 125
WINDOW_SIZE = 30

SENSOR_COLUMNS = [f"sensor_{i}" for i in range(1, 22)]


def add_rul_labels(df):
    """Calculate capped RUL labels for run-to-failure training engines."""
    df = df.copy()

    max_cycles = df.groupby("engine_id")["cycle"].transform("max")
    df["RUL"] = (max_cycles - df["cycle"]).clip(upper=RUL_CAP)

    return df


def split_by_engine(df, validation_size=0.2):
    """Split entire engines into training and validation sets."""
    engine_ids = np.sort(df["engine_id"].unique())

    train_ids, val_ids = train_test_split(
        engine_ids,
        test_size=validation_size,
        random_state=SEED,
        shuffle=True,
    )

    train_df = df[df["engine_id"].isin(train_ids)].copy()
    val_df = df[df["engine_id"].isin(val_ids)].copy()

    assert set(train_ids).isdisjoint(set(val_ids))

    return train_df, val_df


def find_low_variance_sensors(train_df, threshold=1e-8):
    """Identify sensors with almost no variation in training data."""
    variances = train_df[SENSOR_COLUMNS].var()
    low_variance = variances[variances <= threshold].index.tolist()

    return low_variance, variances.sort_values()


def make_windows(df, feature_columns, window=WINDOW_SIZE):
    """
    Build chronological windows independently for each engine.

    Returns:
        X: array with shape (number_of_windows, window, features)
        y: RUL at the final cycle of each window
    """
    X, y = [], []

    for _, engine_df in df.groupby("engine_id", sort=False):
        engine_df = engine_df.sort_values("cycle")

        features = engine_df[feature_columns].to_numpy(dtype=np.float32)
        targets = engine_df["RUL"].to_numpy(dtype=np.float32)

        for end in range(window - 1, len(engine_df)):
            X.append(features[end - window + 1:end + 1])
            y.append(targets[end])

    if not X:
        return (
            np.empty((0, window, len(feature_columns)), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )

    return np.stack(X), np.asarray(y, dtype=np.float32)


def process_dataset(dataset_name):
    """Prepare one NASA training dataset and save reproducible splits."""
    print(f"\nProcessing {dataset_name}")

    df = load_cmapss_file(f"train_{dataset_name}.txt")
    df = add_rul_labels(df)

    train_df, val_df = split_by_engine(df)

    # Detect sensors using training engines only.
    low_variance, variances = find_low_variance_sensors(train_df)

    print("\nSensor variances:")
    print(variances)
    print("\nLow-variance sensors:", low_variance)

    # Keep a consistent feature list for this dataset.
    feature_columns = [
        col for col in columns
        if col not in ["engine_id", "cycle"]
        and col not in low_variance
    ]

    train_df = train_df.drop(columns=low_variance)
    val_df = val_df.drop(columns=low_variance)

    # Save row-level datasets; retain engine_id and cycle for traceability.
    train_df.to_csv(
        PROCESSED_DIR / f"{dataset_name}_train.csv", index=False
    )
    val_df.to_csv(
        PROCESSED_DIR / f"{dataset_name}_validation.csv", index=False
    )

    X_train, y_train = make_windows(train_df, feature_columns)
    X_val, y_val = make_windows(val_df, feature_columns)

    np.savez_compressed(
        PROCESSED_DIR / f"{dataset_name}_train_windows.npz",
        X=X_train, y=y_train
    )
    np.savez_compressed(
        PROCESSED_DIR / f"{dataset_name}_validation_windows.npz",
        X=X_val, y=y_val
    )

    print(f"Training engines: {train_df['engine_id'].nunique()}")
    print(f"Validation engines: {val_df['engine_id'].nunique()}")
    print(f"Feature columns: {len(feature_columns)}")
    print(f"Train windows: {X_train.shape}, targets: {y_train.shape}")
    print(f"Validation windows: {X_val.shape}, targets: {y_val.shape}")

    return train_df, val_df


if __name__ == "__main__":
    for dataset in ["FD001", "FD003"]:
        process_dataset(dataset)

    print("\nPreprocessing complete.")
    print("Processed files saved in:", PROCESSED_DIR)
