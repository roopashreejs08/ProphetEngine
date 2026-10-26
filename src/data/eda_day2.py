
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "report" / "figures"

FIGURES.mkdir(parents=True, exist_ok=True)


def save_correlation_heatmap():
    df = pd.read_csv(PROCESSED / "ai4i_cleaned.csv")

    # Descriptive EDA only. Failure-mode flags are not model features.
    numeric = df.select_dtypes(include="number")
    corr = numeric.corr()

    fig, ax = plt.subplots(figsize=(13, 10))
    image = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=8)
    ax.set_yticklabels(corr.columns, fontsize=8)
    ax.set_title("AI4I 2020 — Correlation Heatmap")
    fig.colorbar(image, ax=ax, label="Pearson correlation")
    fig.tight_layout()
    fig.savefig(FIGURES / "ai4i_correlation_heatmap.png", dpi=200)
    plt.close(fig)


def save_failure_class_chart():
    df = pd.read_csv(PROCESSED / "ai4i_cleaned.csv")
    counts = df["Machine failure"].value_counts().reindex(
        [0, 1], fill_value=0
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(
        ["No failure (0)", "Failure (1)"],
        counts.values
    )
    ax.set_title("AI4I 2020 — Machine Failure Class Distribution")
    ax.set_ylabel("Number of observations")
    ax.bar_label(bars, fmt="%d")
    fig.tight_layout()
    fig.savefig(FIGURES / "ai4i_class_imbalance.png", dpi=200)
    plt.close(fig)


def save_rul_distribution(dataset):
    path = PROCESSED / f"{dataset}_train.csv"
    df = pd.read_csv(path)

    if "RUL" not in df.columns:
        raise ValueError(
            f"'RUL' column not found in {path.name}. "
            f"Available columns: {df.columns.tolist()}"
        )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df["RUL"].dropna(), bins=30, edgecolor="black")
    ax.set_title(f"{dataset} — Training RUL Distribution")
    ax.set_xlabel("Remaining Useful Life (cycles)")
    ax.set_ylabel("Number of observations")
    fig.tight_layout()
    fig.savefig(FIGURES / f"{dataset}_RUL_distribution.png", dpi=200)
    plt.close(fig)


def main():
    save_correlation_heatmap()
    save_failure_class_chart()
    save_rul_distribution("FD001")
    save_rul_distribution("FD003")

    print("EDA figures saved to:", FIGURES)
    for path in sorted(FIGURES.glob("*.png")):
        print(f"{path.name}: {path.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
