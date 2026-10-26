
from pathlib import Path
import pandas as pd

# Project root: predictive-maintenance/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "cmapss"

# NASA C-MAPSS column names
columns = (
    ["engine_id", "cycle"]
    + [f"op_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)


def load_cmapss_file(filename):
    """Load a whitespace-separated NASA C-MAPSS file."""
    path = DATA_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Missing file: {path}\n"
            "Check that the extracted NASA files are in data/cmapss/."
        )

    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=columns
    )

    # Remove trailing empty columns if the source contains them.
    df = df.dropna(axis=1, how="all")

    return df


if __name__ == "__main__":
    for filename in [
        "train_FD001.txt",
        "test_FD001.txt",
        "RUL_FD001.txt",
        "train_FD003.txt",
        "test_FD003.txt",
        "RUL_FD003.txt",
    ]:
        if filename.startswith("RUL_"):
            path = DATA_DIR / filename

            if not path.exists():
                raise FileNotFoundError(f"Missing file: {path}")

            df = pd.read_csv(path, sep=r"\s+", header=None)
            print(f"\n{filename}: {df.shape[0]} reference values")
        else:
            df = load_cmapss_file(filename)
            print(f"\n{filename}: {df.shape}")
            print(df.head(3))
