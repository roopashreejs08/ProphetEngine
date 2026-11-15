"""Paths, column names and shared constants for the ProphetEngine dashboard.

Everything here mirrors what the training code in ``src/`` produced, so the
dashboard and the models always agree on feature names and order.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "processed"
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = REPO_ROOT / "report" / "figures"
LIVE_DIR = REPO_ROOT / "data" / "live"

METADATA_PATH = DATA_DIR / "ai4i_metadata.json"
TRAIN_PATH = DATA_DIR / "ai4i_train.csv"
VALIDATION_PATH = DATA_DIR / "ai4i_validation.csv"
CLEANED_PATH = DATA_DIR / "ai4i_cleaned.csv"
METRICS_PATH = RESULTS_DIR / "classical_ml_metrics.csv"
IMPORTANCE_PATH = RESULTS_DIR / "random_forest_feature_importance.csv"
STREAM_PATH = LIVE_DIR / "stream.csv"

REPO_URL = "https://github.com/roopashreejs08/ProphetEngine"

TARGET = "Machine failure"

# (column, short label, unit, slider step)
SENSORS = [
    ("Air temperature [K]", "Air temperature", "K", 0.1),
    ("Process temperature [K]", "Process temperature", "K", 0.1),
    ("Rotational speed [rpm]", "Rotational speed", "rpm", 1.0),
    ("Torque [Nm]", "Torque", "Nm", 0.1),
    ("Tool wear [min]", "Tool wear", "min", 1.0),
]
SENSOR_COLUMNS = [column for column, _, _, _ in SENSORS]
SENSOR_LABEL = {column: label for column, label, _, _ in SENSORS}
SENSOR_UNIT = {column: unit for column, _, unit, _ in SENSORS}
SENSOR_STEP = {column: step for column, _, _, step in SENSORS}

TYPE_COLUMNS = ["Type_H", "Type_L", "Type_M"]
FEATURE_COLUMNS = SENSOR_COLUMNS + TYPE_COLUMNS  # order used at training time
MACHINE_TYPES = ["L", "M", "H"]
MACHINE_TYPE_LABEL = {"L": "L - low quality", "M": "M - medium quality", "H": "H - high quality"}

FAILURE_MODES = {
    "TWF": "Tool wear failure",
    "HDF": "Heat dissipation failure",
    "PWF": "Power failure",
    "OSF": "Overstrain failure",
    "RNF": "Random failure",
}

# Model registry. The tuned entry reuses the Random Forest artifact with the
# threshold chosen by src/models/tune_random_forest.py (0.23).
TUNED_LABEL = "Random Forest (tuned)"
TUNED_THRESHOLD = 0.23
DEFAULT_THRESHOLD = 0.5
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "Random Forest": "random_forest.joblib",
    TUNED_LABEL: "random_forest.joblib",
}
MODEL_ORDER = list(MODEL_FILES)
BASE_MODELS = ["Logistic Regression", "Decision Tree", "Random Forest"]
DEFAULT_MODEL = TUNED_LABEL
CONFUSION_PNGS = {
    "Logistic Regression": "logistic_regression_confusion_matrix.png",
    "Decision Tree": "decision_tree_confusion_matrix.png",
    "Random Forest": "random_forest_confusion_matrix.png",
}


def default_threshold(model_label):
    return TUNED_THRESHOLD if model_label == TUNED_LABEL else DEFAULT_THRESHOLD


def artifact_file(model_label):
    return MODEL_FILES[model_label]


# Live telemetry feed format (see data/live/README.md).
STREAM_COLUMNS = [
    "timestamp",
    "air_temperature",
    "process_temperature",
    "rotational_speed",
    "torque",
    "tool_wear",
    "machine_type",
]
STREAM_SENSOR_MAP = {
    "air_temperature": "Air temperature [K]",
    "process_temperature": "Process temperature [K]",
    "rotational_speed": "Rotational speed [rpm]",
    "torque": "Torque [Nm]",
    "tool_wear": "Tool wear [min]",
}
STREAM_STALE_SECONDS = 30

# C-MAPSS (engine degradation) data prepared by the preprocessing pipeline.
CMAPSS_SUBSETS = ["FD001", "FD003"]
RUL_CAP = 125

# Colour system. Categorical slots were validated for colour-vision
# deficiency separation on light and dark surfaces; status colours are
# reserved for health states and always shipped with a text label.
MODEL_COLORS = {
    "light": {
        "Logistic Regression": "#2a78d6",
        "Decision Tree": "#eb6834",
        "Random Forest": "#1baf7a",
        TUNED_LABEL: "#6250d6",
    },
    "dark": {
        "Logistic Regression": "#3987e5",
        "Decision Tree": "#d95926",
        "Random Forest": "#199e70",
        TUNED_LABEL: "#9085e9",
    },
}
STATUS_COLORS = {
    "healthy": "#0ca30c",
    "watch": "#fab219",
    "critical": "#d03b3b",
}
ACCENT = {"light": "#2a78d6", "dark": "#3987e5"}
