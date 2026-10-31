"""
ProphetEngine predictive maintenance console.

Single-file Streamlit app. Every value shown comes from repository artifacts or
live inference over repository data:
- data/processed/ai4i_validation.csv (live scoring fleet)
- data/processed/ai4i_train.csv (simulator statistics and input ranges)
- data/live/stream.csv (optional external telemetry feed)
- results/classical_ml_metrics.csv
- results/random_forest_feature_importance.csv
- results/*.joblib
- results/*_confusion_matrix.png
- data/processed/ai4i_metadata.json

Run with: streamlit run app.py
"""

import json
import math
import time
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data" / "processed"
RESULTS_DIR = REPO_ROOT / "results"

METADATA_PATH = DATA_DIR / "ai4i_metadata.json"
TRAIN_PATH = DATA_DIR / "ai4i_train.csv"
VALIDATION_PATH = DATA_DIR / "ai4i_validation.csv"
METRICS_PATH = RESULTS_DIR / "classical_ml_metrics.csv"
IMPORTANCE_PATH = RESULTS_DIR / "random_forest_feature_importance.csv"
LIVE_DIR = REPO_ROOT / "data" / "live"
STREAM_PATH = LIVE_DIR / "stream.csv"

PREPROCESS_COMMAND = "python src/data/preprocess_ai4i.py"
TRAIN_COMMAND = "python src/models/classical_ml.py"
TUNE_COMMAND = "python src/models/tune_random_forest.py"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SENSOR_COLUMNS = [
    ("Air temperature [K]", "Air temperature (K)", "%.1f", 0.1),
    ("Process temperature [K]", "Process temperature (K)", "%.1f", 0.1),
    ("Rotational speed [rpm]", "Rotational speed (rpm)", "%.0f", 1.0),
    ("Torque [Nm]", "Torque (Nm)", "%.1f", 0.1),
    ("Tool wear [min]", "Tool wear (min)", "%.0f", 1.0),
]

FEATURE_COLUMNS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Type_H",
    "Type_L",
    "Type_M",
]

TARGET = "Machine failure"
TUNED_LABEL = "Random Forest (tuned)"
TUNED_THRESHOLD = 0.23
DEFAULT_THRESHOLD = 0.5

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "Random Forest": "random_forest.joblib",
    TUNED_LABEL: "random_forest.joblib",
}

MODEL_ORDER = [
    "Logistic Regression",
    "Decision Tree",
    "Random Forest",
    TUNED_LABEL,
]

CM_FILES = {
    "Logistic Regression": "logistic_regression_confusion_matrix.png",
    "Decision Tree": "decision_tree_confusion_matrix.png",
    "Random Forest": "random_forest_confusion_matrix.png",
}

MACHINE_TYPES = ["L", "M", "H"]

SENSOR_SHORT = {
    "Air temperature [K]": "Air temperature",
    "Process temperature [K]": "Process temperature",
    "Rotational speed [rpm]": "Rotational speed",
    "Torque [Nm]": "Torque",
    "Tool wear [min]": "Tool wear",
}

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
LIVE_HISTORY_LIMIT = 3600

LIVE_SOURCES = ["Simulated engine", "External feed", "Manual"]

PAGES = [
    "Engine health and analytics",
    "Live prediction",
    "Model Comparison",
    "Feature Importance",
    "Confusion matrices",
    "About",
]

# Palette
BASE = "#0F1419"
CARD = "#171D25"
BORDER = "#263040"
TEXT = "#E6EAF0"
MUTED = "#8B97A7"
AMBER = "#F5A623"
GREEN = "#3FB950"
RED = "#F85149"
NEUTRAL = "#4A5568"
CHART_FONT = "Inter, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif"

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    layout="wide",
    page_title="ProphetEngine",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Single stylesheet
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    :root {
        --base: #0F1419;
        --card: #171D25;
        --border: #263040;
        --text: #E6EAF0;
        --muted: #8B97A7;
        --amber: #F5A623;
        --green: #3FB950;
        --red: #F85149;
    }

    html, body, [class*="css"], .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }

    .stApp {
        background: var(--base);
        color: var(--text);
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stAppDeployButton"] {
        display: none;
    }

    [data-testid="stMainMenu"] {
        display: none;
    }

    footer {
        visibility: hidden;
    }

    [data-testid="stSidebar"] {
        background: var(--card);
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 24px;
    }

    [data-testid="stSidebar"] [role="radiogroup"] label {
        padding: 8px 10px;
        border-radius: 8px;
        transition: background-color 150ms ease;
    }

    [data-testid="stSidebar"] [role="radiogroup"] label:hover {
        background: #1F2733;
    }

    .block-container {
        padding-top: 24px;
        padding-bottom: 48px;
        max-width: 1280px;
    }

    *:focus-visible {
        outline: 2px solid var(--amber) !important;
        outline-offset: 2px !important;
    }

    h1, h2, h3 {
        color: var(--text);
        letter-spacing: -0.01em;
    }

    h2 {
        font-size: 22px;
        font-weight: 600;
        margin: 32px 0 16px 0;
    }

    h3 {
        font-size: 16px;
        font-weight: 600;
        margin: 24px 0 12px 0;
    }

    p, li {
        color: var(--text);
        font-size: 15px;
        line-height: 1.6;
    }

    [data-testid="stMarkdownContainer"] p code,
    [data-testid="stMarkdownContainer"] li code {
        color: var(--text) !important;
        background: var(--card) !important;
        border: 1px solid var(--border);
        border-radius: 4px;
        padding: 1px 5px;
        font-size: 0.9em;
    }

    [data-testid="stCaptionContainer"] p {
        color: var(--muted) !important;
        font-size: 12px !important;
    }

    /* Header band */
    .pe-header {
        background: linear-gradient(180deg, #171D25 0%, #0F1419 100%);
        border: 1px solid var(--border);
        border-bottom: 2px solid var(--amber);
        border-radius: 12px;
        padding: 24px 28px;
        margin-bottom: 24px;
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        justify-content: space-between;
        gap: 16px;
    }

    .pe-brand {
        font-size: 26px;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: var(--text);
    }

    .pe-desc {
        color: var(--muted);
        font-size: 14px;
        margin-top: 4px;
    }

    .pe-chips {
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
    }

    .pe-chip {
        font-size: 12px;
        font-weight: 500;
        color: var(--muted);
        background: var(--base);
        border: 1px solid var(--border);
        border-radius: 999px;
        padding: 6px 12px;
        font-variant-numeric: tabular-nums;
    }

    /* Metric cards */
    .pe-card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 16px 18px;
        min-height: 150px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        transition: border-color 150ms ease;
    }

    .pe-card:hover {
        border-color: #3A4657;
    }

    .pe-card-label {
        color: var(--muted);
        font-size: 12px;
        font-weight: 500;
        letter-spacing: 0.02em;
    }

    .pe-card-value {
        color: var(--text);
        font-size: 30px;
        font-weight: 700;
        margin: 8px 0 4px 0;
        font-variant-numeric: tabular-nums;
        line-height: 1.15;
    }

    .pe-card-caption {
        color: var(--muted);
        font-size: 12px;
        line-height: 1.45;
    }

    /* Status banner */
    .pe-banner {
        border-radius: 12px;
        padding: 16px 18px;
        border: 1px solid var(--border);
        background: var(--card);
        margin-top: 16px;
    }

    .pe-banner-safe {
        border-left: 4px solid var(--green);
    }

    .pe-banner-risk {
        border-left: 4px solid var(--red);
    }

    .pe-banner-title {
        font-size: 18px;
        font-weight: 700;
        margin-bottom: 4px;
    }

    .pe-banner-safe .pe-banner-title {
        color: var(--green);
    }

    .pe-banner-risk .pe-banner-title {
        color: var(--red);
    }

    .pe-banner-body {
        color: var(--muted);
        font-size: 13px;
        line-height: 1.5;
        font-variant-numeric: tabular-nums;
    }

    /* Readout tiles */
    .pe-tile-row {
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
        margin-top: 16px;
    }

    .pe-tile {
        flex: 1 1 120px;
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 10px 12px;
    }

    .pe-tile-label {
        color: var(--muted);
        font-size: 11px;
        letter-spacing: 0.02em;
    }

    .pe-tile-value {
        color: var(--text);
        font-size: 15px;
        font-weight: 600;
        margin-top: 2px;
        font-variant-numeric: tabular-nums;
    }

    .pe-tile-sub {
        color: var(--muted);
        font-size: 10px;
        line-height: 1.5;
        margin-top: 3px;
        font-variant-numeric: tabular-nums;
    }

    /* Tables */
    .pe-table-wrap {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 4px 16px;
        overflow-x: auto;
    }

    .pe-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 13px;
        font-variant-numeric: tabular-nums;
    }

    .pe-table th {
        text-align: right;
        color: var(--muted);
        font-weight: 500;
        font-size: 12px;
        border-bottom: 1px solid var(--border);
        padding: 10px 12px;
        white-space: nowrap;
    }

    .pe-table th:first-child,
    .pe-table td:first-child {
        text-align: left;
    }

    .pe-table td {
        text-align: right;
        padding: 10px 12px;
        border-bottom: 1px solid var(--border);
        color: var(--text);
        white-space: nowrap;
    }

    .pe-table tr:last-child td {
        border-bottom: none;
    }

    .pe-muted-cell {
        color: var(--muted) !important;
    }

    .pe-sensor-cell {
        font-size: 12px !important;
        color: var(--muted) !important;
    }

    .pe-row-selected td {
        background: rgba(245, 166, 35, 0.08);
    }

    .pe-verdict-risk {
        color: var(--red) !important;
        font-weight: 600;
    }

    .pe-verdict-safe {
        color: var(--green) !important;
        font-weight: 600;
    }

    .pe-event-raised {
        color: var(--red) !important;
        font-weight: 600;
    }

    .pe-event-cleared {
        color: var(--green) !important;
        font-weight: 600;
    }

    /* Live confusion matrix */
    .pe-cm th,
    .pe-cm td {
        text-align: center;
        padding: 14px 16px;
    }

    .pe-cm thead th:first-child {
        text-align: left;
    }

    .pe-cm tbody th {
        color: var(--muted);
        font-weight: 500;
        text-align: left;
        border-bottom: 1px solid var(--border);
        padding: 14px 16px;
    }

    .pe-cm-correct {
        color: var(--green) !important;
        font-weight: 700;
    }

    .pe-cm-error {
        color: var(--red) !important;
        font-weight: 700;
    }

    /* Empty states */
    .pe-empty {
        background: var(--card);
        border: 1px dashed var(--border);
        border-radius: 12px;
        padding: 24px;
        margin: 12px 0;
    }

    .pe-empty-title {
        font-size: 17px;
        font-weight: 600;
        margin-bottom: 8px;
        color: var(--text);
    }

    .pe-empty-body {
        color: var(--muted);
        font-size: 13px;
        line-height: 1.6;
        margin-bottom: 12px;
    }

    .pe-empty pre {
        background: var(--base);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 12px 16px;
        color: var(--text);
        font-size: 13px;
        overflow-x: auto;
        margin: 0;
    }

    .pe-empty-compact {
        padding: 16px 18px;
    }

    /* Cards for images and text blocks */
    [data-testid="stImage"] {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 12px;
        display: flex;
        justify-content: center;
    }

    [data-testid="stImage"] img {
        max-width: 760px !important;
        width: auto !important;
        height: auto !important;
        border-radius: 6px;
    }

    /* Sidebar */
    .pe-side-brand {
        font-size: 20px;
        font-weight: 700;
        color: var(--text);
    }

    .pe-side-sub {
        color: var(--muted);
        font-size: 12px;
        margin-top: 2px;
        margin-bottom: 16px;
    }

    .pe-side-foot {
        color: var(--muted);
        font-size: 12px;
        line-height: 1.7;
    }

    .pe-side-foot a {
        color: var(--amber);
        text-decoration: none;
    }

    .pe-side-foot a:hover {
        text-decoration: underline;
    }

    /* Buttons */
    .stButton > button {
        background: var(--amber);
        color: #0F1419;
        border: 1px solid var(--amber);
        border-radius: 8px;
        font-weight: 600;
        padding: 10px 16px;
        transition: background-color 150ms ease, border-color 150ms ease;
    }

    .stButton > button:hover {
        background: #E0940F;
        border-color: #E0940F;
        color: #0F1419;
    }

    .stButton > button:disabled {
        background: #263040;
        border-color: #263040;
        color: var(--muted);
    }

    /* Inputs */
    [data-testid="stNumberInput"] input,
    [data-baseweb="select"] > div {
        background: var(--card) !important;
        border-color: var(--border) !important;
        color: var(--text) !important;
        font-variant-numeric: tabular-nums;
    }

    [data-testid="stSlider"] {
        font-variant-numeric: tabular-nums;
    }

    /* Reduced motion */
    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
            transition: none !important;
            animation: none !important;
            scroll-behavior: auto !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_data
def load_metadata():
    if not METADATA_PATH.exists():
        return None
    with open(METADATA_PATH, encoding="utf-8") as handle:
        return json.load(handle)


@st.cache_data
def load_metrics():
    if not METRICS_PATH.exists():
        return None
    return pd.read_csv(METRICS_PATH)


@st.cache_data
def load_importance():
    if not IMPORTANCE_PATH.exists():
        return None
    frame = pd.read_csv(IMPORTANCE_PATH, header=None, names=["Feature", "Importance"])
    frame = frame[frame["Feature"].astype(str).str.strip() != ""]
    frame["Importance"] = pd.to_numeric(frame["Importance"], errors="coerce")
    frame = frame.dropna(subset=["Importance"]).reset_index(drop=True)
    return frame


@st.cache_data
def load_train_frame():
    """Training split, used for input bounds, percentiles and simulator stats."""
    if not TRAIN_PATH.exists():
        return None
    try:
        train = pd.read_csv(TRAIN_PATH)
    except Exception:
        return None
    required = FEATURE_COLUMNS + [TARGET]
    if train.empty or not set(required).issubset(train.columns):
        return None
    if not all(pd.api.types.is_numeric_dtype(train[column]) for column in required):
        return None
    return train


@st.cache_data
def load_validation():
    """Validation fleet used for every live score on the first page."""
    if not VALIDATION_PATH.exists():
        return None
    try:
        validation = pd.read_csv(VALIDATION_PATH)
    except Exception:
        return None
    required = FEATURE_COLUMNS + [TARGET]
    if validation.empty or not set(required).issubset(validation.columns):
        return None
    if not all(
        pd.api.types.is_numeric_dtype(validation[column]) for column in required
    ):
        return None
    return validation


@st.cache_data
def load_input_ranges():
    """Derive simulator and manual-input bounds from the training split."""
    train = load_train_frame()
    if train is None:
        return None
    ranges = {}
    for column, _, _, _ in SENSOR_COLUMNS:
        series = pd.to_numeric(train[column], errors="coerce").dropna()
        if series.empty:
            return None
        ranges[column] = {
            "min": float(series.min()),
            "max": float(series.max()),
            "default": float(series.median()),
        }
    return ranges


@st.cache_data
def load_train_stats():
    """Statistics that parameterise the simulator, all estimated from training data."""
    train = load_train_frame()
    if train is None:
        return None

    sensors = {}
    for column, _, _, _ in SENSOR_COLUMNS:
        series = pd.to_numeric(train[column], errors="coerce").dropna()
        if series.empty:
            return None
        std = float(series.std(ddof=0))
        sensors[column] = {
            "min": float(series.min()),
            "max": float(series.max()),
            "median": float(series.median()),
            "mean": float(series.mean()),
            "std": std if std > 0 else float(series.max() - series.min()) / 4.0,
            "q25": float(series.quantile(0.25)),
            "q75": float(series.quantile(0.75)),
        }

    rpm = train["Rotational speed [rpm]"].to_numpy(dtype=float)
    torque = train["Torque [Nm]"].to_numpy(dtype=float)
    slope, intercept = np.polyfit(rpm, torque, 1)
    residuals = torque - (intercept + slope * rpm)

    offset = (
        train["Process temperature [K]"] - train["Air temperature [K]"]
    ).to_numpy(dtype=float)

    failure_wear = train.loc[
        train[TARGET] == 1, "Tool wear [min]"
    ].to_numpy(dtype=float)
    typical_wear_life = float(np.median(failure_wear)) if len(failure_wear) else 0.0
    if typical_wear_life <= 0:
        typical_wear_life = sensors["Tool wear [min]"]["max"]
    wear_range = (
        sensors["Tool wear [min]"]["max"] - sensors["Tool wear [min]"]["min"]
    )

    type_stats = {}
    for machine_type in MACHINE_TYPES:
        column = f"Type_{machine_type}"
        subset = train[train[column] == 1]
        if subset.empty:
            type_stats[machine_type] = {
                "count": 0,
                "rpm_median": sensors["Rotational speed [rpm]"]["median"],
                "torque_median": sensors["Torque [Nm]"]["median"],
                "air_median": sensors["Air temperature [K]"]["median"],
                "wear_median": sensors["Tool wear [min]"]["median"],
            }
        else:
            type_stats[machine_type] = {
                "count": int(len(subset)),
                "rpm_median": float(subset["Rotational speed [rpm]"].median()),
                "torque_median": float(subset["Torque [Nm]"].median()),
                "air_median": float(subset["Air temperature [K]"].median()),
                "wear_median": float(subset["Tool wear [min]"].median()),
            }

    return {
        "rows": int(len(train)),
        "sensors": sensors,
        "air_process_offset_mean": float(np.mean(offset)),
        "air_process_offset_std": float(np.std(offset, ddof=0)),
        "torque_rpm_slope": float(slope),
        "torque_rpm_intercept": float(intercept),
        "torque_residual_std": float(np.std(residuals, ddof=0)),
        "rpm_torque_correlation": float(np.corrcoef(rpm, torque)[0, 1]),
        "type_stats": type_stats,
        "typical_wear_life": float(typical_wear_life),
        "wear_increment": float(wear_range / max(1.0, typical_wear_life)),
        "failure_rate": float(train[TARGET].mean()),
    }


@st.cache_data
def cached_validation_probabilities(model_label):
    """Cache validation probabilities per model. Threshold is deliberately absent."""
    validation = load_validation()
    if validation is None:
        return {"probabilities": None, "error": "validation_missing"}
    filename = MODEL_FILES.get(model_label)
    if filename is None or not (RESULTS_DIR / filename).exists():
        return {"probabilities": None, "error": "model_missing"}
    try:
        model = load_model(filename)
        probabilities = model.predict_proba(validation[FEATURE_COLUMNS])[:, 1]
    except Exception:
        return {"probabilities": None, "error": "model_failed"}
    return {
        "probabilities": np.asarray(probabilities, dtype=float),
        "error": None,
    }


@st.cache_resource
def load_model(filename):
    return joblib.load(RESULTS_DIR / filename)


# ---------------------------------------------------------------------------
# Formatting and shared UI helpers
# ---------------------------------------------------------------------------
def render_header():
    st.markdown(
        """
        <div class="pe-header">
            <div>
                <div class="pe-brand">ProphetEngine</div>
                <div class="pe-desc">Predictive maintenance for industrial engines.</div>
            </div>
            <div class="pe-chips">
                <span class="pe-chip">AI4I 2020</span>
                <span class="pe-chip">Classical ML</span>
                <span class="pe-chip">Seed 42</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label, value, caption):
    st.markdown(
        f"""
        <div class="pe-card">
            <div class="pe-card-label">{label}</div>
            <div class="pe-card-value">{value}</div>
            <div class="pe-card-caption">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def empty_state(title, body, commands=None):
    commands_html = ""
    if commands:
        joined = "\n".join(commands)
        commands_html = f"<pre>{joined}</pre>"
    st.markdown(
        f"""
        <div class="pe-empty">
            <div class="pe-empty-title">{title}</div>
            <div class="pe-empty-body">{body}</div>
            {commands_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def last_loaded(paths):
    joined = ", ".join(f"`{path}`" for path in paths)
    st.caption(f"Last loaded: {joined}")


def format_sensor(column, value):
    units = {
        "Air temperature [K]": (",.1f", "K"),
        "Process temperature [K]": (",.1f", "K"),
        "Rotational speed [rpm]": (",.0f", "rpm"),
        "Torque [Nm]": (",.1f", "Nm"),
        "Tool wear [min]": (",.0f", "min"),
    }
    number_format, unit = units.get(column, (",.2f", ""))
    return f"{value:{number_format}} {unit}".strip()


def format_percent(value, digits=2):
    """Round percentages half up so 90.625% matches the reported 90.63%."""
    quantum = Decimal(1).scaleb(-digits)
    rounded = (Decimal(str(value)) * 100).quantize(
        quantum, rounding=ROUND_HALF_UP
    )
    return f"{rounded:.{digits}f}%"


def format_metric(column, value):
    if pd.isna(value):
        return "not reported"
    if column in ("Accuracy", "Precision", "Recall", "F1-score"):
        return format_percent(value)
    return f"{value:.4f}"


def build_metrics_with_tuned(metrics):
    """Saved rows plus the tuned row recomputed live from validation data."""
    tuned_result = cached_validation_probabilities(TUNED_LABEL)
    tuned_values = {
        "Accuracy": np.nan,
        "Precision": np.nan,
        "Recall": np.nan,
        "F1-score": np.nan,
        "Average Precision (PR-AUC)": np.nan,
        "ROC-AUC": np.nan,
    }
    if tuned_result["error"] is None:
        validation = load_validation()
        if validation is not None:
            probabilities = tuned_result["probabilities"]
            y_true = validation[TARGET].to_numpy(dtype=int)
            y_pred = (probabilities >= TUNED_THRESHOLD).astype(int)
            tuned_values.update(
                {
                    "Accuracy": float(accuracy_score(y_true, y_pred)),
                    "Precision": float(
                        precision_score(y_true, y_pred, zero_division=0)
                    ),
                    "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
                    "F1-score": float(f1_score(y_true, y_pred, zero_division=0)),
                    "Average Precision (PR-AUC)": float(
                        average_precision_score(y_true, probabilities)
                    ),
                }
            )
    tuned = pd.DataFrame([{"Model": TUNED_LABEL, **tuned_values}])
    combined = pd.concat([metrics, tuned], ignore_index=True)
    combined["Model"] = pd.Categorical(
        combined["Model"], categories=MODEL_ORDER, ordered=True
    )
    return combined.sort_values("Model").reset_index(drop=True)


def render_metrics_table(metrics):
    frame = build_metrics_with_tuned(metrics)
    headers = ["Model", "Accuracy", "Precision", "Recall", "F1-score", "PR-AUC", "ROC-AUC"]
    keys = [
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score",
        "Average Precision (PR-AUC)",
        "ROC-AUC",
    ]
    head_html = "".join(f"<th>{header}</th>" for header in headers)
    rows_html = []
    for _, row in frame.iterrows():
        cells = [f"<td>{row['Model']}</td>"]
        for key in keys[1:]:
            value = row[key]
            if pd.isna(value):
                cells.append('<td class="pe-muted-cell">not reported</td>')
            else:
                cells.append(f"<td>{format_metric(key, value)}</td>")
        rows_html.append("<tr>" + "".join(cells) + "</tr>")
    st.markdown(
        f"""
        <div class="pe-table-wrap">
            <table class="pe-table">
                <thead><tr>{head_html}</tr></thead>
                <tbody>{''.join(rows_html)}</tbody>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_importance_table(frame):
    frame = frame.sort_values("Importance", ascending=False).reset_index(drop=True)
    rows = []
    for _, row in frame.iterrows():
        rows.append(
            f"<tr><td>{row['Feature']}</td><td>{format_percent(row['Importance'])}</td></tr>"
        )
    st.markdown(
        f"""
        <div class="pe-table-wrap">
            <table class="pe-table">
                <thead><tr><th>Feature</th><th>Importance</th></tr></thead>
                <tbody>{''.join(rows)}</tbody>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )


def base_layout(title, x_title=None):
    layout = {
        "title": {
            "text": title,
            "font": {"size": 15, "color": TEXT},
            "x": 0,
            "xanchor": "left",
        },
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"family": CHART_FONT, "size": 13, "color": TEXT},
        "margin": {"l": 8, "r": 48, "t": 48, "b": 40},
        "hoverlabel": {
            "bgcolor": CARD,
            "bordercolor": BORDER,
            "font": {"color": TEXT, "family": CHART_FONT, "size": 12},
        },
        "showlegend": False,
        "height": 320,
    }
    if x_title:
        layout["xaxis"] = {
            "title": {"text": x_title, "font": {"size": 12, "color": MUTED}},
            "gridcolor": BORDER,
            "zeroline": False,
            "tickfont": {"color": MUTED, "size": 11},
        }
    return layout


def bar_figure(pairs, title, x_title, percent=True, x_max=None):
    """Horizontal bars sorted ascending so the largest value sits on top."""
    ordered = sorted(pairs, key=lambda item: item[1])
    labels = [item[0] for item in ordered]
    values = [item[1] for item in ordered]

    colors = [NEUTRAL] * len(ordered)
    if colors:
        colors[-1] = AMBER

    if percent:
        text = [format_percent(value) for value in values]
        custom = [format_percent(value) for value in values]
        tickformat = ".0%"
        range_max = x_max if x_max is not None else min(1.15, max(values) * 1.35)
    else:
        text = [f"{value:.4f}" for value in values]
        custom = [f"{value:.4f}" for value in values]
        tickformat = ".2f"
        range_max = x_max if x_max is not None else max(values) * 1.25

    figure = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker={"color": colors},
            text=text,
            textposition="outside",
            textfont={"color": MUTED, "size": 11},
            customdata=custom,
            hovertemplate="%{y}: %{customdata}<extra></extra>",
            cliponaxis=False,
        )
    )
    figure.update_layout(**base_layout(title, x_title))
    figure.update_xaxes(tickformat=tickformat, range=[0, range_max])
    figure.update_yaxes(tickfont={"color": TEXT, "size": 12})
    return figure


def probability_figure(probability, threshold):
    figure = go.Figure(
        go.Bar(
            x=[probability],
            y=["Failure probability"],
            orientation="h",
            marker={"color": AMBER},
            width=0.42,
            text=[format_percent(probability)],
            textposition="outside",
            textfont={"color": TEXT, "size": 13},
            hovertemplate="Failure probability: %{x:.2%}<extra></extra>",
            cliponaxis=False,
        )
    )
    figure.update_layout(**base_layout("Failure probability", "Probability"))
    figure.update_layout(height=220)
    figure.update_xaxes(range=[0, 1], tickformat=".0%")
    figure.update_yaxes(showticklabels=False)
    figure.add_vline(
        x=threshold,
        line={"color": TEXT, "width": 2, "dash": "dot"},
        annotation_text=f"threshold {threshold:.2f}",
        annotation_position="top",
        annotation_font={"color": MUTED, "size": 11},
    )
    return figure


def build_input_frame(values, machine_type):
    row = {column: values[column] for column, _, _, _ in SENSOR_COLUMNS}
    row["Type_H"] = 1 if machine_type == "H" else 0
    row["Type_L"] = 1 if machine_type == "L" else 0
    row["Type_M"] = 1 if machine_type == "M" else 0
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def model_path(model_label):
    return RESULTS_DIR / MODEL_FILES[model_label]


def init_shared_scoring_state():
    """Shared model and threshold, kept in session state across pages."""
    if st.session_state.get("pe_model") not in MODEL_ORDER:
        st.session_state["pe_model"] = "Random Forest"
    if "pe_threshold" not in st.session_state:
        st.session_state["pe_threshold"] = (
            TUNED_THRESHOLD
            if st.session_state["pe_model"] == TUNED_LABEL
            else DEFAULT_THRESHOLD
        )


def on_model_change():
    selected = st.session_state.get("pe_model", "Random Forest")
    st.session_state["pe_threshold"] = (
        TUNED_THRESHOLD if selected == TUNED_LABEL else DEFAULT_THRESHOLD
    )


def machine_type_series(frame):
    """Recover L/M/H from one-hot columns without inventing a type."""
    result = pd.Series("M", index=frame.index, dtype=object)
    result.loc[frame["Type_L"] == 1] = "L"
    result.loc[frame["Type_H"] == 1] = "H"
    return result


def live_performance(y_true, probabilities, threshold):
    y_pred = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "actual_failures": int(np.sum(y_true == 1)),
    }
    return metrics


def failure_sentence(metrics, threshold):
    failures = metrics["actual_failures"]
    false_alarms = metrics["fp"]
    failure_word = "failure" if failures == 1 else "failures"
    alarm_word = "false alarm" if false_alarms == 1 else "false alarms"
    return (
        f"At threshold {threshold:.2f} the model catches {metrics['tp']} of "
        f"{failures} actual {failure_word} and raises {false_alarms} {alarm_word}."
    )


def confusion_matrix_html(tn, fp, fn, tp):
    return f"""
    <div class="pe-table-wrap pe-cm-wrap">
        <table class="pe-table pe-cm">
            <thead>
                <tr>
                    <th>Live counts</th>
                    <th>Predicted healthy</th>
                    <th>Predicted failure</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <th>Actual healthy</th>
                    <td>{tn:,}</td>
                    <td class="pe-cm-error">{fp:,}</td>
                </tr>
                <tr>
                    <th>Actual failure</th>
                    <td class="pe-cm-error">{fn:,}</td>
                    <td class="pe-cm-correct">{tp:,}</td>
                </tr>
            </tbody>
        </table>
    </div>
    """


def gauge_figure(probability, threshold):
    alerted = probability >= threshold
    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=float(probability),
            number={
                "valueformat": ".3f",
                "font": {"color": TEXT, "size": 36, "family": CHART_FONT},
            },
            gauge={
                "axis": {
                    "range": [0, 1],
                    "tickformat": ".0%",
                    "tickcolor": MUTED,
                    "tickfont": {"color": MUTED, "size": 10},
                },
                "bar": {"color": RED if alerted else GREEN, "thickness": 0.32},
                "bgcolor": CARD,
                "borderwidth": 0,
                "steps": [
                    {"range": [0, threshold], "color": "rgba(63,185,80,0.12)"},
                    {"range": [threshold, 1], "color": "rgba(248,81,73,0.12)"},
                ],
                "threshold": {
                    "line": {"color": AMBER, "width": 3},
                    "thickness": 0.82,
                    "value": float(threshold),
                },
            },
            title={
                "text": "Failure probability gauge",
                "font": {"size": 14, "color": MUTED, "family": CHART_FONT},
            },
        )
    )
    figure.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": CHART_FONT, "color": TEXT},
        height=250,
        margin={"l": 24, "r": 24, "t": 56, "b": 16},
    )
    return figure


def probability_history_figure(history, model_label, threshold):
    points = [
        (reading["timestamp"], reading["probabilities"][model_label])
        for reading in history
        if reading["probabilities"].get(model_label) is not None
    ][-60:]
    figure = go.Figure()
    if points:
        times = [item[0] for item in points]
        values = [item[1] for item in points]
        figure.add_trace(
            go.Scatter(
                x=times,
                y=values,
                mode="lines+markers",
                line={"color": AMBER, "width": 2},
                marker={"size": 4, "color": AMBER},
                hovertemplate="%{x|%H:%M:%S} UTC<br>probability %{y:.3f}<extra></extra>",
                name="Failure probability",
            )
        )
    figure.add_hline(
        y=threshold,
        line={"color": RED, "width": 1.5, "dash": "dot"},
        annotation_text=f"threshold {threshold:.2f}",
        annotation_position="top left",
        annotation_font={"color": MUTED, "size": 11},
    )
    figure.update_layout(
        **base_layout(f"{model_label} probability, last 60 ticks")
    )
    figure.update_layout(
        height=300,
        showlegend=False,
        xaxis={
            "gridcolor": "rgba(0,0,0,0)",
            "tickfont": {"color": MUTED, "size": 10},
            "tickformat": "%H:%M:%S",
        },
        yaxis={
            "range": [0, 1],
            "tickformat": ".0%",
            "gridcolor": BORDER,
            "zeroline": False,
            "tickfont": {"color": MUTED, "size": 10},
        },
    )
    return figure


def sensor_sparkline(history, column, label, stats):
    points = [
        (reading["timestamp"], reading["sensors"][column])
        for reading in history
        if column in reading["sensors"]
    ][-60:]
    figure = go.Figure()
    if points:
        figure.add_trace(
            go.Scatter(
                x=[item[0] for item in points],
                y=[item[1] for item in points],
                mode="lines",
                line={"color": AMBER, "width": 2},
                hovertemplate="%{x|%H:%M:%S} UTC<br>%{y}<extra></extra>",
            )
        )
    figure.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": CHART_FONT, "size": 10, "color": MUTED},
        height=130,
        margin={"l": 8, "r": 8, "t": 30, "b": 8},
        title={
            "text": label,
            "font": {"size": 11, "color": MUTED},
            "x": 0,
            "xanchor": "left",
        },
        showlegend=False,
        xaxis={"visible": False},
        yaxis={
            "gridcolor": BORDER,
            "zeroline": False,
            "showticklabels": False,
        },
    )
    return figure


def render_tile_row(items):
    """Render a row of metric tiles using native Streamlit columns."""
    cols = st.columns(len(items), gap="small")
    for col, (label, value, sub_lines) in zip(cols, items):
        with col:
            sub_text = "  \n".join(sub_lines) if sub_lines else ""
            st.markdown(
                f"""
                <div class="pe-tile">
                    <div class="pe-tile-label">{label}</div>
                    <div class="pe-tile-value">{value}</div>
                    {f'<div class="pe-tile-sub">{sub_text}</div>' if sub_text else ""}
                </div>
                """,
                unsafe_allow_html=True,
            )


def format_delta(column, current, previous):
    if previous is None:
        return "First reading"
    delta = float(current) - float(previous)
    units = {
        "Air temperature [K]": ("K", 1),
        "Process temperature [K]": ("K", 1),
        "Rotational speed [rpm]": ("rpm", 0),
        "Torque [Nm]": ("Nm", 1),
        "Tool wear [min]": ("min", 0),
    }
    unit, digits = units.get(column, ("", 2))
    if abs(delta) < (0.05 if digits == 1 else 0.5):
        return "No change vs previous"
    sign = "+" if delta > 0 else "-"
    return f"{sign}{abs(delta):,.{digits}f} {unit} vs previous".strip()


def training_percentile(train, column, value):
    if train is None or column not in train:
        return None
    series = pd.to_numeric(train[column], errors="coerce").dropna()
    if series.empty:
        return None
    return float((series <= value).mean() * 100.0)


def percentile_label(percentile):
    if percentile is None:
        return "Training percentile unavailable"
    return f"P{percentile:.0f} vs training readings"


def predict_all_models(feature_frame):
    """Run live inference for every label. Returns probabilities and real latency."""
    probabilities = {}
    errors = {}
    start = time.perf_counter()
    for label in MODEL_ORDER:
        path = model_path(label)
        if not path.exists():
            probabilities[label] = None
            errors[label] = "model file missing"
            continue
        try:
            model = load_model(MODEL_FILES[label])
            probabilities[label] = float(
                model.predict_proba(feature_frame)[0, 1]
            )
        except Exception:
            probabilities[label] = None
            errors[label] = "model could not be scored"
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return probabilities, elapsed_ms, errors


def score_reading(sensors, machine_type, source, timestamp=None):
    feature_frame = build_input_frame(sensors, machine_type)
    probabilities, elapsed_ms, errors = predict_all_models(feature_frame)
    return {
        "timestamp": timestamp or datetime.now(timezone.utc),
        "sensors": dict(sensors),
        "machine_type": machine_type,
        "probabilities": probabilities,
        "inference_ms": float(elapsed_ms),
        "model_errors": errors,
        "source": source,
    }


def append_history(key, reading):
    history = st.session_state.setdefault(key, [])
    history.append(reading)
    if len(history) > LIVE_HISTORY_LIMIT:
        history = history[-LIVE_HISTORY_LIMIT:]
    st.session_state[key] = history


def derive_alert_events(history, model_label, threshold):
    events = []
    previously_flagged = False
    for reading in history:
        probability = reading["probabilities"].get(model_label)
        if probability is None:
            continue
        flagged = probability >= threshold
        if flagged != previously_flagged:
            events.append(
                {
                    "timestamp": reading["timestamp"],
                    "action": "raised" if flagged else "cleared",
                    "probability": probability,
                    "sensors": reading["sensors"],
                    "machine_type": reading["machine_type"],
                }
            )
        previously_flagged = flagged
    return events


def human_age(seconds):
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.0f} s"
    if seconds < 3600:
        return f"{seconds / 60.0:.1f} min"
    return f"{seconds / 3600.0:.1f} h"


def live_narrative(history, model_label, stats):
    points = [
        reading
        for reading in history
        if reading["probabilities"].get(model_label) is not None
    ][-10:]
    if not points:
        return (
            "No probability is available yet for the selected model. "
            "Check the model artifacts before reading the stream."
        )
    latest = points[-1]
    probability = latest["probabilities"][model_label]
    if len(points) == 1:
        sensor_column = "Torque [Nm]"
        return (
            f"First reading: failure probability is {probability:.2f} and "
            f"{SENSOR_SHORT[sensor_column].lower()} is "
            f"{format_sensor(sensor_column, latest['sensors'][sensor_column])}."
        )

    first = points[0]
    window = len(points)
    change = probability - first["probabilities"][model_label]

    sensor_changes = []
    for column, _, _, _ in SENSOR_COLUMNS:
        series = stats["sensors"][column]
        scale = series["std"] if series["std"] > 0 else 1.0
        delta = latest["sensors"][column] - first["sensors"][column]
        sensor_changes.append((abs(delta) / scale, delta, column))
    _, sensor_delta, sensor_column = max(sensor_changes, key=lambda item: item[0])

    if abs(sensor_delta) > 1e-9:
        sensor_verb = "climbed" if sensor_delta > 0 else "dropped"
        sensor_phrase = (
            f"{SENSOR_SHORT[sensor_column].lower()} {sensor_verb} to "
            f"{format_sensor(sensor_column, latest['sensors'][sensor_column])}"
        )
    else:
        sensor_phrase = (
            f"{SENSOR_SHORT[sensor_column].lower()} is "
            f"{format_sensor(sensor_column, latest['sensors'][sensor_column])}"
        )

    if change > 0.005:
        trend = f"rose from {first['probabilities'][model_label]:.2f} to {probability:.2f}"
    elif change < -0.005:
        trend = f"fell from {first['probabilities'][model_label]:.2f} to {probability:.2f}"
    else:
        trend = f"held near {probability:.2f}"

    return (
        f"Failure probability {trend} over the last {window} readings "
        f"as {sensor_phrase}."
    )


def render_live_model_table(history, model_label, threshold):
    latest = history[-1] if history else None
    rows = []
    for label in MODEL_ORDER:
        probability = (
            latest["probabilities"].get(label) if latest is not None else None
        )
        if probability is None:
            cells = (
                f"<td>{label}</td>"
                '<td class="pe-muted-cell">not available</td>'
                '<td class="pe-muted-cell">not scored</td>'
            )
            rows.append(f"<tr>{cells}</tr>")
            continue
        verdict = "Failure risk" if probability >= threshold else "Healthy"
        verdict_class = "pe-verdict-risk" if verdict == "Failure risk" else "pe-verdict-safe"
        selected_class = "pe-row-selected" if label == model_label else ""
        rows.append(
            f'<tr class="{selected_class}">'
            f"<td>{label}</td>"
            f"<td>{format_percent(probability, 2)}</td>"
            f'<td class="{verdict_class}">{verdict}</td>'
            "</tr>"
        )
    st.markdown(
        f"""
        <div class="pe-table-wrap">
            <table class="pe-table">
                <thead>
                    <tr><th>Model</th><th>Probability</th><th>Verdict</th></tr>
                </thead>
                <tbody>{''.join(rows)}</tbody>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_alert_log(events):
    if not events:
        st.markdown(
            '<div class="pe-empty pe-empty-compact">'
            '<div class="pe-empty-title">No alerts at the current threshold</div>'
            '<div class="pe-empty-body">The selected model has not crossed the '
            "threshold in the retained stream.</div></div>",
            unsafe_allow_html=True,
        )
        return
    rows = []
    for event in reversed(events[-20:]):
        action = event["action"]
        action_class = "pe-event-raised" if action == "raised" else "pe-event-cleared"
        sensors = " | ".join(
            format_sensor(column, event["sensors"][column])
            for column, _, _, _ in SENSOR_COLUMNS
        )
        rows.append(
            "<tr>"
            f"<td>{event['timestamp'].strftime('%H:%M:%S')} UTC</td>"
            f'<td class="{action_class}">{action}</td>'
            f"<td>{format_percent(event['probability'], 2)}</td>"
            f"<td>{event['machine_type']}</td>"
            f"<td class=\"pe-sensor-cell\">{sensors}</td>"
            "</tr>"
        )
    st.markdown(
        f"""
        <div class="pe-table-wrap">
            <table class="pe-table">
                <thead>
                    <tr><th>Time</th><th>Event</th><th>Probability</th>
                    <th>Type</th><th>Sensors</th></tr>
                </thead>
                <tbody>{''.join(rows)}</tbody>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_live_panel(
    history,
    model_label,
    threshold,
    stats,
    train_frame,
    total_ticks,
):
    if not history:
        st.markdown(
            '<div class="pe-empty"><div class="pe-empty-title">Waiting for telemetry</div>'
            '<div class="pe-empty-body">Start the simulator, append a valid row to the '
            "external feed, or move a manual control to create the first reading.</div></div>",
            unsafe_allow_html=True,
        )
        return

    latest = history[-1]
    probability = latest["probabilities"].get(model_label)
    events = derive_alert_events(history, model_label, threshold)
    raised_events = [event for event in events if event["action"] == "raised"]
    if raised_events:
        last_alert_age = (
            datetime.now(timezone.utc) - raised_events[-1]["timestamp"]
        ).total_seconds()
        last_alert_text = human_age(last_alert_age)
    else:
        last_alert_text = "No alerts yet"
    inference_values = [
        reading["inference_ms"] for reading in history if reading["inference_ms"] is not None
    ]
    average_inference = (
        float(np.mean(inference_values)) if inference_values else None
    )

    status_col, gauge_col = st.columns([1.2, 1], gap="large")
    with status_col:
        if probability is None:
            st.markdown(
                '<div class="pe-banner pe-banner-risk">'
                '<div class="pe-banner-title">Model unavailable</div>'
                '<div class="pe-banner-body">The selected model has no probability '
                "for this reading. Generate the model artifacts and reload.</div></div>",
                unsafe_allow_html=True,
            )
        else:
            flagged = probability >= threshold
            banner_class = "pe-banner-risk" if flagged else "pe-banner-safe"
            title = "Failure risk" if flagged else "Healthy"
            st.markdown(
                f"""
                <div class="pe-banner {banner_class}">
                    <div class="pe-banner-title">{title}</div>
                    <div class="pe-banner-body">
                        Failure probability {probability:.3f} with threshold
                        {threshold:.2f} for {model_label}.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown(
            f"**{live_narrative(history, model_label, stats)}**"
        )

    with gauge_col:
        if probability is None:
            st.markdown(
                '<div class="pe-empty pe-empty-compact">'
                '<div class="pe-empty-title">No gauge reading</div>'
                '<div class="pe-empty-body">The selected model is not available.</div>'
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.plotly_chart(
                gauge_figure(probability, threshold),
                width="stretch",
                config={"displayModeBar": False},
                key="live_gauge",
            )

    render_tile_row(
        [
            ("Ticks processed", f"{total_ticks:,}", ["Current stream"]),
            ("Alerts raised", f"{len(raised_events):,}", ["At current threshold"]),
            ("Since last alert", last_alert_text, ["Raised alerts only"]),
            (
                "Average inference",
                f"{average_inference:.2f} ms" if average_inference is not None else "n/a",
                ["All four models per tick"],
            ),
        ]
    )

    chart_col, table_col = st.columns([1.45, 1], gap="large")
    with chart_col:
        st.markdown("### Rolling failure probability")
        st.plotly_chart(
            probability_history_figure(history, model_label, threshold),
            width="stretch",
            config={"displayModeBar": False},
            key="live_probability_history",
        )
    with table_col:
        st.markdown("### Model probabilities")
        render_live_model_table(history, model_label, threshold)
        st.caption(
            "The shared threshold applies to all four rows; the tuned label defaults "
            "to 0.23. Values are live predict_proba outputs for this reading."
        )

    st.markdown("### Sensor trends")
    spark_columns = st.columns(5)
    for target, (column, label, _, _) in zip(spark_columns, SENSOR_COLUMNS):
        with target:
            st.plotly_chart(
                sensor_sparkline(history, column, label, stats),
                width="stretch",
                config={"displayModeBar": False},
                key=f"spark_{column}",
            )

    st.markdown("### Current readings")
    previous = history[-2] if len(history) > 1 else None
    items = []
    for column, _, _, _ in SENSOR_COLUMNS:
        value = latest["sensors"][column]
        delta_text = format_delta(
            column, value, previous["sensors"][column] if previous else None
        )
        percentile = training_percentile(train_frame, column, value)
        items.append(
            (
                SENSOR_SHORT[column],
                format_sensor(column, value),
                [delta_text, percentile_label(percentile)],
            )
        )
    render_tile_row(items)
    st.caption(
        "Percentiles compare the current reading with every row of "
        "data/processed/ai4i_train.csv."
    )

    st.markdown("### Alert log, last 20 threshold crossings")
    render_alert_log(events)


def empty_stream_state(title, body, stale_note=None):
    header = ",".join(STREAM_COLUMNS)
    lines = [
        "mkdir -p data/live",
        f"printf '{header}\\n' > data/live/stream.csv",
    ]
    if stale_note:
        body = f"{body} {stale_note}"
    empty_state(title, body, lines)


def parse_stream_csv(path):
    if not path.exists():
        return {
            "status": "missing",
            "frame": None,
            "warnings": [],
            "newest": None,
        }
    try:
        raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    except pd.errors.EmptyDataError:
        return {
            "status": "empty",
            "frame": None,
            "warnings": [],
            "newest": None,
        }
    except Exception:
        return {
            "status": "unreadable",
            "frame": None,
            "warnings": [],
            "newest": None,
        }

    raw.columns = [str(column).strip() for column in raw.columns]
    if len(set(raw.columns)) != len(raw.columns):
        return {
            "status": "invalid_columns",
            "frame": None,
            "warnings": ["Duplicate column names are not allowed."],
            "newest": None,
        }
    missing = [column for column in STREAM_COLUMNS if column not in raw.columns]
    extra = [column for column in raw.columns if column not in STREAM_COLUMNS]
    if missing or extra:
        warnings = []
        if missing:
            warnings.append("Missing columns: " + ", ".join(missing))
        if extra:
            warnings.append("Unexpected columns: " + ", ".join(extra))
        return {
            "status": "invalid_columns",
            "frame": None,
            "warnings": warnings,
            "newest": None,
        }
    if raw.empty:
        return {
            "status": "empty",
            "frame": None,
            "warnings": [],
            "newest": None,
        }

    timestamps = pd.to_datetime(raw["timestamp"], errors="coerce", utc=True)
    numeric = {}
    for column in STREAM_SENSOR_MAP:
        numeric[column] = pd.to_numeric(raw[column], errors="coerce")
    machine_type = raw["machine_type"].astype(str).str.strip().str.upper()

    valid = timestamps.notna()
    for column in STREAM_SENSOR_MAP:
        valid = valid & np.isfinite(numeric[column].to_numpy(dtype=float))
    valid = valid & machine_type.isin(MACHINE_TYPES)

    warnings = []
    invalid_count = int((~valid).sum())
    if invalid_count:
        bad_rows = ", ".join(
            str(index + 2) for index in raw.index[~valid][:8]
        )
        warnings.append(
            f"Skipped {invalid_count} malformed row(s). CSV row numbers: {bad_rows}"
            + (" ..." if invalid_count > 8 else "")
        )

    if not valid.any():
        return {
            "status": "no_valid_rows",
            "frame": None,
            "warnings": warnings,
            "newest": None,
        }

    frame = pd.DataFrame(
        {
            "timestamp": timestamps[valid],
            "machine_type": machine_type[valid],
        }
    )
    for column, target_column in STREAM_SENSOR_MAP.items():
        frame[target_column] = numeric[column][valid].astype(float)
    frame = frame.reset_index(drop=True)
    return {
        "status": "ok",
        "frame": frame,
        "warnings": warnings,
        "newest": frame["timestamp"].max(),
    }


def new_sim_state(stats, start_wear, machine_type="M", running=True):
    sensors = stats["sensors"]
    type_stat = stats["type_stats"][machine_type]
    wear = float(
        np.clip(
            start_wear,
            sensors["Tool wear [min]"]["min"],
            sensors["Tool wear [min]"]["max"],
        )
    )
    air = float(type_stat["air_median"])
    process = float(
        np.clip(
            air + stats["air_process_offset_mean"],
            sensors["Process temperature [K]"]["min"],
            sensors["Process temperature [K]"]["max"],
        )
    )
    rpm = float(
        np.clip(
            type_stat["rpm_median"],
            sensors["Rotational speed [rpm]"]["min"],
            sensors["Rotational speed [rpm]"]["max"],
        )
    )
    torque = float(
        np.clip(
            stats["torque_rpm_intercept"] + stats["torque_rpm_slope"] * rpm,
            sensors["Torque [Nm]"]["min"],
            sensors["Torque [Nm]"]["max"],
        )
    )
    return {
        "running": bool(running),
        "speed": 1.0,
        "machine_type": machine_type,
        "start_wear": float(start_wear),
        "wear": wear,
        "air": air,
        "process": process,
        "rpm": rpm,
        "torque": torque,
        "faults": [],
        "fault_air": 0.0,
        "fault_process": 0.0,
        "fault_torque": 0.0,
        "fault_rpm": 0.0,
        "rng": np.random.default_rng(42),
        "last_tick_wall": 0.0,
        "tick": 0,
    }


def apply_sim_fault(sim, fault, stats):
    sensors = stats["sensors"]
    if fault in sim["faults"]:
        return
    torque_max = sensors["Torque [Nm]"]["max"]
    rpm_step = sensors["Rotational speed [rpm]"]["std"]
    wear_max = sensors["Tool wear [min]"]["max"]

    if fault == "overload":
        sim["fault_air"] += 1.0 * sensors["Air temperature [K]"]["std"]
        sim["fault_process"] += 1.0 * sensors["Process temperature [K]"]["std"]
        sim["fault_torque"] = torque_max
        sim["fault_rpm"] -= 1.0 * rpm_step
        sim["wear"] = wear_max
    elif fault == "overheat":
        sim["fault_air"] += 2.5 * sensors["Air temperature [K]"]["std"]
        sim["fault_process"] += 2.5 * sensors["Process temperature [K]"]["std"]
        sim["fault_torque"] = torque_max
        sim["fault_rpm"] -= 1.0 * rpm_step
        sim["wear"] = wear_max
    elif fault == "accelerate_wear":
        sim["fault_air"] += 1.0 * sensors["Air temperature [K]"]["std"]
        sim["fault_process"] += 1.0 * sensors["Process temperature [K]"]["std"]
        sim["wear"] = wear_max
        sim["fault_torque"] = torque_max
        sim["fault_rpm"] -= 1.0 * rpm_step
    sim["faults"].append(fault)


def clear_sim_faults(sim):
    sim["faults"] = []
    sim["fault_air"] = 0.0
    sim["fault_process"] = 0.0
    sim["fault_torque"] = 0.0
    sim["fault_rpm"] = 0.0


def simulate_next_reading(sim, stats):
    sensors = stats["sensors"]
    rng = sim["rng"]
    speed = float(sim["speed"])
    machine_type = sim["machine_type"]
    type_stat = stats["type_stats"][machine_type]

    air_stats = sensors["Air temperature [K]"]
    air = sim["air"] + rng.normal(
        0.0, air_stats["std"] * 0.03 * speed
    ) + 0.02 * (air_stats["median"] - sim["air"])
    sim["air"] = float(
        np.clip(air, air_stats["min"], air_stats["max"])
    )

    offset = stats["air_process_offset_mean"] + rng.normal(
        0.0, stats["air_process_offset_std"] * 0.3
    )
    process_stats = sensors["Process temperature [K]"]
    sim["process"] = float(
        np.clip(
            sim["air"] + offset, process_stats["min"], process_stats["max"]
        )
    )

    rpm_stats = sensors["Rotational speed [rpm]"]
    rpm = sim["rpm"] + rng.normal(
        0.0, rpm_stats["std"] * 0.01 * speed
    ) + 0.05 * (type_stat["rpm_median"] - sim["rpm"])
    sim["rpm"] = float(np.clip(rpm, rpm_stats["min"], rpm_stats["max"]))

    torque_stats = sensors["Torque [Nm]"]
    torque = (
        stats["torque_rpm_intercept"]
        + stats["torque_rpm_slope"] * sim["rpm"]
        + rng.normal(0.0, stats["torque_residual_std"] * 0.2 * speed)
    )
    sim["torque"] = float(
        np.clip(torque, torque_stats["min"], torque_stats["max"])
    )

    wear_stats = sensors["Tool wear [min]"]
    sim["wear"] = float(
        min(
            wear_stats["max"],
            sim["wear"] + stats["wear_increment"] * speed,
        )
    )
    sim["tick"] += 1

    values = {
        "Air temperature [K]": float(
            np.clip(
                sim["air"] + sim["fault_air"],
                air_stats["min"],
                air_stats["max"],
            )
        ),
        "Process temperature [K]": float(
            np.clip(
                sim["process"] + sim["fault_process"],
                process_stats["min"],
                process_stats["max"],
            )
        ),
        "Rotational speed [rpm]": float(
            np.clip(
                sim["rpm"] + sim["fault_rpm"],
                rpm_stats["min"],
                rpm_stats["max"],
            )
        ),
        "Torque [Nm]": float(
            np.clip(
                sim["torque"] + sim["fault_torque"],
                torque_stats["min"],
                torque_stats["max"],
            )
        ),
        "Tool wear [min]": float(
            np.clip(sim["wear"], wear_stats["min"], wear_stats["max"])
        ),
    }
    return values, machine_type


# ---------------------------------------------------------------------------
# Shared data
# ---------------------------------------------------------------------------
metadata = load_metadata()
metrics = load_metrics()
importance = load_importance()
input_ranges = load_input_ranges()
train_frame = load_train_frame()
train_stats = load_train_stats()
validation_frame = load_validation()

init_shared_scoring_state()

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="pe-side-brand">ProphetEngine</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="pe-side-sub">Predictive maintenance console</div>',
        unsafe_allow_html=True,
    )
    page = st.radio("Navigation", PAGES, label_visibility="collapsed")

    st.markdown("---")
    st.markdown(
        '<div class="pe-side-sub" style="margin-bottom:8px;">Scoring controls</div>',
        unsafe_allow_html=True,
    )
    st.selectbox(
        "Model",
        MODEL_ORDER,
        key="pe_model",
        on_change=on_model_change,
        help=(
            "The tuned label reuses random_forest.joblib with its 0.23 "
            "default threshold. Used by Engine health and Live prediction."
        ),
    )
    st.slider(
        "Failure threshold",
        min_value=0.05,
        max_value=0.95,
        step=0.01,
        key="pe_threshold",
        help=(
            "A reading is flagged when its failure probability is at or "
            "above this threshold. Used by Engine health and Live prediction."
        ),
    )
    st.caption(
        f"Active: {st.session_state['pe_model']} at "
        f"{st.session_state['pe_threshold']:.2f}."
    )

    st.markdown("---")
    st.markdown(
        """
        <div class="pe-side-foot">
            <div>roopashreejs08/ProphetEngine</div>
            <div><a href="https://github.com/roopashreejs08/ProphetEngine" target="_blank" rel="noopener">View repository on GitHub</a></div>
            <div>AI4I 2020 Predictive Maintenance Dataset</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ===========================================================================
# Engine health and analytics
# ===========================================================================
if page == "Engine health and analytics":
    render_header()
    st.markdown("## Engine health and analytics")

    if validation_frame is not None:
        fleet_label = (
            "Validation fleet (historical records, "
            f"{len(validation_frame):,} machines)"
        )
    else:
        fleet_label = "Validation fleet (historical records)"
    st.markdown(f"### {fleet_label}")
    st.caption(
        "Every chart and number below is computed live by scoring "
        "data/processed/ai4i_validation.csv with the selected model and threshold."
    )

    model_label = st.session_state["pe_model"]
    threshold = float(st.session_state["pe_threshold"])

    if validation_frame is None:
        empty_state(
            "Validation fleet not found",
            "data/processed/ai4i_validation.csv is missing or unreadable, so the "
            "fleet cannot be scored. Regenerate the processed splits, then reload "
            "this page.",
            [PREPROCESS_COMMAND],
        )
    else:
        result = cached_validation_probabilities(model_label)
        if result["error"] == "model_missing":
            empty_state(
                "Trained model not found",
                f"results/{MODEL_FILES[model_label]} is missing, so the validation "
                "fleet cannot be scored. Generate the model artifacts, then reload.",
                [TRAIN_COMMAND, TUNE_COMMAND],
            )
        elif result["error"] is not None:
            empty_state(
                "Model could not score the fleet",
                f"results/{MODEL_FILES[model_label]} could not be loaded or used "
                "for inference. Regenerate the model artifacts, then reload.",
                [TRAIN_COMMAND, TUNE_COMMAND],
            )
        else:
            probabilities = result["probabilities"]
            y_true = validation_frame[TARGET].to_numpy(dtype=int)
            metrics = live_performance(y_true, probabilities, threshold)
            predicted_flagged = probabilities >= threshold
            flagged_count = int(predicted_flagged.sum())
            healthy_fraction = float((~predicted_flagged).mean())
            fleet_health = float(np.mean(1.0 - probabilities))

            card_columns = st.columns(5)
            with card_columns[0]:
                metric_card(
                    "Machines scored",
                    f"{len(validation_frame):,}",
                    "Validation records",
                )
            with card_columns[1]:
                metric_card(
                    "Flagged at risk",
                    f"{flagged_count:,}",
                    f"Probability at or above {threshold:.2f}",
                )
            with card_columns[2]:
                metric_card(
                    "Healthy machines",
                    format_percent(healthy_fraction),
                    f"{len(validation_frame) - flagged_count:,} not flagged",
                )
            with card_columns[3]:
                metric_card(
                    "Fleet health score",
                    format_percent(fleet_health),
                    "Mean of 1 - failure probability",
                )
            with card_columns[4]:
                metric_card(
                    "Actual failures",
                    f"{metrics['actual_failures']:,}",
                    "Historical labels",
                )

            st.markdown("### Live model performance")
            st.caption(
                f"{model_label} at threshold {threshold:.2f}. Metrics are recomputed "
                "from the cached validation probabilities on every threshold change."
            )
            performance_columns = st.columns(5)
            performance_cards = [
                ("Accuracy", format_percent(metrics["accuracy"]), "All scored records"),
                ("Precision", format_percent(metrics["precision"]), "Flagged records that failed"),
                ("Recall", format_percent(metrics["recall"]), "Actual failures caught"),
                ("F1-score", format_percent(metrics["f1"]), "Precision and recall balance"),
                (
                    "PR-AUC",
                    f"{metrics['pr_auc']:.4f}",
                    "Ranking metric, threshold-independent",
                ),
            ]
            for target, (label, value, caption) in zip(
                performance_columns, performance_cards
            ):
                with target:
                    metric_card(label, value, caption)

            matrix_col, sentence_col = st.columns([1, 1.1], gap="large")
            with matrix_col:
                st.markdown("#### Confusion matrix from live counts")
                st.markdown(
                    confusion_matrix_html(
                        metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"]
                    ),
                    unsafe_allow_html=True,
                )
                st.caption(
                    "Rows are actual outcomes and columns are threshold decisions. "
                    "Positive class is failure."
                )
            with sentence_col:
                st.markdown("#### Threshold outcome")
                st.markdown(
                    f"""
                    <div class="pe-banner pe-banner-safe">
                        <div class="pe-banner-body">
                            {failure_sentence(metrics, threshold)}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"""
                    The selected model flags {flagged_count:,} of
                    {len(validation_frame):,} validation records. Of the
                    {metrics['actual_failures']:,} historical failures, it catches
                    {metrics['tp']:,} and misses {metrics['fn']:,}.
                    """
                )

            st.markdown("### Risk distribution")
            healthy_probabilities = probabilities[y_true == 0]
            failure_probabilities = probabilities[y_true == 1]
            distribution = go.Figure()
            distribution.add_trace(
                go.Histogram(
                    x=healthy_probabilities,
                    name=f"Actual healthy ({len(healthy_probabilities):,})",
                    marker={"color": GREEN},
                    opacity=0.62,
                    autobinx=False,
                    xbins={"start": 0.0, "end": 1.0, "size": 0.05},
                    hovertemplate="Probability %{x:.2f}<br>Count %{y}<extra></extra>",
                )
            )
            distribution.add_trace(
                go.Histogram(
                    x=failure_probabilities,
                    name=f"Actual failure ({len(failure_probabilities):,})",
                    marker={"color": RED},
                    opacity=0.72,
                    autobinx=False,
                    xbins={"start": 0.0, "end": 1.0, "size": 0.05},
                    hovertemplate="Probability %{x:.2f}<br>Count %{y}<extra></extra>",
                )
            )
            distribution.update_layout(
                **base_layout("Failure probability by actual outcome", "Failure probability")
            )
            distribution.update_layout(
                barmode="overlay",
                showlegend=True,
                bargap=0.04,
                height=360,
                legend={
                    "orientation": "h",
                    "yanchor": "bottom",
                    "y": 1.02,
                    "xanchor": "right",
                    "x": 1,
                    "font": {"color": MUTED, "size": 11},
                },
            )
            distribution.update_xaxes(
                range=[0, 1], tickformat=".0%", gridcolor=BORDER
            )
            distribution.update_yaxes(gridcolor=BORDER, zeroline=False)
            distribution.add_vline(
                x=threshold,
                line={"color": AMBER, "width": 2, "dash": "dot"},
                annotation_text=f"threshold {threshold:.2f}",
                annotation_position="top",
                annotation_font={"color": MUTED, "size": 11},
            )
            st.plotly_chart(
                distribution,
                width="stretch",
                config={"displayModeBar": False},
                key="health_risk_distribution",
            )
            st.caption(
                f"Bars are split by the historical {TARGET} label. The vertical line "
                "marks the active decision threshold."
            )

            st.markdown("### Risk by machine type")
            type_series = machine_type_series(validation_frame)
            type_records = []
            type_rows_html = []
            for machine_type in MACHINE_TYPES:
                mask = type_series == machine_type
                count = int(mask.sum())
                flagged = int(predicted_flagged[mask.to_numpy()].sum())
                failures = int(y_true[mask.to_numpy()].sum())
                flagged_rate = flagged / count if count else 0.0
                failure_rate = failures / count if count else 0.0
                type_records.append(
                    {
                        "type": machine_type,
                        "count": count,
                        "flagged": flagged,
                        "flagged_rate": flagged_rate,
                        "failures": failures,
                        "failure_rate": failure_rate,
                    }
                )
                type_rows_html.append(
                    f"<tr><td>{machine_type}</td><td>{count:,}</td>"
                    f"<td>{flagged:,}</td><td>{format_percent(flagged_rate)}</td>"
                    f"<td>{failures:,}</td><td>{format_percent(failure_rate)}</td></tr>"
                )
            type_figure = go.Figure()
            type_figure.add_trace(
                go.Bar(
                    x=[row["type"] for row in type_records],
                    y=[row["flagged_rate"] for row in type_records],
                    name="Flagged at risk",
                    marker={"color": AMBER},
                    text=[
                        format_percent(row["flagged_rate"]) for row in type_records
                    ],
                    textposition="outside",
                    hovertemplate="Type %{x}<br>Flagged %{y:.2%}<extra></extra>",
                )
            )
            type_figure.add_trace(
                go.Bar(
                    x=[row["type"] for row in type_records],
                    y=[row["failure_rate"] for row in type_records],
                    name="Actual failure rate",
                    marker={"color": RED},
                    text=[
                        format_percent(row["failure_rate"]) for row in type_records
                    ],
                    textposition="outside",
                    hovertemplate="Type %{x}<br>Actual failure %{y:.2%}<extra></extra>",
                )
            )
            type_figure.update_layout(
                **base_layout("Flagged rate and actual failure rate by type", "Machine type")
            )
            max_rate = max(
                [row["flagged_rate"] for row in type_records]
                + [row["failure_rate"] for row in type_records]
                + [0.01]
            )
            type_figure.update_layout(
                barmode="group",
                showlegend=True,
                height=340,
                legend={
                    "orientation": "h",
                    "yanchor": "bottom",
                    "y": 1.02,
                    "xanchor": "right",
                    "x": 1,
                    "font": {"color": MUTED, "size": 11},
                },
            )
            type_figure.update_yaxes(
                range=[0, max(0.1, max_rate * 1.35)],
                tickformat=".0%",
                gridcolor=BORDER,
                zeroline=False,
            )
            st.plotly_chart(
                type_figure,
                width="stretch",
                config={"displayModeBar": False},
                key="health_type_risk",
            )
            st.markdown(
                f"""
                <div class="pe-table-wrap">
                    <table class="pe-table">
                        <thead><tr><th>Type</th><th>Machines</th><th>Flagged</th>
                        <th>Flagged rate</th><th>Actual failures</th>
                        <th>Actual failure rate</th></tr></thead>
                        <tbody>{''.join(type_rows_html)}</tbody>
                    </table>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("### Sensor analytics")
            sensor_labels = [label for _, label, _, _ in SENSOR_COLUMNS]
            selected_sensor_label = st.selectbox(
                "Sensor",
                sensor_labels,
                index=3,
                key="health_sensor_picker",
            )
            selected_sensor = next(
                column
                for column, label, _, _ in SENSOR_COLUMNS
                if label == selected_sensor_label
            )
            healthy_values = validation_frame.loc[
                y_true == 0, selected_sensor
            ].to_numpy(dtype=float)
            failed_values = validation_frame.loc[
                y_true == 1, selected_sensor
            ].to_numpy(dtype=float)
            sensor_min = float(validation_frame[selected_sensor].min())
            sensor_max = float(validation_frame[selected_sensor].max())
            bin_size = (sensor_max - sensor_min) / 30.0
            if bin_size <= 0:
                bin_size = 1.0
            sensor_figure = go.Figure()
            sensor_figure.add_trace(
                go.Histogram(
                    x=healthy_values,
                    name=f"Actual healthy ({len(healthy_values):,})",
                    marker={"color": GREEN},
                    opacity=0.62,
                    histnorm="probability density",
                    autobinx=False,
                    xbins={
                        "start": sensor_min,
                        "end": sensor_max + bin_size,
                        "size": bin_size,
                    },
                    hovertemplate="%{x}<br>Density %{y:.4f}<extra></extra>",
                )
            )
            sensor_figure.add_trace(
                go.Histogram(
                    x=failed_values,
                    name=f"Actual failure ({len(failed_values):,})",
                    marker={"color": RED},
                    opacity=0.62,
                    histnorm="probability density",
                    autobinx=False,
                    xbins={
                        "start": sensor_min,
                        "end": sensor_max + bin_size,
                        "size": bin_size,
                    },
                    hovertemplate="%{x}<br>Density %{y:.4f}<extra></extra>",
                )
            )
            sensor_figure.update_layout(
                **base_layout(
                    f"{selected_sensor_label} distribution by actual outcome",
                    selected_sensor_label,
                )
            )
            sensor_figure.update_layout(
                barmode="overlay",
                showlegend=True,
                bargap=0.02,
                height=360,
                legend={
                    "orientation": "h",
                    "yanchor": "bottom",
                    "y": 1.02,
                    "xanchor": "right",
                    "x": 1,
                    "font": {"color": MUTED, "size": 11},
                },
            )
            sensor_figure.update_xaxes(gridcolor=BORDER)
            sensor_figure.update_yaxes(gridcolor=BORDER, zeroline=False)
            st.plotly_chart(
                sensor_figure,
                width="stretch",
                config={"displayModeBar": False},
                key="health_sensor_distribution",
            )
            healthy_median = float(np.median(healthy_values))
            failed_median = float(np.median(failed_values))
            st.caption(
                f"Healthy median {format_sensor(selected_sensor, healthy_median)}; "
                f"failure median {format_sensor(selected_sensor, failed_median)}. "
                "Density normalizes the two groups so their shapes can be compared."
            )

            st.markdown("### Highest-risk machines")
            top_frame = validation_frame.copy()
            top_frame["_probability"] = probabilities
            top_frame["_machine_type"] = type_series.to_numpy()
            top_frame = top_frame.sort_values(
                "_probability", ascending=False
            ).head(10)
            top_rows = []
            for rank, (_, row) in enumerate(top_frame.iterrows(), start=1):
                outcome = "Failure" if int(row[TARGET]) == 1 else "Healthy"
                outcome_class = (
                    "pe-verdict-risk" if outcome == "Failure" else "pe-verdict-safe"
                )
                sensor_cells = "".join(
                    f"<td>{format_sensor(column, row[column])}</td>"
                    for column, _, _, _ in SENSOR_COLUMNS
                )
                top_rows.append(
                    f"<tr><td>{rank}</td>"
                    f"<td>{format_percent(row['_probability'], 2)}</td>"
                    f"<td>{row['_machine_type']}</td>{sensor_cells}"
                    f'<td class="{outcome_class}">{outcome}</td></tr>'
                )
            st.markdown(
                f"""
                <div class="pe-table-wrap">
                    <table class="pe-table">
                        <thead><tr><th>Rank</th><th>Probability</th><th>Type</th>
                        <th>Air temperature</th><th>Process temperature</th>
                        <th>Rotational speed</th><th>Torque</th><th>Tool wear</th>
                        <th>Actual outcome</th></tr></thead>
                        <tbody>{''.join(top_rows)}</tbody>
                    </table>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.caption(
                f"Top 10 of {len(validation_frame):,} historical validation records, "
                f"ranked by {model_label} failure probability."
            )

            last_loaded(
                [
                    "data/processed/ai4i_validation.csv",
                    f"results/{MODEL_FILES[model_label]}",
                ]
            )

# ===========================================================================
# Live prediction
# ===========================================================================
elif page == "Live prediction":
    render_header()
    st.markdown("## Live prediction")
    st.markdown(
        "Score a continuous telemetry stream. The selected model and threshold are "
        "shared with the health page and apply to every source below."
    )

    source = st.radio(
        "Telemetry source",
        LIVE_SOURCES,
        horizontal=True,
        key="live_source",
    )

    available_models = [
        label for label in MODEL_ORDER if model_path(label).exists()
    ]
    missing_models = [
        label for label in MODEL_ORDER if label not in available_models
    ]

    if train_stats is None or input_ranges is None:
        empty_state(
            "Training data not found",
            "data/processed/ai4i_train.csv is missing or unreadable. The simulator, "
            "manual controls, and training percentiles all need this file. Regenerate "
            "the processed splits, then reload this page.",
            [PREPROCESS_COMMAND],
        )
    elif not available_models:
        empty_state(
            "Trained models not found",
            "No results/*.joblib model artifacts are available, so telemetry cannot "
            "be scored. Generate the model artifacts, then reload this page.",
            [TRAIN_COMMAND, TUNE_COMMAND],
        )
    else:
        if missing_models:
            st.warning(
                "Model artifacts are unavailable for: "
                + ", ".join(missing_models)
                + ". Those rows show not available."
            )

        model_label = st.session_state["pe_model"]
        threshold = float(st.session_state["pe_threshold"])

        if source == "Simulated engine":
            st.markdown("### Simulated engine")
            st.caption(
                "Simulated telemetry (generated in real time, not recorded data)"
            )
            st.caption(
                "Noise levels, ranges, machine-type operating points, the "
                "torque-speed regression and the tool-wear increment are estimated "
                f"from data/processed/ai4i_train.csv ({train_stats['rows']:,} rows)."
            )

            sim = st.session_state.get("sim_state")
            if sim is None:
                initial_wear = train_stats["sensors"]["Tool wear [min]"]["median"]
                sim = new_sim_state(
                    train_stats, initial_wear, machine_type="M", running=True
                )
                st.session_state["sim_state"] = sim
                st.session_state.setdefault("sim_history", [])
                st.session_state.setdefault("sim_total_ticks", 0)

            controls = st.columns(4)
            with controls[0]:
                machine_type = st.selectbox(
                    "Machine type",
                    MACHINE_TYPES,
                    index=MACHINE_TYPES.index(sim["machine_type"]),
                    key="sim_machine_type",
                )
            with controls[1]:
                speed_label = st.radio(
                    "Speed",
                    ["0.5x", "1x", "2x"],
                    index=1,
                    horizontal=True,
                    key="sim_speed",
                )
            with controls[2]:
                start_wear = st.slider(
                    "Starting tool wear (min)",
                    min_value=float(input_ranges["Tool wear [min]"]["min"]),
                    max_value=float(input_ranges["Tool wear [min]"]["max"]),
                    value=float(sim["start_wear"]),
                    step=1.0,
                    key="sim_start_wear",
                    help="Applied when the simulator is reset.",
                )
            with controls[3]:
                st.caption("Run state")
                st.markdown(
                    f"**{'Running' if sim['running'] else 'Paused'}**"
                )
                st.caption(f"Tick {sim['tick']:,}")

            if machine_type != sim["machine_type"]:
                sim["machine_type"] = machine_type
                sim["rpm"] = train_stats["type_stats"][machine_type]["rpm_median"]
                sim["air"] = train_stats["type_stats"][machine_type]["air_median"]
            sim["start_wear"] = float(start_wear)
            sim["speed"] = {"0.5x": 0.5, "1x": 1.0, "2x": 2.0}[speed_label]

            run_columns = st.columns(4)
            if run_columns[0].button(
                "Start",
                width="stretch",
                disabled=sim["running"],
                key="sim_start_button",
            ):
                sim["running"] = True
                sim["last_tick_wall"] = 0.0
            if run_columns[1].button(
                "Pause",
                width="stretch",
                disabled=not sim["running"],
                key="sim_pause_button",
            ):
                sim["running"] = False
            if run_columns[2].button(
                "Reset",
                width="stretch",
                key="sim_reset_button",
            ):
                st.session_state["sim_state"] = new_sim_state(
                    train_stats,
                    start_wear,
                    machine_type=machine_type,
                    running=False,
                )
                st.session_state["sim_history"] = []
                st.session_state["sim_total_ticks"] = 0
                st.rerun()
            run_columns[3].caption(
                "Readings are generated once per second at 1x speed."
            )

            st.markdown("#### Fault injection")
            fault_columns = st.columns(4)
            if fault_columns[0].button(
                "Overload (raise torque)",
                width="stretch",
                disabled="overload" in sim["faults"],
                key="fault_overload",
            ):
                apply_sim_fault(sim, "overload", train_stats)
            if fault_columns[1].button(
                "Overheat (raise temperature)",
                width="stretch",
                disabled="overheat" in sim["faults"],
                key="fault_overheat",
            ):
                apply_sim_fault(sim, "overheat", train_stats)
            if fault_columns[2].button(
                "Accelerate wear",
                width="stretch",
                disabled="accelerate_wear" in sim["faults"],
                key="fault_wear",
            ):
                apply_sim_fault(sim, "accelerate_wear", train_stats)
            if fault_columns[3].button(
                "Clear faults",
                width="stretch",
                disabled=not sim["faults"],
                key="fault_clear",
            ):
                clear_sim_faults(sim)

            if sim["faults"]:
                st.caption(
                    "Active faults: " + ", ".join(sim["faults"]).replace("_", " ")
                )
            else:
                st.caption("No injected faults. Simulation continues from training-derived conditions.")

            @st.fragment(run_every=1.0)
            def simulated_live_fragment():
                live_sim = st.session_state.get("sim_state")
                if live_sim is None:
                    return
                if live_sim["running"]:
                    now_wall = time.monotonic()
                    if now_wall - live_sim["last_tick_wall"] >= 0.55:
                        sensor_values, machine = simulate_next_reading(
                            live_sim, train_stats
                        )
                        reading = score_reading(
                            sensor_values, machine, "Simulated engine"
                        )
                        append_history("sim_history", reading)
                        st.session_state["sim_total_ticks"] = (
                            st.session_state.get("sim_total_ticks", 0) + 1
                        )
                        live_sim["last_tick_wall"] = now_wall
                if not st.session_state.get("sim_history"):
                    st.markdown(
                        '<div class="pe-empty"><div class="pe-empty-title">'
                        "Simulator paused</div><div class=\"pe-empty-body\">Press "
                        "Start to generate a reading every second.</div></div>",
                        unsafe_allow_html=True,
                    )
                    return
                render_live_panel(
                    st.session_state["sim_history"],
                    st.session_state["pe_model"],
                    float(st.session_state["pe_threshold"]),
                    train_stats,
                    train_frame,
                    int(st.session_state.get("sim_total_ticks", 0)),
                )
                if not live_sim["running"]:
                    st.caption(
                        "Simulator paused. The panel keeps the last reading and does "
                        "not generate new ticks."
                    )

            simulated_live_fragment()

        elif source == "External feed":
            st.markdown("### External feed")
            st.caption(
                "Watching data/live/stream.csv. Rows are scored in file order as they "
                "are appended."
            )

            @st.fragment(run_every=1.0)
            def external_live_fragment():
                parsed = parse_stream_csv(STREAM_PATH)
                if parsed["status"] in ("missing", "empty", "unreadable"):
                    titles = {
                        "missing": "External feed not found",
                        "empty": "External feed is empty",
                        "unreadable": "External feed could not be read",
                    }
                    empty_stream_state(
                        titles[parsed["status"]],
                        "data/live/stream.csv does not contain any telemetry rows. "
                        "Append rows with the columns listed below. Expected columns: "
                        + ", ".join(STREAM_COLUMNS)
                        + ".",
                    )
                    return
                if parsed["status"] == "invalid_columns":
                    empty_stream_state(
                        "External feed has the wrong columns",
                        "The file header does not match the required stream format. "
                        + " ".join(parsed["warnings"]),
                    )
                    return
                if parsed["status"] == "no_valid_rows":
                    empty_stream_state(
                        "External feed has no valid rows",
                        "At least one row must have an ISO timestamp, five finite "
                        "numeric sensors and machine_type L, M or H. "
                        + " ".join(parsed["warnings"]),
                    )
                    return

                for warning in parsed["warnings"]:
                    st.warning(warning)

                newest = parsed["newest"]
                age_seconds = (
                    datetime.now(timezone.utc) - newest
                ).total_seconds()
                if age_seconds > STREAM_STALE_SECONDS:
                    empty_stream_state(
                        "External feed is stale",
                        f"The newest valid row is {human_age(age_seconds)} old, "
                        f"above the {STREAM_STALE_SECONDS} s freshness limit. Append "
                        "fresh telemetry to data/live/stream.csv. Expected columns: "
                        + ", ".join(STREAM_COLUMNS)
                        + ".",
                        f"Newest timestamp: {newest.strftime('%Y-%m-%d %H:%M:%S')} UTC.",
                    )
                    return

                frame = parsed["frame"]
                processed = int(st.session_state.get("external_processed", 0))
                if len(frame) < processed:
                    st.session_state["external_history"] = []
                    st.session_state["external_total_ticks"] = 0
                    processed = 0
                if len(frame) > processed:
                    for _, row in frame.iloc[processed:].iterrows():
                        sensor_values = {
                            target_column: float(row[target_column])
                            for target_column in STREAM_SENSOR_MAP.values()
                        }
                        reading = score_reading(
                            sensor_values,
                            str(row["machine_type"]),
                            "External feed",
                            timestamp=row["timestamp"].to_pydatetime(),
                        )
                        append_history("external_history", reading)
                    st.session_state["external_total_ticks"] = (
                        st.session_state.get("external_total_ticks", 0)
                        + (len(frame) - processed)
                    )
                    st.session_state["external_processed"] = len(frame)

                history = st.session_state.get("external_history", [])
                if not history:
                    empty_stream_state(
                        "No external readings scored",
                        "The file is fresh but no rows have been scored yet. "
                        "Expected columns: " + ", ".join(STREAM_COLUMNS) + ".",
                    )
                    return
                render_live_panel(
                    history,
                    st.session_state["pe_model"],
                    float(st.session_state["pe_threshold"]),
                    train_stats,
                    train_frame,
                    int(st.session_state.get("external_total_ticks", 0)),
                )
                st.caption(
                    f"Newest valid row: {newest.strftime('%Y-%m-%d %H:%M:%S')} UTC. "
                    f"Valid rows scored: {len(frame):,}."
                )

            external_live_fragment()

        else:
            st.markdown("### Manual reading")
            st.caption("Manual telemetry (operator input, not recorded data)")
            st.caption(
                "Move a control to append a new manual reading. No reading is added "
                "when only the model or threshold changes."
            )

            manual_left, manual_right = st.columns(2, gap="large")
            manual_values = {}
            for index, (column, label, number_format, step) in enumerate(
                SENSOR_COLUMNS
            ):
                bounds = input_ranges[column]
                target = manual_left if index % 2 == 0 else manual_right
                with target:
                    if column == "Rotational speed [rpm]":
                        manual_values[column] = st.slider(
                            label,
                            min_value=int(bounds["min"]),
                            max_value=int(bounds["max"]),
                            value=int(bounds["default"]),
                            step=int(step),
                            key=f"manual_{column}",
                        )
                    elif column == "Tool wear [min]":
                        manual_values[column] = st.slider(
                            label,
                            min_value=int(bounds["min"]),
                            max_value=int(bounds["max"]),
                            value=int(bounds["default"]),
                            step=1,
                            key=f"manual_{column}",
                        )
                    else:
                        manual_values[column] = st.slider(
                            label,
                            min_value=float(bounds["min"]),
                            max_value=float(bounds["max"]),
                            value=float(bounds["default"]),
                            step=float(step),
                            format=number_format,
                            key=f"manual_{column}",
                        )
            manual_type = st.selectbox(
                "Machine type",
                MACHINE_TYPES,
                index=1,
                key="manual_machine_type",
            )

            signature = tuple(
                round(float(manual_values[column]), 6)
                for column, _, _, _ in SENSOR_COLUMNS
            ) + (manual_type,)
            if signature != st.session_state.get("manual_signature"):
                reading = score_reading(
                    manual_values, manual_type, "Manual reading"
                )
                append_history("manual_history", reading)
                st.session_state["manual_total_ticks"] = (
                    st.session_state.get("manual_total_ticks", 0) + 1
                )
                st.session_state["manual_signature"] = signature

            history = st.session_state.get("manual_history", [])
            render_live_panel(
                history,
                model_label,
                threshold,
                train_stats,
                train_frame,
                int(st.session_state.get("manual_total_ticks", 0)),
            )
            last_loaded(
                [
                    "data/processed/ai4i_train.csv",
                    f"results/{MODEL_FILES[model_label]}",
                ]
            )

# ===========================================================================
# Model comparison
# ===========================================================================
elif page == "Model Comparison":
    render_header()
    st.markdown("## Model Comparison")
    st.caption("Saved training-time results.")

    if metrics is None:
        empty_state(
            "Metrics file not found",
            "results/classical_ml_metrics.csv is missing. Run the training script "
            "to generate the comparison table and charts.",
            [TRAIN_COMMAND],
        )
    else:
        st.markdown("### Validation metrics")
        render_metrics_table(metrics)
        st.caption(
            "Validation set, 2,000 samples. Saved rows come from "
            "results/classical_ml_metrics.csv. The tuned row is recomputed live from "
            "ai4i_validation.csv at its 0.23 threshold."
        )

        st.markdown("### Metric charts")
        chart_specs = [
            ("Precision", "Precision by model", True, 1.12),
            ("Recall", "Recall by model", True, 1.12),
            ("F1-score", "F1-score by model", True, 1.12),
            ("Average Precision (PR-AUC)", "PR-AUC by model", False, 1.0),
        ]
        chart_left, chart_right = st.columns(2)
        for index, (column, title, percent, x_max) in enumerate(chart_specs):
            pairs = list(zip(metrics["Model"], metrics[column]))
            target = chart_left if index % 2 == 0 else chart_right
            with target:
                st.plotly_chart(
                    bar_figure(
                        pairs, title, column, percent=percent, x_max=x_max
                    ),
                    width="stretch",
                    config={"displayModeBar": False},
                )

        st.markdown("### Trade-off")
        combined = build_metrics_with_tuned(metrics)
        lookup = combined.set_index("Model")
        try:
            lr_row = lookup.loc["Logistic Regression"]
            rf_row = lookup.loc["Random Forest"]
            tuned_row = lookup.loc[TUNED_LABEL]
            st.markdown(
                f"""
                Logistic Regression catches
                {format_percent(lr_row['Recall'])} of actual failures but only
                {format_percent(lr_row['Precision'])} of its failure alarms are
                correct. Random Forest is precise at
                {format_percent(rf_row['Precision'])} but catches
                {format_percent(rf_row['Recall'])} of failures at the saved
                threshold. The tuned row at 0.23 raises recall to
                {format_percent(tuned_row['Recall'])} while accuracy moves from
                {format_percent(rf_row['Accuracy'])} to
                {format_percent(tuned_row['Accuracy'])}. The tuned setting trades a
                small amount of accuracy for more real failures caught.
                """
            )
        except (KeyError, TypeError):
            st.markdown(
                "Saved rows are available, but the tuned row could not be computed "
                "because a model artifact or validation file is missing."
            )

    last_loaded(["results/classical_ml_metrics.csv"])

# ===========================================================================
# Feature importance
# ===========================================================================
elif page == "Feature Importance":
    render_header()
    st.markdown("## Feature Importance")

    if importance is None:
        empty_state(
            "Feature importance file not found",
            "results/random_forest_feature_importance.csv is missing. Run the "
            "threshold tuning script to generate it.",
            [TUNE_COMMAND],
        )
    else:
        frame = importance.sort_values("Importance", ascending=False)
        pairs = list(zip(frame["Feature"], frame["Importance"]))
        st.plotly_chart(
            bar_figure(pairs, "Random Forest feature importance", "Share of importance"),
            width="stretch",
            config={"displayModeBar": False},
        )

        st.markdown("### Values")
        render_importance_table(importance)

        top_three = frame.head(3)["Importance"].sum()
        st.markdown(
            f"""
            Rotational speed, torque and tool wear together account for
            {format_percent(top_three)} of the Random Forest's importance, about 80% in
            total. The two temperature features and the machine-type indicators make
            up the remainder.
            """
        )
        st.caption(
            "Importance indicates predictive contribution, not causation. Source: "
            "results/random_forest_feature_importance.csv."
        )

    last_loaded(["results/random_forest_feature_importance.csv"])

# ===========================================================================
# Confusion matrices
# ===========================================================================
elif page == "Confusion matrices":
    render_header()
    st.markdown("## Confusion matrices")
    st.markdown(
        "Rows are actual classes and columns are predicted classes. The diagonal "
        "shows correct predictions; off-diagonal cells show errors."
    )

    tabs = st.tabs(list(CM_FILES.keys()))
    for tab, (model_name, filename) in zip(tabs, CM_FILES.items()):
        with tab:
            path = RESULTS_DIR / filename
            if path.exists():
                st.image(str(path), width="stretch")
                st.caption(
                    "0 is no failure and 1 is failure. Rows are actual, columns are "
                    "predicted."
                )
            else:
                empty_state(
                    "Confusion matrix not found",
                    f"results/{filename} is missing. Run the training script to "
                    "regenerate the confusion matrices.",
                    [TRAIN_COMMAND],
                )

    last_loaded([f"results/{filename}" for filename in CM_FILES.values()])

# ===========================================================================
# About and methodology
# ===========================================================================
else:
    render_header()
    st.markdown("## About and methodology")

    st.markdown("### Pages")
    st.markdown(
        """
        **Engine health and analytics** is the default page. It scores the historical
        validation fleet with the selected model and threshold, showing fleet health
        cards, live performance metrics, a live confusion matrix, risk distribution,
        risk by machine type, sensor distributions, and the highest-risk machines.

        **Live prediction** scores a continuous stream from a training-derived
        simulator, an external CSV feed, or manual controls. It shows a probability
        gauge, status banner, rolling probability and sensor charts, readout tiles,
        all four model probabilities, an alert log, counters, and training
        percentiles.

        **Model Comparison**, **Feature Importance**, and **Confusion matrices** show
        saved training artifacts. **About** documents the methodology.
        """
    )

    st.markdown("### Data and leakage prevention")
    st.markdown(
        """
        The models use eight inputs: air temperature, process temperature, rotational
        speed, torque, tool wear, plus one-hot columns for machine type H, L, and M.
        The identifiers UDI and Product ID are excluded because they carry no physical
        signal. The failure-mode columns TWF, HDF, PWF, OSF and RNF are excluded
        because they reveal the failure outcome and would leak the target.
        """
    )

    st.markdown("### Training choices")
    st.markdown(
        """
        All experiments use seed 42 and balanced class weights to account for the
        rare failure class. Logistic Regression uses a StandardScaler pipeline with
        class_weight set to balanced and max_iter 2000. The Decision Tree uses
        balanced class weights. The Random Forest uses 200 estimators, balanced
        class weights, n_jobs set to -1, and random_state 42.
        """
    )

    st.markdown("### Threshold tuning")
    st.markdown(
        """
        The tuned Random Forest keeps the same trained model and changes only the
        decision threshold. The threshold 0.23 was chosen using stratified 3-fold
        cross-validation on the training data only, maximizing F1 while requiring at
        least 75% recall. The validation set was not used to choose the threshold.
        Lowering the threshold to 0.23 raises failure recall at a small accuracy
        cost. Model Comparison recomputes the tuned row from validation data.
        """
    )

    st.markdown("### Validation fleet")
    st.markdown(
        """
        Engine health and analytics scores data/processed/ai4i_validation.csv, a
        historical validation split of 2,000 records. Those readings are historical
        records, not live sensor data. Every metric, confusion matrix, probability
        distribution and risk table on that page is calculated live from the cached
        model probabilities for the active threshold. The page does not read
        results/classical_ml_metrics.csv.
        """
    )

    st.markdown("### Simulator methodology")
    st.markdown(
        """
        The simulator initialises each sensor near a training median and then
        generates new readings in the browser session. It is simulated telemetry,
        not recorded data. Ranges, medians, standard deviations, machine-type
        operating points, the air-to-process temperature offset, the torque-speed
        regression and its residual noise, and the tool-wear increment are all
        estimated from data/processed/ai4i_train.csv. Air temperature follows a
        bounded random walk inside the training range. Process temperature tracks
        air temperature plus the estimated offset. Rotational speed moves around the
        selected machine type's operating point, and torque follows the fitted
        inverse torque-speed relationship with training residual noise. Tool wear
        advances by a trained-range increment derived from the median wear at
        training failures.

        Fault buttons apply training-scale stress profiles. Overload raises torque
        and lowers speed, overheat raises temperature and the coupled thermal load,
        and accelerate wear advances tool wear while raising the associated load.
        Clear faults removes the active offsets; cumulative tool wear is not
        reversed.
        """
    )

    st.markdown("### External feed")
    st.markdown(
        """
        Live prediction can watch data/live/stream.csv with columns timestamp,
        air_temperature, process_temperature, rotational_speed, torque, tool_wear and
        machine_type. Timestamps are parsed as UTC, values must be finite numbers,
        and machine_type must be L, M or H. Malformed rows are skipped with a visible
        message, and a feed older than 30 seconds is shown as stale rather than
        scored as fresh. Missing sensor values are never filled in.
        """
    )

    st.markdown("### Scope")
    st.markdown(
        """
        The repository also contains NASA C-MAPSS FD001 and FD003 data and a shared
        contract for Remaining Useful Life prediction. The models shipped in this app
        only perform binary failure classification on the AI4I 2020 dataset. No RUL
        model is loaded or displayed here.
        """
    )

    st.caption(
        "Validation-set results and simulated telemetry are shown for engineering "
        "review only. These models are not intended for production decisions without "
        "independent test-set evaluation."
    )
    last_loaded(
        [
            "data/processed/ai4i_metadata.json",
            "data/processed/ai4i_train.csv",
            "data/processed/ai4i_validation.csv",
        ]
    )
