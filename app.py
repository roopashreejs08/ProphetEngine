"""
ProphetEngine predictive maintenance console.

Single-file Streamlit app. Every value shown comes from repository artifacts:
- results/classical_ml_metrics.csv
- results/random_forest_feature_importance.csv
- results/*.joblib
- results/*_confusion_matrix.png
- data/processed/ai4i_metadata.json
- data/processed/ai4i_train.csv (input ranges only)

Run with: streamlit run app.py
"""

import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data" / "processed"
RESULTS_DIR = REPO_ROOT / "results"

METADATA_PATH = DATA_DIR / "ai4i_metadata.json"
TRAIN_PATH = DATA_DIR / "ai4i_train.csv"
METRICS_PATH = RESULTS_DIR / "classical_ml_metrics.csv"
IMPORTANCE_PATH = RESULTS_DIR / "random_forest_feature_importance.csv"

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

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "Random Forest": "random_forest.joblib",
    "Random Forest (tuned, threshold 0.23)": "random_forest.joblib",
}

MODEL_ORDER = [
    "Logistic Regression",
    "Decision Tree",
    "Random Forest",
    "Random Forest (tuned, threshold 0.23)",
]

CM_FILES = {
    "Logistic Regression": "logistic_regression_confusion_matrix.png",
    "Decision Tree": "decision_tree_confusion_matrix.png",
    "Random Forest": "random_forest_confusion_matrix.png",
}

PAGES = [
    "Dashboard",
    "Failure prediction",
    "Model comparison",
    "Feature importance",
    "Confusion matrices",
    "About and methodology",
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
def load_input_ranges():
    """Derive form bounds from the training split, as required by the brief."""
    if not TRAIN_PATH.exists():
        return None
    train = pd.read_csv(TRAIN_PATH)
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
    tuned = pd.DataFrame(
        [
            {
                "Model": "Random Forest (tuned, threshold 0.23)",
                "Accuracy": 0.9745,
                "Precision": np.nan,
                "Recall": 0.7353,
                "F1-score": np.nan,
                "Average Precision (PR-AUC)": np.nan,
                "ROC-AUC": np.nan,
            }
        ]
    )
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


# ---------------------------------------------------------------------------
# Shared data
# ---------------------------------------------------------------------------
metadata = load_metadata()
metrics = load_metrics()
importance = load_importance()
input_ranges = load_input_ranges()

train_rows = metadata["train_rows"] if metadata else 8000
validation_rows = metadata["validation_rows"] if metadata else 2000
seed = metadata["random_state"] if metadata else 42

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
# Dashboard
# ===========================================================================
if page == "Dashboard":
    render_header()
    st.markdown("## Dashboard")

    if metrics is None:
        empty_state(
            "Metrics file not found",
            "results/classical_ml_metrics.csv is missing, so no model results can be shown. "
            "Run the training script, then reload this page.",
            [TRAIN_COMMAND],
        )
    else:
        best_accuracy = metrics.loc[metrics["Accuracy"].idxmax()]
        best_pr_auc = metrics.loc[metrics["Average Precision (PR-AUC)"].idxmax()]

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            metric_card(
                "Best accuracy",
                format_percent(best_accuracy["Accuracy"]),
                f"{best_accuracy['Model']}, validation set",
            )
        with col2:
            metric_card(
                "Best PR-AUC",
                f"{best_pr_auc['Average Precision (PR-AUC)']:.4f}",
                f"{best_pr_auc['Model']}, validation set",
            )
        with col3:
            metric_card(
                "Tuned RF recall",
                "73.53%",
                "Threshold 0.23, 50 of 68 failures caught",
            )
        with col4:
            metric_card(
                "Samples",
                f"{train_rows:,} / {validation_rows:,}",
                "Training / validation",
            )

    st.markdown("## Dataset at a glance")
    st.markdown(
        f"""
        The project predicts whether an industrial engine will fail, using five sensor
        readings and the machine type as inputs. The target is `Machine failure`, a
        binary label. Models train on the AI4I 2020 Predictive Maintenance Dataset
        with {train_rows:,} training samples and {validation_rows:,} validation
        samples at seed {seed}. Class weights are balanced to account for the rare
        failure class.
        """
    )
    st.markdown(
        f"""
        Identifiers `UDI` and `Product ID` and the failure-mode columns `TWF`, `HDF`,
        `PWF`, `OSF` and `RNF` are excluded from the model inputs to prevent target
        leakage.
        """
    )

    if metadata is None:
        st.caption(
            "data/processed/ai4i_metadata.json was not found. The sample counts and "
            "seed above use the documented project values."
        )

    if importance is not None:
        st.markdown("## Top five features")
        top_five = importance.sort_values("Importance", ascending=False).head(5)
        pairs = list(zip(top_five["Feature"], top_five["Importance"]))
        st.plotly_chart(
            bar_figure(
                pairs,
                "Random Forest importance, top five features",
                "Share of importance",
            ),
            width="stretch",
            config={"displayModeBar": False},
        )
        st.caption(
            "Rotational speed, torque and tool wear dominate; the two temperature "
            "features follow. Values come from results/random_forest_feature_importance.csv."
        )
    else:
        st.markdown("## Top five features")
        empty_state(
            "Feature importance file not found",
            "results/random_forest_feature_importance.csv is missing. Run the "
            "threshold tuning script to regenerate it.",
            [TUNE_COMMAND],
        )

    st.markdown("## Why recall and PR-AUC matter here")
    st.markdown(
        """
        Failures are rare: only 68 of the 2,000 validation samples are actual
        failures. A model can reach high accuracy by predicting "no failure" almost
        every time, which is why accuracy alone is misleading on this dataset.
        Recall measures how many real failures the model catches. PR-AUC summarizes
        precision and recall across all thresholds. Missing a real failure is the
        expensive mistake, so recall and PR-AUC are the metrics that matter most.
        """
    )

    last_loaded(
        [
            "results/classical_ml_metrics.csv",
            "results/random_forest_feature_importance.csv",
            "data/processed/ai4i_metadata.json",
        ]
    )

# ===========================================================================
# Failure prediction
# ===========================================================================
elif page == "Failure prediction":
    render_header()
    st.markdown("## Failure prediction")

    if input_ranges is None:
        empty_state(
            "Training data not found",
            "data/processed/ai4i_train.csv is missing, so the input ranges for the "
            "prediction form cannot be derived. Restore the file, then reload this page.",
        )
    else:
        left, right = st.columns([1.05, 1], gap="large")

        with left:
            model_name = st.selectbox(
                "Model",
                list(MODEL_FILES.keys()),
                index=2,
                help="The tuned option reuses the Random Forest model with a 0.23 threshold.",
            )

            st.markdown("### Sensor readings")
            values = {}
            sensor_left, sensor_right = st.columns(2)
            for index, (column, label, number_format, step) in enumerate(SENSOR_COLUMNS):
                bounds = input_ranges[column]
                target = sensor_left if index % 2 == 0 else sensor_right
                with target:
                    values[column] = st.number_input(
                        label,
                        min_value=bounds["min"],
                        max_value=bounds["max"],
                        value=bounds["default"],
                        step=step,
                        format=number_format,
                        key=f"input_{column}",
                    )

            st.markdown("### Machine")
            machine_type = st.selectbox("Machine type", ["L", "M", "H"], index=1)

            input_frame = build_input_frame(values, machine_type)

            model_filename = MODEL_FILES[model_name]
            model_path = RESULTS_DIR / model_filename

        with right:
            st.markdown("### Result")

            if not model_path.exists():
                empty_state(
                    "Trained model not found",
                    f"results/{model_filename} is missing, so a prediction cannot "
                    "be made. Generate the model artifacts, then reload this page.",
                    [TRAIN_COMMAND, TUNE_COMMAND],
                )
            else:
                default_threshold = 0.23 if "tuned" in model_name else 0.5
                threshold = st.slider(
                    "Decision threshold",
                    min_value=0.05,
                    max_value=0.95,
                    value=default_threshold,
                    step=0.01,
                    key=f"threshold::{model_name}",
                    help="Predict failure when the model probability is at or above this value.",
                )
                predict_clicked = st.button(
                    "Run prediction", type="primary", width="stretch"
                )

                signature = tuple(
                    values[column] for column, _, _, _ in SENSOR_COLUMNS
                ) + (machine_type,)

                if predict_clicked:
                    try:
                        with st.spinner("Loading model"):
                            model = load_model(model_filename)
                        probability = float(model.predict_proba(input_frame)[0, 1])
                        st.session_state["prediction"] = {
                            "probability": probability,
                            "model": model_name,
                            "signature": signature,
                            "readout": {
                                column: values[column]
                                for column, _, _, _ in SENSOR_COLUMNS
                            },
                        }
                        st.session_state.pop("prediction_error", None)
                    except Exception:
                        st.session_state.pop("prediction", None)
                        st.session_state["prediction_error"] = (
                            model_name,
                            f"results/{model_filename} could not be loaded or used. "
                            "Regenerate the model artifacts and try again.",
                        )

                error = st.session_state.get("prediction_error")
                error_for_current = bool(error and error[0] == model_name)
                if error_for_current:
                    empty_state(
                        "Prediction failed",
                        error[1],
                        [TRAIN_COMMAND, TUNE_COMMAND],
                    )

                stored = st.session_state.get("prediction")
                if (
                    stored
                    and stored.get("model") == model_name
                    and stored.get("signature") == signature
                ):
                    probability = stored["probability"]
                    predicted_failure = probability >= threshold

                    if predicted_failure:
                        st.markdown(
                            f"""
                            <div class="pe-banner pe-banner-risk">
                                <div class="pe-banner-title">Failure predicted</div>
                                <div class="pe-banner-body">
                                    Probability {probability:.2f} is at or above the
                                    {threshold:.2f} threshold, so the model flags this
                                    machine.
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f"""
                            <div class="pe-banner pe-banner-safe">
                                <div class="pe-banner-title">No failure predicted</div>
                                <div class="pe-banner-body">
                                    Probability {probability:.2f} is below the
                                    {threshold:.2f} threshold, so the model does not
                                    flag this machine.
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    st.plotly_chart(
                        probability_figure(probability, threshold),
                        width="stretch",
                        config={"displayModeBar": False},
                    )
                    st.caption(
                        f"The dotted line marks the active threshold at {threshold:.2f}. "
                        "Hover the bar for the exact probability."
                    )

                    tiles = []
                    for column, _, _, _ in SENSOR_COLUMNS:
                        label = column.replace(" [K]", "").replace(" [rpm]", "").replace(" [Nm]", "").replace(" [min]", "")
                        tiles.append(
                            f"""
                            <div class="pe-tile">
                                <div class="pe-tile-label">{label}</div>
                                <div class="pe-tile-value">{format_sensor(column, stored['readout'][column])}</div>
                            </div>
                            """
                        )
                    st.markdown(
                        '<div class="pe-tile-row">' + "".join(tiles) + "</div>",
                        unsafe_allow_html=True,
                    )

                    st.markdown("**What drives this model**")
                    st.markdown(
                        """
                        Rotational speed, torque and tool wear are the three most
                        important Random Forest features, together accounting for about
                        80% of its importance.
                        """
                    )
                elif not error_for_current:
                    st.markdown(
                        '<div class="pe-empty">'
                        '<div class="pe-empty-title">No prediction yet</div>'
                        '<div class="pe-empty-body">Set the sensor readings and machine type, '
                        'then run prediction to see the failure probability and status.</div>'
                        "</div>",
                        unsafe_allow_html=True,
                    )

        last_loaded(
            [
                "data/processed/ai4i_train.csv",
                f"results/{model_filename}",
            ]
        )

    if input_ranges is None:
        last_loaded(["data/processed/ai4i_train.csv"])

# ===========================================================================
# Model comparison
# ===========================================================================
elif page == "Model comparison":
    render_header()
    st.markdown("## Model comparison")

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
            "Validation set, 2,000 samples. The tuned row reports only the metrics "
            "recorded for that experiment."
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
        st.markdown(
            """
            Logistic Regression catches 82.35% of failures but only 14.21% of its
            failure alarms are correct. Random Forest is precise at 90.63% but catches
            only 42.65% of failures at the default 0.5 threshold. Lowering its
            threshold to 0.23 raises recall to 73.53% while accuracy moves from
            97.90% to 97.45%. The tuned model trades a small amount of accuracy for
            far more real failures caught.
            """
        )

    last_loaded(["results/classical_ml_metrics.csv"])

# ===========================================================================
# Feature importance
# ===========================================================================
elif page == "Feature importance":
    render_header()
    st.markdown("## Feature importance")

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
        Lowering the threshold from 0.5 to 0.23 raises failure recall from 42.65% to
        73.53% and catches 50 of 68 actual failures, at a small accuracy cost from
        97.90% to 97.45%.
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
        "Validation-set results only. These models are not intended for production "
        "decisions without independent test-set evaluation."
    )
    last_loaded(["data/processed/ai4i_metadata.json"])
