"""Streamlit glue: cached loaders, shared controls, styling and small components."""

import html

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from dashboard import config as C
from dashboard import data as D
from dashboard import scoring as S
from dashboard import simulator as SIM


# ---------------------------------------------------------------------------
# Theme helpers
# ---------------------------------------------------------------------------
def mode():
    """'light' or 'dark' - follows the viewer's active Streamlit theme."""
    try:
        kind = st.context.theme.type
    except Exception:  # older Streamlit or no browser context (tests)
        kind = None
    return "light" if kind == "light" else "dark"


def model_color(label):
    return C.MODEL_COLORS[mode()][label]


CSS = """
<style>
.block-container {padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1320px;}
[data-testid="stAppDeployButton"] {display: none;}
h1, h2, h3 {letter-spacing: -0.015em;}
[data-testid="stMetricLabel"] p {font-size: 0.82rem; opacity: 0.8; font-weight: 500;}
[data-testid="stMetricValue"] {font-size: 1.6rem;}
[data-testid="stCaptionContainer"] {opacity: 0.85;}

.pe-hero {
  border-radius: 14px; padding: 22px 26px; margin-bottom: 18px;
  background: linear-gradient(120deg, #1c5cab 0%, #2a78d6 55%, #6250d6 100%);
  color: #ffffff;
}
.pe-hero .eyebrow {font-size: 0.75rem; letter-spacing: 0.12em; text-transform: uppercase; opacity: 0.85; font-weight: 600;}
.pe-hero h1 {color: #ffffff; font-size: 1.85rem; margin: 4px 0 6px 0; padding: 0;}
.pe-hero p {color: #ffffff; opacity: 0.92; margin: 0; font-size: 0.98rem; max-width: 860px;}
.pe-hero .chips {margin-top: 12px; display: flex; flex-wrap: wrap; gap: 6px;}
.pe-hero .chip {background: rgba(255,255,255,0.16); border: 1px solid rgba(255,255,255,0.28);
  border-radius: 999px; padding: 2px 10px; font-size: 0.78rem; font-weight: 500;}

.pe-page-title {margin: 0 0 2px 0;}
.pe-page-sub {opacity: 0.75; margin: 0 0 14px 0; font-size: 0.97rem;}

.pe-pill {display: inline-flex; align-items: center; gap: 6px; border-radius: 999px;
  padding: 3px 12px; font-weight: 600; font-size: 0.85rem;}
.pe-pill.critical {background: #d03b3b; color: #ffffff;}
.pe-pill.watch {background: #fab219; color: #0b0b0b;}
.pe-pill.healthy {background: #0ca30c; color: #ffffff;}
.pe-pill.unknown {background: #898781; color: #ffffff;}

.pe-verdict {border-radius: 12px; padding: 16px 18px; border: 1px solid rgba(128,128,128,0.25);}
.pe-verdict .big {font-size: 2.6rem; font-weight: 700; line-height: 1.1; margin: 6px 0 2px 0;}
.pe-verdict .small {opacity: 0.8; font-size: 0.9rem;}

.pe-rule {display: flex; justify-content: space-between; gap: 10px; padding: 8px 0;
  border-bottom: 1px solid rgba(128,128,128,0.18); font-size: 0.9rem;}
.pe-rule:last-child {border-bottom: none;}
.pe-rule .name {font-weight: 600;}
.pe-rule .desc {opacity: 0.72; font-size: 0.8rem;}
.pe-rule .val {text-align: right; white-space: nowrap;}

.pe-side-brand {font-weight: 700; font-size: 1.15rem; letter-spacing: -0.01em;}
.pe-side-sub {opacity: 0.7; font-size: 0.82rem; margin-bottom: 4px;}

@media (max-width: 640px) {
  .pe-hero {padding: 16px;}
  .pe-hero h1 {font-size: 1.4rem;}
  .pe-verdict .big {font-size: 2rem;}
}
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_metadata():
    return D.read_json(C.METADATA_PATH)


@st.cache_data(show_spinner=False)
def load_train():
    return D.read_model_frame(C.TRAIN_PATH)


@st.cache_data(show_spinner=False)
def load_validation():
    return D.read_model_frame(C.VALIDATION_PATH)


@st.cache_data(show_spinner=False)
def load_cleaned():
    try:
        return pd.read_csv(C.CLEANED_PATH)
    except (OSError, ValueError):
        return None


@st.cache_data(show_spinner=False)
def load_validation_with_modes():
    return D.attach_failure_modes(load_validation(), load_cleaned())


@st.cache_data(show_spinner=False)
def load_metrics():
    return D.read_metrics(C.METRICS_PATH)


@st.cache_data(show_spinner=False)
def load_importance():
    return D.read_importance(C.IMPORTANCE_PATH)


@st.cache_data(show_spinner=False)
def load_ranges():
    train = load_train()
    return None if train is None else D.training_ranges(train)


@st.cache_data(show_spinner=False)
def load_sim_stats():
    train = load_train()
    return None if train is None else SIM.training_stats(train)


@st.cache_data(show_spinner=False)
def load_cmapss(subset, split):
    try:
        return pd.read_csv(C.DATA_DIR / f"{subset}_{split}.csv")
    except (OSError, ValueError):
        return None


@st.cache_resource(show_spinner="Loading model...")
def _load_artifact(filename):
    return joblib.load(C.RESULTS_DIR / filename)


def get_model(label):
    """Trained estimator for a model label, or None if the artifact is missing/unreadable."""
    filename = C.artifact_file(label)
    if not (C.RESULTS_DIR / filename).exists():
        return None
    try:
        return _load_artifact(filename)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def validation_probabilities(label):
    """Validation-set failure probabilities for a model (threshold-free, so cacheable)."""
    validation = load_validation()
    model = get_model(label)
    if validation is None or model is None:
        return None
    return S.predict_proba(model, validation)


@st.cache_data(show_spinner="Computing permutation importance on the validation set...")
def permutation_importance(label):
    validation = load_validation()
    model = get_model(label)
    if validation is None or model is None:
        return None
    return S.permutation_drop(model, validation, validation[C.TARGET].to_numpy(), repeats=5)


@st.cache_data(show_spinner=False)
def partial_dependence(label, column, points=30):
    validation = load_validation()
    ranges = load_ranges()
    model = get_model(label)
    if validation is None or model is None or ranges is None:
        return None
    grid = np.linspace(ranges[column]["min"], ranges[column]["max"], points)
    return grid, S.partial_dependence(model, validation, column, grid)


def score_all(features):
    """Probabilities from every model for a features frame. Missing models -> None."""
    out = {}
    for label in C.BASE_MODELS:
        model = get_model(label)
        out[label] = None if model is None else S.predict_proba(model, features)
    out[C.TUNED_LABEL] = out["Random Forest"]
    return out


# ---------------------------------------------------------------------------
# Shared scoring controls (sidebar, persisted across pages)
# ---------------------------------------------------------------------------
def init_state():
    if st.session_state.get("pe_model") not in C.MODEL_ORDER:
        st.session_state["pe_model"] = C.DEFAULT_MODEL
    if "pe_threshold" not in st.session_state:
        st.session_state["pe_threshold"] = C.default_threshold(st.session_state["pe_model"])


def _on_model_change():
    st.session_state["pe_threshold"] = C.default_threshold(st.session_state["pe_model"])


def sidebar_controls():
    init_state()
    with st.sidebar:
        st.markdown("**Decision settings**")
        st.selectbox(
            "Model",
            C.MODEL_ORDER,
            key="pe_model",
            on_change=_on_model_change,
            help="Used on every page. 'Random Forest (tuned)' is the team's recommended "
                 "setup: the Random Forest with its alert threshold lowered to 0.23.",
        )
        st.slider(
            "Alert threshold",
            min_value=0.01,
            max_value=0.99,
            step=0.01,
            key="pe_threshold",
            help="A machine is flagged when its predicted failure probability is at or above this value. "
                 "Lower = catch more failures but raise more false alarms.",
        )
        if st.session_state["pe_threshold"] != C.default_threshold(st.session_state["pe_model"]):
            st.button("Reset to model default", on_click=_on_model_change, width="stretch")


def active_model():
    init_state()
    return st.session_state["pe_model"], float(st.session_state["pe_threshold"])


# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------
def hero(title, subtitle, chips=()):
    chip_html = "".join(f'<span class="chip">{html.escape(c)}</span>' for c in chips)
    st.markdown(
        f"""<div class="pe-hero"><div class="eyebrow">ProphetEngine - predictive maintenance</div>
        <h1>{html.escape(title)}</h1><p>{html.escape(subtitle)}</p>
        <div class="chips">{chip_html}</div></div>""",
        unsafe_allow_html=True,
    )


def page_header(title, subtitle):
    st.markdown(f"## {title}")
    st.markdown(f'<p class="pe-page-sub">{html.escape(subtitle)}</p>', unsafe_allow_html=True)


TIER_TEXT = {"critical": "At risk", "watch": "Watch", "healthy": "Healthy", "unknown": "Unavailable"}
TIER_ICON = {"critical": "&#9888;", "watch": "&#9679;", "healthy": "&#10003;", "unknown": "?"}


def status_pill(tier, text=None):
    label = text or TIER_TEXT[tier]
    return f'<span class="pe-pill {tier}">{TIER_ICON[tier]} {html.escape(label)}</span>'


def verdict_card(probability, threshold, model_label):
    tier = S.risk_tier(probability, threshold)
    st.markdown(
        f"""<div class="pe-verdict">{status_pill(tier)}
        <div class="big">{probability:.1%}</div>
        <div class="small">predicted failure probability ({html.escape(model_label)}, alert at {threshold:.0%})</div>
        </div>""",
        unsafe_allow_html=True,
    )


def missing(what, fix=None):
    """Friendly message when an artifact is missing."""
    body = f"**{what} is not available.**"
    if fix:
        body += f" To regenerate it, run from the repository root:\n\n```bash\n{fix}\n```"
    st.info(body, icon=":material/info:")


def fmt_pct(value, digits=1):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    return f"{value * 100:.{digits}f}%"


def sensor_text(column, value):
    unit = C.SENSOR_UNIT[column]
    if unit in ("rpm", "min"):
        return f"{value:,.0f} {unit}"
    return f"{value:,.1f} {unit}"


def kpi(column, label, value, note=None, help=None, **_ignored):
    """A KPI card: big value, label, and a wrapping one-line explanation underneath."""
    with column.container(border=True):
        st.metric(label, value, help=help)
        if note:
            st.caption(note)
