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
from dashboard import theme as T


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


def tokens():
    """Colour tokens for the viewer's active theme (see dashboard/theme.py)."""
    return T.tokens(mode())


def key_model(present=None):
    """The model drawn in the accent colour: the active model, or the Random Forest when the
    tuned entry (same probabilities, different threshold) is active but not in the chart."""
    present = list(present or C.MODEL_ORDER)
    active = st.session_state.get("pe_model", C.DEFAULT_MODEL)
    if active in present:
        return active
    if active == C.TUNED_LABEL and "Random Forest" in present:
        return "Random Forest"
    return None


def model_style(label, present=None):
    """(colour, dash) for a model series: accent + solid for the key model, greys + dashes for the rest."""
    present = list(present or C.MODEL_ORDER)
    t = tokens()
    key = key_model(present)
    if label == key:
        return t["accent"], "solid"
    others = [m for m in C.MODEL_ORDER if m in present and m != key]
    i = others.index(label) if label in others else 0
    return t["greys"][min(i, 2)], T.DASHES[min(i, 2)]


def model_color(label, present=None):
    return model_style(label, present)[0]


def build_css():
    t = tokens()
    return f"""
<style>
/* Layout */
.block-container {{padding-top: 2.5rem; padding-bottom: 4rem; max-width: 1240px;}}
[data-testid="stAppDeployButton"] {{display: none;}}
h1, h2, h3, h4, h5 {{letter-spacing: -0.015em; font-weight: 600;}}
[data-testid="stMarkdownContainer"] h3 {{margin-top: 1.4rem;}}
[data-testid="stCaptionContainer"] {{opacity: 1;}}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {{color: {t["muted"]};}}
[data-testid="stMarkdownContainer"] code, [data-testid="stCaptionContainer"] code {{color: {t["text2"]} !important;
  background: {t["subtle"]} !important; border-radius: 4px;
  padding: 0.1em 0.35em;}}

/* KPI cards: number first, quiet label, hairline border, no fill */
[data-testid="stMetric"] {{background: transparent;}}
[data-testid="stMetricLabel"] p {{font-size: 0.78rem; color: {t["muted"]}; font-weight: 500;}}
[data-testid="stMetricValue"] {{font-size: 1.6rem; font-weight: 600; letter-spacing: -0.02em;}}
[data-testid="stVerticalBlockBorderWrapper"], [data-testid="stMetric"] {{box-shadow: none;}}

/* Primary buttons: text that stays readable on the accent in both themes */
[data-testid="stBaseButton-primary"], [data-testid="stBaseButton-primary"] p,
[data-testid="stBaseButton-primary"] [data-testid="stIconMaterial"] {{color: {t["on_accent"]} !important;}}
[data-testid="stBaseButton-primary"] {{box-shadow: none;}}

/* Multiselect tags: neutral chips instead of filled accent */
[data-testid="stMultiSelectTagsContainer"] [data-tag] {{background: transparent !important;
  border: 1px solid {t["border"]}; color: {t["text"]} !important;}}
[data-testid="stMultiSelectTagsContainer"] [data-tag] * {{color: {t["text"]} !important;}}

/* Alerts: no coloured fills - hairline border and a neutral surface */
[data-testid="stAlertContainer"] {{background: transparent; border: 1px solid {t["border"]}; color: {t["text"]};}}
[data-testid="stAlertContainer"] [data-testid="stIconMaterial"] {{color: {t["muted"]};}}
[data-testid="stAlertContentError"] [data-testid="stIconMaterial"] {{color: {t["red"]};}}

/* Page intro (overview) */
.pe-hero {{padding: 4px 0 22px 0; margin-bottom: 8px; border-bottom: 1px solid {t["border"]};}}
.pe-hero .eyebrow {{font-size: 0.72rem; letter-spacing: 0.08em; text-transform: uppercase; color: {t["muted"]};
  font-weight: 500;}}
.pe-hero h1 {{font-size: 1.9rem; font-weight: 600; letter-spacing: -0.025em; margin: 6px 0 8px 0; padding: 0;}}
.pe-hero p {{color: {t["text2"]}; margin: 0; font-size: 1rem; line-height: 1.55; max-width: 760px;}}
.pe-hero .chips {{margin-top: 14px; display: flex; flex-wrap: wrap; gap: 6px 18px; font-size: 0.8rem;
  color: {t["muted"]};}}
.pe-hero .chip b {{color: {t["text"]}; font-weight: 500;}}

.pe-page-sub {{color: {t["text2"]}; margin: -4px 0 18px 0; font-size: 0.97rem; max-width: 760px;}}

/* Status labels: a small dot plus text - colour is never the only cue */
.pe-pill {{display: inline-flex; align-items: center; gap: 7px; border-radius: 6px; padding: 2px 9px;
  font-weight: 500; font-size: 0.82rem; border: 1px solid {t["border"]}; color: {t["text"]};
  background: transparent; white-space: nowrap;}}
.pe-pill .dot {{width: 7px; height: 7px; border-radius: 50%; display: inline-block;}}
.pe-pill.critical {{border-color: {t["red"]}; color: {t["red"]};}}
.pe-pill.critical .dot {{background: {t["red"]};}}
.pe-pill.watch .dot {{background: {t["greys"][0]};}}
.pe-pill.healthy .dot {{background: {t["green"]};}}
.pe-pill.unknown .dot {{background: {t["greys"][1]};}}
.pe-pills {{display: flex; flex-wrap: wrap; gap: 8px; margin: 2px 0 6px 0;}}

.pe-verdict {{border-radius: 8px; padding: 16px 18px; border: 1px solid {t["border"]};}}
.pe-verdict .big {{font-size: 2.6rem; font-weight: 600; letter-spacing: -0.03em; line-height: 1.1;
  margin: 10px 0 4px 0;}}
.pe-verdict .small {{color: {t["muted"]}; font-size: 0.86rem;}}

.pe-rule {{display: flex; justify-content: space-between; gap: 12px; padding: 10px 0;
  border-bottom: 1px solid {t["border"]}; font-size: 0.9rem;}}
.pe-rule:last-child {{border-bottom: none;}}
.pe-rule .name {{font-weight: 500;}}
.pe-rule .desc {{color: {t["muted"]}; font-size: 0.8rem;}}
.pe-rule .val {{text-align: right; white-space: nowrap; display: flex; flex-direction: column;
  align-items: flex-end; gap: 4px;}}

/* Sidebar */
[data-testid="stSidebar"] {{border-right: 1px solid {t["border"]};}}
[data-testid="stSidebarHeader"] {{height: 3.25rem; padding-bottom: 0;}}
[data-testid="stSidebarNav"]::before {{
  content: "ProphetEngine"; display: block; padding: 0 0 1.1rem 0; margin-top: -2.35rem;
  font-weight: 600; font-size: 1rem; letter-spacing: -0.015em; color: {t["text"]};}}
[data-testid="stSidebarNav"] {{padding-bottom: 0.5rem;}}
[data-testid="stNavSectionHeader"] {{margin-top: 0.9rem; padding-bottom: 0.15rem;}}
[data-testid="stNavSectionHeader"] p {{font-size: 0.7rem; font-weight: 500; letter-spacing: 0.08em;
  text-transform: uppercase; color: {t["muted"]};}}
[data-testid="stSidebarNavLink"] {{border-radius: 0; margin: 1px 0; padding-left: 0.85rem;
  border-left: 2px solid transparent; background: transparent !important;}}
[data-testid="stSidebarNavLink"] p {{color: {t["text2"]}; font-size: 0.9rem;}}
[data-testid="stSidebarNavLink"] [data-testid="stIconMaterial"] {{color: {t["muted"]} !important;
  font-size: 1.05rem;}}
[data-testid="stSidebarNavLink"]:hover p {{color: {t["text"]};}}
[data-testid="stSidebarNavLink"][aria-current="page"] {{border-left-color: {t["accent"]};}}
[data-testid="stSidebarNavLink"][aria-current="page"] p {{color: {t["text"]}; font-weight: 600;}}
[data-testid="stSidebarNavLink"][aria-current="page"] [data-testid="stIconMaterial"] {{color: {t["text"]} !important;}}
[data-testid="stSidebarNavSeparator"] {{border-color: {t["border"]};}}
.pe-side-label {{font-size: 0.7rem; font-weight: 500; letter-spacing: 0.08em; text-transform: uppercase;
  color: {t["muted"]}; margin: 0.25rem 0 0.4rem 0;}}
.pe-side-foot {{font-size: 0.75rem; color: {t["muted"]}; line-height: 1.6; padding-top: 0.75rem;
  border-top: 1px solid {t["border"]};}}
.pe-side-foot a {{color: {t["muted"]}; text-decoration: underline; text-underline-offset: 2px;}}
[data-testid="stSidebarUserContent"] {{padding-top: 0.5rem;}}

@media (max-width: 640px) {{
  .block-container {{padding-top: 3.5rem;}}
  .pe-hero h1 {{font-size: 1.45rem;}}
  .pe-verdict .big {{font-size: 2rem;}}
  [data-testid="stMetricValue"] {{font-size: 1.4rem;}}
}}
</style>
"""


def inject_css():
    st.markdown(build_css(), unsafe_allow_html=True)


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
        st.markdown('<div class="pe-side-label">Decision settings</div>', unsafe_allow_html=True)
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


def status_pill(tier, text=None):
    label = text or TIER_TEXT[tier]
    return f'<span class="pe-pill {tier}"><span class="dot"></span>{html.escape(label)}</span>'


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
