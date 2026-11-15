"""What-if predictor (one machine) and batch scoring (upload a CSV)."""

import html

import numpy as np
import pandas as pd
import streamlit as st

from dashboard import charts, ui
from dashboard import config as C
from dashboard import data as D
from dashboard import scoring as S

PRESETS = {
    "Typical": None,  # training medians
    "Worn tool": {"Torque [Nm]": 62.0, "Tool wear [min]": 215, "Rotational speed [rpm]": 1300},
    "Overheating": {"Air temperature [K]": 302.6, "Process temperature [K]": 310.4,
                              "Rotational speed [rpm]": 1320, "Torque [Nm]": 52.0},
    "Power failure": {"Rotational speed [rpm]": 2800, "Torque [Nm]": 9.0},
}


def _key(column):
    return f"wi_{column}"


def _apply_preset(name, ranges):
    for column in C.SENSOR_COLUMNS:
        st.session_state[_key(column)] = float(ranges[column]["median"])
    for column, value in (PRESETS[name] or {}).items():
        st.session_state[_key(column)] = float(value)


def _load_real(validation, row_number, ranges):
    row = validation.iloc[row_number - 1]
    for column in C.SENSOR_COLUMNS:
        lo, hi = np.floor(ranges[column]["min"]), np.ceil(ranges[column]["max"])
        st.session_state[_key(column)] = float(np.clip(row[column], lo, hi))
    st.session_state["wi_type"] = D.machine_type_from_onehot(validation.iloc[[row_number - 1]]).iloc[0]


def render_single():
    label, threshold = ui.active_model()
    ui.page_header("What-if predictor", "Set the operating conditions of one machine and see, instantly, how likely "
                                        "each model thinks it is to fail - and why.")
    ranges = ui.load_ranges()
    model = ui.get_model(label)
    validation = ui.load_validation_with_modes()
    if ranges is None or model is None:
        ui.missing("The training data or the selected model", "python src/models/classical_ml.py")
        return

    if _key(C.SENSOR_COLUMNS[0]) not in st.session_state:
        _apply_preset("Worn tool", ranges)  # start on an interesting scenario for demos
    st.session_state.setdefault("wi_type", "L")

    inputs, output = st.columns([1, 1.45], gap="large")
    with inputs:
        with st.container(border=True):
            st.markdown("**Scenario**")
            with st.container(horizontal=True, gap="small"):
                for i, name in enumerate(PRESETS):
                    st.button(name, on_click=_apply_preset, args=(name, ranges), key=f"preset_{i}")
            if validation is not None:
                with st.expander("Load a real machine from the validation set"):
                    failures = np.flatnonzero(validation[C.TARGET].to_numpy() == 1) + 1
                    row_number = st.number_input("Machine # (row in ai4i_validation.csv)", 1, len(validation),
                                                 int(failures[0]) if len(failures) else 1)
                    st.button("Load machine", on_click=_load_real, args=(validation, int(row_number), ranges))
                    outcome = validation.iloc[int(row_number) - 1]
                    st.caption("This machine " + ("**failed** (" + D.failure_mode_label(outcome) + ")."
                                                  if outcome[C.TARGET] == 1 else "did **not** fail."))
            st.divider()
            machine_type = st.segmented_control("Machine type", C.MACHINE_TYPES, key="wi_type",
                                                format_func=lambda t: {"L": "Low (L)", "M": "Medium (M)", "H": "High (H)"}[t],
                                                required=True, help="Product quality variant of the machine.")
            values = {}
            for column in C.SENSOR_COLUMNS:
                r = ranges[column]
                step = C.SENSOR_STEP[column]
                values[column] = st.slider(
                    f"{C.SENSOR_LABEL[column]} ({C.SENSOR_UNIT[column]})",
                    min_value=float(np.floor(r["min"])), max_value=float(np.ceil(r["max"])), step=step,
                    key=_key(column), format="%.1f" if step < 1 else "%d",
                    help=f"Training range {r['min']:,.1f} - {r['max']:,.1f}; typical {r['median']:,.1f}.",
                )

    features = D.build_feature_row(values, machine_type)
    probability, contributions = S.local_contributions(model, values, machine_type, ranges)
    all_probs = ui.score_all(features)

    with output:
        top = st.columns([1, 1], gap="medium")
        with top[0]:
            ui.verdict_card(probability, threshold, label)
        with top[1]:
            with st.container(border=True):
                st.markdown("**Every model on this machine**")
                for name in C.BASE_MODELS:
                    p = all_probs[name]
                    t = threshold if name == label else C.DEFAULT_THRESHOLD
                    flag = "Alert" if p is not None and p[0] >= t else "OK"
                    st.progress(float(p[0]) if p is not None else 0.0,
                                text=f"{name}: {p[0]:.1%} ({flag})" if p is not None else f"{name}: unavailable")
        if contributions["contribution"].abs().max() < 0.005:
            st.success("Every reading is close to typical - no single sensor is pushing the risk up.",
                       icon=":material/check_circle:")
        else:
            charts.show(charts.contribution_bars(contributions), key="contrib")
        st.caption("Each bar compares this machine with the same machine where that one sensor is reset to a typical "
                   "value. Red bars push the risk up - they are the reasons this machine looks risky; blue bars "
                   "pull it down.")

        with st.container(border=True):
            st.markdown("**Engineering check** - the failure conditions documented for the AI4I dataset")
            rows = []
            for check in S.physics_checks(values, machine_type):
                pill = ui.status_pill("critical", "In danger zone") if check["triggered"] else \
                    ui.status_pill("healthy", "Within limits")
                rows.append(
                    f'<div class="pe-rule"><div><div class="name">{html.escape(check["name"])} ({check["code"]})</div>'
                    f'<div class="desc">{html.escape(check["rule"])}</div></div>'
                    f'<div class="val">{html.escape(check["value"])}<br>{pill}</div></div>'
                )
            st.markdown("".join(rows), unsafe_allow_html=True)
            st.caption("These published rules generated the dataset's failure labels. The model never sees them - "
                       "it learned the patterns from data - so this is an independent sanity check.")

    st.markdown("### Sensitivity")
    tab_line, tab_map = st.tabs(["One sensor at a time", "Speed x torque risk map"])
    with tab_line:
        column = st.segmented_control("Sensor", C.SENSOR_COLUMNS, format_func=C.SENSOR_LABEL.get,
                                      default="Tool wear [min]", key="wi_sens_col") or "Tool wear [min]"
        grid = np.linspace(ranges[column]["min"], ranges[column]["max"], 120)
        probs = S.sensitivity(model, values, machine_type, column, grid)
        charts.show(charts.sensitivity_line(grid, probs, values[column], probability, threshold, column, label),
                    key="sens_line")
    with tab_map:
        x_col, y_col = "Rotational speed [rpm]", "Torque [Nm]"
        xg = np.linspace(ranges[x_col]["min"], ranges[x_col]["max"], 70)
        yg = np.linspace(ranges[y_col]["min"], ranges[y_col]["max"], 70)
        z = S.probability_surface(model, x_col, xg, y_col, yg, values, machine_type)
        train = ui.load_train()
        failures = train[train[C.TARGET] == 1] if train is not None else None
        charts.show(charts.surface_heatmap(xg, yg, z, x_col, y_col, (values[x_col], values[y_col]), failures),
                    key="surface")
        st.caption("Colour = failure probability for every speed/torque combination, with the other inputs fixed at "
                   "the values above. Circles are real failures from the training data; the x is your machine.")


# ---------------------------------------------------------------------------
# Batch scoring
# ---------------------------------------------------------------------------
def render_batch():
    label, threshold = ui.active_model()
    ui.page_header("Batch scoring", "Upload readings for many machines and get a risk score for each one, "
                                    "ready to download as a maintenance worklist.")
    ranges = ui.load_ranges()
    if ui.get_model(label) is None:
        ui.missing("The selected model", "python src/models/classical_ml.py")
        return

    left, right = st.columns([2, 1], gap="large")
    with left:
        upload = st.file_uploader("Upload a CSV file", type=["csv"],
                                  help=f"Up to {D.MAX_UPLOAD_ROWS:,} rows. Column names are matched flexibly.")
    with right:
        st.markdown("**Accepted formats**")
        st.caption("Five sensor columns (air temperature, process temperature, rotational speed, torque, tool wear) "
                   "plus the machine type as `Type` (L/M/H) or `Type_H/Type_L/Type_M`. The raw AI4I file, the "
                   "processed splits and the live-feed format all work. Add `Machine failure` (0/1) to see accuracy.")
        st.download_button("Download template CSV", D.template_csv(), "prophetengine_template.csv", "text/csv",
                           icon=":material/download:")
        use_sample = st.toggle("Use the validation set as a sample", value=upload is None)

    if upload is not None:
        result = D.validate_upload(upload.getvalue(), ranges)
        source_name = upload.name
    elif use_sample:
        result = D.validate_upload(C.VALIDATION_PATH.read_bytes(), ranges) if C.VALIDATION_PATH.exists() else None
        source_name = "ai4i_validation.csv (sample)"
    else:
        st.info("Upload a CSV or switch on the sample to see batch results.", icon=":material/upload_file:")
        return
    if result is None:
        ui.missing("The validation sample", "python src/data/preprocess_ai4i.py")
        return
    for error in result.errors:
        st.error(error, icon=":material/error:")
    for warning in result.warnings:
        st.warning(warning, icon=":material/warning:")
    if not result.ok:
        return

    probs = ui.score_all(result.features)
    p = probs[label]
    tiers = S.risk_tiers(p, threshold)
    scored = result.original.copy()
    scored.insert(0, "failure_probability", np.round(p, 4))
    scored.insert(1, "status", [ui.TIER_TEXT[t] for t in tiers])
    scored.insert(2, "alert", (p >= threshold).astype(int))
    scored = scored.sort_values("failure_probability", ascending=False)

    st.markdown(f"### Results for `{source_name}`")
    c = st.columns(4)
    ui.kpi(c[0], "Machines scored", f"{len(p):,}", border=True)
    ui.kpi(c[1], "At risk", f"{int((tiers == 'critical').sum()):,}", f"{(tiers == 'critical').mean():.1%} of fleet",
                delta_color="off", border=True)
    ui.kpi(c[2], "Watch", f"{int((tiers == 'watch').sum()):,}", border=True)
    if result.labels is not None and result.labels.nunique() == 2:
        m = S.metrics_at_threshold(result.labels.to_numpy(), p, threshold)
        ui.kpi(c[3], "Failures caught", f"{m['tp']} of {m['failures']}",
                    f"precision {m['precision']:.0%}", delta_color="off", border=True)
    else:
        ui.kpi(c[3], "Median risk", f"{np.median(p):.1%}", border=True)

    tab_list, tab_chart, tab_quality = st.tabs(["Worklist", "Risk distribution", "Accuracy (if labelled)"])
    with tab_list:
        st.dataframe(scored.head(500), hide_index=True, width="stretch", height=420, column_config={
            "failure_probability": st.column_config.ProgressColumn("Failure probability", format="percent",
                                                                   min_value=0, max_value=1),
        })
        st.download_button("Download scored CSV", scored.to_csv(index=False).encode("utf-8"),
                           "prophetengine_scored.csv", "text/csv", type="primary", icon=":material/download:")
        if len(scored) > 500:
            st.caption("Showing the 500 riskiest rows; the download contains all of them.")
    with tab_chart:
        labels = result.labels.to_numpy() if result.labels is not None else np.zeros(len(p), dtype=int)
        charts.show(charts.risk_histogram(p, labels, threshold), key="batch_hist")
    with tab_quality:
        if result.labels is None or result.labels.nunique() < 2:
            st.info("Add a `Machine failure` column with 0/1 values (and at least one of each) to evaluate accuracy.")
        else:
            m = S.metrics_at_threshold(result.labels.to_numpy(), p, threshold)
            q = st.columns([1, 1.2], gap="large")
            with q[0]:
                charts.show(charts.confusion_heatmap(m), key="batch_cm")
            with q[1]:
                st.dataframe(pd.DataFrame({
                    "Metric": ["Accuracy", "Precision", "Recall", "F1-score", "PR-AUC", "ROC-AUC"],
                    "Value": [m["accuracy"], m["precision"], m["recall"], m["f1"], m["pr_auc"], m["roc_auc"]],
                }), hide_index=True, width="stretch",
                    column_config={"Value": st.column_config.NumberColumn(format="%.3f")})
