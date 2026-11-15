"""Data explorer: filter, summarise and download the project's datasets."""

import pandas as pd
import streamlit as st

from dashboard import charts, ui
from dashboard import config as C
from dashboard import data as D


def render():
    ui.page_header("Data explorer", "Browse the data behind the models: filter it, compare failures against normal "
                                    "operation and download any slice.")
    tab_ai4i, tab_cmapss = st.tabs(["AI4I 2020 - machine failures (used by the models)",
                                    "NASA C-MAPSS - engine degradation (prepared, no model yet)"])
    with tab_ai4i:
        _ai4i()
    with tab_cmapss:
        _cmapss()


def _ai4i():
    cleaned = ui.load_cleaned()
    train, validation = ui.load_train(), ui.load_validation()
    if cleaned is None or train is None or validation is None:
        ui.missing("The processed AI4I data", "python src/data/preprocess_ai4i.py")
        return
    train_m = D.attach_failure_modes(train, cleaned)
    valid_m = D.attach_failure_modes(validation, cleaned)
    frame = pd.concat([train_m.assign(Split="Train"), valid_m.assign(Split="Validation")], ignore_index=True)
    frame.insert(0, "Type", D.machine_type_from_onehot(frame))

    with st.container(border=True):
        f = st.columns([1.2, 1.1, 1, 1.4])
        splits = f[0].multiselect("Split", ["Train", "Validation"], default=["Train", "Validation"])
        types = f[1].multiselect("Machine type", C.MACHINE_TYPES, default=C.MACHINE_TYPES)
        outcome = f[2].selectbox("Outcome", ["All", "Failures only", "No failure"], key="dx_outcome")
        mode_filter = f[3].multiselect("Failure type", list(C.FAILURE_MODES), format_func=C.FAILURE_MODES.get,
                                       placeholder="Any")
        with st.expander("Sensor ranges"):
            rcols = st.columns(5)
            bounds = {}
            for col, column in zip(rcols, C.SENSOR_COLUMNS):
                lo, hi = float(frame[column].min()), float(frame[column].max())
                bounds[column] = col.slider(C.SENSOR_LABEL[column], lo, hi, (lo, hi), key=f"dx_{column}")

    mask = frame["Split"].isin(splits) & frame["Type"].isin(types)
    if outcome == "Failures only":
        mask &= frame[C.TARGET] == 1
    elif outcome == "No failure":
        mask &= frame[C.TARGET] == 0
    if mode_filter:
        mask &= frame[mode_filter].eq(1).any(axis=1)
    for column, (lo, hi) in bounds.items():
        mask &= frame[column].between(lo, hi)
    view = frame[mask]

    k = st.columns(4)
    ui.kpi(k[0], "Records", f"{len(view):,}", f"of {len(frame):,}", delta_color="off", border=True)
    ui.kpi(k[1], "Failures", f"{int(view[C.TARGET].sum()):,}", border=True)
    ui.kpi(k[2], "Failure rate", ui.fmt_pct(view[C.TARGET].mean() if len(view) else float("nan"), 2), border=True)
    ui.kpi(k[3], "Machine types", ", ".join(sorted(view["Type"].unique())) or "-", border=True)
    if view.empty:
        st.warning("No records match these filters.")
        return

    t1, t2, t3, t4 = st.tabs(["Distributions", "Relationships", "Summary statistics", "Records"])
    with t1:
        column = st.segmented_control("Sensor", C.SENSOR_COLUMNS, format_func=C.SENSOR_LABEL.get,
                                      default="Torque [Nm]", key="dx_dist") or "Torque [Nm]"
        if view[C.TARGET].nunique() == 2:
            charts.show(charts.distribution(view, column, C.TARGET), key="dist")
        else:
            st.caption("Both outcomes are needed to compare distributions - widen the Outcome filter.")
        counts = {code: int(cleaned[code].sum()) for code in C.FAILURE_MODES}
        charts.show(charts.failure_mode_counts(counts), key="mode_counts")
        st.caption(f"{int(cleaned[C.TARGET].sum())} failures in {len(cleaned):,} records "
                   f"({cleaned[C.TARGET].mean():.2%}). These failure-type columns are excluded from model inputs "
                   "to prevent leakage.")
    with t2:
        c = st.columns(2)
        x_col = c[0].selectbox("X axis", C.SENSOR_COLUMNS, index=2)
        y_col = c[1].selectbox("Y axis", C.SENSOR_COLUMNS, index=3)
        charts.show(charts.scatter(view, x_col, y_col, C.TARGET), key="scatter")
        st.caption("Failures (red) cluster at the edges of the speed-torque band - very high torque at low speed "
                   "(overstrain, heat) and very high speed at low torque (power).")
        st.markdown("**Correlation matrix**")
        charts.show(charts.correlation(view[C.SENSOR_COLUMNS + [C.TARGET]]), key="corr")
    with t3:
        st.markdown("**All selected records**")
        st.dataframe(view[C.SENSOR_COLUMNS].describe().T.round(2), width="stretch")
        st.markdown("**By outcome**")
        st.dataframe(view.groupby(C.TARGET)[C.SENSOR_COLUMNS].mean().rename(index={0: "No failure", 1: "Failure"})
                     .round(2), width="stretch")
        st.markdown("**By machine type**")
        st.dataframe(view.groupby("Type").agg(records=(C.TARGET, "size"), failures=(C.TARGET, "sum"),
                                              failure_rate=(C.TARGET, "mean")),
                     width="stretch", column_config={"failure_rate": st.column_config.NumberColumn(format="percent")})
    with t4:
        show_cols = ["Split", "Type"] + C.SENSOR_COLUMNS + [C.TARGET] + list(C.FAILURE_MODES)
        st.dataframe(view[show_cols].head(1000), hide_index=True, width="stretch", height=420)
        st.download_button("Download filtered data (CSV)", view[show_cols].to_csv(index=False).encode("utf-8"),
                           "ai4i_filtered.csv", "text/csv", icon=":material/download:")
        if len(view) > 1000:
            st.caption("Showing the first 1,000 rows; the download contains all of them.")


def _cmapss():
    st.info("The team has preprocessed NASA's C-MAPSS turbofan data for a future **remaining-useful-life** model "
            "(how many cycles an engine has left). No RUL model has been trained yet - `shared/results.csv` is still "
            "empty - so this tab shows the prepared data only.", icon=":material/construction:")
    c = st.columns([1, 1, 2])
    subset = c[0].selectbox("Subset", C.CMAPSS_SUBSETS, help="FD001: one operating condition, one fault mode. "
                                                              "FD003: one condition, two fault modes.")
    split = c[1].selectbox("Split", ["train", "validation"])
    frame = ui.load_cmapss(subset, split)
    if frame is None:
        ui.missing(f"{subset} {split} data", "python src/data/preprocess_cmapss.py")
        return
    engines = sorted(frame["engine_id"].unique())
    sensors_all = [col for col in frame.columns if col.startswith("sensor_")]
    chosen = c[2].multiselect("Engines (up to 4)", engines, default=engines[:3], max_selections=4)
    sensors = st.multiselect("Sensors", sensors_all, default=[s for s in ["sensor_4", "sensor_11", "sensor_15"]
                                                              if s in sensors_all] or sensors_all[:3])
    lifetimes = frame.groupby("engine_id")["cycle"].max()
    k = st.columns(4)
    ui.kpi(k[0], "Engines", f"{len(engines)}", border=True)
    ui.kpi(k[1], "Rows (cycles)", f"{len(frame):,}", border=True)
    ui.kpi(k[2], "Median life", f"{lifetimes.median():.0f} cycles", border=True)
    ui.kpi(k[3], "Sensors kept", f"{len(sensors_all)}", help="Low-variance sensors were removed during preprocessing.",
                border=True)
    if chosen and sensors:
        charts.show(charts.engine_trends(frame, sensors, chosen), key="cmapss_trends")
        st.caption("Each line is one engine from first cycle to failure. Drifting sensors are the degradation signal "
                   "a future RUL model would learn from.")
    charts.show(charts.rul_histogram(frame), key="rul_hist")
