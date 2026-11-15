"""Overview: fleet health and headline results at the active model and threshold."""

import numpy as np
import pandas as pd
import streamlit as st

from dashboard import charts, ui
from dashboard import config as C
from dashboard import data as D
from dashboard import scoring as S


def render():
    label, threshold = ui.active_model()
    ui.hero(
        "Spot machine failures before they happen",
        "Machine-learning models trained on 10,000 sensor readings from industrial milling machines "
        "estimate the chance that each machine is about to fail, so maintenance can be scheduled "
        "before a breakdown instead of after it.",
        chips=("AI4I 2020 dataset", "3 trained models", "2,000 held-out machines", f"Active: {label} @ {threshold:.2f}"),
    )

    validation = ui.load_validation_with_modes()
    probabilities = ui.validation_probabilities(label)
    if validation is None or probabilities is None:
        ui.missing("The validation data or the trained model",
                   "python src/data/preprocess_ai4i.py\npython src/models/classical_ml.py")
        return

    y = validation[C.TARGET].to_numpy(dtype=int)
    m = S.metrics_at_threshold(y, probabilities, threshold)
    healthy_total = int((y == 0).sum())

    st.markdown("### Headline results")
    st.caption(f"{label} scoring the {len(y):,} validation machines the models never saw during training. "
               "Change the model or alert threshold in the sidebar and every number updates.")
    cols = st.columns(5)
    ui.kpi(cols[0], "Failures caught", f"{m['tp']} of {m['failures']}", f"{m['recall']:.0%} recall",
                   delta_color="off", border=True,
                   help="Real failures in the validation set that the model flagged in advance.")
    ui.kpi(cols[1], "False alarms", f"{m['fp']:,}", f"{m['fp'] / healthy_total:.1%} of healthy machines",
                   delta_color="off", border=True, help="Healthy machines that were flagged anyway.")
    ui.kpi(cols[2], "Alert precision", ui.fmt_pct(m["precision"]), "alerts that were real",
                   delta_color="off", border=True)
    ui.kpi(cols[3], "PR-AUC", f"{m['pr_auc']:.3f}", f"random guess: {y.mean():.3f}", delta_color="off",
                   border=True, help="Average precision across all thresholds - the project's main ranking metric "
                                     "because failures are rare (3.4%).")
    ui.kpi(cols[4], "ROC-AUC", f"{m['roc_auc']:.3f}", "ranking quality",
                   delta_color="off", border=True)

    st.markdown("### Fleet health")
    tiers = S.risk_tiers(probabilities, threshold)
    counts = {t: int((tiers == t).sum()) for t in ("healthy", "watch", "critical")}
    st.markdown(
        '<div class="pe-pills">' + "".join([
            ui.status_pill("critical", f"At risk: {counts['critical']:,}"),
            ui.status_pill("watch", f"Watch: {counts['watch']:,}"),
            ui.status_pill("healthy", f"Healthy: {counts['healthy']:,}"),
        ]) + "</div>",
        unsafe_allow_html=True,
    )
    st.caption(f"At risk = probability at or above {threshold:.2f}. Watch = at least half the threshold "
               f"({threshold / 2:.2f}). Healthy = below that.")
    charts.show(charts.tier_bar(counts), key="tier_bar")

    left, right = st.columns([1.35, 1], gap="large")
    with left:
        charts.show(charts.risk_histogram(probabilities, y, threshold), key="risk_hist")
        st.caption("Red bars to the right of the dotted line are failures the model catches; red bars to the left "
                   "are misses. Grey bars to the right are false alarms.")
    with right:
        modes = S.recall_by_failure_mode(validation, probabilities, threshold)
        if modes.empty:
            st.info("Failure-type labels are not available.")
        else:
            charts.show(charts.failure_mode_bars(modes), key="mode_bars")
            weakest = modes.sort_values("recall").iloc[0]
            strongest = modes.sort_values("recall").iloc[-1]
            st.caption(f"Strongest on **{strongest['mode'].lower()}** ({strongest['caught']}/{strongest['failures']}); "
                       f"weakest on **{weakest['mode'].lower()}** ({weakest['caught']}/{weakest['failures']}). "
                       "Failure types are used for this breakdown only - the model never sees them.")

    st.markdown("### Highest-risk machines")
    table = validation[C.SENSOR_COLUMNS].copy()
    table.insert(0, "Type", D.machine_type_from_onehot(validation))
    table.insert(0, "Risk", probabilities)
    table["Status"] = [ui.TIER_TEXT[t] for t in tiers]
    table["Actually failed?"] = np.where(y == 1, "Yes", "No")
    table["Failure type"] = validation.apply(D.failure_mode_label, axis=1)
    table.index = pd.RangeIndex(1, len(table) + 1, name="Machine #")
    top = table.sort_values("Risk", ascending=False).head(12)
    st.dataframe(
        top,
        width="stretch",
        column_config={
            "Risk": st.column_config.ProgressColumn("Failure probability", format="percent", min_value=0, max_value=1),
            "Air temperature [K]": st.column_config.NumberColumn("Air temp (K)", format="%.1f"),
            "Process temperature [K]": st.column_config.NumberColumn("Process temp (K)", format="%.1f"),
            "Rotational speed [rpm]": st.column_config.NumberColumn("Speed (rpm)", format="%d"),
            "Torque [Nm]": st.column_config.NumberColumn("Torque (Nm)", format="%.1f"),
            "Tool wear [min]": st.column_config.NumberColumn("Tool wear (min)", format="%d"),
        },
    )
    hits = int((top["Actually failed?"] == "Yes").sum())
    st.caption(f"{hits} of the {len(top)} machines the model ranks as riskiest really did fail. "
               "Machine # is the row number in data/processed/ai4i_validation.csv.")

    st.markdown("### What this means")
    st.markdown(
        f"- Out of **{m['failures']} real failures**, the model warns about **{m['tp']}** "
        f"and misses **{m['fn']}**.\n"
        f"- To do that it raises **{m['flagged']} alerts**, of which **{m['tp']}** are real - "
        f"so roughly **1 in {max(1, round(m['flagged'] / max(m['tp'], 1)))}** alerts leads to a genuine failure.\n"
        f"- Inspecting just the flagged **{m['flagged'] / len(y):.1%}** of machines would catch "
        f"**{m['recall']:.0%}** of failures.\n"
        "- Use the **Performance** page to pick the threshold that best fits the cost of a missed failure "
        "versus a false alarm."
    )
