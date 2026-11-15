"""Model performance: saved results, threshold trade-offs, curves and failure types."""

import pandas as pd
import streamlit as st

from dashboard import charts, ui
from dashboard import config as C
from dashboard import scoring as S

METRIC_COLUMNS = ["Accuracy", "Precision", "Recall", "F1-score", "Average Precision (PR-AUC)", "ROC-AUC"]


def _results_table(saved, y):
    """Saved results for the three models plus the tuned Random Forest recomputed at 0.23."""
    rows = saved[["Model"] + METRIC_COLUMNS].copy()
    rows["Threshold"] = C.DEFAULT_THRESHOLD
    rows["Source"] = "results/classical_ml_metrics.csv"
    rf = ui.validation_probabilities("Random Forest")
    if rf is not None:
        m = S.metrics_at_threshold(y, rf, C.TUNED_THRESHOLD)
        rows.loc[len(rows)] = {
            "Model": C.TUNED_LABEL, "Accuracy": m["accuracy"], "Precision": m["precision"], "Recall": m["recall"],
            "F1-score": m["f1"], "Average Precision (PR-AUC)": m["pr_auc"], "ROC-AUC": m["roc_auc"],
            "Threshold": C.TUNED_THRESHOLD, "Source": "recomputed from random_forest.joblib",
        }
    rows["Model"] = pd.Categorical(rows["Model"], C.MODEL_ORDER, ordered=True)
    return rows.sort_values("Model").reset_index(drop=True)


def render():
    label, threshold = ui.active_model()
    ui.page_header("Model performance", "How well each model separates failing machines from healthy ones on the "
                                        "2,000 validation machines held out from training.")
    saved = ui.load_metrics()
    validation = ui.load_validation_with_modes()
    if saved is None or validation is None:
        ui.missing("The saved metrics or validation data", "python src/models/classical_ml.py")
        return
    y = validation[C.TARGET].to_numpy(dtype=int)

    st.markdown("### Model comparison")
    table = _results_table(saved, y)
    best = table.loc[table["Average Precision (PR-AUC)"].idxmax(), "Model"]
    st.dataframe(table, hide_index=True, width="stretch", column_config={
        "Model": st.column_config.TextColumn(width="medium"),
        **{col: st.column_config.NumberColumn(col.replace("Average Precision (PR-AUC)", "PR-AUC"), format="%.3f")
           for col in METRIC_COLUMNS},
        "Average Precision (PR-AUC)": st.column_config.ProgressColumn("PR-AUC (main metric)", format="%.3f",
                                                                      min_value=0, max_value=1),
        "Threshold": st.column_config.NumberColumn(format="%.2f"),
        "Source": None,
    })
    st.caption("The first three rows are read from `results/classical_ml_metrics.csv`; the tuned row is recomputed "
               "from `random_forest.joblib` at threshold 0.23.")
    st.caption(f"Best ranking quality: **{best}**. Accuracy looks high for every model because only 3.4% of machines "
               "fail - a model that never raised an alert would still score 96.6% - so the team judges models on "
               "precision, recall, F1 and PR-AUC instead.")
    metric_choice = st.segmented_control("Metrics to compare", METRIC_COLUMNS, default=METRIC_COLUMNS[1:5],
                                         selection_mode="multi", key="perf_metrics",
                                         format_func=lambda m: m.replace("Average Precision (PR-AUC)", "PR-AUC"))
    if metric_choice:
        chart_frame = table.copy()
        chart_frame["Model"] = chart_frame["Model"].astype(str)
        charts.show(charts.metric_bars(chart_frame, [m for m in METRIC_COLUMNS if m in metric_choice]),
                    key="metric_bars")

    st.markdown(f"### Threshold explorer - {label}")
    probabilities = ui.validation_probabilities(label)
    if probabilities is None:
        ui.missing(f"The {label} model")
        return
    m = S.metrics_at_threshold(y, probabilities, threshold)
    st.caption("The model outputs a probability; the alert threshold turns it into a yes/no decision. "
               "Move the threshold in the sidebar to see the trade-off live.")
    k = st.columns(5)
    ui.kpi(k[0], "Threshold", f"{threshold:.2f}", border=True)
    ui.kpi(k[1], "Recall", ui.fmt_pct(m["recall"]), f"{m['tp']} of {m['failures']} failures", delta_color="off",
                border=True)
    ui.kpi(k[2], "Precision", ui.fmt_pct(m["precision"]), f"{m['flagged']} alerts", delta_color="off", border=True)
    ui.kpi(k[3], "F1-score", f"{m['f1']:.3f}", border=True)
    ui.kpi(k[4], "Accuracy", ui.fmt_pct(m["accuracy"], 2), border=True)

    sweep = S.threshold_sweep(y, probabilities)
    left, right = st.columns([1.5, 1], gap="large")
    with left:
        charts.show(charts.threshold_tradeoff(sweep, threshold), key="tradeoff")
    with right:
        st.markdown("**Outcomes at this threshold**")
        charts.show(charts.confusion_heatmap(m), key="cm")

    with st.container(border=True):
        st.markdown("**What does a mistake cost?** Put rough costs on each kind of error to find the cheapest threshold.")
        c = st.columns([1, 1, 2], vertical_alignment="bottom")
        missed = c[0].number_input("Cost of a missed failure", min_value=0, value=10_000, step=500,
                                   help="Unplanned downtime, repairs, scrap.")
        false_alarm = c[1].number_input("Cost of a false alarm", min_value=0, value=500, step=100,
                                        help="An unnecessary inspection.")
        costed = S.add_costs(sweep, missed, false_alarm)
        best_row = S.best_threshold(costed)
        current_cost = m["fn"] * missed + m["fp"] * false_alarm
        with c[2]:
            st.markdown(
                f"Lowest total cost at threshold **{best_row['threshold']:.2f}**: {int(best_row['fn'])} missed failures "
                f"and {int(best_row['fp'])} false alarms = **{best_row['cost']:,.0f}**. "
                f"Your current threshold costs **{current_cost:,.0f}**."
            )
        charts.show(charts.cost_curve(costed, threshold, best_row), key="cost")

    st.markdown("### Ranking quality across all thresholds")
    curve_map_pr, curve_map_roc, calib = {}, {}, {}
    for name in C.BASE_MODELS:
        p = ui.validation_probabilities(name)
        if p is None:
            continue
        cv = S.curves(y, p)
        mm = S.metrics_at_threshold(y, p, 0.5)
        curve_map_pr[name] = (cv, mm["pr_auc"])
        curve_map_roc[name] = (cv, mm["roc_auc"])
        calib[name] = S.reliability(y, p)
    a, b = st.columns(2, gap="large")
    with a:
        charts.show(charts.pr_curves(curve_map_pr, y.mean()), key="pr")
    with b:
        charts.show(charts.roc_curves(curve_map_roc), key="roc")
    st.caption("Precision-recall is the more honest view for rare failures: further up and to the right is better. "
               "The Random Forest dominates the other two models almost everywhere.")

    a, b = st.columns(2, gap="large")
    with a:
        modes = S.recall_by_failure_mode(validation, probabilities, threshold)
        if not modes.empty:
            charts.show(charts.failure_mode_bars(modes), key="perf_modes")
            st.caption("Tool-wear failures are the hardest to anticipate: in the dataset they happen at a random "
                       "wear time between 200 and 240 minutes, so the sensors give little warning.")
    with b:
        charts.show(charts.reliability_chart(calib), key="calib")
        st.caption("Points on the diagonal mean a predicted 30% really fails about 30% of the time. Models trained "
                   "with class balancing (all three here) tend to over-state risk, so treat outputs as risk scores "
                   "for ranking rather than exact odds.")

    with st.expander("Original confusion-matrix images from training (results/*.png)"):
        cols = st.columns(3)
        for col, (name, filename) in zip(cols, C.CONFUSION_PNGS.items()):
            path = C.RESULTS_DIR / filename
            if path.exists():
                col.image(str(path), caption=f"{name} (threshold 0.5)")
            else:
                col.caption(f"{filename} not found")
    st.caption("Note: these images and the saved table use scikit-learn's `predict` (a probability of exactly 0.50 "
               "counts as 'no failure'); the dashboard flags `probability >= threshold`, the rule used for threshold "
               "tuning. For the plain Random Forest at 0.50 this changes one machine (30 vs 29 failures caught).")
