"""Explainability: which inputs the models rely on, and how risk responds to each."""

import streamlit as st

from dashboard import charts, ui
from dashboard import config as C


def render():
    label, _ = ui.active_model()
    ui.page_header("Explainability", "Which measurements drive the predictions, and how predicted risk changes as "
                                     "each one moves.")

    a, b = st.columns(2, gap="large")
    with a:
        importance = ui.load_importance()
        if importance is None:
            ui.missing("Feature importance", "python src/models/tune_random_forest.py")
        else:
            names = [C.SENSOR_LABEL.get(f, f.replace("Type_", "Machine type ")) for f in importance["Feature"]]
            charts.show(charts.importance_bars(names, importance["Importance"].tolist(),
                                               "Random Forest built-in importance", "Share of total importance"),
                        key="imp_saved")
            st.caption("From `results/random_forest_feature_importance.csv`: how much each input reduced impurity "
                       "across the forest's 200 trees.")
    with b:
        result = ui.permutation_importance(label)
        if result is None:
            ui.missing(f"The {label} model")
        else:
            baseline, frame = result
            charts.show(charts.importance_bars(frame["feature"].tolist(), frame["drop"].tolist(),
                                               f"Permutation importance - {label}",
                                               "Drop in PR-AUC when the input is shuffled", fmt=".3f",
                                               errors=frame["std"].tolist()),
                        key="imp_perm")
            st.caption(f"Computed live on the validation set (baseline PR-AUC {baseline:.3f}, 5 shuffles each). "
                       "A bigger drop means the model leans on that input more.")

    st.info("Both views put **torque and rotational speed** on top - which matches the physics of the failure types "
            "(power = speed x torque; overstrain = tool wear x torque). Shuffling also shows the forest leans on "
            "**air temperature** and **tool wear**, which it needs to spot heat-dissipation and overstrain failures. "
            "Importance shows what the model uses, not proof of cause.", icon=":material/lightbulb:")

    st.markdown("### How risk responds to each sensor")
    column = st.segmented_control("Sensor", C.SENSOR_COLUMNS, format_func=C.SENSOR_LABEL.get,
                                  default="Torque [Nm]", key="pdp_col") or "Torque [Nm]"
    series = {}
    for name in C.BASE_MODELS:
        result = ui.partial_dependence(name, column)
        if result is not None:
            series[name] = result
    if series:
        charts.show(charts.pdp_lines(series, column), key="pdp")
        st.caption("Partial dependence: every validation machine is re-scored with this sensor set to each value, "
                   "and the predictions are averaged. The tree models capture sharp danger zones (very low or very "
                   "high torque, tool wear above ~200 min) that the linear model can only approximate.")
    st.caption("Want to see this for a single machine? Use the **What-if predictor** page.")
