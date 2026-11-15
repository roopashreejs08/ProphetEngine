"""About & methodology: what the project is, how the models were built, limitations, team."""

import streamlit as st

from dashboard import config as C
from dashboard import ui

CONTRIBUTORS = [
    ("roopashreejs08", "Repository owner. Data pipeline for NASA C-MAPSS and AI4I 2020 (cleaning, encoding, "
                       "leakage-safe splits, 30-cycle windows), exploratory analysis and figures, the shared team "
                       "contract and the RUL metric functions."),
    ("saniyafatima07", "Trained and compared the Logistic Regression, Decision Tree and Random Forest models, tuned "
                       "the Random Forest alert threshold, produced the results files and wrote the project README."),
    ("samay43", "Built the Streamlit dashboard (PR #1) and this redesigned version."),
]


def render():
    ui.page_header("About & method", "What the project does, how the models were built and what to keep in mind "
                                     "when reading the results.")
    metadata = ui.load_metadata() or {}

    a, b = st.columns([1.3, 1], gap="large")
    with a:
        st.markdown("### The problem")
        st.markdown(
            "Unplanned machine breakdowns are expensive. **Predictive maintenance** uses sensor data to estimate "
            "which machines are likely to fail soon, so they can be serviced first. ProphetEngine frames this as a "
            "yes/no question for each reading: *is this machine about to fail?*"
        )
        st.markdown("### The data")
        st.markdown(
            f"- **{metadata.get('dataset', 'AI4I 2020 Predictive Maintenance Dataset')}** - 10,000 readings from "
            "milling machines, each with air and process temperature, rotational speed, torque, tool wear and a "
            "product quality type (L, M, H).\n"
            "- Target: **Machine failure** (339 failures, 3.39% - heavily imbalanced).\n"
            f"- Split: {metadata.get('train_rows', 8000):,} training / {metadata.get('validation_rows', 2000):,} "
            f"validation rows, stratified, random seed {metadata.get('random_state', 42)}.\n"
            "- Inputs: the five sensors plus one-hot machine type (8 features). Identifiers (UDI, Product ID) and "
            "the five failure-type flags (TWF, HDF, PWF, OSF, RNF) are **excluded** because they would leak the answer."
        )
        st.markdown("### The models")
        st.markdown(
            "Three scikit-learn classifiers, each trained with `class_weight='balanced'` so rare failures are not "
            "ignored:\n"
            "- **Logistic Regression** (with standard scaling) - a simple linear baseline.\n"
            "- **Decision Tree** - captures sharp rules but overfits.\n"
            "- **Random Forest** (200 trees) - best overall; PR-AUC 0.773, ROC-AUC 0.964.\n\n"
            "The Random Forest's alert threshold was then tuned using **cross-validation on the training data only**: "
            "the threshold with the best F1 while keeping recall of at least 75% was 0.23. On validation this raises "
            "failures caught from 29 to 50 of 68 (recall 42.6% -> 73.5%) at the cost of 30 extra false alarms."
        )
    with b:
        st.markdown("### How to read the dashboard")
        st.markdown(
            "- **Overview** - fleet health and headline results.\n"
            "- **Live monitor** - stream real or simulated readings through the models.\n"
            "- **What-if predictor** - one machine, every input adjustable, with explanations.\n"
            "- **Batch scoring** - upload a CSV, download a risk-ranked worklist.\n"
            "- **Performance** - metrics, threshold and cost trade-off, curves.\n"
            "- **Explainability** - which inputs matter and how.\n"
            "- **Data explorer** - filter and download the data."
        )
        st.markdown("### Limitations")
        st.markdown(
            "- AI4I 2020 is a **synthetic** dataset modelled on a real milling process, so results describe that "
            "simulation, not a specific factory.\n"
            "- All numbers are **validation** results; no separate test set has been evaluated yet.\n"
            "- Probabilities from class-balanced models over-state absolute risk - use them for ranking and "
            "thresholding.\n"
            "- Each reading is scored independently; the models have no memory of a machine's history.\n"
            "- Live monitor and simulator readings are synthetic; predictions on them are real model outputs."
        )
        st.markdown("### Team")
        for user, text in CONTRIBUTORS:
            st.markdown(f"- **[{user}](https://github.com/{user})** - {text}")

    st.markdown("### Artifacts used by this dashboard")
    artifacts = [
        C.TRAIN_PATH, C.VALIDATION_PATH, C.CLEANED_PATH, C.METADATA_PATH, C.METRICS_PATH, C.IMPORTANCE_PATH,
        *[C.RESULTS_DIR / f for f in dict.fromkeys(C.MODEL_FILES.values())],
    ]
    rows = [{"File": str(p.relative_to(C.REPO_ROOT)), "Status": "found" if p.exists() else "MISSING",
             "Size (KB)": round(p.stat().st_size / 1024, 1) if p.exists() else None} for p in artifacts]
    st.dataframe(rows, hide_index=True, width="stretch")
    st.markdown("### Reproduce the results")
    st.code("python src/data/preprocess_ai4i.py      # needs data/ai4i/ai4i2020.csv (raw file, not in git)\n"
            "python src/models/classical_ml.py       # trains the three models, writes results/\n"
            "python src/models/tune_random_forest.py # threshold tuning + feature importance\n"
            "streamlit run app.py                    # this dashboard", language="bash")
