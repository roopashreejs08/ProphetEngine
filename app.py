"""ProphetEngine - predictive maintenance dashboard.

Run from the repository root:

    streamlit run app.py

Every number on screen comes from the repository: the trained models in
``results/*.joblib``, the processed data in ``data/processed/`` and the saved
evaluation results in ``results/``. Nothing is hard-coded.
"""

import streamlit as st

from dashboard import config as C
from dashboard import ui
from dashboard.views import about, data_explorer, explain, live, overview, performance, predict

st.set_page_config(
    page_title="ProphetEngine | Predictive maintenance",
    page_icon=":material/precision_manufacturing:",
    layout="wide",
    initial_sidebar_state="auto",
)
ui.inject_css()

pages = {
    "Monitor": [
        st.Page(overview.render, title="Overview", icon=":material/space_dashboard:", url_path="overview", default=True),
        st.Page(live.render, title="Live monitor", icon=":material/sensors:", url_path="live"),
    ],
    "Predict": [
        st.Page(predict.render_single, title="What-if predictor", icon=":material/tune:", url_path="predict"),
        st.Page(predict.render_batch, title="Batch scoring", icon=":material/upload_file:", url_path="batch"),
    ],
    "Model": [
        st.Page(performance.render, title="Performance", icon=":material/monitoring:", url_path="performance"),
        st.Page(explain.render, title="Explainability", icon=":material/psychology:", url_path="explain"),
    ],
    "Data": [
        st.Page(data_explorer.render, title="Data explorer", icon=":material/table_chart:", url_path="data"),
    ],
    "Project": [
        st.Page(about.render, title="About & method", icon=":material/info:", url_path="about"),
    ],
}

navigation = st.navigation(pages, position="sidebar", expanded=True)
ui.sidebar_controls()
with st.sidebar:
    st.markdown(f'<div class="pe-side-foot">Data: AI4I 2020 &middot; Models: scikit-learn<br>'
                f'<a href="{C.REPO_URL}" target="_blank">Source code</a></div>', unsafe_allow_html=True)

navigation.run()
