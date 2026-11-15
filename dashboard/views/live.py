"""Live monitor: stream readings through the trained models in real time."""

import time
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from dashboard import charts, ui
from dashboard import config as C
from dashboard import data as D
from dashboard import scoring as S
from dashboard import simulator as SIM

SOURCES = {
    "replay": "Replay real machines",
    "sim": "Simulated machine",
    "feed": "External feed",
}
SPEEDS = {"Slow (2 s)": 2.0, "Normal (1 s)": 1.0, "Fast (0.5 s)": 0.5}
HISTORY_LIMIT = 600


def _state():
    ss = st.session_state
    ss.setdefault("live_source", "replay")
    ss.setdefault("live_running", False)
    ss.setdefault("live_history", {key: [] for key in SOURCES})
    ss.setdefault("live_replay_pos", 0)
    ss.setdefault("live_feed_pos", 0)
    return ss


def _score(values, machine_type):
    features = D.build_feature_row(values, machine_type)
    start = time.perf_counter()
    probs = {}
    for label, p in ui.score_all(features).items():
        probs[label] = None if p is None else float(p[0])
    return probs, (time.perf_counter() - start) * 1000


def _append(source, values, machine_type, actual=None, timestamp=None):
    probs, ms = _score(values, machine_type)
    history = st.session_state["live_history"][source]
    history.append({
        "time": timestamp or datetime.now(timezone.utc),
        "values": values, "type": machine_type, "probs": probs, "ms": ms, "actual": actual,
    })
    del history[:-HISTORY_LIMIT]


def _tick(source):
    """Produce and score the next reading for the active source."""
    ss = st.session_state
    if source == "replay":
        validation = ui.load_validation_with_modes()
        if validation is None:
            return
        pos = ss["live_replay_pos"] % len(validation)
        row = validation.iloc[pos]
        values = {c: float(row[c]) for c in C.SENSOR_COLUMNS}
        machine_type = D.machine_type_from_onehot(validation.iloc[[pos]]).iloc[0]
        _append("replay", values, machine_type, actual=int(row[C.TARGET]))
        ss["live_replay_pos"] = pos + 1
    elif source == "sim":
        stats = ui.load_sim_stats()
        if stats is None:
            return
        if "live_sim" not in ss:
            ss["live_sim"] = SIM.new_state(stats)
        values = SIM.step(ss["live_sim"], stats)
        _append("sim", values, ss["live_sim"]["machine_type"])
    elif source == "feed":
        parsed = D.parse_stream_csv(C.STREAM_PATH)
        ss["live_feed_status"] = parsed
        if parsed["frame"] is None:
            return
        frame = parsed["frame"]
        for _, row in frame.iloc[ss["live_feed_pos"]:].iterrows():
            values = {c: float(row[c]) for c in C.SENSOR_COLUMNS}
            _append("feed", values, row["machine_type"], timestamp=row["timestamp"].to_pydatetime())
        ss["live_feed_pos"] = len(frame)


def _append_demo_feed_row():
    """Write one simulated reading to data/live/stream.csv (handy for demos)."""
    stats = ui.load_sim_stats()
    if stats is None:
        return
    ss = st.session_state
    if "feed_sim" not in ss:
        ss["feed_sim"] = SIM.new_state(stats, machine_type="L", start_wear=150, seed=7)
    values = SIM.step(ss["feed_sim"], stats, wear_per_tick=4)
    C.LIVE_DIR.mkdir(parents=True, exist_ok=True)
    new_file = not C.STREAM_PATH.exists() or C.STREAM_PATH.stat().st_size == 0
    row = pd.DataFrame([{
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        **{k: round(values[v], 2) for k, v in C.STREAM_SENSOR_MAP.items()},
        "machine_type": ss["feed_sim"]["machine_type"],
    }], columns=C.STREAM_COLUMNS)
    row.to_csv(C.STREAM_PATH, mode="a", header=new_file, index=False)


def render():
    ss = _state()
    label, threshold = ui.active_model()
    ui.page_header("Live monitor", "Readings stream in one at a time and every model scores each one the moment "
                                   "it arrives. Press Start, then watch the risk line react.")

    if ui.load_train() is None or ui.get_model("Random Forest") is None:
        ui.missing("The training data or trained models", "python src/models/classical_ml.py")
        return

    top = st.columns([2.6, 1.1, 0.9, 0.9], vertical_alignment="bottom")
    with top[0]:
        source = st.segmented_control("Data source", list(SOURCES), format_func=SOURCES.get, key="live_source", required=True)
    with top[1]:
        speed = st.selectbox("Update speed", list(SPEEDS), index=1, key="live_speed")
    with top[2]:
        if st.button("Pause" if ss["live_running"] else "Start", type="primary", width="stretch",
                     icon=":material/pause:" if ss["live_running"] else ":material/play_arrow:"):
            ss["live_running"] = not ss["live_running"]
            st.rerun()
    with top[3]:
        if st.button("Reset", width="stretch", icon=":material/restart_alt:"):
            ss["live_history"][source] = []
            ss["live_running"] = False
            ss.pop("live_sim", None)
            ss["live_replay_pos"] = 0
            ss["live_feed_pos"] = 0
            st.rerun()

    if source == "replay":
        st.caption("Streams the 2,000 held-out validation machines in file order - real readings with known outcomes, "
                   "so you can see whether each alert was right. Failures are marked with a red x.")
    elif source == "sim":
        _sim_controls(ss)
    else:
        _feed_controls()

    interval = SPEEDS[speed] if ss["live_running"] else None

    @st.fragment(run_every=interval)
    def stream():
        if ss["live_running"]:
            _tick(source)
        _render_stream(source, label, threshold)

    stream()


def _toggle_fault(key):
    sim = st.session_state["live_sim"]
    sim["fault"] = None if sim["fault"] == key else key


def _new_tool():
    st.session_state["live_sim"]["wear"] = 0.0
    st.session_state["live_sim"]["fault"] = None


def _sim_controls(ss):
    stats = ui.load_sim_stats()
    st.caption("A synthetic machine whose normal behaviour is learned from the training data. Inject a fault and "
               "watch the models respond; the readings are simulated, the predictions are real.")
    cols = st.columns([1, 1, 1, 1, 1], vertical_alignment="bottom")
    with cols[0]:
        machine_type = st.selectbox("Machine type", C.MACHINE_TYPES, index=1, format_func=C.MACHINE_TYPE_LABEL.get)
    if "live_sim" not in ss or ss["live_sim"]["machine_type"] != machine_type:
        ss["live_sim"] = SIM.new_state(stats, machine_type=machine_type)
    sim = ss["live_sim"]
    for i, (key, name) in enumerate(SIM.FAULTS.items(), start=1):
        with cols[i]:
            active = sim["fault"] == key
            st.button(("Stop: " if active else "") + name.split(" (")[0], key=f"fault_{key}", width="stretch",
                      help=name, type="primary" if active else "secondary", icon=":material/bolt:",
                      on_click=_toggle_fault, args=(key,))
    with cols[4]:
        st.button("New tool (wear = 0)", width="stretch", icon=":material/build:", on_click=_new_tool)


def _feed_controls():
    st.caption(f"Scores new rows appended to `{C.STREAM_PATH.relative_to(C.REPO_ROOT)}` by any external process "
               "(format in data/live/README.md). Use the button to append a demo reading.")
    if st.button("Append a demo reading to the feed", icon=":material/add:"):
        _append_demo_feed_row()
    status = st.session_state.get("live_feed_status")
    if status and status["status"] != "ok":
        messages = {
            "missing": "The feed file does not exist yet. Append a demo reading or create it as described in data/live/README.md.",
            "empty": "The feed file has no readings yet.",
            "unreadable": "The feed file could not be read.",
            "invalid_columns": "The feed file has the wrong columns. " + " ".join(status["warnings"]),
            "no_valid_rows": "No valid rows in the feed. " + " ".join(status["warnings"]),
        }
        st.warning(messages.get(status["status"], "Feed unavailable."), icon=":material/warning:")
    elif status and status["warnings"]:
        st.warning(" ".join(status["warnings"]), icon=":material/warning:")
    if status and status.get("newest") is not None:
        age = (datetime.now(timezone.utc) - status["newest"].to_pydatetime()).total_seconds()
        if age > C.STREAM_STALE_SECONDS:
            st.caption(f"Newest feed reading is {age:,.0f} s old - the feed looks stale.")


def _render_stream(source, label, threshold):
    history = st.session_state["live_history"][source]
    if not history:
        st.info("No readings yet. Press **Start** to begin streaming.", icon=":material/play_circle:")
        return

    latest = history[-1]
    p = latest["probs"].get(label)
    if p is None:
        st.error(f"The {label} model could not score this reading.")
        return
    tier = S.risk_tier(p, threshold)
    probs = [h["probs"][label] for h in history]
    alerts = sum(1 for i, x in enumerate(probs) if x >= threshold and (i == 0 or probs[i - 1] < threshold))

    left, right = st.columns([1, 2.2], gap="large")
    with left:
        with st.container(border=True):
            st.markdown(ui.status_pill(tier), unsafe_allow_html=True)
            charts.show(charts.gauge(p, threshold, height=200), key=f"gauge_{source}")
            extra = ""
            if latest["actual"] is not None:
                extra = " | actual outcome: **" + ("failed" if latest["actual"] else "no failure") + "**"
            st.caption(f"Reading #{len(history):,} | type {latest['type']} | scored in {latest['ms']:.0f} ms{extra}")
    with right:
        times = [h["time"] for h in history][-120:]
        actual = [h["actual"] for h in history][-120:] if source == "replay" else None
        charts.show(charts.live_probability(times, probs[-120:], threshold, label, actual), key=f"line_{source}")
        stats = st.columns(3)
        stats[0].metric("Readings scored", f"{len(history):,}")
        stats[1].metric("Alerts raised", f"{alerts:,}", help="Times the risk crossed the threshold from below.")
        if source == "replay":
            seen = [h for h in history if h["actual"] is not None]
            caught = sum(1 for h in seen if h["actual"] == 1 and h["probs"][label] >= threshold)
            failed = sum(1 for h in seen if h["actual"] == 1)
            stats[2].metric("Failures caught so far", f"{caught} of {failed}")
        else:
            stats[2].metric("Peak risk", f"{max(probs):.1%}")

    st.markdown("##### Sensor readings")
    short = {"Air temperature [K]": "Air temp", "Process temperature [K]": "Process temp",
             "Rotational speed [rpm]": "Speed", "Torque [Nm]": "Torque", "Tool wear [min]": "Tool wear"}
    cols = st.columns(5)
    recent = history[-40:]
    for col, column in zip(cols, C.SENSOR_COLUMNS):
        series = [h["values"][column] for h in recent]
        delta = series[-1] - series[-2] if len(series) > 1 else None
        col.metric(short[column], ui.sensor_text(column, series[-1]),
                   None if delta is None else f"{delta:+.1f}", delta_color="off", delta_arrow="off", border=True,
                   chart_data=series, chart_type="line")

    a, b = st.columns([1, 1.3], gap="large")
    with a:
        st.markdown("##### All models on this reading")
        rows = []
        for model in C.MODEL_ORDER:
            prob = latest["probs"].get(model)
            t = C.default_threshold(model) if model != label else threshold
            rows.append({"Model": model, "Probability": prob,
                         "Verdict": "n/a" if prob is None else ("Alert" if prob >= t else "OK")})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
            "Probability": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
        })
    with b:
        st.markdown("##### Alert log")
        events = []
        for i, h in enumerate(history):
            prev = probs[i - 1] if i else 0.0
            if (probs[i] >= threshold) != (prev >= threshold):
                events.append({"Time (UTC)": pd.Timestamp(h["time"]).strftime("%H:%M:%S"),
                               "Event": "Alert raised" if probs[i] >= threshold else "Alert cleared",
                               "Probability": probs[i],
                               "Torque (Nm)": h["values"]["Torque [Nm]"],
                               "Tool wear (min)": h["values"]["Tool wear [min]"]})
        if events:
            st.dataframe(pd.DataFrame(events[::-1][:50]), hide_index=True, width="stretch",
                         column_config={"Probability": st.column_config.NumberColumn(format="percent"),
                                        "Torque (Nm)": st.column_config.NumberColumn(format="%.1f"),
                                        "Tool wear (min)": st.column_config.NumberColumn(format="%.0f")})
        else:
            st.caption("No alerts yet.")
