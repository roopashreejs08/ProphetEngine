"""Plotly figures with one consistent look.

Charts are rendered with ``theme="streamlit"`` so fonts and backgrounds follow
the viewer's light/dark theme. Colours come from ``dashboard/theme.py``: neutral
greys for context, the accent for the key series, red only for failures/alerts.
"""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from dashboard import config as C
from dashboard import theme as T
from dashboard import ui

PLOT_CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"]}


def _t():
    return ui.tokens()


def _key_last(series_map):
    """Items with the accent (key) series drawn last, so it sits on top of the grey context lines."""
    key = ui.key_model(list(series_map))
    return sorted(series_map.items(), key=lambda item: item[0] == key)


def show(fig, key=None, height=None):
    """Render a figure. Its title is lifted out into a Streamlit heading so it never
    collides with the legend, and stays readable in both themes."""
    if height:
        fig.update_layout(height=height)
    title = fig.layout.title.text if fig.layout.title else None
    if title:
        st.markdown(f"**{title}**")
        fig.update_layout(title=None, margin={"t": 28})
    st.plotly_chart(fig, width="stretch", theme="streamlit", config=PLOT_CONFIG, key=key)


def _layout(fig, title=None, x=None, y=None, height=340, legend=True):
    t = _t()
    fig.update_layout(
        title={"text": title, "x": 0, "xanchor": "left", "font": {"size": 15}} if title else None,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"family": T.FONT, "color": t["text2"]},
        hoverlabel={"font": {"family": T.FONT}},
        height=height,
        margin={"l": 8, "r": 16, "t": 44 if title else 12, "b": 8},
        showlegend=legend,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1, "title": None,
                "font": {"size": 12, "color": t["text2"]}},
        hovermode="closest",
        bargap=0.25,
    )
    axis = {"zeroline": False, "linecolor": t["border"], "tickcolor": t["border"],
            "tickfont": {"color": t["muted"], "size": 11}, "title_font": {"color": t["muted"], "size": 12}}
    fig.update_xaxes(title=x, showgrid=False, showline=True, **axis)
    fig.update_yaxes(title=y, showgrid=True, gridcolor=t["grid"], showline=False, **axis)
    return fig


def _threshold_line(fig, threshold, axis="x"):
    t = _t()
    kwargs = {"line": {"dash": "dot", "width": 1.2, "color": t["muted"]},
              "annotation_text": f"alert threshold {threshold:.2f}", "annotation_font_size": 11,
              "annotation_font_color": t["muted"]}
    if axis == "x":
        fig.add_vline(x=threshold, annotation_position="top right", **kwargs)
    else:
        fig.add_hline(y=threshold, annotation_position="top left", **kwargs)


# ---------------------------------------------------------------------------
# Model performance
# ---------------------------------------------------------------------------
def metric_bars(frame, metrics):
    fig = go.Figure()
    present = list(frame["Model"])
    for _, row in frame.iterrows():
        label = row["Model"]
        values = [row[m] for m in metrics]
        fig.add_bar(
            name=label, x=metrics, y=values, marker_color=ui.model_color(label, present),
            marker_line_width=0,
            text=[f"{v:.2f}" if np.isfinite(v) else "" for v in values], textposition="outside",
            textfont={"size": 10, "color": _t()["muted"]},
            hovertemplate=f"<b>{label}</b><br>%{{x}}: %{{y:.3f}}<extra></extra>",
        )
    _layout(fig, None, None, "Score (0-1)", height=360)
    fig.update_layout(barmode="group", bargroupgap=0.08)
    fig.update_yaxes(range=[0, 1.12])
    return fig


def pr_curves(curve_map, base_rate):
    fig = go.Figure()
    for label, (curve, ap) in _key_last(curve_map):
        color, dash = ui.model_style(label, list(curve_map))
        fig.add_scatter(x=curve["recall"], y=curve["precision"], mode="lines", name=f"{label} (AP {ap:.3f})",
                        line={"color": color, "width": 2, "shape": "hv", "dash": dash},
                        hovertemplate=f"<b>{label}</b><br>recall %{{x:.2f}}<br>precision %{{y:.2f}}<extra></extra>")
    fig.add_hline(y=base_rate, line={"dash": "dot", "color": _t()["muted"], "width": 1},
                  annotation_text=f"random guess ({base_rate:.1%})", annotation_position="bottom right",
                  annotation_font_size=11, annotation_font_color=_t()["muted"])
    _layout(fig, "Precision-recall curve", "Recall (share of failures caught)", "Precision (alerts that were real)")
    fig.update_xaxes(range=[0, 1.01])
    fig.update_yaxes(range=[0, 1.05])
    return fig


def roc_curves(curve_map):
    fig = go.Figure()
    for label, (curve, auc) in _key_last(curve_map):
        color, dash = ui.model_style(label, list(curve_map))
        fig.add_scatter(x=curve["fpr"], y=curve["tpr"], mode="lines", name=f"{label} (AUC {auc:.3f})",
                        line={"color": color, "width": 2, "dash": dash},
                        hovertemplate=f"<b>{label}</b><br>false-alarm rate %{{x:.2f}}<br>recall %{{y:.2f}}<extra></extra>")
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random guess",
                    line={"dash": "dot", "color": _t()["border"], "width": 1}, hoverinfo="skip")
    _layout(fig, "ROC curve", "False-alarm rate", "Recall")
    return fig


def threshold_tradeoff(sweep, threshold):
    fig = go.Figure()
    t = _t()
    series = [("precision", "Precision", t["accent"], "solid"), ("recall", "Recall", t["greys"][0], "dash"),
              ("f1", "F1-score", t["greys"][1], "dot")]
    for column, name, color, dash in series:
        fig.add_scatter(x=sweep["threshold"], y=sweep[column], name=name, mode="lines",
                        line={"color": color, "width": 2, "dash": dash},
                        hovertemplate=f"{name} %{{y:.2f}}<extra>threshold %{{x:.2f}}</extra>")
    _threshold_line(fig, threshold)
    _layout(fig, "How the threshold trades precision for recall", "Alert threshold", "Score")
    fig.update_layout(hovermode="x unified")
    fig.update_yaxes(range=[0, 1.05])
    return fig


def cost_curve(sweep, threshold, best):
    fig = go.Figure()
    fig.add_scatter(x=sweep["threshold"], y=sweep["cost"], mode="lines", name="Total cost",
                    line={"color": _t()["accent"], "width": 2},
                    customdata=np.stack([sweep["fn"], sweep["fp"]], axis=1),
                    hovertemplate="threshold %{x:.2f}<br>cost %{y:,.0f}<br>missed %{customdata[0]} | "
                                  "false alarms %{customdata[1]}<extra></extra>")
    fig.add_scatter(x=[best["threshold"]], y=[best["cost"]], mode="markers+text", name="Lowest cost",
                    marker={"size": 10, "color": _t()["text"], "line": {"width": 2, "color": _t()["bg"]}},
                    text=[f"best {best['threshold']:.2f}"], textposition="top center",
                    textfont={"size": 11, "color": _t()["text2"]},
                    hovertemplate="lowest cost at %{x:.2f}: %{y:,.0f}<extra></extra>")
    _threshold_line(fig, threshold)
    _layout(fig, "Expected cost on the 2,000 validation machines", "Alert threshold", "Cost", legend=False)
    return fig


def confusion_heatmap(m):
    z = [[m["tn"], m["fp"]], [m["fn"], m["tp"]]]
    labels = [["Correctly left alone", "False alarm"], ["Missed failure", "Caught failure"]]
    text = [[f"<b>{z[r][c]:,}</b><br>{labels[r][c]}" for c in range(2)] for r in range(2)]
    t = _t()
    # Correct outcomes stay neutral; errors get a faint red tint (missed failures strongest).
    fig = go.Figure(go.Heatmap(
        z=[[0, 0.5], [1, 0]], x=["Predicted OK", "Predicted failure"], y=["Actually OK", "Actually failed"],
        text=text, texttemplate="%{text}", textfont={"size": 13, "color": t["text"]},
        colorscale=[[0, t["cell"]], [0.5, t["cell_warn"]], [1, t["cell_bad"]]], zmin=0, zmax=1,
        showscale=False, hoverinfo="skip", xgap=2, ygap=2,
    ))
    _layout(fig, None, None, None, height=280, legend=False)
    fig.update_yaxes(autorange="reversed", showgrid=False, showline=False)
    fig.update_xaxes(side="top", showgrid=False, showline=False)
    return fig


def failure_mode_bars(frame):
    frame = frame.sort_values("recall")
    fig = go.Figure(go.Bar(
        x=frame["recall"], y=frame["mode"], orientation="h",
        marker_color=_t()["accent"], marker_line_width=0,
        text=[f"{c} of {t}" for c, t in zip(frame["caught"], frame["failures"])], textposition="outside",
        textfont={"size": 11, "color": _t()["muted"]},
        hovertemplate="%{y}: %{x:.0%} caught<extra></extra>", cliponaxis=False,
    ))
    _layout(fig, "Failures caught, by failure type", "Share caught", None, height=260, legend=False)
    fig.update_xaxes(range=[0, 1.18], tickformat=".0%")
    return fig


def reliability_chart(curve_map):
    fig = go.Figure()
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfectly calibrated",
                    line={"dash": "dot", "color": _t()["border"], "width": 1}, hoverinfo="skip")
    for label, frame in _key_last(curve_map):
        color, dash = ui.model_style(label, list(curve_map))
        fig.add_scatter(x=frame["predicted"], y=frame["observed"], mode="lines+markers", name=label,
                        line={"color": color, "width": 2, "dash": dash}, marker={"size": 7},
                        hovertemplate=f"<b>{label}</b><br>predicted %{{x:.2f}}<br>observed %{{y:.2f}}<extra></extra>")
    _layout(fig, "Calibration: do predicted probabilities match reality?", "Mean predicted probability",
            "Observed failure rate")
    return fig


# ---------------------------------------------------------------------------
# Fleet / overview
# ---------------------------------------------------------------------------
def risk_histogram(probabilities, labels, threshold):
    fig = go.Figure()
    bins = {"start": 0, "end": 1.0001, "size": 0.025}
    fig.add_histogram(x=probabilities[labels == 0], name="Did not fail", xbins=bins,
                      marker_color=_t()["greys"][1], opacity=0.9,
                      hovertemplate="%{y} machines<extra>did not fail</extra>")
    fig.add_histogram(x=probabilities[labels == 1], name="Failed", xbins=bins,
                      marker_color=_t()["red"], opacity=0.9,
                      hovertemplate="%{y} machines<extra>failed</extra>")
    _threshold_line(fig, threshold)
    _layout(fig, "Predicted risk across the fleet", "Predicted failure probability", "Machines (log scale)")
    fig.update_layout(barmode="overlay")
    fig.update_yaxes(type="log", dtick=1)
    return fig


def tier_bar(counts):
    t = _t()
    fill = {"healthy": t["greys"][2], "watch": t["greys"][0], "critical": t["red"]}
    ink = {"healthy": t["text"], "watch": t["bg"], "critical": "#ffffff"}
    fig = go.Figure()
    order = [("healthy", "Healthy"), ("watch", "Watch"), ("critical", "At risk")]
    total = sum(counts.values()) or 1
    for tier, name in order:
        value = counts.get(tier, 0)
        fig.add_bar(y=["Fleet"], x=[value], orientation="h", name=f"{name} ({value:,})",
                    marker_color=fill[tier], marker_line_width=2, marker_line_color=t["bg"],
                    text=[f"{name}: {value:,} ({value / total:.0%})" if value / total > 0.06 else ""],
                    textposition="inside", insidetextanchor="middle",
                    textfont={"color": ink[tier], "size": 12},
                    hovertemplate=f"{name}: %{{x:,}} machines<extra></extra>")
    _layout(fig, None, None, None, height=96)
    fig.update_layout(barmode="stack", margin={"l": 0, "r": 0, "t": 28, "b": 0}, bargap=0.1)
    fig.update_xaxes(visible=False, range=[0, total])
    fig.update_yaxes(visible=False)
    return fig


def gauge(probability, threshold, height=230):
    tier = "critical" if probability >= threshold else ("watch" if probability >= threshold / 2 else "healthy")
    t = _t()
    bar = {"critical": t["red"], "watch": t["greys"][0], "healthy": t["accent"]}[tier]
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=probability * 100,
        number={"suffix": "%", "valueformat": ".1f", "font": {"color": t["text"], "size": 30}},
        gauge={
            "axis": {"range": [0, 100], "ticksuffix": "%", "tickcolor": t["border"],
                     "tickfont": {"color": t["muted"], "size": 10}},
            "bar": {"color": bar, "thickness": 0.3},
            "bgcolor": t["subtle"], "borderwidth": 0,
            "steps": [{"range": [threshold * 100, 100], "color": t["red_soft"]}],
            "threshold": {"line": {"color": t["text2"], "width": 2}, "value": threshold * 100, "thickness": 0.9},
        },
    ))
    fig.update_layout(height=height, margin={"l": 40, "r": 40, "t": 16, "b": 0},
                      paper_bgcolor="rgba(0,0,0,0)", font={"family": T.FONT})
    return fig


# ---------------------------------------------------------------------------
# Explainability
# ---------------------------------------------------------------------------
def importance_bars(names, values, title, x_title, fmt=".0%", errors=None):
    order = np.argsort(values)
    names = [names[i] for i in order]
    values = [values[i] for i in order]
    errors = None if errors is None else [errors[i] for i in order]
    colors = [_t()["greys"][1]] * len(values)
    colors[-1] = _t()["accent"]
    hover = "%{y}: %{x:" + fmt + "}"
    if errors is not None:
        hover += " (+/- %{customdata:" + fmt + "} across shuffles)"
    fig = go.Figure(go.Bar(
        x=values, y=names, orientation="h", marker_color=colors, customdata=errors, marker_line_width=0,
        text=[format(v, fmt) for v in values], textposition="outside", cliponaxis=False,
        textfont={"size": 11, "color": _t()["muted"]},
        hovertemplate=hover + "<extra></extra>",
    ))
    _layout(fig, title, x_title, None, height=320, legend=False)
    fig.update_xaxes(tickformat=fmt, range=[0, max(values) * 1.25 if max(values) > 0 else 1])
    return fig


def contribution_bars(frame):
    frame = frame.iloc[np.argsort(np.abs(frame["contribution"].to_numpy()))]
    t = _t()
    colors = [t["red"] if v > 0 else t["accent"] for v in frame["contribution"]]
    fig = go.Figure(go.Bar(
        x=frame["contribution"], y=frame["feature"], orientation="h", marker_color=colors, marker_line_width=0,
        text=[f"{v:+.1%}" for v in frame["contribution"]], textposition="outside", cliponaxis=False,
        textfont={"size": 11, "color": t["muted"]},
        hovertemplate="%{y}: %{x:+.1%} vs a typical value<extra></extra>",
    ))
    span = max(0.05, float(np.abs(frame["contribution"]).max()) * 1.35)
    _layout(fig, "What is driving this prediction", "Change in failure probability vs a typical reading", None,
            height=270, legend=False)
    fig.update_xaxes(range=[-span, span], tickformat="+.0%", zeroline=True, zerolinecolor=t["muted"],
                     zerolinewidth=1)
    return fig


def sensitivity_line(grid, probs, current_x, current_p, threshold, column, label):
    fig = go.Figure()
    fig.add_scatter(x=grid, y=probs, mode="lines", line={"color": _t()["accent"], "width": 2},
                    name="Failure probability",
                    hovertemplate=f"{C.SENSOR_LABEL[column]} %{{x:,.1f}}<br>probability %{{y:.1%}}<extra></extra>")
    fig.add_scatter(x=[current_x], y=[current_p], mode="markers", name="Current reading",
                    marker={"size": 10, "color": _t()["accent"], "line": {"width": 2, "color": _t()["bg"]}},
                    hovertemplate="current: %{y:.1%}<extra></extra>")
    _threshold_line(fig, threshold, axis="y")
    _layout(fig, f"If only {C.SENSOR_LABEL[column].lower()} changed...",
            f"{C.SENSOR_LABEL[column]} ({C.SENSOR_UNIT[column]})", "Failure probability", height=300, legend=False)
    fig.update_yaxes(range=[0, 1.02], tickformat=".0%")
    return fig


def pdp_lines(series, column):
    fig = go.Figure()
    for label, (grid, values) in _key_last(series):
        color, dash = ui.model_style(label, list(series))
        fig.add_scatter(x=grid, y=values, mode="lines", name=label,
                        line={"color": color, "width": 2, "dash": dash},
                        hovertemplate=f"<b>{label}</b><br>%{{x:,.1f}}: %{{y:.1%}}<extra></extra>")
    _layout(fig, f"Average predicted risk as {C.SENSOR_LABEL[column].lower()} changes",
            f"{C.SENSOR_LABEL[column]} ({C.SENSOR_UNIT[column]})", "Mean failure probability")
    fig.update_yaxes(tickformat=".0%", rangemode="tozero")
    return fig


def surface_heatmap(x_grid, y_grid, z, x_col, y_col, point=None, train_points=None):
    fig = go.Figure(go.Heatmap(
        x=x_grid, y=y_grid, z=z, zmin=0, zmax=1,
        colorscale=_t()["seq"],
        colorbar={"title": {"text": "P(fail)", "font": {"size": 11, "color": _t()["muted"]}}, "tickformat": ".0%",
                  "thickness": 10, "outlinewidth": 0, "tickfont": {"size": 10, "color": _t()["muted"]}},
        hovertemplate=f"{C.SENSOR_LABEL[x_col]} %{{x:,.0f}}<br>{C.SENSOR_LABEL[y_col]} %{{y:,.1f}}"
                      "<br>probability %{z:.1%}<extra></extra>",
    ))
    if train_points is not None:
        fig.add_scatter(x=train_points[x_col], y=train_points[y_col], mode="markers", name="Real failures (training)",
                        marker={"size": 6, "color": "rgba(0,0,0,0)", "line": {"width": 1, "color": _t()["text2"]}},
                        hovertemplate="real failure<extra></extra>")
    if point is not None:
        fig.add_scatter(x=[point[0]], y=[point[1]], mode="markers", name="Current reading",
                        marker={"size": 13, "symbol": "x", "color": _t()["accent"],
                                "line": {"width": 1.5, "color": _t()["bg"]}})
    _layout(fig, None, f"{C.SENSOR_LABEL[x_col]} ({C.SENSOR_UNIT[x_col]})",
            f"{C.SENSOR_LABEL[y_col]} ({C.SENSOR_UNIT[y_col]})", height=430)
    return fig


# ---------------------------------------------------------------------------
# Live
# ---------------------------------------------------------------------------
def live_probability(times, probs, threshold, label, actual=None):
    fig = go.Figure()
    fig.add_scatter(x=times, y=probs, mode="lines+markers", name=label,
                    line={"color": _t()["accent"], "width": 2}, marker={"size": 4},
                    hovertemplate="%{x}<br>probability %{y:.1%}<extra></extra>")
    if actual is not None:
        failed = [i for i, a in enumerate(actual) if a == 1]
        if failed:
            fig.add_scatter(x=[times[i] for i in failed], y=[probs[i] for i in failed], mode="markers",
                            name="Actually failed", marker={"size": 12, "symbol": "x", "color": _t()["red"]},
                            hovertemplate="actual failure<extra></extra>")
    _threshold_line(fig, threshold, axis="y")
    _layout(fig, None, None, "Failure probability", height=300)
    fig.update_yaxes(range=[0, 1.02], tickformat=".0%")
    return fig


# ---------------------------------------------------------------------------
# Data explorer
# ---------------------------------------------------------------------------
def distribution(frame, column, label_col):
    fig = go.Figure()
    for value, name, color in [(0, "No failure", _t()["greys"][1]), (1, "Failure", _t()["red"])]:
        subset = frame.loc[frame[label_col] == value, column]
        fig.add_histogram(x=subset, name=f"{name} ({len(subset):,})", histnorm="probability density",
                          marker_color=color, opacity=0.7, nbinsx=40,
                          hovertemplate="%{x}<br>density %{y:.4f}<extra>" + name + "</extra>")
    _layout(fig, f"{C.SENSOR_LABEL.get(column, column)} - failures vs normal operation",
            column, "Density (each group sums to 1)")
    fig.update_layout(barmode="overlay")
    return fig


def scatter(frame, x_col, y_col, label_col):
    fig = go.Figure()
    for value, name, color, size in [(0, "No failure", _t()["greys"][1], 5), (1, "Failure", _t()["red"], 7)]:
        subset = frame[frame[label_col] == value]
        fig.add_scattergl(x=subset[x_col], y=subset[y_col], mode="markers", name=name,
                          marker={"size": size, "color": color},
                          hovertemplate=f"{x_col} %{{x}}<br>{y_col} %{{y}}<extra>{name}</extra>")
    _layout(fig, None, x_col, y_col, height=420)
    return fig


def correlation(frame):
    corr = frame.corr(numeric_only=True)
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.columns, zmin=-1, zmax=1,
        colorscale=_t()["div"],
        text=np.round(corr.values, 2), texttemplate="%{text}", textfont={"size": 10},
        colorbar={"thickness": 10, "outlinewidth": 0, "tickfont": {"size": 10, "color": _t()["muted"]}},
        xgap=1, ygap=1, hovertemplate="%{x} vs %{y}: %{z:.2f}<extra></extra>",
    ))
    _layout(fig, None, None, None, height=520, legend=False)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showline=False)
    return fig


def failure_mode_counts(counts):
    fig = go.Figure(go.Bar(
        x=list(counts.values()), y=[f"{C.FAILURE_MODES[k]} ({k})" for k in counts], orientation="h",
        marker_color=_t()["greys"][0], marker_line_width=0, text=list(counts.values()), textposition="outside",
        cliponaxis=False, textfont={"size": 11, "color": _t()["muted"]},
        hovertemplate="%{y}: %{x}<extra></extra>",
    ))
    _layout(fig, "Failure types in the full dataset (a failure can have several)", "Records", None, height=280,
            legend=False)
    fig.update_yaxes(autorange="reversed")
    return fig


def engine_trends(frame, sensors, engine_ids):
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=len(sensors), cols=1, shared_xaxes=True, vertical_spacing=0.04,
                        subplot_titles=sensors)
    t = _t()
    colors = [t["accent"]] + t["greys"]
    for i, engine in enumerate(engine_ids):
        sub = frame[frame["engine_id"] == engine]
        for r, sensor in enumerate(sensors, start=1):
            fig.add_scatter(x=sub["cycle"], y=sub[sensor], mode="lines", name=f"Engine {engine}",
                            legendgroup=str(engine), showlegend=(r == 1), line={"color": colors[i % 4], "width": 1.6},
                            hovertemplate=f"engine {engine}<br>cycle %{{x}}<br>{sensor} %{{y}}<extra></extra>",
                            row=r, col=1)
    _layout(fig, None, None, None, height=170 * len(sensors) + 40)
    fig.update_annotations(font={"size": 12, "color": t["muted"]})
    fig.update_xaxes(title="Cycle", row=len(sensors), col=1)
    return fig


def rul_histogram(frame):
    fig = go.Figure(go.Histogram(x=frame["RUL"], nbinsx=50, marker_color=_t()["accent"],
                                 hovertemplate="RUL %{x}: %{y:,} rows<extra></extra>"))
    _layout(fig, "Remaining useful life labels (capped at 125 cycles)", "RUL (cycles)", "Rows", legend=False)
    return fig
