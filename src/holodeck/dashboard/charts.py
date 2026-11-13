"""Plotly figure factories for the Summary view.

Pure functions returning ``plotly.graph_objects.Figure``. Consumers wrap them
in ``dcc.Graph`` with the mode bar hidden. Every chart calls
``_apply_theme(fig)``: white plot, ``#dde0e8`` grid, house-blue lines, Public
Sans labels. Revision bands come from :mod:`holodeck.dashboard.revisions` so
a prompt version has the same colour here as on every chip.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd
import plotly.graph_objects as go

from holodeck.dashboard.data_loader import (
    to_breakdown_dataframe,
    to_metric_trend_dataframe,
    to_summary_dataframe,
)
from holodeck.dashboard.revisions import (
    RevisionColour,
    revision_colour,
    revision_colour_map,
    run_label,
    run_numbers,
)
from holodeck.dashboard.verdict import Regression, find_regressions
from holodeck.models.eval_run import EvalRun

MetricKind = Literal["standard", "rag", "geval", "code"]

HOUSE = "#1d2347"
INK = "#15171e"
SECONDARY = "#555b6b"
GRID = "#dde0e8"
GRID_INNER = "#eceef3"
FAIL = "#c2312a"
SANS = "Public Sans, system-ui, sans-serif"
MONO = "Courier Prime, ui-monospace, monospace"

# Series colours for metric lines. Revision colours are reserved for prompt
# versions, so metric lines use house blue and neutral data inks, each with
# its own dash pattern so lines never rely on colour alone.
SERIES = ["#1d2347", "#3d6fc4", "#b07300", "#2c7a7b", "#6a4fb0", "#5d6274"]
SERIES_DASH = ["solid", "dash", "dot", "dashdot", "longdash", "longdashdot"]
PALETTE = SERIES
ACCENT = HOUSE


def _apply_theme(fig: go.Figure, *, show_legend: bool = False) -> go.Figure:
    fig.update_layout(
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        font={"color": INK, "family": SANS, "size": 12},
        margin={"l": 48, "r": 16, "t": 28, "b": 36},
        showlegend=show_legend,
        legend={
            "orientation": "h",
            "x": 0,
            "y": -0.18,
            "xanchor": "left",
            "yanchor": "top",
            "bgcolor": "rgba(0,0,0,0)",
            "font": {"size": 12, "color": SECONDARY, "family": MONO},
        },
        hoverlabel={
            "bgcolor": "#ffffff",
            "bordercolor": GRID,
            "font": {"color": INK, "family": SANS, "size": 12},
        },
        hovermode="closest",
    )
    axis = {
        "gridcolor": GRID,
        "zerolinecolor": GRID,
        "linecolor": GRID,
        "tickfont": {"size": 11, "color": SECONDARY, "family": SANS},
        "fixedrange": True,
    }
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    return fig


def _version_segments(
    df: pd.DataFrame,
) -> list[tuple[int, int, str]]:
    """Contiguous ``(first_idx, last_idx, version)`` runs in time order."""
    segments: list[tuple[int, int, str]] = []
    for i, version in enumerate(df["prompt_version"]):
        if segments and segments[-1][2] == version:
            first, _, v = segments[-1]
            segments[-1] = (first, i, v)
        else:
            segments.append((i, i, version))
    return segments


def _edges(times: list[pd.Timestamp]) -> list[pd.Timestamp]:
    """Band edges: midpoints between runs, padded half a gap at each end."""
    if len(times) == 1:
        pad = pd.Timedelta(hours=12)
        return [times[0] - pad, times[0] + pad]
    mids = [times[i] + (times[i + 1] - times[i]) / 2 for i in range(len(times) - 1)]
    first = times[0] - (times[1] - times[0]) / 2
    last = times[-1] + (times[-1] - times[-2]) / 2
    return [first, *mids, last]


def pass_rate_chart(
    runs: list[EvalRun],
    *,
    numbers: dict[str, int] | None = None,
    colours: dict[str, RevisionColour] | None = None,
    regressions: list[Regression] | None = None,
) -> go.Figure:
    """Pass rate by run over revision bands, with numbered regression rings."""
    fig = go.Figure()
    if not runs:
        return _apply_theme(fig)

    numbers = numbers if numbers is not None else run_numbers(runs)
    colours = colours if colours is not None else revision_colour_map(runs)
    if regressions is None:
        regressions = find_regressions(runs, numbers)

    df = to_summary_dataframe(runs).sort_values("timestamp").reset_index(drop=True)
    times = list(df["timestamp"])
    edges = _edges(times)
    labels = [run_label(rid, numbers) for rid in df["id"]]

    for first, last, version in _version_segments(df):
        colour = revision_colour(version, colours)
        fig.add_shape(
            type="rect",
            xref="x",
            yref="paper",
            x0=edges[first],
            x1=edges[last + 1],
            y0=0,
            y1=1,
            fillcolor=colour.tint,
            opacity=0.6,
            line={"width": 0},
            layer="below",
        )
        fig.add_annotation(
            xref="x",
            yref="paper",
            x=edges[first],
            y=1,
            xanchor="left",
            yanchor="top",
            xshift=4,
            yshift=-2,
            text=f"<b>{version}</b>",
            showarrow=False,
            font={"size": 12, "color": INK, "family": MONO},
        )

    custom = [
        [label, version]
        for label, version in zip(labels, df["prompt_version"], strict=True)
    ]
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["pass_rate"],
            mode="lines+markers",
            line={"color": HOUSE, "width": 2.2},
            marker={"color": HOUSE, "size": 5},
            customdata=custom,
            hovertemplate=(
                "<b>%{customdata[0]}</b> · %{customdata[1]}"
                "<br>%{x|%-d %b %H:%M}<br>Pass rate %{y:.1%}<extra></extra>"
            ),
            showlegend=False,
        )
    )

    by_id = {rid: i for i, rid in enumerate(df["id"])}
    ring_x: list[pd.Timestamp] = []
    ring_y: list[float] = []
    ring_text: list[str] = []
    for reg in regressions:
        idx = by_id.get(reg.run_id)
        if idx is None:
            continue
        ring_x.append(df.at[idx, "timestamp"])
        ring_y.append(float(df.at[idx, "pass_rate"]))
        ring_text.append(str(reg.number))
    if ring_x:
        fig.add_trace(
            go.Scatter(
                x=ring_x,
                y=ring_y,
                mode="markers+text",
                marker={
                    "symbol": "circle-open",
                    "color": FAIL,
                    "size": 15,
                    "line": {"color": FAIL, "width": 2},
                },
                text=ring_text,
                textposition="bottom center",
                textfont={"color": FAIL, "family": MONO, "size": 13},
                hovertemplate="Regression %{text}<br>%{y:.1%}<extra></extra>",
                showlegend=False,
            )
        )

    fig.update_yaxes(
        range=[0, 1.04],
        tickmode="array",
        tickvals=[0, 0.25, 0.5, 0.75, 1.0],
        tickformat=".0%",
    )
    fig.update_xaxes(tickformat="%-d %b", showgrid=False)
    return _apply_theme(fig)


def metric_trend_chart(
    runs: list[EvalRun], kind: MetricKind, threshold: float | None = 0.7
) -> go.Figure:
    """Mean score per run for every metric of ``kind``; dashed threshold."""
    fig = go.Figure()
    if not runs:
        return _apply_theme(fig)

    df = to_metric_trend_dataframe(runs, kind)
    if df.empty or len(df.columns) < 2:
        return _apply_theme(fig, show_legend=True)

    metric_cols = [c for c in df.columns if c != "timestamp"]
    for i, col in enumerate(metric_cols):
        color = SERIES[i % len(SERIES)]
        dash = SERIES_DASH[i % len(SERIES_DASH)]
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df[col],
                mode="lines",
                name=col,
                line={"color": color, "width": 2, "dash": dash},
                hovertemplate=f"<b>{col}</b><br>%{{x|%-d %b}}: %{{y:.2f}}<extra></extra>",
            )
        )

    if threshold is not None:
        fig.add_hline(
            y=threshold,
            line_dash="dash",
            line_color=FAIL,
            line_width=1.5,
            annotation_text=f"{threshold:.2f}",
            annotation_position="top right",
            annotation_font={"size": 12, "color": FAIL, "family": MONO},
        )

    fig.update_yaxes(range=[0, 1.02], tickformat=".2f", tickvals=[0, 0.5, 1.0])
    fig.update_xaxes(tickformat="%-d %b", showgrid=False)
    fig = _apply_theme(fig, show_legend=True)
    fig.update_layout(margin={"l": 48, "r": 16, "t": 16, "b": 72})
    return fig


def breakdown_bar(df: pd.DataFrame, palette: list[str]) -> go.Figure:
    fig = go.Figure()
    if df.empty:
        return _apply_theme(fig)

    df = df.sort_values("avg_score", ascending=True).reset_index(drop=True)
    colors = [palette[i % len(palette)] for i in range(len(df))]

    fig.add_trace(
        go.Bar(
            y=df["metric_name"],
            x=df["avg_score"],
            orientation="h",
            marker={"color": colors},
            text=[f"{v:.2f}" for v in df["avg_score"]],
            textposition="outside",
            textfont={"color": INK, "family": MONO, "size": 12},
            hovertemplate="<b>%{y}</b><br>avg=%{x:.2f}<extra></extra>",
            showlegend=False,
        )
    )
    fig.add_vline(x=0.7, line_dash="dash", line_color=FAIL, line_width=1.5)
    height = max(180, 48 * len(df) + 40)
    fig.update_layout(height=height, bargap=0.4)
    fig.update_xaxes(range=[0, 1.08], tickformat=".1f")
    fig.update_yaxes(tickfont={"size": 12, "color": INK, "family": MONO})
    return _apply_theme(fig)


def sparkline(values: list[float], color: str = HOUSE) -> go.Figure:
    fig = go.Figure()
    if not values:
        return fig
    fig.add_trace(
        go.Scatter(
            y=values,
            mode="lines",
            line={"color": color, "width": 1.5},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        xaxis={"visible": False, "fixedrange": True},
        yaxis={"visible": False, "fixedrange": True},
        showlegend=False,
    )
    return fig


def breakdown_dataframe(
    runs: list[EvalRun], kind: MetricKind, recent_n: int = 6
) -> pd.DataFrame:
    """Re-export for views that want the bar chart alongside raw rows."""
    return to_breakdown_dataframe(runs, kind, recent_n=recent_n)
