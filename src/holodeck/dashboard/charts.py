"""Plotly figure factories for the Summary view.

Pure functions returning `plotly.graph_objects.Figure`. Consumers wrap them
in `dcc.Graph`. Every chart calls `_apply_theme(fig)` so colors, fonts,
gridlines, and margins align with the handoff dark terminal-green aesthetic.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd
import plotly.graph_objects as go

from holodeck.dashboard.data_loader import (
    detect_regressions,
    detect_version_boundaries,
    to_breakdown_dataframe,
    to_metric_trend_dataframe,
    to_summary_dataframe,
)
from holodeck.models.eval_run import EvalRun

MetricKind = Literal["standard", "rag", "geval", "code"]

PALETTE = ["#7bff5a", "#5ae0a6", "#53ff9c", "#9bff5f", "#a7f0ba", "#ffcf5a"]
ACCENT = "#7bff5a"
ACCENT_FILL = "rgba(123,255,90,0.15)"
CORAL = "#ff9d7e"
BG_CARD = "#070c0a"
FG = "#e8f5ec"
MUTED = "#9bb3a5"
GRID = "rgba(28,43,37,0.5)"
BORDER_VERSION = "rgba(123,255,90,0.35)"
THRESHOLD_COLOR = "rgba(255,120,80,0.7)"


def _apply_theme(fig: go.Figure, *, show_legend: bool = False) -> go.Figure:
    fig.update_layout(
        plot_bgcolor=BG_CARD,
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": FG, "family": "Inter, system-ui, sans-serif", "size": 12},
        margin={"l": 40, "r": 16, "t": 16, "b": 28},
        showlegend=show_legend,
        legend={
            "bgcolor": "rgba(0,0,0,0)",
            "bordercolor": "rgba(0,0,0,0)",
            "font": {"size": 11, "color": MUTED},
        },
        hoverlabel={
            "bgcolor": "#0a110f",
            "bordercolor": "#1c2b25",
            "font": {"color": FG, "family": "JetBrains Mono, monospace", "size": 12},
        },
    )
    fig.update_xaxes(
        gridcolor=GRID,
        zerolinecolor=GRID,
        linecolor="rgba(28,43,37,0.8)",
        tickfont={"size": 10, "color": MUTED, "family": "JetBrains Mono, monospace"},
    )
    fig.update_yaxes(
        gridcolor=GRID,
        zerolinecolor=GRID,
        linecolor="rgba(28,43,37,0.8)",
        tickfont={"size": 10, "color": MUTED, "family": "JetBrains Mono, monospace"},
    )
    return fig


def pass_rate_chart(runs: list[EvalRun]) -> go.Figure:
    """Filled area + line + regression dots + version-boundary annotations."""
    fig = go.Figure()
    if not runs:
        return _apply_theme(fig)

    df = to_summary_dataframe(runs).sort_values("timestamp").reset_index(drop=True)
    regression_idx = set(detect_regressions(runs))
    boundaries = detect_version_boundaries(runs)

    # Filled area + main line
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["pass_rate"],
            mode="lines",
            line={"color": ACCENT, "width": 2, "shape": "spline", "smoothing": 0.6},
            fill="tozeroy",
            fillcolor=ACCENT_FILL,
            hovertemplate="%{x|%b %d}<br>%{y:.0%}<extra></extra>",
            showlegend=False,
        )
    )

    # Normal dots
    normal_mask = [i not in regression_idx for i in range(len(df))]
    fig.add_trace(
        go.Scatter(
            x=df.loc[normal_mask, "timestamp"],
            y=df.loc[normal_mask, "pass_rate"],
            mode="markers",
            marker={
                "color": ACCENT,
                "size": 7,
                "line": {"color": "#050b09", "width": 2},
            },
            hovertemplate="%{x|%b %d}<br>%{y:.0%}<extra></extra>",
            showlegend=False,
        )
    )

    # Regression dots (coral)
    if regression_idx:
        reg_df = df.iloc[sorted(regression_idx)]
        fig.add_trace(
            go.Scatter(
                x=reg_df["timestamp"],
                y=reg_df["pass_rate"],
                mode="markers",
                marker={
                    "color": CORAL,
                    "size": 10,
                    "line": {"color": "#050b09", "width": 2},
                },
                hovertemplate="regression<br>%{x|%b %d}<br>%{y:.0%}<extra></extra>",
                showlegend=False,
            )
        )

    # Prompt-version boundary lines
    # NOTE: add_vline() with annotation_text crashes on pandas Timestamp x-values
    # (plotly.shapeannotation averages X and blows up). Use add_shape + add_annotation
    # separately so we can still label each boundary.
    for idx, version in boundaries:
        if idx < len(df):
            ts = df.iloc[idx]["timestamp"]
            fig.add_shape(
                type="line",
                xref="x",
                yref="paper",
                x0=ts,
                x1=ts,
                y0=0,
                y1=1,
                line={"color": BORDER_VERSION, "width": 1, "dash": "dot"},
            )
            fig.add_annotation(
                xref="x",
                yref="paper",
                x=ts,
                y=1.02,
                text=version,
                showarrow=False,
                font={
                    "size": 10,
                    "color": "#5ae0a6",
                    "family": "JetBrains Mono, monospace",
                },
                xanchor="center",
                yanchor="bottom",
            )

    fig.update_yaxes(
        range=[0, 1],
        tickmode="array",
        tickvals=[0, 0.25, 0.5, 0.75, 1.0],
        tickformat=".0%",
    )
    fig.update_xaxes(tickformat="%b %d")
    return _apply_theme(fig)


def metric_trend_chart(runs: list[EvalRun], kind: MetricKind) -> go.Figure:
    fig = go.Figure()
    if not runs:
        return _apply_theme(fig)

    df = to_metric_trend_dataframe(runs, kind)
    if df.empty or len(df.columns) < 2:
        return _apply_theme(fig, show_legend=True)

    metric_cols = [c for c in df.columns if c != "timestamp"]
    for i, col in enumerate(metric_cols):
        color = PALETTE[i % len(PALETTE)]
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df[col],
                mode="lines+markers",
                name=col,
                line={"color": color, "width": 2, "shape": "spline", "smoothing": 0.5},
                marker={
                    "color": color,
                    "size": 5,
                    "line": {"color": "#050b09", "width": 1},
                },
                hovertemplate=f"<b>{col}</b><br>%{{x|%b %d}}: %{{y:.2f}}<extra></extra>",
            )
        )

    fig.add_hline(
        y=0.7,
        line_dash="dash",
        line_color=THRESHOLD_COLOR,
        line_width=1,
        annotation_text="thresh 0.7",
        annotation_position="top right",
        annotation_font={
            "size": 10,
            "color": "#ff9d7e",
            "family": "JetBrains Mono, monospace",
        },
    )

    fig.update_yaxes(range=[0, 1], tickformat=".2f")
    fig.update_xaxes(tickformat="%b %d")
    return _apply_theme(fig, show_legend=True)


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
            marker={"color": colors, "line": {"color": "#050b09", "width": 0.5}},
            text=[f"{v:.2f}" for v in df["avg_score"]],
            textposition="outside",
            textfont={"color": FG, "family": "JetBrains Mono, monospace", "size": 11},
            hovertemplate="<b>%{y}</b><br>avg=%{x:.2f}<extra></extra>",
            showlegend=False,
        )
    )

    fig.add_vline(
        x=0.7,
        line_dash="dash",
        line_color=THRESHOLD_COLOR,
        line_width=1,
    )

    height = max(180, 48 * len(df) + 40)
    fig.update_layout(height=height, bargap=0.4)
    fig.update_xaxes(range=[0, 1.08], tickformat=".1f")
    fig.update_yaxes(
        tickfont={"size": 11, "color": FG, "family": "JetBrains Mono, monospace"}
    )
    return _apply_theme(fig)


def sparkline(values: list[float], color: str = ACCENT) -> go.Figure:
    fig = go.Figure()
    if not values:
        return fig
    fig.add_trace(
        go.Scatter(
            y=values,
            mode="lines",
            line={"color": color, "width": 1.5},
            fill="tozeroy",
            fillcolor="rgba(123,255,90,0.18)",
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
