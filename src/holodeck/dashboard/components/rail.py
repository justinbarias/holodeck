"""House-blue rail: brand, agent card, views, and the Summary filters.

The rail is the same on every view. Summary shows the filters (they only
apply there); Explorer gets the compact 72 px icon rail so its three columns
fit; Compare shows the full rail without filters.
"""

from __future__ import annotations

from datetime import UTC, date
from typing import Any

from dash import dcc, html

from holodeck.dashboard.components.ui import icon, sr_only
from holodeck.dashboard.data_loader import distinct_values
from holodeck.dashboard.filters import Filters
from holodeck.dashboard.loaded_run import LoadedRun
from holodeck.dashboard.revisions import (
    revision_colour,
    revision_colour_map,
    run_time,
    sorted_versions,
)
from holodeck.dashboard.state import state_filters

_VIEWS: tuple[tuple[str, str, str], ...] = (
    ("summary", "Summary", "summary"),
    ("explorer", "Explorer", "explorer"),
    ("compare", "Compare", "compare"),
)


def active_filter_count(filters: Filters) -> int:
    """How many filter groups narrow the run list."""
    return sum(
        [
            bool(filters.date_from or filters.date_to),
            bool(filters.prompt_versions),
            bool(filters.model_names),
            filters.min_pass_rate > 0,
            bool(filters.tags),
            bool(filters.search),
        ]
    )


def _brand(compact: bool) -> html.Div:
    children: list[Any] = [icon("brand", "brand-mark")]
    if not compact:
        children.append(
            html.Span(
                ["HoloDeck ", html.Span("test view", className="brand-sub")],
                className="brand-name",
            )
        )
    else:
        children.append(sr_only("HoloDeck test view"))
    return html.Div(children, className="rail-brand")


def _agent_card(agent_display_name: str, agent_name: str, seed_mode: bool) -> html.Div:
    source = "seed fixture (--seed)" if seed_mode else f"results/{agent_name}/"
    return html.Div(
        [
            html.Div(agent_display_name, className="agent-name"),
            html.Div(source, className="cue agent-path"),
            html.Div(
                [icon("saved"), "Saved results · nothing runs here"],
                className="agent-saved",
            ),
        ],
        className="agent-card",
    )


def _nav(active: str, counts: dict[str, int], compact: bool) -> html.Nav:
    items = []
    for value, label, icon_name in _VIEWS:
        is_active = value == active
        count = counts.get(value, 0)
        attrs: dict[str, Any] = {}
        if is_active:
            attrs["aria-current"] = "page"
        if compact:
            children: list[Any] = [icon(icon_name), sr_only(f"{label} ({count})")]
            attrs["title"] = label
        else:
            children = [
                html.Span([icon(icon_name), label], className="nav-label"),
                html.Span(str(count), className="cue nav-count"),
            ]
        items.append(
            html.Button(
                children,
                id={"type": "tab-btn", "value": value},
                className="nav-item active" if is_active else "nav-item",
                n_clicks=0,
                **attrs,
            )
        )
    return html.Nav(items, className="rail-nav", **{"aria-label": "Views"})


def _group(
    label: Any, control: Any, value_text: str | None = None, label_for: str = ""
) -> html.Div:
    head: list[Any] = [
        html.Label(label, htmlFor=label_for) if label_for else html.Span(label)
    ]
    if value_text is not None:
        head.append(html.Span(value_text, className="rail-value"))
    return html.Div(
        [html.Div(head, className="rail-label"), control], className="rail-group"
    )


def _chip(kind: str, value: str, text: str, on: bool, style: dict | None = None):
    children: list[Any] = []
    if on:
        children.append(icon("check"))
    children.append(text)
    return html.Button(
        children,
        id={"type": kind, "value": value},
        className="rchip on" if on else "rchip",
        n_clicks=0,
        style=style or {},
        **{"aria-pressed": "true" if on else "false"},
    )


def _filters(state: dict[str, Any], runs: list[LoadedRun]) -> html.Div:
    filters = state_filters(state)
    colours = revision_colour_map(runs)
    versions = sorted_versions(r.metadata.prompt_version.version for r in runs)
    models = distinct_values(runs, "model_name")
    tags = distinct_values(runs, "tags")
    stamps = sorted(run_time(r).astimezone(UTC).date().isoformat() for r in runs)
    pct = int(round(filters.min_pass_rate * 100))

    version_chips = []
    for v in versions:
        colour = revision_colour(v, colours)
        on = v in filters.prompt_versions
        version_chips.append(
            _chip(
                "chip-version",
                v,
                v,
                on,
                style={"background": colour.tint, "color": colour.ink},
            )
        )

    groups: list[Any] = [
        _group(
            "Date range",
            dcc.DatePickerRange(
                id="filter-date-range",
                start_date=filters.date_from,
                end_date=filters.date_to,
                min_date_allowed=stamps[0] if stamps else None,
                max_date_allowed=stamps[-1] if stamps else None,
                initial_visible_month=stamps[-1] if stamps else None,
                display_format="D MMM YYYY",
                start_date_placeholder_text=_human_date(stamps[0]) if stamps else "",
                end_date_placeholder_text=_human_date(stamps[-1]) if stamps else "",
                clearable=True,
                className="rail-dates",
            ),
        ),
        _group(
            "Prompt version",
            html.Div(version_chips, className="chip-row"),
            ", ".join(filters.prompt_versions) if filters.prompt_versions else "any",
        ),
        _group(
            "Model",
            html.Div(
                [_chip("chip-model", m, m, m in filters.model_names) for m in models],
                className="chip-row",
            ),
            ", ".join(filters.model_names) if filters.model_names else "any",
        ),
        _group(
            "Pass-rate threshold",
            dcc.Slider(
                id="filter-min-pass",
                min=0,
                max=100,
                step=5,
                value=pct,
                marks={0: "0%", 50: "50%", 100: "100%"},
                tooltip={"placement": "bottom", "always_visible": False},
                className="rail-slider",
            ),
            f"≥ {pct}%",
        ),
    ]
    if tags:
        groups.append(
            _group(
                "Tags",
                html.Div(
                    [
                        _chip("chip-tag", f"#{t}", f"#{t}", t in filters.tags)
                        for t in tags
                    ],
                    className="chip-row",
                ),
                ", ".join(f"#{t}" for t in filters.tags) if filters.tags else "any",
            )
        )

    n_active = active_filter_count(filters)
    active_text = (
        "No filters active"
        if n_active == 0
        else f"{n_active} filter{'s' if n_active != 1 else ''} active"
    )
    share = html.Div(
        [
            html.Div(active_text, className="share-count"),
            html.Button(
                [icon("link"), "Copy link to this view"],
                id="filter-copy-url",
                className="btn btn-on-house",
                n_clicks=0,
            ),
            html.Span(
                "",
                id="filter-copy-status",
                className="share-status",
                **{"aria-live": "polite"},
            ),
        ],
        className="share-card",
    )

    return html.Div(
        [
            html.Div(
                [
                    html.Span(
                        "Filters", className="rail-title", id="rail-filters-title"
                    ),
                    html.Button(
                        "Reset", id="filter-reset", className="btn-text", n_clicks=0
                    ),
                ],
                className="rail-filters-head",
            ),
            *groups,
            share,
        ],
        className="rail-filters",
        role="group",
        **{"aria-labelledby": "rail-filters-title"},
    )


def _human_date(iso_date: str) -> str:
    try:
        d = date.fromisoformat(iso_date)
    except ValueError:
        return iso_date
    return f"{d.day} {d.strftime('%b %Y')}"


def render_rail(
    state: dict[str, Any],
    runs: list[LoadedRun],
    *,
    active: str,
    counts: dict[str, int],
    agent_display_name: str,
    agent_name: str,
    seed_mode: bool,
) -> html.Aside:
    compact = active == "explorer"
    children: list[Any] = [_brand(compact)]
    if not compact:
        children.append(_agent_card(agent_display_name, agent_name, seed_mode))
    children.append(_nav(active, counts, compact))
    if active == "summary":
        children.append(_filters(state, runs))
    return html.Aside(
        children,
        className="rail rail-compact" if compact else "rail",
        **{"aria-label": "HoloDeck test view"},
    )
