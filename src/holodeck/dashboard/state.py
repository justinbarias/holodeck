"""Pure helpers for the `dcc.Store`-backed app state.

All functions take/return plain JSON-serializable dicts so they're trivially
unit-testable and work with Dash's `dcc.Store` without serialization glue.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from holodeck.dashboard.filters import (
    Filters,
    filters_from_query_params,
    filters_to_query_params,
)


def default_state() -> dict[str, Any]:
    return {
        "tab": "summary",
        "filters": asdict(Filters()),
        "explorer_run_id": None,
        "explorer_case_name": None,
        "compare_queue": [],
        "compare_variant": 1,
    }


def state_filters(state: dict[str, Any]) -> Filters:
    f = state.get("filters") or {}
    return Filters(**f)


def set_filters(state: dict[str, Any], filters: Filters) -> dict[str, Any]:
    new_state = {**state, "filters": asdict(filters)}
    return new_state


def set_tab(state: dict[str, Any], tab: str) -> dict[str, Any]:
    return {**state, "tab": tab}


def push_to_compare_queue(state: dict[str, Any], run_id: str) -> dict[str, Any]:
    q = list(state.get("compare_queue") or [])
    if run_id in q:
        return state
    if len(q) >= 3:
        q = q[1:]
    q.append(run_id)
    return {**state, "compare_queue": q}


def remove_from_compare_queue(state: dict[str, Any], run_id: str) -> dict[str, Any]:
    q = [r for r in (state.get("compare_queue") or []) if r != run_id]
    return {**state, "compare_queue": q}


def open_in_explorer(state: dict[str, Any], run_id: str) -> dict[str, Any]:
    return {**state, "tab": "explorer", "explorer_run_id": run_id}


def url_search_from_state(state: dict[str, Any]) -> str:
    filters = state_filters(state)
    params = filters_to_query_params(filters)
    if state.get("tab") and state["tab"] != "summary":
        params["tab"] = state["tab"]
    if not params:
        return ""
    return "?" + "&".join(f"{k}={v}" for k, v in params.items())


def state_from_url_search(search: str) -> dict[str, Any]:
    state = default_state()
    if not search:
        return state
    search = search.lstrip("?")
    if not search:
        return state
    params: dict[str, str] = {}
    for part in search.split("&"):
        if "=" in part:
            k, v = part.split("=", 1)
            params[k] = v
    if "tab" in params and params["tab"] in ("summary", "explorer", "compare"):
        state = set_tab(state, params.pop("tab"))
    filters = filters_from_query_params(params)
    return set_filters(state, filters)
