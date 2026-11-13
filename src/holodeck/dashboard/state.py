"""Pure helpers for the `dcc.Store`-backed app state.

All functions take/return plain JSON-serializable dicts so they're trivially
unit-testable and work with Dash's `dcc.Store` without serialization glue.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any
from urllib.parse import parse_qsl, urlencode

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


_TABS = ("summary", "explorer", "compare")


def url_search_from_state(state: dict[str, Any]) -> str:
    """Query string for the address bar: filters, view, compare selection
    and the Explorer selection. :func:`state_from_url_search` reverses it."""
    params = filters_to_query_params(state_filters(state))
    if state.get("tab") and state["tab"] != "summary":
        params["tab"] = state["tab"]
    queue = list(state.get("compare_queue") or [])
    if queue:
        params["cmp"] = ",".join(queue)
    variant = int(state.get("compare_variant", 1) or 1)
    if variant != 1:
        params["layout"] = str(variant)
    if state.get("explorer_run_id"):
        params["run"] = str(state["explorer_run_id"])
    if state.get("explorer_case_name"):
        params["case"] = str(state["explorer_case_name"])
    if not params:
        return ""
    return "?" + urlencode(params, safe=",:")


def state_from_url_search(search: str) -> dict[str, Any]:
    """Rebuild app state from a query string written by
    :func:`url_search_from_state`; unknown or malformed values fall back to
    the defaults."""
    state = default_state()
    query = (search or "").lstrip("?")
    if not query:
        return state
    params = dict(parse_qsl(query))
    tab = params.pop("tab", None)
    if tab in _TABS:
        state = set_tab(state, tab)
    cmp = params.pop("cmp", "")
    if cmp:
        state["compare_queue"] = [r for r in cmp.split(",") if r][:3]
    layout = params.pop("layout", "")
    if layout in ("1", "2", "3"):
        state["compare_variant"] = int(layout)
    run_id = params.pop("run", "")
    if run_id:
        state["explorer_run_id"] = run_id
    case = params.pop("case", "")
    if case:
        state["explorer_case_name"] = case
    return set_filters(state, filters_from_query_params(params))
