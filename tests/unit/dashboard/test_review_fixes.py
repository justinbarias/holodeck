"""Regression tests for the adversarial-review findings on the dashboard.

Runs are copies of the seed fixture with only the field under test changed;
the seed file itself is untouched.
"""

from __future__ import annotations

import io
import json
from collections.abc import Iterator
from csv import DictReader
from pathlib import Path

import pytest
from dash import html
from dash.development.base_component import Component
from dash.exceptions import PreventUpdate
from plotly.utils import PlotlyJSONEncoder

from holodeck.dashboard import app as app_mod
from holodeck.dashboard.compare_data import compute_case_matrix
from holodeck.dashboard.components.compare_tray import render_tray
from holodeck.dashboard.data_loader import load_all
from holodeck.dashboard.explorer_data import ExpectedToolsCoverage, select_run
from holodeck.dashboard.filters import Filters
from holodeck.dashboard.loaded_run import LoadedRun, run_id_from_timestamp
from holodeck.dashboard.revisions import (
    chronological,
    revision_colour_map,
    run_label,
    run_numbers,
)
from holodeck.dashboard.seed_data import build_seed_runs
from holodeck.dashboard.state import (
    default_state,
    open_in_explorer,
    push_to_compare_queue,
    set_filters,
    state_from_url_search,
    url_search_from_state,
)
from holodeck.dashboard.verdict import build_verdict
from holodeck.dashboard.views.compare import (
    _Ctx,
    _narrative,
    render_compare,
    render_compare_header,
)
from holodeck.dashboard.views.explorer import _expected_tools_section, render_explorer
from holodeck.dashboard.views.summary import render_summary
from holodeck.lib.eval_run.writer import write_eval_run


@pytest.fixture(scope="module")
def seed() -> list[LoadedRun]:
    runs: list[LoadedRun] = chronological(build_seed_runs())
    return runs


def _copy(run: LoadedRun, timestamp: str, suffix: str = "") -> LoadedRun:
    """Copy of ``run`` saved at ``timestamp``; ``suffix`` mimics the writer's
    collision suffix so equal timestamps keep distinct ids."""
    out = run.model_copy(
        deep=True, update={"run_id": run_id_from_timestamp(timestamp) + suffix}
    )
    out.report.timestamp = timestamp
    return out


def _json(node: object) -> str:
    return json.dumps(node, cls=PlotlyJSONEncoder, ensure_ascii=False)


# ---------- #1 mixed naive and aware timestamps --------------------------


@pytest.mark.unit
def test_summary_renders_with_naive_and_aware_timestamps(seed: list[LoadedRun]) -> None:
    runs = [
        _copy(seed[0], "2026-04-01T09:00:00"),
        _copy(seed[1], "2026-04-02T09:00:00Z"),
    ]
    rendered = _json(render_summary(default_state(), runs))
    assert "R2 passes 7 of 12 cases, the same as R1." in rendered
    assert "1 days" in rendered  # runs-in-view span, from aware datetimes


# ---------- #2 order by instant, not by string ----------------------------


@pytest.mark.unit
def test_offset_timestamps_sort_by_instant(
    seed: list[LoadedRun], monkeypatch: pytest.MonkeyPatch
) -> None:
    # 10:00+10:00 is 00:00Z, so it happened before 01:00Z although it sorts
    # after it as a string.
    early = _copy(seed[0], "2026-11-13T10:00:00+10:00")  # 7 of 12 pass
    late = _copy(seed[23], "2026-11-13T01:00:00Z")  # 11 of 12 pass
    runs = [late, early]
    numbers = run_numbers(runs)
    assert (run_label(early, numbers), run_label(late, numbers)) == ("R1", "R2")
    verdict = build_verdict(runs, numbers)
    assert verdict is not None
    assert verdict.headline() == "R2 passes 11 of 12 cases, 4 more than R1."
    monkeypatch.setattr(app_mod, "get_runs", lambda: runs)
    download = app_mod.on_download_latest(1)
    assert download["filename"] == "holodeck-run-2026-11-13T01-00-00Z.json"


# ---------- #7 copied link round-trips ------------------------------------


@pytest.mark.unit
def test_url_search_round_trips_state(seed: list[LoadedRun]) -> None:
    state = set_filters(
        default_state(),
        Filters(
            date_from="2026-03-20",
            prompt_versions=["v1.2.1", "v1.4"],
            min_pass_rate=0.75,
            tags=["rag-tuning"],
            metric_kind="geval",
            search="a3f9",
        ),
    )
    state = {
        **state,
        "tab": "compare",
        "compare_queue": [seed[14].run_id, "2026-11-13T10-00-00+10-00"],
        "compare_variant": 3,
        "explorer_run_id": seed[20].run_id,
        "explorer_case_name": "billing_dispute",
    }
    search = url_search_from_state(state)
    assert "%2B10-00" in search  # "+" survives as an encoded offset
    assert state_from_url_search(search) == state


@pytest.mark.unit
def test_copied_link_restores_filter_controls(
    seed: list[LoadedRun], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(app_mod, "get_runs", lambda: seed)
    search = "?versions=v1.4&min=0.75&cmp=" + seed[23].run_id
    restored = app_mod.restore_from_url(search, default_state())
    assert restored["filters"]["prompt_versions"] == ["v1.4"]
    assert restored["compare_queue"] == [seed[23].run_id]
    rail, header, view, _tray = app_mod.render_all(restored, 0)
    pressed = [
        n
        for n in _walk(rail)
        if getattr(n, "id", None) == {"type": "chip-version", "value": "v1.4"}
    ]
    assert pressed and pressed[0].to_plotly_json()["props"]["aria-pressed"] == "true"
    slider = next(n for n in _walk(rail) if getattr(n, "id", None) == "filter-min-pass")
    assert slider.to_plotly_json()["props"]["value"] == 75
    assert "4 of 24 runs match" in _json(header)
    # Address bar already matches: no second update, so no loop with sync_url.
    with pytest.raises(PreventUpdate):
        app_mod.restore_from_url(app_mod.sync_url(restored), restored)


def _run_rows(tree: Component, type_: str) -> dict[str, str]:
    """``run_id`` to class name for each pattern-id component of ``type_``."""
    out: dict[str, str] = {}
    for node in _walk(tree):
        props = node.to_plotly_json()["props"]
        node_id = props.get("id")
        if isinstance(node_id, dict) and node_id.get("type") == type_:
            out[str(node_id["run_id"])] = str(props.get("className", ""))
    return out


def _walk(node: Component) -> Iterator[Component]:
    yield node
    children = getattr(node, "children", None)
    if isinstance(children, list | tuple):
        for c in children:
            if isinstance(c, Component):
                yield from _walk(c)
    elif isinstance(children, Component):
        yield from _walk(children)


# ---------- #8 narrative counts every change -----------------------------


@pytest.mark.unit
def test_narrative_counts_more_than_three_regressions(seed: list[LoadedRun]) -> None:
    baseline = seed[23]  # every case saved as passed
    worse = _copy(seed[23], "2026-04-20T00:00:00Z")
    for case in worse.report.results[:4]:
        case.passed = False
    runs = [baseline, worse]
    text = _json(_narrative(runs, _Ctx([*seed, worse])))
    assert "4 regressions" in text
    for case in worse.report.results[:4]:
        assert (case.test_name or "") in text


# ---------- #11 best vs worst never falls back to all runs ---------------


@pytest.mark.unit
def test_best_vs_worst_with_empty_range_picks_nothing(seed: list[LoadedRun]) -> None:
    state = set_filters(default_state(), Filters(date_from="2030-01-01"))
    assert app_mod.quick_pick_queue("compare-quick-best", state, seed) == []
    tree = render_compare(state, seed)
    best = next(
        n for n in _walk(tree) if getattr(n, "id", None) == "compare-quick-best"
    )
    assert best.to_plotly_json()["props"]["disabled"] is True
    assert "No runs match the Summary filters" in _json(tree)


# ---------- #12 repeated unexpected calls point at their own cue ---------


@pytest.mark.unit
def test_repeated_unexpected_calls_get_their_own_cues() -> None:
    cov = ExpectedToolsCoverage(
        total=1,
        matched=0,
        missed=1,
        unexpected=2,
        rows=[("expected_y", False)],
        unexpected_names=["unexpected_x", "unexpected_x"],
    )
    text = _json(_expected_tools_section(cov, ["unexpected_x", "unexpected_x"]))
    assert "Unexpected call at Q1" in text
    assert "Unexpected call at Q2" in text


# ---------- #13 equal timestamps keep distinct numbers -------------------


@pytest.mark.unit
def test_equal_timestamps_get_distinct_numbers(seed: list[LoadedRun]) -> None:
    a = _copy(seed[0], "2026-05-01T00:00:00Z")
    b = _copy(seed[1], "2026-05-01T00:00:00Z", "-b2c3")
    numbers = run_numbers([a, b])
    assert [run_label(a, numbers), run_label(b, numbers)] == ["R1", "R2"]
    text = _json(render_summary(default_state(), [a, b]))
    assert "R2 passes 7 of 12 cases, the same as R1." in text


@pytest.mark.unit
def test_equal_timestamps_open_compare_and_export_as_two_runs(
    seed: list[LoadedRun], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stamp = "2026-05-01T00:00:00Z"
    paths = [
        write_eval_run(_copy(seed[i], stamp), tmp_path) for i in (0, 23)
    ]  # the writer suffixes the second file name, which then loads first
    runs = load_all(paths[0].parent)
    a, b = runs
    assert a.run_id != b.run_id
    assert {a.run_id, b.run_id} == {p.stem for p in paths}
    assert a.report.timestamp == b.report.timestamp == stamp
    assert "run_id" not in paths[0].read_text(encoding="utf-8")
    monkeypatch.setattr(app_mod, "get_runs", lambda: runs)

    # Open: each id selects and highlights its own run.
    for run in runs:
        state = open_in_explorer(default_state(), run.run_id)
        assert select_run(runs, run.run_id) is run
        rows = _run_rows(render_explorer(state, runs), "explorer-run-row")
        assert rows == {
            r.run_id: "list-row on" if r is run else "list-row" for r in runs
        }
    summary_rows = _run_rows(render_summary(default_state(), runs), "run-row")
    assert set(summary_rows) == {a.run_id, b.run_id}

    # Compare: both runs queue, get their own matrix columns and tray chips.
    state = push_to_compare_queue(default_state(), a.run_id)
    state = push_to_compare_queue(state, b.run_id)
    assert state["compare_queue"] == [a.run_id, b.run_id]
    matrix = compute_case_matrix(runs)
    passed = {r.run_id: int(matrix[f"passed::{r.run_id}"].sum()) for r in runs}
    assert passed == {r.run_id: sum(c.passed for c in r.report.results) for r in runs}
    assert len(set(passed.values())) == 2
    tray = _json(render_tray(state, runs))
    assert "R1 · " in tray and "R2 · " in tray
    assert "2 runs · baseline R1" in _json(render_compare_header(state, runs))

    # Copied link: the queue and Explorer selection survive the round trip.
    linked = {**state, "explorer_run_id": b.run_id}
    assert state_from_url_search(url_search_from_state(linked)) == linked

    # Export: one CSV row per run, each with its own id.
    csv = app_mod.on_export_csv(1, default_state())["content"]
    csv_rows = list(DictReader(io.StringIO(csv)))
    assert sorted(r["id"] for r in csv_rows) == sorted([a.run_id, b.run_id])
    assert {r["passed"] for r in csv_rows} == {"7", "11"}


# ---------- #15 colours survive a filter in a rendered view --------------


@pytest.mark.unit
@pytest.mark.parametrize("versions", [["v1.4"], ["v1.2.1", "v1.3"]])
def test_rendered_chips_keep_unfiltered_colours(
    seed: list[LoadedRun], versions: list[str]
) -> None:
    full = revision_colour_map(seed)
    state = set_filters(default_state(), Filters(prompt_versions=versions))
    tree = render_summary(state, seed)
    chips = [
        n.to_plotly_json()["props"]
        for n in _walk(tree)
        if isinstance(n, html.Span)
        and str(n.to_plotly_json()["props"].get("className", "")).split()[:1] == ["rv"]
    ]
    seen = {c["children"]: c["style"]["background"] for c in chips}
    assert set(seen) == set(versions)
    for version, tint in seen.items():
        assert tint == full[version].tint, version
