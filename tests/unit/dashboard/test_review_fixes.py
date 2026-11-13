"""Regression tests for the adversarial-review findings on the dashboard.

Runs are copies of the seed fixture with only the field under test changed;
the seed file itself is untouched.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

import pytest
from dash import html
from dash.development.base_component import Component
from dash.exceptions import PreventUpdate
from plotly.utils import PlotlyJSONEncoder

from holodeck.dashboard import app as app_mod
from holodeck.dashboard.explorer_data import ExpectedToolsCoverage
from holodeck.dashboard.filters import Filters
from holodeck.dashboard.revisions import (
    chronological,
    revision_colour_map,
    run_label,
    run_numbers,
)
from holodeck.dashboard.seed_data import build_seed_runs
from holodeck.dashboard.state import (
    default_state,
    set_filters,
    state_from_url_search,
    url_search_from_state,
)
from holodeck.dashboard.verdict import build_verdict
from holodeck.dashboard.views.compare import _Ctx, _narrative, render_compare
from holodeck.dashboard.views.explorer import _expected_tools_section
from holodeck.dashboard.views.summary import render_summary
from holodeck.models.eval_run import EvalRun


@pytest.fixture(scope="module")
def seed() -> list[EvalRun]:
    runs: list[EvalRun] = chronological(build_seed_runs())
    return runs


def _copy(run: EvalRun, timestamp: str) -> EvalRun:
    out = run.model_copy(deep=True)
    out.report.timestamp = timestamp
    return out


def _json(node: object) -> str:
    return json.dumps(node, cls=PlotlyJSONEncoder, ensure_ascii=False)


# ---------- #1 mixed naive and aware timestamps --------------------------


@pytest.mark.unit
def test_summary_renders_with_naive_and_aware_timestamps(seed: list[EvalRun]) -> None:
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
    seed: list[EvalRun], monkeypatch: pytest.MonkeyPatch
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
def test_url_search_round_trips_state(seed: list[EvalRun]) -> None:
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
        "compare_queue": [seed[14].report.timestamp, "2026-11-13T10:00:00+10:00"],
        "compare_variant": 3,
        "explorer_run_id": seed[20].report.timestamp,
        "explorer_case_name": "billing_dispute",
    }
    search = url_search_from_state(state)
    assert "%2B10:00" in search  # "+" survives as an encoded offset
    assert state_from_url_search(search) == state


@pytest.mark.unit
def test_copied_link_restores_filter_controls(
    seed: list[EvalRun], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(app_mod, "get_runs", lambda: seed)
    search = "?versions=v1.4&min=0.75&cmp=" + seed[23].report.timestamp
    restored = app_mod.restore_from_url(search, default_state())
    assert restored["filters"]["prompt_versions"] == ["v1.4"]
    assert restored["compare_queue"] == [seed[23].report.timestamp]
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
def test_narrative_counts_more_than_three_regressions(seed: list[EvalRun]) -> None:
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
def test_best_vs_worst_with_empty_range_picks_nothing(seed: list[EvalRun]) -> None:
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
def test_equal_timestamps_get_distinct_numbers(seed: list[EvalRun]) -> None:
    a = _copy(seed[0], "2026-05-01T00:00:00Z")
    b = _copy(seed[1], "2026-05-01T00:00:00Z")
    numbers = run_numbers([a, b])
    assert [run_label(a, numbers), run_label(b, numbers)] == ["R1", "R2"]
    text = _json(render_summary(default_state(), [a, b]))
    assert "R2 passes 7 of 12 cases, the same as R1." in text


# ---------- #15 colours survive a filter in a rendered view --------------


@pytest.mark.unit
@pytest.mark.parametrize("versions", [["v1.4"], ["v1.2.1", "v1.3"]])
def test_rendered_chips_keep_unfiltered_colours(
    seed: list[EvalRun], versions: list[str]
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
