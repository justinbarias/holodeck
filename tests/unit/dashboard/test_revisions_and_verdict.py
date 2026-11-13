"""Revision colours, cue run numbers, and the Summary verdict (seed fixture)."""

from __future__ import annotations

import pytest

from holodeck.dashboard.compare_data import (
    quick_pick_best_worst,
    quick_pick_latest,
    quick_pick_latest_per_revision,
)
from holodeck.dashboard.filters import Filters, apply
from holodeck.dashboard.revisions import (
    REVISION_PALETTE,
    UNKNOWN_REVISION,
    revision_colour,
    revision_colour_map,
    run_label,
    run_numbers,
    sorted_versions,
)
from holodeck.dashboard.seed_data import build_seed_runs
from holodeck.dashboard.verdict import (
    FailingCase,
    build_verdict,
    find_regressions,
)
from holodeck.models.eval_run import EvalRun


@pytest.fixture(scope="module")
def runs() -> list[EvalRun]:
    return sorted(build_seed_runs(), key=lambda r: r.report.timestamp)


# ---------- revision colours ----------------------------------------------


@pytest.mark.unit
def test_versions_sort_naturally() -> None:
    assert sorted_versions(["v1.10", "v1.2.1", "v1.9", "v1.2", "v1.2"]) == [
        "v1.2",
        "v1.2.1",
        "v1.9",
        "v1.10",
    ]


@pytest.mark.unit
def test_nth_version_takes_nth_colour(runs: list[EvalRun]) -> None:
    cmap = revision_colour_map(runs)
    assert list(cmap) == ["v1.0", "v1.1", "v1.2", "v1.2.1", "v1.3", "v1.4"]
    assert cmap["v1.0"].tint == "#cfe0f8" and cmap["v1.0"].ink == "#234f99"
    assert cmap["v1.2.1"].name == "green"
    assert cmap["v1.4"].tint == "#e8dcc4" and cmap["v1.4"].ink == "#5b4a2a"
    assert revision_colour("v9.9", cmap) is UNKNOWN_REVISION


@pytest.mark.unit
def test_colours_do_not_shift_under_a_filter(runs: list[EvalRun]) -> None:
    full = revision_colour_map(runs)
    only_late = [r for r in runs if r.metadata.prompt_version.version == "v1.4"]
    # Views always build the map from every loaded run; a subset would
    # renumber, which is why callers must not pass filtered runs.
    assert revision_colour_map(only_late)["v1.4"] != full["v1.4"]
    assert full["v1.4"] == REVISION_PALETTE[5]


@pytest.mark.unit
def test_palette_wraps_after_eight() -> None:
    versions = [f"v{i}" for i in range(1, 10)]
    from types import SimpleNamespace

    fake = [
        SimpleNamespace(
            metadata=SimpleNamespace(prompt_version=SimpleNamespace(version=v))
        )
        for v in versions
    ]
    cmap = revision_colour_map(fake)  # type: ignore[arg-type]
    assert cmap["v8"].name == "cherry"
    assert cmap["v9"] == REVISION_PALETTE[0]


# ---------- R numbering ---------------------------------------------------


@pytest.mark.unit
def test_runs_are_numbered_chronologically(runs: list[EvalRun]) -> None:
    numbers = run_numbers(reversed(runs))
    assert numbers["2026-03-07T09:00:09.812000Z"] == 1
    assert numbers["2026-04-16T22:59:09.812000Z"] == 24
    assert run_label("2026-04-01T03:02:09.812000Z", numbers) == "R15"
    assert run_label("2026-05-01T00:00:00Z", numbers) == "2026-05-01T00:00"


# ---------- regressions + verdict ----------------------------------------


@pytest.mark.unit
def test_regressions_in_seed(runs: list[EvalRun]) -> None:
    regs = find_regressions(runs, run_numbers(runs))
    assert [(r.number, r.label, r.version, r.previous_label) for r in regs] == [
        (1, "R5", "v1.1", "R4"),
        (2, "R15", "v1.2.1", "R14"),
    ]
    assert regs[0].change_text() == "58.3% → 50.0%"
    assert regs[1].change_text() == "83.3% → 75.0%"
    assert regs[1].newly_failed == ()


@pytest.mark.unit
def test_verdict_for_latest_seed_run(runs: list[EvalRun]) -> None:
    v = build_verdict(runs, run_numbers(runs))
    assert v is not None
    assert v.headline() == "R24 passes 11 of 12 cases, the same as R23."
    assert v.duration_sentence() == "Run time is 0.5 s longer than R23."
    assert v.failing == ()
    assert not v.is_regression


@pytest.mark.unit
def test_verdict_names_a_regression_and_its_persistent_failure(
    runs: list[EvalRun],
) -> None:
    v = build_verdict(runs[:15], run_numbers(runs))
    assert v is not None
    assert v.headline() == (
        "R15 is a regression: it passes 9 of 12 cases, 1 fewer than R14."
    )
    assert v.regression_number == 2
    assert [f.name for f in v.failing] == ["greeting_smalltalk"]
    fc = v.failing[0]
    assert (fc.streak, fc.streak_start_label) == (3, "R13")
    assert fc.sentence_tail(v.label, v.previous_label) == (
        " has failed in every run since R13."
    )


@pytest.mark.unit
def test_verdict_orders_failures_by_streak_and_flags_new_ones(
    runs: list[EvalRun],
) -> None:
    v = build_verdict(runs[:13], run_numbers(runs))
    assert v is not None
    assert v.headline() == "R13 passes 10 of 12 cases, 1 more than R12."
    assert [(f.name, f.streak) for f in v.failing] == [
        ("escalate_to_human", 3),
        ("refund_eligible_standard", 3),
        ("cancel_subscription", 2),
        ("password_reset", 2),
        ("greeting_smalltalk", 1),
        ("multi_item_return", 1),
    ]
    new = v.failing[-1]
    assert new.passed_in_previous
    assert new.sentence_tail("R13", "R12") == " fails in R13 after passing in R12."


@pytest.mark.unit
def test_failing_sentence_includes_metric_below_threshold() -> None:
    fc = FailingCase(
        name="billing_dispute",
        streak=11,
        streak_start_label="R14",
        passed_in_previous=False,
        metric_name="policy_compliance",
        score=0.64,
        threshold=0.70,
    )
    assert fc.sentence_tail("R24", "R23") == (
        " has failed in every run since R14 (policy_compliance 0.64 against 0.70)."
    )


@pytest.mark.unit
def test_single_run_verdict(runs: list[EvalRun]) -> None:
    v = build_verdict(runs[:1], run_numbers(runs))
    assert v is not None
    assert v.headline() == "R1 passes 7 of 12 cases."
    assert v.duration_sentence() is None
    assert build_verdict([], {}) is None


# ---------- quick picks + date filter ------------------------------------


@pytest.mark.unit
def test_quick_picks(runs: list[EvalRun]) -> None:
    numbers = run_numbers(runs)

    def labels(ids: list[str]) -> list[str]:
        return [run_label(i, numbers) for i in ids]

    assert labels(quick_pick_latest(runs, 2)) == ["R23", "R24"]
    assert labels(quick_pick_best_worst(runs)) == ["R6", "R24"]
    assert labels(quick_pick_latest_per_revision(runs)) == ["R16", "R20", "R24"]


@pytest.mark.unit
def test_date_filter_is_inclusive_and_timezone_safe(runs: list[EvalRun]) -> None:
    f = Filters(date_from="2026-04-01", date_to="2026-04-16")
    numbers = run_numbers(runs)
    picked = [run_label(r.report.timestamp, numbers) for r in apply(f, runs)]
    assert picked[0] == "R15"
    assert picked[-1] == "R24"  # 16 Apr 22:59 is inside an inclusive end date
    assert len(picked) == 10


# ---------- regression rows name the case; verdict link targets it ------


@pytest.mark.unit
def test_regression_row_names_first_newly_failed_case_and_more() -> None:
    from holodeck.dashboard.verdict import Regression

    reg = Regression(
        number=1,
        run_id="x",
        label="R5",
        version="v1.1",
        previous_label="R4",
        previous_rate=7 / 12,
        rate=6 / 12,
        newly_failed=("refund_outside_window", "order_status_query", "billing"),
    )
    assert reg.case_text() == "refund_outside_window +2 more"
    assert reg.change_text() == ("58.3% → 50.0% · refund_outside_window +2 more failed")
    one = Regression(1, "x", "R5", "v1.1", "R4", 7 / 12, 6 / 12, ("a_case",))
    assert one.change_text() == "58.3% → 50.0% · a_case failed"


@pytest.mark.unit
def test_seed_regressions_have_no_pass_to_fail_case(runs: list[EvalRun]) -> None:
    # The seed's per-case flags do not change from pass to fail at R5 or R15
    # (its summary counts and case flags disagree), so no case is named.
    regs = find_regressions(runs, run_numbers(runs))
    assert [(r.case_text(), r.change_text()) for r in regs] == [
        (None, "58.3% → 50.0%"),
        (None, "83.3% → 75.0%"),
    ]


@pytest.mark.unit
def test_r15_verdict_link_opens_the_persistent_failing_case(
    runs: list[EvalRun],
) -> None:
    import json

    from plotly.utils import PlotlyJSONEncoder

    from holodeck.dashboard.state import default_state
    from holodeck.dashboard.views.summary import render_summary

    state = {
        **default_state(),
        "filters": {
            **default_state()["filters"],
            "date_to": "2026-04-01",
        },
    }
    tree = render_summary(state, runs)
    rendered = json.dumps(tree, cls=PlotlyJSONEncoder)
    assert "R15 is a regression: it passes 9 of 12 cases, 1 fewer than R14." in rendered
    assert " has failed in every run since R13." in rendered
    assert (
        '"id": {"type": "open-case", "run_id": "2026-04-01T03:02:09.812000Z", '
        '"case_name": "greeting_smalltalk"}'
    ) in rendered


@pytest.mark.unit
def test_tray_hides_open_compare_on_compare_view(runs: list[EvalRun]) -> None:
    from holodeck.dashboard.components.compare_tray import render_tray

    queue = [r.report.timestamp for r in runs[-2:]]
    for tab, hidden in (("summary", False), ("compare", True)):
        tray = render_tray({"tab": tab, "compare_queue": queue}, runs)
        button = next(
            c for c in tray.children if getattr(c, "id", None) == "compare-tray-open"
        )
        assert bool(getattr(button, "hidden", False)) is hidden
