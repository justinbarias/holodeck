"""Compare data-assembly tests (US5 T424–T429)."""

from __future__ import annotations

import pandas as pd
import pytest

from holodeck.dashboard.compare_data import (
    COMPARE_PALETTE,
    Callout,
    ConfigRow,
    RunStats,
    StatRow,
    compute_case_matrix,
    compute_compare_callouts,
    compute_config_diff,
    compute_summary_rows,
    delta_pill_class,
    run_stats,
)
from holodeck.dashboard.seed_data import build_seed_runs


@pytest.fixture(scope="module")
def seed_runs():
    return build_seed_runs()


# ---------- T424: run_stats + cost fallback -------------------------------


@pytest.mark.unit
def test_run_stats_synthetic_cost_for_sonnet(seed_runs):
    run = next(r for r in seed_runs if "sonnet" in r.metadata.agent_config.model.name)
    stats = run_stats(run)
    assert isinstance(stats, RunStats)
    expected_pr = run.report.summary.passed / run.report.summary.total_tests
    assert stats.pass_rate == pytest.approx(expected_pr)
    assert stats.passed == run.report.summary.passed
    assert stats.total == run.report.summary.total_tests
    assert stats.duration_ms == run.report.summary.total_duration_ms
    # All seed token_usage is None, so we're in synthetic mode
    assert stats.total_tokens is None
    expected_cost = (run.report.summary.total_duration_ms / 1000) * 0.018
    assert stats.est_cost == pytest.approx(expected_cost, rel=1e-4)


@pytest.mark.unit
def test_run_stats_synthetic_cost_default_rate(seed_runs):
    run = next(
        (r for r in seed_runs if "sonnet" not in r.metadata.agent_config.model.name),
        None,
    )
    if run is None:
        pytest.skip("seed has no non-sonnet runs")
    stats = run_stats(run)
    expected_cost = (run.report.summary.total_duration_ms / 1000) * 0.012
    assert stats.est_cost == pytest.approx(expected_cost, rel=1e-4)


# ---------- T425: case matrix ---------------------------------------------


@pytest.mark.unit
def test_compute_case_matrix_shape(seed_runs):
    runs = seed_runs[:3]
    df = compute_case_matrix(runs)
    assert isinstance(df, pd.DataFrame)
    # Columns: one score/passed/regression/improvement group per run plus the name
    assert "case_name" in df.columns
    for run in runs:
        rid = run.report.timestamp
        assert f"score::{rid}" in df.columns
        assert f"passed::{rid}" in df.columns
        assert f"regression::{rid}" in df.columns
        assert f"improvement::{rid}" in df.columns
    # Case names sorted alphabetically
    names = df["case_name"].tolist()
    assert names == sorted(names)


# ---------- T426: regression/improvement flags ----------------------------


@pytest.mark.unit
def test_regression_improvement_flags_from_baseline(seed_runs):
    runs = seed_runs[:2]
    df = compute_case_matrix(runs)
    base_id = runs[0].report.timestamp
    other_id = runs[1].report.timestamp

    # Baseline must never have regression/improvement flags set
    assert not df[f"regression::{base_id}"].any()
    assert not df[f"improvement::{base_id}"].any()

    for _, row in df.iterrows():
        name = row["case_name"]
        base_passed = row[f"passed::{base_id}"]
        other_passed = row[f"passed::{other_id}"]
        if base_passed and not other_passed:
            assert row[f"regression::{other_id}"], name
        elif (not base_passed) and other_passed:
            assert row[f"improvement::{other_id}"], name


# ---------- T427: config diff ---------------------------------------------


@pytest.mark.unit
def test_compute_config_diff_labels_and_same_flag(seed_runs):
    runs = seed_runs[:3]
    rows = compute_config_diff(runs)
    labels = [r.label for r in rows]
    assert labels == [
        "prompt_version",
        "model_name",
        "temperature",
        "tags_joined",
        "git_commit",
        "extended_thinking",
    ]
    for r in rows:
        assert isinstance(r, ConfigRow)
        assert len(r.values) == len(runs)


# ---------- T428: callouts ------------------------------------------------


@pytest.mark.unit
def test_compute_compare_callouts_limits_to_three(seed_runs):
    runs = seed_runs[:3]
    callouts = compute_compare_callouts(runs)
    # Baseline is skipped; one Callout per non-baseline run
    assert len(callouts) == len(runs) - 1
    for c in callouts:
        assert isinstance(c, Callout)
        assert len(c.regressions) <= 3
        assert len(c.improvements) <= 3
        assert c.regressions_extra >= 0
        assert c.improvements_extra >= 0


# ---------- T429: summary rows --------------------------------------------


@pytest.mark.unit
def test_compute_summary_rows_polarity_and_token_omission(seed_runs):
    runs = seed_runs[:3]
    rows = compute_summary_rows(runs)
    assert all(isinstance(r, StatRow) for r in rows)
    labels = {r.key: r for r in rows}
    # total_tokens omitted when every run is in synthetic mode (seed)
    assert "total_tokens" not in labels
    # Duration & cost inverted
    assert labels["duration_ms"].delta_polarity == "invert"
    assert labels["est_cost"].delta_polarity == "invert"
    assert labels["pass_rate"].delta_polarity == "normal"


# ---------- delta_pill_class ----------------------------------------------


@pytest.mark.unit
def test_delta_pill_class_normal_and_invert():
    assert delta_pill_class(0) == "hd-delta-neutral"
    assert delta_pill_class(0.1) == "hd-delta-pos"
    assert delta_pill_class(-0.1) == "hd-delta-neg"
    # Inverted polarity: lower value = positive
    assert delta_pill_class(0.1, invert=True) == "hd-delta-neg"
    assert delta_pill_class(-0.1, invert=True) == "hd-delta-pos"


# ---------- palette constant ----------------------------------------------


@pytest.mark.unit
def test_palette_constant():
    assert COMPARE_PALETTE == ["#7bff5a", "#5ae0a6", "#ffcf5a"]
