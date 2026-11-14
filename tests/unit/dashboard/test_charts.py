"""Pass-rate chart labels and time axes on real-shaped run histories.

Runs are copies of the seed fixture with only the timestamp and prompt
version changed; the seed file itself is untouched.
"""

from __future__ import annotations

import pytest

from holodeck.dashboard.charts import (
    TIME_TICK_FORMATS,
    band_label_fits,
    metric_trend_chart,
    pass_rate_chart,
)
from holodeck.dashboard.revisions import chronological
from holodeck.dashboard.seed_data import build_seed_runs
from holodeck.models.eval_run import EvalRun


def _runs(spec: list[tuple[str, str]]) -> list[EvalRun]:
    """Seed copies with the given ``(timestamp, prompt version)`` pairs."""
    seed = chronological(build_seed_runs())
    out: list[EvalRun] = []
    for i, (timestamp, version) in enumerate(spec):
        run = seed[i % len(seed)].model_copy(deep=True)
        run.report.timestamp = timestamp
        run.metadata.prompt_version.version = version
        out.append(run)
    return out


def _band_labels(runs: list[EvalRun]) -> list[str]:
    fig = pass_rate_chart(runs)
    return [str(a.text) for a in fig.layout.annotations]


@pytest.mark.unit
def test_narrow_bands_are_not_labelled() -> None:
    # The warehouse-concierge history: two short versions, then one long one,
    # all on one day. Only the long band has room for its name.
    runs = _runs(
        [
            ("2026-09-06T08:25:33Z", "auto-48c1dda6"),
            ("2026-09-06T08:27:56Z", "auto-d1aa0519"),
            ("2026-09-06T08:32:03Z", "auto-d545d6f7"),
            ("2026-09-06T08:34:45Z", "auto-d545d6f7"),
            ("2026-09-06T08:36:45Z", "auto-d545d6f7"),
            ("2026-09-06T08:39:15Z", "auto-d545d6f7"),
            ("2026-09-06T10:24:02Z", "auto-d545d6f7"),
            ("2026-09-06T11:15:16Z", "auto-d545d6f7"),
        ]
    )
    assert _band_labels(runs) == ["<b>auto-d545d6f7</b>"]


@pytest.mark.unit
def test_wide_bands_keep_their_labels() -> None:
    runs = _runs(
        [
            ("2026-04-01T09:00:00Z", "v1.0"),
            ("2026-04-08T09:00:00Z", "v1.1"),
            ("2026-04-15T09:00:00Z", "v1.2"),
        ]
    )
    assert _band_labels(runs) == ["<b>v1.0</b>", "<b>v1.1</b>", "<b>v1.2</b>"]


@pytest.mark.unit
def test_band_label_fit_threshold() -> None:
    # 13 characters need 13 * 7.2 + 12 = 105.6 px of a 480 px plot: 22 %.
    assert band_label_fits(0.23, "auto-d545d6f7")
    assert not band_label_fits(0.21, "auto-d545d6f7")


@pytest.mark.unit
def test_time_axes_show_clock_times_below_a_day() -> None:
    runs = _runs([("2026-09-06T08:25:33Z", "v1"), ("2026-09-06T13:36:20Z", "v1")])
    for fig in (pass_rate_chart(runs), metric_trend_chart(runs, "geval")):
        stops = [
            (list(s.dtickrange), s.value) for s in fig.layout.xaxis.tickformatstops
        ]
        assert stops == [(list(s["dtickrange"]), s["value"]) for s in TIME_TICK_FORMATS]
        assert stops[0][1] == "%H:%M<br>%-d %b"
        assert fig.layout.xaxis.tickformat is None
