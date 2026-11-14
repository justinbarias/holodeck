"""Verdict and regression derivation for the Summary view.

Pure functions over saved runs. The Summary opens with a sentence about the
latest run in view against the run before it, followed by a numbered list of
regressions. The pass-rate chart draws a ring for each regression with the
same number, so both come from :func:`find_regressions`.

Definitions:

* A run is a *regression* when its pass rate drops more than four points
  below the previous run in view (``data_loader.detect_regressions``).
* A case is *failing* when its saved ``TestResult.passed`` is false.
* A failing case's *streak* counts the consecutive runs in view, ending at
  the latest run, in which that case was present and failed.
"""

from __future__ import annotations

from dataclasses import dataclass

from holodeck.dashboard.data_loader import detect_regressions
from holodeck.dashboard.loaded_run import LoadedRun
from holodeck.dashboard.revisions import RunNumbers, chronological, run_label
from holodeck.models.test_result import MetricResult, TestResult


def pass_rate(run: LoadedRun) -> float:
    """Pass rate on the 0..1 scale from ``passed / total_tests``."""
    s = run.report.summary
    return (s.passed / s.total_tests) if s.total_tests > 0 else 0.0


def _chronological(runs: list[LoadedRun]) -> list[LoadedRun]:
    return chronological(runs)


def _case_map(run: LoadedRun) -> dict[str, TestResult]:
    return {c.test_name: c for c in run.report.results if c.test_name}


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


# --------------------------------------------------------------------------- #
# Regressions                                                                 #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Regression:
    """One numbered regression; the number matches its chart ring."""

    number: int
    run_id: str
    label: str
    version: str
    previous_label: str
    previous_rate: float
    rate: float
    newly_failed: tuple[str, ...]

    def rate_text(self) -> str:
        """``58.3% → 50.0%``."""
        return f"{self.previous_rate * 100:.1f}% → {self.rate * 100:.1f}%"

    def case_text(self) -> str | None:
        """First case that passed in the previous run and fails now, plus
        ``+n more``; ``None`` when no case changed from pass to fail."""
        if not self.newly_failed:
            return None
        text = self.newly_failed[0]
        if len(self.newly_failed) > 1:
            text += f" +{len(self.newly_failed) - 1} more"
        return text

    def change_text(self) -> str:
        """``58.3% → 50.0% · refund_outside_window failed`` (case if any)."""
        case = self.case_text()
        return f"{self.rate_text()} · {case} failed" if case else self.rate_text()


def _newly_failed(previous: LoadedRun, run: LoadedRun) -> tuple[str, ...]:
    before = _case_map(previous)
    out: list[str] = []
    for case in run.report.results:
        name = case.test_name or ""
        prior = before.get(name)
        if prior is not None and prior.passed and not case.passed:
            out.append(name)
    return tuple(out)


def find_regressions(runs: list[LoadedRun], numbers: RunNumbers) -> list[Regression]:
    """Numbered regressions across ``runs`` in chronological order."""
    ordered = _chronological(runs)
    out: list[Regression] = []
    for k, idx in enumerate(detect_regressions(ordered), start=1):
        run = ordered[idx]
        previous = ordered[idx - 1]
        out.append(
            Regression(
                number=k,
                run_id=run.run_id,
                label=run_label(run, numbers),
                version=run.metadata.prompt_version.version,
                previous_label=run_label(previous, numbers),
                previous_rate=pass_rate(previous),
                rate=pass_rate(run),
                newly_failed=_newly_failed(previous, run),
            )
        )
    return out


# --------------------------------------------------------------------------- #
# Verdict                                                                     #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class FailingCase:
    """A case that fails in the latest run, with its failure streak."""

    name: str
    streak: int
    streak_start_label: str
    passed_in_previous: bool
    metric_name: str | None
    score: float | None
    threshold: float | None

    def sentence_tail(self, latest_label: str, previous_label: str | None) -> str:
        """Text that follows the case name in the verdict."""
        if self.streak > 1:
            text = f" has failed in every run since {self.streak_start_label}"
        elif self.passed_in_previous and previous_label is not None:
            text = f" fails in {latest_label} after passing in {previous_label}"
        else:
            text = f" fails in {latest_label}"
        if self.metric_name is not None and self.score is not None:
            if self.threshold is not None:
                text += (
                    f" ({self.metric_name} {self.score:.2f} "
                    f"against {self.threshold:.2f})"
                )
            else:
                text += f" ({self.metric_name} {self.score:.2f})"
        return text + "."


def _weakest_failed_metric(
    case: TestResult,
) -> tuple[str | None, float | None, float | None]:
    # Only name a metric that visibly explains the failure: saved as failed
    # and scored below its threshold (or saved without a threshold).
    failed = [
        m
        for m in case.metric_results
        if m.passed is False and (m.threshold is None or m.score < m.threshold)
    ]
    if not failed:
        return None, None, None

    def margin(m: MetricResult) -> float:
        return float(m.score) - float(m.threshold or 0.0)

    worst = min(failed, key=margin)
    threshold = float(worst.threshold) if worst.threshold is not None else None
    return worst.metric_name, float(worst.score), threshold


def _failing_cases(ordered: list[LoadedRun], numbers: RunNumbers) -> list[FailingCase]:
    latest = ordered[-1]
    previous = ordered[-2] if len(ordered) > 1 else None
    out: list[FailingCase] = []
    for case in latest.report.results:
        if case.passed or not case.test_name:
            continue
        name = case.test_name
        streak = 0
        start = latest
        for run in reversed(ordered):
            match = _case_map(run).get(name)
            if match is None or match.passed:
                break
            streak += 1
            start = run
        prior = _case_map(previous).get(name) if previous is not None else None
        metric, score, threshold = _weakest_failed_metric(case)
        out.append(
            FailingCase(
                name=name,
                streak=streak,
                streak_start_label=run_label(start, numbers),
                passed_in_previous=bool(prior is not None and prior.passed),
                metric_name=metric,
                score=score,
                threshold=threshold,
            )
        )
    # Longest-running failures first: they are the persistent problems.
    out.sort(key=lambda f: (-f.streak, f.name))
    return out


@dataclass(frozen=True)
class Verdict:
    """The latest run in view against the run before it."""

    run_id: str
    label: str
    version: str
    passed: int
    total: int
    previous_run_id: str | None
    previous_label: str | None
    previous_passed: int | None
    previous_total: int | None
    regression_number: int | None
    duration_delta_ms: int | None
    failing: tuple[FailingCase, ...]

    @property
    def is_regression(self) -> bool:
        return self.regression_number is not None

    def headline(self) -> str:
        counts = f"passes {self.passed} of {_plural(self.total, 'case')}"
        if self.previous_label is None or self.previous_passed is None:
            return f"{self.label} {counts}."
        diff = self.passed - self.previous_passed
        if diff == 0:
            compare = f"the same as {self.previous_label}"
        elif diff > 0:
            compare = f"{diff} more than {self.previous_label}"
        else:
            compare = f"{-diff} fewer than {self.previous_label}"
        if self.is_regression:
            return f"{self.label} is a regression: it {counts}, {compare}."
        return f"{self.label} {counts}, {compare}."

    def duration_sentence(self) -> str | None:
        if self.duration_delta_ms is None or self.previous_label is None:
            return None
        delta_s = self.duration_delta_ms / 1000
        if abs(delta_s) < 0.05:
            return f"Run time is unchanged from {self.previous_label}."
        direction = "longer" if delta_s > 0 else "shorter"
        return (
            f"Run time is {abs(delta_s):.1f} s {direction} than {self.previous_label}."
        )


def build_verdict(
    runs: list[LoadedRun],
    numbers: RunNumbers,
    regressions: list[Regression] | None = None,
) -> Verdict | None:
    """Verdict for the latest run in ``runs``; ``None`` when there are none."""
    if not runs:
        return None
    ordered = _chronological(runs)
    latest = ordered[-1]
    previous = ordered[-2] if len(ordered) > 1 else None
    if regressions is None:
        regressions = find_regressions(ordered, numbers)
    reg_number = next(
        (r.number for r in regressions if r.run_id == latest.run_id),
        None,
    )
    duration_delta: int | None = None
    if previous is not None:
        duration_delta = (
            latest.report.summary.total_duration_ms
            - previous.report.summary.total_duration_ms
        )
    return Verdict(
        run_id=latest.run_id,
        label=run_label(latest, numbers),
        version=latest.metadata.prompt_version.version,
        passed=latest.report.summary.passed,
        total=latest.report.summary.total_tests,
        previous_run_id=previous.run_id if previous else None,
        previous_label=(run_label(previous, numbers) if previous else None),
        previous_passed=previous.report.summary.passed if previous else None,
        previous_total=previous.report.summary.total_tests if previous else None,
        regression_number=reg_number,
        duration_delta_ms=duration_delta,
        failing=tuple(_failing_cases(ordered, numbers)),
    )
