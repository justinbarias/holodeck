"""Write optimizer run artifacts.

Produces ``<output_dir>/<run-id>/`` containing:

- ``best.yaml`` — the best candidate agent, ready to copy over the original.
- ``trials.jsonl`` — one ``TrialRecord`` per line (the full audit trail).
- ``report.md`` — baseline vs best, accepted edits, and a per-phase summary.

The original ``agent.yaml`` is never read or written here; candidates are
always fresh copies produced by the mutator.
"""

import json
from pathlib import Path

import yaml

from holodeck.optimizer.models import OptimizationResult, TrialRecord


def _agent_to_yaml(result: OptimizationResult) -> str:
    """Serialize the best candidate agent to YAML, dropping unset fields."""
    data = result.best_agent.model_dump(mode="json", exclude_none=True)
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)


def _trials_to_jsonl(trials: list[TrialRecord]) -> str:
    """Serialize trials to newline-delimited JSON (one row per trial)."""
    return "".join(json.dumps(t.model_dump()) + "\n" for t in trials)


def _build_report(result: OptimizationResult) -> str:
    """Build the Markdown summary report."""
    delta = result.best_score - result.baseline_score
    lines = [
        f"# Optimization Report: {result.agent_name}",
        "",
        f"- **Run ID:** `{result.run_id}`",
        f"- **Baseline score:** {result.baseline_score:.2f}",
        f"- **Best score:** {result.best_score:.2f} (Δ {delta:+.2f})",
        f"- **Cycles run:** {result.cycles_run}",
        f"- **Accepted improvements:** {result.accepted_count}",
        "",
        "## Accepted edits",
        "",
    ]
    accepted = [t for t in result.trials if t.accepted]
    if accepted:
        for trial in accepted:
            detail = (
                f"params {trial.params}"
                if trial.params is not None
                else f"{trial.textual_axis}: {trial.edit_summary or 'rewritten'}"
            )
            lines.append(
                f"- Trial {trial.trial_id} ({trial.phase}): {detail} "
                f"→ score {trial.score:.3f}"
            )
    else:
        lines.append("_No improvements were accepted._")

    lines += ["", "## Per-phase summary", ""]
    for phase in ("numeric", "textual"):
        phase_trials = [t for t in result.trials if t.phase == phase]
        accepts = sum(1 for t in phase_trials if t.accepted)
        lines.append(f"- **{phase}:** {len(phase_trials)} trials, {accepts} accepted")

    lines += ["", "## All trials", ""]
    for trial in result.trials:
        status = "accepted" if trial.accepted else "rejected"
        if trial.error:
            status = f"skipped ({trial.error})"
        lines.append(
            f"- Trial {trial.trial_id} [{trial.phase}, cycle {trial.cycle}]: "
            f"score {trial.score:.3f} vs {trial.baseline_score:.3f} — {status}"
        )

    return "\n".join(lines) + "\n"


def write_outputs(result: OptimizationResult, output_dir: Path) -> Path:
    """Write the run's three artifacts under ``output_dir/<run-id>/``.

    Args:
        result: The completed optimization result.
        output_dir: Base directory for optimizer runs.

    Returns:
        The run directory the artifacts were written to.
    """
    run_dir = output_dir / result.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "best.yaml").write_text(_agent_to_yaml(result))
    (run_dir / "trials.jsonl").write_text(_trials_to_jsonl(result.trials))
    (run_dir / "report.md").write_text(_build_report(result))

    return run_dir
