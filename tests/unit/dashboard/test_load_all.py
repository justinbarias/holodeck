"""The dashboard opens stored runs away from their agent's files.

Regression guard: a run's agent snapshot names its code graders by import
path and its skills by directory. Both live next to the agent, not next to
the results, so the dashboard skipped every run that used one.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from holodeck.dashboard.data_loader import load_all
from holodeck.dashboard.explorer_data import build_case_detail
from holodeck.dashboard.seed_data import build_seed_runs
from holodeck.lib.errors import ConfigError
from holodeck.models.eval_run import EvalRun
from holodeck.models.evaluation import CodeMetric, EvaluationMetric
from holodeck.models.tool import SkillTool

MISSING_GRADER = "grader_module_that_is_not_installed:check"
MISSING_SKILL_DIR = "skills/directory-that-does-not-exist"


def _run_with_missing_grader() -> dict[str, object]:
    run = build_seed_runs()[0].model_dump(mode="json")
    run["metadata"]["agent_config"]["evaluations"] = {
        "metrics": [{"type": "code", "grader": MISSING_GRADER}]
    }
    return run


@pytest.mark.unit
def test_load_all_keeps_runs_whose_grader_is_not_importable(tmp_path: Path) -> None:
    (tmp_path / "run.json").write_text(json.dumps(_run_with_missing_grader()))

    runs = load_all(tmp_path)

    assert len(runs) == 1
    evaluations = runs[0].metadata.agent_config.evaluations
    assert evaluations is not None
    metric = evaluations.metrics[0]
    assert isinstance(metric, CodeMetric)
    assert metric.grader == MISSING_GRADER
    assert metric.resolved_callable is None


@pytest.mark.unit
def test_agent_config_still_rejects_a_grader_that_is_not_importable() -> None:
    with pytest.raises(ConfigError, match="cannot import grader module"):
        EvalRun.model_validate(_run_with_missing_grader())


def _run_with_missing_skill_dir() -> dict[str, object]:
    run = build_seed_runs()[0].model_dump(mode="json")
    run["metadata"]["agent_config"]["tools"] = [
        {
            "type": "skill",
            "name": "research",
            "description": "Research a topic.",
            "path": MISSING_SKILL_DIR,
        }
    ]
    return run


@pytest.mark.unit
def test_load_all_keeps_runs_whose_skill_dir_is_missing(tmp_path: Path) -> None:
    (tmp_path / "run.json").write_text(json.dumps(_run_with_missing_skill_dir()))

    runs = load_all(tmp_path)

    assert len(runs) == 1
    tools = runs[0].metadata.agent_config.tools
    assert tools is not None
    skill = tools[0]
    assert isinstance(skill, SkillTool)
    assert skill.path == MISSING_SKILL_DIR


@pytest.mark.unit
def test_agent_config_still_rejects_a_missing_skill_dir() -> None:
    with pytest.raises(ValueError, match="skill path"):
        EvalRun.model_validate(_run_with_missing_skill_dir())


def _run_with_retired_metric() -> dict[str, object]:
    """A run written before the Azure AI metrics were retired (042 T18)."""
    run = build_seed_runs()[0].model_dump(mode="json")
    run["metadata"]["agent_config"]["evaluations"] = {
        "metrics": [{"type": "standard", "metric": "groundedness", "threshold": 0.8}]
    }
    run["report"]["results"][0]["metric_results"].append(
        {
            "metric_name": "groundedness",
            "kind": "standard",
            "score": 0.8,
            "threshold": 0.8,
            "passed": True,
        }
    )
    return run


@pytest.mark.unit
def test_load_all_keeps_runs_that_used_a_retired_metric(tmp_path: Path) -> None:
    (tmp_path / "run.json").write_text(json.dumps(_run_with_retired_metric()))

    runs = load_all(tmp_path)

    assert len(runs) == 1
    evaluations = runs[0].metadata.agent_config.evaluations
    assert evaluations is not None
    metric = evaluations.metrics[0]
    assert isinstance(metric, EvaluationMetric)
    assert metric.metric == "groundedness"

    # The Explorer detail panel shows the stored name and score.
    case = runs[0].report.results[0]
    detail = build_case_detail(runs[0], case.test_name or "", {})
    assert detail is not None
    rows = [(r.name, r.score) for r in detail.evaluations["standard"]]
    assert ("groundedness", 0.8) in rows
    raw_metrics = detail.agent_snapshot.raw_config["evaluations"]["metrics"]
    assert raw_metrics[0]["metric"] == "groundedness"


@pytest.mark.unit
def test_agent_config_rejects_a_retired_metric_outside_a_snapshot() -> None:
    with pytest.raises(ConfigError, match="Metric 'groundedness' was removed"):
        EvalRun.model_validate(_run_with_retired_metric())
