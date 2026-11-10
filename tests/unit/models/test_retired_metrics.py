"""Retired Azure AI metric names fail at config load (042 T18, FR-031, AC-13).

The Azure AI evaluation stack is removed. New configuration that names one of
its metrics must fail during validation, before any backend or evaluator is
built, and the error must name the DeepEval replacement. Stored runs that
used those names still load (see tests/unit/dashboard/test_load_all.py).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from holodeck.config.context import STORED_SNAPSHOT
from holodeck.config.loader import ConfigLoader
from holodeck.lib.errors import ConfigError
from holodeck.models.evaluation import (
    RETIRED_METRIC_REPLACEMENTS,
    RETIRED_METRICS_REMOVED_IN,
    EvaluationConfig,
    EvaluationMetric,
)

EXPECTED_REPLACEMENT = {
    "groundedness": "metric_type: faithfulness",
    "relevance": "metric_type: answer_relevancy",
    "coherence": "type: geval",
    "fluency": "type: geval",
    "similarity": "type: geval",
    "safety": "type: geval",
}

AGENT_YAML = """\
name: retired-metric-agent
model:
  provider: openai
  name: gpt-4o
instructions:
  inline: You are a helpful assistant.
evaluations:
  metrics:
    - type: standard
      metric: {metric}
      threshold: 0.7
test_cases:
  - input: "Hello"
    ground_truth: "Hi"
"""


@pytest.mark.unit
def test_every_retired_name_has_a_replacement() -> None:
    assert set(RETIRED_METRIC_REPLACEMENTS) == set(EXPECTED_REPLACEMENT)


@pytest.mark.unit
@pytest.mark.parametrize("metric", sorted(EXPECTED_REPLACEMENT))
def test_retired_metric_names_the_replacement(metric: str) -> None:
    with pytest.raises(ConfigError) as exc_info:
        EvaluationMetric.model_validate({"metric": metric})

    message = str(exc_info.value)
    assert (
        f"Metric '{metric}' was removed in HoloDeck {RETIRED_METRICS_REMOVED_IN}"
        in message
    )
    assert EXPECTED_REPLACEMENT[metric] in message
    assert "(DeepEval)" in message


@pytest.mark.unit
def test_retired_metric_fails_in_the_config_loader(tmp_path: Path) -> None:
    agent_yaml = tmp_path / "agent.yaml"
    agent_yaml.write_text(AGENT_YAML.format(metric="groundedness"))

    with (
        patch("holodeck.lib.backends.selector.BackendSelector.select") as select,
        pytest.raises(ConfigError, match="Use 'type: rag' with 'metric_type: "),
    ):
        ConfigLoader().load_agent_yaml(str(agent_yaml))

    select.assert_not_called()


@pytest.mark.unit
def test_retired_metric_fails_before_the_executor_builds_anything(
    tmp_path: Path,
) -> None:
    from holodeck.lib.test_runner.executor import TestExecutor

    agent_yaml = tmp_path / "agent.yaml"
    agent_yaml.write_text(AGENT_YAML.format(metric="coherence"))

    with (
        patch("holodeck.lib.backends.selector.BackendSelector.select") as select,
        patch.object(TestExecutor, "_create_evaluators") as create_evaluators,
        pytest.raises(ConfigError, match="Metric 'coherence' was removed"),
    ):
        TestExecutor(agent_config_path=str(agent_yaml))

    select.assert_not_called()
    create_evaluators.assert_not_called()


@pytest.mark.unit
def test_stored_snapshot_keeps_a_retired_metric() -> None:
    config = EvaluationConfig.model_validate(
        {"metrics": [{"type": "standard", "metric": "fluency"}]},
        context={STORED_SNAPSHOT: True},
    )

    metric = config.metrics[0]
    assert isinstance(metric, EvaluationMetric)
    assert metric.metric == "fluency"


@pytest.mark.unit
def test_published_schema_lists_only_supported_metrics() -> None:
    schema = EvaluationMetric.model_json_schema()
    assert schema["properties"]["metric"]["enum"] == [
        "bleu",
        "rouge",
        "meteor",
        "equality",
        "numeric",
    ]


@pytest.mark.unit
def test_test_runner_imports_with_azure_ai_evaluation_blocked() -> None:
    """The test command and its evaluators load without the removed SDK."""
    code = (
        "import sys\n"
        "sys.modules['azure.ai.evaluation'] = None\n"
        "import holodeck.lib.test_runner.executor\n"
        "import holodeck.lib.evaluators\n"
        "from holodeck.cli.main import main\n"
        "main.get_command(None, 'test')\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True, timeout=180)


@pytest.mark.unit
def test_no_source_file_references_azure_ai_evaluation() -> None:
    src = Path(__file__).resolve().parents[3] / "src"
    offenders = [
        str(path.relative_to(src))
        for path in src.rglob("*")
        if path.is_file()
        and path.suffix in {".py", ".j2", ".yaml", ".toml", ".json"}
        and (
            "azure.ai.evaluation" in (text := path.read_text(errors="ignore"))
            or "azure_ai_evaluation" in text
            or "azure-ai-evaluation" in text
        )
    ]
    assert offenders == []
