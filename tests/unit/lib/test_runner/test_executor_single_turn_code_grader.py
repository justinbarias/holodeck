"""Single-turn ``type: code`` graders see the turn's tool invocations.

Regression: the single-turn path ran evaluations before pairing tool calls,
so ``GraderContext.tool_invocations`` was always empty there.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock, patch

import pytest

from holodeck.config.loader import ConfigLoader
from holodeck.lib.backends.base import ExecutionResult
from holodeck.lib.file_processor import FileProcessor
from holodeck.lib.test_runner.executor import TestExecutor
from holodeck.models.agent import Agent, Instructions
from holodeck.models.config import ExecutionConfig
from holodeck.models.evaluation import CodeMetric, EvaluationConfig
from holodeck.models.llm import LLMProvider, ProviderEnum
from holodeck.models.test_case import TestCaseModel
from holodeck.models.test_result import TestResult
from holodeck.models.token_usage import TokenUsage


async def _run_single_turn_lookup() -> TestResult:
    """Run one single-turn case whose backend calls ``lookup`` once."""
    test_case = TestCaseModel(name="single", input="look it up")
    agent = Agent(
        name="single-turn-grader-agent",
        model=LLMProvider(
            provider=ProviderEnum.OPENAI, name="gpt-4", api_key="test-key"
        ),
        instructions=Instructions(inline="be concise"),
        test_cases=[test_case],
        evaluations=EvaluationConfig(
            metrics=[CodeMetric(grader="my_benchmarks:lookup_returned_marker")]
        ),
    )
    loader = Mock(spec=ConfigLoader)
    loader.load_agent_yaml.return_value = agent
    loader.resolve_execution_config.return_value = ExecutionConfig(llm_timeout=60)
    backend = Mock()
    backend.invoke_once = AsyncMock(
        return_value=ExecutionResult(
            response="done",
            tool_calls=[{"name": "lookup", "args": {"q": "x"}, "id": "c1"}],
            tool_results=[{"name": "lookup", "result": "MARKER-42"}],
            token_usage=TokenUsage.zero(),
        )
    )
    backend.teardown = AsyncMock()
    executor = TestExecutor(
        agent_config_path="agent.yaml",
        config_loader=loader,
        file_processor=Mock(spec=FileProcessor),
        backend=backend,
        agent_config=agent,
        resolved_execution_config=ExecutionConfig(llm_timeout=60),
    )

    report = await executor.execute_tests()
    return report.results[0]


@pytest.mark.asyncio
async def test_single_turn_code_grader_receives_tool_invocations() -> None:
    result = await _run_single_turn_lookup()

    assert [inv.name for inv in result.tool_invocations] == ["lookup"]
    [grader] = result.metric_results
    assert grader.metric_name == "lookup_returned_marker"
    assert grader.passed is True, grader.reasoning
    assert result.passed is True


@pytest.mark.asyncio
async def test_unpairable_tool_calls_fall_back_to_no_invocations() -> None:
    """A TypeError from pairing yields no invocations instead of failing the run."""
    with patch(
        "holodeck.lib.test_runner.executor.pair_tool_calls",
        side_effect=TypeError("not a list"),
    ):
        result = await _run_single_turn_lookup()

    assert result.tool_invocations == []
    [grader] = result.metric_results
    assert grader.passed is False
    assert "[]" in (grader.reasoning or "")
