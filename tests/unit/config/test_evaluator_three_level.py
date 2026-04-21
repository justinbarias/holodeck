"""Three-level configuration tests for US4 metrics (T045–T046).

Verifies that ``CodeMetric`` and the new deterministic metrics (``equality`` /
``numeric``) are usable at each of the three configuration rungs — per-turn,
per-test-case, and agent-global — per FR-023. Each rung attaching the same
metric should produce identical ``MetricResult`` kinds / pass outcomes.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock

import pytest

from holodeck.config.loader import ConfigLoader
from holodeck.lib.backends.base import ExecutionResult
from holodeck.lib.file_processor import FileProcessor
from holodeck.lib.test_runner.executor import TestExecutor
from holodeck.models.agent import Agent, Instructions
from holodeck.models.config import ExecutionConfig
from holodeck.models.evaluation import (
    CodeMetric,
    EvaluationConfig,
    EvaluationMetric,
)
from holodeck.models.llm import LLMProvider, ProviderEnum
from holodeck.models.test_case import TestCaseModel, Turn
from holodeck.models.token_usage import TokenUsage


def _make_backend(response: str) -> Mock:
    async def send(_msg: str) -> ExecutionResult:
        return ExecutionResult(
            response=response,
            tool_calls=[],
            tool_results=[],
            token_usage=TokenUsage(
                prompt_tokens=1, completion_tokens=1, total_tokens=2
            ),
        )

    session = Mock()
    session.send = AsyncMock(side_effect=send)
    session.close = AsyncMock()

    backend = Mock()
    backend.initialize = AsyncMock()
    backend.teardown = AsyncMock()
    backend.invoke_once = AsyncMock(
        return_value=ExecutionResult(
            response=response,
            tool_calls=[],
            tool_results=[],
            token_usage=TokenUsage(
                prompt_tokens=1, completion_tokens=1, total_tokens=2
            ),
        )
    )
    backend.create_session = AsyncMock(return_value=session)
    backend.__class__.__name__ = "_StubBackend"
    return backend


async def _run(agent: Agent, response: str) -> list:
    loader = Mock(spec=ConfigLoader)
    loader.load_agent_yaml.return_value = agent
    loader.resolve_execution_config.return_value = ExecutionConfig(llm_timeout=60)

    executor = TestExecutor(
        agent_config_path="x.yaml",
        config_loader=loader,
        file_processor=Mock(spec=FileProcessor),
        backend=_make_backend(response),
        agent_config=agent,
        resolved_execution_config=ExecutionConfig(llm_timeout=60),
    )
    report = await executor.execute_tests()
    return report.results[0].metric_results


def _base_agent(
    *,
    agent_eval: EvaluationConfig | None = None,
    tc_eval: list | None = None,
    turn_eval: list | None = None,
    multi_turn: bool,
) -> Agent:
    if multi_turn:
        tc = TestCaseModel(
            name="tc",
            turns=[
                Turn(input="q", ground_truth="25587", evaluations=turn_eval),
            ],
            evaluations=tc_eval,
        )
    else:
        tc = TestCaseModel(
            name="tc",
            input="q",
            ground_truth="25587",
            evaluations=tc_eval,
        )
    return Agent(
        name="three-level-agent",
        description="three-level config test",
        model=LLMProvider(
            provider=ProviderEnum.OPENAI, name="gpt-4", api_key="test-key"
        ),
        instructions=Instructions(inline="x"),
        test_cases=[tc],
        evaluations=agent_eval,
        execution=None,
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_code_metric_usable_at_all_three_levels() -> None:
    code_metric = CodeMetric(grader="my_benchmarks:numeric_equal")

    # 1. Agent-global level.
    agent = _base_agent(
        agent_eval=EvaluationConfig(metrics=[code_metric]),
        multi_turn=False,
    )
    results = await _run(agent, response="25587")
    assert len(results) == 1
    assert results[0].kind == "code"
    assert results[0].passed is True

    # 2. Test-case level.
    agent = _base_agent(tc_eval=[code_metric], multi_turn=False)
    results = await _run(agent, response="25587")
    assert len(results) == 1
    assert results[0].kind == "code"
    assert results[0].passed is True

    # 3. Per-turn level (only meaningful in multi-turn mode).
    agent = _base_agent(turn_eval=[code_metric], multi_turn=True)
    rolled = await _run(agent, response="25587")
    # Rollup emits one row per metric_name.
    code_row = [m for m in rolled if m.metric_name == "numeric_equal"]
    assert len(code_row) == 1
    assert code_row[0].passed is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_equality_numeric_usable_at_all_three_levels() -> None:
    eq = EvaluationMetric(metric="equality")
    num = EvaluationMetric(metric="numeric", absolute_tolerance=0.5)

    # 1. Agent-global.
    agent = _base_agent(
        agent_eval=EvaluationConfig(metrics=[eq, num]),
        multi_turn=False,
    )
    results = await _run(agent, response="25587")
    names = {m.metric_name: m for m in results}
    assert names["equality"].kind == "standard"
    assert names["equality"].passed is True
    assert names["numeric"].kind == "standard"
    assert names["numeric"].passed is True

    # 2. Test-case level.
    agent = _base_agent(tc_eval=[eq, num], multi_turn=False)
    results = await _run(agent, response="25587")
    names = {m.metric_name: m for m in results}
    assert names["equality"].passed is True
    assert names["numeric"].passed is True

    # 3. Per-turn level.
    agent = _base_agent(turn_eval=[eq, num], multi_turn=True)
    rolled = await _run(agent, response="25587")
    names = {m.metric_name: m for m in rolled}
    assert names["equality"].passed is True
    assert names["numeric"].passed is True
