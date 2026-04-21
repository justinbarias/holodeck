"""Executor-level test for deterministic evaluator wiring (US4 T012).

Asserts that when the agent config declares ``metric: equality`` or
``metric: numeric``, the executor builds a ``MetricResult`` with
``kind == "standard"`` (data-model.md §7a) and a score in ``[0.0, 1.0]``.

This is an executor-level integration — evaluators return dicts; the
executor builds the ``MetricResult`` envelope at ``executor.py:~1295``.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock

import pytest

from holodeck.config.loader import ConfigLoader
from holodeck.lib.backends.base import ExecutionResult
from holodeck.lib.file_processor import FileProcessor
from holodeck.lib.test_runner.executor import TestExecutor as _TestExecutor
from holodeck.models.agent import Agent, Instructions
from holodeck.models.config import ExecutionConfig
from holodeck.models.evaluation import EvaluationConfig, EvaluationMetric
from holodeck.models.llm import LLMProvider, ProviderEnum
from holodeck.models.test_case import TestCaseModel as _TestCaseModel
from holodeck.models.token_usage import TokenUsage


def _agent_for(metric: str) -> Agent:
    return Agent(
        name="determ-agent",
        description="deterministic evaluator wiring",
        model=LLMProvider(
            provider=ProviderEnum.OPENAI, name="gpt-4", api_key="test-key"
        ),
        instructions=Instructions(inline="hi"),
        test_cases=[
            _TestCaseModel(
                name="single",
                input="q",
                ground_truth="hello",
            )
        ],
        evaluations=EvaluationConfig(
            metrics=[EvaluationMetric(metric=metric)],
        ),
        execution=None,
    )


async def _run(metric: str, agent_response: str) -> list:
    agent = _agent_for(metric)
    loader = Mock(spec=ConfigLoader)
    loader.load_agent_yaml.return_value = agent
    loader.resolve_execution_config.return_value = ExecutionConfig(llm_timeout=60)

    exec_result = ExecutionResult(
        response=agent_response,
        tool_calls=[],
        tool_results=[],
        token_usage=TokenUsage.zero(),
    )
    backend = Mock()
    backend.initialize = AsyncMock()
    backend.teardown = AsyncMock()
    backend.invoke_once = AsyncMock(return_value=exec_result)
    backend.create_session = AsyncMock()
    backend.__class__.__name__ = "_StubBackend"

    executor = _TestExecutor(
        agent_config_path="x.yaml",
        config_loader=loader,
        file_processor=Mock(spec=FileProcessor),
        backend=backend,
        agent_config=agent,
        resolved_execution_config=ExecutionConfig(llm_timeout=60),
    )
    report = await executor.execute_tests()
    return report.results[0].metric_results


@pytest.mark.unit
@pytest.mark.asyncio
async def test_equality_produces_metric_result_with_kind_standard() -> None:
    results = await _run("equality", agent_response="hello")
    assert len(results) == 1
    mr = results[0]
    assert mr.metric_name == "equality"
    assert mr.kind == "standard"
    assert 0.0 <= mr.score <= 1.0
    assert mr.score == 1.0
    assert mr.passed is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_numeric_produces_metric_result_with_kind_standard() -> None:
    # ground_truth is "hello" — numeric parse will fail, so score=0 / passed=False.
    # Using a numeric-parseable gt instead:
    agent = Agent(
        name="numeric-agent",
        description="numeric wiring",
        model=LLMProvider(
            provider=ProviderEnum.OPENAI, name="gpt-4", api_key="test-key"
        ),
        instructions=Instructions(inline="hi"),
        test_cases=[
            _TestCaseModel(
                name="single",
                input="q",
                ground_truth="1",
            )
        ],
        evaluations=EvaluationConfig(
            metrics=[EvaluationMetric(metric="numeric")],
        ),
    )
    loader = Mock(spec=ConfigLoader)
    loader.load_agent_yaml.return_value = agent
    loader.resolve_execution_config.return_value = ExecutionConfig(llm_timeout=60)

    exec_result = ExecutionResult(
        response="1",
        tool_calls=[],
        tool_results=[],
        token_usage=TokenUsage.zero(),
    )
    backend = Mock()
    backend.initialize = AsyncMock()
    backend.teardown = AsyncMock()
    backend.invoke_once = AsyncMock(return_value=exec_result)
    backend.create_session = AsyncMock()
    backend.__class__.__name__ = "_StubBackend"

    executor = _TestExecutor(
        agent_config_path="x.yaml",
        config_loader=loader,
        file_processor=Mock(spec=FileProcessor),
        backend=backend,
        agent_config=agent,
        resolved_execution_config=ExecutionConfig(llm_timeout=60),
    )
    report = await executor.execute_tests()
    results = report.results[0].metric_results
    assert len(results) == 1
    mr = results[0]
    assert mr.metric_name == "numeric"
    assert mr.kind == "standard"
    assert 0.0 <= mr.score <= 1.0
    assert mr.passed is True
