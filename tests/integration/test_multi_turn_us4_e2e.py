"""E2E integration for US4 — numeric built-in + code grader (T047).

A 1-turn multi-turn case with:

- `numeric` built-in (with `accept_thousands_separators=True`) — compares the
  scripted agent response ``"25,587"`` against ground truth ``"25587"``.
- `code` grader ``my_benchmarks:numeric_equal`` — exercises the fixture
  grader path at ``tests/fixtures/graders/my_benchmarks.py``.

Asserts both MetricResults land in ``TurnResult.metric_results`` with the
correct ``kind`` values and that the rollup marks the case passed.
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


@pytest.mark.integration
@pytest.mark.asyncio
async def test_convfinqa_numeric_plus_code_grader() -> None:
    test_case = TestCaseModel(
        name="convfinqa-us4",
        turns=[
            Turn(
                input="what is the difference?",
                ground_truth="25587",
            ),
        ],
    )
    evaluations = EvaluationConfig(
        metrics=[
            EvaluationMetric(
                metric="numeric",
                absolute_tolerance=0.01,
                accept_thousands_separators=True,
            ),
            CodeMetric(grader="my_benchmarks:numeric_equal"),
        ],
    )
    agent = Agent(
        name="convfinqa-us4-agent",
        description="US4 numeric + code-grader e2e",
        model=LLMProvider(
            provider=ProviderEnum.OPENAI, name="gpt-4", api_key="test-key"
        ),
        instructions=Instructions(inline="answer numerically"),
        test_cases=[test_case],
        evaluations=evaluations,
        execution=None,
    )

    async def send(_msg: str) -> ExecutionResult:
        return ExecutionResult(
            response="25,587",
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
    backend.invoke_once = AsyncMock()
    backend.create_session = AsyncMock(return_value=session)
    backend.__class__.__name__ = "_StubBackend"

    loader = Mock(spec=ConfigLoader)
    loader.load_agent_yaml.return_value = agent
    loader.resolve_execution_config.return_value = ExecutionConfig(llm_timeout=60)

    executor = TestExecutor(
        agent_config_path="tests/fixtures/multi_turn/convfinqa_sample.yaml",
        config_loader=loader,
        file_processor=Mock(spec=FileProcessor),
        backend=backend,
        agent_config=agent,
        resolved_execution_config=ExecutionConfig(llm_timeout=60),
    )

    report = await executor.execute_tests()
    result = report.results[0]

    assert result.turns is not None and len(result.turns) == 1
    turn = result.turns[0]

    names = {m.metric_name for m in turn.metric_results}
    assert "numeric" in names
    assert "numeric_equal" in names

    numeric_mr = next(m for m in turn.metric_results if m.metric_name == "numeric")
    assert numeric_mr.kind == "standard"
    assert numeric_mr.passed is True

    grader_mr = next(m for m in turn.metric_results if m.metric_name == "numeric_equal")
    assert grader_mr.kind == "code"
    assert grader_mr.passed is True

    # Rollup still passes (all turn metrics passed).
    assert turn.passed is True
    assert result.passed is True


@pytest.mark.integration
@pytest.mark.asyncio
async def test_grader_exception_captured_only_that_turn() -> None:
    """T037 — grader raises, only that turn's metric fails, other turns run."""
    test_case = TestCaseModel(
        name="two-turn-exception",
        turns=[
            Turn(input="turn 1", ground_truth="25587"),
            Turn(input="turn 2", ground_truth="25587"),
        ],
    )
    evaluations = EvaluationConfig(
        metrics=[CodeMetric(grader="my_benchmarks:raises_value_error")],
    )
    agent = Agent(
        name="grader-exc-agent",
        description="grader exception test",
        model=LLMProvider(
            provider=ProviderEnum.OPENAI, name="gpt-4", api_key="test-key"
        ),
        instructions=Instructions(inline="x"),
        test_cases=[test_case],
        evaluations=evaluations,
        execution=None,
    )

    async def send(_msg: str) -> ExecutionResult:
        return ExecutionResult(
            response="answer",
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
    backend.invoke_once = AsyncMock()
    backend.create_session = AsyncMock(return_value=session)
    backend.__class__.__name__ = "_StubBackend"

    loader = Mock(spec=ConfigLoader)
    loader.load_agent_yaml.return_value = agent
    loader.resolve_execution_config.return_value = ExecutionConfig(llm_timeout=60)

    executor = TestExecutor(
        agent_config_path="x.yaml",
        config_loader=loader,
        file_processor=Mock(spec=FileProcessor),
        backend=backend,
        agent_config=agent,
        resolved_execution_config=ExecutionConfig(llm_timeout=60),
    )
    report = await executor.execute_tests()
    result = report.results[0]
    assert result.turns is not None and len(result.turns) == 2
    # Both turns ran (exception did not halt the case).
    for turn in result.turns:
        assert turn.skipped is False
        grader = next(
            (m for m in turn.metric_results if m.metric_name == "raises_value_error"),
            None,
        )
        assert grader is not None
        assert grader.passed is False
        assert grader.error is not None
        assert "ValueError" in grader.error


@pytest.mark.integration
@pytest.mark.asyncio
async def test_fail_on_error_halts_test_case() -> None:
    """T038 — ``fail_on_error=True`` stops subsequent turns for this case only."""
    halting_case = TestCaseModel(
        name="halting",
        turns=[
            Turn(input="turn 1", ground_truth="x"),
            Turn(input="turn 2", ground_truth="x"),
            Turn(input="turn 3", ground_truth="x"),
        ],
    )
    other_case = TestCaseModel(
        name="other",
        turns=[Turn(input="only turn", ground_truth="x")],
    )
    evaluations_halting = EvaluationConfig(
        metrics=[
            CodeMetric(
                grader="my_benchmarks:raises_value_error",
                fail_on_error=True,
            )
        ],
    )
    agent = Agent(
        name="fail-on-error-agent",
        description="fail_on_error test",
        model=LLMProvider(
            provider=ProviderEnum.OPENAI, name="gpt-4", api_key="test-key"
        ),
        instructions=Instructions(inline="x"),
        test_cases=[halting_case, other_case],
        evaluations=evaluations_halting,
        execution=None,
    )

    call_count = {"halting": 0, "other": 0}

    def make_session(name: str) -> Mock:
        async def send(_msg: str) -> ExecutionResult:
            call_count[name] += 1
            return ExecutionResult(
                response="a",
                tool_calls=[],
                tool_results=[],
                token_usage=TokenUsage(
                    prompt_tokens=1, completion_tokens=1, total_tokens=2
                ),
            )

        s = Mock()
        s.send = AsyncMock(side_effect=send)
        s.close = AsyncMock()
        return s

    sessions = [make_session("halting"), make_session("other")]
    backend = Mock()
    backend.initialize = AsyncMock()
    backend.teardown = AsyncMock()
    backend.invoke_once = AsyncMock()
    backend.create_session = AsyncMock(side_effect=sessions)
    backend.__class__.__name__ = "_StubBackend"

    loader = Mock(spec=ConfigLoader)
    loader.load_agent_yaml.return_value = agent
    loader.resolve_execution_config.return_value = ExecutionConfig(llm_timeout=60)

    executor = TestExecutor(
        agent_config_path="x.yaml",
        config_loader=loader,
        file_processor=Mock(spec=FileProcessor),
        backend=backend,
        agent_config=agent,
        resolved_execution_config=ExecutionConfig(llm_timeout=60),
    )
    report = await executor.execute_tests()

    halting = next(r for r in report.results if r.test_name == "halting")
    other = next(r for r in report.results if r.test_name == "other")

    assert halting.turns is not None and len(halting.turns) == 3
    # Turn 0 ran and failed; turns 1+2 were skipped after TestCaseFatal.
    assert halting.turns[0].skipped is False
    assert halting.turns[1].skipped is True
    assert halting.turns[2].skipped is True

    # The OTHER test case still ran unaffected.
    assert other.turns is not None and len(other.turns) == 1
    assert other.turns[0].skipped is False
    # Only turn 0 of halting + turn 0 of other should have been sent.
    assert call_count["halting"] == 1
    assert call_count["other"] == 1


@pytest.mark.integration
@pytest.mark.asyncio
async def test_grader_details_preserved_on_turn_result() -> None:
    """T040 — grader details surfaced via ``TurnResult.grader_details``."""
    test_case = TestCaseModel(
        name="details-case",
        turns=[Turn(input="q", ground_truth="anything")],
    )
    evaluations = EvaluationConfig(
        metrics=[CodeMetric(grader="my_benchmarks:returns_grader_result")],
    )
    agent = Agent(
        name="details-agent",
        description="grader details e2e",
        model=LLMProvider(
            provider=ProviderEnum.OPENAI, name="gpt-4", api_key="test-key"
        ),
        instructions=Instructions(inline="x"),
        test_cases=[test_case],
        evaluations=evaluations,
        execution=None,
    )

    async def send(_msg: str) -> ExecutionResult:
        return ExecutionResult(
            response="resp",
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
    backend.invoke_once = AsyncMock()
    backend.create_session = AsyncMock(return_value=session)
    backend.__class__.__name__ = "_StubBackend"

    loader = Mock(spec=ConfigLoader)
    loader.load_agent_yaml.return_value = agent
    loader.resolve_execution_config.return_value = ExecutionConfig(llm_timeout=60)

    executor = TestExecutor(
        agent_config_path="x.yaml",
        config_loader=loader,
        file_processor=Mock(spec=FileProcessor),
        backend=backend,
        agent_config=agent,
        resolved_execution_config=ExecutionConfig(llm_timeout=60),
    )
    report = await executor.execute_tests()
    turn = report.results[0].turns[0]  # type: ignore[index]
    assert turn.grader_details is not None
    assert turn.grader_details["returns_grader_result"] == {"foo": "bar"}


@pytest.mark.integration
@pytest.mark.asyncio
async def test_grader_details_none_when_no_code_graders() -> None:
    """T044 regression — ``grader_details`` stays None without code graders."""
    test_case = TestCaseModel(
        name="no-grader",
        turns=[Turn(input="q", ground_truth="hello")],
    )
    evaluations = EvaluationConfig(
        metrics=[EvaluationMetric(metric="equality")],
    )
    agent = Agent(
        name="no-grader-agent",
        description="no code graders",
        model=LLMProvider(
            provider=ProviderEnum.OPENAI, name="gpt-4", api_key="test-key"
        ),
        instructions=Instructions(inline="x"),
        test_cases=[test_case],
        evaluations=evaluations,
        execution=None,
    )

    async def send(_msg: str) -> ExecutionResult:
        return ExecutionResult(
            response="hello",
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
    backend.invoke_once = AsyncMock()
    backend.create_session = AsyncMock(return_value=session)
    backend.__class__.__name__ = "_StubBackend"

    loader = Mock(spec=ConfigLoader)
    loader.load_agent_yaml.return_value = agent
    loader.resolve_execution_config.return_value = ExecutionConfig(llm_timeout=60)

    executor = TestExecutor(
        agent_config_path="x.yaml",
        config_loader=loader,
        file_processor=Mock(spec=FileProcessor),
        backend=backend,
        agent_config=agent,
        resolved_execution_config=ExecutionConfig(llm_timeout=60),
    )
    report = await executor.execute_tests()
    turn = report.results[0].turns[0]  # type: ignore[index]
    assert turn.grader_details is None
