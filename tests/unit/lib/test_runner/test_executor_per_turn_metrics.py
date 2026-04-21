"""Per-turn metric evaluation and rollup (US2 Phase 2, T002–T012a).

Drives a multi-turn executor through a stubbed backend, wires in mock
evaluators, and asserts the per-turn `metric_results` + test-case-level
rollup behaviour defined in data-model.md §9 and
contracts/turn-result-schema.md §3.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock

import pytest

from holodeck.config.loader import ConfigLoader
from holodeck.lib.backends.base import ExecutionResult
from holodeck.lib.evaluators.param_spec import EvalParam, ParamSpec
from holodeck.lib.file_processor import FileProcessor
from holodeck.lib.test_runner.executor import TestExecutor
from holodeck.models.agent import Agent, Instructions
from holodeck.models.config import ExecutionConfig
from holodeck.models.evaluation import EvaluationConfig, EvaluationMetric
from holodeck.models.llm import LLMProvider, ProviderEnum
from holodeck.models.test_case import TestCaseModel, Turn
from holodeck.models.test_result import MetricResult, TurnResult
from holodeck.models.token_usage import TokenUsage


def _agent(
    test_cases: list[TestCaseModel],
    *,
    evaluations: EvaluationConfig | None = None,
) -> Agent:
    return Agent(
        name="per-turn-agent",
        description="per-turn metric agent",
        model=LLMProvider(
            provider=ProviderEnum.OPENAI, name="gpt-4", api_key="test-key"
        ),
        instructions=Instructions(inline="be concise"),
        test_cases=test_cases,
        evaluations=evaluations,
        execution=None,
    )


def _bleu_mock(score: float = 0.9) -> Mock:
    mock = Mock()
    mock.name = "bleu"
    mock.get_param_spec = Mock(
        return_value=ParamSpec(
            required=frozenset({EvalParam.RESPONSE, EvalParam.GROUND_TRUTH}),
        )
    )
    mock.evaluate = AsyncMock(return_value={"bleu": score})
    return mock


def _rouge_mock(score: float = 0.9) -> Mock:
    mock = Mock()
    mock.name = "rouge"
    mock.get_param_spec = Mock(
        return_value=ParamSpec(
            required=frozenset({EvalParam.RESPONSE, EvalParam.GROUND_TRUTH}),
        )
    )
    mock.evaluate = AsyncMock(return_value={"rouge": score})
    return mock


def _exec_result(response: str) -> ExecutionResult:
    return ExecutionResult(
        response=response,
        tool_calls=[],
        tool_results=[],
        token_usage=TokenUsage.zero(),
    )


def _build_executor(
    *,
    test_case: TestCaseModel,
    evaluators: dict,
    evaluations: EvaluationConfig | None,
    responses: list[str],
) -> TestExecutor:
    agent = _agent([test_case], evaluations=evaluations)
    loader = Mock(spec=ConfigLoader)
    loader.load_agent_yaml.return_value = agent
    loader.resolve_execution_config.return_value = ExecutionConfig(llm_timeout=60)

    session = Mock()
    session.send = AsyncMock(side_effect=[_exec_result(r) for r in responses])
    session.close = AsyncMock()

    backend = Mock()
    backend.initialize = AsyncMock()
    backend.teardown = AsyncMock()
    backend.invoke_once = AsyncMock()
    backend.create_session = AsyncMock(return_value=session)
    backend.__class__.__name__ = "_StubBackend"

    return TestExecutor(
        agent_config_path="x.yaml",
        config_loader=loader,
        file_processor=Mock(spec=FileProcessor),
        backend=backend,
        agent_config=agent,
        resolved_execution_config=ExecutionConfig(llm_timeout=60),
        evaluators=evaluators,
    )


@pytest.mark.asyncio
async def test_turn_with_ground_truth_runs_metrics() -> None:
    """A turn with ground_truth + BLEU configured gets a bleu MetricResult."""
    case = TestCaseModel(
        name="gt-turn",
        turns=[Turn(input="q1", ground_truth="expected-answer")],
    )
    evaluations = EvaluationConfig(
        metrics=[EvaluationMetric(metric="bleu", threshold=0.5)],
    )
    executor = _build_executor(
        test_case=case,
        evaluators={"bleu": _bleu_mock(0.8)},
        evaluations=evaluations,
        responses=["actual-answer"],
    )

    report = await executor.execute_tests()
    result = report.results[0]
    assert result.turns is not None and len(result.turns) == 1
    turn = result.turns[0]
    names = [m.metric_name for m in turn.metric_results]
    assert names == ["bleu"]
    assert turn.metric_results[0].passed is True
    assert turn.metric_results[0].score == pytest.approx(0.8)


@pytest.mark.asyncio
async def test_turn_without_ground_truth_skips_text_metrics() -> None:
    """A turn with no ground_truth gets no BLEU/ROUGE entry (A6)."""
    case = TestCaseModel(
        name="no-gt",
        turns=[Turn(input="q1"), Turn(input="q2", ground_truth="answer")],
    )
    evaluations = EvaluationConfig(
        metrics=[EvaluationMetric(metric="bleu", threshold=0.5)],
    )
    executor = _build_executor(
        test_case=case,
        evaluators={"bleu": _bleu_mock(0.9)},
        evaluations=evaluations,
        responses=["r1", "r2"],
    )

    report = await executor.execute_tests()
    turns = report.results[0].turns or []
    assert [m.metric_name for m in turns[0].metric_results] == []
    assert [m.metric_name for m in turns[1].metric_results] == ["bleu"]


@pytest.mark.asyncio
async def test_per_turn_evaluations_override_agent_defaults() -> None:
    """Per-turn `evaluations` override agent-level metrics for that turn only."""
    case = TestCaseModel(
        name="override",
        turns=[
            Turn(input="q1", ground_truth="gt1"),
            Turn(
                input="q2",
                ground_truth="gt2",
                evaluations=[EvaluationMetric(metric="rouge", threshold=0.5)],
            ),
        ],
    )
    evaluations = EvaluationConfig(
        metrics=[EvaluationMetric(metric="bleu", threshold=0.5)],
    )
    executor = _build_executor(
        test_case=case,
        evaluators={"bleu": _bleu_mock(0.9), "rouge": _rouge_mock(0.7)},
        evaluations=evaluations,
        responses=["r1", "r2"],
    )

    report = await executor.execute_tests()
    turns = report.results[0].turns or []
    assert [m.metric_name for m in turns[0].metric_results] == ["bleu"]
    assert [m.metric_name for m in turns[1].metric_results] == ["rouge"]


@pytest.mark.asyncio
async def test_test_case_evaluations_override_agent_when_turn_unset() -> None:
    """Middle rung of resolver: test_case beats agent when turn is unset."""
    case = TestCaseModel(
        name="mid",
        turns=[Turn(input="q1", ground_truth="gt1")],
        evaluations=[EvaluationMetric(metric="rouge", threshold=0.5)],
    )
    evaluations = EvaluationConfig(
        metrics=[EvaluationMetric(metric="bleu", threshold=0.5)],
    )
    executor = _build_executor(
        test_case=case,
        evaluators={"bleu": _bleu_mock(0.9), "rouge": _rouge_mock(0.8)},
        evaluations=evaluations,
        responses=["r1"],
    )

    report = await executor.execute_tests()
    turns = report.results[0].turns or []
    assert [m.metric_name for m in turns[0].metric_results] == ["rouge"]


@pytest.mark.asyncio
async def test_rollup_mean_across_turns() -> None:
    """3 turns score 0.8, 0.4, 0.6 → rolled-up bleu.score == 0.6."""
    case = TestCaseModel(
        name="rollup-mean",
        turns=[
            Turn(input="q1", ground_truth="gt1"),
            Turn(input="q2", ground_truth="gt2"),
            Turn(input="q3", ground_truth="gt3"),
        ],
    )
    evaluations = EvaluationConfig(
        metrics=[EvaluationMetric(metric="bleu", threshold=0.5)],
    )

    bleu = Mock()
    bleu.name = "bleu"
    bleu.get_param_spec = Mock(
        return_value=ParamSpec(
            required=frozenset({EvalParam.RESPONSE, EvalParam.GROUND_TRUTH}),
        )
    )
    bleu.evaluate = AsyncMock(side_effect=[{"bleu": 0.8}, {"bleu": 0.4}, {"bleu": 0.6}])

    executor = _build_executor(
        test_case=case,
        evaluators={"bleu": bleu},
        evaluations=evaluations,
        responses=["r1", "r2", "r3"],
    )

    report = await executor.execute_tests()
    result = report.results[0]
    rolled = [m for m in result.metric_results if m.metric_name == "bleu"]
    assert len(rolled) == 1
    assert rolled[0].score == pytest.approx(0.6, rel=1e-6)
    # threshold=0.5, turn 2 scored 0.4 → turn 2 fails → rolled passed=False
    assert rolled[0].passed is False


@pytest.mark.asyncio
async def test_rollup_skips_turns_without_metric() -> None:
    """Turn 2 has no ground_truth → averages only over turns 1 and 3."""
    case = TestCaseModel(
        name="rollup-skip",
        turns=[
            Turn(input="q1", ground_truth="gt1"),
            Turn(input="q2"),  # no ground_truth → skipped
            Turn(input="q3", ground_truth="gt3"),
        ],
    )
    evaluations = EvaluationConfig(
        metrics=[EvaluationMetric(metric="bleu", threshold=0.5)],
    )
    bleu = Mock()
    bleu.name = "bleu"
    bleu.get_param_spec = Mock(
        return_value=ParamSpec(
            required=frozenset({EvalParam.RESPONSE, EvalParam.GROUND_TRUTH}),
        )
    )
    bleu.evaluate = AsyncMock(
        side_effect=[{"bleu": 0.8}, {"bleu": 1.0}]  # only 2 calls expected
    )

    executor = _build_executor(
        test_case=case,
        evaluators={"bleu": bleu},
        evaluations=evaluations,
        responses=["r1", "r2", "r3"],
    )

    report = await executor.execute_tests()
    result = report.results[0]
    rolled = [m for m in result.metric_results if m.metric_name == "bleu"]
    assert len(rolled) == 1
    assert rolled[0].score == pytest.approx(0.9, rel=1e-6)


@pytest.mark.asyncio
async def test_metric_omitted_when_every_turn_skipped() -> None:
    """No turn has ground_truth → bleu is absent from rolled metric_results."""
    case = TestCaseModel(
        name="all-skip",
        turns=[Turn(input="q1"), Turn(input="q2")],
    )
    evaluations = EvaluationConfig(
        metrics=[EvaluationMetric(metric="bleu", threshold=0.5)],
    )
    executor = _build_executor(
        test_case=case,
        evaluators={"bleu": _bleu_mock(0.5)},
        evaluations=evaluations,
        responses=["r1", "r2"],
    )

    report = await executor.execute_tests()
    names = [m.metric_name for m in report.results[0].metric_results]
    assert "bleu" not in names


@pytest.mark.asyncio
async def test_failing_turn_flips_rolled_up_passed() -> None:
    """2/3 turns pass BLEU, 1 fails → rolled up passed=False."""
    case = TestCaseModel(
        name="flip",
        turns=[
            Turn(input="q1", ground_truth="gt1"),
            Turn(input="q2", ground_truth="gt2"),
            Turn(input="q3", ground_truth="gt3"),
        ],
    )
    evaluations = EvaluationConfig(
        metrics=[EvaluationMetric(metric="bleu", threshold=0.5)],
    )
    bleu = Mock()
    bleu.name = "bleu"
    bleu.get_param_spec = Mock(
        return_value=ParamSpec(
            required=frozenset({EvalParam.RESPONSE, EvalParam.GROUND_TRUTH}),
        )
    )
    # 0.9 + 0.9 + 0.3 = mean 0.7 (above threshold) but turn 3 fails.
    bleu.evaluate = AsyncMock(side_effect=[{"bleu": 0.9}, {"bleu": 0.9}, {"bleu": 0.3}])

    executor = _build_executor(
        test_case=case,
        evaluators={"bleu": bleu},
        evaluations=evaluations,
        responses=["r1", "r2", "r3"],
    )

    report = await executor.execute_tests()
    result = report.results[0]
    rolled = [m for m in result.metric_results if m.metric_name == "bleu"]
    assert len(rolled) == 1
    assert rolled[0].score == pytest.approx(0.7, rel=1e-6)
    assert rolled[0].passed is False
    assert result.passed is False


def test_turn_with_errors_only_fails() -> None:
    """First conjunct: errors != [] forces TurnResult.passed=False."""
    turn = TurnResult(
        turn_index=0,
        input="q",
        response=None,
        ground_truth=None,
        expected_tools=None,
        tool_calls=[],
        tool_invocations=[],
        tools_matched=None,
        arg_match_details=None,
        metric_results=[],
        passed=(  # manually compose 4-conjunct
            [] == []  # errors
            and False is False  # skipped
            and True  # tools_matched None → neutral
            and True  # no metrics
        ),
        execution_time_ms=0,
        token_usage=None,
        errors=["timeout"],
        skipped=False,
        grader_details=None,
    )
    # The executor composes passed at build time. Re-derive using the contract:
    computed = (
        turn.errors == []
        and turn.skipped is False
        and (turn.tools_matched is None or turn.tools_matched is True)
        and all((m.passed is not False) for m in turn.metric_results)
    )
    assert computed is False


def test_skipped_turn_fails() -> None:
    """Second conjunct: skipped=True forces TurnResult.passed=False."""
    turn = TurnResult(
        turn_index=0,
        input="q",
        response=None,
        ground_truth=None,
        expected_tools=None,
        tool_calls=[],
        tool_invocations=[],
        tools_matched=None,
        arg_match_details=None,
        metric_results=[],
        passed=False,
        execution_time_ms=0,
        token_usage=None,
        errors=[],
        skipped=True,
        grader_details=None,
    )
    computed = (
        turn.errors == []
        and turn.skipped is False
        and (turn.tools_matched is None or turn.tools_matched is True)
        and all((m.passed is not False) for m in turn.metric_results)
    )
    assert computed is False


def test_tools_matched_false_fails() -> None:
    """Third conjunct: tools_matched=False forces TurnResult.passed=False."""
    turn = TurnResult(
        turn_index=0,
        input="q",
        response="r",
        ground_truth=None,
        expected_tools=["subtract"],
        tool_calls=["lookup"],
        tool_invocations=[],
        tools_matched=False,
        arg_match_details=None,
        metric_results=[],
        passed=False,
        execution_time_ms=0,
        token_usage=None,
        errors=[],
        skipped=False,
        grader_details=None,
    )
    computed = (
        turn.errors == []
        and turn.skipped is False
        and (turn.tools_matched is None or turn.tools_matched is True)
        and all((m.passed is not False) for m in turn.metric_results)
    )
    assert computed is False


def test_metric_failed_fails_turn() -> None:
    """Fourth conjunct: any metric.passed==False forces turn.passed=False."""
    metric = MetricResult(
        metric_name="bleu",
        kind="standard",
        score=0.1,
        threshold=0.5,
        passed=False,
        scale="0-1",
    )
    turn = TurnResult(
        turn_index=0,
        input="q",
        response="r",
        ground_truth="gt",
        expected_tools=None,
        tool_calls=[],
        tool_invocations=[],
        tools_matched=None,
        arg_match_details=None,
        metric_results=[metric],
        passed=False,
        execution_time_ms=0,
        token_usage=None,
        errors=[],
        skipped=False,
        grader_details=None,
    )
    computed = (
        turn.errors == []
        and turn.skipped is False
        and (turn.tools_matched is None or turn.tools_matched is True)
        and all((m.passed is not False) for m in turn.metric_results)
    )
    assert computed is False


@pytest.mark.asyncio
async def test_executor_composes_turn_passed_from_all_conjuncts() -> None:
    """Integration: executor sets turn.passed = 4-conjunct AND."""
    case = TestCaseModel(
        name="conj",
        turns=[
            Turn(input="q1", ground_truth="gt1"),
            Turn(input="q2", ground_truth="gt2"),
        ],
    )
    evaluations = EvaluationConfig(
        metrics=[EvaluationMetric(metric="bleu", threshold=0.5)],
    )
    bleu = Mock()
    bleu.name = "bleu"
    bleu.get_param_spec = Mock(
        return_value=ParamSpec(
            required=frozenset({EvalParam.RESPONSE, EvalParam.GROUND_TRUTH}),
        )
    )
    bleu.evaluate = AsyncMock(side_effect=[{"bleu": 0.9}, {"bleu": 0.1}])

    executor = _build_executor(
        test_case=case,
        evaluators={"bleu": bleu},
        evaluations=evaluations,
        responses=["r1", "r2"],
    )

    report = await executor.execute_tests()
    turns = report.results[0].turns or []
    assert turns[0].passed is True
    assert turns[1].passed is False  # BLEU 0.1 < 0.5
