"""Per-turn tool-name assertion (US2 Phase 3, T014–T019).

Drives a multi-turn executor through a stubbed backend, scripts the
per-turn `tool_calls` returned on `ExecutionResult`, and asserts the
per-turn + test-case-level `tools_matched` behaviour from
contracts/turn-result-schema.md §2 and §4.
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
from holodeck.models.llm import LLMProvider, ProviderEnum
from holodeck.models.test_case import TestCaseModel, Turn
from holodeck.models.test_result import TurnResult
from holodeck.models.token_usage import TokenUsage


def _agent(test_cases: list[TestCaseModel]) -> Agent:
    return Agent(
        name="tool-turn-agent",
        description="per-turn tool unit-test agent",
        model=LLMProvider(
            provider=ProviderEnum.OPENAI, name="gpt-4", api_key="test-key"
        ),
        instructions=Instructions(inline="be concise"),
        test_cases=test_cases,
        evaluations=None,
        execution=None,
    )


def _exec_result(
    response: str,
    *,
    tool_call_names: list[str] | None = None,
) -> ExecutionResult:
    tool_call_names = tool_call_names or []
    tool_calls = [{"name": name, "arguments": {}} for name in tool_call_names]
    return ExecutionResult(
        response=response,
        tool_calls=tool_calls,
        tool_results=[],
        token_usage=TokenUsage.zero(),
    )


def _build_executor(
    *,
    test_case: TestCaseModel,
    per_turn_tool_names: list[list[str]],
    responses: list[str] | None = None,
) -> TestExecutor:
    agent = _agent([test_case])
    loader = Mock(spec=ConfigLoader)
    loader.load_agent_yaml.return_value = agent
    loader.resolve_execution_config.return_value = ExecutionConfig(llm_timeout=60)

    responses = responses or [f"r{i}" for i in range(len(per_turn_tool_names))]
    session = Mock()
    session.send = AsyncMock(
        side_effect=[
            _exec_result(r, tool_call_names=tc)
            for r, tc in zip(responses, per_turn_tool_names, strict=True)
        ]
    )
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
    )


@pytest.mark.asyncio
async def test_tool_name_match_substring_scoped_to_turn() -> None:
    """Turn 3 asserts `subtract`; turn 1 also called it. In-turn only counts."""
    case = TestCaseModel(
        name="scoped",
        turns=[
            Turn(input="t1", expected_tools=None),
            Turn(input="t2", expected_tools=None),
            Turn(input="t3", expected_tools=["subtract"]),
        ],
    )
    # Turn 1 calls subtract, turn 3 does NOT → turn 3 should fail.
    executor = _build_executor(
        test_case=case,
        per_turn_tool_names=[["subtract"], [], ["lookup"]],
    )

    report = await executor.execute_tests()
    turns = report.results[0].turns or []
    assert turns[0].tools_matched is None
    assert turns[1].tools_matched is None
    assert turns[2].tools_matched is False
    assert turns[2].passed is False


@pytest.mark.asyncio
async def test_per_turn_tool_match_inherits_substring_contract() -> None:
    """SK plugin prefixing (`Math-subtract`) satisfies expected `subtract`."""
    case = TestCaseModel(
        name="substring",
        turns=[Turn(input="t", expected_tools=["subtract"])],
    )
    executor = _build_executor(
        test_case=case,
        per_turn_tool_names=[["Math-subtract"]],
    )

    report = await executor.execute_tests()
    turns = report.results[0].turns or []
    assert turns[0].tools_matched is True


@pytest.mark.asyncio
async def test_missing_expected_tool_fails_turn() -> None:
    """expected `divide`, agent calls nothing → tools_matched=False + errors set."""
    case = TestCaseModel(
        name="missing",
        turns=[Turn(input="t", expected_tools=["divide"])],
    )
    executor = _build_executor(
        test_case=case,
        per_turn_tool_names=[[]],
    )

    report = await executor.execute_tests()
    turn = (report.results[0].turns or [])[0]
    assert turn.tools_matched is False
    assert turn.passed is False
    assert any("divide" in e for e in turn.errors)


@pytest.mark.asyncio
async def test_extra_tools_do_not_fail_turn() -> None:
    """expected_tools is a lower bound; extra tool calls do not fail assertion."""
    case = TestCaseModel(
        name="extra",
        turns=[Turn(input="t", expected_tools=["subtract"])],
    )
    executor = _build_executor(
        test_case=case,
        per_turn_tool_names=[["subtract", "lookup"]],
    )

    report = await executor.execute_tests()
    turn = (report.results[0].turns or [])[0]
    assert turn.tools_matched is True
    assert turn.passed is True


@pytest.mark.asyncio
async def test_no_expected_tools_leaves_matched_none() -> None:
    """No `expected_tools` → turn.tools_matched is None (neutral)."""
    case = TestCaseModel(
        name="none",
        turns=[Turn(input="t")],
    )
    executor = _build_executor(
        test_case=case,
        per_turn_tool_names=[["lookup"]],
    )

    report = await executor.execute_tests()
    result = report.results[0]
    turn = (result.turns or [])[0]
    assert turn.tools_matched is None
    # Test-case rollup: no turn asserted → TestResult.tools_matched is None
    assert result.tools_matched is None


@pytest.mark.asyncio
async def test_rollup_tools_matched_all_true() -> None:
    """turn1=True, turn2=None, turn3=True → test-case True (None is neutral)."""
    case = TestCaseModel(
        name="all-true",
        turns=[
            Turn(input="t1", expected_tools=["a"]),
            Turn(input="t2"),
            Turn(input="t3", expected_tools=["b"]),
        ],
    )
    executor = _build_executor(
        test_case=case,
        per_turn_tool_names=[["a"], [], ["b"]],
    )

    report = await executor.execute_tests()
    assert report.results[0].tools_matched is True


@pytest.mark.asyncio
async def test_rollup_tools_matched_all_none() -> None:
    """No turn asserts anything → test-case tools_matched is None."""
    case = TestCaseModel(
        name="all-none",
        turns=[Turn(input="t1"), Turn(input="t2")],
    )
    executor = _build_executor(
        test_case=case,
        per_turn_tool_names=[[], []],
    )

    report = await executor.execute_tests()
    assert report.results[0].tools_matched is None


@pytest.mark.asyncio
async def test_rollup_tools_matched_any_false() -> None:
    """Any turn explicitly False → test-case tools_matched is False."""
    case = TestCaseModel(
        name="any-false",
        turns=[
            Turn(input="t1", expected_tools=["a"]),
            Turn(input="t2", expected_tools=["b"]),
        ],
    )
    executor = _build_executor(
        test_case=case,
        per_turn_tool_names=[["a"], ["other"]],
    )

    report = await executor.execute_tests()
    result = report.results[0]
    turns = result.turns or []
    assert turns[0].tools_matched is True
    assert turns[1].tools_matched is False
    assert result.tools_matched is False


def test_rollup_helper_pure_logic() -> None:
    """Direct unit test for the rollup rule (contracts §4)."""
    from holodeck.lib.test_runner.executor import _rollup_tools_matched

    assert _rollup_tools_matched([None, None]) is None
    assert _rollup_tools_matched([True, None]) is True
    assert _rollup_tools_matched([True, False]) is False
    assert _rollup_tools_matched([False]) is False


def _mk_turn(tools_matched: bool | None, skipped: bool = False) -> TurnResult:
    return TurnResult(
        turn_index=0,
        input="x",
        response=None,
        ground_truth=None,
        expected_tools=None,
        tool_calls=[],
        tool_invocations=[],
        tools_matched=tools_matched,
        arg_match_details=None,
        metric_results=[],
        passed=True,
        execution_time_ms=0,
        token_usage=None,
        errors=[],
        skipped=skipped,
        grader_details=None,
    )
