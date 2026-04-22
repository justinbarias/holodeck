"""Executor integration — ExpectedTool arg matchers (US3 T024–T027).

Drives a multi-turn executor through a stubbed backend with scripted
tool_calls that carry real arguments, then asserts that
`TurnResult.arg_match_details` is populated correctly and that turn-level
`tools_matched` reflects the arg-matcher outcome.
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
from holodeck.models.test_case import ExpectedTool, TestCaseModel, Turn
from holodeck.models.token_usage import TokenUsage


def _agent(test_cases: list[TestCaseModel]) -> Agent:
    return Agent(
        name="arg-match-agent",
        description="US3 executor arg-match tests",
        model=LLMProvider(
            provider=ProviderEnum.OPENAI, name="gpt-4", api_key="test-key"
        ),
        instructions=Instructions(inline="answer"),
        test_cases=test_cases,
        evaluations=None,
        execution=None,
    )


def _exec_result(response: str, tool_calls: list[dict]) -> ExecutionResult:
    return ExecutionResult(
        response=response,
        tool_calls=tool_calls,
        tool_results=[],
        token_usage=TokenUsage.zero(),
    )


def _build_executor(
    *,
    test_case: TestCaseModel,
    scripted: list[tuple[str, list[dict]]],
) -> TestExecutor:
    agent = _agent([test_case])
    loader = Mock(spec=ConfigLoader)
    loader.load_agent_yaml.return_value = agent
    loader.resolve_execution_config.return_value = ExecutionConfig(llm_timeout=60)

    session = Mock()
    session.send = AsyncMock(side_effect=[_exec_result(r, tc) for r, tc in scripted])
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
@pytest.mark.unit
async def test_fuzzy_regex_args_populate_arg_match_details() -> None:
    case = TestCaseModel(
        name="fuzzy-regex",
        turns=[
            Turn(
                input="q",
                expected_tools=[
                    ExpectedTool(
                        name="subtract",
                        args={
                            "a": {"fuzzy": "206588"},
                            "b": {"regex": r"^181001(\.0+)?$"},
                            "c": 42,
                        },
                    )
                ],
            )
        ],
    )
    executor = _build_executor(
        test_case=case,
        scripted=[
            (
                "ok",
                [
                    {
                        "name": "Math-subtract",
                        "arguments": {
                            "a": 206588.0,
                            "b": "181001",
                            "c": 42,
                        },
                    }
                ],
            )
        ],
    )

    report = await executor.execute_tests()
    turn = (report.results[0].turns or [])[0]
    assert turn.tools_matched is True
    assert turn.arg_match_details is not None
    assert len(turn.arg_match_details) == 1
    entry = turn.arg_match_details[0]
    assert entry["expected_tool"] == "subtract"
    assert entry["matched_call_index"] == 0
    assert entry["unmatched_reason"] is None


@pytest.mark.asyncio
@pytest.mark.unit
async def test_missing_arg_fails_turn_with_reason() -> None:
    case = TestCaseModel(
        name="missing-arg",
        turns=[
            Turn(
                input="q",
                expected_tools=[ExpectedTool(name="subtract", args={"a": 206588})],
            )
        ],
    )
    executor = _build_executor(
        test_case=case,
        scripted=[
            (
                "ok",
                [{"name": "subtract", "arguments": {"b": 181001}}],
            )
        ],
    )

    report = await executor.execute_tests()
    turn = (report.results[0].turns or [])[0]
    assert turn.tools_matched is False
    assert turn.passed is False
    assert turn.arg_match_details is not None
    entry = turn.arg_match_details[0]
    assert entry["matched_call_index"] == -1
    assert "a" in entry["unmatched_reason"]
    assert "missing" in entry["unmatched_reason"]


@pytest.mark.asyncio
@pytest.mark.unit
async def test_extras_ignored() -> None:
    case = TestCaseModel(
        name="extras",
        turns=[
            Turn(
                input="q",
                expected_tools=[ExpectedTool(name="divide", args={"a": 1})],
            )
        ],
    )
    executor = _build_executor(
        test_case=case,
        scripted=[
            (
                "ok",
                [
                    {
                        "name": "Math-divide",
                        "arguments": {
                            "a": 1,
                            "rounding_mode": "half-up",
                        },
                    }
                ],
            )
        ],
    )

    report = await executor.execute_tests()
    turn = (report.results[0].turns or [])[0]
    assert turn.tools_matched is True
    assert turn.passed is True


@pytest.mark.asyncio
@pytest.mark.unit
async def test_name_matches_but_args_mismatch_fails() -> None:
    case = TestCaseModel(
        name="name-ok-args-bad",
        turns=[
            Turn(
                input="q",
                expected_tools=[ExpectedTool(name="subtract", args={"a": 206588})],
            )
        ],
    )
    executor = _build_executor(
        test_case=case,
        scripted=[
            (
                "ok",
                [{"name": "subtract", "arguments": {"a": 999}}],
            )
        ],
    )

    report = await executor.execute_tests()
    turn = (report.results[0].turns or [])[0]
    assert turn.tools_matched is False
    assert turn.passed is False
    entry = (turn.arg_match_details or [])[0]
    assert entry["matched_call_index"] == -1
    assert "999" in entry["unmatched_reason"]
