"""E2E integration for US3 — object-form expected_tools with fuzzy+regex args (T032).

Uses a scripted backend with two variants: a 'match' variant where all args
satisfy the asserted matchers, and a 'mismatch' variant where the regex
assertion fails. Confirms end-to-end SC-006 behaviour: tools_matched=True
on match, False on mismatch, and `arg_match_details` populated either way.
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


def _agent(test_case: TestCaseModel) -> Agent:
    return Agent(
        name="convfinqa-us3",
        description="ConvFinQA arg-check agent",
        model=LLMProvider(
            provider=ProviderEnum.OPENAI, name="gpt-4", api_key="test-key"
        ),
        instructions=Instructions(inline="answer numerically"),
        test_cases=[test_case],
        evaluations=None,
        execution=None,
    )


def _build_test_case() -> TestCaseModel:
    return TestCaseModel(
        name="convfinqa-us3-argcheck",
        turns=[
            Turn(input="what is 2009 cash?", ground_truth="206588"),
            Turn(
                input="what is the difference between 2009 and 2008?",
                ground_truth="25587",
                expected_tools=[
                    ExpectedTool(
                        name="subtract",
                        args={
                            "a": {"fuzzy": "206588"},
                            "b": {"regex": r"^181001(\.0+)?$"},
                        },
                    )
                ],
            ),
        ],
    )


def _make_executor(script: list[tuple[str, list[dict]]]) -> TestExecutor:
    case = _build_test_case()
    agent = _agent(case)
    session = Mock()

    async def send(_msg: str) -> ExecutionResult:
        resp, tc = script.pop(0)
        return ExecutionResult(
            response=resp,
            tool_calls=tc,
            tool_results=[],
            token_usage=TokenUsage.zero(),
        )

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

    return TestExecutor(
        agent_config_path="tests/fixtures/multi_turn/convfinqa_sample.yaml",
        config_loader=loader,
        file_processor=Mock(spec=FileProcessor),
        backend=backend,
        agent_config=agent,
        resolved_execution_config=ExecutionConfig(llm_timeout=60),
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_convfinqa_arg_match_passes() -> None:
    script = [
        ("206588", []),
        (
            "25587",
            [
                {
                    "name": "Math-subtract",
                    "arguments": {"a": 206588.0, "b": "181001"},
                }
            ],
        ),
    ]
    executor = _make_executor(script)
    report = await executor.execute_tests()
    result = report.results[0]
    assert result.turns is not None and len(result.turns) == 2
    turn2 = result.turns[1]
    assert turn2.tools_matched is True
    assert turn2.arg_match_details is not None
    entry = turn2.arg_match_details[0]
    assert entry["expected_tool"] == "subtract"
    assert entry["matched_call_index"] == 0
    assert entry["unmatched_reason"] is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_convfinqa_arg_match_fails_on_mismatch() -> None:
    script = [
        ("206588", []),
        (
            "wrong",
            [
                {
                    "name": "Math-subtract",
                    "arguments": {"a": 206588.0, "b": "180000"},
                }
            ],
        ),
    ]
    executor = _make_executor(script)
    report = await executor.execute_tests()
    result = report.results[0]
    assert result.turns is not None and len(result.turns) == 2
    turn2 = result.turns[1]
    assert turn2.tools_matched is False
    assert turn2.passed is False
    entry = (turn2.arg_match_details or [])[0]
    assert entry["matched_call_index"] == -1
    assert "b" in entry["unmatched_reason"]
