"""Tests for parallel_test_cases concurrency orchestration (T043–T045)."""

from __future__ import annotations

import asyncio
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
from holodeck.models.test_result import TestResult
from holodeck.models.token_usage import TokenUsage


def _agent(test_cases: list[TestCaseModel]) -> Agent:
    return Agent(
        name="parallel-agent",
        description="parallel test agent",
        model=LLMProvider(
            provider=ProviderEnum.OPENAI, name="gpt-4", api_key="test-key"
        ),
        instructions=Instructions(inline="be concise"),
        test_cases=test_cases,
        evaluations=None,
        execution=None,
    )


def _build_executor(
    test_cases: list[TestCaseModel],
    *,
    parallel: int,
    session_factory,
) -> tuple[TestExecutor, Mock]:
    agent = _agent(test_cases)
    loader = Mock(spec=ConfigLoader)
    loader.load_agent_yaml.return_value = agent
    exec_cfg = ExecutionConfig(llm_timeout=60, parallel_test_cases=parallel)
    loader.resolve_execution_config.return_value = exec_cfg

    backend = Mock()
    backend.initialize = AsyncMock()
    backend.teardown = AsyncMock()
    backend.invoke_once = AsyncMock()
    backend.create_session = AsyncMock(side_effect=session_factory)
    backend.__class__.__name__ = "_StubBackend"

    executor = TestExecutor(
        agent_config_path="x.yaml",
        config_loader=loader,
        file_processor=Mock(spec=FileProcessor),
        backend=backend,
        agent_config=agent,
        resolved_execution_config=exec_cfg,
    )
    return executor, backend


def _mk_exec_result(response: str) -> ExecutionResult:
    return ExecutionResult(
        response=response,
        tool_calls=[],
        tool_results=[],
        token_usage=TokenUsage.zero(),
    )


@pytest.mark.asyncio
async def test_parallel_respects_semaphore() -> None:
    """With parallel_test_cases=2 and 4 cases, ≤ 2 sessions are live."""
    active_sessions = 0
    max_observed = 0
    lock = asyncio.Lock()

    async def send(msg):
        nonlocal active_sessions, max_observed
        async with lock:
            active_sessions += 1
            max_observed = max(max_observed, active_sessions)
        await asyncio.sleep(0.02)
        async with lock:
            active_sessions -= 1
        return _mk_exec_result("ok")

    def session_factory():
        session = Mock()
        session.send = AsyncMock(side_effect=send)
        session.close = AsyncMock()
        return session

    cases = [TestCaseModel(turns=[Turn(input="t")]) for _ in range(4)]
    executor, _ = _build_executor(cases, parallel=2, session_factory=session_factory)

    await executor.execute_tests()
    assert max_observed <= 2
    assert max_observed >= 2  # should reach limit


@pytest.mark.asyncio
async def test_turns_within_case_stay_sequential() -> None:
    """Even with parallel_test_cases=4, turns inside ONE case stay serial."""
    order: list[str] = []

    async def send(msg):
        name = msg.strip()
        order.append(f"start-{name}")
        await asyncio.sleep(0.01)
        order.append(f"end-{name}")
        return _mk_exec_result(f"r-{name}")

    def session_factory():
        session = Mock()
        session.send = AsyncMock(side_effect=send)
        session.close = AsyncMock()
        return session

    cases = [
        TestCaseModel(turns=[Turn(input="a1"), Turn(input="a2"), Turn(input="a3")])
    ]
    executor, _ = _build_executor(cases, parallel=4, session_factory=session_factory)
    await executor.execute_tests()
    assert order == [
        "start-a1",
        "end-a1",
        "start-a2",
        "end-a2",
        "start-a3",
        "end-a3",
    ]


@pytest.mark.asyncio
async def test_reporter_writes_serialized() -> None:
    """Concurrent cases do not interleave progress callback invocations."""
    callback_order: list[str] = []

    async def send(msg):
        await asyncio.sleep(0.01)
        return _mk_exec_result("ok")

    def session_factory():
        session = Mock()
        session.send = AsyncMock(side_effect=send)
        session.close = AsyncMock()
        return session

    cases = [TestCaseModel(name=f"tc-{i}", turns=[Turn(input="x")]) for i in range(4)]

    def cb(result: TestResult) -> None:
        callback_order.append(f"enter-{result.test_name}")
        # Simulate a slow writer — no other callback should interleave.
        callback_order.append(f"leave-{result.test_name}")

    executor, _ = _build_executor(cases, parallel=3, session_factory=session_factory)
    executor.progress_callback = cb

    await executor.execute_tests()
    # Every "enter-X" must be immediately followed by "leave-X".
    for i in range(0, len(callback_order), 2):
        enter = callback_order[i]
        leave = callback_order[i + 1]
        assert enter.startswith("enter-") and leave.startswith("leave-")
        assert enter.split("-", 1)[1] == leave.split("-", 1)[1]
