"""Tests for multi-turn dispatch in TestExecutor (T018–T027).

Covers the US1 core: the executor detects `turns` and routes through
`backend.create_session()` + `session.send()` instead of `invoke_once()`,
drives turns strictly sequentially, closes the session always, handles
per-turn timeouts / backend errors, and rolls up per-turn results
without cross-turn tool bleed.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from holodeck.config.loader import ConfigLoader
from holodeck.lib.backends.base import (
    AgentBackend,
    BackendSessionError,
    ExecutionResult,
)
from holodeck.lib.file_processor import FileProcessor
from holodeck.lib.test_runner.executor import TestExecutor
from holodeck.models.agent import Agent, Instructions
from holodeck.models.config import ExecutionConfig
from holodeck.models.llm import LLMProvider, ProviderEnum
from holodeck.models.test_case import FileInput, TestCaseModel, Turn
from holodeck.models.token_usage import TokenUsage


def _make_agent(test_cases: list[TestCaseModel]) -> Agent:
    return Agent(
        name="multi-turn-agent",
        description="multi-turn unit-test agent",
        model=LLMProvider(
            provider=ProviderEnum.OPENAI, name="gpt-4", api_key="test-key"
        ),
        instructions=Instructions(inline="be concise"),
        test_cases=test_cases,
        evaluations=None,
        execution=None,
    )


def _mk_exec_result(
    response: str,
    *,
    tool_calls: list | None = None,
    tool_results: list | None = None,
    token_usage: TokenUsage | None = None,
) -> ExecutionResult:
    return ExecutionResult(
        response=response,
        tool_calls=tool_calls or [],
        tool_results=tool_results or [],
        token_usage=token_usage or TokenUsage.zero(),
    )


class _StubBackend:
    """Minimal AgentBackend stub that records session usage."""

    def __init__(self, session_factory):
        self._factory = session_factory
        self.create_session = AsyncMock(side_effect=self._create)
        self.invoke_once = AsyncMock()
        self.initialize = AsyncMock()
        self.teardown = AsyncMock()
        self.sessions: list = []

    async def _create(self):
        session = self._factory()
        self.sessions.append(session)
        return session


def _stub_session(send_side_effects):
    session = Mock()
    session.send = AsyncMock(side_effect=send_side_effects)
    session.send_streaming = AsyncMock()
    session.close = AsyncMock()
    return session


def _make_executor(backend: AgentBackend, test_case: TestCaseModel) -> TestExecutor:
    agent = _make_agent([test_case])
    loader = Mock(spec=ConfigLoader)
    loader.load_agent_yaml.return_value = agent
    loader.resolve_execution_config.return_value = ExecutionConfig(llm_timeout=60)
    file_proc = Mock(spec=FileProcessor)
    return TestExecutor(
        agent_config_path="tests/fixtures/multi_turn/convfinqa_sample.yaml",
        config_loader=loader,
        file_processor=file_proc,
        backend=backend,
        agent_config=agent,
        resolved_execution_config=ExecutionConfig(llm_timeout=60),
    )


@pytest.mark.asyncio
async def test_dispatch_detects_turns() -> None:
    tc = TestCaseModel(
        name="mt",
        turns=[Turn(input="a"), Turn(input="b")],
    )
    session = _stub_session([_mk_exec_result("resp-a"), _mk_exec_result("resp-b")])
    backend = _StubBackend(lambda: session)
    executor = _make_executor(backend, tc)

    report = await executor.execute_tests()

    backend.create_session.assert_awaited_once()
    backend.invoke_once.assert_not_called()
    assert session.send.await_count == 2
    assert report.summary.total_tests == 1
    assert report.results[0].turns is not None
    assert len(report.results[0].turns) == 2


@pytest.mark.asyncio
async def test_legacy_single_turn_unchanged() -> None:
    tc = TestCaseModel(name="legacy", input="hi")
    backend = _StubBackend(lambda: _stub_session([]))
    backend.invoke_once = AsyncMock(return_value=_mk_exec_result("hi back"))
    executor = _make_executor(backend, tc)

    report = await executor.execute_tests()

    backend.invoke_once.assert_awaited_once()
    backend.create_session.assert_not_called()
    assert report.results[0].turns is None


@pytest.mark.asyncio
async def test_turns_strictly_sequential() -> None:
    """Turn N+1's send() may not start until turn N resolves."""
    order: list[str] = []

    async def make_send(name: str, delay: float):
        await asyncio.sleep(delay)
        order.append(f"end-{name}")
        return _mk_exec_result(f"r-{name}")

    async def send(msg):
        name = msg.strip()
        order.append(f"start-{name}")
        # Yield to the scheduler so any concurrent start would be observed.
        await asyncio.sleep(0.01)
        order.append(f"end-{name}")
        return _mk_exec_result(f"r-{name}")

    session = Mock()
    session.send = AsyncMock(side_effect=send)
    session.close = AsyncMock()
    backend = _StubBackend(lambda: session)

    tc = TestCaseModel(turns=[Turn(input="a"), Turn(input="b"), Turn(input="c")])
    executor = _make_executor(backend, tc)
    await executor.execute_tests()

    # Every start must be followed immediately by its own end — no overlap.
    assert order == [
        "start-a",
        "end-a",
        "start-b",
        "end-b",
        "start-c",
        "end-c",
    ]


@pytest.mark.asyncio
async def test_session_closed_on_completion() -> None:
    session = _stub_session([_mk_exec_result("ok"), _mk_exec_result("ok")])
    backend = _StubBackend(lambda: session)
    tc = TestCaseModel(turns=[Turn(input="a"), Turn(input="b")])
    executor = _make_executor(backend, tc)

    await executor.execute_tests()
    session.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_session_closed_on_failure() -> None:
    session = _stub_session([RuntimeError("boom")])
    backend = _StubBackend(lambda: session)
    tc = TestCaseModel(turns=[Turn(input="a")])
    executor = _make_executor(backend, tc)

    await executor.execute_tests()
    session.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_turn_timeout_fails_turn_continues_next() -> None:
    """A TimeoutError on turn 1 fails that turn but turn 2 still runs."""
    call_count = 0

    async def send(msg):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise asyncio.TimeoutError()
        return _mk_exec_result("second-response")

    session = Mock()
    session.send = AsyncMock(side_effect=send)
    session.close = AsyncMock()
    backend = _StubBackend(lambda: session)

    tc = TestCaseModel(turns=[Turn(input="a"), Turn(input="b")])
    executor = _make_executor(backend, tc)
    report = await executor.execute_tests()

    turns = report.results[0].turns
    assert turns is not None
    assert turns[0].passed is False
    assert turns[0].errors == ["timeout"]
    assert turns[0].skipped is False
    assert turns[1].response == "second-response"
    assert turns[1].passed is True


@pytest.mark.asyncio
async def test_two_consecutive_session_errors_mark_remaining_skipped() -> None:
    """Two consecutive BackendSessionErrors → remaining turns are skipped."""
    call_count = 0

    async def send(msg):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _mk_exec_result("first ok")
        raise BackendSessionError("session lost")

    session = Mock()
    session.send = AsyncMock(side_effect=send)
    session.close = AsyncMock()
    backend = _StubBackend(lambda: session)

    tc = TestCaseModel(
        turns=[Turn(input="a"), Turn(input="b"), Turn(input="c"), Turn(input="d")]
    )
    executor = _make_executor(backend, tc)
    report = await executor.execute_tests()

    turns = report.results[0].turns
    assert turns is not None
    assert len(turns) == 4
    assert turns[0].passed is True
    assert turns[1].passed is False and turns[1].skipped is False
    assert turns[2].passed is False and turns[2].skipped is False
    # After two consecutive session errors (turns 2 and 3, indexes 1 and 2),
    # subsequent turns are skipped.
    assert turns[3].skipped is True
    assert any("unrecoverable" in e.lower() for e in turns[3].errors)
    # A single isolated BackendSessionError should NOT trigger skipping.


@pytest.mark.asyncio
async def test_isolated_session_error_does_not_skip() -> None:
    call_count = 0

    async def send(msg):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise BackendSessionError("blip")
        return _mk_exec_result(f"r{call_count}")

    session = Mock()
    session.send = AsyncMock(side_effect=send)
    session.close = AsyncMock()
    backend = _StubBackend(lambda: session)

    tc = TestCaseModel(
        turns=[Turn(input="a"), Turn(input="b"), Turn(input="c"), Turn(input="d")]
    )
    executor = _make_executor(backend, tc)
    report = await executor.execute_tests()

    turns = report.results[0].turns
    assert turns is not None
    assert len(turns) == 4
    assert turns[1].passed is False
    assert turns[1].skipped is False
    assert turns[2].passed is True  # recovered
    assert turns[3].passed is True


@pytest.mark.asyncio
async def test_per_turn_tool_invocations_partitioned() -> None:
    """Each turn owns its tool invocations — no cross-turn bleed."""
    session = _stub_session(
        [
            _mk_exec_result(
                "r1",
                tool_calls=[{"name": "search", "args": {}, "id": "c1"}],
                tool_results=[{"name": "search", "result": "a"}],
            ),
            _mk_exec_result(
                "r2",
                tool_calls=[{"name": "calc", "args": {}, "id": "c2"}],
                tool_results=[{"name": "calc", "result": "b"}],
            ),
        ]
    )
    backend = _StubBackend(lambda: session)
    tc = TestCaseModel(turns=[Turn(input="a"), Turn(input="b")])
    executor = _make_executor(backend, tc)
    report = await executor.execute_tests()
    turns = report.results[0].turns
    assert turns is not None
    assert [ti.name for ti in turns[0].tool_invocations] == ["search"]
    assert [ti.name for ti in turns[1].tool_invocations] == ["calc"]


@pytest.mark.asyncio
async def test_token_usage_rollup_sum() -> None:
    session = _stub_session(
        [
            _mk_exec_result(
                "r1",
                token_usage=TokenUsage(
                    prompt_tokens=10,
                    completion_tokens=5,
                    total_tokens=15,
                    cache_creation_tokens=2,
                    cache_read_tokens=3,
                ),
            ),
            _mk_exec_result(
                "r2",
                token_usage=TokenUsage(
                    prompt_tokens=20,
                    completion_tokens=10,
                    total_tokens=30,
                    cache_creation_tokens=4,
                    cache_read_tokens=6,
                ),
            ),
        ]
    )
    backend = _StubBackend(lambda: session)
    tc = TestCaseModel(turns=[Turn(input="a"), Turn(input="b")])
    executor = _make_executor(backend, tc)
    report = await executor.execute_tests()
    usage = report.results[0].token_usage
    assert usage is not None
    assert usage.prompt_tokens == 30
    assert usage.completion_tokens == 15
    assert usage.total_tokens == 45
    assert usage.cache_creation_tokens == 6
    assert usage.cache_read_tokens == 9


@pytest.mark.asyncio
async def test_files_flattened_into_turn_prompt_not_replayed() -> None:
    sent_messages: list[str] = []

    async def send(msg: str):
        sent_messages.append(msg)
        return _mk_exec_result(f"r{len(sent_messages)}")

    session = Mock()
    session.send = AsyncMock(side_effect=send)
    session.close = AsyncMock()
    backend = _StubBackend(lambda: session)

    fi = FileInput(path="tests/fixtures/chart.png", type="image")
    tc = TestCaseModel(
        turns=[Turn(input="look at this", files=[fi]), Turn(input="plain second")]
    )
    # Patch file_processor to return a deterministic processed file.
    from holodeck.models.test_result import ProcessedFileInput

    executor = _make_executor(backend, tc)
    executor.file_processor = Mock(spec=FileProcessor)
    executor.file_processor.process_file = Mock(
        return_value=ProcessedFileInput(
            original=fi,
            markdown_content="<<chart bytes>>",
            metadata=None,
            cached_path=None,
            processing_time_ms=None,
            error=None,
        )
    )

    await executor.execute_tests()

    assert len(sent_messages) == 2
    assert "<<chart bytes>>" in sent_messages[0]
    assert "look at this" in sent_messages[0]
    # Turn 2 must have plain input only — no file residue.
    assert "<<chart bytes>>" not in sent_messages[1]
    assert sent_messages[1].strip() == "plain second"


@pytest.mark.asyncio
async def test_test_case_passed_requires_all_turns_passed() -> None:
    """FR-016: one failing turn flips test-case passed to False."""

    async def send(msg):
        if "fail" in msg:
            raise asyncio.TimeoutError()
        return _mk_exec_result("ok")

    session = Mock()
    session.send = AsyncMock(side_effect=send)
    session.close = AsyncMock()
    backend = _StubBackend(lambda: session)

    tc = TestCaseModel(
        turns=[Turn(input="ok1"), Turn(input="fail-me"), Turn(input="ok2")]
    )
    executor = _make_executor(backend, tc)
    report = await executor.execute_tests()

    assert report.results[0].passed is False
    assert report.summary.passed == 0
    assert report.summary.failed == 1
