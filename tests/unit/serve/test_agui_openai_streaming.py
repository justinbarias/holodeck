"""Regression: AG-UI streams OpenAI Agents runs live, not in one end-of-run burst.

Drives ``AGUIProtocol.handle_request`` through a real ``AgentExecutor`` (with
the ``_TaskBoundSession`` actor that ``holodeck serve`` uses), a real
``OpenAIAgentsSession`` and the real SDK ``Runner``, backed by a scripted
streaming ``Model`` (no network). The ``lookup`` tool blocks on a gate, so the
test proves ``TOOL_CALL_START`` reaches the client while the run is still in
flight, and that text arrives as several ``TEXT_MESSAGE_CONTENT`` events in run
order around the tool card. Further tests pin parity with the non-streaming
path (final answer, usage, persisted session) and the close-out on failure.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator, AsyncIterator
from types import SimpleNamespace
from typing import Any

import pytest
from ag_ui.core import RunAgentInput, UserMessage

from holodeck.chat.executor import AgentExecutor
from holodeck.lib.backends.base import AgentSession, ExecutionResult, ToolEvent
from holodeck.lib.backends.openai_agents_backend import OpenAIAgentsSession
from holodeck.models.agent import Agent, Instructions
from holodeck.models.llm import LLMProvider, ProviderEnum
from holodeck.serve.protocols.agui import AGUIProtocol

_STEP_TIMEOUT = 5.0


def _response_payload(output: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "id": "resp_1",
        "created_at": 1.0,
        "model": "scripted",
        "object": "response",
        "output": output,
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
        "usage": {
            "input_tokens": 1,
            "output_tokens": 1,
            "total_tokens": 2,
            "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    }


def _text_message(text: str) -> dict[str, Any]:
    return {
        "type": "message",
        "id": "msg_1",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def _function_call(name: str, call_id: str) -> dict[str, Any]:
    return {
        "type": "function_call",
        "id": f"fc_{call_id}",
        "call_id": call_id,
        "name": name,
        "arguments": '{"q": "answer"}',
        "status": "completed",
    }


# Each turn: (text deltas streamed first, completed ``output`` items).
_TURNS: list[tuple[list[str], list[dict[str, Any]]]] = [
    (
        ["Let me check."],
        [_text_message("Let me check."), _function_call("lookup", "c1")],
    ),
    (["The answer ", "is 42."], [_text_message("The answer is 42.")]),
]


def _scripted_streaming_model(
    delta_gate: asyncio.Event | None = None, *, allow_non_streaming: bool = False
) -> Any:
    """A ``Model`` that streams text deltas, then completes each scripted turn.

    With *delta_gate*, the stream pauses after the first delta of a turn
    with several deltas until the gate is set. With *allow_non_streaming*,
    ``get_response`` returns the same scripted turns.
    """
    from agents.items import ModelResponse
    from agents.models.interface import Model
    from agents.usage import Usage
    from openai.types.responses import (
        Response,
        ResponseCompletedEvent,
        ResponseTextDeltaEvent,
    )

    class _Scripted(Model):
        def __init__(self) -> None:
            self.calls = 0

        async def get_response(self, *_a: Any, **_k: Any) -> Any:
            if not allow_non_streaming:
                raise AssertionError("AG-UI must use the streaming path")
            _deltas, output = _TURNS[self.calls]
            self.calls += 1
            response = Response.model_validate(_response_payload(output))
            return ModelResponse(
                output=response.output,
                usage=Usage(
                    requests=1, input_tokens=1, output_tokens=1, total_tokens=2
                ),
                response_id=response.id,
            )

        async def stream_response(self, *_a: Any, **_k: Any) -> AsyncIterator[Any]:
            deltas, output = _TURNS[self.calls]
            self.calls += 1
            for seq, delta in enumerate(deltas):
                if seq == 1 and delta_gate is not None:
                    await delta_gate.wait()
                yield ResponseTextDeltaEvent(
                    type="response.output_text.delta",
                    content_index=0,
                    delta=delta,
                    item_id="msg_1",
                    logprobs=[],
                    output_index=0,
                    sequence_number=seq,
                )
                await asyncio.sleep(0)
            yield ResponseCompletedEvent(
                type="response.completed",
                response=Response.model_validate(_response_payload(output)),
                sequence_number=len(deltas),
            )

    return _Scripted()


def _sdk_agent(
    gate: asyncio.Event, tool_entered: asyncio.Event, model: Any | None = None
) -> Any:
    from agents import Agent as SDKAgent
    from agents import FunctionTool as SDKFunctionTool

    async def _lookup(_ctx: Any, _input: str) -> str:
        tool_entered.set()
        await gate.wait()
        return "42"

    lookup = SDKFunctionTool(
        name="lookup",
        description="lookup",
        params_json_schema={"type": "object", "properties": {}},
        on_invoke_tool=_lookup,
        strict_json_schema=False,
    )
    return SDKAgent(
        name="entry",
        instructions="answer",
        tools=[lookup],
        model=model if model is not None else _scripted_streaming_model(),
    )


class _ScriptedBackend:
    """``AgentBackend`` returning a real session over the scripted SDK agent."""

    def __init__(self, sdk_agent: Any, sqlite_session: Any = None) -> None:
        self._sdk_agent = sdk_agent
        self._sqlite_session = sqlite_session

    async def initialize(self) -> None:
        return None

    async def invoke_once(
        self, message: str, context: list[dict[str, Any]] | None = None
    ) -> ExecutionResult:
        raise AssertionError("not used")

    async def create_session(self, *, eager_connect: bool = True) -> AgentSession:
        return OpenAIAgentsSession(self._sdk_agent, self._sqlite_session)

    async def teardown(self) -> None:
        return None


def _agent_config() -> Agent:
    return Agent(
        name="streaming-test",
        model=LLMProvider(provider=ProviderEnum.OPENAI, name="gpt-4o-mini"),
        instructions=Instructions(inline="answer"),
    )


def _decode(chunk: bytes) -> dict[str, Any]:
    text = chunk.decode()
    payload = next(
        line[len("data: ") :] for line in text.splitlines() if line.startswith("data: ")
    )
    decoded: dict[str, Any] = json.loads(payload)
    return decoded


async def _next_event(stream: AsyncGenerator[bytes, None]) -> dict[str, Any]:
    return _decode(await asyncio.wait_for(stream.__anext__(), _STEP_TIMEOUT))


@pytest.mark.unit
@pytest.mark.asyncio
async def test_openai_agui_streams_tool_events_and_text_live() -> None:
    gate = asyncio.Event()
    tool_entered = asyncio.Event()
    delta_gate = asyncio.Event()
    model = _scripted_streaming_model(delta_gate)
    executor = AgentExecutor(
        _agent_config(),
        backend=_ScriptedBackend(_sdk_agent(gate, tool_entered, model)),
        release_transport_after_turn=True,
    )
    server_session = SimpleNamespace(agent_executor=executor, session_id="s1")
    request = RunAgentInput(
        thread_id="thread-1",
        run_id="run-1",
        messages=[UserMessage(id="u1", role="user", content="What is it?")],
        state=None,
        tools=[],
        context=[],
        forwarded_props=None,
    )
    stream = AGUIProtocol().handle_request(request, server_session)  # type: ignore[arg-type]

    try:
        # Phase 1: everything up to the tool call arrives while the tool is
        # still blocked, i.e. before the run can complete.
        before: list[dict[str, Any]] = []
        while not before or before[-1]["type"] != "TOOL_CALL_ARGS":
            before.append(await _next_event(stream))
        await asyncio.wait_for(tool_entered.wait(), _STEP_TIMEOUT)
        assert not gate.is_set()

        # Phase 2: release the tool; the final answer's first chunk arrives
        # while the model is still holding back the second one.
        gate.set()
        after: list[dict[str, Any]] = []
        while not after or after[-1]["type"] != "TEXT_MESSAGE_CONTENT":
            after.append(await _next_event(stream))
        assert after[-1]["delta"] == "The answer "
        assert not delta_gate.is_set()

        # Phase 3: release the rest of the final answer.
        delta_gate.set()
        while after[-1]["type"] not in ("RUN_FINISHED", "RUN_ERROR"):
            after.append(await _next_event(stream))
    finally:
        await stream.aclose()
        await executor.shutdown()

    assert [e["type"] for e in before] == [
        "RUN_STARTED",
        "TEXT_MESSAGE_START",
        "TEXT_MESSAGE_CONTENT",
        "TEXT_MESSAGE_END",
        "TEXT_MESSAGE_START",
        "TOOL_CALL_START",
        "TOOL_CALL_ARGS",
    ]
    assert [e["type"] for e in after] == [
        "TOOL_CALL_END",
        "TOOL_CALL_RESULT",
        "TEXT_MESSAGE_END",
        "TEXT_MESSAGE_START",
        "TEXT_MESSAGE_CONTENT",
        "TEXT_MESSAGE_CONTENT",
        "TEXT_MESSAGE_END",
        "RUN_FINISHED",
    ]

    events = before + after
    contents = [e["delta"] for e in events if e["type"] == "TEXT_MESSAGE_CONTENT"]
    assert contents == ["Let me check.", "The answer ", "is 42."]

    tool_start = next(e for e in events if e["type"] == "TOOL_CALL_START")
    assert tool_start["toolCallName"] == "lookup"
    assert tool_start["toolCallId"] == "c1"
    result = next(e for e in events if e["type"] == "TOOL_CALL_RESULT")
    assert result["content"] == "42"

    # Text before and after the tool card are separate assistant messages,
    # and each text message is opened and closed exactly once.
    text_ids = [e["messageId"] for e in events if e["type"] == "TEXT_MESSAGE_CONTENT"]
    assert text_ids[0] != text_ids[1] == text_ids[2]
    starts = [e["messageId"] for e in events if e["type"] == "TEXT_MESSAGE_START"]
    ends = [e["messageId"] for e in events if e["type"] == "TEXT_MESSAGE_END"]
    assert sorted(starts) == sorted(ends)
    assert len(set(starts)) == len(starts)

    # History records the run's final output, as the non-streaming path does.
    assert executor.get_history() == [
        {"role": "user", "content": "What is it?"},
        {"role": "assistant", "content": "The answer is 42."},
    ]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_streamed_turn_matches_non_streaming_result() -> None:
    """Final answer, usage, tool calls and persisted session match ``send``."""
    from agents import SQLiteSession

    async def _run(streamed: bool) -> tuple[Any, list[Any], list[Any]]:
        gate = asyncio.Event()
        gate.set()
        sqlite = SQLiteSession("parity")
        model = _scripted_streaming_model(allow_non_streaming=not streamed)
        executor = AgentExecutor(
            _agent_config(),
            backend=_ScriptedBackend(_sdk_agent(gate, asyncio.Event(), model), sqlite),
            release_transport_after_turn=True,
        )
        try:
            if streamed:
                items = [i async for i in executor.execute_turn_events("What is it?")]
                response = items[-1]
            else:
                response = await executor.execute_turn("What is it?")
            stored = await sqlite.get_items()
        finally:
            await executor.shutdown()
        return response, executor.get_history(), stored

    plain, plain_history, plain_items = await _run(streamed=False)
    live, live_history, live_items = await _run(streamed=True)

    assert live.content == plain.content == "The answer is 42."
    assert live.tokens_used == plain.tokens_used
    assert live.tokens_used.total_tokens == 4
    assert [t.tool_name for t in live.tool_executions] == ["lookup"]
    assert [t.tool_name for t in plain.tool_executions] == ["lookup"]
    assert live_history == plain_history
    assert live_items == plain_items
    assert len(live_items) == 5  # user, text, call, output, final text


class _FailingEventSession:
    """Event session that streams text, opens a tool call, then fails."""

    def __init__(self, *, open_tool: bool, block: bool) -> None:
        self._open_tool = open_tool
        self._block = block
        self.tool_events = None

    async def prepare(self) -> None:
        return None

    async def send(self, message: str) -> ExecutionResult:
        raise AssertionError("send must not be used")

    async def send_streaming(self, message: str) -> AsyncGenerator[str, None]:
        raise AssertionError("send_streaming must not be used")
        yield ""  # pragma: no cover

    async def send_events(self, message: str) -> AsyncGenerator[Any, None]:
        yield "partial answer"
        if self._open_tool:
            yield ToolEvent(kind="start", tool_name="lookup", tool_use_id="c9")
        if self._block:
            await asyncio.Event().wait()
        raise RuntimeError("upstream exploded")

    async def close(self) -> None:
        return None


class _FailingBackend(_ScriptedBackend):
    def __init__(self, session: _FailingEventSession) -> None:
        super().__init__(None)
        self._session = session

    async def create_session(self, *, eager_connect: bool = True) -> AgentSession:
        return self._session  # type: ignore[return-value]


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("open_tool", "block", "llm_timeout"),
    [(False, False, None), (True, False, None), (True, True, 0.2)],
    ids=["text-then-error", "tool-open-then-error", "tool-open-then-timeout"],
)
async def test_failure_closes_open_messages_before_run_error(
    open_tool: bool, block: bool, llm_timeout: float | None
) -> None:
    executor = AgentExecutor(
        _agent_config(),
        backend=_FailingBackend(_FailingEventSession(open_tool=open_tool, block=block)),
        release_transport_after_turn=True,
        llm_timeout=llm_timeout,
    )
    request = RunAgentInput(
        thread_id="thread-1",
        run_id="run-1",
        messages=[UserMessage(id="u1", role="user", content="What is it?")],
        state=None,
        tools=[],
        context=[],
        forwarded_props=None,
    )
    server_session = SimpleNamespace(agent_executor=executor, session_id="s1")
    stream = AGUIProtocol().handle_request(request, server_session)  # type: ignore[arg-type]
    try:
        events: list[dict[str, Any]] = []
        while not events or events[-1]["type"] not in ("RUN_FINISHED", "RUN_ERROR"):
            events.append(await _next_event(stream))
    finally:
        await stream.aclose()
        # Bounded: a turn left running after the timeout would block shutdown.
        await asyncio.wait_for(executor.shutdown(), _STEP_TIMEOUT)

    tool_events = [
        "TEXT_MESSAGE_START",
        "TOOL_CALL_START",
        "TOOL_CALL_ARGS",
        "TOOL_CALL_END",
        "TOOL_CALL_RESULT",
        "TEXT_MESSAGE_END",
    ]
    assert [e["type"] for e in events] == [
        "RUN_STARTED",
        "TEXT_MESSAGE_START",
        "TEXT_MESSAGE_CONTENT",
        "TEXT_MESSAGE_END",
        *(tool_events if open_tool else []),
        "RUN_ERROR",
    ]
    starts = [e["messageId"] for e in events if e["type"] == "TEXT_MESSAGE_START"]
    ends = [e["messageId"] for e in events if e["type"] == "TEXT_MESSAGE_END"]
    assert sorted(starts) == sorted(ends)
