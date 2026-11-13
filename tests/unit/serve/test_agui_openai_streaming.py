"""Regression: AG-UI streams OpenAI Agents runs live, not in one end-of-run burst.

Drives ``AGUIProtocol.handle_request`` through a real ``AgentExecutor`` (with
the ``_TaskBoundSession`` actor that ``holodeck serve`` uses), a real
``OpenAIAgentsSession`` and the real SDK ``Runner``, backed by a scripted
streaming ``Model`` (no network). The ``lookup`` tool blocks on a gate, so the
test proves ``TOOL_CALL_START`` reaches the client while the run is still in
flight, and that text arrives as several ``TEXT_MESSAGE_CONTENT`` events in run
order around the tool card.
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
from holodeck.lib.backends.base import AgentSession, ExecutionResult
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


def _scripted_streaming_model() -> Any:
    """A ``Model`` that streams text deltas, then completes each scripted turn."""
    from agents.models.interface import Model
    from openai.types.responses import (
        Response,
        ResponseCompletedEvent,
        ResponseTextDeltaEvent,
    )

    class _Scripted(Model):
        def __init__(self) -> None:
            self.calls = 0

        async def get_response(self, *_a: Any, **_k: Any) -> Any:
            raise AssertionError("AG-UI must use the streaming path")

        async def stream_response(self, *_a: Any, **_k: Any) -> AsyncIterator[Any]:
            deltas, output = _TURNS[self.calls]
            self.calls += 1
            for seq, delta in enumerate(deltas):
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


def _sdk_agent(gate: asyncio.Event, tool_entered: asyncio.Event) -> Any:
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
        model=_scripted_streaming_model(),
    )


class _ScriptedBackend:
    """``AgentBackend`` returning a real session over the scripted SDK agent."""

    def __init__(self, sdk_agent: Any) -> None:
        self._sdk_agent = sdk_agent

    async def initialize(self) -> None:
        return None

    async def invoke_once(
        self, message: str, context: list[dict[str, Any]] | None = None
    ) -> ExecutionResult:
        raise AssertionError("not used")

    async def create_session(self, *, eager_connect: bool = True) -> AgentSession:
        return OpenAIAgentsSession(self._sdk_agent, None)

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
    executor = AgentExecutor(
        _agent_config(),
        backend=_ScriptedBackend(_sdk_agent(gate, tool_entered)),
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

        # Phase 2: release the tool and read the rest of the run.
        gate.set()
        after: list[dict[str, Any]] = []
        while not after or after[-1]["type"] not in ("RUN_FINISHED", "RUN_ERROR"):
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

    assert executor.get_history() == [
        {"role": "user", "content": "What is it?"},
        {"role": "assistant", "content": "Let me check.The answer is 42."},
    ]
