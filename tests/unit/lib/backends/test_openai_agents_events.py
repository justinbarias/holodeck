"""Tests for handoff / tool ``ToolEvent`` emission on the OpenAI Agents backend.

Two layers:

* ``TestItemMapping`` drives ``openai_agents_events`` directly with SDK run
  items and stream events and pins the exact event shapes (FR-006).
* ``TestScriptedHandoffRun`` drives the **real** SDK ``Runner`` through
  ``OpenAIAgentsSession`` with a scripted ``Model`` (no network) whose turns
  are: entry agent hands off to ``researcher``; researcher calls the inherited
  ``lookup`` tool; researcher answers. It asserts the ordered events reaching
  the session's ``tool_events`` queue on both the streaming and post-hoc
  paths, and that the ``ExecutionResult`` records the handoff. This is the
  bounded fixture-based handoff acceptance for T3; T10 covers a live handoff.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import MagicMock

import pytest

from holodeck.lib.backends.base import ToolEvent
from holodeck.lib.backends.openai_agents_backend import (
    OpenAIAgentsSession,
    _to_execution_result,
)
from holodeck.lib.backends.openai_agents_events import (
    HandoffTracker,
    tool_events_for_item,
    tool_events_for_run_items,
    tool_events_for_stream_event,
)

# ---------------------------------------------------------------------------
# Item helpers
# ---------------------------------------------------------------------------


def _agent_named(name: str) -> MagicMock:
    agent = MagicMock()
    agent.name = name
    return agent


def _tool_call(name: str, call_id: str, arguments: str = "{}") -> Any:
    from agents.items import ToolCallItem
    from openai.types.responses import ResponseFunctionToolCall

    return ToolCallItem(
        agent=_agent_named("x"),
        raw_item=ResponseFunctionToolCall(
            call_id=call_id, name=name, arguments=arguments, type="function_call"
        ),
    )


def _tool_output(call_id: str, output: str) -> Any:
    from agents.items import ToolCallOutputItem

    return ToolCallOutputItem(
        agent=_agent_named("x"),
        raw_item={"call_id": call_id, "output": output, "type": "function_call_output"},
        output=output,
    )


def _handoff_call(target: str, call_id: str) -> Any:
    from agents.items import HandoffCallItem
    from openai.types.responses import ResponseFunctionToolCall

    return HandoffCallItem(
        agent=_agent_named("entry"),
        raw_item=ResponseFunctionToolCall(
            call_id=call_id,
            name=f"transfer_to_{target}",
            arguments="{}",
            type="function_call",
        ),
    )


def _handoff_output(source: str, target: str, call_id: str) -> Any:
    from agents.items import HandoffOutputItem

    return HandoffOutputItem(
        agent=_agent_named(source),
        raw_item={
            "call_id": call_id,
            "output": f"{{'assistant': '{target}'}}",
            "type": "function_call_output",
        },
        source_agent=_agent_named(source),
        target_agent=_agent_named(target),
    )


def _message(text: str) -> Any:
    from agents.items import MessageOutputItem
    from openai.types.responses import ResponseOutputMessage, ResponseOutputText

    return MessageOutputItem(
        agent=_agent_named("x"),
        raw_item=ResponseOutputMessage(
            id="msg_1",
            role="assistant",
            status="completed",
            type="message",
            content=[ResponseOutputText(type="output_text", text=text, annotations=[])],
        ),
    )


def _reasoning(*summaries: str) -> Any:
    from agents.items import ReasoningItem
    from openai.types.responses.response_reasoning_item import (
        ResponseReasoningItem,
        Summary,
    )

    return ReasoningItem(
        agent=_agent_named("x"),
        raw_item=ResponseReasoningItem(
            id="reason-1",
            type="reasoning",
            summary=[Summary(text=s, type="summary_text") for s in summaries],
        ),
    )


def _kinds(events: list[ToolEvent]) -> list[tuple[str, str]]:
    return [(e.kind, e.tool_name) for e in events]


# ---------------------------------------------------------------------------
# Pure mapping
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestItemMapping:
    def test_tool_call_and_output_outside_handoff(self) -> None:
        tracker = HandoffTracker()
        start = tool_events_for_item(
            _tool_call("add", "call-1", '{"a": 1}'), tracker, announce_handoff=True
        )
        end = tool_events_for_item(
            _tool_output("call-1", "3"), tracker, announce_handoff=True
        )
        assert _kinds(start) == [("start", "add")]
        assert start[0].tool_use_id == "call-1"
        assert start[0].tool_input == {"a": 1}
        assert _kinds(end) == [("end", "add")]
        assert end[0].tool_response == "3"
        assert end[0].tool_use_id == "call-1"

    def test_handoff_sequence_post_hoc(self) -> None:
        items = [
            _handoff_call("researcher", "call-h"),
            _handoff_output("entry", "researcher", "call-h"),
            _tool_call("lookup", "call-t"),
            _tool_output("call-t", "42"),
            _message("Researched: 42"),
        ]
        events = tool_events_for_run_items(items)
        assert _kinds(events) == [
            ("start", "transfer_to_researcher"),
            ("subagent_message", "transfer_to_researcher"),
            ("start", "lookup"),
            ("parent_link", "lookup"),
            ("end", "lookup"),
            ("subagent_message", "transfer_to_researcher"),
            ("end", "transfer_to_researcher"),
        ]
        announce = events[1]
        assert announce.text == "Agent 'researcher' is now active."
        assert announce.parent_tool_use_id == "call-h"
        assert announce.tool_use_id == "call-h"
        link = events[3]
        assert link.tool_use_id == "call-t"
        assert link.parent_tool_use_id == "call-h"
        snapshot = events[5]
        assert snapshot.text == "Researched: 42"
        assert snapshot.parent_tool_use_id == "call-h"
        # The handoff closes only at run end, after the subagent's activity,
        # so the chat panel keeps nesting children under it.
        handoff_end = events[6]
        assert handoff_end.tool_use_id == "call-h"
        assert handoff_end.tool_response == "Handoff to 'researcher' completed."

    def test_message_before_any_handoff_emits_nothing(self) -> None:
        assert tool_events_for_run_items([_message("hello")]) == []

    def test_reasoning_maps_to_thinking(self) -> None:
        events = tool_events_for_run_items([_reasoning("first", "second")])
        assert _kinds(events) == [("thinking", "")]
        assert events[0].text == "first\n\nsecond"
        assert events[0].tool_use_id == "reason-1"

    def test_reasoning_without_summary_is_silent(self) -> None:
        assert tool_events_for_run_items([_reasoning()]) == []

    def test_streaming_announces_from_agent_updated_event(self) -> None:
        from agents.stream_events import AgentUpdatedStreamEvent, RunItemStreamEvent

        tracker = HandoffTracker()
        # SDK announces the entry agent first; nothing to nest under yet.
        assert (
            tool_events_for_stream_event(
                AgentUpdatedStreamEvent(new_agent=_agent_named("entry")), tracker
            )
            == []
        )
        requested = tool_events_for_stream_event(
            RunItemStreamEvent(
                item=_handoff_call("researcher", "call-h"), name="handoff_requested"
            ),
            tracker,
        )
        occurred = tool_events_for_stream_event(
            RunItemStreamEvent(
                item=_handoff_output("entry", "researcher", "call-h"),
                name="handoff_occured",
            ),
            tracker,
        )
        updated = tool_events_for_stream_event(
            AgentUpdatedStreamEvent(new_agent=_agent_named("researcher")), tracker
        )
        assert _kinds(requested) == [("start", "transfer_to_researcher")]
        # No announcement (and no premature end) from the output item on the
        # stream path; the agent-updated event announces instead.
        assert occurred == []
        assert _kinds(updated) == [("subagent_message", "transfer_to_researcher")]
        assert updated[0].text == "Agent 'researcher' is now active."
        assert _kinds(tracker.close()) == [("end", "transfer_to_researcher")]

    def test_close_with_error_fails_open_calls_and_handoffs(self) -> None:
        tracker = HandoffTracker()
        tool_events_for_item(
            _handoff_call("r", "call-h"), tracker, announce_handoff=False
        )
        tool_events_for_item(
            _handoff_output("e", "r", "call-h"), tracker, announce_handoff=False
        )
        tool_events_for_item(
            _tool_call("lookup", "call-t"), tracker, announce_handoff=False
        )
        closing = tracker.close(error="RuntimeError: boom")
        assert _kinds(closing) == [("error", "lookup"), ("error", "transfer_to_r")]
        assert all(e.error == "RuntimeError: boom" for e in closing)
        assert tracker.open_tool_calls == [] and tracker.open_handoffs == {}

    def test_nested_handoff_links_to_previous_handoff(self) -> None:
        items = [
            _handoff_call("a", "call-1"),
            _handoff_output("entry", "a", "call-1"),
            _handoff_call("b", "call-2"),
            _handoff_output("a", "b", "call-2"),
            _message("from b"),
        ]
        events = tool_events_for_run_items(items)
        assert _kinds(events) == [
            ("start", "transfer_to_a"),
            ("subagent_message", "transfer_to_a"),
            ("start", "transfer_to_b"),
            ("parent_link", "transfer_to_b"),
            ("subagent_message", "transfer_to_b"),
            ("subagent_message", "transfer_to_b"),
            ("end", "transfer_to_a"),
            ("end", "transfer_to_b"),
        ]
        assert events[3].parent_tool_use_id == "call-1"
        assert events[5].parent_tool_use_id == "call-2"

    def test_raw_response_events_are_ignored(self) -> None:
        raw = MagicMock()
        raw.type = "raw_response_event"
        assert tool_events_for_stream_event(raw, HandoffTracker()) == []

    def test_execution_result_records_handoff(self) -> None:
        result = MagicMock()
        result.final_output = "done"
        result.new_items = [
            _handoff_call("researcher", "call-h"),
            _handoff_output("entry", "researcher", "call-h"),
        ]
        result.raw_responses = [MagicMock()]
        result.context_wrapper = MagicMock(usage=None)
        execution = _to_execution_result(result)
        assert execution.tool_calls == [
            {"name": "transfer_to_researcher", "arguments": {}, "call_id": "call-h"}
        ]
        assert execution.tool_results == [
            {
                "name": "transfer_to_researcher",
                "result": "handoff:researcher",
                "call_id": "call-h",
            }
        ]


# ---------------------------------------------------------------------------
# Scripted Runner
# ---------------------------------------------------------------------------


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
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    }


def _function_call(name: str, call_id: str) -> dict[str, Any]:
    return {
        "type": "function_call",
        "id": f"fc_{call_id}",
        "call_id": call_id,
        "name": name,
        "arguments": "{}",
        "status": "completed",
    }


def _text_message(text: str) -> dict[str, Any]:
    return {
        "type": "message",
        "id": "msg_1",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def _scripted_model(turns: list[list[dict[str, Any]]]) -> Any:
    """A ``Model`` that replays *turns* (Responses ``output`` lists) in order."""
    from agents.items import ModelResponse
    from agents.models.interface import Model
    from agents.usage import Usage
    from openai.types.responses import Response, ResponseCompletedEvent

    class _Scripted(Model):
        def __init__(self) -> None:
            self.calls = 0

        def _next(self) -> Response:
            output = turns[self.calls]
            self.calls += 1
            return Response.model_validate(_response_payload(output))

        async def get_response(self, *_a: Any, **_k: Any) -> ModelResponse:
            response = self._next()
            return ModelResponse(
                output=list(response.output), usage=Usage(), response_id=None
            )

        async def stream_response(self, *_a: Any, **_k: Any) -> AsyncIterator[Any]:
            yield ResponseCompletedEvent(
                type="response.completed", response=self._next(), sequence_number=0
            )

    return _Scripted()


def _handoff_agent_pair(model: Any) -> Any:
    """Entry agent with an inherited ``lookup`` tool and a ``researcher`` handoff."""
    from agents import Agent as SDKAgent
    from agents import FunctionTool as SDKFunctionTool

    async def _lookup(_ctx: Any, _input: str) -> str:
        return "42"

    lookup = SDKFunctionTool(
        name="lookup",
        description="lookup",
        params_json_schema={"type": "object", "properties": {}},
        on_invoke_tool=_lookup,
        strict_json_schema=False,
    )
    researcher = SDKAgent(
        name="researcher",
        instructions="research",
        handoff_description="Finds facts",
        tools=[lookup],
        model=model,
    )
    return SDKAgent(
        name="entry",
        instructions="route",
        tools=[lookup],
        handoffs=[researcher],
        model=model,
    )


SCRIPT: list[list[dict[str, Any]]] = [
    [_function_call("transfer_to_researcher", "call-h")],
    [_function_call("lookup", "call-t")],
    [_text_message("Researched: 42")],
]

EXPECTED_ORDER = [
    ("start", "transfer_to_researcher"),
    ("subagent_message", "transfer_to_researcher"),
    ("start", "lookup"),
    ("parent_link", "lookup"),
    ("end", "lookup"),
    ("subagent_message", "transfer_to_researcher"),
    ("end", "transfer_to_researcher"),
]


def _drain(queue: asyncio.Queue[ToolEvent]) -> list[ToolEvent]:
    events: list[ToolEvent] = []
    while not queue.empty():
        events.append(queue.get_nowait())
    return events


@pytest.mark.unit
class TestScriptedHandoffRun:
    @pytest.mark.asyncio
    async def test_streaming_emits_ordered_handoff_events(self) -> None:
        model = _scripted_model(SCRIPT)
        session = OpenAIAgentsSession(_handoff_agent_pair(model), None)
        chunks = [chunk async for chunk in session.send_streaming("go")]
        events = _drain(session.tool_events)
        assert _kinds(events) == EXPECTED_ORDER
        assert model.calls == 3
        # The scripted stream carries no text deltas; only events are asserted.
        assert chunks == []
        assert events[1].text == "Agent 'researcher' is now active."
        assert events[5].text == "Researched: 42"
        assert events[3].parent_tool_use_id == events[0].tool_use_id == "call-h"
        assert events[6].tool_response == "Handoff to 'researcher' completed."

    @pytest.mark.asyncio
    async def test_stream_failure_closes_open_entries_with_error(self) -> None:
        """A run that raises after a tool call started leaves no entry running."""
        from agents.models.interface import Model
        from openai.types.responses import Response, ResponseCompletedEvent

        class _FailsAfterCall(Model):
            def __init__(self) -> None:
                self.calls = 0

            async def get_response(self, *_a: Any, **_k: Any) -> Any:
                raise AssertionError("non-streaming path not exercised")

            async def stream_response(self, *_a: Any, **_k: Any) -> AsyncIterator[Any]:
                self.calls += 1
                if self.calls == 1:
                    yield ResponseCompletedEvent(
                        type="response.completed",
                        response=Response.model_validate(
                            _response_payload(
                                [_function_call("transfer_to_researcher", "call-h")]
                            )
                        ),
                        sequence_number=0,
                    )
                    return
                raise RuntimeError("upstream exploded")

        session = OpenAIAgentsSession(_handoff_agent_pair(_FailsAfterCall()), None)
        with pytest.raises(RuntimeError, match="upstream exploded"):
            async for _ in session.send_streaming("go"):
                pass
        events = _drain(session.tool_events)
        assert _kinds(events) == [
            ("start", "transfer_to_researcher"),
            ("subagent_message", "transfer_to_researcher"),
            ("error", "transfer_to_researcher"),
        ]
        assert "upstream exploded" in (events[-1].error or "")

    @pytest.mark.asyncio
    async def test_send_emits_same_events_post_hoc(self) -> None:
        model = _scripted_model(SCRIPT)
        session = OpenAIAgentsSession(_handoff_agent_pair(model), None)
        result = await session.send("go")
        events = _drain(session.tool_events)
        assert result.is_error is False
        assert result.response == "Researched: 42"
        assert _kinds(events) == EXPECTED_ORDER
        assert [c["name"] for c in result.tool_calls] == [
            "transfer_to_researcher",
            "lookup",
        ]
        assert result.tool_results[0]["result"] == "handoff:researcher"
        assert result.tool_results[1]["result"] == "42"
