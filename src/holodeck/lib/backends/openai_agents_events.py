"""``ToolEvent`` mapping for the OpenAI Agents backend (FR-006).

Translates SDK run items and stream events into the provider-agnostic
:class:`~holodeck.lib.backends.base.ToolEvent` records the chat tools panel
and the AG-UI bridge already consume for the Claude backend, so handoffs and
tool calls render identically across backends.

Mapping (one HoloDeck event list per SDK item, in SDK order):

* ``ToolCallItem`` → ``start`` (``tool_name``, ``tool_use_id=call_id``,
  ``tool_input``). While a handoff is active, a ``parent_link`` follows so the
  panel nests the call under the handoff.
* Hosted (server-side) call items — web search, file search, code
  interpreter, image generation, hosted MCP — never get an output item, so
  they emit ``start`` and ``end`` together (see :func:`hosted_call_for`).
* ``ToolCallOutputItem`` → ``end`` with ``tool_response``. A local tool that
  raised inside the SDK loop surfaces here with the SDK's error text as the
  output (the SDK's default failure handler converts the exception to a
  model-visible string), so the panel shows it as a completed call carrying
  the error message rather than an ``error`` event.
* ``HandoffCallItem`` (``transfer_to_<agent>``) → ``start`` (+ ``parent_link``
  when nested under an earlier handoff).
* ``HandoffOutputItem`` → the handoff becomes the *active parent* for later
  events. No ``end`` yet: the target agent now owns the conversation, so the
  handoff entry stays active (like a Claude ``Task``) until the run ends and
  :meth:`HandoffTracker.close` emits its ``end``. That keeps the chat panel
  nesting the subagent's tool calls and text under the handoff.
* ``AgentUpdatedStreamEvent`` (streaming only) → ``subagent_message`` under
  the active handoff announcing the newly active agent. In the post-hoc path
  (non-streaming ``send``) the same announcement is emitted from the
  ``HandoffOutputItem`` instead, since no agent-updated event exists there.
* ``MessageOutputItem`` produced while a handoff is active →
  ``subagent_message`` carrying the subagent's text snapshot.
* ``ReasoningItem`` with summary text → ``thinking``.
* Run end → ``end`` for every still-open handoff (:meth:`HandoffTracker.close`);
  a run that fails mid-stream closes every open tool call and handoff with an
  ``error`` event instead, so nothing is left "running" in the panel.

Every ``import agents`` happens inside functions (SC-005).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from holodeck.lib.backends.base import ToolEvent


@dataclass
class HandoffTracker:
    """Per-run state linking events to the active handoff.

    Attributes:
        active_handoff_id: ``call_id`` of the most recent completed handoff,
            used as ``parent_tool_use_id`` for nested events; ``None`` while
            the entry agent is running.
        active_handoff_name: The ``transfer_to_*`` tool name of that handoff.
        names_by_call_id: ``call_id`` → tool name, so ``end`` events can name
            the tool their SDK output item omits.
        open_tool_calls: ``call_id``s of tool calls that have started but not
            produced an output item yet (in start order).
        open_handoffs: ``call_id`` → target agent name for handoffs that have
            occurred and not been closed (in occurrence order).
    """

    active_handoff_id: str | None = None
    active_handoff_name: str = ""
    names_by_call_id: dict[str, str] = field(default_factory=dict)
    open_tool_calls: list[str] = field(default_factory=list)
    open_handoffs: dict[str, str] = field(default_factory=dict)

    def remember(self, call_id: str, name: str) -> None:
        """Record the tool name for *call_id* and mark the call open."""
        self.names_by_call_id[call_id] = name
        if call_id not in self.open_tool_calls:
            self.open_tool_calls.append(call_id)

    def name_for(self, call_id: str) -> str:
        """Return the recorded tool name for *call_id* (empty if unknown)."""
        return self.names_by_call_id.get(call_id, "")

    def settle(self, call_id: str) -> None:
        """Mark a tool call as having produced its output."""
        if call_id in self.open_tool_calls:
            self.open_tool_calls.remove(call_id)

    def close(self, *, error: str | None = None) -> list[ToolEvent]:
        """Close every open tool call and handoff at run end.

        Args:
            error: When set, the run failed: every open entry gets an
                ``error`` event carrying this text. When ``None`` open tool
                calls (which should not exist after a clean run) are ended
                with an empty response and open handoffs are ended with a
                completion note.

        Returns:
            The closing events, tool calls first, then handoffs in occurrence
            order.
        """
        events: list[ToolEvent] = []
        for call_id in list(self.open_tool_calls):
            name = self.name_for(call_id)
            if error is not None:
                events.append(
                    ToolEvent(
                        kind="error", tool_name=name, tool_use_id=call_id, error=error
                    )
                )
            else:
                events.append(
                    ToolEvent(
                        kind="end",
                        tool_name=name,
                        tool_use_id=call_id,
                        tool_response="",
                    )
                )
        self.open_tool_calls.clear()
        for call_id, target in list(self.open_handoffs.items()):
            name = self.name_for(call_id)
            if error is not None:
                events.append(
                    ToolEvent(
                        kind="error", tool_name=name, tool_use_id=call_id, error=error
                    )
                )
            else:
                events.append(
                    ToolEvent(
                        kind="end",
                        tool_name=name,
                        tool_use_id=call_id,
                        tool_response=f"Handoff to '{target}' completed.",
                    )
                )
        self.open_handoffs.clear()
        self.active_handoff_id = None
        self.active_handoff_name = ""
        return events


def _arguments(raw: Any) -> dict[str, Any]:
    """Coerce a raw ``arguments`` value into a dict (JSON string or dict)."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {"raw": raw}
        return parsed if isinstance(parsed, dict) else {"raw": raw}
    return {}


_HOSTED_CALL_NAMES: dict[str, str] = {
    "web_search_call": "web_search",
    "file_search_call": "file_search",
    "code_interpreter_call": "code_interpreter",
    "image_generation_call": "image_generation",
    "mcp_call": "hosted_mcp",
}
"""Responses output-item ``type`` → tool name for server-side (hosted) calls."""


@dataclass
class HostedCall:
    """A hosted (server-side) tool call reconstructed from its output item.

    Hosted tools execute on the OpenAI platform, so the SDK never produces a
    ``ToolCallOutputItem`` for them: the call item already carries the
    outcome. ``response`` summarises that outcome for the ``end`` event and
    the ``ExecutionResult`` tool results.
    """

    name: str
    call_id: str
    arguments: dict[str, Any]
    response: str


def _field(raw: Any, key: str) -> Any:
    """Read *key* from a dict or attribute-style raw item."""
    if isinstance(raw, dict):
        return raw.get(key)
    return getattr(raw, key, None)


def _dump(value: Any) -> Any:
    """JSON-friendly view of a nested SDK value (pydantic model, list, scalar)."""
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=True)
    if isinstance(value, list):
        return [_dump(v) for v in value]
    return value


def hosted_call_for(raw: Any) -> HostedCall | None:
    """Return the :class:`HostedCall` view of a hosted tool-call item.

    Args:
        raw: A ``ToolCallItem.raw_item`` (SDK model or dict).

    Returns:
        The reconstructed call, or ``None`` when *raw* is an ordinary
        function call (or any item type that is not a hosted call).
    """
    item_type = str(_field(raw, "type") or "")
    name = _HOSTED_CALL_NAMES.get(item_type)
    if name is None:
        return None
    call_id = _call_id(raw)
    status = str(_field(raw, "status") or "")
    arguments: dict[str, Any] = {}
    response = status
    if item_type == "web_search_call":
        action = _dump(_field(raw, "action"))
        if action:
            arguments = {"action": action}
    elif item_type == "file_search_call":
        queries = _field(raw, "queries") or []
        arguments = {"queries": list(queries)}
        results = _field(raw, "results")
        if results:
            response = json.dumps(_dump(results))
    elif item_type == "code_interpreter_call":
        code = _field(raw, "code")
        if code:
            arguments = {"code": code}
        outputs = _field(raw, "outputs")
        if outputs:
            response = json.dumps(_dump(outputs))
    elif item_type == "image_generation_call":
        if _field(raw, "result"):
            response = f"{status or 'completed'} (image data omitted)"
    elif item_type == "mcp_call":
        # A remote MCP call names the remote tool; keep the server label so
        # graders can tell hosted MCP calls from local MCP tools.
        name = str(_field(raw, "name") or name)
        arguments = _arguments(_field(raw, "arguments"))
        server = _field(raw, "server_label")
        if server:
            # The SDK's ``server_label`` is authoritative; it must win over a
            # same-named key inside the remote tool's arguments.
            arguments = {**arguments, "server_label": server}
        error = _field(raw, "error")
        output = _field(raw, "output")
        response = str(error) if error else str(output or status)
    return HostedCall(
        name=name, call_id=call_id, arguments=arguments, response=response
    )


def _call_id(raw: Any) -> str:
    """Return the ``call_id`` (or ``id``) of a raw call / output item."""
    if isinstance(raw, dict):
        return str(raw.get("call_id") or raw.get("id") or "")
    return str(getattr(raw, "call_id", None) or getattr(raw, "id", None) or "")


def _agent_name(agent: Any) -> str:
    """Return an SDK agent's ``name`` as a string (empty when unavailable)."""
    return str(getattr(agent, "name", "") or "")


def _start_events(
    tracker: HandoffTracker, name: str, call_id: str, arguments: Any
) -> list[ToolEvent]:
    """Build the ``start`` (+ nested ``parent_link``) events for a call."""
    tracker.remember(call_id, name)
    events = [
        ToolEvent(
            kind="start",
            tool_name=name,
            tool_use_id=call_id,
            tool_input=_arguments(arguments),
        )
    ]
    if tracker.active_handoff_id is not None:
        events.append(
            ToolEvent(
                kind="parent_link",
                tool_name=name,
                tool_use_id=call_id,
                parent_tool_use_id=tracker.active_handoff_id,
            )
        )
    return events


def _agent_active_event(
    tracker: HandoffTracker, handoff_id: str, agent_name: str
) -> ToolEvent:
    """Build the ``subagent_message`` announcing *agent_name* took over."""
    return ToolEvent(
        kind="subagent_message",
        tool_name=tracker.active_handoff_name,
        tool_use_id=handoff_id,
        parent_tool_use_id=handoff_id,
        text=f"Agent '{agent_name}' is now active.",
    )


def tool_events_for_item(
    item: Any, tracker: HandoffTracker, *, announce_handoff: bool
) -> list[ToolEvent]:
    """Map one SDK ``RunItem`` onto zero or more ``ToolEvent`` records.

    Args:
        item: A ``RunItem`` from ``RunResult.new_items`` or a
            ``RunItemStreamEvent.item``.
        tracker: The run's handoff state; mutated as handoffs occur.
        announce_handoff: When ``True`` a completed handoff also emits the
            "agent is now active" ``subagent_message`` (post-hoc path). The
            streaming path passes ``False`` and emits that message from the
            ``AgentUpdatedStreamEvent`` instead.

    Returns:
        The events for *item*, in emission order.
    """
    from agents.items import (
        HandoffCallItem,
        HandoffOutputItem,
        ItemHelpers,
        MessageOutputItem,
        ReasoningItem,
        ToolCallItem,
        ToolCallOutputItem,
    )

    if isinstance(item, HandoffCallItem | ToolCallItem):
        raw = item.raw_item
        hosted = hosted_call_for(raw)
        if hosted is not None:
            # Server-side call: no output item follows, so close it here.
            events = _start_events(
                tracker, hosted.name, hosted.call_id, hosted.arguments
            )
            tracker.settle(hosted.call_id)
            events.append(
                ToolEvent(
                    kind="end",
                    tool_name=hosted.name,
                    tool_use_id=hosted.call_id,
                    tool_response=hosted.response,
                )
            )
            return events
        if isinstance(raw, dict):
            name = str(raw.get("name", "") or "")
            arguments = raw.get("arguments")
        else:
            name = str(getattr(raw, "name", "") or "")
            arguments = getattr(raw, "arguments", None)
        return _start_events(tracker, name, _call_id(raw), arguments)

    if isinstance(item, HandoffOutputItem):
        call_id = _call_id(item.raw_item)
        name = tracker.name_for(call_id)
        target = _agent_name(item.target_agent)
        # The handoff call is no longer a pending tool call; it becomes an
        # open handoff that stays active until the run ends.
        tracker.settle(call_id)
        tracker.open_handoffs[call_id] = target
        tracker.active_handoff_id = call_id
        tracker.active_handoff_name = name
        if announce_handoff:
            return [_agent_active_event(tracker, call_id, target)]
        return []

    if isinstance(item, ToolCallOutputItem):
        call_id = _call_id(item.raw_item)
        tracker.settle(call_id)
        return [
            ToolEvent(
                kind="end",
                tool_name=tracker.name_for(call_id),
                tool_use_id=call_id,
                tool_response=str(item.output),
            )
        ]

    if isinstance(item, ReasoningItem):
        summaries = getattr(item.raw_item, "summary", None) or []
        text = "\n\n".join(
            str(getattr(entry, "text", "") or "") for entry in summaries
        ).strip()
        if not text:
            return []
        return [
            ToolEvent(
                kind="thinking",
                tool_name="",
                tool_use_id=str(getattr(item.raw_item, "id", "") or ""),
                text=text,
            )
        ]

    if isinstance(item, MessageOutputItem) and tracker.active_handoff_id is not None:
        text = ItemHelpers.text_message_output(item).strip()
        if not text:
            return []
        return [
            ToolEvent(
                kind="subagent_message",
                tool_name=tracker.active_handoff_name,
                tool_use_id=tracker.active_handoff_id,
                parent_tool_use_id=tracker.active_handoff_id,
                text=text,
            )
        ]

    return []


def tool_events_for_stream_event(
    event: Any, tracker: HandoffTracker
) -> list[ToolEvent]:
    """Map one ``Runner.run_streamed`` event onto ``ToolEvent`` records.

    Args:
        event: A ``RunItemStreamEvent``, ``AgentUpdatedStreamEvent``, or
            ``RawResponsesStreamEvent`` (the last yields nothing here).
        tracker: The run's handoff state.

    Returns:
        The events for *event*, in emission order.
    """
    event_type = getattr(event, "type", "")
    if event_type == "run_item_stream_event":
        return tool_events_for_item(event.item, tracker, announce_handoff=False)
    if event_type == "agent_updated_stream_event":
        if tracker.active_handoff_id is None:
            # The SDK announces the entry agent before any handoff; nothing to
            # nest under.
            return []
        return [
            _agent_active_event(
                tracker, tracker.active_handoff_id, _agent_name(event.new_agent)
            )
        ]
    return []


def tool_events_for_run_items(items: list[Any]) -> list[ToolEvent]:
    """Map a completed run's ``new_items`` onto ``ToolEvent`` records (post-hoc).

    Used by the non-streaming session path, where no stream events exist: the
    same ordered events are reconstructed from ``RunResult.new_items``.

    Args:
        items: ``RunResult.new_items``.

    Returns:
        All events for the run, in item order.
    """
    tracker = HandoffTracker()
    events: list[ToolEvent] = []
    for item in items:
        events.extend(tool_events_for_item(item, tracker, announce_handoff=True))
    events.extend(tracker.close())
    return events
