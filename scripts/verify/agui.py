"""AG-UI SSE driver and event assertions shared by the smoke and live tiers.

The driver posts ``RunAgentInput`` bodies to ``POST /awp`` on a running
``holodeck serve`` process, parses the SSE stream into event dicts, and keeps a
per-thread message history so each thread's later turns carry its own context.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import httpx

JsonObject = dict[str, object]

RUN_STARTED = "RUN_STARTED"
RUN_FINISHED = "RUN_FINISHED"
RUN_ERROR = "RUN_ERROR"
TOOL_EVENTS = ("TOOL_CALL_START", "TOOL_CALL_ARGS", "TOOL_CALL_END")


@dataclass(frozen=True)
class TurnSpec:
    """One user turn to send on one thread."""

    thread: str
    prompt: str
    expect_tool: bool = False
    # When set, a TOOL_CALL_START must name this tool (substring match, so
    # Claude's ``mcp__<server>__<tool>`` names match the bare tool name).
    tool_name: str | None = None


@dataclass
class TurnResult:
    """Events and transport status for one sent turn."""

    spec: TurnSpec
    thread_id: str
    index: int
    status_code: int
    events: list[JsonObject] = field(default_factory=list)
    transport_error: str | None = None

    @property
    def types(self) -> list[str]:
        """Return event types in order."""
        return event_types(self.events)

    @property
    def text(self) -> str:
        """Return the concatenated assistant text deltas."""
        return assistant_text(self.events)


def parse_sse_lines(lines: Iterable[str]) -> Iterator[JsonObject]:
    """Parse SSE lines into JSON event objects.

    Joins multi-line ``data:`` fields, ignores comments and other fields, and
    yields one object per blank-line-terminated event.

    Raises:
        ValueError: If an event's data is not a JSON object.
    """
    data: list[str] = []
    for raw in lines:
        line = raw.rstrip("\r\n")
        if not line:
            if data:
                yield _decode("\n".join(data))
                data = []
            continue
        if line.startswith(":"):
            continue
        name, _, value = line.partition(":")
        if name == "data":
            data.append(value[1:] if value.startswith(" ") else value)
    if data:
        yield _decode("\n".join(data))


def _decode(payload: str) -> JsonObject:
    parsed: object = json.loads(payload)
    if not isinstance(parsed, dict):
        raise ValueError(f"SSE event is not a JSON object: {payload[:80]}")
    return cast(JsonObject, parsed)


def parse_sse(text: str) -> list[JsonObject]:
    """Parse a complete SSE body."""
    return list(parse_sse_lines(text.splitlines()))


def event_types(events: list[JsonObject]) -> list[str]:
    """Return the ``type`` of each event."""
    return [str(event.get("type", "")) for event in events]


def assistant_text(events: list[JsonObject]) -> str:
    """Concatenate ``TEXT_MESSAGE_CONTENT`` deltas."""
    return "".join(
        str(event.get("delta", ""))
        for event in events
        if event.get("type") == "TEXT_MESSAGE_CONTENT"
    )


def tool_call_names(events: list[JsonObject]) -> list[str]:
    """Return the ``toolCallName`` of each ``TOOL_CALL_START`` event."""
    return [
        str(event.get("toolCallName", ""))
        for event in events
        if event.get("type") == "TOOL_CALL_START"
    ]


def recall_problems(text: str, *, expected: str, forbidden: str) -> list[str]:
    """Check a recall reply names *expected* and not *forbidden* (any case)."""
    folded = text.casefold()
    problems: list[str] = []
    if expected.casefold() not in folded:
        problems.append(f"reply does not contain {expected!r}: {text[:120]!r}")
    if forbidden.casefold() in folded:
        problems.append(f"reply contains {forbidden!r} from another thread")
    return problems


def check_turn(
    events: list[JsonObject],
    *,
    expect_text: str | None = None,
    expect_tool: bool = False,
    expect_tool_name: str | None = None,
) -> list[str]:
    """Return problems with one run's event stream (empty when it passes).

    Always checks: first event ``RUN_STARTED``, last ``RUN_FINISHED``, no
    ``RUN_ERROR``, some assistant text, balanced text and tool-call events.
    ``expect_tool_name`` implies ``expect_tool`` and also requires a
    ``TOOL_CALL_START`` whose name contains it.
    """
    types = event_types(events)
    problems: list[str] = []
    if not types:
        return ["no events received"]
    if types[0] != RUN_STARTED:
        problems.append(f"first event is {types[0]}, expected {RUN_STARTED}")
    if types[-1] != RUN_FINISHED:
        problems.append(f"last event is {types[-1]}, expected {RUN_FINISHED}")
    for event in events:
        if event.get("type") == RUN_ERROR:
            problems.append(f"RUN_ERROR: {event.get('message', '')}"[:300])
    text = assistant_text(events)
    if not text.strip():
        problems.append("no TEXT_MESSAGE_CONTENT text")
    if expect_text is not None and expect_text not in text:
        problems.append(f"text does not contain {expect_text!r}: {text[:120]!r}")
    if types.count("TEXT_MESSAGE_START") != types.count("TEXT_MESSAGE_END"):
        problems.append("unbalanced TEXT_MESSAGE_START/END")
    started = {
        str(e.get("toolCallId")) for e in events if e.get("type") == "TOOL_CALL_START"
    }
    ended = {
        str(e.get("toolCallId")) for e in events if e.get("type") == "TOOL_CALL_END"
    }
    if started != ended:
        problems.append("unbalanced TOOL_CALL_START/END")
    if expect_tool or expect_tool_name:
        missing = [name for name in TOOL_EVENTS if name not in types]
        if missing:
            problems.append(f"tool turn missing {', '.join(missing)}")
    if expect_tool_name:
        names = tool_call_names(events)
        if not any(expect_tool_name in name for name in names):
            problems.append(f"no TOOL_CALL_START for {expect_tool_name!r}: {names}")
    return problems


def run_input(thread_id: str, messages: list[JsonObject]) -> JsonObject:
    """Build a ``RunAgentInput`` body."""
    return {
        "threadId": thread_id,
        "runId": f"run-{uuid.uuid4().hex[:12]}",
        "state": {},
        "messages": messages,
        "tools": [],
        "context": [],
        "forwardedProps": {},
    }


def drive(
    base_url: str,
    turns: list[TurnSpec],
    *,
    timeout: float = 180.0,
) -> list[TurnResult]:
    """Send *turns* in order and return their results.

    Each distinct ``TurnSpec.thread`` maps to its own AG-UI thread id; requests
    carry that thread's full user/assistant history, like CopilotKit does.
    """
    thread_ids: dict[str, str] = {}
    histories: dict[str, list[JsonObject]] = {}
    counts: dict[str, int] = {}
    results: list[TurnResult] = []
    with httpx.Client(base_url=base_url, timeout=timeout) as client:
        for spec in turns:
            thread_id = thread_ids.setdefault(
                spec.thread, f"verify-{spec.thread}-{uuid.uuid4().hex[:8]}"
            )
            history = histories.setdefault(spec.thread, [])
            counts[spec.thread] = counts.get(spec.thread, 0) + 1
            history.append(
                {"id": uuid.uuid4().hex, "role": "user", "content": spec.prompt}
            )
            result = TurnResult(
                spec=spec,
                thread_id=thread_id,
                index=counts[spec.thread],
                status_code=0,
            )
            try:
                with client.stream(
                    "POST",
                    "/awp",
                    json=run_input(thread_id, list(history)),
                    headers={"Accept": "text/event-stream"},
                ) as response:
                    result.status_code = response.status_code
                    if response.status_code != 200:
                        response.read()
                        result.transport_error = (
                            f"HTTP {response.status_code}: {response.text[:200]}"
                        )
                    else:
                        result.events = list(parse_sse_lines(response.iter_lines()))
            except (httpx.HTTPError, ValueError) as exc:
                result.transport_error = f"{type(exc).__name__}: {exc}"
            if result.text:
                history.append(
                    {
                        "id": uuid.uuid4().hex,
                        "role": "assistant",
                        "content": result.text,
                    }
                )
            results.append(result)
    return results


def save_transcript(results: list[TurnResult], path: Path) -> None:
    """Write one JSONL record per turn with its events."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for result in results:
            record = {
                "thread": result.spec.thread,
                "thread_id": result.thread_id,
                "turn": result.index,
                "prompt": result.spec.prompt,
                "status_code": result.status_code,
                "transport_error": result.transport_error,
                "events": result.events,
            }
            handle.write(json.dumps(record) + "\n")
