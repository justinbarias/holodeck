"""Deterministic offline LLM provider for the verify smoke tier.

Implements just enough of two wire protocols for HoloDeck's backends to finish
real turns without network access or credentials:

* OpenAI Responses API: ``POST /v1/responses`` (JSON and SSE streaming).
* Anthropic Messages API: ``POST /v1/messages`` (JSON and SSE streaming) and
  ``POST /v1/messages/count_tokens``.

Scripted behaviour: when the latest user message contains ``SMOKE_TOOL`` and no
tool result follows it, reply with one call to the first offered tool whose
name contains ``smoke``. Otherwise reply ``SMOKE-OK`` plus a short echo.

Every request is appended to a JSONL log. Unknown paths return 404 and are
logged, so the log also shows which endpoints a client probes.

Usage::

    uv run python scripts/verify/fake_provider.py --port 8765 --log provider.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import threading
import time
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

TRIGGER = "SMOKE_TOOL"
OK_TEXT = "SMOKE-OK"
# The smoke tool returns "SMOKE-TOOL-RESULT: <text>" and is always called with
# "smoke-ping"; only that exact output earns the verified reply, so a tool
# that raised or returned an error cannot pass the tool turn.
TOOL_RESULT = "SMOKE-TOOL-RESULT: smoke-ping"
TOOL_VERIFIED = "tool-verified"
TOOL_FAILED = "SMOKE-TOOL-FAILED"
TOOL_ERROR = "[tool error]"
ECHO_CHARS = 60

JsonObject = dict[str, object]


@dataclass(frozen=True)
class ToolCall:
    """Scripted decision: call one tool."""

    name: str
    arguments: JsonObject


@dataclass(frozen=True)
class Text:
    """Scripted decision: reply with text."""

    text: str


Decision = ToolCall | Text


def _as_object(value: object) -> JsonObject:
    return cast(JsonObject, value) if isinstance(value, dict) else {}


def _as_list(value: object) -> list[object]:
    return cast(list[object], value) if isinstance(value, list) else []


def _content_text(content: object) -> str:
    """Return the last text part of a message ``content`` (string or part list).

    Claude Code prepends ``<system-reminder>`` text parts to user messages, so
    the user's own text is the last part.
    """
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for raw in _as_list(content):
        part = _as_object(raw)
        text = part.get("text")
        if isinstance(text, str) and part.get("type") in (
            "text",
            "input_text",
            "output_text",
        ):
            parts.append(text)
    return parts[-1] if parts else ""


def _smoke_arguments(schema: object) -> JsonObject:
    """Fill the required properties of a JSON schema with fixed values."""
    obj = _as_object(schema)
    properties = _as_object(obj.get("properties"))
    required = [str(name) for name in _as_list(obj.get("required"))] or list(properties)
    arguments: JsonObject = {}
    for name in required:
        kind = _as_object(properties.get(name)).get("type")
        if kind in ("integer", "number"):
            arguments[name] = 1
        elif kind == "boolean":
            arguments[name] = True
        else:
            arguments[name] = "smoke-ping"
    return arguments


def _any_text(content: object) -> str:
    """Return all text in a tool output (string or part list), joined."""
    if isinstance(content, str):
        return content
    texts = [_as_object(part).get("text") for part in _as_list(content)]
    return " ".join(text for text in texts if isinstance(text, str))


def _decide(
    latest_user: str,
    tool_output: str | None,
    tool: tuple[str, object] | None,
) -> Decision:
    """Pick the scripted reply.

    *tool_output* is the tool result that follows the latest user message, or
    ``None`` when no tool has run since that message.
    """
    if TRIGGER in latest_user and tool_output is None and tool is not None:
        return ToolCall(name=tool[0], arguments=_smoke_arguments(tool[1]))
    echo = " ".join(latest_user.split())[:ECHO_CHARS]
    if tool_output is None:
        return Text(text=f"{OK_TEXT} echo: {echo}")
    if TOOL_RESULT in tool_output and TOOL_ERROR not in tool_output:
        return Text(text=f"{OK_TEXT} {TOOL_VERIFIED} echo: {echo}")
    return Text(text=f"{TOOL_FAILED}: {' '.join(tool_output.split())[:120]}")


def _find_smoke_tool(tools: object, schema_key: str) -> tuple[str, object] | None:
    for raw in _as_list(tools):
        tool = _as_object(raw)
        # Chat-completions style nests the function; Responses and Messages do not.
        function = _as_object(tool.get("function"))
        name = tool.get("name") or function.get("name")
        if isinstance(name, str) and "smoke" in name.lower():
            schema = tool.get(schema_key) or function.get("parameters")
            return name, schema
    return None


def decide_openai(body: JsonObject) -> Decision:
    """Decide the reply for an OpenAI Responses API request body."""
    items = body.get("input")
    if isinstance(items, str):
        items = [{"role": "user", "content": items}]
    latest_user = ""
    tool_output: str | None = None
    for raw in _as_list(items):
        item = _as_object(raw)
        kind = item.get("type")
        if kind == "function_call_output":
            tool_output = (tool_output or "") + _any_text(item.get("output"))
        elif item.get("role") == "user" and kind in (None, "message"):
            latest_user = _content_text(item.get("content"))
            tool_output = None
    return _decide(
        latest_user, tool_output, _find_smoke_tool(body.get("tools"), "parameters")
    )


def decide_anthropic(body: JsonObject) -> Decision:
    """Decide the reply for an Anthropic Messages API request body."""
    latest_user = ""
    tool_output: str | None = None
    for raw in _as_list(body.get("messages")):
        message = _as_object(raw)
        if message.get("role") != "user":
            continue
        content = message.get("content")
        results = [
            block
            for block in (_as_object(raw_block) for raw_block in _as_list(content))
            if block.get("type") == "tool_result"
        ]
        if results:
            for block in results:
                output = _any_text(block.get("content"))
                if block.get("is_error") is True:
                    output = f"{TOOL_ERROR} {output}"
                tool_output = (tool_output or "") + output
            continue
        text = _content_text(content)
        if text:
            latest_user = text
            tool_output = None
    return _decide(
        latest_user,
        tool_output,
        _find_smoke_tool(body.get("tools"), "input_schema"),
    )


_ids = itertools.count(1)


def _next_id(prefix: str) -> str:
    return f"{prefix}_smoke{next(_ids):06d}"


def _sse(event: str, data: JsonObject) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


# --------------------------------------------------------------------------- #
# OpenAI Responses API                                                        #
# --------------------------------------------------------------------------- #


def _openai_output_item(decision: Decision) -> JsonObject:
    if isinstance(decision, ToolCall):
        return {
            "type": "function_call",
            "id": _next_id("fc"),
            "call_id": _next_id("call"),
            "name": decision.name,
            "arguments": json.dumps(decision.arguments),
            "status": "completed",
        }
    return {
        "type": "message",
        "id": _next_id("msg"),
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": decision.text, "annotations": []}],
    }


def openai_response(body: JsonObject, decision: Decision) -> JsonObject:
    """Build a completed Responses API ``response`` object."""
    return {
        "id": _next_id("resp"),
        "object": "response",
        "created_at": int(time.time()),
        "model": str(body.get("model", "smoke-model")),
        "status": "completed",
        "output": [_openai_output_item(decision)],
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
        "usage": {
            "input_tokens": 10,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 5,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 15,
        },
    }


def openai_stream(body: JsonObject, decision: Decision) -> Iterator[str]:
    """Yield Responses API SSE frames for one scripted reply."""
    response = openai_response(body, decision)
    item = _as_object(_as_list(response["output"])[0])
    sequence = itertools.count()
    in_progress = {**response, "status": "in_progress", "output": []}
    yield _sse(
        "response.created",
        {
            "type": "response.created",
            "sequence_number": next(sequence),
            "response": in_progress,
        },
    )
    item_id = str(item["id"])
    if item["type"] == "function_call":
        yield _sse(
            "response.output_item.added",
            {
                "type": "response.output_item.added",
                "sequence_number": next(sequence),
                "output_index": 0,
                "item": {**item, "arguments": "", "status": "in_progress"},
            },
        )
        yield _sse(
            "response.function_call_arguments.delta",
            {
                "type": "response.function_call_arguments.delta",
                "sequence_number": next(sequence),
                "item_id": item_id,
                "output_index": 0,
                "delta": item["arguments"],
            },
        )
        yield _sse(
            "response.function_call_arguments.done",
            {
                "type": "response.function_call_arguments.done",
                "sequence_number": next(sequence),
                "item_id": item_id,
                "output_index": 0,
                "arguments": item["arguments"],
            },
        )
    else:
        text = str(_as_object(_as_list(item["content"])[0])["text"])
        part: JsonObject = {"type": "output_text", "text": "", "annotations": []}
        yield _sse(
            "response.output_item.added",
            {
                "type": "response.output_item.added",
                "sequence_number": next(sequence),
                "output_index": 0,
                "item": {**item, "content": [], "status": "in_progress"},
            },
        )
        yield _sse(
            "response.content_part.added",
            {
                "type": "response.content_part.added",
                "sequence_number": next(sequence),
                "item_id": item_id,
                "output_index": 0,
                "content_index": 0,
                "part": part,
            },
        )
        yield _sse(
            "response.output_text.delta",
            {
                "type": "response.output_text.delta",
                "sequence_number": next(sequence),
                "item_id": item_id,
                "output_index": 0,
                "content_index": 0,
                "delta": text,
                "logprobs": [],
            },
        )
        yield _sse(
            "response.output_text.done",
            {
                "type": "response.output_text.done",
                "sequence_number": next(sequence),
                "item_id": item_id,
                "output_index": 0,
                "content_index": 0,
                "text": text,
                "logprobs": [],
            },
        )
        yield _sse(
            "response.content_part.done",
            {
                "type": "response.content_part.done",
                "sequence_number": next(sequence),
                "item_id": item_id,
                "output_index": 0,
                "content_index": 0,
                "part": {**part, "text": text},
            },
        )
    yield _sse(
        "response.output_item.done",
        {
            "type": "response.output_item.done",
            "sequence_number": next(sequence),
            "output_index": 0,
            "item": item,
        },
    )
    yield _sse(
        "response.completed",
        {
            "type": "response.completed",
            "sequence_number": next(sequence),
            "response": response,
        },
    )


# --------------------------------------------------------------------------- #
# Anthropic Messages API                                                      #
# --------------------------------------------------------------------------- #


def _anthropic_block(decision: Decision) -> JsonObject:
    if isinstance(decision, ToolCall):
        return {
            "type": "tool_use",
            "id": _next_id("toolu"),
            "name": decision.name,
            "input": decision.arguments,
        }
    return {"type": "text", "text": decision.text}


def anthropic_message(body: JsonObject, decision: Decision) -> JsonObject:
    """Build a complete Messages API response."""
    return {
        "id": _next_id("msg"),
        "type": "message",
        "role": "assistant",
        "model": str(body.get("model", "smoke-model")),
        "content": [_anthropic_block(decision)],
        "stop_reason": "tool_use" if isinstance(decision, ToolCall) else "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }


def anthropic_stream(body: JsonObject, decision: Decision) -> Iterator[str]:
    """Yield Messages API SSE frames for one scripted reply."""
    message = anthropic_message(body, decision)
    block = _as_object(_as_list(message["content"])[0])
    yield _sse(
        "message_start",
        {
            "type": "message_start",
            "message": {
                **message,
                "content": [],
                "stop_reason": None,
                "usage": {"input_tokens": 10, "output_tokens": 1},
            },
        },
    )
    if block["type"] == "tool_use":
        start: JsonObject = {**block, "input": {}}
        delta: JsonObject = {
            "type": "input_json_delta",
            "partial_json": json.dumps(block["input"]),
        }
    else:
        start = {"type": "text", "text": ""}
        delta = {"type": "text_delta", "text": block["text"]}
    yield _sse(
        "content_block_start",
        {"type": "content_block_start", "index": 0, "content_block": start},
    )
    yield _sse(
        "content_block_delta",
        {"type": "content_block_delta", "index": 0, "delta": delta},
    )
    yield _sse("content_block_stop", {"type": "content_block_stop", "index": 0})
    yield _sse(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": message["stop_reason"], "stop_sequence": None},
            "usage": {"output_tokens": 5},
        },
    )
    yield _sse("message_stop", {"type": "message_stop"})


# --------------------------------------------------------------------------- #
# App                                                                         #
# --------------------------------------------------------------------------- #


class RequestLog:
    """Append-only JSONL request log, safe across worker threads."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)

    def _write(self, record: JsonObject) -> None:
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    async def write(self, record: JsonObject) -> None:
        """Append one record without blocking the event loop."""
        await asyncio.to_thread(self._write, record)


def read_log(path: Path) -> list[JsonObject]:
    """Read every record of a request log."""
    if not path.exists():
        return []
    return [
        _as_object(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _stream(frames: Iterator[str]) -> StreamingResponse:
    async def body() -> AsyncIterator[bytes]:
        for frame in frames:
            yield frame.encode()

    return StreamingResponse(body(), media_type="text/event-stream")


def create_app(log_path: Path) -> FastAPI:
    """Create the fake provider app writing to *log_path*."""
    app = FastAPI(title="HoloDeck verify fake provider")
    log = RequestLog(log_path)

    async def record(request: Request, status: int) -> JsonObject:
        raw = await request.body()
        try:
            parsed: object = json.loads(raw) if raw else None
        except ValueError:
            parsed = raw.decode(errors="replace")
        await log.write(
            {
                "ts": time.time(),
                "method": request.method,
                "path": request.url.path,
                "query": request.url.query,
                "status": status,
                "body": parsed,
            }
        )
        return _as_object(parsed)

    @app.get("/health")
    async def health() -> Response:
        return JSONResponse({"status": "ok"})

    @app.post("/v1/responses")
    async def responses(request: Request) -> Response:
        body = await record(request, 200)
        decision = decide_openai(body)
        if body.get("stream") is True:
            return _stream(openai_stream(body, decision))
        return JSONResponse(openai_response(body, decision))

    @app.post("/v1/messages")
    async def messages(request: Request) -> Response:
        body = await record(request, 200)
        decision = decide_anthropic(body)
        if body.get("stream") is True:
            return _stream(anthropic_stream(body, decision))
        return JSONResponse(anthropic_message(body, decision))

    @app.post("/v1/messages/count_tokens")
    async def count_tokens(request: Request) -> Response:
        await record(request, 200)
        return JSONResponse({"input_tokens": 10})

    @app.api_route(
        "/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD"]
    )
    async def unhandled(request: Request, path: str) -> Response:
        await record(request, 404)
        return JSONResponse(
            {"type": "error", "error": {"type": "not_found_error", "message": path}},
            status_code=404,
        )

    return app


def main(argv: list[str] | None = None) -> int:
    """Serve the fake provider until interrupted."""
    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--log", type=Path, required=True)
    args = parser.parse_args(argv)
    uvicorn.run(
        create_app(args.log), host=args.host, port=args.port, log_level="warning"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
