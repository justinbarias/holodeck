"""Offline coverage for the verify smoke/live helpers (SSE driver, fake provider)."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from scripts.verify.agui import assistant_text, check_turn, parse_sse
from scripts.verify.fake_provider import (
    Text,
    ToolCall,
    create_app,
    decide_anthropic,
    decide_openai,
    read_log,
)
from scripts.verify.smoke import BACKENDS, isolation_problems, missing_prerequisite

pytestmark = pytest.mark.unit


def _sse(*events: dict[str, object]) -> str:
    return "".join(f"data: {json.dumps(e)}\n\n" for e in events)


GOOD = [
    {"type": "RUN_STARTED", "threadId": "t", "runId": "r"},
    {"type": "TOOL_CALL_START", "toolCallId": "c1", "toolCallName": "smoke_echo"},
    {"type": "TOOL_CALL_ARGS", "toolCallId": "c1", "delta": "{}"},
    {"type": "TOOL_CALL_END", "toolCallId": "c1"},
    {"type": "TEXT_MESSAGE_START", "messageId": "m"},
    {"type": "TEXT_MESSAGE_CONTENT", "messageId": "m", "delta": "SMOKE-OK "},
    {"type": "TEXT_MESSAGE_CONTENT", "messageId": "m", "delta": "echo"},
    {"type": "TEXT_MESSAGE_END", "messageId": "m"},
    {"type": "RUN_FINISHED", "threadId": "t", "runId": "r"},
]


def test_parse_sse_handles_comments_multiline_and_crlf() -> None:
    body = ': keepalive\r\nevent: x\r\ndata: {"type":\r\ndata: "RUN_STARTED"}\r\n\r\n'
    body += 'data: {"type": "RUN_FINISHED"}'  # no trailing blank line
    assert [e["type"] for e in parse_sse(body)] == ["RUN_STARTED", "RUN_FINISHED"]


def test_parse_sse_rejects_non_object() -> None:
    with pytest.raises(ValueError, match="not a JSON object"):
        parse_sse("data: [1]\n\n")


def test_check_turn_passes_complete_tool_run() -> None:
    events = parse_sse(_sse(*GOOD))
    assert assistant_text(events) == "SMOKE-OK echo"
    assert check_turn(events, expect_text="SMOKE-OK", expect_tool=True) == []


def test_check_turn_reports_each_structural_problem() -> None:
    events = [
        {"type": "TEXT_MESSAGE_START", "messageId": "m"},
        {"type": "TOOL_CALL_START", "toolCallId": "c1"},
        {"type": "RUN_ERROR", "message": "boom"},
    ]
    problems = check_turn(events, expect_text="SMOKE-OK", expect_tool=True)
    assert problems == [
        "first event is TEXT_MESSAGE_START, expected RUN_STARTED",
        "last event is RUN_ERROR, expected RUN_FINISHED",
        "RUN_ERROR: boom",
        "no TEXT_MESSAGE_CONTENT text",
        "text does not contain 'SMOKE-OK': ''",
        "unbalanced TEXT_MESSAGE_START/END",
        "unbalanced TOOL_CALL_START/END",
        "tool turn missing TOOL_CALL_ARGS, TOOL_CALL_END",
    ]
    assert check_turn([]) == ["no events received"]


OPENAI_TOOLS = [
    {
        "type": "function",
        "name": "smoke_echo",
        "parameters": {
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    }
]


def test_openai_decision_calls_tool_once_then_answers() -> None:
    first = {
        "input": [{"role": "user", "content": "go SMOKE_TOOL"}],
        "tools": OPENAI_TOOLS,
    }
    assert decide_openai(first) == ToolCall("smoke_echo", {"text": "smoke-ping"})
    after_tool = {
        "input": [
            *first["input"],
            {"type": "function_call", "call_id": "c", "name": "smoke_echo"},
            {
                "type": "function_call_output",
                "call_id": "c",
                "output": "SMOKE-TOOL-RESULT: smoke-ping",
            },
        ],
        "tools": OPENAI_TOOLS,
    }
    decision = decide_openai(after_tool)
    assert decision == Text("SMOKE-OK tool-verified echo: go SMOKE_TOOL")
    # A tool that failed (any other output) never earns the verified reply.
    after_tool["input"][-1]["output"] = "Error: tool raised ValueError"
    decision = decide_openai(after_tool)
    assert isinstance(decision, Text)
    assert decision.text.startswith("SMOKE-TOOL-FAILED")
    # No smoke tool offered: answer in text even when triggered.
    assert isinstance(decide_openai({"input": "SMOKE_TOOL"}), Text)


def test_anthropic_decision_uses_last_text_block() -> None:
    tools = [{"name": "mcp__holodeck_tools__smoke_echo", "input_schema": {}}]
    reminder = {"type": "text", "text": "<system-reminder>context</system-reminder>"}
    user = {"type": "text", "text": "please SMOKE_TOOL"}
    body = {"messages": [{"role": "user", "content": [reminder, user]}], "tools": tools}
    assert decide_anthropic(body) == ToolCall("mcp__holodeck_tools__smoke_echo", {})
    body["messages"] = [
        *body["messages"],
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t"}]},
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "t",
                    "content": [
                        {"type": "text", "text": "SMOKE-TOOL-RESULT: smoke-ping"}
                    ],
                }
            ],
        },
    ]
    assert decide_anthropic(body) == Text(
        "SMOKE-OK tool-verified echo: please SMOKE_TOOL"
    )
    # An errored tool result fails even if its text carries the marker.
    body["messages"][-1]["content"][0]["is_error"] = True
    decision = decide_anthropic(body)
    assert isinstance(decision, Text)
    assert decision.text.startswith("SMOKE-TOOL-FAILED")


def test_fake_provider_streams_anthropic_and_logs_unknown_paths(tmp_path: Path) -> None:
    log = tmp_path / "provider.jsonl"
    client = TestClient(create_app(log))
    body = {"stream": True, "messages": [{"role": "user", "content": "hi"}]}
    response = client.post("/v1/messages?beta=true", json=body)
    types = [e["type"] for e in parse_sse(response.text)]
    assert types == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    assert client.post("/v1/unknown", json={}).status_code == 404
    records = read_log(log)
    assert [(r["path"], r["status"]) for r in records] == [
        ("/v1/messages", 200),
        ("/v1/unknown", 404),
    ]


def test_fake_provider_openai_stream_ends_with_completed(tmp_path: Path) -> None:
    client = TestClient(create_app(tmp_path / "log.jsonl"))
    body = {"stream": True, "input": "hi", "model": "m"}
    events = parse_sse(client.post("/v1/responses", json=body).text)
    assert events[0]["type"] == "response.created"
    assert events[-1]["type"] == "response.completed"
    response = client.post("/v1/responses", json={**body, "stream": False}).json()
    assert response["output"][0]["content"][0]["text"] == "SMOKE-OK echo: hi"


def test_isolation_detects_leak_and_missing_thread(tmp_path: Path) -> None:
    log = tmp_path / "log.jsonl"
    log.write_text(
        json.dumps({"path": "/v1/messages", "body": {"m": "ALPHA BRAVO"}}) + "\n",
        encoding="utf-8",
    )
    assert isolation_problems(log, "ALPHA", "BRAVO") == [
        "thread B request to /v1/messages leaks thread A"
    ]
    assert isolation_problems(log, "ALPHA", "CHARLIE") == [
        "no provider request carried thread B's message"
    ]


def test_claude_on_linux_without_bwrap_reports_install_hint() -> None:
    claude = next(b for b in BACKENDS if b.name == "claude")
    openai = next(b for b in BACKENDS if b.name == "openai")

    def no_bwrap(_: str) -> str | None:
        return None

    problem = missing_prerequisite(claude, "linux", no_bwrap)
    assert problem is not None
    assert "apt-get install -y bubblewrap" in problem
    assert missing_prerequisite(claude, "linux", lambda _: "/usr/bin/bwrap") is None
    assert missing_prerequisite(claude, "darwin", no_bwrap) is None
    assert missing_prerequisite(openai, "linux", no_bwrap) is None
