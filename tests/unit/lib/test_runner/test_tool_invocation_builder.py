"""Tests for pair_tool_calls — SK index-pairing vs Claude id-pairing (T010g).

Per research.md R8:
- Semantic Kernel: positional index pairing of parallel `tool_calls` /
  `tool_results` lists. Shorter `tool_results` is padded with
  `error="no result received"` preserving call order.
- Claude Agent SDK: id pairing on the `call_id` key (value populated from
  `ToolUseBlock.id` / `ToolResultBlock.tool_use_id`). Unmatched call →
  same padding rule; unmatched result is a protocol violation and is skipped
  with a WARNING.
"""

import json
import logging
from datetime import datetime

import pytest

from holodeck.lib.test_runner.tool_invocation_builder import pair_tool_calls


@pytest.mark.unit
class TestSKIndexPairing:
    """SK backend: positional pairing."""

    def test_equal_length_lists_pair_in_order(self) -> None:
        tool_calls = [
            {"name": "a", "arguments": {}},
            {"name": "b", "arguments": {"x": 1}},
        ]
        tool_results = [
            {"name": "a", "result": 1},
            {"name": "b", "result": 2},
        ]
        invocations = pair_tool_calls(tool_calls, tool_results, backend_kind="sk")

        assert len(invocations) == 2
        assert invocations[0].name == "a"
        assert invocations[0].args == {}
        assert invocations[0].result == 1
        assert invocations[0].error is None

        assert invocations[1].name == "b"
        assert invocations[1].args == {"x": 1}
        assert invocations[1].result == 2

    def test_shorter_results_list_pads_with_error(self) -> None:
        tool_calls = [
            {"name": "a", "arguments": {}},
            {"name": "b", "arguments": {"x": 1}},
        ]
        tool_results = [{"name": "a", "result": 1}]
        invocations = pair_tool_calls(tool_calls, tool_results, backend_kind="sk")

        assert len(invocations) == 2
        assert invocations[0].error is None
        assert invocations[1].name == "b"
        assert invocations[1].result is None
        assert invocations[1].error == "no result received"

    def test_bytes_computed_for_unserialisable_result(self) -> None:
        dt = datetime(2026, 4, 18, 12, 0, 0)
        tool_calls = [{"name": "clock", "arguments": {}}]
        tool_results = [{"name": "clock", "result": dt}]

        invocations = pair_tool_calls(tool_calls, tool_results, backend_kind="sk")

        assert len(invocations) == 1
        # `default=str` coerces datetime to isoformat string before length.
        expected_bytes = len(json.dumps(dt.isoformat()))
        assert invocations[0].bytes == expected_bytes

    def test_empty_inputs_produce_empty_output(self) -> None:
        assert pair_tool_calls([], [], backend_kind="sk") == []


@pytest.mark.unit
class TestClaudeIdPairing:
    """Claude backend: pair by `call_id`."""

    def test_id_pairing_ignores_order(self) -> None:
        tool_calls = [
            {"name": "a", "arguments": {"q": 1}, "call_id": "id-a"},
            {"name": "b", "arguments": {"q": 2}, "call_id": "id-b"},
        ]
        tool_results = [
            {"call_id": "id-b", "result": "second"},
            {"call_id": "id-a", "result": "first"},
        ]

        invocations = pair_tool_calls(tool_calls, tool_results, backend_kind="claude")

        by_name = {inv.name: inv for inv in invocations}
        assert by_name["a"].result == "first"
        assert by_name["b"].result == "second"

    def test_unmatched_call_id_pads_with_error(self) -> None:
        tool_calls = [
            {"name": "a", "arguments": {}, "call_id": "id-a"},
            {"name": "b", "arguments": {}, "call_id": "id-b"},
        ]
        tool_results = [{"call_id": "id-a", "result": "first"}]

        invocations = pair_tool_calls(tool_calls, tool_results, backend_kind="claude")

        assert len(invocations) == 2
        by_name = {inv.name: inv for inv in invocations}
        assert by_name["a"].result == "first"
        assert by_name["a"].error is None
        assert by_name["b"].result is None
        assert by_name["b"].error == "no result received"

    def test_result_without_matching_call_logs_warning_and_is_skipped(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        tool_calls = [{"name": "a", "arguments": {}, "call_id": "id-a"}]
        tool_results = [
            {"call_id": "id-a", "result": 1},
            {"call_id": "id-ghost", "result": "orphan"},
        ]

        with caplog.at_level(
            logging.WARNING, logger="holodeck.lib.test_runner.tool_invocation_builder"
        ):
            invocations = pair_tool_calls(
                tool_calls, tool_results, backend_kind="claude"
            )

        assert len(invocations) == 1
        assert invocations[0].name == "a"
        assert any("id-ghost" in record.message for record in caplog.records)

    def test_preserves_original_call_order(self) -> None:
        tool_calls = [
            {"name": "first", "arguments": {}, "call_id": "id-1"},
            {"name": "second", "arguments": {}, "call_id": "id-2"},
            {"name": "third", "arguments": {}, "call_id": "id-3"},
        ]
        tool_results = [
            {"call_id": "id-3", "result": 3},
            {"call_id": "id-1", "result": 1},
            {"call_id": "id-2", "result": 2},
        ]

        invocations = pair_tool_calls(tool_calls, tool_results, backend_kind="claude")

        assert [inv.name for inv in invocations] == ["first", "second", "third"]
