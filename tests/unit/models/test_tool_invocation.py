"""Tests for the ToolInvocation model and TestResult.tool_invocations (T010e).

The dashboard Explorer renders tool-use panels from `{name, args, result, bytes}`
tuples. This test locks the on-disk shape that the runtime produces.
"""

import json
from datetime import datetime
from typing import get_args

import pytest
from pydantic import ValidationError

from holodeck.models.test_result import TestResult, ToolInvocation


@pytest.mark.unit
class TestToolInvocationModel:
    """Lock the ToolInvocation field contract."""

    def test_required_fields(self) -> None:
        invocation = ToolInvocation(
            name="lookup_order",
            args={"order_id": "abc"},
            result={"status": "shipped"},
            bytes=22,
        )
        assert invocation.name == "lookup_order"
        assert invocation.args == {"order_id": "abc"}
        assert invocation.result == {"status": "shipped"}
        assert invocation.bytes == 22
        assert invocation.duration_ms is None
        assert invocation.error is None

    def test_minimal_construction_uses_field_defaults(self) -> None:
        invocation = ToolInvocation(name="noop", bytes=4)
        assert invocation.args == {}
        assert invocation.result is None
        assert invocation.duration_ms is None
        assert invocation.error is None

    def test_forbids_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            ToolInvocation(
                name="x",
                bytes=0,
                not_a_field="nope",  # type: ignore[call-arg]
            )

    def test_bytes_must_be_non_negative(self) -> None:
        with pytest.raises(ValidationError):
            ToolInvocation(name="x", bytes=-1)

    def test_round_trip_preserves_all_fields(self) -> None:
        original = ToolInvocation(
            name="search",
            args={"q": "hello"},
            result=[1, 2, 3],
            bytes=len(json.dumps([1, 2, 3])),
            duration_ms=42,
            error=None,
        )
        rehydrated = ToolInvocation.model_validate_json(original.model_dump_json())
        assert rehydrated == original

    def test_round_trip_preserves_error_case(self) -> None:
        original = ToolInvocation(
            name="search",
            args={"q": "boom"},
            result=None,
            bytes=4,
            error="connection timed out",
        )
        rehydrated = ToolInvocation.model_validate_json(original.model_dump_json())
        assert rehydrated == original
        assert rehydrated.result is None
        assert rehydrated.error == "connection timed out"


@pytest.mark.unit
class TestTestResultToolInvocations:
    """Lock the TestResult.tool_invocations field contract."""

    def test_field_defaults_to_empty_list(self) -> None:
        result = TestResult(
            test_input="hi",
            passed=True,
            execution_time_ms=1,
            timestamp="2026-04-18T00:00:00Z",
        )
        assert result.tool_invocations == []

    def test_legacy_tool_calls_field_preserved(self) -> None:
        """The legacy `tool_calls: list[str]` field must stay for back-compat."""
        result = TestResult(
            test_input="hi",
            passed=True,
            execution_time_ms=1,
            timestamp="2026-04-18T00:00:00Z",
            tool_calls=["name_only"],
        )
        assert result.tool_calls == ["name_only"]
        assert result.tool_invocations == []

    def test_round_trip_preserves_tool_invocations(self) -> None:
        invocation = ToolInvocation(
            name="search",
            args={"q": "x"},
            result="ok",
            bytes=4,
        )
        original = TestResult(
            test_input="hi",
            passed=True,
            execution_time_ms=1,
            timestamp="2026-04-18T00:00:00Z",
            tool_invocations=[invocation],
        )
        rehydrated = TestResult.model_validate_json(original.model_dump_json())
        assert rehydrated.tool_invocations == [invocation]

    def test_legacy_json_without_field_loads_with_empty_list(self) -> None:
        legacy = (
            '{"test_input":"hi","passed":true,"execution_time_ms":1,'
            '"timestamp":"2026-04-18T00:00:00Z"}'
        )
        result = TestResult.model_validate_json(legacy)
        assert result.tool_invocations == []


@pytest.mark.unit
class TestToolInvocationFieldSchema:
    """Locks the Literal / annotation shapes consumed by external readers."""

    def test_tool_invocations_annotation_is_list_of_tool_invocation(self) -> None:
        annotation = TestResult.model_fields["tool_invocations"].annotation
        # list[ToolInvocation] → args is (ToolInvocation,)
        assert ToolInvocation in get_args(annotation)

    def test_datetime_result_is_serialised_via_json_default_str(self) -> None:
        """Sanity-check: consumers that want datetime results must JSON-coerce them;
        the builder (T010h) does this before constructing ToolInvocation."""
        raw = {"now": datetime(2026, 4, 18, 12, 0, 0).isoformat()}
        invocation = ToolInvocation(
            name="clock",
            args={},
            result=raw,
            bytes=len(json.dumps(raw)),
        )
        rehydrated = ToolInvocation.model_validate_json(invocation.model_dump_json())
        assert rehydrated.result == raw
