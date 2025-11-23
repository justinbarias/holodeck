"""Tests for ToolExecution and TokenUsage models."""

import re

import pytest
from pydantic import ValidationError

from holodeck.models.token_usage import TokenUsage
from holodeck.models.tool_execution import ToolExecution, ToolStatus


class TestToolExecution:
    """ToolExecution validation behavior."""

    def test_error_message_required_when_failed(self) -> None:
        """Ensure failed status requires an error message."""
        with pytest.raises(ValidationError):
            ToolExecution(tool_name="search", status=ToolStatus.FAILED)

    def test_execution_time_must_be_non_negative(self) -> None:
        """Negative execution_time is rejected."""
        with pytest.raises(ValidationError):
            ToolExecution(
                tool_name="search",
                status=ToolStatus.SUCCESS,
                execution_time=-0.1,
            )

    def test_result_is_sanitized(self) -> None:
        """Result output should be stripped of ANSI sequences."""
        raw_result = "done\x1b[31mERROR\x1b[0m"
        execution = ToolExecution(
            tool_name="search",
            status=ToolStatus.SUCCESS,
            result=raw_result,
        )
        assert "\x1b" not in execution.result
        assert re.sub(r"\s+", "", execution.result or "") == "doneERROR"


class TestTokenUsage:
    """TokenUsage validation behavior."""

    def test_total_tokens_must_match_components(self) -> None:
        """Validate total_tokens equality."""
        with pytest.raises(ValidationError):
            TokenUsage(prompt_tokens=5, completion_tokens=10, total_tokens=20)

    def test_valid_token_usage(self) -> None:
        """Valid token usage passes."""
        usage = TokenUsage(prompt_tokens=5, completion_tokens=10, total_tokens=15)
        assert usage.total_tokens == 15
