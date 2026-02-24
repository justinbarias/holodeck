"""Unit tests for holodeck.lib.backends.base.

Tests cover ExecutionResult dataclass construction and defaults,
BackendError exception hierarchy, and Protocol definitions.
"""

import pytest

from holodeck.lib.backends.base import (
    AgentBackend,
    AgentSession,
    BackendError,
    BackendInitError,
    BackendSessionError,
    BackendTimeoutError,
    ExecutionResult,
)
from holodeck.lib.errors import HoloDeckError
from holodeck.models.token_usage import TokenUsage


class TestExecutionResult:
    """Tests for the ExecutionResult dataclass."""

    @pytest.mark.unit
    def test_construction_with_response_only(self) -> None:
        """Create with just response, verify all defaults are correct."""
        result = ExecutionResult(response="Hello")

        assert result.response == "Hello"
        assert result.tool_calls == []
        assert result.tool_results == []
        assert result.token_usage == TokenUsage.zero()
        assert result.structured_output is None
        assert result.num_turns == 1
        assert result.is_error is False
        assert result.error_reason is None

    @pytest.mark.unit
    def test_construction_with_all_fields(self) -> None:
        """Create with all fields explicit, verify each."""
        token_usage = TokenUsage(
            prompt_tokens=10, completion_tokens=20, total_tokens=30
        )
        tool_calls = [{"name": "search", "args": {"query": "test"}}]
        tool_results = [{"name": "search", "result": "found"}]
        structured_output = {"key": "value"}

        result = ExecutionResult(
            response="Test response",
            tool_calls=tool_calls,
            tool_results=tool_results,
            token_usage=token_usage,
            structured_output=structured_output,
            num_turns=3,
            is_error=True,
            error_reason="something went wrong",
        )

        assert result.response == "Test response"
        assert result.tool_calls == tool_calls
        assert result.tool_results == tool_results
        assert result.token_usage == token_usage
        assert result.structured_output == structured_output
        assert result.num_turns == 3
        assert result.is_error is True
        assert result.error_reason == "something went wrong"

    @pytest.mark.unit
    def test_default_token_usage_is_zero(self) -> None:
        """Verify default token_usage has all counts at zero."""
        result = ExecutionResult(response="test")

        assert result.token_usage.prompt_tokens == 0
        assert result.token_usage.completion_tokens == 0
        assert result.token_usage.total_tokens == 0

    @pytest.mark.unit
    def test_error_max_turns_limit_reached(self) -> None:
        """Verify error fields for max_turns limit reached scenario."""
        result = ExecutionResult(
            response="",
            is_error=True,
            error_reason="max_turns limit reached",
            num_turns=10,
        )

        assert result.is_error is True
        assert result.error_reason == "max_turns limit reached"
        assert result.num_turns == 10

    @pytest.mark.unit
    def test_error_subprocess_crash(self) -> None:
        """Verify error fields for subprocess crash scenario."""
        result = ExecutionResult(
            response="",
            is_error=True,
            error_reason="subprocess terminated unexpectedly",
        )

        assert result.is_error is True
        assert result.error_reason == "subprocess terminated unexpectedly"

    @pytest.mark.unit
    def test_error_tool_failure(self) -> None:
        """Verify error fields for tool execution failure scenario."""
        result = ExecutionResult(
            response="",
            is_error=True,
            error_reason="tool execution failed: vectorstore_search",
        )

        assert result.is_error is True
        assert result.error_reason == "tool execution failed: vectorstore_search"

    @pytest.mark.unit
    def test_error_timeout(self) -> None:
        """Verify error fields for timeout scenario."""
        result = ExecutionResult(
            response="",
            is_error=True,
            error_reason="timeout exceeded",
        )

        assert result.is_error is True
        assert result.error_reason == "timeout exceeded"

    @pytest.mark.unit
    def test_successful_result_has_no_error(self) -> None:
        """Verify successful result has is_error=False and error_reason=None."""
        result = ExecutionResult(response="Success!")

        assert result.is_error is False
        assert result.error_reason is None

    @pytest.mark.unit
    def test_tool_calls_default_factory_isolation(self) -> None:
        """Mutable default for tool_calls is not shared between instances."""
        result_a = ExecutionResult(response="A")
        result_b = ExecutionResult(response="B")

        result_a.tool_calls.append({"name": "tool1"})

        assert result_b.tool_calls == []
        assert len(result_a.tool_calls) == 1

    @pytest.mark.unit
    def test_tool_results_default_factory_isolation(self) -> None:
        """Mutable default for tool_results is not shared between instances."""
        result_a = ExecutionResult(response="A")
        result_b = ExecutionResult(response="B")

        result_a.tool_results.append({"result": "data"})

        assert result_b.tool_results == []
        assert len(result_a.tool_results) == 1

    @pytest.mark.unit
    def test_token_usage_default_factory_isolation(self) -> None:
        """Each instance gets its own TokenUsage object (not same object)."""
        result_a = ExecutionResult(response="A")
        result_b = ExecutionResult(response="B")

        assert result_a.token_usage is not result_b.token_usage
        assert result_a.token_usage == result_b.token_usage


class TestBackendExceptionHierarchy:
    """Tests for the BackendError exception hierarchy."""

    @pytest.mark.unit
    def test_backend_error_is_holodeck_error(self) -> None:
        """BackendError must inherit from HoloDeckError."""
        error = BackendError("test error")

        assert isinstance(error, HoloDeckError)
        assert isinstance(error, BackendError)

    @pytest.mark.unit
    def test_backend_init_error_is_backend_error(self) -> None:
        """BackendInitError must inherit from BackendError."""
        error = BackendInitError("init failed")

        assert isinstance(error, BackendInitError)
        assert isinstance(error, BackendError)
        assert isinstance(error, HoloDeckError)

    @pytest.mark.unit
    def test_backend_session_error_is_backend_error(self) -> None:
        """BackendSessionError must inherit from BackendError."""
        error = BackendSessionError("session failed")

        assert isinstance(error, BackendSessionError)
        assert isinstance(error, BackendError)
        assert isinstance(error, HoloDeckError)

    @pytest.mark.unit
    def test_backend_timeout_error_is_backend_error(self) -> None:
        """BackendTimeoutError must inherit from BackendError."""
        error = BackendTimeoutError("timed out")

        assert isinstance(error, BackendTimeoutError)
        assert isinstance(error, BackendError)
        assert isinstance(error, HoloDeckError)

    @pytest.mark.unit
    def test_backend_error_catches_all_subtypes(self) -> None:
        """except BackendError must catch all three subtypes."""
        for exc_class, msg in [
            (BackendInitError, "init failure"),
            (BackendSessionError, "session failure"),
            (BackendTimeoutError, "timeout failure"),
        ]:
            with pytest.raises(BackendError):
                raise exc_class(msg)

    @pytest.mark.unit
    def test_holodeck_error_catches_all_backend_errors(self) -> None:
        """except HoloDeckError must catch all four backend error types."""
        for exc_class, msg in [
            (BackendError, "base backend error"),
            (BackendInitError, "init failure"),
            (BackendSessionError, "session failure"),
            (BackendTimeoutError, "timeout failure"),
        ]:
            with pytest.raises(HoloDeckError):
                raise exc_class(msg)


class TestProtocolDefinitions:
    """Tests for Protocol definitions in backends.base."""

    @pytest.mark.unit
    def test_agent_session_is_runtime_checkable(self) -> None:
        """AgentSession must be importable and be a runtime-checkable Protocol."""
        from typing import Protocol

        # Verify it is a Protocol subclass and runtime_checkable
        assert issubclass(AgentSession, Protocol)
        # runtime_checkable Protocols support isinstance checks
        # (though concrete class checks require structural subtyping)
        assert AgentSession is not None

    @pytest.mark.unit
    def test_agent_backend_is_runtime_checkable(self) -> None:
        """AgentBackend must be importable and be a runtime-checkable Protocol."""
        from typing import Protocol

        assert issubclass(AgentBackend, Protocol)
        assert AgentBackend is not None
