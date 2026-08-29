"""Retry classification at the activity boundary (spec 040, T4).

SC-003's model-fault vs authoring-fault split, translated to Temporal:

* Gate rejections (``GateValidationError``) and broken invocations
  (``ExecutionError``) stay plain exceptions — Temporal converts them into
  retryable ``ApplicationError`` s typed by class name, and the class-name
  string is the contract ``RetryPolicy(non_retryable_error_types=[...])``
  matches against.
* Authoring faults (``ConfigError``, ``GateSchemaError``) cross the boundary
  as ``ApplicationError(non_retryable=True)`` typed by the original class
  name.

The channels must never mix: a gate rejection is never non-retryable and an
authoring fault never surfaces as a plain exception.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from temporalio.exceptions import ApplicationError

import holodeck.temporal.activity as activity_module
from holodeck.lib.errors import (
    ConfigError,
    ExecutionError,
    GateSchemaError,
    GateValidationError,
)
from holodeck.models.workflow import EdgeNode
from holodeck.temporal.activity import agent_activity
from holodeck.temporal.models import AgentActivityInput

from .test_activity_factory import (
    RAW_TEXT,
    VALID_OUTPUT,
    _install_backend,
    _result,
)

# Reuse the factory suite's fixtures (base_dir, node, forbid_real_backends) so
# both suites classify errors for the exact same worker layout.
from .test_activity_factory import base_dir as base_dir  # noqa: F401
from .test_activity_factory import forbid_real_backends as forbid_real_backends
from .test_activity_factory import node as node  # noqa: F401

pytestmark = pytest.mark.unit

__all__ = ["base_dir", "forbid_real_backends", "node"]


class TestRetryableChannel:
    """Model and transport faults stay plain exceptions (retryable)."""

    @pytest.mark.asyncio
    async def test_gate_rejection_is_retryable_and_carries_detail(
        self, monkeypatch: pytest.MonkeyPatch, base_dir: Path, node: EdgeNode
    ) -> None:
        """Free text is a gate rejection: plain, typed by class name, detailed."""
        # Arrange
        _install_backend(monkeypatch, _result(structured_output=None))
        fn = agent_activity(node, base_dir)

        # Act / Assert
        with pytest.raises(GateValidationError) as excinfo:
            await fn(AgentActivityInput(message="extract the evidence"))
        assert not isinstance(excinfo.value, ApplicationError)
        assert excinfo.value.node_id == "evidence"
        assert "gate rejected output" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_schema_invalid_output_is_retryable(
        self, monkeypatch: pytest.MonkeyPatch, base_dir: Path, node: EdgeNode
    ) -> None:
        """Output the gate refuses is evidence about the model — retryable."""
        # Arrange
        bad = {"net_income": "not-a-number", "residency_status": "verified"}
        _install_backend(monkeypatch, _result(structured_output=bad))
        fn = agent_activity(node, base_dir)

        # Act / Assert
        with pytest.raises(GateValidationError) as excinfo:
            await fn(AgentActivityInput(message="extract the evidence"))
        assert not isinstance(excinfo.value, ApplicationError)

    @pytest.mark.asyncio
    async def test_raising_invocation_is_retryable(
        self, monkeypatch: pytest.MonkeyPatch, base_dir: Path, node: EdgeNode
    ) -> None:
        """A transport failure stays a plain ExecutionError — retryable."""
        # Arrange
        _install_backend(
            monkeypatch,
            _result(structured_output=VALID_OUTPUT),
            invoke_error=ConnectionError("api unreachable"),
        )
        fn = agent_activity(node, base_dir)

        # Act / Assert
        with pytest.raises(ExecutionError) as excinfo:
            await fn(AgentActivityInput(message="extract the evidence"))
        assert not isinstance(excinfo.value, ApplicationError)

    @pytest.mark.asyncio
    async def test_error_result_without_output_is_retryable(
        self, monkeypatch: pytest.MonkeyPatch, base_dir: Path, node: EdgeNode
    ) -> None:
        """is_error with nothing to judge is an ExecutionError — retryable."""
        # Arrange
        _install_backend(
            monkeypatch,
            _result(
                structured_output=None,
                response="",
                is_error=True,
                error_reason="model overloaded",
            ),
        )
        fn = agent_activity(node, base_dir)

        # Act / Assert
        with pytest.raises(ExecutionError) as excinfo:
            await fn(AgentActivityInput(message="extract the evidence"))
        assert not isinstance(excinfo.value, ApplicationError)


class TestNonRetryableChannel:
    """Authoring faults cross as ApplicationError(non_retryable=True)."""

    @pytest.mark.asyncio
    async def test_runtime_gate_schema_fault_is_non_retryable(
        self, monkeypatch: pytest.MonkeyPatch, base_dir: Path, node: EdgeNode
    ) -> None:
        """A per-call GateSchemaError is an authoring fault — never retried."""
        # Arrange
        _install_backend(monkeypatch, _result(structured_output=VALID_OUTPUT))
        fn = agent_activity(node, base_dir)

        def _broken_gate(*args: Any, **kwargs: Any) -> Any:
            raise GateSchemaError("evidence", "schema became unusable")

        # Patch the module object, not the dotted string: the import-guard suite
        # re-imports holodeck.temporal fresh, and on the same xdist worker the
        # package attribute 'activity' may be absent when a string path resolves.
        monkeypatch.setattr(activity_module, "_apply_gate", _broken_gate)

        # Act / Assert
        with pytest.raises(ApplicationError) as excinfo:
            await fn(AgentActivityInput(message="extract the evidence"))
        assert excinfo.value.non_retryable is True
        assert excinfo.value.type == "GateSchemaError"
        assert "schema became unusable" in str(excinfo.value)
        assert isinstance(excinfo.value.__cause__, GateSchemaError)

    @pytest.mark.asyncio
    async def test_runtime_config_fault_is_non_retryable(
        self, monkeypatch: pytest.MonkeyPatch, base_dir: Path, node: EdgeNode
    ) -> None:
        """A per-call ConfigError is an authoring fault — never retried."""
        # Arrange
        _install_backend(monkeypatch, _result(structured_output=VALID_OUTPUT))
        fn = agent_activity(node, base_dir)

        async def _broken_select(*args: Any, **kwargs: Any) -> Any:
            raise ConfigError("model.provider", "no credentials configured")

        monkeypatch.setattr(
            "holodeck.lib.backends.selector.BackendSelector.select", _broken_select
        )

        # Act / Assert
        with pytest.raises(ApplicationError) as excinfo:
            await fn(AgentActivityInput(message="extract the evidence"))
        assert excinfo.value.non_retryable is True
        assert excinfo.value.type == "ConfigError"


class TestChannelsNeverMix:
    """SC-003: the model-fault and authoring-fault channels stay separate."""

    @pytest.mark.asyncio
    async def test_gate_rejection_never_becomes_non_retryable(
        self, monkeypatch: pytest.MonkeyPatch, base_dir: Path, node: EdgeNode
    ) -> None:
        """The free-text rejection carries no non_retryable marking anywhere."""
        # Arrange
        _install_backend(
            monkeypatch, _result(structured_output=None, response=RAW_TEXT)
        )
        fn = agent_activity(node, base_dir)

        # Act
        with pytest.raises(GateValidationError) as excinfo:
            await fn(AgentActivityInput(message="extract the evidence"))

        # Assert — no ApplicationError anywhere in the cause chain
        exc: BaseException | None = excinfo.value
        while exc is not None:
            assert not isinstance(exc, ApplicationError)
            exc = exc.__cause__

    def test_class_name_contract_is_stable(self) -> None:
        """RetryPolicy matches by class-name string; pin the public names."""
        assert GateValidationError.__name__ == "GateValidationError"
        assert ExecutionError.__name__ == "ExecutionError"
        assert ConfigError.__name__ == "ConfigError"
        assert GateSchemaError.__name__ == "GateSchemaError"
