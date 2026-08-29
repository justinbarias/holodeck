"""Payload and parameter models for the Temporal integration (spec 040).

Two kinds of model live here:

* **Payloads** — :class:`AgentActivityInput` and :class:`AgentActivityResult`
  cross the Temporal wire and are encoded by
  ``temporalio.contrib.pydantic.pydantic_data_converter``. Every field is
  JSON-serializable. Per FR-008 the gate-validated object is the canonical
  value, so the result carries a plain ``dict`` and never the raw model text.
* **Parameters** — :class:`ActivityParameters` never crosses the wire. It is a
  workflow-side helper (decision 10) that expands into ``execute_activity``
  keyword arguments.

This module is imported by workflow code and must stay sandbox-safe: only
``temporalio.common`` (a passthrough module) and pure-Python imports belong
here — never ``temporalio.worker``, ``temporalio.client``, or
``temporalio.activity``.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator
from temporalio.common import RetryPolicy

from holodeck.models.token_usage import TokenUsage


class AgentActivityInput(BaseModel):
    """Input payload for a HoloDeck agent activity.

    Attributes:
        message: The user-facing message handed to the agent for this turn.
        context: Optional caller-supplied context object, passed through to
            the agent. ``None`` when the caller supplies nothing — never an
            empty dict sentinel.
    """

    model_config = ConfigDict(extra="forbid")

    message: str
    context: dict[str, Any] | None = None


class AgentActivityResult(BaseModel):
    """Result payload returned by a HoloDeck agent activity.

    Attributes:
        output: The gate-validated object produced by the agent. This is the
            canonical value for downstream consumers (FR-008); the raw model
            response text is deliberately absent from the envelope.
        token_usage: Token consumption for the turn, mirroring
            ``ExecutionResult.token_usage``.
        num_turns: Number of agent turns taken to produce ``output``.
        agent_id: Identifier of the agent (the edge node id) that ran.
    """

    model_config = ConfigDict(extra="forbid")

    output: dict[str, Any]
    token_usage: TokenUsage = Field(default_factory=TokenUsage.zero)
    num_turns: int = Field(default=1, ge=0)
    agent_id: str


class ActivityParameters(BaseModel):
    """Workflow-side scheduling options for an agent activity.

    Temporal carries timeouts and the retry policy on the workflow's
    ``execute_activity`` command rather than on the activity definition
    (decision 10), so this model is a caller-side helper and is never sent
    over the wire.

    Temporal has no server-side default for the closing timeouts, so an
    instance must set at least one of ``start_to_close`` or
    ``schedule_to_close``.

    ``heartbeat_timeout`` is deliberately unsupported in v1: the agent
    activity does not heartbeat, and exposing the knob without heartbeats
    invites concurrent duplicate LLM calls.

    Attributes:
        start_to_close: Maximum duration of a single activity attempt.
        schedule_to_close: Maximum duration of the activity including retries
            and queue time.
        schedule_to_start: Maximum time the activity may sit on the task queue
            before a worker picks it up.
        initial_interval: First retry backoff interval.
        backoff_coefficient: Multiplier applied to the interval per retry.
        maximum_interval: Cap on the retry backoff interval.
        maximum_attempts: Total attempts before the activity fails; ``0`` means
            unlimited, matching ``temporalio.common.RetryPolicy``.
        non_retryable_error_types: Error type names (matched by string against
            the exception class name) that must not be retried.
    """

    model_config = ConfigDict(extra="forbid")

    start_to_close: timedelta | None = None
    schedule_to_close: timedelta | None = None
    schedule_to_start: timedelta | None = None

    initial_interval: timedelta | None = None
    backoff_coefficient: float | None = None
    maximum_interval: timedelta | None = None
    maximum_attempts: int | None = None
    non_retryable_error_types: list[str] | None = None

    @model_validator(mode="after")
    def _require_a_closing_timeout(self) -> ActivityParameters:
        """Refuse an instance with neither closing timeout.

        Returns:
            The validated model.

        Raises:
            ValueError: If both ``start_to_close`` and ``schedule_to_close``
                are unset. Pydantic surfaces this as a ``ValidationError``.
        """
        if self.start_to_close is None and self.schedule_to_close is None:
            raise ValueError(
                "ActivityParameters requires start_to_close or schedule_to_close: "
                "Temporal has no server-side default for either timeout."
            )
        return self

    def _retry_policy(self) -> RetryPolicy | None:
        """Build the retry policy, or ``None`` when no retry field is set.

        Returns:
            A ``RetryPolicy`` carrying only the fields the caller set, or
            ``None`` so the server default policy applies.
        """
        fields: dict[str, Any] = {
            "initial_interval": self.initial_interval,
            "backoff_coefficient": self.backoff_coefficient,
            "maximum_interval": self.maximum_interval,
            "maximum_attempts": self.maximum_attempts,
            "non_retryable_error_types": self.non_retryable_error_types,
        }
        set_fields = {
            name: value for name, value in fields.items() if value is not None
        }
        if not set_fields:
            return None
        return RetryPolicy(**set_fields)

    def to_activity_kwargs(self) -> dict[str, Any]:
        """Expand into keyword arguments for ``workflow.execute_activity``.

        Unset fields are omitted so Temporal's own defaults apply.

        Returns:
            A mapping with any of ``start_to_close_timeout``,
            ``schedule_to_close_timeout``, ``schedule_to_start_timeout`` (all
            ``timedelta``) and ``retry_policy`` (a
            ``temporalio.common.RetryPolicy``).
        """
        kwargs: dict[str, Any] = {}
        if self.start_to_close is not None:
            kwargs["start_to_close_timeout"] = self.start_to_close
        if self.schedule_to_close is not None:
            kwargs["schedule_to_close_timeout"] = self.schedule_to_close
        if self.schedule_to_start is not None:
            kwargs["schedule_to_start_timeout"] = self.schedule_to_start
        retry_policy = self._retry_policy()
        if retry_policy is not None:
            kwargs["retry_policy"] = retry_policy
        return kwargs


__all__ = ["ActivityParameters", "AgentActivityInput", "AgentActivityResult"]
