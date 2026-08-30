"""Payload and parameter models for the Temporal integration (spec 040, T2).

Covers the wire contract of the activity payloads (they must survive the
``pydantic_data_converter`` round trip and keep ``output`` a plain dict per
FR-008) and the workflow-side scheduling helper (decision 10: caller-side
timeouts, no heartbeat knob, at least one closing timeout).
"""

from __future__ import annotations

from datetime import timedelta

import pytest
from pydantic import BaseModel, ValidationError
from temporalio.common import RetryPolicy
from temporalio.contrib.pydantic import pydantic_data_converter

from holodeck.models.token_usage import TokenUsage
from holodeck.temporal.models import (
    ActivityParameters,
    AgentActivityInput,
    AgentActivityResult,
)

pytestmark = pytest.mark.unit


async def _round_trip(value: object, as_type: type) -> object:
    """Encode a value through the Temporal converter and decode it back."""
    payloads = await pydantic_data_converter.encode([value])
    decoded = await pydantic_data_converter.decode(payloads, [as_type])
    return decoded[0]


class TestAgentActivityInput:
    """Input payload."""

    def test_context_defaults_to_none(self):
        """The optional context uses a None sentinel, never an empty dict."""
        # Act
        payload = AgentActivityInput(message="hello")

        # Assert
        assert payload.context is None

    @pytest.mark.asyncio
    async def test_round_trips_through_pydantic_data_converter(self):
        """Input survives Temporal's Pydantic converter unchanged."""
        # Arrange
        payload = AgentActivityInput(
            message="summarize the filing", context={"ticker": "ALXN", "year": 2007}
        )

        # Act
        decoded = await _round_trip(payload, AgentActivityInput)

        # Assert
        assert decoded == payload
        assert decoded.context == {"ticker": "ALXN", "year": 2007}

    def test_non_json_context_value_is_refused_at_construction(self):
        """A non-JSON context value is a typed authoring error, refused here.

        Without this, the arbitrary object validates and then blows up inside
        the data converter while the workflow schedules the activity — a
        workflow-task failure instead of an error at the call site.
        """
        # Act / Assert
        with pytest.raises(ValidationError):
            AgentActivityInput(message="hi", context={"bad": object()})

    def test_set_context_value_is_refused_at_construction(self):
        """Sets are not JSON; refused at construction, not in the converter."""
        # Act / Assert
        with pytest.raises(ValidationError):
            AgentActivityInput(message="hi", context={"bad": {1, 2}})

    def test_nested_json_context_survives(self):
        """Nested JSON values (lists, dicts, null) validate and round-trip."""
        # Act
        payload = AgentActivityInput(
            message="hi",
            context={"a": [1, 2.5, "x", None, {"b": True}]},
        )

        # Assert
        assert payload.context == {"a": [1, 2.5, "x", None, {"b": True}]}

    def test_unknown_field_is_refused(self):
        """Extra payload fields are a contract break, not a silent pass."""
        # Act / Assert
        with pytest.raises(ValidationError):
            AgentActivityInput(message="hi", extra_field="nope")


class TestAgentActivityResult:
    """Result payload."""

    def test_output_is_a_plain_dict(self):
        """The gate-validated object lands as a plain dict (FR-008)."""
        # Arrange
        result = AgentActivityResult(
            output={"verdict": "approve", "score": 0.91}, agent_id="triage"
        )

        # Act
        output = result.output

        # Assert
        assert type(output) is dict
        assert output == {"verdict": "approve", "score": 0.91}

    def test_defaults_are_zero_usage_and_one_turn(self):
        """Unspecified usage and turn count take conservative defaults."""
        # Act
        result = AgentActivityResult(output={}, agent_id="triage")

        # Assert
        assert result.token_usage == TokenUsage.zero()
        assert result.num_turns == 1

    def test_raw_response_text_has_no_field_on_the_envelope(self):
        """The envelope carries no channel for raw model text (FR-008)."""
        # Act
        field_names = set(AgentActivityResult.model_fields)

        # Assert
        assert field_names == {"output", "token_usage", "num_turns", "agent_id"}

    @pytest.mark.asyncio
    async def test_round_trips_through_pydantic_data_converter(self):
        """Result, including nested TokenUsage, survives the converter."""
        # Arrange
        result = AgentActivityResult(
            output={"verdict": "approve", "items": [1, 2, 3], "nested": {"a": None}},
            token_usage=TokenUsage(
                prompt_tokens=120, completion_tokens=40, total_tokens=160
            ),
            num_turns=3,
            agent_id="triage",
        )

        # Act
        decoded = await _round_trip(result, AgentActivityResult)

        # Assert
        assert decoded == result
        assert type(decoded.output) is dict
        assert decoded.token_usage.total_tokens == 160

    def test_output_as_validates_into_caller_model(self) -> None:
        # Arrange
        class Verdict(BaseModel):
            eligible: bool
            reason: str

        result = AgentActivityResult(
            output={"eligible": True, "reason": "income below threshold"},
            agent_id="evidence-extractor",
        )

        # Act
        verdict = result.output_as(Verdict)

        # Assert
        assert isinstance(verdict, Verdict)
        assert verdict.eligible is True
        assert verdict.reason == "income below threshold"

    def test_output_as_raises_on_nonconforming_output(self) -> None:
        # Arrange
        class Verdict(BaseModel):
            eligible: bool

        result = AgentActivityResult(output={"wrong": 1}, agent_id="a")

        # Act / Assert
        with pytest.raises(ValidationError):
            result.output_as(Verdict)


class TestActivityParametersValidation:
    """Closing-timeout requirement and the deliberate heartbeat omission."""

    def test_neither_closing_timeout_is_refused(self):
        """Temporal has no server default, so one closing timeout is required."""
        # Act / Assert
        with pytest.raises(ValidationError) as excinfo:
            ActivityParameters(schedule_to_start=timedelta(seconds=10))

        assert "start_to_close or schedule_to_close" in str(excinfo.value)

    def test_empty_parameters_are_refused(self):
        """An all-defaults instance is not a usable scheduling option set."""
        # Act / Assert
        with pytest.raises(ValidationError):
            ActivityParameters()

    @pytest.mark.parametrize("field", ["start_to_close", "schedule_to_close"])
    def test_either_closing_timeout_alone_is_accepted(self, field):
        """Either closing timeout on its own satisfies validation."""
        # Act
        params = ActivityParameters(**{field: timedelta(minutes=5)})

        # Assert
        assert getattr(params, field) == timedelta(minutes=5)

    def test_heartbeat_timeout_is_not_a_field(self):
        """Heartbeat is unsupported in v1 (decision 10) and must not be settable."""
        # Assert
        assert "heartbeat_timeout" not in ActivityParameters.model_fields
        with pytest.raises(ValidationError):
            ActivityParameters(
                start_to_close=timedelta(minutes=5),
                heartbeat_timeout=timedelta(seconds=30),
            )


class TestActivityParametersKwargs:
    """``to_activity_kwargs`` expansion."""

    def test_only_set_timeouts_are_emitted(self):
        """Unset timeout fields are omitted so Temporal's defaults apply."""
        # Arrange
        params = ActivityParameters(start_to_close=timedelta(minutes=5))

        # Act
        kwargs = params.to_activity_kwargs()

        # Assert
        assert kwargs == {"start_to_close_timeout": timedelta(minutes=5)}

    def test_all_timeouts_map_to_execute_activity_names(self):
        """Each timeout field maps onto its ``execute_activity`` keyword."""
        # Arrange
        params = ActivityParameters(
            start_to_close=timedelta(minutes=5),
            schedule_to_close=timedelta(minutes=30),
            schedule_to_start=timedelta(seconds=45),
        )

        # Act
        kwargs = params.to_activity_kwargs()

        # Assert
        assert kwargs == {
            "start_to_close_timeout": timedelta(minutes=5),
            "schedule_to_close_timeout": timedelta(minutes=30),
            "schedule_to_start_timeout": timedelta(seconds=45),
        }
        assert all(isinstance(value, timedelta) for value in kwargs.values())

    def test_no_retry_policy_when_no_retry_field_is_set(self):
        """Absent retry fields leave the server default retry policy in place."""
        # Arrange
        params = ActivityParameters(start_to_close=timedelta(minutes=5))

        # Act
        kwargs = params.to_activity_kwargs()

        # Assert
        assert "retry_policy" not in kwargs

    def test_retry_fields_build_a_retry_policy(self):
        """Retry fields become a ``temporalio.common.RetryPolicy``."""
        # Arrange
        params = ActivityParameters(
            start_to_close=timedelta(minutes=5),
            initial_interval=timedelta(seconds=2),
            backoff_coefficient=1.5,
            maximum_interval=timedelta(seconds=30),
            maximum_attempts=4,
            non_retryable_error_types=["ConfigError"],
        )

        # Act
        policy = params.to_activity_kwargs()["retry_policy"]

        # Assert
        assert isinstance(policy, RetryPolicy)
        assert policy.initial_interval == timedelta(seconds=2)
        assert policy.backoff_coefficient == 1.5
        assert policy.maximum_interval == timedelta(seconds=30)
        assert policy.maximum_attempts == 4
        assert policy.non_retryable_error_types == ["ConfigError"]

    def test_partial_retry_policy_keeps_sdk_defaults(self):
        """Only the retry fields the caller set are passed to RetryPolicy."""
        # Arrange
        params = ActivityParameters(
            schedule_to_close=timedelta(minutes=10), maximum_attempts=3
        )

        # Act
        policy = params.to_activity_kwargs()["retry_policy"]

        # Assert
        assert policy.maximum_attempts == 3
        assert policy.initial_interval == RetryPolicy().initial_interval

    def test_kwargs_bind_to_execute_activity(self):
        """Every emitted keyword is one ``workflow.execute_activity`` accepts."""
        # Arrange
        import inspect

        from temporalio import workflow

        params = ActivityParameters(
            start_to_close=timedelta(minutes=5),
            schedule_to_close=timedelta(minutes=30),
            schedule_to_start=timedelta(seconds=45),
            maximum_attempts=2,
        )

        # Act
        signature = inspect.signature(workflow.execute_activity)
        bound = signature.bind("activity-name", **params.to_activity_kwargs())

        # Assert
        assert set(params.to_activity_kwargs()) <= set(signature.parameters)
        assert bound.arguments["retry_policy"].maximum_attempts == 2


class TestSandboxSafety:
    """The module ships to workflow code with the D3 surface."""

    def test_module_imports_no_worker_or_client_modules(self):
        """Only sandbox-safe temporalio imports belong in the module."""
        # Arrange
        import holodeck.temporal.models as models_module

        source = models_module.__file__
        assert source is not None

        # Act
        with open(source, encoding="utf-8") as handle:
            lines = [
                line
                for line in handle
                if line.startswith(("import ", "from ")) and "temporalio" in line
            ]

        # Assert
        assert lines == ["from temporalio.common import RetryPolicy\n"]
