"""Tests for the Turn model (multi-turn test cases, T003)."""

import pytest
from pydantic import ValidationError

from holodeck.models.test_case import Turn


@pytest.mark.unit
class TestTurnModel:
    """Lock the Turn shape used by multi-turn test cases."""

    def test_round_trip_input_only(self) -> None:
        turn = Turn(input="hi")
        rehydrated = Turn.model_validate_json(turn.model_dump_json())
        assert rehydrated.input == "hi"
        assert rehydrated.ground_truth is None
        assert rehydrated.expected_tools is None
        assert rehydrated.files is None
        assert rehydrated.retrieval_context is None
        assert rehydrated.evaluations is None

    def test_empty_input_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Turn(input="")
        with pytest.raises(ValidationError):
            Turn(input="   ")

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            Turn(input="hi", unknown_field="nope")  # type: ignore[call-arg]

    def test_empty_ground_truth_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Turn(input="hi", ground_truth="")

    def test_turn_config_round_trip(self) -> None:
        turn = Turn(
            input="x",
            turn_config={"turn_program": "subtract(a,b)"},
        )
        assert turn.turn_config == {"turn_program": "subtract(a,b)"}
        rehydrated = Turn.model_validate_json(turn.model_dump_json())
        assert rehydrated.turn_config == {"turn_program": "subtract(a,b)"}

    def test_turn_config_defaults_to_none(self) -> None:
        turn = Turn(input="hi")
        assert turn.turn_config is None

    def test_unknown_top_level_key_still_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Turn(input="hi", turn_program="subtract(a,b)")  # type: ignore[call-arg]
