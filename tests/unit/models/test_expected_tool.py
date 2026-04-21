"""Tests for ExpectedTool and ArgMatcher parsing (US3 T002–T004, T009)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from holodeck.models.test_case import (
    ExpectedTool,
    FuzzyMatcher,
    LiteralMatcher,
    TestCaseModel,
    Turn,
)


@pytest.mark.unit
class TestExpectedToolLegacyString:
    """Bare-string form must still parse (FR-024)."""

    def test_bare_string_parses_as_legacy(self) -> None:
        # Turn with the bare-string form — normalized to ExpectedTool
        # internally but visible here as a round-trippable form.
        turn = Turn(input="q", expected_tools=["subtract"])
        assert turn.expected_tools is not None
        # One element, either as-is string or as promoted ExpectedTool with
        # name-only and count==1.
        first = turn.expected_tools[0]
        if isinstance(first, str):
            assert first == "subtract"
        else:
            assert isinstance(first, ExpectedTool)
            assert first.name == "subtract"
            assert first.args is None
            assert first.count == 1


@pytest.mark.unit
class TestExpectedToolObjectForm:
    """Full object form with args + count (T003)."""

    def test_object_form_full(self) -> None:
        turn = Turn(
            input="q",
            expected_tools=[
                {
                    "name": "subtract",
                    "args": {"a": 206588, "b": {"fuzzy": "181001"}},
                    "count": 2,
                }
            ],
        )
        assert turn.expected_tools is not None
        only = turn.expected_tools[0]
        assert isinstance(only, ExpectedTool)
        assert only.name == "subtract"
        assert only.count == 2
        assert only.args is not None
        assert isinstance(only.args["a"], LiteralMatcher)
        assert only.args["a"].value == 206588
        assert isinstance(only.args["b"], FuzzyMatcher)
        assert only.args["b"].pattern == "181001"

    def test_count_defaults_to_one(self) -> None:
        et = ExpectedTool(name="subtract")
        assert et.count == 1
        assert et.args is None

    def test_count_rejects_zero_and_negative(self) -> None:
        with pytest.raises(ValidationError):
            ExpectedTool(name="subtract", count=0)
        with pytest.raises(ValidationError):
            ExpectedTool(name="subtract", count=-1)

    def test_name_rejects_empty(self) -> None:
        with pytest.raises(ValidationError):
            ExpectedTool(name="")
        with pytest.raises(ValidationError):
            ExpectedTool(name="   ")


@pytest.mark.unit
class TestMixedListSupport:
    """Mix of str and object entries (FR-003, T009)."""

    def test_mixed_str_and_object_expected_tools(self) -> None:
        tc = TestCaseModel(
            name="mix",
            turns=[
                Turn(
                    input="q",
                    expected_tools=[
                        "lookup",
                        {"name": "subtract", "args": {"a": 1}},
                    ],
                )
            ],
        )
        assert tc.turns is not None
        et = tc.turns[0].expected_tools
        assert et is not None
        assert len(et) == 2
        # The object entry should be an ExpectedTool with args
        found_subtract = False
        for entry in et:
            if isinstance(entry, ExpectedTool) and entry.name == "subtract":
                assert entry.args is not None
                assert isinstance(entry.args["a"], LiteralMatcher)
                found_subtract = True
        assert found_subtract
