"""Round-trip and error-path tests for ExpectedTool (US3 T014a, T014b)."""

from __future__ import annotations

import pytest

from holodeck.lib.errors import ConfigError
from holodeck.models.test_case import TestCaseModel


@pytest.mark.unit
def test_bare_string_round_trip_preserves_str_on_dump() -> None:
    """FR-024 / SC-002 — legacy list[str] JSON output must not regress."""
    tc = TestCaseModel(input="x", expected_tools=["lookup"])
    dumped = tc.model_dump()
    assert dumped["expected_tools"] == ["lookup"]


@pytest.mark.unit
def test_bad_regex_surfaces_full_field_path() -> None:
    """Loading a bad regex yields a ConfigError with the test-case name and
    the full field path under `turns[0].expected_tools[0].args.a.regex`.
    """
    with pytest.raises((ConfigError, ValueError)) as exc_info:
        TestCaseModel(
            name="tc1",
            turns=[
                {
                    "input": "x",
                    "expected_tools": [
                        {"name": "x", "args": {"a": {"regex": "("}}},
                    ],
                }
            ],
        )
    msg = str(exc_info.value)
    assert "tc1" in msg
    assert "turns[0].expected_tools[0].args.a.regex" in msg
