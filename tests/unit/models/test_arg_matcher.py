"""Tests for ArgMatcher discriminator (US3 T005–T008)."""

from __future__ import annotations

import re

import pytest

from holodeck.lib.errors import ConfigError
from holodeck.models.test_case import (
    ExpectedTool,
    FuzzyMatcher,
    LiteralMatcher,
    RegexMatcher,
)


@pytest.mark.unit
class TestLiteralRouting:
    """Scalars, lists, dicts go to LiteralMatcher."""

    def test_literal_scalar_list_dict(self) -> None:
        et = ExpectedTool(
            name="tool",
            args={
                "scalar_int": 5,
                "scalar_float": 1.5,
                "scalar_str": "hello",
                "scalar_bool": True,
                "scalar_none": None,
                "list_val": [1, 2, 3],
                "dict_val": {"mode": "fast"},
            },
        )
        assert et.args is not None
        for key in et.args:
            assert isinstance(
                et.args[key], LiteralMatcher
            ), f"{key} should be LiteralMatcher"
        assert et.args["scalar_int"].value == 5
        assert et.args["list_val"].value == [1, 2, 3]
        assert et.args["dict_val"].value == {"mode": "fast"}


@pytest.mark.unit
class TestFuzzyAndRegexShapes:
    """{fuzzy:...} and {regex:...} dicts coerce to matcher kinds."""

    def test_fuzzy_shape_parses(self) -> None:
        et = ExpectedTool(name="tool", args={"a": {"fuzzy": "206588"}})
        assert et.args is not None
        assert isinstance(et.args["a"], FuzzyMatcher)
        assert et.args["a"].pattern == "206588"

    def test_regex_shape_parses_and_compiles(self) -> None:
        et = ExpectedTool(name="tool", args={"a": {"regex": "^foo$"}})
        assert et.args is not None
        assert isinstance(et.args["a"], RegexMatcher)
        assert isinstance(et.args["a"].compiled, re.Pattern)
        assert et.args["a"].compiled.fullmatch("foo") is not None
        assert et.args["a"].compiled.fullmatch("bar") is None


@pytest.mark.unit
class TestMalformedMatchers:
    """Reject bad shapes at config load (FR-025)."""

    def test_both_matcher_keys_rejected(self) -> None:
        with pytest.raises((ConfigError, ValueError)):
            ExpectedTool(name="tool", args={"a": {"fuzzy": "x", "regex": "y"}})

    def test_unknown_matcher_key_treated_as_literal(self) -> None:
        """Matrix row 22: a dict without `fuzzy`/`regex` keys is a literal
        dict, not a malformed matcher. `{foo:"x"}` routes to LiteralMatcher.

        Deviation from tasks-us3.md T007 wording — the authoritative
        contracts/tool-arg-matchers.md §7 row 22 (`{mode:"fast"}` as literal
        dict) prevents us from rejecting these. Malformed matchers are
        caught when a dict *does* contain a matcher key but also has other
        keys (see test_extra_key_alongside_matcher_rejected below).
        """
        et = ExpectedTool(name="tool", args={"a": {"foo": "x"}})
        assert et.args is not None
        assert isinstance(et.args["a"], LiteralMatcher)
        assert et.args["a"].value == {"foo": "x"}

    def test_extra_key_alongside_matcher_rejected(self) -> None:
        """A dict mixing a matcher key and an extra key is malformed."""
        with pytest.raises((ConfigError, ValueError)):
            ExpectedTool(name="tool", args={"a": {"fuzzy": "x", "extra": 1}})

    def test_bad_regex_rejected_at_load(self) -> None:
        with pytest.raises((ConfigError, ValueError)):
            ExpectedTool(name="tool", args={"a": {"regex": "("}})
