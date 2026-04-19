"""Unit tests for the eval-run slugifier (T014).

Implements research.md R4 rules:
- lowercase
- alphanumerics + hyphen
- non-ASCII → hyphen
- spaces → hyphen
- consecutive hyphens collapsed
- leading/trailing hyphens stripped
- empty result raises ValueError
"""

from __future__ import annotations

import pytest

from holodeck.lib.eval_run.slugify import slugify


@pytest.mark.unit
class TestSlugify:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("MyAgent", "myagent"),
            ("My Agent", "my-agent"),
            ("My  Agent", "my-agent"),
            ("Agent_v1.0", "agent-v1-0"),
            ("café-bot", "caf-bot"),
            ("---hello---", "hello"),
            ("a___b", "a-b"),
            ("Already-Slug", "already-slug"),
            ("a/b\\c:d", "a-b-c-d"),
        ],
    )
    def test_slugify_cases(self, raw: str, expected: str):
        assert slugify(raw) == expected

    def test_collapses_repeated_hyphens(self):
        assert slugify("foo----bar") == "foo-bar"

    def test_strips_leading_trailing_hyphens(self):
        assert slugify("-foo-bar-") == "foo-bar"

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            slugify("")

    def test_only_separators_raises(self):
        with pytest.raises(ValueError):
            slugify("___---***")
