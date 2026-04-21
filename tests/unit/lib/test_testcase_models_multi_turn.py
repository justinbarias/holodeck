"""Tests for TestCaseModel multi-turn validation rules (T004).

Covers mutual-exclusion between legacy single-turn fields and the new
`turns` list, empty-turn rejection, and top-level file/retrieval_context
rejection when `turns` is present (contracts/test-case-schema.md §2).
"""

import pytest
from pydantic import ValidationError

from holodeck.models.test_case import TestCaseModel, Turn


@pytest.mark.unit
class TestMultiTurnTestCaseValidation:
    def test_turns_with_input_raises(self) -> None:
        with pytest.raises(ValidationError):
            TestCaseModel(input="hi", turns=[Turn(input="hi")])

    def test_turns_with_ground_truth_raises(self) -> None:
        with pytest.raises(ValidationError):
            TestCaseModel(ground_truth="yes", turns=[Turn(input="hi")])

    def test_turns_with_top_level_expected_tools_raises(self) -> None:
        with pytest.raises(ValidationError):
            TestCaseModel(expected_tools=["foo"], turns=[Turn(input="hi")])

    def test_empty_turns_raises(self) -> None:
        with pytest.raises(ValidationError):
            TestCaseModel(turns=[])

    def test_legacy_single_turn_still_parses(self) -> None:
        case = TestCaseModel(input="hello")
        assert case.input == "hello"
        assert case.turns is None

    def test_turns_only_is_valid(self) -> None:
        case = TestCaseModel(turns=[Turn(input="a"), Turn(input="b")])
        assert case.turns is not None
        assert len(case.turns) == 2
        assert case.input is None

    def test_turns_with_top_level_files_raises(self) -> None:
        from holodeck.models.test_case import FileInput

        with pytest.raises(ValidationError):
            TestCaseModel(
                turns=[Turn(input="hi")],
                files=[FileInput(path="foo.pdf", type="pdf")],
            )

    def test_turns_with_top_level_retrieval_context_raises(self) -> None:
        with pytest.raises(ValidationError):
            TestCaseModel(
                turns=[Turn(input="hi")],
                retrieval_context=["fact 1"],
            )
