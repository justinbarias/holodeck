"""Tests for TestCase model in holodeck.models.test_case."""

import pytest
from pydantic import ValidationError

from holodeck.models.evaluation import EvaluationMetric
from holodeck.models.test_case import FileInput, TestCase


class TestFileInput:
    """Tests for FileInput model."""

    def test_file_input_with_local_path(self) -> None:
        """Test creating FileInput with local path."""
        file_input = FileInput(
            path="data/document.pdf",
            type="pdf",
        )
        assert file_input.path == "data/document.pdf"
        assert file_input.type == "pdf"
        assert file_input.url is None

    def test_file_input_with_url(self) -> None:
        """Test creating FileInput with URL."""
        file_input = FileInput(
            url="https://example.com/document.pdf",
            type="pdf",
        )
        assert file_input.url == "https://example.com/document.pdf"
        assert file_input.type == "pdf"
        assert file_input.path is None

    def test_file_input_type_required(self) -> None:
        """Test that type field is required."""
        with pytest.raises(ValidationError) as exc_info:
            FileInput(path="data.txt")
        assert "type" in str(exc_info.value).lower()

    def test_file_input_path_and_url_mutually_exclusive(self) -> None:
        """Test that path and url are mutually exclusive."""
        with pytest.raises(ValidationError):
            FileInput(
                path="data.txt",
                url="https://example.com/data.txt",
                type="text",
            )

    def test_file_input_path_or_url_required(self) -> None:
        """Test that either path or url is required."""
        with pytest.raises(ValidationError):
            FileInput(type="text")

    @pytest.mark.parametrize(
        "field,file_type,path_or_url,value,expected",
        [
            (
                "description",
                "text",
                {"path": "data.txt"},
                "Test data file",
                "Test data file",
            ),
            ("pages", "pdf", {"path": "document.pdf"}, [1, 2, 3], [1, 2, 3]),
            ("sheet", "excel", {"path": "data.xlsx"}, "Sheet1", "Sheet1"),
            ("range", "excel", {"path": "data.xlsx"}, "A1:E100", "A1:E100"),
            ("cache", "text", {"url": "https://example.com/data.txt"}, True, True),
        ],
        ids=["description", "pages", "sheet", "range", "cache"],
    )
    def test_file_input_optional_fields_with_value(
        self,
        field: str,
        file_type: str,
        path_or_url: dict,
        value,
        expected,
    ) -> None:
        """Test FileInput optional fields can be set."""
        kwargs = {**path_or_url, "type": file_type, field: value}
        file_input = FileInput(**kwargs)
        assert getattr(file_input, field) == expected

    @pytest.mark.parametrize(
        "field,file_type,path_or_url",
        [
            ("description", "text", {"path": "data.txt"}),
            ("pages", "pdf", {"path": "document.pdf"}),
            ("sheet", "excel", {"path": "data.xlsx"}),
            ("range", "excel", {"path": "data.xlsx"}),
            ("cache", "text", {"path": "data.txt"}),
        ],
        ids=["description", "pages", "sheet", "range", "cache"],
    )
    def test_file_input_optional_fields_default(
        self, field: str, file_type: str, path_or_url: dict
    ) -> None:
        """Test FileInput optional fields default to None or expected value."""
        kwargs = {**path_or_url, "type": file_type}
        file_input = FileInput(**kwargs)
        value = getattr(file_input, field)
        # cache can be None or False by default, others should be None
        if field == "cache":
            assert value is None or isinstance(value, bool)
        else:
            assert value is None

    def test_file_input_valid_types(self) -> None:
        """Test FileInput accepts valid file types."""
        for file_type in ["image", "pdf", "text", "excel", "word", "powerpoint", "csv"]:
            file_input = FileInput(
                path="file",
                type=file_type,
            )
            assert file_input.type == file_type


class TestTestCase:
    """Tests for TestCase model."""

    def test_test_case_valid_creation(self) -> None:
        """Test creating a valid TestCase."""
        test_case = TestCase(
            input="What is the weather?",
        )
        assert test_case.input == "What is the weather?"
        assert test_case.expected_tools is None
        assert test_case.ground_truth is None

    def test_test_case_input_required(self) -> None:
        """Test that input field is required."""
        with pytest.raises(ValidationError) as exc_info:
            TestCase()
        assert "input" in str(exc_info.value).lower()

    def test_test_case_input_not_empty(self) -> None:
        """Test that input cannot be empty string."""
        with pytest.raises(ValidationError):
            TestCase(input="")

    @pytest.mark.parametrize(
        "field,value,expected_check",
        [
            ("name", "Test 1", lambda v: v == "Test 1"),
            (
                "expected_tools",
                ["search_tool", "rank_tool"],
                lambda v: v == ["search_tool", "rank_tool"],
            ),
            ("ground_truth", "4", lambda v: v == "4"),
            (
                "evaluations",
                [
                    EvaluationMetric(metric="groundedness", threshold=0.7),
                    EvaluationMetric(metric="relevance", threshold=0.8),
                ],
                lambda v: len(v) == 2
                and v[0].metric == "groundedness"
                and v[1].metric == "relevance",
            ),
        ],
        ids=["name", "expected_tools", "ground_truth", "evaluations"],
    )
    def test_test_case_optional_fields_with_value(
        self, field: str, value, expected_check
    ) -> None:
        """Test TestCase optional fields can be set."""
        kwargs = {"input": "Test input", field: value}
        test_case = TestCase(**kwargs)
        assert expected_check(getattr(test_case, field))

    @pytest.mark.parametrize(
        "field",
        ["name", "expected_tools", "ground_truth", "evaluations", "files"],
        ids=["name", "expected_tools", "ground_truth", "evaluations", "files"],
    )
    def test_test_case_optional_fields_default(self, field: str) -> None:
        """Test TestCase optional fields default to None."""
        test_case = TestCase(input="Test")
        assert getattr(test_case, field) is None

    def test_test_case_with_files(self) -> None:
        """Test TestCase with files."""
        file_input = FileInput(
            path="data.pdf",
            type="pdf",
        )
        test_case = TestCase(
            input="Analyze this document",
            files=[file_input],
        )
        assert len(test_case.files) == 1
        assert test_case.files[0].path == "data.pdf"

    def test_test_case_all_fields(self) -> None:
        """Test TestCase with all optional fields."""
        file_input = FileInput(
            path="document.pdf",
            type="pdf",
        )
        test_case = TestCase(
            name="Test case 1",
            input="Process document",
            expected_tools=["extractor"],
            ground_truth="Expected output",
            files=[file_input],
            evaluations=[
                EvaluationMetric(metric="groundedness", threshold=0.7),
                EvaluationMetric(metric="relevance", threshold=0.8),
            ],
        )
        assert test_case.name == "Test case 1"
        assert test_case.input == "Process document"
        assert test_case.expected_tools == ["extractor"]
        assert test_case.ground_truth == "Expected output"
        assert len(test_case.files) == 1
        assert len(test_case.evaluations) == 2
        assert test_case.evaluations[0].metric == "groundedness"
        assert test_case.evaluations[1].metric == "relevance"

    def test_test_case_max_input_length(self) -> None:
        """Test that long inputs are accepted (up to reasonable limit)."""
        long_input = "x" * 5000
        test_case = TestCase(input=long_input)
        assert test_case.input == long_input
