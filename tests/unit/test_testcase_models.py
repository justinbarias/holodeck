"""Tests for TestCase model in holodeck.models.test_case."""

import pytest
from pydantic import ValidationError

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

    def test_file_input_description_optional(self) -> None:
        """Test that description is optional."""
        file_input = FileInput(
            path="data.txt",
            type="text",
        )
        assert file_input.description is None

    def test_file_input_with_description(self) -> None:
        """Test FileInput with description."""
        file_input = FileInput(
            path="data.txt",
            type="text",
            description="Test data file",
        )
        assert file_input.description == "Test data file"

    def test_file_input_pages_optional(self) -> None:
        """Test that pages is optional."""
        file_input = FileInput(
            path="document.pdf",
            type="pdf",
        )
        assert file_input.pages is None

    def test_file_input_with_pages(self) -> None:
        """Test FileInput with specific pages."""
        file_input = FileInput(
            path="document.pdf",
            type="pdf",
            pages=[1, 2, 3],
        )
        assert file_input.pages == [1, 2, 3]

    def test_file_input_sheet_optional(self) -> None:
        """Test that sheet is optional."""
        file_input = FileInput(
            path="data.xlsx",
            type="excel",
        )
        assert file_input.sheet is None

    def test_file_input_with_sheet(self) -> None:
        """Test FileInput with Excel sheet."""
        file_input = FileInput(
            path="data.xlsx",
            type="excel",
            sheet="Sheet1",
        )
        assert file_input.sheet == "Sheet1"

    def test_file_input_range_optional(self) -> None:
        """Test that range is optional."""
        file_input = FileInput(
            path="data.xlsx",
            type="excel",
        )
        assert file_input.range is None

    def test_file_input_with_range(self) -> None:
        """Test FileInput with Excel range."""
        file_input = FileInput(
            path="data.xlsx",
            type="excel",
            range="A1:E100",
        )
        assert file_input.range == "A1:E100"

    def test_file_input_cache_optional(self) -> None:
        """Test that cache is optional."""
        file_input = FileInput(
            path="data.txt",
            type="text",
        )
        assert file_input.cache is None or isinstance(file_input.cache, bool)

    def test_file_input_cache_with_url(self) -> None:
        """Test FileInput cache with URL."""
        file_input = FileInput(
            url="https://example.com/data.txt",
            type="text",
            cache=True,
        )
        assert file_input.cache is True

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

    def test_test_case_name_optional(self) -> None:
        """Test that name is optional."""
        test_case = TestCase(input="Test")
        assert test_case.name is None

    def test_test_case_with_name(self) -> None:
        """Test TestCase with name."""
        test_case = TestCase(
            name="Test 1",
            input="Test input",
        )
        assert test_case.name == "Test 1"

    def test_test_case_expected_tools_optional(self) -> None:
        """Test that expected_tools is optional."""
        test_case = TestCase(input="Test")
        assert test_case.expected_tools is None

    def test_test_case_with_expected_tools(self) -> None:
        """Test TestCase with expected_tools."""
        test_case = TestCase(
            input="Search for documents",
            expected_tools=["search_tool", "rank_tool"],
        )
        assert test_case.expected_tools == ["search_tool", "rank_tool"]

    def test_test_case_ground_truth_optional(self) -> None:
        """Test that ground_truth is optional."""
        test_case = TestCase(input="Test")
        assert test_case.ground_truth is None

    def test_test_case_with_ground_truth(self) -> None:
        """Test TestCase with ground_truth."""
        test_case = TestCase(
            input="What is 2+2?",
            ground_truth="4",
        )
        assert test_case.ground_truth == "4"

    def test_test_case_evaluations_optional(self) -> None:
        """Test that evaluations is optional."""
        test_case = TestCase(input="Test")
        assert test_case.evaluations is None

    def test_test_case_with_evaluations(self) -> None:
        """Test TestCase with specific evaluations."""
        test_case = TestCase(
            input="Test",
            evaluations=["groundedness", "relevance"],
        )
        assert test_case.evaluations == ["groundedness", "relevance"]

    def test_test_case_files_optional(self) -> None:
        """Test that files is optional."""
        test_case = TestCase(input="Test")
        assert test_case.files is None

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
            evaluations=["groundedness", "relevance"],
        )
        assert test_case.name == "Test case 1"
        assert test_case.input == "Process document"
        assert test_case.expected_tools == ["extractor"]
        assert test_case.ground_truth == "Expected output"
        assert len(test_case.files) == 1
        assert test_case.evaluations == ["groundedness", "relevance"]

    def test_test_case_max_input_length(self) -> None:
        """Test that long inputs are accepted (up to reasonable limit)."""
        long_input = "x" * 5000
        test_case = TestCase(input=long_input)
        assert test_case.input == long_input
