"""Tests for StructuredDataLoader in holodeck.lib.structured_loader.

Tests for file type detection, CSV delimiter detection, record loading,
and batch iteration. These tests should FAIL initially (TDD RED phase)
until T025-T033 are implemented.
"""

import json
from pathlib import Path

import pytest

from holodeck.lib.errors import ConfigError

# These imports will fail until the module is created
from holodeck.lib.structured_loader import (
    FileType,
    StructuredDataLoader,
    detect_csv_delimiter,
    detect_file_type,
    load_csv_records,
    load_json_records,
    load_jsonl_records,
)

# Path to test fixtures
FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "structured"


class TestDetectFileType:
    """Tests for detect_file_type() function (T015)."""

    def test_detect_csv_file(self) -> None:
        """Test detection of .csv file type."""
        result = detect_file_type("data/products.csv")
        assert result == FileType.CSV

    def test_detect_json_file(self) -> None:
        """Test detection of .json file type."""
        result = detect_file_type("data/faqs.json")
        assert result == FileType.JSON

    def test_detect_jsonl_file(self) -> None:
        """Test detection of .jsonl file type."""
        result = detect_file_type("data/products.jsonl")
        assert result == FileType.JSONL

    def test_detect_case_insensitive(self) -> None:
        """Test that file type detection is case-insensitive."""
        assert detect_file_type("DATA.CSV") == FileType.CSV
        assert detect_file_type("data.JSON") == FileType.JSON
        assert detect_file_type("data.JSONL") == FileType.JSONL

    def test_detect_unsupported_extension(self) -> None:
        """Test that unsupported file type raises error."""
        with pytest.raises(ValueError, match="Unsupported file type"):
            detect_file_type("data/file.xml")

    def test_detect_no_extension(self) -> None:
        """Test that file without extension raises error."""
        with pytest.raises(ValueError, match="Unsupported file type"):
            detect_file_type("data/file")


class TestDetectCsvDelimiter:
    """Tests for detect_csv_delimiter() function (T016)."""

    def test_detect_comma_delimiter(self) -> None:
        """Test detection of comma delimiter."""
        content = "id,name,value\n1,test,100\n2,test2,200"
        assert detect_csv_delimiter(content) == ","

    def test_detect_semicolon_delimiter(self) -> None:
        """Test detection of semicolon delimiter."""
        content = "id;name;value\n1;test;100\n2;test2;200"
        assert detect_csv_delimiter(content) == ";"

    def test_detect_tab_delimiter(self) -> None:
        """Test detection of tab delimiter."""
        content = "id\tname\tvalue\n1\ttest\t100\n2\ttest2\t200"
        assert detect_csv_delimiter(content) == "\t"

    def test_detect_pipe_delimiter(self) -> None:
        """Test detection of pipe delimiter."""
        content = "id|name|value\n1|test|100\n2|test2|200"
        assert detect_csv_delimiter(content) == "|"

    def test_detect_with_quotes(self) -> None:
        """Test detection with quoted fields."""
        content = 'id,name,value\n1,"test, with comma",100'
        assert detect_csv_delimiter(content) == ","

    def test_detect_fallback_to_comma(self) -> None:
        """Test fallback to comma when detection fails."""
        # Single line or ambiguous content should default to comma
        content = "single line content"
        assert detect_csv_delimiter(content) == ","


class TestValidateSchema:
    """Tests for StructuredDataLoader.validate_schema() method (T017)."""

    def test_validate_schema_all_fields_exist(self) -> None:
        """Test validation passes when all fields exist in source."""
        loader = StructuredDataLoader(
            source_path=str(FIXTURES_DIR / "products.csv"),
            id_field="id",
            vector_fields=["description"],
            metadata_fields=["title", "category"],
        )
        available_fields = loader.validate_schema()
        assert "id" in available_fields
        assert "description" in available_fields
        assert "title" in available_fields
        assert "category" in available_fields

    def test_validate_schema_missing_id_field(self) -> None:
        """Test validation fails when id_field doesn't exist."""
        loader = StructuredDataLoader(
            source_path=str(FIXTURES_DIR / "products.csv"),
            id_field="nonexistent_id",
            vector_fields=["description"],
        )
        with pytest.raises(ConfigError, match="id_field"):
            loader.validate_schema()

    def test_validate_schema_missing_vector_field(self) -> None:
        """Test validation fails when vector_field doesn't exist."""
        loader = StructuredDataLoader(
            source_path=str(FIXTURES_DIR / "products.csv"),
            id_field="id",
            vector_fields=["nonexistent_field"],
        )
        with pytest.raises(ConfigError, match="vector_field"):
            loader.validate_schema()

    def test_validate_schema_missing_metadata_field(self) -> None:
        """Test validation fails when metadata_field doesn't exist."""
        loader = StructuredDataLoader(
            source_path=str(FIXTURES_DIR / "products.csv"),
            id_field="id",
            vector_fields=["description"],
            metadata_fields=["nonexistent_metadata"],
        )
        with pytest.raises(ConfigError, match="metadata_field"):
            loader.validate_schema()

    def test_validate_schema_returns_available_fields(self) -> None:
        """Test that validation returns list of available fields."""
        loader = StructuredDataLoader(
            source_path=str(FIXTURES_DIR / "products.csv"),
            id_field="id",
            vector_fields=["description"],
        )
        available_fields = loader.validate_schema()
        assert isinstance(available_fields, list)
        assert len(available_fields) == 5  # id, title, description, category, price


class TestLoadCsvRecords:
    """Tests for load_csv_records() function (T018)."""

    def test_load_csv_records_basic(self) -> None:
        """Test loading CSV records with auto-detected delimiter."""
        records = list(load_csv_records(str(FIXTURES_DIR / "products.csv"), ","))
        assert len(records) == 3
        assert records[0]["id"] == "1"
        assert records[0]["title"] == "Widget Pro"

    def test_load_csv_records_all_fields(self) -> None:
        """Test that all fields are loaded from CSV."""
        records = list(load_csv_records(str(FIXTURES_DIR / "products.csv"), ","))
        first_record = records[0]
        assert "id" in first_record
        assert "title" in first_record
        assert "description" in first_record
        assert "category" in first_record
        assert "price" in first_record

    def test_load_csv_records_with_quotes(self) -> None:
        """Test loading CSV with quoted fields."""
        records = list(load_csv_records(str(FIXTURES_DIR / "products.csv"), ","))
        # description field has quotes in the source
        assert records[0]["description"] == "Advanced widget with AI features"

    def test_load_csv_records_generator(self) -> None:
        """Test that load_csv_records returns a generator."""
        result = load_csv_records(str(FIXTURES_DIR / "products.csv"), ",")
        # Check it's an iterator/generator
        assert hasattr(result, "__iter__")
        assert hasattr(result, "__next__")

    def test_load_csv_records_nonexistent_file(self) -> None:
        """Test loading nonexistent CSV file raises error."""
        with pytest.raises(FileNotFoundError):
            list(load_csv_records("/nonexistent/path.csv", ","))


class TestLoadJsonRecords:
    """Tests for load_json_records() function (T019)."""

    def test_load_json_records_basic(self) -> None:
        """Test loading JSON array records."""
        records = load_json_records(str(FIXTURES_DIR / "faqs.json"))
        assert len(records) == 2
        assert records[0]["faq_id"] == "FAQ001"
        assert records[1]["faq_id"] == "FAQ002"

    def test_load_json_records_all_fields(self) -> None:
        """Test that all fields are loaded from JSON."""
        records = load_json_records(str(FIXTURES_DIR / "faqs.json"))
        first_record = records[0]
        assert "faq_id" in first_record
        assert "question" in first_record
        assert "answer" in first_record
        assert "category" in first_record

    def test_load_json_records_returns_list(self) -> None:
        """Test that load_json_records returns a list."""
        result = load_json_records(str(FIXTURES_DIR / "faqs.json"))
        assert isinstance(result, list)

    def test_load_json_records_nonexistent_file(self) -> None:
        """Test loading nonexistent JSON file raises error."""
        with pytest.raises(FileNotFoundError):
            load_json_records("/nonexistent/path.json")

    def test_load_json_records_invalid_json(self, tmp_path: Path) -> None:
        """Test loading invalid JSON raises error."""
        bad_json = tmp_path / "bad.json"
        bad_json.write_text("{ invalid json }")
        with pytest.raises(json.JSONDecodeError):
            load_json_records(str(bad_json))


class TestLoadJsonlRecords:
    """Tests for load_jsonl_records() function (T020)."""

    def test_load_jsonl_records_basic(self) -> None:
        """Test loading JSONL records."""
        records = list(load_jsonl_records(str(FIXTURES_DIR / "products.jsonl")))
        assert len(records) == 2
        assert records[0]["product_id"] == "P001"
        assert records[1]["product_id"] == "P002"

    def test_load_jsonl_records_all_fields(self) -> None:
        """Test that all fields are loaded from JSONL."""
        records = list(load_jsonl_records(str(FIXTURES_DIR / "products.jsonl")))
        first_record = records[0]
        assert "product_id" in first_record
        assert "name" in first_record
        assert "desc" in first_record

    def test_load_jsonl_records_generator(self) -> None:
        """Test that load_jsonl_records returns a generator."""
        result = load_jsonl_records(str(FIXTURES_DIR / "products.jsonl"))
        assert hasattr(result, "__iter__")
        assert hasattr(result, "__next__")

    def test_load_jsonl_records_nonexistent_file(self) -> None:
        """Test loading nonexistent JSONL file raises error."""
        with pytest.raises(FileNotFoundError):
            list(load_jsonl_records("/nonexistent/path.jsonl"))

    def test_load_jsonl_records_skips_empty_lines(self, tmp_path: Path) -> None:
        """Test that empty lines in JSONL are skipped."""
        jsonl_file = tmp_path / "test.jsonl"
        jsonl_file.write_text('{"id": "1"}\n\n{"id": "2"}\n')
        records = list(load_jsonl_records(str(jsonl_file)))
        assert len(records) == 2


class TestIterRecords:
    """Tests for StructuredDataLoader.iter_records() method (T021)."""

    def test_iter_records_basic(self) -> None:
        """Test iterating records with field mapping."""
        loader = StructuredDataLoader(
            source_path=str(FIXTURES_DIR / "products.csv"),
            id_field="id",
            vector_fields=["description"],
            metadata_fields=["title", "category"],
        )
        records = list(loader.iter_records())
        assert len(records) == 3

        first = records[0]
        assert first["id"] == "1"
        assert first["content"] == "Advanced widget with AI features"
        assert first["metadata"]["title"] == "Widget Pro"
        assert first["metadata"]["category"] == "Electronics"

    def test_iter_records_multiple_vector_fields(self) -> None:
        """Test concatenation of multiple vector fields."""
        loader = StructuredDataLoader(
            source_path=str(FIXTURES_DIR / "faqs.json"),
            id_field="faq_id",
            vector_fields=["question", "answer"],
            metadata_fields=["category"],
            field_separator="\n",
        )
        records = list(loader.iter_records())
        first = records[0]

        # Content should be question + separator + answer
        expected_content = (
            "How do I reset my password?\n" "Go to Settings > Security > Reset Password"
        )
        assert first["content"] == expected_content

    def test_iter_records_custom_separator(self) -> None:
        """Test custom field separator for multiple vector fields."""
        loader = StructuredDataLoader(
            source_path=str(FIXTURES_DIR / "faqs.json"),
            id_field="faq_id",
            vector_fields=["question", "answer"],
            field_separator=" | ",
        )
        records = list(loader.iter_records())
        first = records[0]

        assert " | " in first["content"]

    def test_iter_records_no_metadata_includes_all(self) -> None:
        """Test that None metadata_fields includes all non-vector fields."""
        loader = StructuredDataLoader(
            source_path=str(FIXTURES_DIR / "products.csv"),
            id_field="id",
            vector_fields=["description"],
            metadata_fields=None,  # Should include all other fields
        )
        records = list(loader.iter_records())
        first = records[0]

        # Metadata should include title, category, price (not id or description)
        assert "title" in first["metadata"]
        assert "category" in first["metadata"]
        assert "price" in first["metadata"]
        # id and description should NOT be in metadata
        assert "id" not in first["metadata"]
        assert "description" not in first["metadata"]

    def test_iter_records_is_generator(self) -> None:
        """Test that iter_records returns a generator."""
        loader = StructuredDataLoader(
            source_path=str(FIXTURES_DIR / "products.csv"),
            id_field="id",
            vector_fields=["description"],
        )
        result = loader.iter_records()
        assert hasattr(result, "__iter__")
        assert hasattr(result, "__next__")

    def test_iter_records_skips_empty_vector_field(self, tmp_path: Path) -> None:
        """Test that records with empty vector fields are skipped."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("id,description\n1,valid content\n2,\n3,more content\n")

        loader = StructuredDataLoader(
            source_path=str(csv_file),
            id_field="id",
            vector_fields=["description"],
        )
        records = list(loader.iter_records())

        # Record with empty description should be skipped
        assert len(records) == 2
        assert records[0]["id"] == "1"
        assert records[1]["id"] == "3"

    def test_iter_records_type_coercion(self, tmp_path: Path) -> None:
        """Test that non-string values are converted to strings."""
        json_file = tmp_path / "test.json"
        json_file.write_text('[{"id": 1, "content": 123, "meta": true}]')

        loader = StructuredDataLoader(
            source_path=str(json_file),
            id_field="id",
            vector_fields=["content"],
            metadata_fields=["meta"],
        )
        records = list(loader.iter_records())

        assert records[0]["id"] == "1"
        assert records[0]["content"] == "123"
        assert records[0]["metadata"]["meta"] == "True"


class TestIterBatches:
    """Tests for StructuredDataLoader.iter_batches() method (T022)."""

    def test_iter_batches_default_size(self) -> None:
        """Test batch iteration with default batch size."""
        loader = StructuredDataLoader(
            source_path=str(FIXTURES_DIR / "products.csv"),
            id_field="id",
            vector_fields=["description"],
        )
        batches = list(loader.iter_batches())

        # With 3 records and default batch_size (10000), should be 1 batch
        assert len(batches) == 1
        assert len(batches[0]) == 3

    def test_iter_batches_custom_size(self, tmp_path: Path) -> None:
        """Test batch iteration with custom batch size."""
        # Create a CSV with 5 records
        csv_file = tmp_path / "test.csv"
        lines = ["id,content"]
        for i in range(5):
            lines.append(f"{i},content {i}")
        csv_file.write_text("\n".join(lines))

        loader = StructuredDataLoader(
            source_path=str(csv_file),
            id_field="id",
            vector_fields=["content"],
            batch_size=2,
        )
        batches = list(loader.iter_batches())

        # 5 records with batch_size=2 should give 3 batches
        assert len(batches) == 3
        assert len(batches[0]) == 2
        assert len(batches[1]) == 2
        assert len(batches[2]) == 1

    def test_iter_batches_is_generator(self) -> None:
        """Test that iter_batches returns a generator."""
        loader = StructuredDataLoader(
            source_path=str(FIXTURES_DIR / "products.csv"),
            id_field="id",
            vector_fields=["description"],
        )
        result = loader.iter_batches()
        assert hasattr(result, "__iter__")
        assert hasattr(result, "__next__")

    def test_iter_batches_preserves_record_structure(self) -> None:
        """Test that batched records have correct structure."""
        loader = StructuredDataLoader(
            source_path=str(FIXTURES_DIR / "products.csv"),
            id_field="id",
            vector_fields=["description"],
            metadata_fields=["title"],
        )
        batches = list(loader.iter_batches())

        first_record = batches[0][0]
        assert "id" in first_record
        assert "content" in first_record
        assert "metadata" in first_record
        assert "title" in first_record["metadata"]


class TestStructuredDataLoaderInit:
    """Tests for StructuredDataLoader initialization."""

    def test_init_with_required_params(self) -> None:
        """Test initialization with required parameters."""
        loader = StructuredDataLoader(
            source_path=str(FIXTURES_DIR / "products.csv"),
            id_field="id",
            vector_fields=["description"],
        )
        assert loader.source_path == str(FIXTURES_DIR / "products.csv")
        assert loader.id_field == "id"
        assert loader.vector_fields == ["description"]

    def test_init_with_all_params(self) -> None:
        """Test initialization with all parameters."""
        loader = StructuredDataLoader(
            source_path=str(FIXTURES_DIR / "products.csv"),
            id_field="id",
            vector_fields=["description", "title"],
            metadata_fields=["category", "price"],
            field_separator=" | ",
            delimiter=",",
            batch_size=100,
        )
        assert loader.field_separator == " | "
        assert loader.delimiter == ","
        assert loader.batch_size == 100

    def test_init_defaults(self) -> None:
        """Test default parameter values."""
        loader = StructuredDataLoader(
            source_path="test.csv",
            id_field="id",
            vector_fields=["content"],
        )
        assert loader.field_separator == "\n"
        assert loader.delimiter is None  # Auto-detect
        assert loader.batch_size == 10_000
        assert loader.metadata_fields is None
