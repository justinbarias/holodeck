"""Tests for structured data record types in holodeck.lib.vector_store.

Tests for StructuredQueryResult dataclass and create_structured_record_class() factory.
These tests should FAIL initially (TDD RED phase) until T013-T014 are implemented.
"""

import importlib
import sys
from typing import Any

import pytest

# Force reload of holodeck.lib.vector_store to ensure we get real SK imports
# This is needed because test_vector_store.py uses module-level mocking that
# can pollute the module cache when tests run in certain orders.
if "holodeck.lib.vector_store" in sys.modules:
    importlib.reload(sys.modules["holodeck.lib.vector_store"])

from semantic_kernel.data.vector import VectorStoreCollectionDefinition

from holodeck.lib.vector_store import (
    StructuredQueryResult,
    create_structured_record_class,
)


class TestStructuredQueryResult:
    """Tests for StructuredQueryResult dataclass.

    Tests the search result type for structured data queries.
    """

    def test_structured_query_result_creation(self) -> None:
        """Test creating a valid StructuredQueryResult."""
        result = StructuredQueryResult(
            id="P001",
            content="Product description text",
            score=0.85,
            source_file="products.csv",
            metadata={"title": "Widget Pro", "category": "Electronics"},
        )
        assert result.id == "P001"
        assert result.content == "Product description text"
        assert result.score == 0.85
        assert result.source_file == "products.csv"
        assert result.metadata == {"title": "Widget Pro", "category": "Electronics"}

    def test_structured_query_result_score_validation_min(self) -> None:
        """Test that score must be >= 0.0."""
        with pytest.raises(ValueError, match="0.0 and 1.0"):
            StructuredQueryResult(
                id="P001",
                content="Test",
                score=-0.1,
                source_file="test.csv",
                metadata={},
            )

    def test_structured_query_result_score_validation_max(self) -> None:
        """Test that score must be <= 1.0."""
        with pytest.raises(ValueError, match="0.0 and 1.0"):
            StructuredQueryResult(
                id="P001",
                content="Test",
                score=1.1,
                source_file="test.csv",
                metadata={},
            )

    def test_structured_query_result_score_boundary_zero(self) -> None:
        """Test that score of exactly 0.0 is valid."""
        result = StructuredQueryResult(
            id="P001",
            content="Test",
            score=0.0,
            source_file="test.csv",
            metadata={},
        )
        assert result.score == 0.0

    def test_structured_query_result_score_boundary_one(self) -> None:
        """Test that score of exactly 1.0 is valid."""
        result = StructuredQueryResult(
            id="P001",
            content="Test",
            score=1.0,
            source_file="test.csv",
            metadata={},
        )
        assert result.score == 1.0

    def test_structured_query_result_empty_metadata(self) -> None:
        """Test StructuredQueryResult with empty metadata."""
        result = StructuredQueryResult(
            id="P001",
            content="Test",
            score=0.5,
            source_file="test.csv",
            metadata={},
        )
        assert result.metadata == {}

    def test_structured_query_result_complex_metadata(self) -> None:
        """Test StructuredQueryResult with complex metadata."""
        metadata: dict[str, Any] = {
            "title": "Product Name",
            "price": "99.99",
            "tags": ["electronics", "gadget"],
            "nested": {"key": "value"},
        }
        result = StructuredQueryResult(
            id="P001",
            content="Test",
            score=0.75,
            source_file="products.json",
            metadata=metadata,
        )
        assert result.metadata == metadata
        assert result.metadata["tags"] == ["electronics", "gadget"]


class TestCreateStructuredRecordClass:
    """Tests for create_structured_record_class() factory function.

    Tests the dynamic record class generation for Semantic Kernel vector stores.
    The factory returns a tuple of (record_class, definition).
    """

    def test_factory_creates_class_and_definition(self) -> None:
        """Test that factory creates a class type and definition."""
        record_class, definition = create_structured_record_class()
        assert isinstance(record_class, type)
        assert isinstance(definition, VectorStoreCollectionDefinition)

    def test_factory_default_dimensions(self) -> None:
        """Test that factory uses default 1536 dimensions."""
        record_class, definition = create_structured_record_class()
        # The class should be created successfully with default dimensions
        assert record_class is not None
        assert definition is not None

    def test_factory_custom_dimensions(self) -> None:
        """Test that factory accepts custom dimensions."""
        record_class, definition = create_structured_record_class(dimensions=768)
        assert record_class is not None
        assert definition is not None

    def test_factory_invalid_dimensions_zero(self) -> None:
        """Test that factory rejects zero dimensions."""
        with pytest.raises(ValueError):
            create_structured_record_class(dimensions=0)

    def test_factory_invalid_dimensions_negative(self) -> None:
        """Test that factory rejects negative dimensions."""
        with pytest.raises(ValueError):
            create_structured_record_class(dimensions=-1)

    def test_factory_invalid_dimensions_too_large(self) -> None:
        """Test that factory rejects dimensions > 10000."""
        with pytest.raises(ValueError):
            create_structured_record_class(dimensions=10001)

    def test_factory_with_metadata_fields(self) -> None:
        """Test factory with metadata field names."""
        record_class, definition = create_structured_record_class(
            metadata_field_names=["title", "category", "price"],
        )
        assert record_class is not None
        assert definition is not None

    def test_factory_with_custom_collection_name(self) -> None:
        """Test factory with custom collection name."""
        record_class, definition = create_structured_record_class(
            collection_name="products_collection",
        )
        assert record_class is not None
        assert definition.collection_name == "products_collection"

    def test_generated_class_has_id_field(self) -> None:
        """Test that generated class has id field."""
        record_class, _ = create_structured_record_class()
        # Create an instance and check id field exists
        instance = record_class(
            id="test-id",
            content="test content",
            embedding=None,
            source_file="test.csv",
        )
        assert instance.id == "test-id"

    def test_generated_class_has_content_field(self) -> None:
        """Test that generated class has content field."""
        record_class, _ = create_structured_record_class()
        instance = record_class(
            id="test-id",
            content="test content text",
            embedding=None,
            source_file="test.csv",
        )
        assert instance.content == "test content text"

    def test_generated_class_has_embedding_field(self) -> None:
        """Test that generated class has embedding field."""
        record_class, _ = create_structured_record_class(dimensions=3)
        instance = record_class(
            id="test-id",
            content="test",
            embedding=[0.1, 0.2, 0.3],
            source_file="test.csv",
        )
        assert instance.embedding == [0.1, 0.2, 0.3]

    def test_generated_class_has_source_file_field(self) -> None:
        """Test that generated class has source_file field."""
        record_class, _ = create_structured_record_class()
        instance = record_class(
            id="test-id",
            content="test",
            embedding=None,
            source_file="data/products.csv",
        )
        assert instance.source_file == "data/products.csv"

    def test_generated_class_with_dynamic_metadata(self) -> None:
        """Test that generated class includes dynamic metadata fields."""
        record_class, _ = create_structured_record_class(
            metadata_field_names=["title", "category"],
        )
        instance = record_class(
            id="test-id",
            content="test",
            embedding=None,
            source_file="test.csv",
            title="Widget Pro",
            category="Electronics",
        )
        assert instance.title == "Widget Pro"
        assert instance.category == "Electronics"

    def test_generated_class_embedding_accepts_none(self) -> None:
        """Test that embedding field can be None."""
        record_class, _ = create_structured_record_class()
        instance = record_class(
            id="test-id",
            content="test",
            embedding=None,
            source_file="test.csv",
        )
        assert instance.embedding is None

    def test_factory_empty_metadata_fields(self) -> None:
        """Test factory with empty metadata field list."""
        record_class, definition = create_structured_record_class(
            metadata_field_names=[],
        )
        assert record_class is not None
        assert definition is not None

    def test_factory_none_metadata_fields(self) -> None:
        """Test factory with None metadata fields (default)."""
        record_class, definition = create_structured_record_class(
            metadata_field_names=None,
        )
        assert record_class is not None
        assert definition is not None

    def test_factory_invalid_field_name_starts_with_digit(self) -> None:
        """Test that factory rejects field names starting with a digit."""
        with pytest.raises(ValueError, match="Invalid field name.*123category"):
            create_structured_record_class(
                metadata_field_names=["123category"],
            )

    def test_factory_invalid_field_name_with_hyphen(self) -> None:
        """Test that factory rejects field names containing hyphens."""
        with pytest.raises(ValueError, match="Invalid field name.*price-usd"):
            create_structured_record_class(
                metadata_field_names=["price-usd"],
            )

    def test_factory_invalid_field_name_with_space(self) -> None:
        """Test that factory rejects field names containing spaces."""
        with pytest.raises(ValueError, match="Invalid field name.*my field"):
            create_structured_record_class(
                metadata_field_names=["my field"],
            )

    def test_factory_metadata_config_stored(self) -> None:
        """Test that __metadata_config__ stores field names."""
        record_class, _ = create_structured_record_class(
            metadata_field_names=["title", "category"],
        )
        assert hasattr(record_class, "__metadata_config__")
        assert record_class.__metadata_config__["field_names"] == ["title", "category"]

    def test_factory_metadata_fields_in_annotations(self) -> None:
        """Test that metadata fields are registered in __annotations__."""
        record_class, _ = create_structured_record_class(
            metadata_field_names=["title", "category"],
        )
        assert "title" in record_class.__annotations__
        assert "category" in record_class.__annotations__
        assert record_class.__annotations__["title"] is str
        assert record_class.__annotations__["category"] is str

    def test_factory_no_metadata_config_when_no_fields(self) -> None:
        """Test that __metadata_config__ is not set when no metadata fields."""
        record_class, _ = create_structured_record_class()
        assert not hasattr(record_class, "__metadata_config__")

    def test_metadata_fields_in_definition(self) -> None:
        """Test that metadata fields are included in the definition."""
        record_class, definition = create_structured_record_class(
            metadata_field_names=["title", "category"],
        )
        field_names = [f.name for f in definition.fields]
        # Core fields
        assert "id" in field_names
        assert "content" in field_names
        assert "embedding" in field_names
        assert "source_file" in field_names
        # Dynamic metadata fields
        assert "title" in field_names
        assert "category" in field_names

    def test_definition_has_correct_field_count(self) -> None:
        """Test that definition has correct number of fields."""
        # Without metadata: 4 fields (id, content, embedding, source_file)
        _, definition_no_meta = create_structured_record_class()
        assert len(definition_no_meta.fields) == 4

        # With 2 metadata fields: 6 fields total
        _, definition_with_meta = create_structured_record_class(
            metadata_field_names=["title", "category"],
        )
        assert len(definition_with_meta.fields) == 6
