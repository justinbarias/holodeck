"""Tests for ``StructuredQueryResult``, the structured-data search result."""

from typing import Any

import pytest

from holodeck.lib.retrieval import StructuredQueryResult


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
