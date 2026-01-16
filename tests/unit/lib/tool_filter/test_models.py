"""Unit tests for tool_filter models."""

import pytest
from pydantic import ValidationError

from holodeck.lib.tool_filter.models import ToolFilterConfig, ToolMetadata


class TestToolMetadata:
    """Tests for ToolMetadata model."""

    def test_valid_tool_metadata(self) -> None:
        """Test creating valid ToolMetadata."""
        metadata = ToolMetadata(
            name="search",
            plugin_name="vectorstore",
            full_name="vectorstore-search",
            description="Search the knowledge base",
            parameters=["query: search query"],
            defer_loading=True,
            usage_count=0,
        )

        assert metadata.name == "search"
        assert metadata.plugin_name == "vectorstore"
        assert metadata.full_name == "vectorstore-search"
        assert metadata.description == "Search the knowledge base"
        assert metadata.parameters == ["query: search query"]
        assert metadata.defer_loading is True
        assert metadata.embedding is None
        assert metadata.usage_count == 0

    def test_minimal_tool_metadata(self) -> None:
        """Test creating ToolMetadata with only required fields."""
        metadata = ToolMetadata(
            name="search",
            full_name="search",
            description="Search function",
        )

        assert metadata.name == "search"
        assert metadata.plugin_name == ""
        assert metadata.full_name == "search"
        assert metadata.description == "Search function"
        assert metadata.parameters == []
        assert metadata.defer_loading is True
        assert metadata.embedding is None
        assert metadata.usage_count == 0

    def test_tool_metadata_with_embedding(self) -> None:
        """Test creating ToolMetadata with embedding vector."""
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        metadata = ToolMetadata(
            name="search",
            full_name="search",
            description="Search function",
            embedding=embedding,
        )

        assert metadata.embedding == embedding

    def test_empty_name_raises_error(self) -> None:
        """Test that empty name raises validation error."""
        with pytest.raises(ValidationError, match="name must be a non-empty string"):
            ToolMetadata(
                name="",
                full_name="search",
                description="Search function",
            )

    def test_empty_full_name_raises_error(self) -> None:
        """Test that empty full_name raises validation error."""
        with pytest.raises(
            ValidationError, match="full_name must be a non-empty string"
        ):
            ToolMetadata(
                name="search",
                full_name="",
                description="Search function",
            )

    def test_empty_description_raises_error(self) -> None:
        """Test that empty description raises validation error."""
        with pytest.raises(
            ValidationError, match="description must be a non-empty string"
        ):
            ToolMetadata(
                name="search",
                full_name="search",
                description="",
            )

    def test_negative_usage_count_raises_error(self) -> None:
        """Test that negative usage_count raises validation error."""
        with pytest.raises(ValidationError, match="usage_count must be non-negative"):
            ToolMetadata(
                name="search",
                full_name="search",
                description="Search function",
                usage_count=-1,
            )

    def test_extra_fields_forbidden(self) -> None:
        """Test that extra fields raise validation error."""
        with pytest.raises(ValidationError):
            ToolMetadata(
                name="search",
                full_name="search",
                description="Search function",
                extra_field="not_allowed",  # type: ignore[call-arg]
            )


class TestToolFilterConfig:
    """Tests for ToolFilterConfig model."""

    def test_default_config(self) -> None:
        """Test creating ToolFilterConfig with defaults."""
        config = ToolFilterConfig()

        assert config.enabled is False
        assert config.top_k == 5
        assert config.similarity_threshold == 0.3
        assert config.always_include == []
        assert config.always_include_top_n_used == 3
        assert config.search_method == "semantic"

    def test_custom_config(self) -> None:
        """Test creating ToolFilterConfig with custom values."""
        config = ToolFilterConfig(
            enabled=True,
            top_k=10,
            similarity_threshold=0.5,
            always_include=["get_user", "search"],
            always_include_top_n_used=5,
            search_method="hybrid",
        )

        assert config.enabled is True
        assert config.top_k == 10
        assert config.similarity_threshold == 0.5
        assert config.always_include == ["get_user", "search"]
        assert config.always_include_top_n_used == 5
        assert config.search_method == "hybrid"

    def test_top_k_minimum(self) -> None:
        """Test that top_k must be at least 1."""
        with pytest.raises(ValidationError, match="greater than or equal to 1"):
            ToolFilterConfig(top_k=0)

    def test_top_k_maximum(self) -> None:
        """Test that top_k cannot exceed 50."""
        with pytest.raises(ValidationError, match="less than or equal to 50"):
            ToolFilterConfig(top_k=51)

    def test_similarity_threshold_range(self) -> None:
        """Test that similarity_threshold must be between 0.0 and 1.0."""
        # Valid boundaries
        ToolFilterConfig(similarity_threshold=0.0)
        ToolFilterConfig(similarity_threshold=1.0)

        # Invalid values
        with pytest.raises(ValidationError):
            ToolFilterConfig(similarity_threshold=-0.1)

        with pytest.raises(ValidationError):
            ToolFilterConfig(similarity_threshold=1.1)

    def test_always_include_top_n_used_range(self) -> None:
        """Test always_include_top_n_used range validation."""
        # Valid boundaries
        ToolFilterConfig(always_include_top_n_used=0)
        ToolFilterConfig(always_include_top_n_used=20)

        # Invalid values
        with pytest.raises(ValidationError):
            ToolFilterConfig(always_include_top_n_used=-1)

        with pytest.raises(ValidationError):
            ToolFilterConfig(always_include_top_n_used=21)

    def test_valid_search_methods(self) -> None:
        """Test that only valid search methods are accepted."""
        ToolFilterConfig(search_method="semantic")
        ToolFilterConfig(search_method="bm25")
        ToolFilterConfig(search_method="hybrid")

        with pytest.raises(ValidationError):
            ToolFilterConfig(search_method="invalid")  # type: ignore[arg-type]

    def test_empty_always_include_entries_raises_error(self) -> None:
        """Test that empty strings in always_include raise error."""
        with pytest.raises(
            ValidationError, match="always_include entries must be non-empty"
        ):
            ToolFilterConfig(always_include=["search", ""])

    def test_extra_fields_forbidden(self) -> None:
        """Test that extra fields raise validation error."""
        with pytest.raises(ValidationError):
            ToolFilterConfig(extra_field="not_allowed")  # type: ignore[call-arg]
