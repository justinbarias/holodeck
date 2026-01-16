"""Unit tests for tool_filter index."""

import pytest

from holodeck.lib.tool_filter.index import ToolIndex, _cosine_similarity, _tokenize
from holodeck.lib.tool_filter.models import ToolMetadata


class TestCosineSimililarity:
    """Tests for cosine similarity helper function."""

    def test_identical_vectors(self) -> None:
        """Test cosine similarity of identical vectors is 1.0."""
        vec = [1.0, 2.0, 3.0]
        result = _cosine_similarity(vec, vec)
        assert abs(result - 1.0) < 1e-6

    def test_orthogonal_vectors(self) -> None:
        """Test cosine similarity of orthogonal vectors is 0.0."""
        vec_a = [1.0, 0.0]
        vec_b = [0.0, 1.0]
        result = _cosine_similarity(vec_a, vec_b)
        assert abs(result) < 1e-6

    def test_opposite_vectors(self) -> None:
        """Test cosine similarity of opposite vectors is -1.0."""
        vec_a = [1.0, 0.0]
        vec_b = [-1.0, 0.0]
        result = _cosine_similarity(vec_a, vec_b)
        assert abs(result - (-1.0)) < 1e-6

    def test_different_lengths(self) -> None:
        """Test that different length vectors return 0.0."""
        vec_a = [1.0, 2.0]
        vec_b = [1.0, 2.0, 3.0]
        result = _cosine_similarity(vec_a, vec_b)
        assert result == 0.0

    def test_zero_vector(self) -> None:
        """Test that zero vector returns 0.0."""
        vec_a = [0.0, 0.0]
        vec_b = [1.0, 2.0]
        result = _cosine_similarity(vec_a, vec_b)
        assert result == 0.0


class TestTokenize:
    """Tests for tokenize helper function."""

    def test_simple_text(self) -> None:
        """Test tokenizing simple text."""
        result = _tokenize("hello world")
        assert result == ["hello", "world"]

    def test_mixed_case(self) -> None:
        """Test that tokenizing converts to lowercase."""
        result = _tokenize("Hello WORLD")
        assert result == ["hello", "world"]

    def test_special_characters(self) -> None:
        """Test tokenizing text with special characters."""
        result = _tokenize("hello-world, how's it_going?")
        assert result == ["hello", "world", "how", "s", "it_going"]

    def test_numbers(self) -> None:
        """Test tokenizing text with numbers."""
        result = _tokenize("test123 456test")
        assert result == ["test123", "456test"]

    def test_empty_string(self) -> None:
        """Test tokenizing empty string."""
        result = _tokenize("")
        assert result == []


class TestToolIndex:
    """Tests for ToolIndex class."""

    @pytest.fixture
    def sample_index(self) -> ToolIndex:
        """Create a sample index with pre-populated tools."""
        index = ToolIndex()

        # Add sample tools directly
        index.tools["vectorstore-search"] = ToolMetadata(
            name="search",
            plugin_name="vectorstore",
            full_name="vectorstore-search",
            description="Search the knowledge base for relevant documents",
            parameters=["query: search query string"],
            defer_loading=True,
            usage_count=5,
        )

        index.tools["mcp-weather"] = ToolMetadata(
            name="weather",
            plugin_name="mcp",
            full_name="mcp-weather",
            description="Get current weather for a location",
            parameters=["location: city name"],
            defer_loading=True,
            usage_count=10,
        )

        index.tools["vectorstore-summarize"] = ToolMetadata(
            name="summarize",
            plugin_name="vectorstore",
            full_name="vectorstore-summarize",
            description="Summarize document content",
            parameters=["text: text to summarize"],
            defer_loading=False,
            usage_count=2,
        )

        # Build BM25 index
        documents = [
            (tool.full_name, index._create_searchable_text(tool))
            for tool in index.tools.values()
        ]
        index._build_bm25_index(documents)

        return index

    def test_empty_index(self) -> None:
        """Test that empty index has no tools."""
        index = ToolIndex()
        assert len(index.tools) == 0
        assert index.get_all_tool_names() == []

    def test_get_tool(self, sample_index: ToolIndex) -> None:
        """Test retrieving a tool by full name."""
        tool = sample_index.get_tool("vectorstore-search")
        assert tool is not None
        assert tool.name == "search"

    def test_get_nonexistent_tool(self, sample_index: ToolIndex) -> None:
        """Test retrieving a non-existent tool returns None."""
        tool = sample_index.get_tool("nonexistent")
        assert tool is None

    def test_get_all_tool_names(self, sample_index: ToolIndex) -> None:
        """Test getting all tool names."""
        names = sample_index.get_all_tool_names()
        assert len(names) == 3
        assert "vectorstore-search" in names
        assert "mcp-weather" in names
        assert "vectorstore-summarize" in names

    def test_update_usage(self, sample_index: ToolIndex) -> None:
        """Test updating tool usage count."""
        initial_count = sample_index.tools["vectorstore-search"].usage_count
        sample_index.update_usage("vectorstore-search")
        assert sample_index.tools["vectorstore-search"].usage_count == initial_count + 1

    def test_update_usage_nonexistent(self, sample_index: ToolIndex) -> None:
        """Test that updating non-existent tool does nothing."""
        sample_index.update_usage("nonexistent")
        # Should not raise an error

    def test_get_top_n_used(self, sample_index: ToolIndex) -> None:
        """Test getting most used tools."""
        top_tools = sample_index.get_top_n_used(2)
        assert len(top_tools) == 2
        # Should be sorted by usage_count descending
        assert top_tools[0].full_name == "mcp-weather"  # usage_count=10
        assert top_tools[1].full_name == "vectorstore-search"  # usage_count=5

    def test_get_top_n_used_zero(self, sample_index: ToolIndex) -> None:
        """Test getting zero top tools."""
        top_tools = sample_index.get_top_n_used(0)
        assert len(top_tools) == 0

    def test_get_top_n_used_more_than_available(self, sample_index: ToolIndex) -> None:
        """Test getting more top tools than available."""
        top_tools = sample_index.get_top_n_used(100)
        assert len(top_tools) == 3

    @pytest.mark.asyncio
    async def test_bm25_search(self, sample_index: ToolIndex) -> None:
        """Test BM25 keyword search."""
        results = await sample_index.search(
            query="weather location",
            top_k=3,
            method="bm25",
            threshold=0.0,
        )

        assert len(results) > 0
        # Weather tool should rank highly for "weather location"
        tool_names = [tool.full_name for tool, score in results]
        assert "mcp-weather" in tool_names

    @pytest.mark.asyncio
    async def test_search_with_threshold(self, sample_index: ToolIndex) -> None:
        """Test search with similarity threshold."""
        results = await sample_index.search(
            query="weather",
            top_k=3,
            method="bm25",
            threshold=10.0,  # Very high threshold
        )

        # May return fewer results due to threshold
        assert len(results) <= 3

    @pytest.mark.asyncio
    async def test_search_empty_index(self) -> None:
        """Test searching empty index returns empty results."""
        index = ToolIndex()
        results = await index.search(
            query="test",
            top_k=5,
            method="bm25",
        )
        assert results == []

    def test_create_searchable_text(self, sample_index: ToolIndex) -> None:
        """Test creating searchable text from tool metadata."""
        tool = sample_index.tools["vectorstore-search"]
        text = sample_index._create_searchable_text(tool)

        assert "search" in text
        assert "vectorstore" in text
        assert "knowledge base" in text
        assert "query" in text
