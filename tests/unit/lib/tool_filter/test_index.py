"""Unit tests for tool_filter index."""

from unittest.mock import AsyncMock, MagicMock

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
        """Test tokenizing text with special characters and underscores."""
        result = _tokenize("hello-world, how's it_going?")
        # Underscores are now treated as separators (not word chars)
        # to enable matching "web" against "brave_web_search"
        assert result == ["hello", "world", "how", "s", "it", "going"]

    def test_underscore_splitting(self) -> None:
        """Test that underscores split tokens for tool name matching."""
        result = _tokenize("brave_web_search")
        # Tool names like "brave_web_search" should split into components
        # so that queries containing "web" can match the tool
        assert result == ["brave", "web", "search"]

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

    @pytest.fixture
    def mock_kernel(self) -> MagicMock:
        """Create a mock kernel with no plugins."""
        kernel = MagicMock()
        kernel.plugins = {}
        return kernel

    @pytest.fixture
    def kernel_with_plugins(self) -> MagicMock:
        """Create a mock kernel with plugins and functions."""
        metadata = MagicMock()
        metadata.description = "Metadata description"

        param_query = MagicMock()
        param_query.name = "query"
        param_query.description = "Search query"

        param_limit = MagicMock()
        param_limit.name = "limit"
        param_limit.description = ""

        func_search = MagicMock()
        func_search.description = ""
        func_search.metadata = metadata
        func_search.parameters = [param_query, param_limit]

        func_broken = MagicMock()
        func_broken.description = ""
        func_broken.metadata = None
        broken_params = MagicMock()
        broken_params.__iter__.side_effect = Exception("boom")
        func_broken.parameters = broken_params

        plugin = MagicMock()
        plugin.functions = {"search": func_search, "broken": func_broken}

        kernel = MagicMock()
        kernel.plugins = {"plugin": plugin}
        return kernel

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

    @pytest.mark.asyncio
    async def test_build_from_kernel_no_plugins(self, mock_kernel: MagicMock) -> None:
        """Test build_from_kernel exits when no plugins exist."""
        index = ToolIndex()

        await index.build_from_kernel(mock_kernel)

        assert index.tools == {}

    @pytest.mark.asyncio
    async def test_build_from_kernel_extracts_metadata(
        self, kernel_with_plugins: MagicMock
    ) -> None:
        """Test build_from_kernel extracts metadata and parameters."""
        index = ToolIndex()

        await index.build_from_kernel(
            kernel_with_plugins,
            defer_loading_map={"plugin-search": False},
        )

        assert "plugin-search" in index.tools
        tool = index.tools["plugin-search"]
        assert tool.description == "Metadata description"
        assert tool.parameters == ["query: Search query", "limit"]
        assert tool.defer_loading is False

        fallback_tool = index.tools["plugin-broken"]
        assert fallback_tool.description == "Function broken from plugin plugin"
        assert fallback_tool.parameters == []

    @pytest.mark.asyncio
    async def test_generate_embeddings_assigns_values(self) -> None:
        """Test that embeddings are assigned when generated."""
        index = ToolIndex()
        index.tools["plugin-search"] = ToolMetadata(
            name="search",
            plugin_name="plugin",
            full_name="plugin-search",
            description="Search the docs",
            parameters=["query"],
        )
        embedding_service = AsyncMock()
        embedding_service.generate_embeddings = AsyncMock(return_value=[[0.1, 0.2]])

        await index._generate_embeddings(embedding_service)

        assert index.tools["plugin-search"].embedding == [0.1, 0.2]

    @pytest.mark.asyncio
    async def test_generate_embeddings_handles_error(self) -> None:
        """Test that embedding generation failures are handled gracefully."""
        index = ToolIndex()
        index.tools["plugin-search"] = ToolMetadata(
            name="search",
            plugin_name="plugin",
            full_name="plugin-search",
            description="Search the docs",
            parameters=["query"],
        )
        embedding_service = AsyncMock()
        embedding_service.generate_embeddings = AsyncMock(side_effect=Exception("fail"))

        await index._generate_embeddings(embedding_service)

        assert index.tools["plugin-search"].embedding is None

    @pytest.mark.asyncio
    async def test_search_unknown_method_falls_back_to_semantic(
        self, sample_index: ToolIndex
    ) -> None:
        """Test unknown search method falls back to semantic search."""
        embedding_service = AsyncMock()
        embedding_service.generate_embeddings = AsyncMock(return_value=[[1.0, 0.0]])

        sample_index.tools["vectorstore-search"].embedding = [1.0, 0.0]

        results = await sample_index.search(
            query="search",
            top_k=2,
            method="unknown",
            embedding_service=embedding_service,
        )

        assert results
        assert results[0][0].full_name == "vectorstore-search"

    @pytest.mark.asyncio
    async def test_semantic_search_without_embeddings_falls_back_to_bm25(
        self, sample_index: ToolIndex
    ) -> None:
        """Test semantic search falls back to BM25 without embeddings."""
        results = await sample_index.search(
            query="weather",
            top_k=3,
            method="semantic",
        )

        assert results
        tool_names = [tool.full_name for tool, _ in results]
        assert "mcp-weather" in tool_names

    @pytest.mark.asyncio
    async def test_semantic_search_mixed_embeddings(self) -> None:
        """Test semantic search falls back to BM25 for missing embeddings."""
        index = ToolIndex()
        index.tools["plugin-embedded"] = ToolMetadata(
            name="embedded",
            plugin_name="plugin",
            full_name="plugin-embedded",
            description="Embedded tool",
            embedding=[0.2, 0.0],
        )
        index.tools["plugin-bm25"] = ToolMetadata(
            name="bm25",
            plugin_name="plugin",
            full_name="plugin-bm25",
            description="BM25 tool",
        )
        docs = [
            (tool.full_name, index._create_searchable_text(tool))
            for tool in index.tools.values()
        ]
        index._build_bm25_index(docs)

        embedding_service = AsyncMock()
        embedding_service.generate_embeddings = AsyncMock(return_value=[[0.2, 0.0]])

        results = await index._semantic_search("bm25", embedding_service)

        assert {tool.full_name for tool, _ in results} == {
            "plugin-embedded",
            "plugin-bm25",
        }

    @pytest.mark.asyncio
    async def test_semantic_search_error_falls_back_to_bm25(
        self, sample_index: ToolIndex
    ) -> None:
        """Test semantic search errors fall back to BM25."""
        embedding_service = AsyncMock()
        embedding_service.generate_embeddings = AsyncMock(side_effect=Exception("fail"))

        results = await sample_index._semantic_search("weather", embedding_service)

        assert results
        tool_names = [tool.full_name for tool, _ in results]
        assert "mcp-weather" in tool_names

    def test_bm25_search_normalizes_scores(self, sample_index: ToolIndex) -> None:
        """Test BM25 search normalization to 0-1 range."""
        results = sample_index._bm25_search("weather")

        assert results
        max_score = max(score for _, score in results)
        assert max_score == 1.0

    def test_bm25_search_all_zero_scores(self) -> None:
        """Test BM25 search with all zero scores returns raw values."""
        index = ToolIndex()
        index.tools["empty"] = ToolMetadata(
            name="empty",
            plugin_name="plugin",
            full_name="plugin-empty",
            description="Empty tool",
        )
        docs = [
            (tool.full_name, index._create_searchable_text(tool))
            for tool in index.tools.values()
        ]
        index._build_bm25_index(docs)

        results = index._bm25_search("missing")

        assert results == [(index.tools["empty"], 0.0)]

    def test_bm25_score_single_handles_empty_inputs(self) -> None:
        """Test BM25 score returns 0.0 for empty query or document."""
        index = ToolIndex()
        tool = ToolMetadata(
            name="empty",
            plugin_name="plugin",
            full_name="plugin-empty",
            description="Empty tool",
        )

        assert index._bm25_score_single("", tool) == 0.0

        index.tools[tool.full_name] = tool
        index._build_bm25_index([(tool.full_name, "")])
        assert index._bm25_score_single("query", tool) == 0.0

    @pytest.mark.asyncio
    async def test_hybrid_search_normalizes_scores(self) -> None:
        """Test hybrid search normalizes RRF scores and filters tools."""
        index = ToolIndex()
        index.tools["plugin-a"] = ToolMetadata(
            name="alpha",
            plugin_name="plugin",
            full_name="plugin-a",
            description="Alpha tool",
            embedding=[1.0, 0.0],
        )
        index.tools["plugin-b"] = ToolMetadata(
            name="beta",
            plugin_name="plugin",
            full_name="plugin-b",
            description="Beta tool",
        )
        docs = [
            (tool.full_name, index._create_searchable_text(tool))
            for tool in index.tools.values()
        ]
        index._build_bm25_index(docs)

        embedding_service = AsyncMock()
        embedding_service.generate_embeddings = AsyncMock(return_value=[[1.0, 0.0]])

        results = await index._hybrid_search("alpha", embedding_service)

        assert results
        max_score = max(score for _, score in results)
        assert max_score == 1.0
        assert {tool.full_name for tool, _ in results} == {"plugin-a", "plugin-b"}
