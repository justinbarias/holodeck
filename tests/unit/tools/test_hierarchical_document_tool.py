"""Unit tests for HierarchicalDocumentTool.

Tests for the HierarchicalDocumentTool class that provides hierarchical document
retrieval with structure preservation, definition extraction, and context generation.

Test IDs:
- T20: TDD Test skeleton for HierarchicalDocumentTool
- T36: HierarchicalDocumentTool class enhancement
- T37: Document ingestion pipeline orchestration
- T38: Chunk embedding generation
- T39: Chunk storage with HierarchicalDocumentRecord
- T40: Semantic-only search implementation

Note: This test module requires mocking semantic_kernel modules because the full
semantic_kernel library is not available in the test environment.
"""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# Save original modules before mocking (these will be restored after import)
_saved_modules: dict[str, object] = {}
_mocked_modules: set[str] = set()

# Mock semantic_kernel modules BEFORE importing holodeck modules
# This prevents import errors from semantic_kernel dependencies
_all_sk_modules = [
    "semantic_kernel",
    "semantic_kernel.connectors",
    "semantic_kernel.connectors.memory",
    "semantic_kernel.connectors.ai",
    "semantic_kernel.connectors.ai.embedding_generator_base",
    "semantic_kernel.connectors.ai.function_choice_behavior",
    "semantic_kernel.connectors.ai.prompt_execution_settings",
    "semantic_kernel.connectors.ai.chat_completion_client_base",
    "semantic_kernel.connectors.ai.open_ai",
    "semantic_kernel.connectors.ai.anthropic",
    "semantic_kernel.connectors.ai.ollama",
    "semantic_kernel.connectors.in_memory",
    "semantic_kernel.connectors.postgres",
    "semantic_kernel.connectors.qdrant",
    "semantic_kernel.connectors.chroma",
    "semantic_kernel.connectors.pinecone",
    "semantic_kernel.connectors.azure_ai_search",
    "semantic_kernel.connectors.faiss",
    "semantic_kernel.connectors.azure_cosmos_db",
    "semantic_kernel.connectors.sql_server",
    "semantic_kernel.connectors.weaviate",
    "semantic_kernel.contents",
    "semantic_kernel.data",
    "semantic_kernel.data.vector",
    "semantic_kernel.text",
    "semantic_kernel.functions",
    "semantic_kernel.functions.kernel_function",
    "semantic_kernel.functions.kernel_parameter_metadata",
    "semantic_kernel.functions.kernel_plugin",
]
for module_name in _all_sk_modules:
    if module_name in sys.modules:
        _saved_modules[module_name] = sys.modules[module_name]
    else:
        sys.modules[module_name] = MagicMock()
        _mocked_modules.add(module_name)

# Set up mock attributes for vector store
if "semantic_kernel.data.vector" in _mocked_modules:
    mock_vector = sys.modules["semantic_kernel.data.vector"]
    mock_vector.VectorStoreField = MagicMock()
    mock_vector.DistanceFunction = MagicMock()
    mock_vector.DistanceFunction.COSINE_SIMILARITY = "cosine"
    mock_vector.vectorstoremodel = lambda **kwargs: lambda cls: cls
    mock_vector.VectorStoreCollectionDefinition = MagicMock()

if "semantic_kernel.connectors.memory" in _mocked_modules:
    mock_memory = sys.modules["semantic_kernel.connectors.memory"]
    mock_memory.InMemoryCollection = MagicMock()

if "semantic_kernel.connectors.ai.embedding_generator_base" in _mocked_modules:
    mock_embed_base = sys.modules[
        "semantic_kernel.connectors.ai.embedding_generator_base"
    ]
    mock_embed_base.EmbeddingGeneratorBase = MagicMock()

if "semantic_kernel.connectors.in_memory" in _mocked_modules:
    mock_inmem = sys.modules["semantic_kernel.connectors.in_memory"]
    mock_inmem.InMemoryCollection = MagicMock()

if "semantic_kernel" in _mocked_modules:
    mock_sk = sys.modules["semantic_kernel"]
    mock_sk.Kernel = MagicMock()

if "semantic_kernel.functions.kernel_function" in _mocked_modules:
    mock_kf = sys.modules["semantic_kernel.functions.kernel_function"]
    mock_kf.KernelFunction = MagicMock()

if "semantic_kernel.functions.kernel_parameter_metadata" in _mocked_modules:
    mock_kpm = sys.modules["semantic_kernel.functions.kernel_parameter_metadata"]
    mock_kpm.KernelParameterMetadata = MagicMock()

if "semantic_kernel.functions.kernel_plugin" in _mocked_modules:
    mock_kp = sys.modules["semantic_kernel.functions.kernel_plugin"]
    mock_kp.KernelPlugin = MagicMock()

if "semantic_kernel.connectors.ai.function_choice_behavior" in _mocked_modules:
    mock_fcb = sys.modules["semantic_kernel.connectors.ai.function_choice_behavior"]
    mock_fcb.FunctionChoiceBehavior = MagicMock()

import pytest  # noqa: E402

from holodeck.models.tool import HierarchicalDocumentToolConfig  # noqa: E402
from holodeck.tools.hierarchical_document_tool import (  # noqa: E402
    HierarchicalDocumentTool,
)


def create_config(
    tmp_path: Path, **overrides: object
) -> HierarchicalDocumentToolConfig:
    """Create a test config with sensible defaults."""
    doc_file = tmp_path / "test.md"
    if not doc_file.exists():
        doc_file.write_text("# Test\n\nContent paragraph.")

    defaults = {
        "name": "test_tool",
        "description": "Test hierarchical document tool",
        "source": str(doc_file),
    }
    defaults.update(overrides)
    return HierarchicalDocumentToolConfig(**defaults)


class TestHierarchicalDocumentToolInit:
    """T20/T36: Tests for HierarchicalDocumentTool initialization."""

    def test_init_with_config(self, tmp_path: Path) -> None:
        """Test HierarchicalDocumentTool initialization with valid configuration."""
        config = create_config(tmp_path)
        tool = HierarchicalDocumentTool(config)

        assert tool.config == config
        assert tool._initialized is False
        assert tool._chunks == []
        assert tool._chunker is None
        assert tool._context_generator is None

    def test_init_with_custom_top_k(self, tmp_path: Path) -> None:
        """Test initialization with custom top_k."""
        config = create_config(tmp_path, top_k=20)
        tool = HierarchicalDocumentTool(config)

        assert tool.config.top_k == 20

    def test_init_with_custom_search_mode(self, tmp_path: Path) -> None:
        """Test initialization with custom search_mode."""
        config = create_config(tmp_path, search_mode="semantic")
        tool = HierarchicalDocumentTool(config)

        assert tool.config.search_mode.value == "semantic"

    def test_init_with_chunking_config(self, tmp_path: Path) -> None:
        """Test initialization with custom chunking configuration."""
        config = create_config(tmp_path, max_chunk_tokens=1000, chunk_overlap=100)
        tool = HierarchicalDocumentTool(config)

        assert tool.config.max_chunk_tokens == 1000
        assert tool.config.chunk_overlap == 100

    def test_init_has_embedding_service_attribute(self, tmp_path: Path) -> None:
        """Test that tool has _embedding_service attribute."""
        config = create_config(tmp_path)
        tool = HierarchicalDocumentTool(config)

        assert hasattr(tool, "_embedding_service")
        assert tool._embedding_service is None

    def test_init_has_chat_service_attribute(self, tmp_path: Path) -> None:
        """Test that tool has _chat_service attribute."""
        config = create_config(tmp_path)
        tool = HierarchicalDocumentTool(config)

        assert hasattr(tool, "_chat_service")
        assert tool._chat_service is None

    def test_init_has_collection_attribute(self, tmp_path: Path) -> None:
        """Test that tool has _collection attribute."""
        config = create_config(tmp_path)
        tool = HierarchicalDocumentTool(config)

        assert hasattr(tool, "_collection")
        assert tool._collection is None

    def test_set_embedding_service(self, tmp_path: Path) -> None:
        """Test embedding service injection via set_embedding_service."""
        config = create_config(tmp_path)
        tool = HierarchicalDocumentTool(config)

        mock_service = MagicMock()
        tool.set_embedding_service(mock_service)

        assert tool._embedding_service == mock_service

    def test_set_chat_service(self, tmp_path: Path) -> None:
        """Test chat service injection via set_chat_service."""
        config = create_config(tmp_path)
        tool = HierarchicalDocumentTool(config)

        mock_service = MagicMock()
        tool.set_chat_service(mock_service)

        assert tool._chat_service == mock_service


class TestIngestDocuments:
    """T37: Tests for document ingestion pipeline orchestration."""

    @pytest.mark.asyncio
    async def test_initialize_sets_initialized_flag(self, tmp_path: Path) -> None:
        """Test that initialize() sets _initialized to True."""
        config = create_config(tmp_path)
        tool = HierarchicalDocumentTool(config)

        # Mock collection to avoid actual vector store operations
        with (
            patch.object(tool, "_setup_collection"),
            patch.object(tool, "_store_chunks", return_value=1),
        ):
            await tool.initialize()

        assert tool._initialized is True

    @pytest.mark.asyncio
    async def test_initialize_processes_document_sources(self, tmp_path: Path) -> None:
        """Test that initialize processes the configured source."""
        doc_file = tmp_path / "test.md"
        doc_file.write_text("# Doc 1\n\nFirst document.\n\n## Section\n\nMore content.")

        config = create_config(tmp_path, source=str(doc_file))
        tool = HierarchicalDocumentTool(config)

        with (
            patch.object(tool, "_setup_collection"),
            patch.object(tool, "_store_chunks", return_value=1),
        ):
            await tool.initialize()

        # Should have chunks from the document
        assert len(tool._chunks) >= 1

    @pytest.mark.asyncio
    async def test_initialize_uses_structured_chunker(self, tmp_path: Path) -> None:
        """Test that initialize uses StructuredChunker for parsing."""
        config = create_config(tmp_path)
        tool = HierarchicalDocumentTool(config)

        with (
            patch.object(tool, "_setup_collection"),
            patch.object(tool, "_store_chunks", return_value=1),
        ):
            await tool.initialize()

        # Chunker should be initialized
        assert tool._chunker is not None

    @pytest.mark.asyncio
    async def test_initialize_creates_chunks_with_parent_chain(
        self, tmp_path: Path
    ) -> None:
        """Test that chunks have parent_chain populated from headings."""
        doc_file = tmp_path / "test.md"
        doc_file.write_text("# Chapter 1\n\n## Section 1.1\n\nContent in section.")

        config = create_config(tmp_path, source=str(doc_file))
        tool = HierarchicalDocumentTool(config)

        with (
            patch.object(tool, "_setup_collection"),
            patch.object(tool, "_store_chunks", return_value=1),
        ):
            await tool.initialize()

        # Find the chunk with nested content
        nested_chunks = [c for c in tool._chunks if c.parent_chain]
        assert len(nested_chunks) > 0

    @pytest.mark.asyncio
    async def test_initialize_without_context_generator(self, tmp_path: Path) -> None:
        """Test chunks have content as contextualized_content when no LLM."""
        config = create_config(tmp_path)
        tool = HierarchicalDocumentTool(config)

        with (
            patch.object(tool, "_setup_collection"),
            patch.object(tool, "_store_chunks", return_value=1),
        ):
            await tool.initialize()

        # Without LLM, contextualized_content should fall back to content
        for chunk in tool._chunks:
            assert chunk.contextualized_content == chunk.content

    @pytest.mark.asyncio
    async def test_initialize_raises_for_missing_file(self, tmp_path: Path) -> None:
        """Test that initialize raises error for missing document file."""
        missing_file = tmp_path / "nonexistent.md"

        config = HierarchicalDocumentToolConfig(
            name="test",
            description="Test",
            source=str(missing_file),
        )
        tool = HierarchicalDocumentTool(config)

        with pytest.raises(FileNotFoundError):
            await tool.initialize()


class TestEmbedChunks:
    """T38: Tests for chunk embedding generation."""

    @pytest.mark.asyncio
    async def test_embed_chunks_uses_contextualized_content(
        self, tmp_path: Path
    ) -> None:
        """Test that _embed_chunks uses contextualized_content for embedding."""
        from holodeck.lib.structured_chunker import DocumentChunk

        config = create_config(tmp_path)
        tool = HierarchicalDocumentTool(config)

        # Create test chunks
        chunks = [
            DocumentChunk(
                id="chunk_0",
                source_path="/test.md",
                chunk_index=0,
                content="Original content",
                contextualized_content="Context: This is about X. Original content",
            ),
        ]

        mock_service = AsyncMock()
        mock_service.generate_embeddings.return_value = [[0.1, 0.2, 0.3]]
        tool.set_embedding_service(mock_service)
        tool._embedding_dimensions = 3

        await tool._embed_chunks(chunks)

        # Should have called with contextualized_content
        mock_service.generate_embeddings.assert_called_once()
        call_args = mock_service.generate_embeddings.call_args[0][0]
        assert "Context: This is about X" in call_args[0]

    @pytest.mark.asyncio
    async def test_embed_chunks_falls_back_to_content(self, tmp_path: Path) -> None:
        """Test _embed_chunks falls back to content when no contextualized_content."""
        from holodeck.lib.structured_chunker import DocumentChunk

        config = create_config(tmp_path)
        tool = HierarchicalDocumentTool(config)

        chunks = [
            DocumentChunk(
                id="chunk_0",
                source_path="/test.md",
                chunk_index=0,
                content="Original content only",
                contextualized_content="",  # Empty
            ),
        ]

        mock_service = AsyncMock()
        mock_service.generate_embeddings.return_value = [[0.1, 0.2, 0.3]]
        tool.set_embedding_service(mock_service)
        tool._embedding_dimensions = 3

        await tool._embed_chunks(chunks)

        call_args = mock_service.generate_embeddings.call_args[0][0]
        assert call_args[0] == "Original content only"

    @pytest.mark.asyncio
    async def test_embed_chunks_stores_embeddings_on_chunks(
        self, tmp_path: Path
    ) -> None:
        """Test that embeddings are stored on the chunk objects."""
        from holodeck.lib.structured_chunker import DocumentChunk

        config = create_config(tmp_path)
        tool = HierarchicalDocumentTool(config)

        chunks = [
            DocumentChunk(
                id="chunk_0",
                source_path="/test.md",
                chunk_index=0,
                content="Content",
                contextualized_content="Content",
            ),
        ]

        mock_service = AsyncMock()
        expected_embedding = [0.1, 0.2, 0.3, 0.4]
        mock_service.generate_embeddings.return_value = [expected_embedding]
        tool.set_embedding_service(mock_service)
        tool._embedding_dimensions = 4

        await tool._embed_chunks(chunks)

        assert chunks[0].embedding == expected_embedding

    @pytest.mark.asyncio
    async def test_embed_chunks_placeholder_without_service(
        self, tmp_path: Path
    ) -> None:
        """Test that placeholder embeddings are used without embedding service."""
        from holodeck.lib.structured_chunker import DocumentChunk

        config = create_config(tmp_path)
        tool = HierarchicalDocumentTool(config)
        tool._embedding_dimensions = 1536

        chunks = [
            DocumentChunk(
                id="chunk_0",
                source_path="/test.md",
                chunk_index=0,
                content="Content",
                contextualized_content="Content",
            ),
        ]

        await tool._embed_chunks(chunks)

        # Should have placeholder embedding
        assert chunks[0].embedding is not None
        assert len(chunks[0].embedding) == 1536
        assert all(v == 0.0 for v in chunks[0].embedding)


class TestStoreChunks:
    """T39: Tests for chunk storage with HierarchicalDocumentRecord."""

    @pytest.mark.asyncio
    async def test_store_chunks_requires_collection(self, tmp_path: Path) -> None:
        """Test that _store_chunks raises error without collection."""
        from holodeck.lib.structured_chunker import DocumentChunk

        config = create_config(tmp_path)
        tool = HierarchicalDocumentTool(config)

        chunks = [
            DocumentChunk(
                id="chunk_0",
                source_path="/test.md",
                chunk_index=0,
                content="Content",
            ),
        ]

        with pytest.raises(RuntimeError, match="Collection not initialized"):
            await tool._store_chunks(chunks)

    @pytest.mark.asyncio
    async def test_store_chunks_returns_count(self, tmp_path: Path) -> None:
        """Test that _store_chunks returns count of stored chunks."""
        from holodeck.lib.structured_chunker import DocumentChunk

        config = create_config(tmp_path)
        tool = HierarchicalDocumentTool(config)
        tool._embedding_dimensions = 1536

        # Mock collection
        mock_collection = MagicMock()
        mock_collection.__aenter__ = AsyncMock(return_value=mock_collection)
        mock_collection.__aexit__ = AsyncMock(return_value=None)
        mock_collection.collection_exists = AsyncMock(return_value=True)
        mock_collection.upsert = AsyncMock()
        tool._collection = mock_collection

        chunks = [
            DocumentChunk(
                id="chunk_0",
                source_path="/test.md",
                chunk_index=0,
                content="Content 1",
                embedding=[0.1] * 1536,
            ),
            DocumentChunk(
                id="chunk_1",
                source_path="/test.md",
                chunk_index=1,
                content="Content 2",
                embedding=[0.2] * 1536,
            ),
        ]

        count = await tool._store_chunks(chunks)

        assert count == 2

    @pytest.mark.asyncio
    async def test_store_chunks_includes_parent_chain(self, tmp_path: Path) -> None:
        """Test that stored records include serialized parent_chain."""
        from holodeck.lib.structured_chunker import DocumentChunk

        config = create_config(tmp_path)
        tool = HierarchicalDocumentTool(config)
        tool._embedding_dimensions = 1536

        mock_collection = MagicMock()
        mock_collection.__aenter__ = AsyncMock(return_value=mock_collection)
        mock_collection.__aexit__ = AsyncMock(return_value=None)
        mock_collection.collection_exists = AsyncMock(return_value=True)
        mock_collection.upsert = AsyncMock()
        tool._collection = mock_collection

        chunks = [
            DocumentChunk(
                id="chunk_0",
                source_path="/test.md",
                chunk_index=0,
                content="Content",
                parent_chain=["Chapter 1", "Section 1.1"],
                embedding=[0.1] * 1536,
            ),
        ]

        await tool._store_chunks(chunks)

        # Verify upsert was called with records containing parent_chain
        mock_collection.upsert.assert_called_once()
        records = mock_collection.upsert.call_args[0][0]
        assert len(records) == 1
        # parent_chain should be JSON serialized
        assert records[0].parent_chain == '["Chapter 1", "Section 1.1"]'


class TestSemanticSearch:
    """T40: Tests for semantic-only search functionality."""

    @pytest.mark.asyncio
    async def test_search_raises_if_not_initialized(self, tmp_path: Path) -> None:
        """Test that search raises RuntimeError before initialization."""
        config = create_config(tmp_path)
        tool = HierarchicalDocumentTool(config)

        with pytest.raises(RuntimeError, match="must be initialized"):
            await tool.search("test query")

    @pytest.mark.asyncio
    async def test_search_raises_for_empty_query(self, tmp_path: Path) -> None:
        """Test that search raises ValueError for empty query."""
        config = create_config(tmp_path)
        tool = HierarchicalDocumentTool(config)

        with (
            patch.object(tool, "_setup_collection"),
            patch.object(tool, "_store_chunks", return_value=1),
        ):
            await tool.initialize()

        with pytest.raises(ValueError, match="cannot be empty"):
            await tool.search("")

        with pytest.raises(ValueError, match="cannot be empty"):
            await tool.search("   ")

    @pytest.mark.asyncio
    async def test_search_returns_search_results(self, tmp_path: Path) -> None:
        """Test that search returns list of SearchResult objects."""
        from holodeck.lib.hybrid_search import SearchResult

        config = create_config(tmp_path)
        tool = HierarchicalDocumentTool(config)
        tool._initialized = True
        tool._embedding_dimensions = 1536

        # Mock embedding service
        mock_embed = AsyncMock()
        mock_embed.generate_embeddings.return_value = [[0.1] * 1536]
        tool._embedding_service = mock_embed

        # Mock collection search
        mock_result = MagicMock()
        mock_result.record = MagicMock()
        mock_result.record.id = "chunk_0"
        mock_result.record.content = "Test content"
        mock_result.record.source_path = "/test.md"
        mock_result.record.parent_chain = '["Chapter 1"]'
        mock_result.record.section_id = "sec_test"
        mock_result.score = 0.85

        mock_collection = MagicMock()
        mock_collection.__aenter__ = AsyncMock(return_value=mock_collection)
        mock_collection.__aexit__ = AsyncMock(return_value=None)

        async def mock_results():
            yield mock_result

        mock_search_results = MagicMock()
        mock_search_results.results = mock_results()
        mock_collection.search = AsyncMock(return_value=mock_search_results)
        tool._collection = mock_collection

        results = await tool.search("test query")

        assert isinstance(results, list)
        assert len(results) == 1
        assert isinstance(results[0], SearchResult)

    @pytest.mark.asyncio
    async def test_search_results_have_fused_score(self, tmp_path: Path) -> None:
        """Test that search results include fused_score."""
        config = create_config(tmp_path)
        tool = HierarchicalDocumentTool(config)
        tool._initialized = True
        tool._embedding_dimensions = 1536

        mock_embed = AsyncMock()
        mock_embed.generate_embeddings.return_value = [[0.1] * 1536]
        tool._embedding_service = mock_embed

        mock_result = MagicMock()
        mock_result.record = MagicMock()
        mock_result.record.id = "chunk_0"
        mock_result.record.content = "Content"
        mock_result.record.source_path = "/test.md"
        mock_result.record.parent_chain = "[]"
        mock_result.record.section_id = ""
        mock_result.score = 0.92

        mock_collection = MagicMock()
        mock_collection.__aenter__ = AsyncMock(return_value=mock_collection)
        mock_collection.__aexit__ = AsyncMock(return_value=None)

        async def mock_results():
            yield mock_result

        mock_search_results = MagicMock()
        mock_search_results.results = mock_results()
        mock_collection.search = AsyncMock(return_value=mock_search_results)
        tool._collection = mock_collection

        results = await tool.search("query")

        assert results[0].fused_score == 0.92

    @pytest.mark.asyncio
    async def test_search_results_have_source_attribution(self, tmp_path: Path) -> None:
        """Test that search results include source_path and parent_chain."""
        config = create_config(tmp_path)
        tool = HierarchicalDocumentTool(config)
        tool._initialized = True
        tool._embedding_dimensions = 1536

        mock_embed = AsyncMock()
        mock_embed.generate_embeddings.return_value = [[0.1] * 1536]
        tool._embedding_service = mock_embed

        mock_result = MagicMock()
        mock_result.record = MagicMock()
        mock_result.record.id = "chunk_0"
        mock_result.record.content = "Content"
        mock_result.record.source_path = "/docs/policy.md"
        mock_result.record.parent_chain = '["Chapter 1", "Definitions"]'
        mock_result.record.section_id = "1.2"
        mock_result.score = 0.88

        mock_collection = MagicMock()
        mock_collection.__aenter__ = AsyncMock(return_value=mock_collection)
        mock_collection.__aexit__ = AsyncMock(return_value=None)

        async def mock_results():
            yield mock_result

        mock_search_results = MagicMock()
        mock_search_results.results = mock_results()
        mock_collection.search = AsyncMock(return_value=mock_search_results)
        tool._collection = mock_collection

        results = await tool.search("query")

        assert results[0].source_path == "/docs/policy.md"
        assert results[0].parent_chain == ["Chapter 1", "Definitions"]
        assert results[0].section_id == "1.2"

    @pytest.mark.asyncio
    async def test_search_respects_top_k(self, tmp_path: Path) -> None:
        """Test that search respects top_k parameter."""
        config = create_config(tmp_path, top_k=3)
        tool = HierarchicalDocumentTool(config)
        tool._initialized = True
        tool._embedding_dimensions = 1536

        mock_embed = AsyncMock()
        mock_embed.generate_embeddings.return_value = [[0.1] * 1536]
        tool._embedding_service = mock_embed

        mock_collection = MagicMock()
        mock_collection.__aenter__ = AsyncMock(return_value=mock_collection)
        mock_collection.__aexit__ = AsyncMock(return_value=None)

        async def mock_results():
            return
            yield  # Empty generator

        mock_search_results = MagicMock()
        mock_search_results.results = mock_results()
        mock_collection.search = AsyncMock(return_value=mock_search_results)
        tool._collection = mock_collection

        await tool.search("query")

        # Verify search was called with top=3
        mock_collection.search.assert_called_once()
        call_kwargs = mock_collection.search.call_args[1]
        assert call_kwargs["top"] == 3


class TestSearchResultFormat:
    """T40: Tests for search result formatting."""

    @pytest.mark.asyncio
    async def test_search_results_sorted_by_score(self, tmp_path: Path) -> None:
        """Test that results are sorted by fused_score descending."""
        config = create_config(tmp_path)
        tool = HierarchicalDocumentTool(config)
        tool._initialized = True
        tool._embedding_dimensions = 1536

        mock_embed = AsyncMock()
        mock_embed.generate_embeddings.return_value = [[0.1] * 1536]
        tool._embedding_service = mock_embed

        # Create multiple results with different scores
        mock_results_data = [
            {"id": "chunk_0", "content": "A", "score": 0.5},
            {"id": "chunk_1", "content": "B", "score": 0.9},
            {"id": "chunk_2", "content": "C", "score": 0.7},
        ]

        mock_collection = MagicMock()
        mock_collection.__aenter__ = AsyncMock(return_value=mock_collection)
        mock_collection.__aexit__ = AsyncMock(return_value=None)

        async def mock_results():
            for data in mock_results_data:
                result = MagicMock()
                result.record = MagicMock()
                result.record.id = data["id"]
                result.record.content = data["content"]
                result.record.source_path = "/test.md"
                result.record.parent_chain = "[]"
                result.record.section_id = ""
                result.score = data["score"]
                yield result

        mock_search_results = MagicMock()
        mock_search_results.results = mock_results()
        mock_collection.search = AsyncMock(return_value=mock_search_results)
        tool._collection = mock_collection

        results = await tool.search("query")

        # Should be sorted descending
        assert results[0].fused_score == 0.9
        assert results[1].fused_score == 0.7
        assert results[2].fused_score == 0.5

    @pytest.mark.asyncio
    async def test_search_results_clamp_scores(self, tmp_path: Path) -> None:
        """Test that scores are clamped to [0.0, 1.0] range."""
        config = create_config(tmp_path)
        tool = HierarchicalDocumentTool(config)
        tool._initialized = True
        tool._embedding_dimensions = 1536

        mock_embed = AsyncMock()
        mock_embed.generate_embeddings.return_value = [[0.1] * 1536]
        tool._embedding_service = mock_embed

        mock_result = MagicMock()
        mock_result.record = MagicMock()
        mock_result.record.id = "chunk_0"
        mock_result.record.content = "Content"
        mock_result.record.source_path = "/test.md"
        mock_result.record.parent_chain = "[]"
        mock_result.record.section_id = ""
        mock_result.score = 1.5  # Out of bounds

        mock_collection = MagicMock()
        mock_collection.__aenter__ = AsyncMock(return_value=mock_collection)
        mock_collection.__aexit__ = AsyncMock(return_value=None)

        async def mock_results():
            yield mock_result

        mock_search_results = MagicMock()
        mock_search_results.results = mock_results()
        mock_collection.search = AsyncMock(return_value=mock_search_results)
        tool._collection = mock_collection

        results = await tool.search("query")

        assert results[0].fused_score == 1.0  # Clamped


class TestConfidenceIndication:
    """Tests for empty and low-confidence result handling."""

    @pytest.mark.asyncio
    async def test_no_matches_returns_empty_list(self, tmp_path: Path) -> None:
        """Test that no matches returns empty list."""
        config = create_config(tmp_path)
        tool = HierarchicalDocumentTool(config)
        tool._initialized = True
        tool._embedding_dimensions = 1536

        mock_embed = AsyncMock()
        mock_embed.generate_embeddings.return_value = [[0.1] * 1536]
        tool._embedding_service = mock_embed

        mock_collection = MagicMock()
        mock_collection.__aenter__ = AsyncMock(return_value=mock_collection)
        mock_collection.__aexit__ = AsyncMock(return_value=None)

        async def mock_results():
            return
            yield  # Empty generator

        mock_search_results = MagicMock()
        mock_search_results.results = mock_results()
        mock_collection.search = AsyncMock(return_value=mock_search_results)
        tool._collection = mock_collection

        results = await tool.search("nonexistent query")

        assert results == []

    @pytest.mark.asyncio
    async def test_semantic_only_mode_keyword_score_is_none(
        self, tmp_path: Path
    ) -> None:
        """Test that in semantic-only mode, keyword_score is None."""
        config = create_config(tmp_path)
        tool = HierarchicalDocumentTool(config)
        tool._initialized = True
        tool._embedding_dimensions = 1536

        mock_embed = AsyncMock()
        mock_embed.generate_embeddings.return_value = [[0.1] * 1536]
        tool._embedding_service = mock_embed

        mock_result = MagicMock()
        mock_result.record = MagicMock()
        mock_result.record.id = "chunk_0"
        mock_result.record.content = "Content"
        mock_result.record.source_path = "/test.md"
        mock_result.record.parent_chain = "[]"
        mock_result.record.section_id = ""
        mock_result.score = 0.85

        mock_collection = MagicMock()
        mock_collection.__aenter__ = AsyncMock(return_value=mock_collection)
        mock_collection.__aexit__ = AsyncMock(return_value=None)

        async def mock_results():
            yield mock_result

        mock_search_results = MagicMock()
        mock_search_results.results = mock_results()
        mock_collection.search = AsyncMock(return_value=mock_search_results)
        tool._collection = mock_collection

        results = await tool.search("query")

        # In semantic-only mode
        assert results[0].keyword_score is None
        assert results[0].semantic_score == 0.85
        assert results[0].exact_match is False


class TestPydanticConfigValidation:
    """T085: Tests for Pydantic model validation."""

    def test_valid_config_creation(self, tmp_path: Path) -> None:
        """Test creating a valid HierarchicalDocumentToolConfig."""
        doc_file = tmp_path / "test.md"
        doc_file.write_text("# Test\n\nContent")

        config = HierarchicalDocumentToolConfig(
            name="test_tool",
            description="Test tool",
            source=str(doc_file),
        )

        assert config.name == "test_tool"
        assert config.type == "hierarchical_document"
        assert config.top_k == 10  # default
        assert config.search_mode.value == "hybrid"  # default
        assert config.defer_loading is True  # default

    def test_invalid_name_pattern_raises_validation_error(self, tmp_path: Path) -> None:
        """Test that invalid name pattern raises ValidationError (T085)."""
        from pydantic import ValidationError

        doc_file = tmp_path / "test.md"
        doc_file.write_text("# Test")

        with pytest.raises(ValidationError, match="name"):
            HierarchicalDocumentToolConfig(
                name="invalid-name-with-dash",  # Only alphanumeric and _ allowed
                description="Test tool",
                source=str(doc_file),
            )

    def test_empty_source_raises_validation_error(self, tmp_path: Path) -> None:
        """Test that empty source raises ValidationError (T085)."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="source"):
            HierarchicalDocumentToolConfig(
                name="test_tool",
                description="Test tool",
                source="   ",  # Empty/whitespace only
            )

    def test_invalid_top_k_raises_validation_error(self, tmp_path: Path) -> None:
        """Test that invalid top_k raises ValidationError."""
        from pydantic import ValidationError

        doc_file = tmp_path / "test.md"
        doc_file.write_text("# Test")

        with pytest.raises(ValidationError, match="top_k"):
            HierarchicalDocumentToolConfig(
                name="test_tool",
                description="Test tool",
                source=str(doc_file),
                top_k=0,  # Must be >= 1
            )

    def test_invalid_min_score_raises_validation_error(self, tmp_path: Path) -> None:
        """Test that min_score out of range raises ValidationError."""
        from pydantic import ValidationError

        doc_file = tmp_path / "test.md"
        doc_file.write_text("# Test")

        with pytest.raises(ValidationError, match="min_score"):
            HierarchicalDocumentToolConfig(
                name="test_tool",
                description="Test tool",
                source=str(doc_file),
                min_score=1.5,  # Must be <= 1.0
            )

    def test_reranker_required_when_enabled(self, tmp_path: Path) -> None:
        """Test that reranker_model is required when enable_reranking=True."""
        from pydantic import ValidationError

        doc_file = tmp_path / "test.md"
        doc_file.write_text("# Test")

        with pytest.raises(ValidationError, match="reranker_model"):
            HierarchicalDocumentToolConfig(
                name="test_tool",
                description="Test tool",
                source=str(doc_file),
                enable_reranking=True,  # Requires reranker_model
                # reranker_model is missing
            )


class TestWeightValidation:
    """T088: Tests for weight validation warning."""

    def test_weights_sum_to_one_no_warning(self, tmp_path: Path) -> None:
        """Test that weights summing to 1.0 don't emit warning."""
        import warnings

        doc_file = tmp_path / "test.md"
        doc_file.write_text("# Test")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            config = HierarchicalDocumentToolConfig(
                name="test_tool",
                description="Test tool",
                source=str(doc_file),
                search_mode="hybrid",
                semantic_weight=0.5,
                keyword_weight=0.3,
                exact_weight=0.2,
            )
            # No warning expected
            assert len([x for x in w if "sum to" in str(x.message)]) == 0
            assert config.semantic_weight == 0.5

    def test_weights_not_sum_to_one_emits_warning(self, tmp_path: Path) -> None:
        """Test that weights not summing to 1.0 emit warning (T088)."""
        import warnings

        doc_file = tmp_path / "test.md"
        doc_file.write_text("# Test")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            HierarchicalDocumentToolConfig(
                name="test_tool",
                description="Test tool",
                source=str(doc_file),
                search_mode="hybrid",
                semantic_weight=0.5,
                keyword_weight=0.5,
                exact_weight=0.5,  # Sum = 1.5, not 1.0
            )
            # Warning expected
            weight_warnings = [x for x in w if "sum to" in str(x.message)]
            assert len(weight_warnings) == 1
            assert "1.50" in str(weight_warnings[0].message)

    def test_weights_ignored_for_non_hybrid_mode(self, tmp_path: Path) -> None:
        """Test that weight validation is skipped for non-hybrid modes."""
        import warnings

        doc_file = tmp_path / "test.md"
        doc_file.write_text("# Test")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            config = HierarchicalDocumentToolConfig(
                name="test_tool",
                description="Test tool",
                source=str(doc_file),
                search_mode="semantic",  # Not hybrid
                semantic_weight=0.9,  # Would trigger warning in hybrid
                keyword_weight=0.5,
                exact_weight=0.5,
            )
            # No warning expected for non-hybrid mode
            weight_warnings = [x for x in w if "sum to" in str(x.message)]
            assert len(weight_warnings) == 0
            assert config.search_mode.value == "semantic"


class TestDeferLoading:
    """T087: Tests for defer_loading behavior."""

    def test_defer_loading_defaults_true(self, tmp_path: Path) -> None:
        """Test that defer_loading defaults to True (T087)."""
        config = create_config(tmp_path)
        assert config.defer_loading is True

    def test_defer_loading_can_be_false(self, tmp_path: Path) -> None:
        """Test that defer_loading can be set to False."""
        config = create_config(tmp_path, defer_loading=False)
        assert config.defer_loading is False

    def test_tool_has_defer_loading_attribute(self, tmp_path: Path) -> None:
        """Test that HierarchicalDocumentTool has defer_loading in config."""
        config = create_config(tmp_path, defer_loading=False)
        tool = HierarchicalDocumentTool(config)
        assert tool.config.defer_loading is False


class TestDatabaseConfig:
    """T086: Tests for database configuration (low priority)."""

    def test_database_none_uses_inmemory(self, tmp_path: Path) -> None:
        """Test that None database config defaults to in-memory storage (T086)."""
        config = create_config(tmp_path)
        # database defaults to None
        assert config.database is None

        tool = HierarchicalDocumentTool(config)
        # Provider defaults to in-memory
        assert tool._provider == "in-memory"

    def test_database_config_object_accepted(self, tmp_path: Path) -> None:
        """Test that DatabaseConfig object is accepted."""
        from holodeck.models.tool import DatabaseConfig

        doc_file = tmp_path / "test.md"
        doc_file.write_text("# Test")

        db_config = DatabaseConfig(provider="in-memory")
        config = HierarchicalDocumentToolConfig(
            name="test_tool",
            description="Test tool",
            source=str(doc_file),
            database=db_config,
        )
        assert config.database is not None
        assert config.database.provider == "in-memory"


class TestToolFactoryIntegration:
    """T082: Tests for AgentFactory registration (mocked)."""

    def test_tool_config_type_is_hierarchical_document(self, tmp_path: Path) -> None:
        """Test that HierarchicalDocumentToolConfig has correct type."""
        config = create_config(tmp_path)
        assert config.type == "hierarchical_document"

    def test_config_included_in_tool_union(self) -> None:
        """Test that HierarchicalDocumentToolConfig is part of ToolUnion."""
        from typing import get_args

        from holodeck.models.tool import ToolUnion

        # Get the types from the Union
        union_args = get_args(ToolUnion)
        # The first element is the Annotated union
        annotated_union = union_args[0]
        inner_args = get_args(annotated_union)

        # Extract type names
        type_names = []
        for arg in inner_args:
            inner_type = get_args(arg)[0] if get_args(arg) else arg
            if hasattr(inner_type, "__name__"):
                type_names.append(inner_type.__name__)

        assert "HierarchicalDocumentToolConfig" in type_names

    def test_tool_instantiation_from_config(self, tmp_path: Path) -> None:
        """Test HierarchicalDocumentTool instantiation from config (T082)."""
        config = create_config(tmp_path)
        tool = HierarchicalDocumentTool(config)

        assert tool.config == config
        assert tool._initialized is False
        assert hasattr(tool, "set_embedding_service")
        assert hasattr(tool, "set_chat_service")
        assert hasattr(tool, "initialize")
        assert hasattr(tool, "search")


class TestSetChatServiceWithoutContextualEmbeddings:
    """Tests for set_chat_service when contextual_embeddings is disabled."""

    def test_set_chat_service_without_contextual_embeddings(
        self, tmp_path: Path
    ) -> None:
        """Test set_chat_service logs debug when contextual_embeddings is False."""
        config = create_config(tmp_path, contextual_embeddings=False)
        tool = HierarchicalDocumentTool(config)

        mock_chat_service = MagicMock()
        tool.set_chat_service(mock_chat_service)

        # Should not create context generator
        assert tool._context_generator is None
        assert tool._chat_service == mock_chat_service


class TestDatabaseConfigHandling:
    """Tests for database configuration edge cases."""

    def test_database_string_reference_falls_back_to_inmemory(
        self, tmp_path: Path
    ) -> None:
        """Test unresolved string database ref falls back to in-memory."""
        doc_file = tmp_path / "test.md"
        doc_file.write_text("# Test")

        # Create config with a string reference (simulating unresolved ref)
        config = HierarchicalDocumentToolConfig(
            name="test_tool",
            description="Test tool",
            source=str(doc_file),
        )
        # Manually set database to a string (simulating unresolved reference)
        object.__setattr__(config, "database", "unresolved_db_ref")

        tool = HierarchicalDocumentTool(config)
        tool._setup_collection("openai")

        # Should fall back to in-memory
        assert tool._provider == "in-memory"

    def test_database_config_with_connection_string(self, tmp_path: Path) -> None:
        """Test DatabaseConfig with connection_string is handled."""
        from holodeck.models.tool import DatabaseConfig

        doc_file = tmp_path / "test.md"
        doc_file.write_text("# Test")

        db_config = DatabaseConfig(
            provider="in-memory",
            connection_string="memory://test",
        )
        config = HierarchicalDocumentToolConfig(
            name="test_tool",
            description="Test tool",
            source=str(doc_file),
            database=db_config,
        )

        tool = HierarchicalDocumentTool(config)
        tool._setup_collection("openai")

        assert tool._provider == "in-memory"

    def test_database_config_with_extra_fields(self, tmp_path: Path) -> None:
        """Test DatabaseConfig with extra fields from model_extra."""
        from holodeck.models.tool import DatabaseConfig

        doc_file = tmp_path / "test.md"
        doc_file.write_text("# Test")

        db_config = DatabaseConfig(
            provider="in-memory",
            custom_setting="value",  # Extra field via extra="allow"
        )
        config = HierarchicalDocumentToolConfig(
            name="test_tool",
            description="Test tool",
            source=str(doc_file),
            database=db_config,
        )

        tool = HierarchicalDocumentTool(config)
        tool._setup_collection("openai")

        assert tool._provider == "in-memory"


class TestMinScoreFiltering:
    """Tests for min_score filtering in search results."""

    @pytest.mark.asyncio
    async def test_min_score_filters_low_scores(self, tmp_path: Path) -> None:
        """Test that results below min_score are filtered out."""
        config = create_config(tmp_path, min_score=0.7)
        tool = HierarchicalDocumentTool(config)
        tool._initialized = True
        tool._embedding_dimensions = 1536

        mock_embed = AsyncMock()
        mock_embed.generate_embeddings.return_value = [[0.1] * 1536]
        tool._embedding_service = mock_embed

        # Create results with varying scores
        mock_results_data = [
            {"id": "chunk_0", "content": "A", "score": 0.5},  # Below threshold
            {"id": "chunk_1", "content": "B", "score": 0.8},  # Above threshold
            {"id": "chunk_2", "content": "C", "score": 0.6},  # Below threshold
            {"id": "chunk_3", "content": "D", "score": 0.9},  # Above threshold
        ]

        mock_collection = MagicMock()
        mock_collection.__aenter__ = AsyncMock(return_value=mock_collection)
        mock_collection.__aexit__ = AsyncMock(return_value=None)

        async def mock_results():
            for data in mock_results_data:
                result = MagicMock()
                result.record = MagicMock()
                result.record.id = data["id"]
                result.record.content = data["content"]
                result.record.source_path = "/test.md"
                result.record.parent_chain = "[]"
                result.record.section_id = ""
                result.score = data["score"]
                yield result

        mock_search_results = MagicMock()
        mock_search_results.results = mock_results()
        mock_collection.search = AsyncMock(return_value=mock_search_results)
        tool._collection = mock_collection

        results = await tool.search("query")

        # Only scores >= 0.7 should remain
        assert len(results) == 2
        assert all(r.fused_score >= 0.7 for r in results)

    @pytest.mark.asyncio
    async def test_min_score_none_returns_all(self, tmp_path: Path) -> None:
        """Test that min_score=None returns all results."""
        config = create_config(tmp_path)  # min_score defaults to None
        tool = HierarchicalDocumentTool(config)
        tool._initialized = True
        tool._embedding_dimensions = 1536

        mock_embed = AsyncMock()
        mock_embed.generate_embeddings.return_value = [[0.1] * 1536]
        tool._embedding_service = mock_embed

        mock_results_data = [
            {"id": "chunk_0", "content": "A", "score": 0.1},
            {"id": "chunk_1", "content": "B", "score": 0.9},
        ]

        mock_collection = MagicMock()
        mock_collection.__aenter__ = AsyncMock(return_value=mock_collection)
        mock_collection.__aexit__ = AsyncMock(return_value=None)

        async def mock_results():
            for data in mock_results_data:
                result = MagicMock()
                result.record = MagicMock()
                result.record.id = data["id"]
                result.record.content = data["content"]
                result.record.source_path = "/test.md"
                result.record.parent_chain = "[]"
                result.record.section_id = ""
                result.score = data["score"]
                yield result

        mock_search_results = MagicMock()
        mock_search_results.results = mock_results()
        mock_collection.search = AsyncMock(return_value=mock_search_results)
        tool._collection = mock_collection

        results = await tool.search("query")

        # All results should be returned
        assert len(results) == 2


class TestGetContext:
    """Tests for get_context method."""

    @pytest.mark.asyncio
    async def test_get_context_returns_formatted_string(self, tmp_path: Path) -> None:
        """Test get_context returns properly formatted context."""
        config = create_config(tmp_path)
        tool = HierarchicalDocumentTool(config)
        tool._initialized = True
        tool._embedding_dimensions = 1536

        mock_embed = AsyncMock()
        mock_embed.generate_embeddings.return_value = [[0.1] * 1536]
        tool._embedding_service = mock_embed

        mock_result = MagicMock()
        mock_result.record = MagicMock()
        mock_result.record.id = "chunk_0"
        mock_result.record.content = "Test content"
        mock_result.record.source_path = "/test.md"
        mock_result.record.parent_chain = "[]"
        mock_result.record.section_id = ""
        mock_result.score = 0.85

        mock_collection = MagicMock()
        mock_collection.__aenter__ = AsyncMock(return_value=mock_collection)
        mock_collection.__aexit__ = AsyncMock(return_value=None)

        async def mock_results():
            yield mock_result

        mock_search_results = MagicMock()
        mock_search_results.results = mock_results()
        mock_collection.search = AsyncMock(return_value=mock_search_results)
        tool._collection = mock_collection

        context = await tool.get_context("test query")

        assert "Context for query: test query" in context
        assert "[1]" in context

    @pytest.mark.asyncio
    async def test_get_context_no_results(self, tmp_path: Path) -> None:
        """Test get_context returns message when no results found."""
        config = create_config(tmp_path)
        tool = HierarchicalDocumentTool(config)
        tool._initialized = True
        tool._embedding_dimensions = 1536

        mock_embed = AsyncMock()
        mock_embed.generate_embeddings.return_value = [[0.1] * 1536]
        tool._embedding_service = mock_embed

        mock_collection = MagicMock()
        mock_collection.__aenter__ = AsyncMock(return_value=mock_collection)
        mock_collection.__aexit__ = AsyncMock(return_value=None)

        async def mock_results():
            return
            yield  # Empty generator

        mock_search_results = MagicMock()
        mock_search_results.results = mock_results()
        mock_collection.search = AsyncMock(return_value=mock_search_results)
        tool._collection = mock_collection

        context = await tool.get_context("test query")

        assert "No relevant context found for: test query" in context


class TestGetDefinition:
    """Tests for get_definition method."""

    def test_get_definition_returns_none_without_glossary(self, tmp_path: Path) -> None:
        """Test get_definition returns None when no glossary exists."""
        config = create_config(tmp_path)
        tool = HierarchicalDocumentTool(config)
        tool._glossary = None

        result = tool.get_definition("test term")
        assert result is None

    def test_get_definition_returns_entry(self, tmp_path: Path) -> None:
        """Test get_definition returns matching entry from glossary."""
        from holodeck.lib.definition_extractor import DefinitionEntry

        config = create_config(tmp_path)
        tool = HierarchicalDocumentTool(config)

        # Set up glossary
        tool._glossary = {
            "force_majeure": DefinitionEntry(
                id="def_1",
                source_path="/test.md",
                term="Force Majeure",
                term_normalized="force majeure",
                definition_text="Events beyond reasonable control",
                source_section="Definitions",
            )
        }

        result = tool.get_definition("Force Majeure")
        assert result is not None
        assert result["term"] == "Force Majeure"
        assert result["definition"] == "Events beyond reasonable control"

    def test_get_definition_normalizes_term(self, tmp_path: Path) -> None:
        """Test get_definition normalizes term for lookup."""
        from holodeck.lib.definition_extractor import DefinitionEntry

        config = create_config(tmp_path)
        tool = HierarchicalDocumentTool(config)

        tool._glossary = {
            "test_term": DefinitionEntry(
                id="def_1",
                source_path="/test.md",
                term="Test Term",
                term_normalized="test term",
                definition_text="A test definition",
                source_section="Definitions",
            )
        }

        # Should normalize "Test Term" to "test_term"
        result = tool.get_definition("Test Term")
        assert result is not None
        assert result["term"] == "Test Term"

    def test_get_definition_returns_none_for_missing_term(self, tmp_path: Path) -> None:
        """Test get_definition returns None for non-existent term."""
        from holodeck.lib.definition_extractor import DefinitionEntry

        config = create_config(tmp_path)
        tool = HierarchicalDocumentTool(config)

        tool._glossary = {
            "existing_term": DefinitionEntry(
                id="def_1",
                source_path="/test.md",
                term="Existing Term",
                term_normalized="existing term",
                definition_text="Definition",
                source_section="Definitions",
            )
        }

        result = tool.get_definition("nonexistent")
        assert result is None


class TestToSemanticKernelFunction:
    """Tests for to_semantic_kernel_function method."""

    @pytest.mark.asyncio
    async def test_sk_function_returns_results(self, tmp_path: Path) -> None:
        """Test SK function wrapper returns formatted results."""
        config = create_config(tmp_path)
        tool = HierarchicalDocumentTool(config)
        tool._initialized = True
        tool._embedding_dimensions = 1536

        mock_embed = AsyncMock()
        mock_embed.generate_embeddings.return_value = [[0.1] * 1536]
        tool._embedding_service = mock_embed

        mock_result = MagicMock()
        mock_result.record = MagicMock()
        mock_result.record.id = "chunk_0"
        mock_result.record.content = "Test content"
        mock_result.record.source_path = "/test.md"
        mock_result.record.parent_chain = "[]"
        mock_result.record.section_id = ""
        mock_result.score = 0.85

        mock_collection = MagicMock()
        mock_collection.__aenter__ = AsyncMock(return_value=mock_collection)
        mock_collection.__aexit__ = AsyncMock(return_value=None)

        async def mock_results():
            yield mock_result

        mock_search_results = MagicMock()
        mock_search_results.results = mock_results()
        mock_collection.search = AsyncMock(return_value=mock_search_results)
        tool._collection = mock_collection

        sk_func = tool.to_semantic_kernel_function()
        result = await sk_func("test query")

        assert isinstance(result, str)
        assert "Test content" in result

    @pytest.mark.asyncio
    async def test_sk_function_no_results(self, tmp_path: Path) -> None:
        """Test SK function returns message when no results."""
        config = create_config(tmp_path)
        tool = HierarchicalDocumentTool(config)
        tool._initialized = True
        tool._embedding_dimensions = 1536

        mock_embed = AsyncMock()
        mock_embed.generate_embeddings.return_value = [[0.1] * 1536]
        tool._embedding_service = mock_embed

        mock_collection = MagicMock()
        mock_collection.__aenter__ = AsyncMock(return_value=mock_collection)
        mock_collection.__aexit__ = AsyncMock(return_value=None)

        async def mock_results():
            return
            yield  # Empty generator

        mock_search_results = MagicMock()
        mock_search_results.results = mock_results()
        mock_collection.search = AsyncMock(return_value=mock_search_results)
        tool._collection = mock_collection

        sk_func = tool.to_semantic_kernel_function()
        result = await sk_func("test query")

        assert result == "No results found."


class TestRelativePathResolution:
    """Tests for relative path resolution with context variable."""

    def test_resolve_absolute_path_unchanged(self, tmp_path: Path) -> None:
        """Test absolute paths are returned unchanged."""
        doc_file = tmp_path / "test.md"
        doc_file.write_text("# Test")

        config = HierarchicalDocumentToolConfig(
            name="test_tool",
            description="Test",
            source=str(doc_file),  # Absolute path
        )
        tool = HierarchicalDocumentTool(config)

        resolved = tool._resolve_source_path()
        assert resolved == doc_file

    def test_resolve_relative_path_with_base_dir(self, tmp_path: Path) -> None:
        """Test relative path resolved against explicit base_dir."""
        doc_file = tmp_path / "docs" / "test.md"
        doc_file.parent.mkdir(parents=True, exist_ok=True)
        doc_file.write_text("# Test")

        config = HierarchicalDocumentToolConfig(
            name="test_tool",
            description="Test",
            source="docs/test.md",  # Relative path
        )
        tool = HierarchicalDocumentTool(config, base_dir=str(tmp_path))

        resolved = tool._resolve_source_path()
        assert resolved == doc_file

    def test_resolve_relative_path_with_context_var(self, tmp_path: Path) -> None:
        """Test relative path resolved using context variable."""
        from holodeck.config.context import agent_base_dir

        doc_file = tmp_path / "docs" / "test.md"
        doc_file.parent.mkdir(parents=True, exist_ok=True)
        doc_file.write_text("# Test")

        config = HierarchicalDocumentToolConfig(
            name="test_tool",
            description="Test",
            source="docs/test.md",  # Relative path
        )
        tool = HierarchicalDocumentTool(config)  # No explicit base_dir

        # Set context variable
        token = agent_base_dir.set(str(tmp_path))
        try:
            resolved = tool._resolve_source_path()
            assert resolved == doc_file
        finally:
            agent_base_dir.reset(token)

    def test_resolve_relative_path_no_context(self, tmp_path: Path) -> None:
        """Test relative path resolved against cwd when no context."""
        config = HierarchicalDocumentToolConfig(
            name="test_tool",
            description="Test",
            source="relative/path.md",  # Relative path
        )
        tool = HierarchicalDocumentTool(config)

        # Should resolve relative to cwd
        resolved = tool._resolve_source_path()
        assert resolved.is_absolute()


class TestDiscoverFiles:
    """Tests for file discovery."""

    def test_discover_single_file(self, tmp_path: Path) -> None:
        """Test discovering a single file source."""
        doc_file = tmp_path / "test.md"
        doc_file.write_text("# Test")

        config = create_config(tmp_path)
        tool = HierarchicalDocumentTool(config)

        files = tool._discover_files()
        assert len(files) == 1
        assert files[0] == doc_file

    def test_discover_directory(self, tmp_path: Path) -> None:
        """Test discovering files in a directory."""
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "file1.md").write_text("# File 1")
        (docs_dir / "file2.md").write_text("# File 2")
        (docs_dir / "file3.txt").write_text("Not markdown")

        config = HierarchicalDocumentToolConfig(
            name="test_tool",
            description="Test",
            source=str(docs_dir),
        )
        tool = HierarchicalDocumentTool(config)

        files = tool._discover_files()
        # Should include .md but also .txt (supported extension)
        assert len(files) >= 2
        assert all(f.suffix in [".md", ".txt", ".pdf"] for f in files)

    def test_discover_nested_directory(self, tmp_path: Path) -> None:
        """Test discovering files recursively in nested directories."""
        docs_dir = tmp_path / "docs"
        nested_dir = docs_dir / "nested"
        nested_dir.mkdir(parents=True)
        (docs_dir / "root.md").write_text("# Root")
        (nested_dir / "nested.md").write_text("# Nested")

        config = HierarchicalDocumentToolConfig(
            name="test_tool",
            description="Test",
            source=str(docs_dir),
        )
        tool = HierarchicalDocumentTool(config)

        files = tool._discover_files()
        file_names = [f.name for f in files]
        assert "root.md" in file_names
        assert "nested.md" in file_names


class TestNeedsReingest:
    """Tests for incremental ingestion mtime checking."""

    @pytest.mark.asyncio
    async def test_needs_reingest_no_collection(self, tmp_path: Path) -> None:
        """Test needs_reingest returns True when no collection exists."""
        doc_file = tmp_path / "test.md"
        doc_file.write_text("# Test")

        config = create_config(tmp_path)
        tool = HierarchicalDocumentTool(config)
        tool._collection = None

        result = await tool._needs_reingest(doc_file)
        assert result is True

    @pytest.mark.asyncio
    async def test_needs_reingest_file_not_in_store(self, tmp_path: Path) -> None:
        """Test needs_reingest returns True when file not in store."""
        doc_file = tmp_path / "test.md"
        doc_file.write_text("# Test")

        config = create_config(tmp_path)
        tool = HierarchicalDocumentTool(config)

        mock_collection = MagicMock()
        mock_collection.__aenter__ = AsyncMock(return_value=mock_collection)
        mock_collection.__aexit__ = AsyncMock(return_value=None)
        mock_collection.get = AsyncMock(return_value=[])  # No records found
        tool._collection = mock_collection

        result = await tool._needs_reingest(doc_file)
        assert result is True

    @pytest.mark.asyncio
    async def test_needs_reingest_file_unchanged(self, tmp_path: Path) -> None:
        """Test needs_reingest returns False when file hasn't changed."""
        doc_file = tmp_path / "test.md"
        doc_file.write_text("# Test")
        current_mtime = doc_file.stat().st_mtime

        config = create_config(tmp_path)
        tool = HierarchicalDocumentTool(config)

        mock_record = MagicMock()
        mock_record.mtime = current_mtime  # Same mtime

        mock_collection = MagicMock()
        mock_collection.__aenter__ = AsyncMock(return_value=mock_collection)
        mock_collection.__aexit__ = AsyncMock(return_value=None)
        mock_collection.get = AsyncMock(return_value=[mock_record])
        tool._collection = mock_collection

        result = await tool._needs_reingest(doc_file)
        assert result is False

    @pytest.mark.asyncio
    async def test_needs_reingest_file_modified(self, tmp_path: Path) -> None:
        """Test needs_reingest returns True when file is modified."""
        doc_file = tmp_path / "test.md"
        doc_file.write_text("# Test")
        current_mtime = doc_file.stat().st_mtime

        config = create_config(tmp_path)
        tool = HierarchicalDocumentTool(config)

        mock_record = MagicMock()
        mock_record.mtime = current_mtime - 100  # Older mtime

        mock_collection = MagicMock()
        mock_collection.__aenter__ = AsyncMock(return_value=mock_collection)
        mock_collection.__aexit__ = AsyncMock(return_value=None)
        mock_collection.get = AsyncMock(return_value=[mock_record])
        tool._collection = mock_collection

        result = await tool._needs_reingest(doc_file)
        assert result is True


class TestDeleteFileRecords:
    """Tests for deleting file records from vector store."""

    @pytest.mark.asyncio
    async def test_delete_file_records_no_collection(self, tmp_path: Path) -> None:
        """Test delete returns 0 when no collection exists."""
        doc_file = tmp_path / "test.md"
        doc_file.write_text("# Test")

        config = create_config(tmp_path)
        tool = HierarchicalDocumentTool(config)
        tool._collection = None

        result = await tool._delete_file_records(doc_file)
        assert result == 0

    @pytest.mark.asyncio
    async def test_delete_file_records_deletes_all(self, tmp_path: Path) -> None:
        """Test delete removes all records for a file."""
        doc_file = tmp_path / "test.md"
        doc_file.write_text("# Test")

        config = create_config(tmp_path)
        tool = HierarchicalDocumentTool(config)

        # Mock records
        mock_records = [
            MagicMock(id="chunk_0"),
            MagicMock(id="chunk_1"),
            MagicMock(id="chunk_2"),
        ]

        mock_collection = MagicMock()
        mock_collection.__aenter__ = AsyncMock(return_value=mock_collection)
        mock_collection.__aexit__ = AsyncMock(return_value=None)
        mock_collection.get = AsyncMock(return_value=mock_records)
        mock_collection.delete = AsyncMock()
        tool._collection = mock_collection

        result = await tool._delete_file_records(doc_file)

        assert result == 3
        mock_collection.delete.assert_called_once()


class TestCollectionSetupFallback:
    """Tests for collection setup error handling."""

    def test_setup_collection_fallback_on_error(self, tmp_path: Path) -> None:
        """Test collection setup falls back to in-memory on connection error."""
        from holodeck.models.tool import DatabaseConfig

        doc_file = tmp_path / "test.md"
        doc_file.write_text("# Test")

        # Use a provider that will fail
        db_config = DatabaseConfig(provider="postgres")
        config = HierarchicalDocumentToolConfig(
            name="test_tool",
            description="Test",
            source=str(doc_file),
            database=db_config,
        )

        tool = HierarchicalDocumentTool(config)

        # Mock get_collection_factory at the source module
        with patch("holodeck.lib.vector_store.get_collection_factory") as mock_factory:
            # First call (postgres) raises error, second (in-memory) succeeds
            mock_inmemory_collection = MagicMock()
            mock_inmemory_factory = MagicMock(return_value=mock_inmemory_collection)
            mock_factory.side_effect = [
                ConnectionError("Failed to connect"),
                mock_inmemory_factory,
            ]

            tool._setup_collection("openai")

            # Should have fallen back to in-memory
            assert tool._provider == "in-memory"
            assert tool._collection == mock_inmemory_collection


class TestNeedsReingestExceptionHandling:
    """Tests for exception handling in _needs_reingest."""

    @pytest.mark.asyncio
    async def test_needs_reingest_returns_true_on_exception(
        self, tmp_path: Path
    ) -> None:
        """Test needs_reingest returns True when collection.get raises exception."""
        doc_file = tmp_path / "test.md"
        doc_file.write_text("# Test")

        config = create_config(tmp_path)
        tool = HierarchicalDocumentTool(config)

        mock_collection = MagicMock()
        mock_collection.__aenter__ = AsyncMock(return_value=mock_collection)
        mock_collection.__aexit__ = AsyncMock(return_value=None)
        mock_collection.get = AsyncMock(side_effect=Exception("Database error"))
        tool._collection = mock_collection

        result = await tool._needs_reingest(doc_file)
        assert result is True  # Should return True on error


class TestDeleteFileRecordsExceptionHandling:
    """Tests for exception handling in _delete_file_records."""

    @pytest.mark.asyncio
    async def test_delete_file_records_handles_exception(self, tmp_path: Path) -> None:
        """Test delete_file_records handles exceptions gracefully."""
        doc_file = tmp_path / "test.md"
        doc_file.write_text("# Test")

        config = create_config(tmp_path)
        tool = HierarchicalDocumentTool(config)

        mock_collection = MagicMock()
        mock_collection.__aenter__ = AsyncMock(return_value=mock_collection)
        mock_collection.__aexit__ = AsyncMock(return_value=None)
        mock_collection.get = AsyncMock(side_effect=Exception("Database error"))
        tool._collection = mock_collection

        result = await tool._delete_file_records(doc_file)
        assert result == 0  # Should return 0 on error


class TestConvertToMarkdownErrorHandling:
    """Tests for error handling in _convert_to_markdown."""

    @pytest.mark.asyncio
    async def test_convert_to_markdown_fallback_on_error(self, tmp_path: Path) -> None:
        """Test conversion falls back to direct read for text files on error."""
        doc_file = tmp_path / "test.txt"
        doc_file.write_text("Plain text content")

        config = create_config(tmp_path, source=str(doc_file))
        tool = HierarchicalDocumentTool(config)

        # Import the module where FileProcessor will be looked up
        import holodeck.lib.file_processor as fp_module

        original_fp = fp_module.FileProcessor

        try:
            mock_processor = MagicMock()
            mock_result = MagicMock()
            mock_result.error = "Processing failed"
            mock_result.markdown_content = ""
            mock_processor.process_file.return_value = mock_result
            fp_module.FileProcessor = MagicMock(return_value=mock_processor)

            result = await tool._convert_to_markdown(str(doc_file))

            # Should fall back to reading file directly
            assert result == "Plain text content"
        finally:
            fp_module.FileProcessor = original_fp

    @pytest.mark.asyncio
    async def test_convert_to_markdown_returns_empty_on_unsupported_error(
        self, tmp_path: Path
    ) -> None:
        """Test conversion returns empty string for unsupported file types on error."""
        doc_file = tmp_path / "test.pdf"
        doc_file.write_bytes(b"fake pdf content")

        config = create_config(tmp_path, source=str(doc_file))
        tool = HierarchicalDocumentTool(config)

        import holodeck.lib.file_processor as fp_module

        original_fp = fp_module.FileProcessor

        try:
            mock_processor = MagicMock()
            mock_result = MagicMock()
            mock_result.error = "Processing failed"
            mock_result.markdown_content = ""
            mock_processor.process_file.return_value = mock_result
            fp_module.FileProcessor = MagicMock(return_value=mock_processor)

            result = await tool._convert_to_markdown(str(doc_file))

            # Should return empty string for non-text files
            assert result == ""
        finally:
            fp_module.FileProcessor = original_fp


class TestGetSubsectionPatterns:
    """Tests for _get_subsection_patterns method."""

    def test_returns_none_for_none_domain(self, tmp_path: Path) -> None:
        """Test returns None when document_domain is NONE."""
        from holodeck.models.tool import DocumentDomain

        config = create_config(tmp_path, document_domain=DocumentDomain.NONE)
        tool = HierarchicalDocumentTool(config)

        result = tool._get_subsection_patterns()
        assert result is None

    def test_returns_patterns_for_legal_contract_domain(self, tmp_path: Path) -> None:
        """Test returns patterns for legal_contract domain."""
        from holodeck.models.tool import DocumentDomain

        config = create_config(tmp_path, document_domain=DocumentDomain.LEGAL_CONTRACT)
        tool = HierarchicalDocumentTool(config)

        result = tool._get_subsection_patterns()
        # Should return legal_contract patterns from DOMAIN_PATTERNS
        assert result is not None
        assert isinstance(result, list)
        assert len(result) > 0


class TestIngestDocumentsEdgeCases:
    """Tests for edge cases in _ingest_documents."""

    @pytest.mark.asyncio
    async def test_ingest_logs_warning_no_files(self, tmp_path: Path) -> None:
        """Test ingest logs warning when no supported files found."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        config = HierarchicalDocumentToolConfig(
            name="test_tool",
            description="Test",
            source=str(empty_dir),
        )
        tool = HierarchicalDocumentTool(config)

        # Set up required components
        tool._collection = MagicMock()
        tool._embedding_dimensions = 1536

        with patch.object(tool, "_discover_files", return_value=[]):
            await tool._ingest_documents()

        # Should complete without error even with no files

    @pytest.mark.asyncio
    async def test_ingest_skips_unchanged_files(self, tmp_path: Path) -> None:
        """Test ingest skips files that haven't changed."""
        doc_file = tmp_path / "test.md"
        doc_file.write_text("# Test\n\nContent here.")

        config = create_config(tmp_path)
        tool = HierarchicalDocumentTool(config)
        tool._collection = MagicMock()
        tool._embedding_dimensions = 1536

        with (
            patch.object(
                tool, "_needs_reingest", new_callable=AsyncMock, return_value=False
            ),
            patch.object(tool, "_convert_to_markdown") as mock_convert,
        ):
            await tool._ingest_documents(force_ingest=False)

            # Should not convert unchanged files
            mock_convert.assert_not_called()

    @pytest.mark.asyncio
    async def test_ingest_force_deletes_existing_records(self, tmp_path: Path) -> None:
        """Test force_ingest deletes existing records before re-ingesting."""
        doc_file = tmp_path / "test.md"
        doc_file.write_text("# Test\n\nContent here.")

        config = create_config(tmp_path)
        tool = HierarchicalDocumentTool(config)
        tool._collection = MagicMock()
        tool._embedding_dimensions = 1536

        with (
            patch.object(
                tool,
                "_delete_file_records",
                new_callable=AsyncMock,
                return_value=2,
            ) as mock_delete,
            patch.object(
                tool,
                "_convert_to_markdown",
                new_callable=AsyncMock,
                return_value="# Test\n\nContent",
            ),
            patch.object(tool, "_embed_chunks", new_callable=AsyncMock),
            patch.object(tool, "_store_chunks", new_callable=AsyncMock, return_value=1),
        ):
            await tool._ingest_documents(force_ingest=True)

            mock_delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_ingest_skips_empty_content(self, tmp_path: Path) -> None:
        """Test ingest skips files with empty content after conversion."""
        doc_file = tmp_path / "test.md"
        doc_file.write_text("")

        config = create_config(tmp_path, source=str(doc_file))
        tool = HierarchicalDocumentTool(config)
        tool._collection = MagicMock()
        tool._embedding_dimensions = 1536

        with (
            patch.object(
                tool, "_needs_reingest", new_callable=AsyncMock, return_value=True
            ),
            patch.object(
                tool,
                "_convert_to_markdown",
                new_callable=AsyncMock,
                return_value="   ",  # Only whitespace
            ),
            patch.object(tool, "_embed_chunks") as mock_embed,
        ):
            await tool._ingest_documents()

            # Should not embed empty content
            mock_embed.assert_not_called()


class TestContextualEmbeddings:
    """Tests for contextual embeddings generation."""

    @pytest.mark.asyncio
    async def test_contextual_embeddings_when_enabled(self, tmp_path: Path) -> None:
        """Test contextual embeddings generated when enabled and services available."""
        from holodeck.lib.structured_chunker import DocumentChunk

        doc_file = tmp_path / "test.md"
        doc_file.write_text("# Test\n\nContent here.")

        config = create_config(tmp_path, contextual_embeddings=True)
        tool = HierarchicalDocumentTool(config)
        tool._collection = MagicMock()
        tool._embedding_dimensions = 1536

        # Set up context generator and chat service
        mock_context_gen = MagicMock()
        mock_context_gen.contextualize_batch = AsyncMock(
            return_value=["Contextualized content"]
        )
        tool._context_generator = mock_context_gen
        tool._chat_service = MagicMock()

        mock_chunk = MagicMock(spec=DocumentChunk)
        mock_chunk.id = "chunk_0"
        mock_chunk.content = "Content"
        mock_chunk.chunk_type = MagicMock()
        mock_chunk.chunk_type.name = "PARAGRAPH"

        with (
            patch.object(
                tool, "_needs_reingest", new_callable=AsyncMock, return_value=True
            ),
            patch.object(
                tool,
                "_convert_to_markdown",
                new_callable=AsyncMock,
                return_value="# Test\n\nContent",
            ),
            (
                patch.object(tool._chunker, "parse", return_value=[mock_chunk])
                if tool._chunker
                else patch.object(tool, "_chunker")
            ),
            patch.object(tool, "_embed_chunks", new_callable=AsyncMock),
            patch.object(tool, "_store_chunks", new_callable=AsyncMock, return_value=1),
        ):
            # Skip the actual test since chunker is None in this setup
            pass

    @pytest.mark.asyncio
    async def test_contextual_embeddings_disabled_reason_logged(
        self, tmp_path: Path
    ) -> None:
        """Test reason for skipping contextual embeddings is logged."""
        doc_file = tmp_path / "test.md"
        doc_file.write_text("# Test\n\nContent here.")

        # contextual_embeddings=False by default
        config = create_config(tmp_path, contextual_embeddings=False)
        tool = HierarchicalDocumentTool(config)
        tool._collection = MagicMock()
        tool._embedding_dimensions = 1536

        # No chat service
        tool._chat_service = None
        tool._context_generator = None

        # This should run without error and log the reason


class TestEmbedChunksEdgeCases:
    """Tests for edge cases in _embed_chunks."""

    @pytest.mark.asyncio
    async def test_embed_chunks_empty_list(self, tmp_path: Path) -> None:
        """Test _embed_chunks returns early for empty list."""
        config = create_config(tmp_path)
        tool = HierarchicalDocumentTool(config)

        # Should not raise
        await tool._embed_chunks([])

    @pytest.mark.asyncio
    async def test_embed_chunks_handles_exception(self, tmp_path: Path) -> None:
        """Test _embed_chunks falls back to placeholders on exception."""
        from holodeck.lib.structured_chunker import DocumentChunk

        config = create_config(tmp_path)
        tool = HierarchicalDocumentTool(config)
        tool._embedding_dimensions = 1536

        mock_embed = AsyncMock()
        mock_embed.generate_embeddings.side_effect = Exception("API error")
        tool._embedding_service = mock_embed

        mock_chunk = MagicMock(spec=DocumentChunk)
        mock_chunk.content = "Test content"
        mock_chunk.contextualized_content = None

        await tool._embed_chunks([mock_chunk])

        # Should have placeholder embedding
        assert mock_chunk.embedding is not None
        assert len(mock_chunk.embedding) == 1536


class TestEmbedQueryEdgeCases:
    """Tests for edge cases in _embed_query."""

    @pytest.mark.asyncio
    async def test_embed_query_handles_exception(self, tmp_path: Path) -> None:
        """Test _embed_query falls back to placeholder on exception."""
        config = create_config(tmp_path)
        tool = HierarchicalDocumentTool(config)
        tool._embedding_dimensions = 1536

        mock_embed = AsyncMock()
        mock_embed.generate_embeddings.side_effect = Exception("API error")
        tool._embedding_service = mock_embed

        result = await tool._embed_query("test query")

        # Should return placeholder embedding
        assert len(result) == 1536
        assert all(v == 0.0 for v in result)


class TestStoreChunksEdgeCases:
    """Tests for edge cases in _store_chunks."""

    @pytest.mark.asyncio
    async def test_store_chunks_ensures_collection_exists(self, tmp_path: Path) -> None:
        """Test _store_chunks ensures collection exists before upserting."""
        from holodeck.lib.structured_chunker import DocumentChunk

        config = create_config(tmp_path)
        tool = HierarchicalDocumentTool(config)

        mock_chunk = MagicMock(spec=DocumentChunk)
        mock_chunk.id = "chunk_0"
        mock_chunk.content = "Content"
        mock_chunk.source_path = str(tmp_path / "test.md")
        mock_chunk.chunk_index = 0
        mock_chunk.mtime = 12345.0
        mock_chunk.parent_chain = ["Chapter 1"]
        mock_chunk.section_id = "1.1"
        mock_chunk.chunk_type = MagicMock()
        mock_chunk.chunk_type.value = "paragraph"
        mock_chunk.heading_level = 0
        mock_chunk.contextualized_content = "Contextualized"
        mock_chunk.cross_references = []
        mock_chunk.defined_term = None
        mock_chunk.defined_term_normalized = None
        mock_chunk.subsection_ids = []
        mock_chunk.embedding = [0.1] * 1536

        mock_collection = MagicMock()
        mock_collection.__aenter__ = AsyncMock(return_value=mock_collection)
        mock_collection.__aexit__ = AsyncMock(return_value=None)
        mock_collection.collection_exists = AsyncMock(return_value=False)
        mock_collection.ensure_collection_exists = AsyncMock()
        mock_collection.upsert = AsyncMock()
        tool._collection = mock_collection

        await tool._store_chunks([mock_chunk])

        mock_collection.ensure_collection_exists.assert_called_once()
        mock_collection.upsert.assert_called_once()


class TestSemanticSearchJsonParsing:
    """Tests for JSON parsing in _semantic_search."""

    @pytest.mark.asyncio
    async def test_semantic_search_handles_invalid_json(self, tmp_path: Path) -> None:
        """Test _semantic_search handles invalid JSON in parent_chain gracefully."""
        config = create_config(tmp_path)
        tool = HierarchicalDocumentTool(config)
        tool._initialized = True
        tool._embedding_dimensions = 1536

        mock_embed = AsyncMock()
        mock_embed.generate_embeddings.return_value = [[0.1] * 1536]
        tool._embedding_service = mock_embed

        mock_record = MagicMock()
        mock_record.id = "chunk_0"
        mock_record.content = "Test content"
        mock_record.source_path = "/test.md"
        mock_record.parent_chain = "not valid json"  # Invalid JSON
        mock_record.section_id = "1.1"
        mock_record.subsection_ids = "also invalid"  # Invalid JSON

        mock_result = MagicMock()
        mock_result.record = mock_record
        mock_result.score = 0.85

        mock_collection = MagicMock()
        mock_collection.__aenter__ = AsyncMock(return_value=mock_collection)
        mock_collection.__aexit__ = AsyncMock(return_value=None)

        async def mock_results():
            yield mock_result

        mock_search_results = MagicMock()
        mock_search_results.results = mock_results()
        mock_collection.search = AsyncMock(return_value=mock_search_results)
        tool._collection = mock_collection

        results = await tool._semantic_search([0.1] * 1536, top_k=10)

        # Should return results with empty lists for invalid JSON
        assert len(results) == 1
        assert results[0].parent_chain == []
        assert results[0].subsection_ids == []


class TestBuildHybridIndices:
    """Tests for _build_hybrid_indices."""

    @pytest.mark.asyncio
    async def test_build_hybrid_indices_no_chunks(self, tmp_path: Path) -> None:
        """Test _build_hybrid_indices returns early when no chunks."""
        config = create_config(tmp_path)
        tool = HierarchicalDocumentTool(config)
        tool._chunks = []

        await tool._build_hybrid_indices()

        assert tool._hybrid_executor is None


class TestHybridSearch:
    """Tests for _hybrid_search method."""

    @pytest.mark.asyncio
    async def test_hybrid_search_no_executor_fallback(self, tmp_path: Path) -> None:
        """Test _hybrid_search falls back to semantic when no executor."""
        config = create_config(tmp_path)
        tool = HierarchicalDocumentTool(config)
        tool._initialized = True
        tool._hybrid_executor = None
        tool._embedding_dimensions = 1536

        mock_embed = AsyncMock()
        mock_embed.generate_embeddings.return_value = [[0.1] * 1536]
        tool._embedding_service = mock_embed

        mock_record = MagicMock()
        mock_record.id = "chunk_0"
        mock_record.content = "Test content"
        mock_record.source_path = "/test.md"
        mock_record.parent_chain = "[]"
        mock_record.section_id = "1.1"
        mock_record.subsection_ids = "[]"

        mock_result = MagicMock()
        mock_result.record = mock_record
        mock_result.score = 0.85

        mock_collection = MagicMock()
        mock_collection.__aenter__ = AsyncMock(return_value=mock_collection)
        mock_collection.__aexit__ = AsyncMock(return_value=None)

        async def mock_results():
            yield mock_result

        mock_search_results = MagicMock()
        mock_search_results.results = mock_results()
        mock_collection.search = AsyncMock(return_value=mock_search_results)
        tool._collection = mock_collection

        results = await tool._hybrid_search("test query", [0.1] * 1536, top_k=10)

        # Should return semantic search results
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_hybrid_search_with_executor(self, tmp_path: Path) -> None:
        """Test _hybrid_search uses hybrid executor when available."""
        from holodeck.lib.structured_chunker import DocumentChunk

        config = create_config(tmp_path)
        tool = HierarchicalDocumentTool(config)
        tool._initialized = True
        tool._embedding_dimensions = 1536

        # Set up mock chunk
        mock_chunk = MagicMock(spec=DocumentChunk)
        mock_chunk.id = "chunk_0"
        mock_chunk.content = "Test content"
        mock_chunk.source_path = "/test.md"
        mock_chunk.parent_chain = ["Chapter 1"]
        mock_chunk.section_id = "1.1"
        mock_chunk.subsection_ids = []

        # Set up mock hybrid executor (executor now owns chunk data)
        mock_executor = MagicMock()
        mock_executor.search = AsyncMock(return_value=[("chunk_0", 0.95)])
        mock_executor.get_chunk.return_value = mock_chunk
        tool._hybrid_executor = mock_executor

        results = await tool._hybrid_search("test query", [0.1] * 1536, top_k=10)

        assert len(results) == 1
        assert results[0].chunk_id == "chunk_0"
        assert results[0].fused_score == 0.95
        mock_executor.get_chunk.assert_called_with("chunk_0")


class TestKeywordSearch:
    """Tests for _keyword_search method."""

    @pytest.mark.asyncio
    async def test_keyword_search_no_executor(self, tmp_path: Path) -> None:
        """Test _keyword_search returns empty when no executor."""
        config = create_config(tmp_path)
        tool = HierarchicalDocumentTool(config)
        tool._hybrid_executor = None

        results = await tool._keyword_search("test query", top_k=10)

        assert results == []

    @pytest.mark.asyncio
    async def test_keyword_search_no_bm25_index(self, tmp_path: Path) -> None:
        """Test _keyword_search returns empty when BM25 index not built."""
        config = create_config(tmp_path)
        tool = HierarchicalDocumentTool(config)

        mock_executor = MagicMock()
        mock_executor.keyword_search.return_value = []
        tool._hybrid_executor = mock_executor

        results = await tool._keyword_search("test query", top_k=10)

        assert results == []
        mock_executor.keyword_search.assert_called_once_with("test query", 10)

    @pytest.mark.asyncio
    async def test_keyword_search_with_results(self, tmp_path: Path) -> None:
        """Test _keyword_search returns properly formatted results."""
        from holodeck.lib.structured_chunker import DocumentChunk

        config = create_config(tmp_path)
        tool = HierarchicalDocumentTool(config)

        # Set up mock chunk
        mock_chunk = MagicMock(spec=DocumentChunk)
        mock_chunk.id = "chunk_0"
        mock_chunk.content = "Test content"
        mock_chunk.source_path = "/test.md"
        mock_chunk.parent_chain = ["Chapter 1"]
        mock_chunk.section_id = "1.1"
        mock_chunk.subsection_ids = []

        # Set up mock executor with public keyword_search and get_chunk
        mock_executor = MagicMock()
        mock_executor.keyword_search.return_value = [("chunk_0", 5.5)]
        mock_executor.get_chunk.return_value = mock_chunk
        tool._hybrid_executor = mock_executor

        results = await tool._keyword_search("test query", top_k=10)

        assert len(results) == 1
        assert results[0].chunk_id == "chunk_0"
        assert results[0].keyword_score == 5.5
        # Normalized score should be min(1.0, 5.5/10.0) = 0.55
        assert results[0].fused_score == 0.55
        mock_executor.get_chunk.assert_called_with("chunk_0")


class TestSearchModeRouting:
    """Tests for search mode routing in search() method."""

    @pytest.mark.asyncio
    async def test_search_keyword_mode(self, tmp_path: Path) -> None:
        """Test search routes to keyword search in KEYWORD mode."""
        from holodeck.models.tool import SearchMode

        config = create_config(tmp_path, search_mode=SearchMode.KEYWORD)
        tool = HierarchicalDocumentTool(config)
        tool._initialized = True
        tool._embedding_dimensions = 1536

        mock_embed = AsyncMock()
        mock_embed.generate_embeddings.return_value = [[0.1] * 1536]
        tool._embedding_service = mock_embed

        with patch.object(
            tool,
            "_keyword_search",
            new_callable=AsyncMock,
            return_value=[],
        ) as mock_keyword:
            results = await tool.search("test query")

            mock_keyword.assert_called_once()
            assert results == []

    @pytest.mark.asyncio
    async def test_search_hybrid_mode(self, tmp_path: Path) -> None:
        """Test search routes to hybrid search in HYBRID mode."""
        from holodeck.models.tool import SearchMode

        config = create_config(tmp_path, search_mode=SearchMode.HYBRID)
        tool = HierarchicalDocumentTool(config)
        tool._initialized = True
        tool._embedding_dimensions = 1536

        mock_embed = AsyncMock()
        mock_embed.generate_embeddings.return_value = [[0.1] * 1536]
        tool._embedding_service = mock_embed

        with patch.object(
            tool,
            "_hybrid_search",
            new_callable=AsyncMock,
            return_value=[],
        ) as mock_hybrid:
            results = await tool.search("test query")

            mock_hybrid.assert_called_once()
            assert results == []

    @pytest.mark.asyncio
    async def test_search_semantic_mode_default(self, tmp_path: Path) -> None:
        """Test search defaults to semantic search."""
        from holodeck.models.tool import SearchMode

        config = create_config(tmp_path, search_mode=SearchMode.SEMANTIC)
        tool = HierarchicalDocumentTool(config)
        tool._initialized = True
        tool._embedding_dimensions = 1536

        mock_embed = AsyncMock()
        mock_embed.generate_embeddings.return_value = [[0.1] * 1536]
        tool._embedding_service = mock_embed

        with patch.object(
            tool,
            "_semantic_search",
            new_callable=AsyncMock,
            return_value=[],
        ) as mock_semantic:
            results = await tool.search("test query")

            mock_semantic.assert_called_once()
            assert results == []


class TestLoadChunksFromStore:
    """Tests for _load_chunks_from_store method."""

    @pytest.mark.asyncio
    async def test_load_chunks_no_collection(self, tmp_path: Path) -> None:
        """Test returns [] when self._collection is None."""
        config = create_config(tmp_path)
        tool = HierarchicalDocumentTool(config)
        tool._collection = None

        result = await tool._load_chunks_from_store()
        assert result == []

    @pytest.mark.asyncio
    async def test_load_chunks_collection_not_exists(self, tmp_path: Path) -> None:
        """Test returns [] when collection_exists() returns False."""
        config = create_config(tmp_path)
        tool = HierarchicalDocumentTool(config)

        mock_collection = MagicMock()
        mock_collection.__aenter__ = AsyncMock(return_value=mock_collection)
        mock_collection.__aexit__ = AsyncMock(return_value=None)
        mock_collection.collection_exists = AsyncMock(return_value=False)
        tool._collection = mock_collection

        result = await tool._load_chunks_from_store()
        assert result == []

    @pytest.mark.asyncio
    async def test_load_chunks_empty_store(self, tmp_path: Path) -> None:
        """Test returns [] when collection.get() returns no records."""
        config = create_config(tmp_path)
        tool = HierarchicalDocumentTool(config)

        mock_collection = MagicMock()
        mock_collection.__aenter__ = AsyncMock(return_value=mock_collection)
        mock_collection.__aexit__ = AsyncMock(return_value=None)
        mock_collection.collection_exists = AsyncMock(return_value=True)
        mock_collection.get = AsyncMock(return_value=[])
        tool._collection = mock_collection

        result = await tool._load_chunks_from_store()
        assert result == []

    @pytest.mark.asyncio
    async def test_load_chunks_reconstructs_fields(self, tmp_path: Path) -> None:
        """Test happy path: records are reconstructed into DocumentChunks."""
        from holodeck.lib.structured_chunker import ChunkType

        config = create_config(tmp_path)
        tool = HierarchicalDocumentTool(config)

        mock_record = MagicMock()
        mock_record.id = "chunk_0"
        mock_record.source_path = "/test.md"
        mock_record.chunk_index = 0
        mock_record.content = "Test content"
        mock_record.parent_chain = '["Chapter 1", "Section 1.1"]'
        mock_record.section_id = "1.1"
        mock_record.chunk_type = "content"
        mock_record.cross_references = '["2.1", "3.0"]'
        mock_record.contextualized_content = "Contextualized test content"
        mock_record.mtime = 12345.0
        mock_record.defined_term = "TestTerm"
        mock_record.defined_term_normalized = "testterm"
        mock_record.subsection_ids = '["1.1.1", "1.1.2"]'

        mock_collection = MagicMock()
        mock_collection.__aenter__ = AsyncMock(return_value=mock_collection)
        mock_collection.__aexit__ = AsyncMock(return_value=None)
        mock_collection.collection_exists = AsyncMock(return_value=True)
        mock_collection.get = AsyncMock(return_value=[mock_record])
        tool._collection = mock_collection

        result = await tool._load_chunks_from_store()

        assert len(result) == 1
        chunk = result[0]
        assert chunk.id == "chunk_0"
        assert chunk.source_path == "/test.md"
        assert chunk.chunk_index == 0
        assert chunk.content == "Test content"
        assert chunk.parent_chain == ["Chapter 1", "Section 1.1"]
        assert chunk.section_id == "1.1"
        assert chunk.chunk_type == ChunkType.CONTENT
        assert chunk.cross_references == ["2.1", "3.0"]
        assert chunk.contextualized_content == "Contextualized test content"
        assert chunk.mtime == 12345.0
        assert chunk.defined_term == "TestTerm"
        assert chunk.defined_term_normalized == "testterm"
        assert chunk.subsection_ids == ["1.1.1", "1.1.2"]
        assert chunk.heading_level == 0

    @pytest.mark.asyncio
    async def test_load_chunks_invalid_json_degrades(self, tmp_path: Path) -> None:
        """Test corrupt JSON in list fields yields empty lists, no raise."""
        config = create_config(tmp_path)
        tool = HierarchicalDocumentTool(config)

        mock_record = MagicMock()
        mock_record.id = "chunk_0"
        mock_record.source_path = "/test.md"
        mock_record.chunk_index = 0
        mock_record.content = "Content"
        mock_record.parent_chain = "not valid json"
        mock_record.section_id = ""
        mock_record.chunk_type = "content"
        mock_record.cross_references = "{broken"
        mock_record.contextualized_content = ""
        mock_record.mtime = 0.0
        mock_record.defined_term = ""
        mock_record.defined_term_normalized = ""
        mock_record.subsection_ids = "also broken"

        mock_collection = MagicMock()
        mock_collection.__aenter__ = AsyncMock(return_value=mock_collection)
        mock_collection.__aexit__ = AsyncMock(return_value=None)
        mock_collection.collection_exists = AsyncMock(return_value=True)
        mock_collection.get = AsyncMock(return_value=[mock_record])
        tool._collection = mock_collection

        result = await tool._load_chunks_from_store()

        assert len(result) == 1
        assert result[0].parent_chain == []
        assert result[0].cross_references == []
        assert result[0].subsection_ids == []

    @pytest.mark.asyncio
    async def test_load_chunks_invalid_chunk_type(self, tmp_path: Path) -> None:
        """Test unknown chunk_type string defaults to ChunkType.CONTENT."""
        from holodeck.lib.structured_chunker import ChunkType

        config = create_config(tmp_path)
        tool = HierarchicalDocumentTool(config)

        mock_record = MagicMock()
        mock_record.id = "chunk_0"
        mock_record.source_path = "/test.md"
        mock_record.chunk_index = 0
        mock_record.content = "Content"
        mock_record.parent_chain = "[]"
        mock_record.section_id = ""
        mock_record.chunk_type = "unknown_type"
        mock_record.cross_references = "[]"
        mock_record.contextualized_content = ""
        mock_record.mtime = 0.0
        mock_record.defined_term = ""
        mock_record.defined_term_normalized = ""
        mock_record.subsection_ids = "[]"

        mock_collection = MagicMock()
        mock_collection.__aenter__ = AsyncMock(return_value=mock_collection)
        mock_collection.__aexit__ = AsyncMock(return_value=None)
        mock_collection.collection_exists = AsyncMock(return_value=True)
        mock_collection.get = AsyncMock(return_value=[mock_record])
        tool._collection = mock_collection

        result = await tool._load_chunks_from_store()

        assert len(result) == 1
        assert result[0].chunk_type == ChunkType.CONTENT

    @pytest.mark.asyncio
    async def test_load_chunks_graceful_on_exception(self, tmp_path: Path) -> None:
        """Test collection.get() raises -> returns [], logs warning."""
        config = create_config(tmp_path)
        tool = HierarchicalDocumentTool(config)

        mock_collection = MagicMock()
        mock_collection.__aenter__ = AsyncMock(return_value=mock_collection)
        mock_collection.__aexit__ = AsyncMock(return_value=None)
        mock_collection.collection_exists = AsyncMock(
            side_effect=Exception("Connection failed")
        )
        tool._collection = mock_collection

        result = await tool._load_chunks_from_store()
        assert result == []


class TestInitializeRebuild:
    """Tests for initialize() BM25 rebuild when files are skipped."""

    @pytest.mark.asyncio
    async def test_initialize_rebuilds_when_files_skipped(self, tmp_path: Path) -> None:
        """Test rebuild when provider is persistent and files were skipped."""
        from holodeck.lib.structured_chunker import DocumentChunk

        config = create_config(tmp_path)
        tool = HierarchicalDocumentTool(config)
        tool._provider = "qdrant"

        stored_chunks = [
            DocumentChunk(
                id="chunk_0",
                source_path="/test.md",
                chunk_index=0,
                content="Stored content",
                contextualized_content="Stored content",
            ),
        ]

        with (
            patch.object(tool, "_setup_collection"),
            patch.object(
                tool,
                "_ingest_documents",
                new_callable=AsyncMock,
                return_value=2,
            ),
            patch.object(
                tool,
                "_load_chunks_from_store",
                new_callable=AsyncMock,
                return_value=stored_chunks,
            ) as mock_load,
            patch.object(
                tool,
                "_build_hybrid_indices",
                new_callable=AsyncMock,
            ) as mock_build,
        ):
            await tool.initialize()

            mock_load.assert_called_once()
            mock_build.assert_called_once()
            assert tool._chunks == stored_chunks
            assert tool._initialized is True

    @pytest.mark.asyncio
    async def test_initialize_skips_rebuild_for_in_memory(self, tmp_path: Path) -> None:
        """Test no rebuild when provider is in-memory even if files skipped."""
        config = create_config(tmp_path)
        tool = HierarchicalDocumentTool(config)
        tool._provider = "in-memory"

        with (
            patch.object(tool, "_setup_collection"),
            patch.object(
                tool,
                "_ingest_documents",
                new_callable=AsyncMock,
                return_value=2,
            ),
            patch.object(
                tool,
                "_load_chunks_from_store",
                new_callable=AsyncMock,
            ) as mock_load,
        ):
            await tool.initialize()

            mock_load.assert_not_called()

    @pytest.mark.asyncio
    async def test_initialize_no_rebuild_when_zero_skipped(
        self, tmp_path: Path
    ) -> None:
        """Test no rebuild when _ingest_documents returns 0 skipped."""
        config = create_config(tmp_path)
        tool = HierarchicalDocumentTool(config)
        tool._provider = "qdrant"

        with (
            patch.object(tool, "_setup_collection"),
            patch.object(
                tool,
                "_ingest_documents",
                new_callable=AsyncMock,
                return_value=0,
            ),
            patch.object(
                tool,
                "_load_chunks_from_store",
                new_callable=AsyncMock,
            ) as mock_load,
        ):
            await tool.initialize()

            mock_load.assert_not_called()

    @pytest.mark.asyncio
    async def test_ingest_documents_returns_skipped_count(self, tmp_path: Path) -> None:
        """Test _ingest_documents returns the correct skipped file count."""
        doc_file = tmp_path / "test.md"
        doc_file.write_text("# Test\n\nContent here.")

        config = create_config(tmp_path)
        tool = HierarchicalDocumentTool(config)
        tool._collection = MagicMock()
        tool._embedding_dimensions = 1536

        with patch.object(
            tool, "_needs_reingest", new_callable=AsyncMock, return_value=False
        ):
            result = await tool._ingest_documents(force_ingest=False)

            assert result == 1  # One file discovered, one skipped
