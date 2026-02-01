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
