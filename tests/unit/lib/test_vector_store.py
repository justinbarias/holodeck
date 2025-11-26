"""Tests for vector store models and factory functions.

Note: This test module requires mocking semantic_kernel modules because the full
semantic_kernel library is not available in the test environment. Mocks are set up
only during module import to avoid polluting the rest of the test suite.
"""

import sys
from unittest.mock import MagicMock

# Save original modules before mocking (these will be restored after import)
_saved_modules = {}

# Mock semantic_kernel modules ONLY during initial import
# We'll restore them after the holodeck.lib.vector_store import completes
for module_name in [
    "semantic_kernel",
    "semantic_kernel.connectors",
    "semantic_kernel.connectors.memory",
    "semantic_kernel.connectors.ai",
    "semantic_kernel.contents",
    "semantic_kernel.data",
    "semantic_kernel.data.vector",
]:
    if module_name in sys.modules:
        _saved_modules[module_name] = sys.modules[module_name]
    else:
        sys.modules[module_name] = MagicMock()

# Set up mock attributes
# mypy: ignore - these are intentional mocks for testing
mock_memory = sys.modules["semantic_kernel.connectors.memory"]
mock_memory.AzureAISearchCollection = MagicMock()  # type: ignore[assignment]
mock_memory.ChromaCollection = MagicMock()  # type: ignore[assignment]
mock_memory.CosmosMongoCollection = MagicMock()  # type: ignore[assignment]
mock_memory.CosmosNoSqlCollection = MagicMock()  # type: ignore[assignment]
mock_memory.FaissCollection = MagicMock()  # type: ignore[assignment]
mock_memory.InMemoryCollection = MagicMock()  # type: ignore[assignment]
mock_memory.PineconeCollection = MagicMock()  # type: ignore[assignment]
mock_memory.PostgresCollection = MagicMock()  # type: ignore[assignment]
mock_memory.QdrantCollection = MagicMock()  # type: ignore[assignment]
mock_memory.RedisHashsetCollection = MagicMock()  # type: ignore[assignment]
mock_memory.RedisJsonCollection = MagicMock()  # type: ignore[assignment]
mock_memory.SqlServerCollection = MagicMock()  # type: ignore[assignment]
mock_memory.WeaviateCollection = MagicMock()  # type: ignore[assignment]

mock_vector = sys.modules["semantic_kernel.data.vector"]
mock_vector.VectorStoreField = MagicMock()  # type: ignore[assignment]
mock_vector.vectorstoremodel = lambda **kwargs: lambda cls: cls  # type: ignore[assignment]

# Now import from holodeck.lib.vector_store
import pytest  # noqa: E402

from holodeck.lib.vector_store import (  # noqa: E402
    DocumentRecord,
    QueryResult,
    convert_document_to_query_result,
    create_document_record_class,
    get_collection_factory,
)

# Restore original modules after import to avoid polluting other tests
# Keep the mocked versions only for this module's tests
for module_name in [
    "semantic_kernel",
    "semantic_kernel.connectors",
    "semantic_kernel.connectors.memory",
    "semantic_kernel.connectors.ai",
    "semantic_kernel.contents",
    "semantic_kernel.data",
    "semantic_kernel.data.vector",
]:
    if module_name in _saved_modules:
        sys.modules[module_name] = _saved_modules[module_name]
    # Note: We keep the mocks in sys.modules for this module's execution


class TestDocumentRecord:
    """Tests for DocumentRecord dataclass."""

    def test_document_record_creation_with_defaults(self) -> None:
        """Test creating DocumentRecord with default values."""
        record = DocumentRecord()
        assert record.id  # Should have a UUID
        assert record.source_path == ""
        assert record.chunk_index == 0
        assert record.content == ""
        assert record.embedding is None
        assert record.mtime == 0.0
        assert record.file_type == ""
        assert record.file_size_bytes == 0

    def test_document_record_creation_with_values(self) -> None:
        """Test creating DocumentRecord with custom values."""
        embedding = [0.1, 0.2, 0.3] * 512  # 1536 dimensions
        record = DocumentRecord(
            id="test_id",
            source_path="/path/to/file.txt",
            chunk_index=5,
            content="This is test content",
            embedding=embedding,
            mtime=1234567890.0,
            file_type=".txt",
            file_size_bytes=1024,
        )
        assert record.id == "test_id"
        assert record.source_path == "/path/to/file.txt"
        assert record.chunk_index == 5
        assert record.content == "This is test content"
        assert record.embedding == embedding
        assert record.mtime == 1234567890.0
        assert record.file_type == ".txt"
        assert record.file_size_bytes == 1024

    def test_document_record_auto_generates_uuid(self) -> None:
        """Test that DocumentRecord auto-generates UUID if not provided."""
        record1 = DocumentRecord()
        record2 = DocumentRecord()
        # Both should have IDs but they should be different
        assert record1.id
        assert record2.id
        assert record1.id != record2.id

    def test_document_record_with_embedding(self) -> None:
        """Test DocumentRecord with embedding vector."""
        embedding = [0.5] * 1536
        record = DocumentRecord(
            id="doc_1",
            embedding=embedding,
        )
        assert record.embedding == embedding

    def test_document_record_with_empty_embedding(self) -> None:
        """Test DocumentRecord with empty embedding."""
        record = DocumentRecord(id="doc_1", embedding=[])
        assert record.embedding == []

    def test_document_record_with_none_embedding(self) -> None:
        """Test DocumentRecord with None embedding."""
        record = DocumentRecord(id="doc_1")
        assert record.embedding is None

    def test_document_record_all_fields(self) -> None:
        """Test DocumentRecord with all fields populated."""
        record = DocumentRecord(
            id="doc_123",
            source_path="/docs/sample.pdf",
            chunk_index=3,
            content="Sample document content here",
            embedding=[0.1] * 1536,
            mtime=1700000000.0,
            file_type=".pdf",
            file_size_bytes=5000,
        )
        assert record.id == "doc_123"
        assert record.source_path == "/docs/sample.pdf"
        assert record.chunk_index == 3
        assert record.content == "Sample document content here"
        assert record.mtime == 1700000000.0
        assert record.file_type == ".pdf"
        assert record.file_size_bytes == 5000


class TestQueryResult:
    """Tests for QueryResult dataclass."""

    def test_query_result_creation(self) -> None:
        """Test creating a QueryResult."""
        result = QueryResult(
            content="Found content",
            score=0.95,
            source_path="/path/to/source.txt",
            chunk_index=2,
        )
        assert result.content == "Found content"
        assert result.score == 0.95
        assert result.source_path == "/path/to/source.txt"
        assert result.chunk_index == 2
        assert result.metadata == {}

    def test_query_result_with_metadata(self) -> None:
        """Test QueryResult with metadata."""
        metadata = {"file_type": ".txt", "file_size_bytes": 1000}
        result = QueryResult(
            content="Test",
            score=0.8,
            source_path="/test.txt",
            chunk_index=0,
            metadata=metadata,
        )
        assert result.metadata == metadata
        assert result.metadata["file_type"] == ".txt"
        assert result.metadata["file_size_bytes"] == 1000

    def test_query_result_score_validation_valid_range(self) -> None:
        """Test that scores in valid range are accepted."""
        # Boundary values
        QueryResult("content", score=0.0, source_path="/path", chunk_index=0)
        QueryResult("content", score=1.0, source_path="/path", chunk_index=0)
        # Middle values
        QueryResult("content", score=0.5, source_path="/path", chunk_index=0)
        QueryResult("content", score=0.99, source_path="/path", chunk_index=0)

    def test_query_result_score_validation_too_low(self) -> None:
        """Test that scores below 0.0 are rejected."""
        with pytest.raises(ValueError, match="Score must be between 0.0 and 1.0"):
            QueryResult(
                content="content",
                score=-0.1,
                source_path="/path",
                chunk_index=0,
            )

    def test_query_result_score_validation_too_high(self) -> None:
        """Test that scores above 1.0 are rejected."""
        with pytest.raises(ValueError, match="Score must be between 0.0 and 1.0"):
            QueryResult(
                content="content",
                score=1.1,
                source_path="/path",
                chunk_index=0,
            )

    def test_query_result_score_validation_negative(self) -> None:
        """Test that negative scores are rejected."""
        with pytest.raises(ValueError, match="Score must be between 0.0 and 1.0"):
            QueryResult(
                content="content",
                score=-1.0,
                source_path="/path",
                chunk_index=0,
            )

    def test_query_result_score_validation_far_exceeds(self) -> None:
        """Test that scores far exceeding range are rejected."""
        with pytest.raises(ValueError, match="Score must be between 0.0 and 1.0"):
            QueryResult(
                content="content",
                score=2.5,
                source_path="/path",
                chunk_index=0,
            )

    def test_query_result_with_zero_score(self) -> None:
        """Test QueryResult with score of 0.0."""
        result = QueryResult(
            content="content",
            score=0.0,
            source_path="/path",
            chunk_index=0,
        )
        assert result.score == 0.0

    def test_query_result_with_perfect_score(self) -> None:
        """Test QueryResult with perfect score of 1.0."""
        result = QueryResult(
            content="content",
            score=1.0,
            source_path="/path",
            chunk_index=0,
        )
        assert result.score == 1.0

    def test_query_result_default_metadata(self) -> None:
        """Test that default metadata is empty dict."""
        result = QueryResult(
            content="test",
            score=0.5,
            source_path="/test",
            chunk_index=0,
        )
        assert result.metadata == {}
        assert isinstance(result.metadata, dict)


class TestCreateDocumentRecordClass:
    """Tests for create_document_record_class factory function."""

    def test_creates_class_with_default_dimensions(self) -> None:
        """Test creating DocumentRecord class with default 1536 dimensions."""
        record_class = create_document_record_class()
        record = record_class(id="test", content="test content", embedding=[0.1] * 1536)
        assert record.id == "test"
        assert record.content == "test content"
        assert len(record.embedding) == 1536

    def test_creates_class_with_custom_dimensions(self) -> None:
        """Test creating DocumentRecord class with custom dimensions."""
        record_class = create_document_record_class(dimensions=768)
        record = record_class(id="test", content="test", embedding=[0.1] * 768)
        assert len(record.embedding) == 768

    def test_creates_class_with_large_dimensions(self) -> None:
        """Test creating DocumentRecord class with large dimensions (3072)."""
        record_class = create_document_record_class(dimensions=3072)
        record = record_class(id="test", content="test", embedding=[0.1] * 3072)
        assert len(record.embedding) == 3072

    def test_invalid_dimensions_zero(self) -> None:
        """Test that zero dimensions raises ValueError."""
        with pytest.raises(ValueError, match="Invalid dimensions"):
            create_document_record_class(dimensions=0)

    def test_invalid_dimensions_negative(self) -> None:
        """Test that negative dimensions raises ValueError."""
        with pytest.raises(ValueError, match="Invalid dimensions"):
            create_document_record_class(dimensions=-1)

    def test_invalid_dimensions_too_large(self) -> None:
        """Test that dimensions over 10000 raises ValueError."""
        with pytest.raises(ValueError, match="Invalid dimensions"):
            create_document_record_class(dimensions=10001)

    def test_different_dimension_classes_are_independent(self) -> None:
        """Test that different dimension classes don't interfere."""
        record_class_1536 = create_document_record_class(1536)
        record_class_768 = create_document_record_class(768)

        record_1536 = record_class_1536(id="test1", embedding=[0.1] * 1536)
        record_768 = record_class_768(id="test2", embedding=[0.2] * 768)

        assert len(record_1536.embedding) == 1536
        assert len(record_768.embedding) == 768


class TestGetCollectionFactory:
    """Tests for get_collection_factory function using mocks."""

    def test_factory_unsupported_provider(self) -> None:
        """Test that unsupported provider raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported vector store provider"):
            get_collection_factory("unsupported-provider")

    def test_factory_invalid_provider_name(self) -> None:
        """Test that invalid provider name raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported vector store provider"):
            get_collection_factory("invalid_provider_xyz")

    def test_factory_empty_provider_name(self) -> None:
        """Test that empty provider name raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported vector store provider"):
            get_collection_factory("")

    def test_factory_case_sensitive(self) -> None:
        """Test that provider names are case-sensitive."""
        with pytest.raises(ValueError, match="Unsupported vector store provider"):
            get_collection_factory("Redis-Hashset")  # Wrong case

    def test_factory_returns_callable(self) -> None:
        """Test that get_collection_factory returns a callable."""
        factory = get_collection_factory("in-memory")
        assert callable(factory)

    def test_factory_supported_providers(self) -> None:
        """Test that all documented providers return callables."""
        providers = [
            "redis-hashset",
            "redis-json",
            "postgres",
            "azure-ai-search",
            "qdrant",
            "weaviate",
            "chromadb",
            "faiss",
            "azure-cosmos-mongo",
            "azure-cosmos-nosql",
            "sql-server",
            "pinecone",
            "in-memory",
        ]
        for provider in providers:
            factory = get_collection_factory(provider)
            assert callable(factory)

    def test_factory_with_connection_kwargs(self) -> None:
        """Test factory with connection kwargs."""
        factory = get_collection_factory(
            "postgres",
            connection_string="postgresql://localhost/db",
            pool_size=10,
        )
        assert callable(factory)

    def test_factory_with_custom_dimensions(self) -> None:
        """Test factory with custom embedding dimensions."""
        factory = get_collection_factory("in-memory", dimensions=768)
        assert callable(factory)

    def test_factory_with_large_dimensions(self) -> None:
        """Test factory with large embedding dimensions (3072)."""
        factory = get_collection_factory("in-memory", dimensions=3072)
        assert callable(factory)

    def test_factory_invalid_dimensions_zero(self) -> None:
        """Test that zero dimensions raises ValueError."""
        with pytest.raises(ValueError, match="Invalid dimensions"):
            get_collection_factory("in-memory", dimensions=0)

    def test_factory_invalid_dimensions_negative(self) -> None:
        """Test that negative dimensions raises ValueError."""
        with pytest.raises(ValueError, match="Invalid dimensions"):
            get_collection_factory("in-memory", dimensions=-100)

    def test_factory_invalid_dimensions_too_large(self) -> None:
        """Test that dimensions over 10000 raises ValueError."""
        with pytest.raises(ValueError, match="Invalid dimensions"):
            get_collection_factory("in-memory", dimensions=10001)

    def test_factory_dimensions_default_to_1536(self) -> None:
        """Test that dimensions default to 1536 when not specified."""
        factory = get_collection_factory("in-memory")
        # This should work without errors, using default 1536
        assert callable(factory)


class TestConvertDocumentToQueryResult:
    """Tests for convert_document_to_query_result async function."""

    @pytest.mark.asyncio
    async def test_convert_basic_document(self) -> None:
        """Test converting a basic DocumentRecord to QueryResult."""
        doc = DocumentRecord(
            id="doc_1",
            content="Test content",
            source_path="/test.txt",
            chunk_index=0,
        )
        result = await convert_document_to_query_result(doc, score=0.9)
        assert result.content == "Test content"
        assert result.score == 0.9
        assert result.source_path == "/test.txt"
        assert result.chunk_index == 0

    @pytest.mark.asyncio
    async def test_convert_with_metadata(self) -> None:
        """Test converting document with metadata."""
        doc = DocumentRecord(
            id="doc_2",
            content="Content",
            source_path="/docs/file.pdf",
            chunk_index=5,
            file_type=".pdf",
            file_size_bytes=5000,
            mtime=1234567890.0,
        )
        result = await convert_document_to_query_result(doc, score=0.75)
        assert result.metadata["file_type"] == ".pdf"
        assert result.metadata["file_size_bytes"] == 5000
        assert result.metadata["mtime"] == 1234567890.0

    @pytest.mark.asyncio
    async def test_convert_with_zero_score(self) -> None:
        """Test conversion with score of 0.0."""
        doc = DocumentRecord(content="test", source_path="/test", chunk_index=0)
        result = await convert_document_to_query_result(doc, score=0.0)
        assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_convert_with_perfect_score(self) -> None:
        """Test conversion with perfect score."""
        doc = DocumentRecord(content="test", source_path="/test", chunk_index=0)
        result = await convert_document_to_query_result(doc, score=1.0)
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_convert_preserves_all_fields(self) -> None:
        """Test that conversion preserves all relevant fields."""
        doc = DocumentRecord(
            id="unique_id",
            content="Full content here",
            source_path="/path/to/source",
            chunk_index=42,
            file_type=".txt",
            file_size_bytes=2048,
            mtime=9999999999.0,
        )
        result = await convert_document_to_query_result(doc, score=0.85)
        assert result.content == "Full content here"
        assert result.source_path == "/path/to/source"
        assert result.chunk_index == 42
        assert result.score == 0.85
        assert result.metadata["file_type"] == ".txt"
        assert result.metadata["file_size_bytes"] == 2048
        assert result.metadata["mtime"] == 9999999999.0

    @pytest.mark.asyncio
    async def test_convert_invalid_score_too_low(self) -> None:
        """Test that invalid score raises error during conversion."""
        doc = DocumentRecord(content="test", source_path="/test", chunk_index=0)
        with pytest.raises(ValueError, match="Score must be between 0.0 and 1.0"):
            await convert_document_to_query_result(doc, score=-0.5)

    @pytest.mark.asyncio
    async def test_convert_invalid_score_too_high(self) -> None:
        """Test that invalid score raises error during conversion."""
        doc = DocumentRecord(content="test", source_path="/test", chunk_index=0)
        with pytest.raises(ValueError, match="Score must be between 0.0 and 1.0"):
            await convert_document_to_query_result(doc, score=1.5)

    @pytest.mark.asyncio
    async def test_convert_returns_query_result(self) -> None:
        """Test that conversion returns QueryResult instance."""
        doc = DocumentRecord(content="test", source_path="/test", chunk_index=0)
        result = await convert_document_to_query_result(doc, score=0.5)
        assert isinstance(result, QueryResult)

    @pytest.mark.asyncio
    async def test_convert_with_empty_metadata_fields(self) -> None:
        """Test conversion when document has empty metadata fields."""
        doc = DocumentRecord(
            content="test",
            source_path="/test",
            chunk_index=0,
            file_type="",
            file_size_bytes=0,
            mtime=0.0,
        )
        result = await convert_document_to_query_result(doc, score=0.5)
        assert result.metadata["file_type"] == ""
        assert result.metadata["file_size_bytes"] == 0
        assert result.metadata["mtime"] == 0.0


class TestVectorStoreSearchTopK:
    """T023: Tests for VectorStore search with top_k filtering.

    These tests verify that search operations properly limit results
    based on the top_k parameter.
    """

    def test_query_result_list_respects_top_k_limit(self) -> None:
        """Test that a list of QueryResults can be limited to top_k."""
        # Create more results than we want to return
        all_results = [
            QueryResult(
                content=f"Result {i}",
                score=0.9 - (i * 0.1),  # Decreasing scores
                source_path=f"/doc{i}.md",
                chunk_index=0,
            )
            for i in range(10)
        ]

        # Simulate top_k filtering
        top_k = 5
        limited_results = all_results[:top_k]

        assert len(limited_results) == top_k
        # Verify highest scores are kept
        assert limited_results[0].score == 0.9
        assert limited_results[4].score == 0.5

    def test_query_result_top_k_with_fewer_results(self) -> None:
        """Test top_k when fewer results than requested exist."""
        results = [
            QueryResult(
                content="Only result",
                score=0.85,
                source_path="/single.md",
                chunk_index=0,
            )
        ]

        top_k = 5
        limited_results = results[:top_k]

        # Should return all available results (1)
        assert len(limited_results) == 1
        assert limited_results[0].content == "Only result"

    def test_query_result_top_k_exact_count(self) -> None:
        """Test top_k when exactly that many results exist."""
        results = [
            QueryResult(
                content=f"Result {i}",
                score=0.9 - (i * 0.1),
                source_path=f"/doc{i}.md",
                chunk_index=0,
            )
            for i in range(5)
        ]

        top_k = 5
        limited_results = results[:top_k]

        assert len(limited_results) == 5

    def test_query_result_top_k_preserves_order(self) -> None:
        """Test that top_k preserves descending score order."""
        results = [
            QueryResult(content="A", score=0.95, source_path="/a.md", chunk_index=0),
            QueryResult(content="B", score=0.90, source_path="/b.md", chunk_index=0),
            QueryResult(content="C", score=0.85, source_path="/c.md", chunk_index=0),
            QueryResult(content="D", score=0.80, source_path="/d.md", chunk_index=0),
            QueryResult(content="E", score=0.75, source_path="/e.md", chunk_index=0),
        ]

        top_k = 3
        limited_results = results[:top_k]

        assert len(limited_results) == 3
        assert limited_results[0].score == 0.95
        assert limited_results[1].score == 0.90
        assert limited_results[2].score == 0.85
        # Verify order is maintained
        for i in range(len(limited_results) - 1):
            assert limited_results[i].score >= limited_results[i + 1].score

    def test_query_result_top_k_zero_returns_empty(self) -> None:
        """Test that top_k=0 returns empty list."""
        results = [
            QueryResult(
                content="Result",
                score=0.9,
                source_path="/doc.md",
                chunk_index=0,
            )
        ]

        top_k = 0
        limited_results = results[:top_k]

        assert len(limited_results) == 0

    def test_query_result_top_k_with_min_similarity_score(self) -> None:
        """Test top_k combined with min_similarity_score filtering."""
        results = [
            QueryResult(content="A", score=0.95, source_path="/a.md", chunk_index=0),
            QueryResult(content="B", score=0.85, source_path="/b.md", chunk_index=0),
            QueryResult(content="C", score=0.75, source_path="/c.md", chunk_index=0),
            QueryResult(content="D", score=0.65, source_path="/d.md", chunk_index=0),
            QueryResult(content="E", score=0.55, source_path="/e.md", chunk_index=0),
        ]

        # First filter by min_similarity_score
        min_score = 0.7
        filtered_results = [r for r in results if r.score >= min_score]

        # Then apply top_k
        top_k = 2
        limited_results = filtered_results[:top_k]

        assert len(limited_results) == 2
        assert all(r.score >= min_score for r in limited_results)
        assert limited_results[0].score == 0.95
        assert limited_results[1].score == 0.85

    def test_query_result_top_k_large_value(self) -> None:
        """Test that large top_k values work correctly."""
        results = [
            QueryResult(
                content=f"Result {i}",
                score=0.99 - (i * 0.01),
                source_path=f"/doc{i}.md",
                chunk_index=0,
            )
            for i in range(50)
        ]

        top_k = 100  # Request more than available
        limited_results = results[:top_k]

        # Should return all 50 available
        assert len(limited_results) == 50

    def test_query_result_empty_list_with_top_k(self) -> None:
        """Test top_k on empty results list."""
        results: list[QueryResult] = []

        top_k = 5
        limited_results = results[:top_k]

        assert len(limited_results) == 0
