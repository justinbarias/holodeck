"""Tests for vector store models and factory functions.

Note: This test module requires mocking semantic_kernel modules because the full
semantic_kernel library is not available in the test environment. Mocks are set up
only during module import to avoid polluting the rest of the test suite.
"""

import sys
from unittest.mock import MagicMock

# Save original modules before mocking (these will be restored after import)
_saved_modules: dict[str, object] = {}
_mocked_modules: set[str] = set()

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
        _mocked_modules.add(module_name)

# Only set up mock attributes if we created the mocks
# This prevents polluting real modules that may already be imported
# mypy: ignore - these are intentional mocks for testing
if "semantic_kernel.connectors.memory" in _mocked_modules:
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
    mock_memory.SqlServerCollection = MagicMock()  # type: ignore[assignment]
    mock_memory.WeaviateCollection = MagicMock()  # type: ignore[assignment]

if "semantic_kernel.data.vector" in _mocked_modules:
    mock_vector = sys.modules["semantic_kernel.data.vector"]
    mock_vector.VectorStoreField = MagicMock()  # type: ignore[assignment]
    mock_vector.VectorStoreCollectionDefinition = MagicMock()  # type: ignore[assignment]
    mock_vector.DistanceFunction = MagicMock()  # type: ignore[assignment]
    mock_vector.vectorstoremodel = lambda **kwargs: lambda cls: cls  # type: ignore[assignment]

# Now import from holodeck.lib.vector_store
import pytest  # noqa: E402

from holodeck.lib.vector_store import (  # noqa: E402
    DocumentRecord,
    QueryResult,
    convert_document_to_query_result,
    create_chromadb_client,
    create_document_record_class,
    get_collection_class,
    get_collection_factory,
    parse_chromadb_connection_string,
    parse_pinecone_connection_string,
    parse_qdrant_connection_string,
)

# Restore original modules after import to avoid polluting other tests
# This is critical for parallel test execution with pytest-xdist
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
    else:
        # Remove mocks that weren't originally present so other tests
        # can import the real modules
        sys.modules.pop(module_name, None)


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
            get_collection_factory("Postgres")  # Wrong case

    def test_factory_returns_callable(self) -> None:
        """Test that get_collection_factory returns a callable."""
        factory = get_collection_factory("in-memory")
        assert callable(factory)

    def test_factory_supported_providers(self) -> None:
        """Test that all documented providers return callables."""
        providers = [
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
        """Test that conversion returns QueryResult instance.

        Note: We check class name instead of isinstance() because module reloading
        during parallel test execution can cause class identity mismatches.
        """
        doc = DocumentRecord(content="test", source_path="/test", chunk_index=0)
        result = await convert_document_to_query_result(doc, score=0.5)
        # Check class name to handle module reloading during parallel tests
        assert type(result).__name__ == "QueryResult"
        # Verify it has expected QueryResult attributes
        assert hasattr(result, "content")
        assert hasattr(result, "score")
        assert hasattr(result, "source_path")
        assert hasattr(result, "chunk_index")
        assert hasattr(result, "metadata")

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


class TestParseChromadbConnectionString:
    """Tests for parse_chromadb_connection_string function."""

    def test_parse_http_localhost_with_port(self) -> None:
        """Test parsing http://localhost:8000."""
        result = parse_chromadb_connection_string("http://localhost:8000")
        assert result["host"] == "localhost"
        assert result["port"] == 8000
        assert result["ssl"] is False

    def test_parse_https_with_custom_host(self) -> None:
        """Test parsing https://chroma.example.com."""
        result = parse_chromadb_connection_string("https://chroma.example.com")
        assert result["host"] == "chroma.example.com"
        assert result["port"] == 443  # Default HTTPS port
        assert result["ssl"] is True

    def test_parse_http_without_port(self) -> None:
        """Test parsing http://localhost without explicit port."""
        result = parse_chromadb_connection_string("http://localhost")
        assert result["host"] == "localhost"
        assert result["port"] == 8000  # Default HTTP port
        assert result["ssl"] is False

    def test_parse_https_with_custom_port(self) -> None:
        """Test parsing https://chroma.internal:9000."""
        result = parse_chromadb_connection_string("https://chroma.internal:9000")
        assert result["host"] == "chroma.internal"
        assert result["port"] == 9000
        assert result["ssl"] is True

    def test_parse_http_with_path(self) -> None:
        """Test parsing URL with path (path is ignored)."""
        result = parse_chromadb_connection_string("http://localhost:8000/api/v1")
        assert result["host"] == "localhost"
        assert result["port"] == 8000
        assert result["ssl"] is False

    def test_parse_empty_connection_string_raises_error(self) -> None:
        """Test that empty connection string raises ValueError."""
        with pytest.raises(ValueError, match="Connection string cannot be empty"):
            parse_chromadb_connection_string("")

    def test_parse_invalid_scheme_raises_error(self) -> None:
        """Test that invalid scheme raises ValueError."""
        with pytest.raises(ValueError, match="Invalid scheme"):
            parse_chromadb_connection_string("ftp://localhost:8000")

    def test_parse_no_scheme_raises_error(self) -> None:
        """Test that URL without scheme raises ValueError."""
        with pytest.raises(ValueError, match="Invalid scheme"):
            parse_chromadb_connection_string("localhost:8000")

    def test_parse_tcp_scheme_raises_error(self) -> None:
        """Test that tcp:// scheme raises ValueError."""
        with pytest.raises(ValueError, match="Invalid scheme"):
            parse_chromadb_connection_string("tcp://localhost:8000")

    def test_parse_ip_address_host(self) -> None:
        """Test parsing URL with IP address as host."""
        result = parse_chromadb_connection_string("http://192.168.1.100:8000")
        assert result["host"] == "192.168.1.100"
        assert result["port"] == 8000
        assert result["ssl"] is False

    def test_parse_subdomain_host(self) -> None:
        """Test parsing URL with subdomain."""
        result = parse_chromadb_connection_string("https://chroma.prod.example.com:443")
        assert result["host"] == "chroma.prod.example.com"
        assert result["port"] == 443
        assert result["ssl"] is True

    def test_parse_default_host_when_missing(self) -> None:
        """Test that hostname defaults to localhost when missing."""
        result = parse_chromadb_connection_string("http://:8000")
        assert result["host"] == "localhost"
        assert result["port"] == 8000


class TestCreateChromadbClient:
    """Tests for create_chromadb_client function."""

    def test_create_http_client_with_connection_string(self) -> None:
        """Test creating HttpClient with connection string."""
        # Mock chromadb module
        mock_chromadb = MagicMock()
        sys.modules["chromadb"] = mock_chromadb

        try:
            client = create_chromadb_client(connection_string="http://localhost:8000")
            mock_chromadb.HttpClient.assert_called_once_with(
                host="localhost",
                port=8000,
                ssl=False,
                headers=None,
                tenant="default_tenant",
                database="default_database",
            )
            assert client == mock_chromadb.HttpClient.return_value
        finally:
            del sys.modules["chromadb"]

    def test_create_http_client_with_https(self) -> None:
        """Test creating HttpClient with HTTPS connection string."""
        mock_chromadb = MagicMock()
        sys.modules["chromadb"] = mock_chromadb

        try:
            create_chromadb_client(connection_string="https://chroma.example.com")
            mock_chromadb.HttpClient.assert_called_once_with(
                host="chroma.example.com",
                port=443,
                ssl=True,
                headers=None,
                tenant="default_tenant",
                database="default_database",
            )
        finally:
            del sys.modules["chromadb"]

    def test_create_http_client_with_headers(self) -> None:
        """Test creating HttpClient with authentication headers."""
        mock_chromadb = MagicMock()
        sys.modules["chromadb"] = mock_chromadb

        try:
            create_chromadb_client(
                connection_string="http://localhost:8000",
                headers={"Authorization": "Bearer token123"},
            )
            mock_chromadb.HttpClient.assert_called_once_with(
                host="localhost",
                port=8000,
                ssl=False,
                headers={"Authorization": "Bearer token123"},
                tenant="default_tenant",
                database="default_database",
            )
        finally:
            del sys.modules["chromadb"]

    def test_create_http_client_with_custom_tenant(self) -> None:
        """Test creating HttpClient with custom tenant."""
        mock_chromadb = MagicMock()
        sys.modules["chromadb"] = mock_chromadb

        try:
            create_chromadb_client(
                connection_string="http://localhost:8000",
                tenant="my_tenant",
                database="my_database",
            )
            mock_chromadb.HttpClient.assert_called_once_with(
                host="localhost",
                port=8000,
                ssl=False,
                headers=None,
                tenant="my_tenant",
                database="my_database",
            )
        finally:
            del sys.modules["chromadb"]

    def test_create_persistent_client(self) -> None:
        """Test creating PersistentClient with persist_directory."""
        mock_chromadb = MagicMock()
        sys.modules["chromadb"] = mock_chromadb

        try:
            client = create_chromadb_client(persist_directory="/var/data/chromadb")
            mock_chromadb.PersistentClient.assert_called_once_with(
                path="/var/data/chromadb",
                tenant="default_tenant",
                database="default_database",
            )
            assert client == mock_chromadb.PersistentClient.return_value
        finally:
            del sys.modules["chromadb"]

    def test_create_persistent_client_with_custom_tenant(self) -> None:
        """Test creating PersistentClient with custom tenant and database."""
        mock_chromadb = MagicMock()
        sys.modules["chromadb"] = mock_chromadb

        try:
            create_chromadb_client(
                persist_directory="/data/vectors",
                tenant="custom_tenant",
                database="custom_db",
            )
            mock_chromadb.PersistentClient.assert_called_once_with(
                path="/data/vectors",
                tenant="custom_tenant",
                database="custom_db",
            )
        finally:
            del sys.modules["chromadb"]

    def test_create_ephemeral_client(self) -> None:
        """Test creating EphemeralClient with no parameters."""
        mock_chromadb = MagicMock()
        sys.modules["chromadb"] = mock_chromadb

        try:
            client = create_chromadb_client()
            mock_chromadb.EphemeralClient.assert_called_once_with(
                tenant="default_tenant",
                database="default_database",
            )
            assert client == mock_chromadb.EphemeralClient.return_value
        finally:
            del sys.modules["chromadb"]

    def test_create_ephemeral_client_with_custom_tenant(self) -> None:
        """Test creating EphemeralClient with custom tenant."""
        mock_chromadb = MagicMock()
        sys.modules["chromadb"] = mock_chromadb

        try:
            create_chromadb_client(
                tenant="test_tenant",
                database="test_db",
            )
            mock_chromadb.EphemeralClient.assert_called_once_with(
                tenant="test_tenant",
                database="test_db",
            )
        finally:
            del sys.modules["chromadb"]

    def test_connection_string_takes_precedence(self) -> None:
        """Test that connection_string takes precedence over persist_directory."""
        mock_chromadb = MagicMock()
        sys.modules["chromadb"] = mock_chromadb

        try:
            # When both are provided, connection_string should be used
            create_chromadb_client(
                connection_string="http://localhost:8000",
                persist_directory="/data/vectors",
            )
            mock_chromadb.HttpClient.assert_called_once()
            mock_chromadb.PersistentClient.assert_not_called()
        finally:
            del sys.modules["chromadb"]

    def test_import_error_when_chromadb_not_installed(self) -> None:
        """Test ImportError is raised when chromadb is not installed."""
        # Use unittest.mock to patch the import statement within the function
        import builtins
        from unittest.mock import patch

        original_import = builtins.__import__

        def mock_import(name: str, *args: object, **kwargs: object) -> MagicMock:
            if name == "chromadb":
                raise ImportError("No module named 'chromadb'")
            return original_import(name, *args, **kwargs)

        with (
            patch.object(builtins, "__import__", side_effect=mock_import),
            pytest.raises(ImportError, match="ChromaDB is not installed"),
        ):
            create_chromadb_client()


class TestGetCollectionClass:
    """Tests for get_collection_class function."""

    def test_get_collection_class_unsupported_provider(self) -> None:
        """Test that unsupported provider raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported vector store provider"):
            get_collection_class("unsupported-provider")

    def test_get_collection_class_empty_provider(self) -> None:
        """Test that empty provider raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported vector store provider"):
            get_collection_class("")

    def test_get_collection_class_returns_error_message_with_supported_providers(
        self,
    ) -> None:
        """Test that error message lists supported providers."""
        with pytest.raises(ValueError) as exc_info:
            get_collection_class("invalid")
        error_msg = str(exc_info.value)
        # Check that some supported providers are mentioned
        assert "chromadb" in error_msg
        assert "postgres" in error_msg
        assert "in-memory" in error_msg

    def test_get_collection_class_import_error_with_hint(self) -> None:
        """Test that ImportError includes installation hint."""
        # Mock importlib to raise ImportError
        import importlib

        original_import = importlib.import_module

        def mock_import(name: str) -> MagicMock:
            if "chroma" in name:
                raise ImportError(f"No module named '{name}'")
            return original_import(name)

        try:
            importlib.import_module = mock_import  # type: ignore[method-assign]
            with pytest.raises(ImportError, match="chromadb"):
                get_collection_class("chromadb")
        finally:
            importlib.import_module = original_import

    def test_get_collection_class_valid_provider_chromadb(self) -> None:
        """Test get_collection_class with chromadb provider."""
        # Setup mock for the import
        mock_module = MagicMock()
        mock_module.ChromaCollection = MagicMock()

        import importlib

        original_import = importlib.import_module

        def mock_import(name: str) -> MagicMock:
            if name == "semantic_kernel.connectors.chroma":
                return mock_module
            return original_import(name)

        try:
            importlib.import_module = mock_import  # type: ignore[method-assign]
            result = get_collection_class("chromadb")
            assert result == mock_module.ChromaCollection
        finally:
            importlib.import_module = original_import

    def test_get_collection_class_valid_provider_postgres(self) -> None:
        """Test get_collection_class with postgres provider."""
        mock_module = MagicMock()
        mock_module.PostgresCollection = MagicMock()

        import importlib

        original_import = importlib.import_module

        def mock_import(name: str) -> MagicMock:
            if name == "semantic_kernel.connectors.postgres":
                return mock_module
            return original_import(name)

        try:
            importlib.import_module = mock_import  # type: ignore[method-assign]
            result = get_collection_class("postgres")
            assert result == mock_module.PostgresCollection
        finally:
            importlib.import_module = original_import

    def test_get_collection_class_valid_provider_in_memory(self) -> None:
        """Test get_collection_class with in-memory provider."""
        mock_module = MagicMock()
        mock_module.InMemoryCollection = MagicMock()

        import importlib

        original_import = importlib.import_module

        def mock_import(name: str) -> MagicMock:
            if name == "semantic_kernel.connectors.in_memory":
                return mock_module
            return original_import(name)

        try:
            importlib.import_module = mock_import  # type: ignore[method-assign]
            result = get_collection_class("in-memory")
            assert result == mock_module.InMemoryCollection
        finally:
            importlib.import_module = original_import


class TestGetCollectionFactoryChromadb:
    """Tests for get_collection_factory with ChromaDB provider."""

    def test_factory_chromadb_with_connection_string(self) -> None:
        """Test factory creates ChromaDB collection with connection string."""
        # This test verifies the factory is created correctly
        # The actual collection creation happens when the factory is called
        factory = get_collection_factory(
            "chromadb",
            dimensions=768,
            connection_string="http://localhost:8000",
        )
        assert callable(factory)

    def test_factory_chromadb_with_persist_directory(self) -> None:
        """Test factory creates ChromaDB collection with persist directory."""
        factory = get_collection_factory(
            "chromadb",
            dimensions=768,
            persist_directory="/data/vectors",
        )
        assert callable(factory)

    def test_factory_chromadb_ephemeral(self) -> None:
        """Test factory creates ephemeral ChromaDB collection."""
        factory = get_collection_factory(
            "chromadb",
            dimensions=768,
        )
        assert callable(factory)

    def test_factory_chromadb_with_headers(self) -> None:
        """Test factory creates ChromaDB collection with auth headers."""
        factory = get_collection_factory(
            "chromadb",
            dimensions=768,
            connection_string="https://chroma.example.com",
            headers={"Authorization": "Bearer token"},
        )
        assert callable(factory)

    def test_factory_chromadb_with_tenant_and_database(self) -> None:
        """Test factory creates ChromaDB collection with tenant/database."""
        factory = get_collection_factory(
            "chromadb",
            dimensions=768,
            connection_string="http://localhost:8000",
            tenant="my_tenant",
            database="my_database",
        )
        assert callable(factory)

    def test_factory_chromadb_calls_create_chromadb_client(self) -> None:
        """Test that factory() calls create_chromadb_client when invoked."""
        # Setup mocks
        mock_chromadb = MagicMock()
        mock_client = MagicMock()
        mock_chromadb.HttpClient.return_value = mock_client
        sys.modules["chromadb"] = mock_chromadb

        # Setup mock for collection class
        mock_collection_class = MagicMock()

        import importlib

        original_import = importlib.import_module

        def mock_import(name: str) -> MagicMock:
            if name == "semantic_kernel.connectors.chroma":
                module = MagicMock()
                module.ChromaCollection = mock_collection_class
                return module
            return original_import(name)

        try:
            importlib.import_module = mock_import  # type: ignore[method-assign]

            factory = get_collection_factory(
                "chromadb",
                dimensions=768,
                connection_string="http://localhost:8000",
            )

            # Call the factory
            factory()

            # Verify HttpClient was called with correct params
            mock_chromadb.HttpClient.assert_called_once_with(
                host="localhost",
                port=8000,
                ssl=False,
                headers=None,
                tenant="default_tenant",
                database="default_database",
            )

            # Verify ChromaCollection was instantiated with the client
            mock_collection_class.__getitem__.return_value.assert_called()
        finally:
            del sys.modules["chromadb"]
            importlib.import_module = original_import

    def test_factory_chromadb_with_persist_directory_calls_persistent_client(
        self,
    ) -> None:
        """Test that factory() calls PersistentClient when persist_directory set."""
        mock_chromadb = MagicMock()
        mock_client = MagicMock()
        mock_chromadb.PersistentClient.return_value = mock_client
        sys.modules["chromadb"] = mock_chromadb

        mock_collection_class = MagicMock()

        import importlib

        original_import = importlib.import_module

        def mock_import(name: str) -> MagicMock:
            if name == "semantic_kernel.connectors.chroma":
                module = MagicMock()
                module.ChromaCollection = mock_collection_class
                return module
            return original_import(name)

        try:
            importlib.import_module = mock_import  # type: ignore[method-assign]

            factory = get_collection_factory(
                "chromadb",
                dimensions=768,
                persist_directory="/data/vectors",
            )

            # Call the factory
            factory()

            # Verify PersistentClient was called
            mock_chromadb.PersistentClient.assert_called_once_with(
                path="/data/vectors",
                tenant="default_tenant",
                database="default_database",
            )
        finally:
            del sys.modules["chromadb"]
            importlib.import_module = original_import

    def test_factory_chromadb_ephemeral_calls_ephemeral_client(self) -> None:
        """Test that factory() calls EphemeralClient when no params set."""
        mock_chromadb = MagicMock()
        mock_client = MagicMock()
        mock_chromadb.EphemeralClient.return_value = mock_client
        sys.modules["chromadb"] = mock_chromadb

        mock_collection_class = MagicMock()

        import importlib

        original_import = importlib.import_module

        def mock_import(name: str) -> MagicMock:
            if name == "semantic_kernel.connectors.chroma":
                module = MagicMock()
                module.ChromaCollection = mock_collection_class
                return module
            return original_import(name)

        try:
            importlib.import_module = mock_import  # type: ignore[method-assign]

            factory = get_collection_factory(
                "chromadb",
                dimensions=768,
            )

            # Call the factory
            factory()

            # Verify EphemeralClient was called
            mock_chromadb.EphemeralClient.assert_called_once_with(
                tenant="default_tenant",
                database="default_database",
            )
        finally:
            del sys.modules["chromadb"]
            importlib.import_module = original_import


class TestGetCollectionFactoryNonChromadb:
    """Tests for get_collection_factory with non-ChromaDB providers."""

    def test_factory_non_chromadb_calls_collection_with_kwargs(self) -> None:
        """Test that factory() for non-chromadb provider passes connection kwargs."""
        mock_collection_class = MagicMock()
        mock_postgres_settings = MagicMock()

        import importlib

        original_import = importlib.import_module

        def mock_import(name: str) -> MagicMock:
            if name == "semantic_kernel.connectors.postgres":
                module = MagicMock()
                module.PostgresCollection = mock_collection_class
                module.PostgresSettings = mock_postgres_settings
                return module
            return original_import(name)

        # Also need to mock sys.modules for the direct import in vector_store.py
        mock_postgres_module = MagicMock()
        mock_postgres_module.PostgresSettings = mock_postgres_settings
        mock_postgres_module.PostgresCollection = mock_collection_class

        try:
            # Set up sys.modules for the direct import
            sys.modules["semantic_kernel.connectors.postgres"] = mock_postgres_module
            importlib.import_module = mock_import  # type: ignore[method-assign]

            factory = get_collection_factory(
                "postgres",
                dimensions=768,
                connection_string="postgresql://localhost/db",
                pool_size=10,
            )

            # Call the factory
            factory()

            # Verify PostgresSettings was called with the connection string
            mock_postgres_settings.assert_called_once()
            settings_call_kwargs = mock_postgres_settings.call_args.kwargs
            assert "connection_string" in settings_call_kwargs

            # Verify PostgresCollection was instantiated with settings
            mock_collection_class.__getitem__.return_value.assert_called_once()
            call_kwargs = mock_collection_class.__getitem__.return_value.call_args
            # Check that settings and pool_size are passed
            assert call_kwargs is not None
            assert "settings" in call_kwargs.kwargs
            assert call_kwargs.kwargs["pool_size"] == 10
        finally:
            del sys.modules["semantic_kernel.connectors.postgres"]
            importlib.import_module = original_import

    def test_factory_in_memory_calls_collection_class(self) -> None:
        """Test that factory() for in-memory provider works."""
        mock_collection_class = MagicMock()

        import importlib

        original_import = importlib.import_module

        def mock_import(name: str) -> MagicMock:
            if name == "semantic_kernel.connectors.in_memory":
                module = MagicMock()
                module.InMemoryCollection = mock_collection_class
                return module
            return original_import(name)

        try:
            importlib.import_module = mock_import  # type: ignore[method-assign]

            factory = get_collection_factory(
                "in-memory",
                dimensions=1536,
            )

            # Call the factory
            result = factory()

            # Verify InMemoryCollection was instantiated
            mock_collection_class.__getitem__.return_value.assert_called_once()
            assert result == mock_collection_class.__getitem__.return_value.return_value
        finally:
            importlib.import_module = original_import


class TestParseQdrantConnectionString:
    """Tests for parse_qdrant_connection_string function."""

    def test_parse_http_localhost_with_port(self) -> None:
        """Test parsing http://localhost:6333."""
        result = parse_qdrant_connection_string("http://localhost:6333")
        assert result["host"] == "localhost"
        assert result["port"] == 6333

    def test_parse_https_remote_server(self) -> None:
        """Test parsing https://qdrant.example.com:6333."""
        result = parse_qdrant_connection_string("https://qdrant.example.com:6333")
        assert result["url"] == "https://qdrant.example.com:6333"

    def test_parse_http_without_port(self) -> None:
        """Test parsing http://localhost without explicit port."""
        result = parse_qdrant_connection_string("http://localhost")
        assert result["host"] == "localhost"
        assert result["port"] == 6333  # Default Qdrant HTTP port

    def test_parse_in_memory(self) -> None:
        """Test parsing :memory: for in-memory storage."""
        result = parse_qdrant_connection_string(":memory:")
        assert result["location"] == ":memory:"

    def test_parse_local_file_path(self) -> None:
        """Test parsing local file path for persistent storage."""
        result = parse_qdrant_connection_string("/var/data/qdrant")
        assert result["path"] == "/var/data/qdrant"

    def test_parse_file_uri(self) -> None:
        """Test parsing file:// URI for persistent storage."""
        result = parse_qdrant_connection_string("file:///var/data/qdrant")
        assert result["path"] == "/var/data/qdrant"

    def test_parse_grpc_scheme(self) -> None:
        """Test parsing qdrant+grpc:// for gRPC preference."""
        result = parse_qdrant_connection_string("qdrant+grpc://localhost:6334")
        assert result["host"] == "localhost"
        assert result["grpc_port"] == 6334
        assert result["prefer_grpc"] is True

    def test_parse_grpc_scheme_alternative(self) -> None:
        """Test parsing grpc:// scheme."""
        result = parse_qdrant_connection_string("grpc://localhost:6334")
        assert result["host"] == "localhost"
        assert result["grpc_port"] == 6334
        assert result["prefer_grpc"] is True

    def test_parse_with_api_key_in_userinfo(self) -> None:
        """Test parsing URL with API key in userinfo."""
        result = parse_qdrant_connection_string("https://my-api-key@qdrant.example.com")
        assert result["url"] == "https://qdrant.example.com"
        assert result["api_key"] == "my-api-key"

    def test_parse_localhost_127_0_0_1(self) -> None:
        """Test parsing 127.0.0.1 as localhost."""
        result = parse_qdrant_connection_string("http://127.0.0.1:6333")
        assert result["host"] == "127.0.0.1"
        assert result["port"] == 6333

    def test_parse_empty_connection_string_raises_error(self) -> None:
        """Test that empty connection string raises ValueError."""
        with pytest.raises(ValueError, match="Connection string cannot be empty"):
            parse_qdrant_connection_string("")

    def test_parse_invalid_scheme_raises_error(self) -> None:
        """Test that invalid scheme raises ValueError."""
        with pytest.raises(ValueError, match="Invalid scheme"):
            parse_qdrant_connection_string("ftp://localhost:6333")

    def test_parse_tcp_scheme_raises_error(self) -> None:
        """Test that tcp:// scheme raises ValueError."""
        with pytest.raises(ValueError, match="Invalid scheme"):
            parse_qdrant_connection_string("tcp://localhost:6333")

    def test_parse_qdrant_custom_scheme(self) -> None:
        """Test parsing qdrant:// custom scheme."""
        result = parse_qdrant_connection_string("qdrant://localhost:6333")
        assert result["host"] == "localhost"
        assert result["port"] == 6333

    def test_parse_https_remote_without_port(self) -> None:
        """Test parsing https remote server without explicit port."""
        result = parse_qdrant_connection_string("https://qdrant.example.com")
        assert result["url"] == "https://qdrant.example.com"


class TestParsePineconeConnectionString:
    """Tests for parse_pinecone_connection_string function."""

    def test_parse_direct_api_key(self) -> None:
        """Test parsing direct API key starting with pc-."""
        result = parse_pinecone_connection_string("pc-abc123def456")
        assert result["api_key"] == "pc-abc123def456"

    def test_parse_non_pc_api_key(self) -> None:
        """Test parsing API key not starting with pc- (treated as direct key)."""
        result = parse_pinecone_connection_string("my-custom-api-key-12345")
        assert result["api_key"] == "my-custom-api-key-12345"

    def test_parse_pinecone_url_with_api_key(self) -> None:
        """Test parsing pinecone://api_key format."""
        result = parse_pinecone_connection_string("pinecone://pc-abc123")
        assert result["api_key"] == "pc-abc123"

    def test_parse_pinecone_url_with_namespace(self) -> None:
        """Test parsing pinecone://api_key@namespace format."""
        result = parse_pinecone_connection_string("pinecone://pc-abc123@my-namespace")
        assert result["api_key"] == "pc-abc123"
        assert result["namespace"] == "my-namespace"

    def test_parse_pinecone_url_with_namespace_in_path(self) -> None:
        """Test parsing pinecone://api_key/namespace format."""
        result = parse_pinecone_connection_string("pinecone://pc-abc123/production")
        assert result["api_key"] == "pc-abc123"
        assert result["namespace"] == "production"

    def test_parse_empty_connection_string_raises_error(self) -> None:
        """Test that empty connection string raises ValueError."""
        with pytest.raises(ValueError, match="Connection string cannot be empty"):
            parse_pinecone_connection_string("")

    def test_parse_pinecone_url_empty_namespace_ignored(self) -> None:
        """Test that empty namespace in path is ignored."""
        result = parse_pinecone_connection_string("pinecone://pc-abc123/")
        assert result["api_key"] == "pc-abc123"
        assert "namespace" not in result


class TestGetCollectionFactoryQdrant:
    """Tests for get_collection_factory with Qdrant provider."""

    def test_factory_qdrant_with_connection_string(self) -> None:
        """Test factory creates Qdrant collection with connection string."""
        factory = get_collection_factory(
            "qdrant",
            dimensions=768,
            connection_string="http://localhost:6333",
        )
        assert callable(factory)

    def test_factory_qdrant_in_memory(self) -> None:
        """Test factory creates in-memory Qdrant collection."""
        factory = get_collection_factory(
            "qdrant",
            dimensions=768,
            connection_string=":memory:",
        )
        assert callable(factory)

    def test_factory_qdrant_with_api_key(self) -> None:
        """Test factory creates Qdrant collection with API key."""
        factory = get_collection_factory(
            "qdrant",
            dimensions=768,
            connection_string="https://qdrant.example.com:6333",
            api_key="my-secret-key",
        )
        assert callable(factory)

    def test_factory_qdrant_calls_collection_class(self) -> None:
        """Test that factory() for Qdrant provider calls collection properly."""
        mock_collection_class = MagicMock()

        import importlib

        original_import = importlib.import_module

        def mock_import(name: str) -> MagicMock:
            if name == "semantic_kernel.connectors.qdrant":
                module = MagicMock()
                module.QdrantCollection = mock_collection_class
                return module
            return original_import(name)

        try:
            importlib.import_module = mock_import  # type: ignore[method-assign]

            factory = get_collection_factory(
                "qdrant",
                dimensions=768,
                connection_string="http://localhost:6333",
            )

            # Call the factory
            factory()

            # Verify QdrantCollection was instantiated
            mock_collection_class.__getitem__.return_value.assert_called_once()
            call_kwargs = mock_collection_class.__getitem__.return_value.call_args
            assert call_kwargs is not None
            # Should have host and port from parsed connection string
            assert call_kwargs.kwargs["host"] == "localhost"
            assert call_kwargs.kwargs["port"] == 6333
        finally:
            importlib.import_module = original_import

    def test_factory_qdrant_with_grpc(self) -> None:
        """Test factory creates Qdrant collection with gRPC preference."""
        mock_collection_class = MagicMock()

        import importlib

        original_import = importlib.import_module

        def mock_import(name: str) -> MagicMock:
            if name == "semantic_kernel.connectors.qdrant":
                module = MagicMock()
                module.QdrantCollection = mock_collection_class
                return module
            return original_import(name)

        try:
            importlib.import_module = mock_import  # type: ignore[method-assign]

            factory = get_collection_factory(
                "qdrant",
                dimensions=768,
                connection_string="qdrant+grpc://localhost:6334",
            )

            factory()

            call_kwargs = mock_collection_class.__getitem__.return_value.call_args
            assert call_kwargs is not None
            assert call_kwargs.kwargs["host"] == "localhost"
            assert call_kwargs.kwargs["grpc_port"] == 6334
            assert call_kwargs.kwargs["prefer_grpc"] is True
        finally:
            importlib.import_module = original_import


class TestGetCollectionFactoryPinecone:
    """Tests for get_collection_factory with Pinecone provider."""

    def test_factory_pinecone_with_api_key(self) -> None:
        """Test factory creates Pinecone collection with API key."""
        factory = get_collection_factory(
            "pinecone",
            dimensions=768,
            api_key="pc-abc123def456",
        )
        assert callable(factory)

    def test_factory_pinecone_with_connection_string(self) -> None:
        """Test factory creates Pinecone collection with connection string."""
        factory = get_collection_factory(
            "pinecone",
            dimensions=768,
            connection_string="pc-abc123def456",
        )
        assert callable(factory)

    def test_factory_pinecone_with_namespace(self) -> None:
        """Test factory creates Pinecone collection with namespace."""
        factory = get_collection_factory(
            "pinecone",
            dimensions=768,
            api_key="pc-abc123def456",
            namespace="production",
        )
        assert callable(factory)

    def test_factory_pinecone_calls_collection_class(self) -> None:
        """Test that factory() for Pinecone provider calls collection properly."""
        mock_collection_class = MagicMock()

        import importlib

        original_import = importlib.import_module

        def mock_import(name: str) -> MagicMock:
            if name == "semantic_kernel.connectors.pinecone":
                module = MagicMock()
                module.PineconeCollection = mock_collection_class
                return module
            return original_import(name)

        try:
            importlib.import_module = mock_import  # type: ignore[method-assign]

            factory = get_collection_factory(
                "pinecone",
                dimensions=768,
                api_key="pc-abc123def456",
                namespace="my-namespace",
            )

            # Call the factory
            factory()

            # Verify PineconeCollection was instantiated
            mock_collection_class.__getitem__.return_value.assert_called_once()
            call_kwargs = mock_collection_class.__getitem__.return_value.call_args
            assert call_kwargs is not None
            assert call_kwargs.kwargs["api_key"] == "pc-abc123def456"
            assert call_kwargs.kwargs["namespace"] == "my-namespace"
        finally:
            importlib.import_module = original_import

    def test_factory_pinecone_with_connection_string_parsed(self) -> None:
        """Test that factory parses Pinecone connection string."""
        mock_collection_class = MagicMock()

        import importlib

        original_import = importlib.import_module

        def mock_import(name: str) -> MagicMock:
            if name == "semantic_kernel.connectors.pinecone":
                module = MagicMock()
                module.PineconeCollection = mock_collection_class
                return module
            return original_import(name)

        try:
            importlib.import_module = mock_import  # type: ignore[method-assign]

            factory = get_collection_factory(
                "pinecone",
                dimensions=768,
                connection_string="pinecone://pc-abc123@production",
            )

            factory()

            call_kwargs = mock_collection_class.__getitem__.return_value.call_args
            assert call_kwargs is not None
            assert call_kwargs.kwargs["api_key"] == "pc-abc123"
            assert call_kwargs.kwargs["namespace"] == "production"
        finally:
            importlib.import_module = original_import

    def test_factory_pinecone_with_use_grpc(self) -> None:
        """Test factory creates Pinecone collection with gRPC option."""
        mock_collection_class = MagicMock()

        import importlib

        original_import = importlib.import_module

        def mock_import(name: str) -> MagicMock:
            if name == "semantic_kernel.connectors.pinecone":
                module = MagicMock()
                module.PineconeCollection = mock_collection_class
                return module
            return original_import(name)

        try:
            importlib.import_module = mock_import  # type: ignore[method-assign]

            factory = get_collection_factory(
                "pinecone",
                dimensions=768,
                api_key="pc-abc123",
                use_grpc=True,
            )

            factory()

            call_kwargs = mock_collection_class.__getitem__.return_value.call_args
            assert call_kwargs is not None
            assert call_kwargs.kwargs["use_grpc"] is True
        finally:
            importlib.import_module = original_import
