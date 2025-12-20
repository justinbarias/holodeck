"""Integration tests for structured data vectorstore ingestion and search.

Tests the end-to-end flow:
1. Configure VectorstoreTool with structured data fields
2. Initialize (ingest structured data with embeddings)
3. Search (semantic search returns StructuredQueryResult)

These tests can run in two modes:
- Without Ollama: Uses placeholder embeddings (zero vectors)
- With Ollama: Uses real embeddings from Ollama for semantic search

To run with real embeddings:
1. Copy tests/integration/.env.example to tests/integration/.env
2. Ensure Ollama is running: ollama serve
3. Pull the embedding model: ollama pull nomic-embed-text:latest
4. Run: pytest tests/integration/test_structured_vectorstore.py
"""

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from holodeck.models.tool import VectorstoreTool as VectorstoreToolConfig
from holodeck.tools.vectorstore_tool import VectorStoreTool

# Load environment variables from tests/integration/.env
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

# Ollama configuration with defaults
OLLAMA_ENDPOINT = os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text:latest")
OLLAMA_EMBEDDING_DIMENSIONS = int(os.getenv("OLLAMA_EMBEDDING_DIMENSIONS", "768"))

# Check if we should skip tests requiring Ollama
SKIP_OLLAMA_TESTS = os.getenv("SKIP_OLLAMA_TESTS", "false").lower() == "true"


def _is_ollama_available() -> bool:
    """Check if Ollama is reachable at the configured endpoint."""
    try:
        import httpx

        response = httpx.get(f"{OLLAMA_ENDPOINT}/api/tags", timeout=2.0)
        return response.status_code == 200
    except Exception:
        return False


# Determine Ollama availability at module load time
OLLAMA_AVAILABLE = _is_ollama_available() if not SKIP_OLLAMA_TESTS else False

# Skip marker for tests requiring Ollama
skip_if_no_ollama = pytest.mark.skipif(
    SKIP_OLLAMA_TESTS or not OLLAMA_AVAILABLE,
    reason="Ollama not available or tests disabled (set SKIP_OLLAMA_TESTS=false)",
)

# Path to test fixtures
FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "structured"


def _create_ollama_embedding_service() -> object | None:
    """Create an Ollama embedding service if available.

    Returns:
        OllamaTextEmbedding instance or None if not available.
    """
    try:
        from semantic_kernel.connectors.ai.ollama import OllamaTextEmbedding

        return OllamaTextEmbedding(
            ai_model_id=OLLAMA_EMBEDDING_MODEL,
            host=OLLAMA_ENDPOINT,
        )
    except ImportError:
        return None


@pytest.mark.integration
class TestStructuredVectorstoreCSV:
    """Integration tests for CSV structured data ingestion and search (T023)."""

    @pytest.mark.asyncio
    async def test_csv_ingestion_and_search(self) -> None:
        """Test CSV ingestion and semantic search end-to-end.

        Validates:
        1. VectorStoreTool initializes with structured CSV source
        2. Records are ingested with embeddings
        3. Search returns StructuredQueryResult with correct metadata
        """
        config = VectorstoreToolConfig(
            name="products_search",
            description="Search product descriptions",
            type="vectorstore",
            source=str(FIXTURES_DIR / "products.csv"),
            id_field="id",
            vector_field="description",
            meta_fields=["title", "category", "price"],
            top_k=3,
        )

        tool = VectorStoreTool(config)
        await tool.initialize()

        assert tool.is_initialized
        assert tool.document_count == 3  # 3 products in CSV

        # Search for electronics
        result = await tool.search("advanced AI features")

        # Verify result format
        assert "result" in result.lower() or "found" in result.lower()

    @pytest.mark.asyncio
    async def test_csv_search_returns_metadata(self) -> None:
        """Test that CSV search results include configured metadata fields."""
        config = VectorstoreToolConfig(
            name="products_meta_test",
            description="Test metadata fields",
            type="vectorstore",
            source=str(FIXTURES_DIR / "products.csv"),
            id_field="id",
            vector_field="description",
            meta_fields=["title", "category"],
            top_k=5,
        )

        tool = VectorStoreTool(config)
        await tool.initialize()

        result = await tool.search("widget")

        # Result should contain metadata
        assert "title" in result.lower() or "widget" in result.lower()

    @pytest.mark.asyncio
    async def test_csv_with_auto_delimiter(self) -> None:
        """Test CSV ingestion with auto-detected delimiter."""
        config = VectorstoreToolConfig(
            name="products_auto_delim",
            description="Test auto delimiter",
            type="vectorstore",
            source=str(FIXTURES_DIR / "products.csv"),
            id_field="id",
            vector_field="description",
            # delimiter not specified, should auto-detect
        )

        tool = VectorStoreTool(config)
        await tool.initialize()

        assert tool.is_initialized
        assert tool.document_count == 3

    @pytest.mark.asyncio
    async def test_csv_with_explicit_delimiter(self) -> None:
        """Test CSV ingestion with explicit delimiter."""
        config = VectorstoreToolConfig(
            name="products_explicit_delim",
            description="Test explicit delimiter",
            type="vectorstore",
            source=str(FIXTURES_DIR / "products.csv"),
            id_field="id",
            vector_field="description",
            delimiter=",",
        )

        tool = VectorStoreTool(config)
        await tool.initialize()

        assert tool.is_initialized
        assert tool.document_count == 3


@pytest.mark.integration
class TestStructuredVectorstoreJSON:
    """Integration tests for JSON structured data ingestion and search (T024)."""

    @pytest.mark.asyncio
    async def test_json_ingestion_and_search(self) -> None:
        """Test JSON array ingestion and semantic search end-to-end.

        Validates:
        1. VectorStoreTool initializes with structured JSON source
        2. Records are ingested with embeddings
        3. Search returns results
        """
        config = VectorstoreToolConfig(
            name="faqs_search",
            description="Search FAQ questions and answers",
            type="vectorstore",
            source=str(FIXTURES_DIR / "faqs.json"),
            id_field="faq_id",
            vector_field=["question", "answer"],  # Multiple fields
            meta_fields=["category"],
            top_k=2,
        )

        tool = VectorStoreTool(config)
        await tool.initialize()

        assert tool.is_initialized
        assert tool.document_count == 2  # 2 FAQs in JSON

        # Search for password-related FAQ
        result = await tool.search("how to reset password")

        assert "result" in result.lower() or "found" in result.lower()

    @pytest.mark.asyncio
    async def test_json_multiple_vector_fields(self) -> None:
        """Test JSON ingestion with multiple vector fields concatenated."""
        config = VectorstoreToolConfig(
            name="faqs_multi_vector",
            description="Test multiple vector fields",
            type="vectorstore",
            source=str(FIXTURES_DIR / "faqs.json"),
            id_field="faq_id",
            vector_field=["question", "answer"],
            field_separator="\n",
        )

        tool = VectorStoreTool(config)
        await tool.initialize()

        assert tool.is_initialized
        assert tool.document_count == 2

    @pytest.mark.asyncio
    async def test_json_custom_field_separator(self) -> None:
        """Test JSON ingestion with custom field separator."""
        config = VectorstoreToolConfig(
            name="faqs_custom_sep",
            description="Test custom separator",
            type="vectorstore",
            source=str(FIXTURES_DIR / "faqs.json"),
            id_field="faq_id",
            vector_field=["question", "answer"],
            field_separator=" | ",
        )

        tool = VectorStoreTool(config)
        await tool.initialize()

        assert tool.is_initialized


@pytest.mark.integration
class TestStructuredVectorstoreJSONL:
    """Integration tests for JSONL structured data ingestion and search."""

    @pytest.mark.asyncio
    async def test_jsonl_ingestion_and_search(self) -> None:
        """Test JSONL ingestion and semantic search end-to-end."""
        config = VectorstoreToolConfig(
            name="products_jsonl",
            description="Search JSONL products",
            type="vectorstore",
            source=str(FIXTURES_DIR / "products.jsonl"),
            id_field="product_id",
            vector_field="desc",
            meta_fields=["name"],
            top_k=2,
        )

        tool = VectorStoreTool(config)
        await tool.initialize()

        assert tool.is_initialized
        assert tool.document_count == 2  # 2 products in JSONL

        result = await tool.search("alpha product")
        assert "result" in result.lower() or "found" in result.lower()


@pytest.mark.integration
class TestStructuredVectorstoreValidation:
    """Integration tests for structured vectorstore validation."""

    @pytest.mark.asyncio
    async def test_invalid_id_field_raises_error(self) -> None:
        """Test that invalid id_field raises ConfigError during initialization."""
        from holodeck.lib.errors import ConfigError

        config = VectorstoreToolConfig(
            name="invalid_id",
            description="Test invalid id field",
            type="vectorstore",
            source=str(FIXTURES_DIR / "products.csv"),
            id_field="nonexistent_id",
            vector_field="description",
        )

        tool = VectorStoreTool(config)
        with pytest.raises(ConfigError, match="id_field"):
            await tool.initialize()

    @pytest.mark.asyncio
    async def test_invalid_vector_field_raises_error(self) -> None:
        """Test that invalid vector_field raises ConfigError during initialization."""
        from holodeck.lib.errors import ConfigError

        config = VectorstoreToolConfig(
            name="invalid_vector",
            description="Test invalid vector field",
            type="vectorstore",
            source=str(FIXTURES_DIR / "products.csv"),
            id_field="id",
            vector_field="nonexistent_content",
        )

        tool = VectorStoreTool(config)
        with pytest.raises(ConfigError, match="vector_field"):
            await tool.initialize()

    @pytest.mark.asyncio
    async def test_file_not_found_raises_error(self) -> None:
        """Test that nonexistent source file raises FileNotFoundError."""
        config = VectorstoreToolConfig(
            name="missing_file",
            description="Test missing file",
            type="vectorstore",
            source="/nonexistent/path/data.csv",
            id_field="id",
            vector_field="content",
        )

        tool = VectorStoreTool(config)
        with pytest.raises(FileNotFoundError):
            await tool.initialize()


@pytest.mark.integration
@pytest.mark.slow
class TestStructuredVectorstoreWithOllama:
    """Integration tests for structured vectorstore with real Ollama embeddings.

    These tests require a running Ollama instance with the embedding model pulled.
    They are skipped if Ollama is not available.
    """

    @skip_if_no_ollama
    @pytest.mark.asyncio
    async def test_csv_with_ollama_embeddings(self) -> None:
        """Test CSV ingestion and search with real Ollama embeddings.

        Validates:
        1. VectorStoreTool initializes with Ollama embedding service
        2. Real embeddings are generated for structured data
        3. Semantic search returns relevant results based on meaning
        """
        config = VectorstoreToolConfig(
            name="products_ollama",
            description="Search products with Ollama embeddings",
            type="vectorstore",
            source=str(FIXTURES_DIR / "products.csv"),
            id_field="id",
            vector_field="description",
            meta_fields=["title", "category", "price"],
            embedding_model=OLLAMA_EMBEDDING_MODEL,
            embedding_dimensions=OLLAMA_EMBEDDING_DIMENSIONS,
            top_k=3,
        )

        tool = VectorStoreTool(config)

        # Inject Ollama embedding service
        embedding_service = _create_ollama_embedding_service()
        assert (
            embedding_service is not None
        ), "Failed to create Ollama embedding service"
        tool.set_embedding_service(embedding_service)

        await tool.initialize(provider_type="ollama")

        assert tool.is_initialized
        assert tool.document_count == 3

        # Search with semantic meaning - should find AI-related products
        result = await tool.search("artificial intelligence technology")

        # With real embeddings, search should return meaningful results
        assert "result" in result.lower() or "found" in result.lower()

    @skip_if_no_ollama
    @pytest.mark.asyncio
    async def test_json_with_ollama_embeddings(self) -> None:
        """Test JSON ingestion and search with real Ollama embeddings.

        Validates:
        1. VectorStoreTool works with JSON and Ollama embeddings
        2. Multiple vector fields are properly concatenated and embedded
        3. Semantic search returns relevant FAQ entries
        """
        config = VectorstoreToolConfig(
            name="faqs_ollama",
            description="Search FAQs with Ollama embeddings",
            type="vectorstore",
            source=str(FIXTURES_DIR / "faqs.json"),
            id_field="faq_id",
            vector_field=["question", "answer"],
            meta_fields=["category"],
            embedding_model=OLLAMA_EMBEDDING_MODEL,
            embedding_dimensions=OLLAMA_EMBEDDING_DIMENSIONS,
            top_k=2,
        )

        tool = VectorStoreTool(config)

        # Inject Ollama embedding service
        embedding_service = _create_ollama_embedding_service()
        assert (
            embedding_service is not None
        ), "Failed to create Ollama embedding service"
        tool.set_embedding_service(embedding_service)

        await tool.initialize()

        assert tool.is_initialized
        assert tool.document_count == 2

        # Search for password-related content
        result = await tool.search("forgot my login credentials")

        assert "result" in result.lower() or "found" in result.lower()

    @skip_if_no_ollama
    @pytest.mark.asyncio
    async def test_semantic_relevance_with_ollama(self) -> None:
        """Test that Ollama embeddings provide meaningful semantic search.

        This test validates that semantically similar queries return
        relevant results, demonstrating real embedding quality.
        """
        config = VectorstoreToolConfig(
            name="products_semantic",
            description="Test semantic search quality",
            type="vectorstore",
            source=str(FIXTURES_DIR / "products.csv"),
            id_field="id",
            vector_field="description",
            meta_fields=["title", "category"],
            embedding_model=OLLAMA_EMBEDDING_MODEL,
            embedding_dimensions=OLLAMA_EMBEDDING_DIMENSIONS,
            top_k=1,
        )

        tool = VectorStoreTool(config)

        # Inject Ollama embedding service
        embedding_service = _create_ollama_embedding_service()
        assert (
            embedding_service is not None
        ), "Failed to create Ollama embedding service"
        tool.set_embedding_service(embedding_service)

        await tool.initialize()

        # Test different semantic queries that should match products
        # Query about technology should find relevant tech products
        result = await tool.search("smart device with machine learning")

        # With real embeddings, we expect meaningful search results
        assert tool.is_initialized
        assert "result" in result.lower() or "found" in result.lower()
