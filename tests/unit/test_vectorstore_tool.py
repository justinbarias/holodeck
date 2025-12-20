"""Unit tests for VectorStoreTool.

Tests for the VectorStoreTool class that provides semantic search over
unstructured documents. These tests are written following TDD - they should
FAIL until the implementation is complete.

Test IDs:
- T016: VectorStoreTool initialization with valid config
- T017: VectorStoreTool initialization with missing source path
- T018: VectorStoreTool file discovery (single file)
- T019: VectorStoreTool file discovery (directory with nested subdirectories)
- T020: VectorStoreTool search result formatting
- T021: VectorStoreTool embedding service injection
- T022: VectorStoreTool collection setup
- T023: VectorStoreTool file processor lazy init
- T024: VectorStoreTool source path resolution with context var
- T025: VectorStoreTool file processing
- T026: VectorStoreTool chunk embedding
- T027: VectorStoreTool chunk storage
- T028: VectorStoreTool initialization full flow
- T029: VectorStoreTool search success path
- T030: VectorStoreTool search documents

Note: This test module requires mocking semantic_kernel modules because the full
semantic_kernel library is not available in the test environment.
"""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# Mock semantic_kernel modules BEFORE importing holodeck modules
# This prevents import errors from semantic_kernel dependencies
# Track which modules we actually mock to avoid polluting real modules
_mocked_modules: set[str] = set()
for module_name in [
    "semantic_kernel",
    "semantic_kernel.connectors",
    "semantic_kernel.connectors.memory",
    "semantic_kernel.connectors.ai",
    "semantic_kernel.contents",
    "semantic_kernel.data",
    "semantic_kernel.data.vector",
    "semantic_kernel.text",
]:
    if module_name not in sys.modules:
        sys.modules[module_name] = MagicMock()
        _mocked_modules.add(module_name)

# Only set up mock attributes if we created the mocks
# This prevents polluting real modules that may already be imported
if "semantic_kernel.connectors.memory" in _mocked_modules:
    mock_memory = sys.modules["semantic_kernel.connectors.memory"]
    mock_memory.AzureAISearchCollection = MagicMock()
    mock_memory.ChromaCollection = MagicMock()
    mock_memory.CosmosMongoCollection = MagicMock()
    mock_memory.CosmosNoSqlCollection = MagicMock()
    mock_memory.FaissCollection = MagicMock()
    mock_memory.InMemoryCollection = MagicMock()
    mock_memory.PineconeCollection = MagicMock()
    mock_memory.PostgresCollection = MagicMock()
    mock_memory.QdrantCollection = MagicMock()
    mock_memory.SqlServerCollection = MagicMock()
    mock_memory.WeaviateCollection = MagicMock()

if "semantic_kernel.data.vector" in _mocked_modules:
    mock_vector = sys.modules["semantic_kernel.data.vector"]
    mock_vector.VectorStoreField = MagicMock()
    mock_vector.vectorstoremodel = lambda **kwargs: lambda cls: cls

if "semantic_kernel.text" in _mocked_modules:
    mock_text = sys.modules["semantic_kernel.text"]
    mock_text.split_plaintext_paragraph = MagicMock(
        side_effect=lambda lines, max_tokens: (
            lines if isinstance(lines, list) else [lines]
        )
    )

import pytest  # noqa: E402

from holodeck.models.tool import VectorstoreTool  # noqa: E402


class TestVectorStoreToolInitialization:
    """T016: Tests for VectorStoreTool initialization with valid config."""

    def test_init_with_valid_config(self, tmp_path: Path) -> None:
        """Test VectorStoreTool initialization with valid configuration."""
        # Create a test file
        source_file = tmp_path / "test.md"
        source_file.write_text("# Test content\n\nThis is test content.")

        # Create valid config
        config = VectorstoreTool(
            name="test_vectorstore",
            description="Test vectorstore tool",
            source=str(source_file),
        )

        # Import and create the tool
        from holodeck.tools.vectorstore_tool import VectorStoreTool

        tool = VectorStoreTool(config)

        # Assertions
        assert tool.config == config
        assert tool.is_initialized is False
        assert tool.document_count == 0

    def test_init_with_custom_embedding_model(self, tmp_path: Path) -> None:
        """Test initialization with custom embedding model."""
        source_file = tmp_path / "test.md"
        source_file.write_text("# Test")

        config = VectorstoreTool(
            name="test_vectorstore",
            description="Test tool",
            source=str(source_file),
            embedding_model="text-embedding-3-large",
        )

        from holodeck.tools.vectorstore_tool import VectorStoreTool

        tool = VectorStoreTool(config)

        assert tool.config.embedding_model == "text-embedding-3-large"

    def test_init_with_custom_top_k(self, tmp_path: Path) -> None:
        """Test initialization with custom top_k parameter."""
        source_file = tmp_path / "test.md"
        source_file.write_text("# Test")

        config = VectorstoreTool(
            name="test_vectorstore",
            description="Test tool",
            source=str(source_file),
            top_k=10,
        )

        from holodeck.tools.vectorstore_tool import VectorStoreTool

        tool = VectorStoreTool(config)

        assert tool.config.top_k == 10

    def test_init_with_min_similarity_score(self, tmp_path: Path) -> None:
        """Test initialization with min_similarity_score parameter."""
        source_file = tmp_path / "test.md"
        source_file.write_text("# Test")

        config = VectorstoreTool(
            name="test_vectorstore",
            description="Test tool",
            source=str(source_file),
            min_similarity_score=0.7,
        )

        from holodeck.tools.vectorstore_tool import VectorStoreTool

        tool = VectorStoreTool(config)

        assert tool.config.min_similarity_score == 0.7

    def test_init_with_chunk_settings(self, tmp_path: Path) -> None:
        """Test initialization with custom chunking settings."""
        source_file = tmp_path / "test.md"
        source_file.write_text("# Test")

        config = VectorstoreTool(
            name="test_vectorstore",
            description="Test tool",
            source=str(source_file),
            chunk_size=256,
            chunk_overlap=25,
        )

        from holodeck.tools.vectorstore_tool import VectorStoreTool

        tool = VectorStoreTool(config)

        assert tool.config.chunk_size == 256
        assert tool.config.chunk_overlap == 25


class TestVectorStoreToolMissingSourcePath:
    """T017: Tests for VectorStoreTool initialization with missing source path."""

    @pytest.mark.asyncio
    async def test_initialize_with_nonexistent_file_raises_error(
        self, tmp_path: Path
    ) -> None:
        """Test that initialize raises FileNotFoundError for nonexistent file."""
        nonexistent_path = tmp_path / "nonexistent.md"

        config = VectorstoreTool(
            name="test_vectorstore",
            description="Test tool",
            source=str(nonexistent_path),
        )

        from holodeck.tools.vectorstore_tool import VectorStoreTool

        tool = VectorStoreTool(config)

        with pytest.raises(FileNotFoundError) as exc_info:
            await tool.initialize()

        assert "nonexistent.md" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_initialize_with_nonexistent_directory_raises_error(
        self, tmp_path: Path
    ) -> None:
        """Test that initialize raises FileNotFoundError for nonexistent directory."""
        nonexistent_dir = tmp_path / "nonexistent_dir"

        config = VectorstoreTool(
            name="test_vectorstore",
            description="Test tool",
            source=str(nonexistent_dir),
        )

        from holodeck.tools.vectorstore_tool import VectorStoreTool

        tool = VectorStoreTool(config)

        with pytest.raises(FileNotFoundError) as exc_info:
            await tool.initialize()

        assert "nonexistent_dir" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_initialize_error_message_contains_path(self, tmp_path: Path) -> None:
        """Test that error message includes the missing path for clarity."""
        missing_path = tmp_path / "missing_data" / "docs"

        config = VectorstoreTool(
            name="test_vectorstore",
            description="Test tool",
            source=str(missing_path),
        )

        from holodeck.tools.vectorstore_tool import VectorStoreTool

        tool = VectorStoreTool(config)

        with pytest.raises(FileNotFoundError) as exc_info:
            await tool.initialize()

        # Error message should contain the full path for debugging
        assert str(missing_path) in str(exc_info.value) or "missing_data" in str(
            exc_info.value
        )


class TestVectorStoreToolFileDiscoverySingleFile:
    """T018: Tests for VectorStoreTool file discovery with single file."""

    def test_discover_files_single_markdown(self, tmp_path: Path) -> None:
        """Test discovery of a single markdown file."""
        md_file = tmp_path / "document.md"
        md_file.write_text("# Markdown content")

        config = VectorstoreTool(
            name="test_vectorstore",
            description="Test tool",
            source=str(md_file),
        )

        from holodeck.tools.vectorstore_tool import VectorStoreTool

        tool = VectorStoreTool(config)
        discovered = tool._discover_files()

        assert len(discovered) == 1
        assert discovered[0] == md_file

    def test_discover_files_single_txt(self, tmp_path: Path) -> None:
        """Test discovery of a single text file."""
        txt_file = tmp_path / "document.txt"
        txt_file.write_text("Plain text content")

        config = VectorstoreTool(
            name="test_vectorstore",
            description="Test tool",
            source=str(txt_file),
        )

        from holodeck.tools.vectorstore_tool import VectorStoreTool

        tool = VectorStoreTool(config)
        discovered = tool._discover_files()

        assert len(discovered) == 1
        assert discovered[0] == txt_file

    def test_discover_files_single_pdf(self, tmp_path: Path) -> None:
        """Test discovery of a single PDF file."""
        # Create a dummy PDF file (just for discovery, not content)
        pdf_file = tmp_path / "document.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 dummy content")

        config = VectorstoreTool(
            name="test_vectorstore",
            description="Test tool",
            source=str(pdf_file),
        )

        from holodeck.tools.vectorstore_tool import VectorStoreTool

        tool = VectorStoreTool(config)
        discovered = tool._discover_files()

        assert len(discovered) == 1
        assert discovered[0] == pdf_file

    def test_discover_files_single_csv(self, tmp_path: Path) -> None:
        """Test discovery of a single CSV file."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("col1,col2\nval1,val2")

        config = VectorstoreTool(
            name="test_vectorstore",
            description="Test tool",
            source=str(csv_file),
        )

        from holodeck.tools.vectorstore_tool import VectorStoreTool

        tool = VectorStoreTool(config)
        discovered = tool._discover_files()

        assert len(discovered) == 1
        assert discovered[0] == csv_file

    def test_discover_files_single_json(self, tmp_path: Path) -> None:
        """Test discovery of a single JSON file."""
        json_file = tmp_path / "data.json"
        json_file.write_text('{"key": "value"}')

        config = VectorstoreTool(
            name="test_vectorstore",
            description="Test tool",
            source=str(json_file),
        )

        from holodeck.tools.vectorstore_tool import VectorStoreTool

        tool = VectorStoreTool(config)
        discovered = tool._discover_files()

        assert len(discovered) == 1
        assert discovered[0] == json_file

    def test_discover_files_returns_path_object(self, tmp_path: Path) -> None:
        """Test that discovered files are returned as Path objects."""
        md_file = tmp_path / "document.md"
        md_file.write_text("# Content")

        config = VectorstoreTool(
            name="test_vectorstore",
            description="Test tool",
            source=str(md_file),
        )

        from holodeck.tools.vectorstore_tool import VectorStoreTool

        tool = VectorStoreTool(config)
        discovered = tool._discover_files()

        assert all(isinstance(f, Path) for f in discovered)


class TestVectorStoreToolFileDiscoveryDirectory:
    """T019: Tests for VectorStoreTool file discovery with directories."""

    def test_discover_files_flat_directory(self, tmp_path: Path) -> None:
        """Test discovery in a flat directory (no subdirectories)."""
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()

        # Create multiple supported files
        (docs_dir / "doc1.md").write_text("# Doc 1")
        (docs_dir / "doc2.txt").write_text("Doc 2")
        (docs_dir / "doc3.csv").write_text("a,b\n1,2")

        config = VectorstoreTool(
            name="test_vectorstore",
            description="Test tool",
            source=str(docs_dir),
        )

        from holodeck.tools.vectorstore_tool import VectorStoreTool

        tool = VectorStoreTool(config)
        discovered = tool._discover_files()

        assert len(discovered) == 3
        extensions = {f.suffix for f in discovered}
        assert extensions == {".md", ".txt", ".csv"}

    def test_discover_files_nested_directories(self, tmp_path: Path) -> None:
        """Test discovery recursively traverses nested subdirectories."""
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()

        # Create nested structure
        api_dir = docs_dir / "api"
        api_dir.mkdir()
        guides_dir = docs_dir / "guides"
        guides_dir.mkdir()
        deep_dir = guides_dir / "advanced"
        deep_dir.mkdir()

        # Create files at different levels
        (docs_dir / "readme.md").write_text("# Root")
        (api_dir / "endpoints.md").write_text("# API")
        (api_dir / "schemas.json").write_text("{}")
        (guides_dir / "quickstart.md").write_text("# Guide")
        (deep_dir / "advanced.txt").write_text("Advanced")

        config = VectorstoreTool(
            name="test_vectorstore",
            description="Test tool",
            source=str(docs_dir),
        )

        from holodeck.tools.vectorstore_tool import VectorStoreTool

        tool = VectorStoreTool(config)
        discovered = tool._discover_files()

        assert len(discovered) == 5
        filenames = {f.name for f in discovered}
        assert filenames == {
            "readme.md",
            "endpoints.md",
            "schemas.json",
            "quickstart.md",
            "advanced.txt",
        }

    def test_discover_files_skips_unsupported_extensions(self, tmp_path: Path) -> None:
        """Test that unsupported file extensions are skipped."""
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()

        # Create supported files
        (docs_dir / "doc.md").write_text("# Supported")
        # Create unsupported files
        (docs_dir / "image.png").write_bytes(b"PNG data")
        (docs_dir / "binary.exe").write_bytes(b"Binary")
        (docs_dir / "config.yaml").write_text("key: value")
        (docs_dir / "script.py").write_text("print('hello')")

        config = VectorstoreTool(
            name="test_vectorstore",
            description="Test tool",
            source=str(docs_dir),
        )

        from holodeck.tools.vectorstore_tool import VectorStoreTool

        tool = VectorStoreTool(config)
        discovered = tool._discover_files()

        # Only the .md file should be discovered
        assert len(discovered) == 1
        assert discovered[0].suffix == ".md"

    def test_discover_files_mixed_supported_unsupported(self, tmp_path: Path) -> None:
        """Test discovery with mix of supported and unsupported files."""
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()

        # Supported
        (docs_dir / "doc1.md").write_text("# MD")
        (docs_dir / "doc2.txt").write_text("TXT")
        (docs_dir / "data.csv").write_text("a,b")
        (docs_dir / "data.json").write_text("{}")
        # Unsupported
        (docs_dir / "img.jpg").write_bytes(b"JPEG")
        (docs_dir / "doc.docx").write_bytes(b"DOCX")

        config = VectorstoreTool(
            name="test_vectorstore",
            description="Test tool",
            source=str(docs_dir),
        )

        from holodeck.tools.vectorstore_tool import VectorStoreTool

        tool = VectorStoreTool(config)
        discovered = tool._discover_files()

        assert len(discovered) == 4
        extensions = {f.suffix for f in discovered}
        assert extensions == {".md", ".txt", ".csv", ".json"}

    def test_discover_files_empty_directory(self, tmp_path: Path) -> None:
        """Test discovery in empty directory returns empty list."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        config = VectorstoreTool(
            name="test_vectorstore",
            description="Test tool",
            source=str(empty_dir),
        )

        from holodeck.tools.vectorstore_tool import VectorStoreTool

        tool = VectorStoreTool(config)
        discovered = tool._discover_files()

        assert len(discovered) == 0

    def test_discover_files_directory_only_unsupported(self, tmp_path: Path) -> None:
        """Test discovery in directory with only unsupported files."""
        unsupported_dir = tmp_path / "unsupported"
        unsupported_dir.mkdir()

        (unsupported_dir / "image.png").write_bytes(b"PNG")
        (unsupported_dir / "config.yaml").write_text("key: value")

        config = VectorstoreTool(
            name="test_vectorstore",
            description="Test tool",
            source=str(unsupported_dir),
        )

        from holodeck.tools.vectorstore_tool import VectorStoreTool

        tool = VectorStoreTool(config)
        discovered = tool._discover_files()

        assert len(discovered) == 0

    def test_discover_files_deeply_nested_structure(self, tmp_path: Path) -> None:
        """Test discovery in deeply nested directory structure."""
        # Create deep nesting: level1/level2/level3/level4/doc.md
        current = tmp_path / "root"
        current.mkdir()

        for i in range(1, 5):
            current = current / f"level{i}"
            current.mkdir()
            (current / f"doc_level{i}.md").write_text(f"# Level {i}")

        config = VectorstoreTool(
            name="test_vectorstore",
            description="Test tool",
            source=str(tmp_path / "root"),
        )

        from holodeck.tools.vectorstore_tool import VectorStoreTool

        tool = VectorStoreTool(config)
        discovered = tool._discover_files()

        # Should find all 4 documents at different levels
        assert len(discovered) == 4
        filenames = {f.name for f in discovered}
        assert filenames == {
            "doc_level1.md",
            "doc_level2.md",
            "doc_level3.md",
            "doc_level4.md",
        }


class TestVectorStoreToolSearchResultFormatting:
    """T020: Tests for VectorStoreTool search result formatting."""

    def test_format_results_single_result(self) -> None:
        """Test formatting of a single search result."""
        from holodeck.lib.vector_store import QueryResult
        from holodeck.tools.vectorstore_tool import VectorStoreTool

        results = [
            QueryResult(
                content="This is the matched content.",
                score=0.89,
                source_path="/path/to/doc.md",
                chunk_index=0,
            )
        ]

        formatted = VectorStoreTool._format_results(results, "test query")

        assert "Found 1 result" in formatted
        assert "[1]" in formatted
        assert "Score: 0.89" in formatted
        assert "/path/to/doc.md" in formatted
        assert "This is the matched content." in formatted

    def test_format_results_multiple_results(self) -> None:
        """Test formatting of multiple search results."""
        from holodeck.lib.vector_store import QueryResult
        from holodeck.tools.vectorstore_tool import VectorStoreTool

        results = [
            QueryResult(
                content="First result content",
                score=0.95,
                source_path="/docs/first.md",
                chunk_index=0,
            ),
            QueryResult(
                content="Second result content",
                score=0.82,
                source_path="/docs/second.md",
                chunk_index=1,
            ),
            QueryResult(
                content="Third result content",
                score=0.75,
                source_path="/docs/third.txt",
                chunk_index=0,
            ),
        ]

        formatted = VectorStoreTool._format_results(results, "test query")

        assert "Found 3 result" in formatted
        assert "[1]" in formatted
        assert "[2]" in formatted
        assert "[3]" in formatted
        assert "Score: 0.95" in formatted
        assert "Score: 0.82" in formatted
        assert "Score: 0.75" in formatted
        assert "/docs/first.md" in formatted
        assert "/docs/second.md" in formatted
        assert "/docs/third.txt" in formatted

    def test_format_results_no_results(self) -> None:
        """Test formatting when no results are found."""
        from holodeck.tools.vectorstore_tool import VectorStoreTool

        formatted = VectorStoreTool._format_results([], "my search query")

        assert "No relevant results found" in formatted
        assert "my search query" in formatted

    def test_format_results_score_formatting(self) -> None:
        """Test that scores are formatted with 2 decimal places."""
        from holodeck.lib.vector_store import QueryResult
        from holodeck.tools.vectorstore_tool import VectorStoreTool

        results = [
            QueryResult(
                content="Content",
                score=0.8888888,
                source_path="/doc.md",
                chunk_index=0,
            )
        ]

        formatted = VectorStoreTool._format_results(results, "query")

        # Score should be formatted to 2 decimal places
        assert "0.89" in formatted
        assert "0.8888888" not in formatted

    def test_format_results_preserves_content(self) -> None:
        """Test that content is preserved without truncation."""
        from holodeck.lib.vector_store import QueryResult
        from holodeck.tools.vectorstore_tool import VectorStoreTool

        long_content = "A" * 500  # Long content
        results = [
            QueryResult(
                content=long_content,
                score=0.9,
                source_path="/doc.md",
                chunk_index=0,
            )
        ]

        formatted = VectorStoreTool._format_results(results, "query")

        assert long_content in formatted

    def test_format_results_includes_source_path(self) -> None:
        """Test that source path is included in output."""
        from holodeck.lib.vector_store import QueryResult
        from holodeck.tools.vectorstore_tool import VectorStoreTool

        results = [
            QueryResult(
                content="Content",
                score=0.9,
                source_path="data/docs/api/endpoints.md",
                chunk_index=0,
            )
        ]

        formatted = VectorStoreTool._format_results(results, "query")

        assert "data/docs/api/endpoints.md" in formatted

    def test_format_results_ordered_by_rank(self) -> None:
        """Test that results are numbered in order (1, 2, 3...)."""
        from holodeck.lib.vector_store import QueryResult
        from holodeck.tools.vectorstore_tool import VectorStoreTool

        results = [
            QueryResult(content="First", score=0.9, source_path="/a.md", chunk_index=0),
            QueryResult(
                content="Second", score=0.8, source_path="/b.md", chunk_index=0
            ),
        ]

        formatted = VectorStoreTool._format_results(results, "query")

        # [1] should appear before [2]
        pos_1 = formatted.find("[1]")
        pos_2 = formatted.find("[2]")
        assert pos_1 < pos_2
        assert pos_1 != -1
        assert pos_2 != -1


class TestVectorStoreToolSearchValidation:
    """Additional tests for search method validation."""

    @pytest.mark.asyncio
    async def test_search_raises_error_if_not_initialized(self, tmp_path: Path) -> None:
        """Test that search raises RuntimeError if tool not initialized."""
        source_file = tmp_path / "test.md"
        source_file.write_text("# Test")

        config = VectorstoreTool(
            name="test_vectorstore",
            description="Test tool",
            source=str(source_file),
        )

        from holodeck.tools.vectorstore_tool import VectorStoreTool

        tool = VectorStoreTool(config)

        with pytest.raises(RuntimeError, match="must be initialized"):
            await tool.search("test query")

    @pytest.mark.asyncio
    async def test_search_raises_error_for_empty_query(self, tmp_path: Path) -> None:
        """Test that search raises ValueError for empty query."""
        source_file = tmp_path / "test.md"
        source_file.write_text("# Test content")

        config = VectorstoreTool(
            name="test_vectorstore",
            description="Test tool",
            source=str(source_file),
        )

        from holodeck.tools.vectorstore_tool import VectorStoreTool

        tool = VectorStoreTool(config)
        tool.is_initialized = True  # Manually set for test

        with pytest.raises(ValueError, match="cannot be empty"):
            await tool.search("")

    @pytest.mark.asyncio
    async def test_search_raises_error_for_whitespace_query(
        self, tmp_path: Path
    ) -> None:
        """Test that search raises ValueError for whitespace-only query."""
        source_file = tmp_path / "test.md"
        source_file.write_text("# Test")

        config = VectorstoreTool(
            name="test_vectorstore",
            description="Test tool",
            source=str(source_file),
        )

        from holodeck.tools.vectorstore_tool import VectorStoreTool

        tool = VectorStoreTool(config)
        tool.is_initialized = True

        with pytest.raises(ValueError, match="cannot be empty"):
            await tool.search("   ")


class TestSetEmbeddingService:
    """T021: Tests for embedding service injection."""

    def test_set_embedding_service_stores_service(self, tmp_path: Path) -> None:
        """Test that set_embedding_service stores the service."""
        source_file = tmp_path / "test.md"
        source_file.write_text("# Test")

        config = VectorstoreTool(
            name="test_vectorstore",
            description="Test tool",
            source=str(source_file),
        )

        from holodeck.tools.vectorstore_tool import VectorStoreTool

        tool = VectorStoreTool(config)

        mock_service = MagicMock()
        tool.set_embedding_service(mock_service)

        assert tool._embedding_service is mock_service

    def test_set_embedding_service_with_none(self, tmp_path: Path) -> None:
        """Test that set_embedding_service can set None."""
        source_file = tmp_path / "test.md"
        source_file.write_text("# Test")

        config = VectorstoreTool(
            name="test_vectorstore",
            description="Test tool",
            source=str(source_file),
        )

        from holodeck.tools.vectorstore_tool import VectorStoreTool

        tool = VectorStoreTool(config)
        tool.set_embedding_service(None)

        assert tool._embedding_service is None


class TestSetupCollection:
    """T022: Tests for collection setup."""

    def test_setup_collection_in_memory_default(self, tmp_path: Path) -> None:
        """Test that _setup_collection defaults to in-memory provider."""
        source_file = tmp_path / "test.md"
        source_file.write_text("# Test")

        config = VectorstoreTool(
            name="test_vectorstore",
            description="Test tool",
            source=str(source_file),
        )

        from holodeck.tools.vectorstore_tool import VectorStoreTool

        tool = VectorStoreTool(config)

        # Mock get_collection_factory to avoid actual import
        mock_collection = MagicMock()
        with patch(
            "holodeck.lib.vector_store.get_collection_factory"
        ) as mock_get_factory:
            mock_get_factory.return_value = MagicMock(return_value=mock_collection)
            tool._setup_collection("openai")

        assert tool._provider == "in-memory"
        assert tool._collection is not None

    def test_setup_collection_with_chromadb_provider(self, tmp_path: Path) -> None:
        """Test _setup_collection with ChromaDB provider."""
        from holodeck.models.tool import DatabaseConfig

        source_file = tmp_path / "test.md"
        source_file.write_text("# Test")

        config = VectorstoreTool(
            name="test_vectorstore",
            description="Test tool",
            source=str(source_file),
            database=DatabaseConfig(provider="chromadb"),
        )

        from holodeck.tools.vectorstore_tool import VectorStoreTool

        tool = VectorStoreTool(config)

        # Mock get_collection_factory to avoid actual import
        mock_collection = MagicMock()
        with patch(
            "holodeck.lib.vector_store.get_collection_factory"
        ) as mock_get_factory:
            mock_get_factory.return_value = MagicMock(return_value=mock_collection)
            tool._setup_collection("openai")

        assert tool._provider == "chromadb"
        assert tool._collection is not None

    def test_setup_collection_with_connection_string(self, tmp_path: Path) -> None:
        """Test _setup_collection with connection string."""
        from holodeck.models.tool import DatabaseConfig

        source_file = tmp_path / "test.md"
        source_file.write_text("# Test")

        config = VectorstoreTool(
            name="test_vectorstore",
            description="Test tool",
            source=str(source_file),
            database=DatabaseConfig(
                provider="chromadb",
                connection_string="http://localhost:8000",
            ),
        )

        from holodeck.tools.vectorstore_tool import VectorStoreTool

        tool = VectorStoreTool(config)

        # Mock get_collection_factory to avoid actual import
        mock_collection = MagicMock()
        with patch(
            "holodeck.lib.vector_store.get_collection_factory"
        ) as mock_get_factory:
            mock_get_factory.return_value = MagicMock(return_value=mock_collection)
            tool._setup_collection("openai")

        assert tool._provider == "chromadb"
        assert tool._collection is not None


class TestGetFileProcessor:
    """T023: Tests for file processor lazy initialization."""

    def test_get_file_processor_lazy_initialization(self, tmp_path: Path) -> None:
        """Test that _get_file_processor creates processor lazily."""
        source_file = tmp_path / "test.md"
        source_file.write_text("# Test")

        config = VectorstoreTool(
            name="test_vectorstore",
            description="Test tool",
            source=str(source_file),
        )

        from holodeck.tools.vectorstore_tool import VectorStoreTool

        tool = VectorStoreTool(config)

        # Initially None
        assert tool._file_processor is None

        # After calling _get_file_processor, it should be created
        processor = tool._get_file_processor()
        assert processor is not None
        assert tool._file_processor is not None

    def test_get_file_processor_returns_same_instance(self, tmp_path: Path) -> None:
        """Test that _get_file_processor returns the same instance."""
        source_file = tmp_path / "test.md"
        source_file.write_text("# Test")

        config = VectorstoreTool(
            name="test_vectorstore",
            description="Test tool",
            source=str(source_file),
        )

        from holodeck.tools.vectorstore_tool import VectorStoreTool

        tool = VectorStoreTool(config)

        processor1 = tool._get_file_processor()
        processor2 = tool._get_file_processor()

        assert processor1 is processor2


class TestResolveSourcePath:
    """T024: Tests for source path resolution with context variable."""

    def test_resolve_absolute_path(self, tmp_path: Path) -> None:
        """Test that absolute paths are returned as-is."""
        source_file = tmp_path / "test.md"
        source_file.write_text("# Test")

        config = VectorstoreTool(
            name="test_vectorstore",
            description="Test tool",
            source=str(source_file),  # Absolute path
        )

        from holodeck.tools.vectorstore_tool import VectorStoreTool

        tool = VectorStoreTool(config)
        resolved = tool._resolve_source_path()

        assert resolved == source_file

    def test_resolve_relative_path_with_base_dir(self, tmp_path: Path) -> None:
        """Test resolution of relative path with explicit base_dir."""
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        source_file = docs_dir / "test.md"
        source_file.write_text("# Test")

        config = VectorstoreTool(
            name="test_vectorstore",
            description="Test tool",
            source="docs/test.md",  # Relative path
        )

        from holodeck.tools.vectorstore_tool import VectorStoreTool

        tool = VectorStoreTool(config, base_dir=str(tmp_path))
        resolved = tool._resolve_source_path()

        assert resolved == source_file

    def test_resolve_relative_path_with_context_var(self, tmp_path: Path) -> None:
        """Test resolution of relative path using context variable."""
        from holodeck.config.context import agent_base_dir

        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        source_file = docs_dir / "test.md"
        source_file.write_text("# Test")

        config = VectorstoreTool(
            name="test_vectorstore",
            description="Test tool",
            source="docs/test.md",  # Relative path
        )

        from holodeck.tools.vectorstore_tool import VectorStoreTool

        # Set context variable
        token = agent_base_dir.set(str(tmp_path))
        try:
            tool = VectorStoreTool(config)  # No explicit base_dir
            resolved = tool._resolve_source_path()

            assert resolved == source_file
        finally:
            agent_base_dir.reset(token)

    def test_resolve_relative_path_fallback_to_cwd(self) -> None:
        """Test that relative path falls back to CWD when no base_dir."""
        from holodeck.config.context import agent_base_dir

        config = VectorstoreTool(
            name="test_vectorstore",
            description="Test tool",
            source="relative/path.md",  # Relative path
        )

        from holodeck.tools.vectorstore_tool import VectorStoreTool

        # Explicitly ensure context variable is None to test CWD fallback
        token = agent_base_dir.set(None)
        try:
            tool = VectorStoreTool(config)  # No base_dir, no context var
            resolved = tool._resolve_source_path()

            # Should resolve relative to CWD
            expected = Path("relative/path.md").resolve()
            assert resolved == expected
        finally:
            agent_base_dir.reset(token)


class TestProcessFile:
    """T025: Tests for file processing."""

    @pytest.mark.asyncio
    async def test_process_file_success_markdown(self, tmp_path: Path) -> None:
        """Test successful processing of a markdown file."""
        source_file = tmp_path / "test.md"
        source_file.write_text("# Test content\n\nThis is test content.")

        config = VectorstoreTool(
            name="test_vectorstore",
            description="Test tool",
            source=str(source_file),
        )

        from holodeck.lib.file_processor import SourceFile
        from holodeck.models.test_case import FileInput
        from holodeck.models.test_result import ProcessedFileInput
        from holodeck.tools.vectorstore_tool import VectorStoreTool

        tool = VectorStoreTool(config)

        # Mock the file processor
        mock_processor = MagicMock()
        mock_processor.process_file.return_value = ProcessedFileInput(
            original=FileInput(path=str(source_file), type="text"),
            markdown_content="# Test content\n\nThis is test content.",
            metadata={},
            error=None,
        )
        tool._file_processor = mock_processor

        result = await tool._process_file(source_file)

        assert result is not None
        assert isinstance(result, SourceFile)
        assert result.path == source_file
        assert result.content == "# Test content\n\nThis is test content."
        assert len(result.chunks) > 0

    @pytest.mark.asyncio
    async def test_process_file_returns_none_on_error(self, tmp_path: Path) -> None:
        """Test that _process_file returns None when processor returns error."""
        source_file = tmp_path / "test.md"
        source_file.write_text("# Test")

        config = VectorstoreTool(
            name="test_vectorstore",
            description="Test tool",
            source=str(source_file),
        )

        from holodeck.models.test_case import FileInput
        from holodeck.models.test_result import ProcessedFileInput
        from holodeck.tools.vectorstore_tool import VectorStoreTool

        tool = VectorStoreTool(config)

        # Mock processor returning error
        mock_processor = MagicMock()
        mock_processor.process_file.return_value = ProcessedFileInput(
            original=FileInput(path=str(source_file), type="text"),
            markdown_content="",
            metadata={},
            error="Processing failed",
        )
        tool._file_processor = mock_processor

        result = await tool._process_file(source_file)

        assert result is None

    @pytest.mark.asyncio
    async def test_process_file_returns_none_on_empty_content(
        self, tmp_path: Path
    ) -> None:
        """Test that _process_file returns None for empty content."""
        source_file = tmp_path / "empty.md"
        source_file.write_text("")  # Empty file

        config = VectorstoreTool(
            name="test_vectorstore",
            description="Test tool",
            source=str(source_file),
        )

        from holodeck.models.test_case import FileInput
        from holodeck.models.test_result import ProcessedFileInput
        from holodeck.tools.vectorstore_tool import VectorStoreTool

        tool = VectorStoreTool(config)

        # Mock processor returning empty content
        mock_processor = MagicMock()
        mock_processor.process_file.return_value = ProcessedFileInput(
            original=FileInput(path=str(source_file), type="text"),
            markdown_content="   ",  # Whitespace only
            metadata={},
            error=None,
        )
        tool._file_processor = mock_processor

        result = await tool._process_file(source_file)

        assert result is None

    @pytest.mark.asyncio
    async def test_process_file_handles_permission_error(self, tmp_path: Path) -> None:
        """Test that _process_file handles PermissionError gracefully."""
        source_file = tmp_path / "test.md"
        source_file.write_text("# Test")

        config = VectorstoreTool(
            name="test_vectorstore",
            description="Test tool",
            source=str(source_file),
        )

        from holodeck.tools.vectorstore_tool import VectorStoreTool

        tool = VectorStoreTool(config)

        # Create a non-existent path that will cause stat() to fail
        non_existent = tmp_path / "nonexistent.md"

        result = await tool._process_file(non_existent)

        assert result is None

    @pytest.mark.asyncio
    async def test_process_file_handles_unexpected_error(self, tmp_path: Path) -> None:
        """Test that _process_file handles unexpected errors gracefully."""
        source_file = tmp_path / "test.md"
        source_file.write_text("# Test")

        config = VectorstoreTool(
            name="test_vectorstore",
            description="Test tool",
            source=str(source_file),
        )

        from holodeck.tools.vectorstore_tool import VectorStoreTool

        tool = VectorStoreTool(config)

        # Mock processor raising exception
        mock_processor = MagicMock()
        mock_processor.process_file.side_effect = RuntimeError("Unexpected error")
        tool._file_processor = mock_processor

        result = await tool._process_file(source_file)

        assert result is None


class TestEmbedChunks:
    """T026: Tests for chunk embedding."""

    @pytest.mark.asyncio
    async def test_embed_chunks_with_embedding_service(self, tmp_path: Path) -> None:
        """Test embedding with injected embedding service."""
        source_file = tmp_path / "test.md"
        source_file.write_text("# Test")

        config = VectorstoreTool(
            name="test_vectorstore",
            description="Test tool",
            source=str(source_file),
        )

        from holodeck.tools.vectorstore_tool import VectorStoreTool

        tool = VectorStoreTool(config)

        # Mock embedding service
        mock_service = MagicMock()
        mock_service.generate_embeddings = AsyncMock(
            return_value=[[0.1] * 1536, [0.2] * 1536]
        )
        tool._embedding_service = mock_service

        chunks = ["chunk1", "chunk2"]
        embeddings = await tool._embed_chunks(chunks)

        assert len(embeddings) == 2
        assert len(embeddings[0]) == 1536
        mock_service.generate_embeddings.assert_called_once_with(chunks)

    @pytest.mark.asyncio
    async def test_embed_chunks_fallback_to_placeholder(self, tmp_path: Path) -> None:
        """Test embedding falls back to placeholder when no service."""
        source_file = tmp_path / "test.md"
        source_file.write_text("# Test")

        config = VectorstoreTool(
            name="test_vectorstore",
            description="Test tool",
            source=str(source_file),
        )

        from holodeck.tools.vectorstore_tool import VectorStoreTool

        tool = VectorStoreTool(config)
        # No embedding service set

        chunks = ["chunk1", "chunk2"]
        embeddings = await tool._embed_chunks(chunks)

        # Should return placeholder embeddings
        assert len(embeddings) == 2
        assert len(embeddings[0]) == 1536
        assert all(v == 0.0 for v in embeddings[0])

    @pytest.mark.asyncio
    async def test_embed_chunks_handles_service_error(self, tmp_path: Path) -> None:
        """Test embedding handles service errors by falling back."""
        source_file = tmp_path / "test.md"
        source_file.write_text("# Test")

        config = VectorstoreTool(
            name="test_vectorstore",
            description="Test tool",
            source=str(source_file),
        )

        from holodeck.tools.vectorstore_tool import VectorStoreTool

        tool = VectorStoreTool(config)

        # Mock embedding service that fails
        mock_service = MagicMock()
        mock_service.generate_embeddings = AsyncMock(
            side_effect=RuntimeError("Service error")
        )
        tool._embedding_service = mock_service

        chunks = ["chunk1"]
        embeddings = await tool._embed_chunks(chunks)

        # Should fall back to placeholder
        assert len(embeddings) == 1
        assert len(embeddings[0]) == 1536
        assert all(v == 0.0 for v in embeddings[0])


class TestStoreChunks:
    """T027: Tests for chunk storage."""

    @pytest.mark.asyncio
    async def test_store_chunks_creates_document_records(self, tmp_path: Path) -> None:
        """Test that _store_chunks creates DocumentRecord instances."""
        source_file = tmp_path / "test.md"
        source_file.write_text("# Test")

        config = VectorstoreTool(
            name="test_vectorstore",
            description="Test tool",
            source=str(source_file),
        )

        from holodeck.lib.file_processor import SourceFile
        from holodeck.tools.vectorstore_tool import VectorStoreTool

        tool = VectorStoreTool(config)

        # Setup mock collection
        mock_collection = MagicMock()
        mock_collection.__aenter__ = AsyncMock(return_value=mock_collection)
        mock_collection.__aexit__ = AsyncMock(return_value=None)
        mock_collection.collection_exists = AsyncMock(return_value=False)
        mock_collection.ensure_collection_exists = AsyncMock()
        mock_collection.upsert = AsyncMock()
        tool._collection = mock_collection

        # Create source file with chunks
        sf = SourceFile(
            path=source_file,
            content="Test content",
            mtime=1234567890.0,
            size_bytes=100,
            file_type=".md",
            chunks=["chunk1", "chunk2"],
        )
        embeddings = [[0.1] * 1536, [0.2] * 1536]

        count = await tool._store_chunks(sf, embeddings)

        assert count == 2
        mock_collection.upsert.assert_called_once()

        # Verify records were created correctly
        call_args = mock_collection.upsert.call_args[0][0]
        assert len(call_args) == 2
        assert call_args[0].content == "chunk1"
        assert call_args[1].content == "chunk2"

    @pytest.mark.asyncio
    async def test_store_chunks_raises_if_collection_none(self, tmp_path: Path) -> None:
        """Test that _store_chunks raises if collection not initialized."""
        source_file = tmp_path / "test.md"
        source_file.write_text("# Test")

        config = VectorstoreTool(
            name="test_vectorstore",
            description="Test tool",
            source=str(source_file),
        )

        from holodeck.lib.file_processor import SourceFile
        from holodeck.tools.vectorstore_tool import VectorStoreTool

        tool = VectorStoreTool(config)
        # _collection is None

        sf = SourceFile(
            path=source_file,
            content="Test",
            chunks=["chunk1"],
        )

        with pytest.raises(RuntimeError, match="Collection not initialized"):
            await tool._store_chunks(sf, [[0.1] * 1536])


class TestInitializeFullFlow:
    """T028: Tests for full initialization flow."""

    @pytest.mark.asyncio
    async def test_initialize_processes_all_files(self, tmp_path: Path) -> None:
        """Test that initialize processes all discovered files."""
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "doc1.md").write_text("# Doc 1")
        (docs_dir / "doc2.md").write_text("# Doc 2")

        config = VectorstoreTool(
            name="test_vectorstore",
            description="Test tool",
            source=str(docs_dir),
        )

        from holodeck.models.test_case import FileInput
        from holodeck.models.test_result import ProcessedFileInput
        from holodeck.tools.vectorstore_tool import VectorStoreTool

        tool = VectorStoreTool(config)

        # Mock file processor
        def mock_process_file(file_input: FileInput) -> ProcessedFileInput:
            return ProcessedFileInput(
                original=file_input,
                markdown_content="# Content",
                metadata={},
                error=None,
            )

        mock_processor = MagicMock()
        mock_processor.process_file.side_effect = mock_process_file
        tool._file_processor = mock_processor

        # Mock collection
        mock_collection = MagicMock()
        mock_collection.__aenter__ = AsyncMock(return_value=mock_collection)
        mock_collection.__aexit__ = AsyncMock(return_value=None)
        mock_collection.collection_exists = AsyncMock(return_value=False)
        mock_collection.ensure_collection_exists = AsyncMock()
        mock_collection.upsert = AsyncMock()
        mock_collection.get = AsyncMock(return_value=None)

        with patch(
            "holodeck.lib.vector_store.get_collection_factory"
        ) as mock_get_factory:
            mock_get_factory.return_value = MagicMock(return_value=mock_collection)

            await tool.initialize()

        assert tool.is_initialized is True
        assert tool.document_count > 0
        assert tool.last_ingest_time is not None

    @pytest.mark.asyncio
    async def test_initialize_sets_state_correctly(self, tmp_path: Path) -> None:
        """Test that initialize sets state attributes correctly."""
        source_file = tmp_path / "test.md"
        source_file.write_text("# Test content")

        config = VectorstoreTool(
            name="test_vectorstore",
            description="Test tool",
            source=str(source_file),
        )

        from holodeck.models.test_case import FileInput
        from holodeck.models.test_result import ProcessedFileInput
        from holodeck.tools.vectorstore_tool import VectorStoreTool

        tool = VectorStoreTool(config)

        # Mock file processor
        mock_processor = MagicMock()
        mock_processor.process_file.return_value = ProcessedFileInput(
            original=FileInput(path=str(source_file), type="text"),
            markdown_content="# Test content",
            metadata={},
            error=None,
        )
        tool._file_processor = mock_processor

        # Mock collection
        mock_collection = MagicMock()
        mock_collection.__aenter__ = AsyncMock(return_value=mock_collection)
        mock_collection.__aexit__ = AsyncMock(return_value=None)
        mock_collection.collection_exists = AsyncMock(return_value=False)
        mock_collection.ensure_collection_exists = AsyncMock()
        mock_collection.upsert = AsyncMock()
        mock_collection.get = AsyncMock(return_value=None)

        with patch(
            "holodeck.lib.vector_store.get_collection_factory"
        ) as mock_get_factory:
            mock_get_factory.return_value = MagicMock(return_value=mock_collection)

            await tool.initialize()

        assert tool.is_initialized is True
        assert tool.document_count >= 0
        assert tool.last_ingest_time is not None

    @pytest.mark.asyncio
    async def test_initialize_with_empty_directory(self, tmp_path: Path) -> None:
        """Test initialization with empty directory."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        config = VectorstoreTool(
            name="test_vectorstore",
            description="Test tool",
            source=str(empty_dir),
        )

        from holodeck.tools.vectorstore_tool import VectorStoreTool

        tool = VectorStoreTool(config)

        # Mock collection
        mock_collection = MagicMock()
        mock_collection.__aenter__ = AsyncMock(return_value=mock_collection)
        mock_collection.__aexit__ = AsyncMock(return_value=None)
        mock_collection.collection_exists = AsyncMock(return_value=False)
        mock_collection.ensure_collection_exists = AsyncMock()
        mock_collection.upsert = AsyncMock()

        with patch(
            "holodeck.lib.vector_store.get_collection_factory"
        ) as mock_get_factory:
            mock_get_factory.return_value = MagicMock(return_value=mock_collection)

            await tool.initialize()

        # Should still be initialized even with no files
        assert tool.is_initialized is True
        assert tool.document_count == 0

    @pytest.mark.asyncio
    async def test_initialize_skips_failed_files(self, tmp_path: Path) -> None:
        """Test that initialize skips files that fail processing."""
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "good.md").write_text("# Good")
        (docs_dir / "bad.md").write_text("# Bad")

        config = VectorstoreTool(
            name="test_vectorstore",
            description="Test tool",
            source=str(docs_dir),
        )

        from holodeck.models.test_case import FileInput
        from holodeck.models.test_result import ProcessedFileInput
        from holodeck.tools.vectorstore_tool import VectorStoreTool

        tool = VectorStoreTool(config)

        # Mock processor that fails on bad.md
        call_count = [0]

        def mock_process_file(file_input: FileInput) -> ProcessedFileInput:
            call_count[0] += 1
            path = file_input.path or ""
            if "bad.md" in path:
                return ProcessedFileInput(
                    original=file_input,
                    markdown_content="",
                    metadata={},
                    error="Processing failed",
                )
            return ProcessedFileInput(
                original=file_input,
                markdown_content="# Content",
                metadata={},
                error=None,
            )

        mock_processor = MagicMock()
        mock_processor.process_file.side_effect = mock_process_file
        tool._file_processor = mock_processor

        # Mock collection
        mock_collection = MagicMock()
        mock_collection.__aenter__ = AsyncMock(return_value=mock_collection)
        mock_collection.__aexit__ = AsyncMock(return_value=None)
        mock_collection.collection_exists = AsyncMock(return_value=False)
        mock_collection.ensure_collection_exists = AsyncMock()
        mock_collection.upsert = AsyncMock()
        mock_collection.get = AsyncMock(return_value=None)

        with patch(
            "holodeck.lib.vector_store.get_collection_factory"
        ) as mock_get_factory:
            mock_get_factory.return_value = MagicMock(return_value=mock_collection)

            await tool.initialize()

        # Should be initialized with only the successful file
        assert tool.is_initialized is True
        assert tool.document_count > 0


class TestSearchSuccessPath:
    """T029: Tests for search success path."""

    @pytest.mark.asyncio
    async def test_search_generates_query_embedding(self, tmp_path: Path) -> None:
        """Test that search generates embedding for query."""
        source_file = tmp_path / "test.md"
        source_file.write_text("# Test")

        config = VectorstoreTool(
            name="test_vectorstore",
            description="Test tool",
            source=str(source_file),
        )

        from holodeck.tools.vectorstore_tool import VectorStoreTool

        tool = VectorStoreTool(config)
        tool.is_initialized = True

        # Mock _embed_chunks to track calls
        mock_embed = AsyncMock(return_value=[[0.1] * 1536])
        tool._embed_chunks = mock_embed

        # Mock _search_documents
        mock_search = AsyncMock(return_value=[])
        tool._search_documents = mock_search

        await tool.search("test query")

        mock_embed.assert_called_once_with(["test query"])

    @pytest.mark.asyncio
    async def test_search_applies_min_similarity_filter(self, tmp_path: Path) -> None:
        """Test that search applies min_similarity_score filter."""
        source_file = tmp_path / "test.md"
        source_file.write_text("# Test")

        config = VectorstoreTool(
            name="test_vectorstore",
            description="Test tool",
            source=str(source_file),
            min_similarity_score=0.5,
        )

        from holodeck.lib.vector_store import QueryResult
        from holodeck.tools.vectorstore_tool import VectorStoreTool

        tool = VectorStoreTool(config)
        tool.is_initialized = True

        # Mock _embed_chunks
        tool._embed_chunks = AsyncMock(return_value=[[0.1] * 1536])

        # Mock _search_documents returning results with various scores
        mock_results = [
            QueryResult(content="High", score=0.9, source_path="/a.md", chunk_index=0),
            QueryResult(content="Low", score=0.3, source_path="/b.md", chunk_index=0),
            QueryResult(content="Mid", score=0.6, source_path="/c.md", chunk_index=0),
        ]
        tool._search_documents = AsyncMock(return_value=mock_results)

        result = await tool.search("test query")

        # Only results with score >= 0.5 should be included
        assert "High" in result
        assert "Mid" in result
        assert "Low" not in result

    @pytest.mark.asyncio
    async def test_search_applies_top_k_limit(self, tmp_path: Path) -> None:
        """Test that search applies top_k limit."""
        source_file = tmp_path / "test.md"
        source_file.write_text("# Test")

        config = VectorstoreTool(
            name="test_vectorstore",
            description="Test tool",
            source=str(source_file),
            top_k=2,
        )

        from holodeck.lib.vector_store import QueryResult
        from holodeck.tools.vectorstore_tool import VectorStoreTool

        tool = VectorStoreTool(config)
        tool.is_initialized = True

        # Mock _embed_chunks
        tool._embed_chunks = AsyncMock(return_value=[[0.1] * 1536])

        # Mock _search_documents returning 3 results
        mock_results = [
            QueryResult(content="First", score=0.9, source_path="/a.md", chunk_index=0),
            QueryResult(
                content="Second", score=0.8, source_path="/b.md", chunk_index=0
            ),
            QueryResult(content="Third", score=0.7, source_path="/c.md", chunk_index=0),
        ]
        tool._search_documents = AsyncMock(return_value=mock_results)

        result = await tool.search("test query")

        # Only top 2 results should be included
        assert "First" in result
        assert "Second" in result
        assert "Third" not in result

    @pytest.mark.asyncio
    async def test_search_returns_formatted_results(self, tmp_path: Path) -> None:
        """Test that search returns properly formatted results."""
        source_file = tmp_path / "test.md"
        source_file.write_text("# Test")

        config = VectorstoreTool(
            name="test_vectorstore",
            description="Test tool",
            source=str(source_file),
        )

        from holodeck.lib.vector_store import QueryResult
        from holodeck.tools.vectorstore_tool import VectorStoreTool

        tool = VectorStoreTool(config)
        tool.is_initialized = True

        tool._embed_chunks = AsyncMock(return_value=[[0.1] * 1536])

        mock_results = [
            QueryResult(
                content="Result content",
                score=0.85,
                source_path="/path/to/doc.md",
                chunk_index=0,
            )
        ]
        tool._search_documents = AsyncMock(return_value=mock_results)

        result = await tool.search("test query")

        assert "Found 1 result" in result
        assert "Score: 0.85" in result
        assert "/path/to/doc.md" in result
        assert "Result content" in result


class TestSearchDocuments:
    """T030: Tests for _search_documents method."""

    @pytest.mark.asyncio
    async def test_search_documents_uses_collection_search(
        self, tmp_path: Path
    ) -> None:
        """Test that _search_documents uses collection's search method."""
        source_file = tmp_path / "test.md"
        source_file.write_text("# Test")

        config = VectorstoreTool(
            name="test_vectorstore",
            description="Test tool",
            source=str(source_file),
            top_k=5,
        )

        from holodeck.tools.vectorstore_tool import VectorStoreTool

        tool = VectorStoreTool(config)

        # Create mock async iterator for search results
        class MockAsyncIterator:
            def __init__(self, items: list) -> None:
                self.items = items
                self.index = 0

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self.index >= len(self.items):
                    raise StopAsyncIteration
                item = self.items[self.index]
                self.index += 1
                return item

        # Mock search result with cosine similarity score (0.0 to 1.0)
        mock_record = MagicMock()
        mock_record.id = "test_id"
        mock_record.source_path = "/test.md"
        mock_record.chunk_index = 0
        mock_record.content = "Test content"
        mock_record.mtime = 1234567890.0
        mock_record.file_type = ".md"
        mock_record.file_size_bytes = 100

        mock_result = MagicMock()
        mock_result.record = mock_record
        mock_result.score = 0.9  # Cosine similarity score

        # Create mock collection
        mock_collection = MagicMock()
        mock_collection.__aenter__ = AsyncMock(return_value=mock_collection)
        mock_collection.__aexit__ = AsyncMock(return_value=None)

        # Mock search results
        mock_search_results = MagicMock()
        mock_search_results.results = MockAsyncIterator([mock_result])
        mock_collection.search = AsyncMock(return_value=mock_search_results)

        tool._collection = mock_collection

        query_embedding = [0.1] * 1536
        results = await tool._search_documents(query_embedding)

        mock_collection.search.assert_called_once_with(
            vector=query_embedding,
            top=5,
        )
        assert len(results) == 1
        assert results[0].content == "Test content"
        # Score is clamped to [0.0, 1.0] range (cosine similarity)
        assert results[0].score == 0.9

    @pytest.mark.asyncio
    async def test_search_documents_sorts_by_score(self, tmp_path: Path) -> None:
        """Test that _search_documents sorts results by cosine similarity descending."""
        source_file = tmp_path / "test.md"
        source_file.write_text("# Test")

        config = VectorstoreTool(
            name="test_vectorstore",
            description="Test tool",
            source=str(source_file),
        )

        from holodeck.tools.vectorstore_tool import VectorStoreTool

        tool = VectorStoreTool(config)

        # Create mock results with unsorted cosine similarity scores
        class MockAsyncIterator:
            def __init__(self, items: list) -> None:
                self.items = items
                self.index = 0

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self.index >= len(self.items):
                    raise StopAsyncIteration
                item = self.items[self.index]
                self.index += 1
                return item

        # Cosine similarity scores in range [0.0, 1.0]
        mock_results = []
        for i, (score, content) in enumerate(
            [(0.5, "Low"), (0.9, "High"), (0.7, "Mid")]
        ):
            mock_record = MagicMock()
            mock_record.id = f"test_{i}"
            mock_record.source_path = f"/test{i}.md"
            mock_record.chunk_index = 0
            mock_record.content = content
            mock_record.mtime = 1234567890.0
            mock_record.file_type = ".md"
            mock_record.file_size_bytes = 100

            mock_result = MagicMock()
            mock_result.record = mock_record
            mock_result.score = score  # Cosine similarity score
            mock_results.append(mock_result)

        mock_collection = MagicMock()
        mock_collection.__aenter__ = AsyncMock(return_value=mock_collection)
        mock_collection.__aexit__ = AsyncMock(return_value=None)

        mock_search_results = MagicMock()
        mock_search_results.results = MockAsyncIterator(mock_results)
        mock_collection.search = AsyncMock(return_value=mock_search_results)

        tool._collection = mock_collection

        results = await tool._search_documents([0.1] * 1536)

        # Results should be sorted by cosine similarity score descending
        assert len(results) == 3
        assert results[0].score == 0.9
        assert results[1].score == 0.7
        assert results[2].score == 0.5

    @pytest.mark.asyncio
    async def test_search_documents_clamps_scores_to_valid_range(
        self, tmp_path: Path
    ) -> None:
        """Test that _search_documents clamps scores to [0.0, 1.0] range."""
        source_file = tmp_path / "test.md"
        source_file.write_text("# Test")

        config = VectorstoreTool(
            name="test_vectorstore",
            description="Test tool",
            source=str(source_file),
        )

        from holodeck.tools.vectorstore_tool import VectorStoreTool

        tool = VectorStoreTool(config)

        class MockAsyncIterator:
            def __init__(self, items: list) -> None:
                self.items = items
                self.index = 0

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self.index >= len(self.items):
                    raise StopAsyncIteration
                item = self.items[self.index]
                self.index += 1
                return item

        # Create results with scores outside valid cosine similarity range
        mock_results = []
        for i, (raw_score, expected_score, content) in enumerate(
            [
                (1.5, 1.0, "Above max"),  # Should clamp to 1.0
                (-0.5, 0.0, "Below min"),  # Should clamp to 0.0
                (0.75, 0.75, "In range"),  # Should stay as-is
            ]
        ):
            mock_record = MagicMock()
            mock_record.id = f"test_{i}"
            mock_record.source_path = f"/test{i}.md"
            mock_record.chunk_index = 0
            mock_record.content = content
            mock_record.mtime = 1234567890.0
            mock_record.file_type = ".md"
            mock_record.file_size_bytes = 100

            mock_result = MagicMock()
            mock_result.record = mock_record
            mock_result.score = raw_score
            mock_results.append((mock_result, expected_score))

        mock_collection = MagicMock()
        mock_collection.__aenter__ = AsyncMock(return_value=mock_collection)
        mock_collection.__aexit__ = AsyncMock(return_value=None)

        mock_search_results = MagicMock()
        mock_search_results.results = MockAsyncIterator([mr for mr, _ in mock_results])
        mock_collection.search = AsyncMock(return_value=mock_search_results)

        tool._collection = mock_collection

        results = await tool._search_documents([0.1] * 1536)

        # Verify scores are clamped to valid range
        assert len(results) == 3
        # Results are sorted by score descending
        assert results[0].score == 1.0  # Was 1.5, clamped to 1.0
        assert results[1].score == 0.75  # Unchanged
        assert results[2].score == 0.0  # Was -0.5, clamped to 0.0

    @pytest.mark.asyncio
    async def test_search_documents_raises_if_collection_none(
        self, tmp_path: Path
    ) -> None:
        """Test that _search_documents raises if collection not initialized."""
        source_file = tmp_path / "test.md"
        source_file.write_text("# Test")

        config = VectorstoreTool(
            name="test_vectorstore",
            description="Test tool",
            source=str(source_file),
        )

        from holodeck.tools.vectorstore_tool import VectorStoreTool

        tool = VectorStoreTool(config)
        # _collection is None

        with pytest.raises(RuntimeError, match="Collection not initialized"):
            await tool._search_documents([0.1] * 1536)


class TestDiscoverFilesUnsupportedExtension:
    """Additional tests for unsupported file extension handling."""

    def test_discover_files_single_unsupported_file_logs_warning(
        self, tmp_path: Path
    ) -> None:
        """Test that single unsupported file logs warning and returns empty."""
        unsupported_file = tmp_path / "image.png"
        unsupported_file.write_bytes(b"PNG data")

        config = VectorstoreTool(
            name="test_vectorstore",
            description="Test tool",
            source=str(unsupported_file),
        )

        from holodeck.tools.vectorstore_tool import VectorStoreTool

        tool = VectorStoreTool(config)
        discovered = tool._discover_files()

        assert len(discovered) == 0


class TestFileModificationTimestampTracking:
    """T052: Tests for file modification timestamp tracking."""

    @pytest.mark.asyncio
    async def test_process_file_captures_mtime(self, tmp_path: Path) -> None:
        """Test that _process_file captures file mtime correctly."""
        source_file = tmp_path / "test.md"
        source_file.write_text("# Test content\n\nThis is test content.")

        # Get expected mtime
        expected_mtime = source_file.stat().st_mtime

        config = VectorstoreTool(
            name="test_vectorstore",
            description="Test tool",
            source=str(source_file),
        )

        from holodeck.models.test_case import FileInput
        from holodeck.models.test_result import ProcessedFileInput
        from holodeck.tools.vectorstore_tool import VectorStoreTool

        tool = VectorStoreTool(config)

        # Mock file processor
        mock_processor = MagicMock()
        mock_processor.process_file.return_value = ProcessedFileInput(
            original=FileInput(path=str(source_file), type="text"),
            markdown_content="# Test content\n\nThis is test content.",
            metadata={},
            error=None,
        )
        tool._file_processor = mock_processor

        result = await tool._process_file(source_file)

        assert result is not None
        # Implementation rounds mtime to 6 decimal places (microseconds) for precision
        assert result.mtime == round(expected_mtime, 6)

    @pytest.mark.asyncio
    async def test_stored_document_includes_mtime(self, tmp_path: Path) -> None:
        """Test that DocumentRecord mtime is populated from SourceFile."""
        source_file = tmp_path / "test.md"
        source_file.write_text("# Test content")

        config = VectorstoreTool(
            name="test_vectorstore",
            description="Test tool",
            source=str(source_file),
        )

        from holodeck.lib.file_processor import SourceFile
        from holodeck.tools.vectorstore_tool import VectorStoreTool

        tool = VectorStoreTool(config)

        # Setup mock collection
        mock_collection = MagicMock()
        mock_collection.__aenter__ = AsyncMock(return_value=mock_collection)
        mock_collection.__aexit__ = AsyncMock(return_value=None)
        mock_collection.collection_exists = AsyncMock(return_value=False)
        mock_collection.ensure_collection_exists = AsyncMock()
        mock_collection.upsert = AsyncMock()
        tool._collection = mock_collection

        # Create source file with specific mtime
        expected_mtime = 1234567890.5
        sf = SourceFile(
            path=source_file,
            content="Test content",
            mtime=expected_mtime,
            size_bytes=100,
            file_type=".md",
            chunks=["chunk1"],
        )
        embeddings = [[0.1] * 1536]

        await tool._store_chunks(sf, embeddings)

        # Verify DocumentRecord was created with correct mtime
        mock_collection.upsert.assert_called_once()
        call_args = mock_collection.upsert.call_args[0][0]
        assert len(call_args) == 1
        assert call_args[0].mtime == expected_mtime


class TestMtimeComparisonLogic:
    """T053: Tests for mtime comparison (needs re-ingestion vs up-to-date)."""

    @pytest.mark.asyncio
    async def test_needs_reingest_true_when_file_modified(self, tmp_path: Path) -> None:
        """Test _needs_reingest returns True when file mtime > stored mtime."""
        source_file = tmp_path / "test.md"
        source_file.write_text("# Test")

        config = VectorstoreTool(
            name="test_vectorstore",
            description="Test tool",
            source=str(source_file),
        )

        from holodeck.tools.vectorstore_tool import VectorStoreTool

        tool = VectorStoreTool(config)

        # Mock collection with older mtime
        mock_record = MagicMock()
        mock_record.mtime = source_file.stat().st_mtime - 100  # Older than current

        mock_collection = MagicMock()
        mock_collection.__aenter__ = AsyncMock(return_value=mock_collection)
        mock_collection.__aexit__ = AsyncMock(return_value=None)
        mock_collection.get = AsyncMock(return_value=mock_record)
        tool._collection = mock_collection

        result = await tool._needs_reingest(source_file)

        assert result is True

    @pytest.mark.asyncio
    async def test_needs_reingest_false_when_file_unchanged(
        self, tmp_path: Path
    ) -> None:
        """Test _needs_reingest returns False when file mtime == stored mtime."""
        source_file = tmp_path / "test.md"
        source_file.write_text("# Test")

        # Round to 6 decimal places to match implementation precision
        current_mtime = round(source_file.stat().st_mtime, 6)

        config = VectorstoreTool(
            name="test_vectorstore",
            description="Test tool",
            source=str(source_file),
        )

        from holodeck.tools.vectorstore_tool import VectorStoreTool

        tool = VectorStoreTool(config)

        # Mock collection with same mtime (rounded to match implementation)
        mock_record = MagicMock()
        mock_record.mtime = current_mtime  # Same as current (rounded)

        mock_collection = MagicMock()
        mock_collection.__aenter__ = AsyncMock(return_value=mock_collection)
        mock_collection.__aexit__ = AsyncMock(return_value=None)
        mock_collection.get = AsyncMock(return_value=mock_record)
        tool._collection = mock_collection

        result = await tool._needs_reingest(source_file)

        assert result is False

    @pytest.mark.asyncio
    async def test_needs_reingest_true_when_no_stored_record(
        self, tmp_path: Path
    ) -> None:
        """Test _needs_reingest returns True when file has no stored records."""
        source_file = tmp_path / "test.md"
        source_file.write_text("# Test")

        config = VectorstoreTool(
            name="test_vectorstore",
            description="Test tool",
            source=str(source_file),
        )

        from holodeck.tools.vectorstore_tool import VectorStoreTool

        tool = VectorStoreTool(config)

        # Mock collection returning None (no record found)
        mock_collection = MagicMock()
        mock_collection.__aenter__ = AsyncMock(return_value=mock_collection)
        mock_collection.__aexit__ = AsyncMock(return_value=None)
        mock_collection.get = AsyncMock(return_value=None)
        tool._collection = mock_collection

        result = await tool._needs_reingest(source_file)

        assert result is True

    @pytest.mark.asyncio
    async def test_needs_reingest_true_when_collection_none(
        self, tmp_path: Path
    ) -> None:
        """Test _needs_reingest returns True when collection is not initialized."""
        source_file = tmp_path / "test.md"
        source_file.write_text("# Test")

        config = VectorstoreTool(
            name="test_vectorstore",
            description="Test tool",
            source=str(source_file),
        )

        from holodeck.tools.vectorstore_tool import VectorStoreTool

        tool = VectorStoreTool(config)
        # _collection is None by default

        result = await tool._needs_reingest(source_file)

        assert result is True


class TestForceIngestFlag:
    """T054: Tests for --force-ingest flag bypassing mtime checks."""

    @pytest.mark.asyncio
    async def test_force_ingest_processes_unchanged_files(self, tmp_path: Path) -> None:
        """Test force_ingest=True processes files regardless of mtime."""
        source_file = tmp_path / "test.md"
        source_file.write_text("# Test content")

        config = VectorstoreTool(
            name="test_vectorstore",
            description="Test tool",
            source=str(source_file),
        )

        from holodeck.models.test_case import FileInput
        from holodeck.models.test_result import ProcessedFileInput
        from holodeck.tools.vectorstore_tool import VectorStoreTool

        tool = VectorStoreTool(config)

        # Mock _needs_reingest to return False (file unchanged)
        tool._needs_reingest = AsyncMock(return_value=False)
        tool._delete_file_records = AsyncMock(return_value=0)

        # Mock file processor
        mock_processor = MagicMock()
        mock_processor.process_file.return_value = ProcessedFileInput(
            original=FileInput(path=str(source_file), type="text"),
            markdown_content="# Test content",
            metadata={},
            error=None,
        )
        tool._file_processor = mock_processor

        # Mock collection
        mock_collection = MagicMock()
        mock_collection.__aenter__ = AsyncMock(return_value=mock_collection)
        mock_collection.__aexit__ = AsyncMock(return_value=None)
        mock_collection.collection_exists = AsyncMock(return_value=False)
        mock_collection.ensure_collection_exists = AsyncMock()
        mock_collection.upsert = AsyncMock()

        with patch(
            "holodeck.lib.vector_store.get_collection_factory"
        ) as mock_get_factory:
            mock_get_factory.return_value = MagicMock(return_value=mock_collection)

            await tool.initialize(force_ingest=True)

        # Should have processed the file despite _needs_reingest returning False
        assert tool.is_initialized is True
        assert tool.document_count > 0
        # _needs_reingest should NOT have been called when force_ingest=True
        # (or if called, its result should be ignored)

    @pytest.mark.asyncio
    async def test_default_skips_unchanged_files(self, tmp_path: Path) -> None:
        """Test force_ingest=False (default) skips unchanged files."""
        source_file = tmp_path / "test.md"
        source_file.write_text("# Test content")

        config = VectorstoreTool(
            name="test_vectorstore",
            description="Test tool",
            source=str(source_file),
        )

        from holodeck.tools.vectorstore_tool import VectorStoreTool

        tool = VectorStoreTool(config)

        # Track if _process_file was called
        process_file_called = False
        original_process_file = tool._process_file

        async def tracking_process_file(file_path):
            nonlocal process_file_called
            process_file_called = True
            return await original_process_file(file_path)

        tool._process_file = tracking_process_file

        # Mock _needs_reingest to return False (file unchanged)
        tool._needs_reingest = AsyncMock(return_value=False)

        # Mock collection
        mock_collection = MagicMock()
        mock_collection.__aenter__ = AsyncMock(return_value=mock_collection)
        mock_collection.__aexit__ = AsyncMock(return_value=None)
        mock_collection.collection_exists = AsyncMock(return_value=False)
        mock_collection.ensure_collection_exists = AsyncMock()
        mock_collection.upsert = AsyncMock()

        with patch(
            "holodeck.lib.vector_store.get_collection_factory"
        ) as mock_get_factory:
            mock_get_factory.return_value = MagicMock(return_value=mock_collection)

            await tool.initialize(force_ingest=False)

        # Should NOT have processed the file
        assert process_file_called is False
        assert tool.is_initialized is True
        assert tool.document_count == 0

    @pytest.mark.asyncio
    async def test_force_ingest_deletes_old_records_before_reingest(
        self, tmp_path: Path
    ) -> None:
        """Test that old records are deleted before re-ingestion with force_ingest."""
        source_file = tmp_path / "test.md"
        source_file.write_text("# Test content")

        config = VectorstoreTool(
            name="test_vectorstore",
            description="Test tool",
            source=str(source_file),
        )

        from holodeck.models.test_case import FileInput
        from holodeck.models.test_result import ProcessedFileInput
        from holodeck.tools.vectorstore_tool import VectorStoreTool

        tool = VectorStoreTool(config)

        # Track _delete_file_records calls
        delete_called = False

        async def mock_delete_file_records(file_path):
            nonlocal delete_called
            delete_called = True
            return 2  # Simulate deleting 2 records

        tool._delete_file_records = mock_delete_file_records

        # Mock file processor
        mock_processor = MagicMock()
        mock_processor.process_file.return_value = ProcessedFileInput(
            original=FileInput(path=str(source_file), type="text"),
            markdown_content="# Test content",
            metadata={},
            error=None,
        )
        tool._file_processor = mock_processor

        # Mock collection
        mock_collection = MagicMock()
        mock_collection.__aenter__ = AsyncMock(return_value=mock_collection)
        mock_collection.__aexit__ = AsyncMock(return_value=None)
        mock_collection.collection_exists = AsyncMock(return_value=False)
        mock_collection.ensure_collection_exists = AsyncMock()
        mock_collection.upsert = AsyncMock()

        with patch(
            "holodeck.lib.vector_store.get_collection_factory"
        ) as mock_get_factory:
            mock_get_factory.return_value = MagicMock(return_value=mock_collection)

            await tool.initialize(force_ingest=True)

        # Should have called _delete_file_records
        assert delete_called is True


class TestDeleteFileRecords:
    """T060: Tests for _delete_file_records method."""

    @pytest.mark.asyncio
    async def test_delete_file_records_deletes_all_chunks(self, tmp_path: Path) -> None:
        """Test that _delete_file_records deletes all chunks for a file."""
        source_file = tmp_path / "test.md"
        source_file.write_text("# Test")

        config = VectorstoreTool(
            name="test_vectorstore",
            description="Test tool",
            source=str(source_file),
        )

        from holodeck.tools.vectorstore_tool import VectorStoreTool

        tool = VectorStoreTool(config)

        # Create mock records for chunks 0, 1, 2
        mock_records = {
            f"{source_file}_chunk_0": MagicMock(),
            f"{source_file}_chunk_1": MagicMock(),
            f"{source_file}_chunk_2": MagicMock(),
        }

        async def mock_get(record_id):
            return mock_records.get(record_id)

        deleted_ids = []

        async def mock_delete(record_id):
            deleted_ids.append(record_id)

        mock_collection = MagicMock()
        mock_collection.__aenter__ = AsyncMock(return_value=mock_collection)
        mock_collection.__aexit__ = AsyncMock(return_value=None)
        mock_collection.get = mock_get
        mock_collection.delete = mock_delete
        tool._collection = mock_collection

        count = await tool._delete_file_records(source_file)

        assert count == 3
        assert len(deleted_ids) == 3

    @pytest.mark.asyncio
    async def test_delete_file_records_returns_zero_when_no_collection(
        self, tmp_path: Path
    ) -> None:
        """Test that _delete_file_records returns 0 when collection is None."""
        source_file = tmp_path / "test.md"
        source_file.write_text("# Test")

        config = VectorstoreTool(
            name="test_vectorstore",
            description="Test tool",
            source=str(source_file),
        )

        from holodeck.tools.vectorstore_tool import VectorStoreTool

        tool = VectorStoreTool(config)
        # _collection is None

        count = await tool._delete_file_records(source_file)

        assert count == 0

    @pytest.mark.asyncio
    async def test_delete_file_records_handles_no_existing_records(
        self, tmp_path: Path
    ) -> None:
        """Test _delete_file_records when file has no records in store."""
        source_file = tmp_path / "test.md"
        source_file.write_text("# Test")

        config = VectorstoreTool(
            name="test_vectorstore",
            description="Test tool",
            source=str(source_file),
        )

        from holodeck.tools.vectorstore_tool import VectorStoreTool

        tool = VectorStoreTool(config)

        # Mock collection returning None for all gets
        mock_collection = MagicMock()
        mock_collection.__aenter__ = AsyncMock(return_value=mock_collection)
        mock_collection.__aexit__ = AsyncMock(return_value=None)
        mock_collection.get = AsyncMock(return_value=None)
        tool._collection = mock_collection

        count = await tool._delete_file_records(source_file)

        assert count == 0
