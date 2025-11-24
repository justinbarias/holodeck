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

Note: This test module requires mocking semantic_kernel modules because the full
semantic_kernel library is not available in the test environment.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

# Mock semantic_kernel modules BEFORE importing holodeck modules
# This prevents import errors from semantic_kernel dependencies
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

# Set up specific mock attributes needed by the modules
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
mock_memory.RedisHashsetCollection = MagicMock()
mock_memory.RedisJsonCollection = MagicMock()
mock_memory.SqlServerCollection = MagicMock()
mock_memory.WeaviateCollection = MagicMock()

mock_vector = sys.modules["semantic_kernel.data.vector"]
mock_vector.VectorStoreField = MagicMock()
mock_vector.vectorstoremodel = lambda **kwargs: lambda cls: cls

mock_text = sys.modules["semantic_kernel.text"]
mock_text.split_plaintext_paragraph = MagicMock(
    side_effect=lambda lines, max_tokens: lines if isinstance(lines, list) else [lines]
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
