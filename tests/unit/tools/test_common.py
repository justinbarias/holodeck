"""Unit tests for holodeck.tools.common utilities.

Tests for shared utility functions used by VectorStoreTool and
HierarchicalDocumentTool.

Test IDs:
- TC001: SUPPORTED_EXTENSIONS contains expected extensions
- TC002: FILE_TYPE_MAPPING maps correctly
- TC003: get_file_type returns correct type for each extension
- TC004: get_file_type returns "text" for unknown extensions
- TC005: resolve_source_path handles absolute paths
- TC006: resolve_source_path handles relative paths with base_dir
- TC007: resolve_source_path uses context var when no base_dir
- TC008: discover_files finds single file
- TC009: discover_files finds files recursively in directory
- TC010: discover_files filters by extension
- TC011: discover_files returns sorted list
- TC012: generate_placeholder_embeddings creates correct shape
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from holodeck.tools.common import (
    FILE_TYPE_MAPPING,
    SUPPORTED_EXTENSIONS,
    discover_files,
    generate_placeholder_embeddings,
    get_file_type,
    resolve_source_path,
)


class TestSupportedExtensions:
    """Tests for SUPPORTED_EXTENSIONS constant."""

    def test_contains_expected_extensions(self) -> None:
        """TC001: SUPPORTED_EXTENSIONS contains all expected extensions."""
        expected = {".txt", ".md", ".pdf", ".csv", ".json"}
        assert expected == SUPPORTED_EXTENSIONS

    def test_is_frozenset(self) -> None:
        """SUPPORTED_EXTENSIONS is immutable."""
        assert isinstance(SUPPORTED_EXTENSIONS, frozenset)


class TestFileTypeMapping:
    """Tests for FILE_TYPE_MAPPING constant."""

    def test_mapping_contains_expected_keys(self) -> None:
        """TC002: FILE_TYPE_MAPPING maps correctly."""
        expected = {
            ".txt": "text",
            ".md": "text",
            ".pdf": "pdf",
            ".csv": "csv",
            ".json": "text",
        }
        assert expected == FILE_TYPE_MAPPING


class TestGetFileType:
    """Tests for get_file_type function."""

    @pytest.mark.parametrize(
        "path,expected",
        [
            ("document.txt", "text"),
            ("readme.md", "text"),
            ("report.pdf", "pdf"),
            ("data.csv", "csv"),
            ("config.json", "text"),
        ],
    )
    def test_returns_correct_type(self, path: str, expected: str) -> None:
        """TC003: get_file_type returns correct type for each extension."""
        assert get_file_type(path) == expected

    @pytest.mark.parametrize(
        "path,expected",
        [
            (Path("document.txt"), "text"),
            (Path("data/report.pdf"), "pdf"),
        ],
    )
    def test_accepts_path_objects(self, path: Path, expected: str) -> None:
        """get_file_type accepts Path objects."""
        assert get_file_type(path) == expected

    def test_returns_text_for_unknown_extension(self) -> None:
        """TC004: get_file_type returns 'text' for unknown extensions."""
        assert get_file_type("unknown.xyz") == "text"
        assert get_file_type("file.docx") == "text"
        assert get_file_type("file") == "text"

    def test_case_insensitive(self) -> None:
        """get_file_type is case-insensitive for extensions."""
        assert get_file_type("FILE.PDF") == "pdf"
        assert get_file_type("FILE.Pdf") == "pdf"


class TestResolveSourcePath:
    """Tests for resolve_source_path function."""

    def test_returns_absolute_path_unchanged(self) -> None:
        """TC005: resolve_source_path handles absolute paths."""
        path = "/absolute/path/to/file.txt"
        result = resolve_source_path(path)
        assert result == Path(path)
        assert result.is_absolute()

    def test_resolves_relative_with_base_dir(self) -> None:
        """TC006: resolve_source_path handles relative paths with base_dir."""
        result = resolve_source_path("relative/file.txt", "/base/dir")
        expected = Path("/base/dir/relative/file.txt").resolve()
        assert result == expected

    def test_uses_context_var_when_no_base_dir(self) -> None:
        """TC007: resolve_source_path uses context var when no base_dir."""
        mock_context = MagicMock()
        mock_context.get.return_value = "/context/base"

        with patch(
            "holodeck.config.context.agent_base_dir",
            mock_context,
        ):
            result = resolve_source_path("relative/file.txt")
            expected = Path("/context/base/relative/file.txt").resolve()
            assert result == expected

    def test_falls_back_to_cwd_when_no_context(self) -> None:
        """resolve_source_path falls back to cwd when no context var."""
        mock_context = MagicMock()
        mock_context.get.return_value = None

        with patch(
            "holodeck.config.context.agent_base_dir",
            mock_context,
        ):
            result = resolve_source_path("relative/file.txt")
            expected = Path("relative/file.txt").resolve()
            assert result == expected


class TestDiscoverFiles:
    """Tests for discover_files function."""

    def test_discovers_single_file(self, tmp_path: Path) -> None:
        """TC008: discover_files finds single file."""
        file = tmp_path / "document.txt"
        file.write_text("content")

        result = discover_files(file)

        assert result == [file]

    def test_discovers_files_recursively(self, tmp_path: Path) -> None:
        """TC009: discover_files finds files recursively in directory."""
        # Create nested structure
        (tmp_path / "file1.txt").write_text("content")
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (subdir / "file2.md").write_text("content")
        nested = subdir / "nested"
        nested.mkdir()
        (nested / "file3.pdf").write_text("content")

        result = discover_files(tmp_path)

        assert len(result) == 3
        assert tmp_path / "file1.txt" in result
        assert subdir / "file2.md" in result
        assert nested / "file3.pdf" in result

    def test_filters_by_extension(self, tmp_path: Path) -> None:
        """TC010: discover_files filters by extension."""
        (tmp_path / "included.txt").write_text("content")
        (tmp_path / "excluded.xyz").write_text("content")
        (tmp_path / "also_included.md").write_text("content")

        result = discover_files(tmp_path)

        assert len(result) == 2
        assert all(f.suffix in SUPPORTED_EXTENSIONS for f in result)

    def test_returns_sorted_list(self, tmp_path: Path) -> None:
        """TC011: discover_files returns sorted list."""
        (tmp_path / "z_file.txt").write_text("content")
        (tmp_path / "a_file.txt").write_text("content")
        (tmp_path / "m_file.txt").write_text("content")

        result = discover_files(tmp_path)

        assert result == sorted(result)

    def test_returns_empty_for_unsupported_single_file(self, tmp_path: Path) -> None:
        """discover_files returns empty list for unsupported single file."""
        file = tmp_path / "document.xyz"
        file.write_text("content")

        result = discover_files(file)

        assert result == []

    def test_returns_empty_for_nonexistent_path(self, tmp_path: Path) -> None:
        """discover_files returns empty list for nonexistent path."""
        nonexistent = tmp_path / "does_not_exist"

        result = discover_files(nonexistent)

        assert result == []

    def test_accepts_custom_extensions(self, tmp_path: Path) -> None:
        """discover_files accepts custom extension set."""
        (tmp_path / "file.abc").write_text("content")
        (tmp_path / "file.xyz").write_text("content")

        result = discover_files(tmp_path, extensions=frozenset({".abc"}))

        assert len(result) == 1
        assert result[0].suffix == ".abc"


class TestGeneratePlaceholderEmbeddings:
    """Tests for generate_placeholder_embeddings function."""

    def test_creates_correct_count(self) -> None:
        """TC012: generate_placeholder_embeddings creates correct shape."""
        result = generate_placeholder_embeddings(5)
        assert len(result) == 5

    def test_creates_correct_dimensions(self) -> None:
        """generate_placeholder_embeddings creates correct dimensions."""
        result = generate_placeholder_embeddings(3, dimensions=768)
        assert all(len(emb) == 768 for emb in result)

    def test_uses_default_dimensions(self) -> None:
        """generate_placeholder_embeddings uses default 1536 dimensions."""
        result = generate_placeholder_embeddings(1)
        assert len(result[0]) == 1536

    def test_creates_zero_vectors(self) -> None:
        """generate_placeholder_embeddings creates zero-valued vectors."""
        result = generate_placeholder_embeddings(2, dimensions=3)
        assert result == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]

    def test_handles_zero_count(self) -> None:
        """generate_placeholder_embeddings handles zero count."""
        result = generate_placeholder_embeddings(0)
        assert result == []
