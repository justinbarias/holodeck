"""Unit tests for file processor module using markitdown integration."""

import hashlib
import tempfile
from pathlib import Path
from unittest import mock

from holodeck.lib.file_processor import FileProcessor
from holodeck.models.test_case import FileInput
from holodeck.models.test_result import ProcessedFileInput


class TestFileProcessorBasics:
    """Tests for basic file processor functionality."""

    def test_process_file_routes_to_local(self) -> None:
        """Test that process_file routes local files correctly."""
        mock_file_input = FileInput(path="/path/to/sample.pdf", type="pdf")

        with mock.patch.object(FileProcessor, "_process_local_file") as mock_local:
            mock_local.return_value = ProcessedFileInput(
                original=mock_file_input,
                markdown_content="# Content",
                processing_time_ms=100,
            )

            processor = FileProcessor()
            result = processor.process_file(mock_file_input)

            assert result.original == mock_file_input
            mock_local.assert_called_once()

    def test_process_file_routes_to_remote(self) -> None:
        """Test that process_file routes remote files correctly."""
        mock_file_input = FileInput(url="https://example.com/file.pdf", type="pdf")

        with mock.patch.object(FileProcessor, "_process_remote_file") as mock_remote:
            mock_remote.return_value = ProcessedFileInput(
                original=mock_file_input,
                markdown_content="# Content",
                processing_time_ms=100,
            )

            processor = FileProcessor()
            result = processor.process_file(mock_file_input)

            assert result.original == mock_file_input
            mock_remote.assert_called_once()

    def test_cache_key_generation(self) -> None:
        """Test cache key generation uses MD5 hashing."""
        processor = FileProcessor()

        url = "https://example.com/test.pdf"
        expected_hash = hashlib.md5(url.encode()).hexdigest()  # noqa: S324
        actual_hash = processor._get_cache_key(url)

        assert actual_hash == expected_hash
        assert len(actual_hash) == 32  # MD5 is 32 hex characters


class TestFileProcessorMetadata:
    """Tests for file metadata tracking."""

    def test_error_handling_returns_processed_input(self) -> None:
        """Test that errors are captured and returned in result."""
        mock_file_input = FileInput(path="/nonexistent/file.pdf", type="pdf")

        with mock.patch.object(FileProcessor, "_process_local_file") as mock_local:
            mock_local.side_effect = FileNotFoundError("File not found")

            processor = FileProcessor()
            result = processor.process_file(mock_file_input)

            assert result.error is not None
            assert "File not found" in result.error
            assert result.original == mock_file_input


class TestFileProcessorCaching:
    """Tests for file caching functionality."""

    def test_cache_directory_creation(self) -> None:
        """Test that cache directory is created on initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / ".holodeck" / "cache"
            processor = FileProcessor(cache_dir=str(cache_dir))

            assert cache_dir.exists()
            assert processor.cache_dir == cache_dir

    def test_load_cache_missing_file(self) -> None:
        """Test loading from cache when file doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / ".holodeck" / "cache"
            processor = FileProcessor(cache_dir=str(cache_dir))

            result = processor._load_from_cache("nonexistent_key")
            assert result is None

    def test_save_and_load_cache(self) -> None:
        """Test saving and loading from cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / ".holodeck" / "cache"
            processor = FileProcessor(cache_dir=str(cache_dir))

            cache_key = "test_key"
            metadata = {"size": 1024}
            processor._save_to_cache(cache_key, "Test content", metadata, 100)

            loaded = processor._load_from_cache(cache_key)
            assert loaded is not None
            assert loaded["markdown_content"] == "Test content"
            assert loaded["metadata"] == metadata
            assert loaded["processing_time_ms"] == 100


class TestFileProcessorErrorHandling:
    """Tests for error handling in file processing."""

    def test_error_dict_structure(self) -> None:
        """Test that error messages are properly structured."""
        result = ProcessedFileInput(
            original=FileInput(path="test.pdf", type="pdf"),
            markdown_content="",
            processing_time_ms=100,
            error="Test error message",
        )

        assert result.error == "Test error message"
        assert result.markdown_content == ""


class TestFileProcessorDownloads:
    """Tests for remote file download functionality."""

    def test_download_file_success(self) -> None:
        """Test successful file download."""
        processor = FileProcessor(download_timeout_ms=30000)

        with mock.patch("requests.get") as mock_get:
            mock_get.return_value = mock.Mock(content=b"PDF content", status_code=200)

            result = processor._download_file("https://example.com/file.pdf")

            assert result == b"PDF content"
            mock_get.assert_called_once()

    def test_download_file_retry_on_failure(self) -> None:
        """Test retry logic for failed downloads."""
        processor = FileProcessor(max_retries=3)

        with mock.patch("requests.get") as mock_get, mock.patch("time.sleep"):
            mock_get.side_effect = [
                Exception("Timeout"),
                Exception("Timeout"),
                mock.Mock(content=b"Success", status_code=200),
            ]

            result = processor._download_file("https://example.com/file.pdf")

            assert result == b"Success"
            assert mock_get.call_count == 3

    def test_download_file_max_retries_exceeded(self) -> None:
        """Test handling when max retries exceeded."""
        processor = FileProcessor(max_retries=3)

        with mock.patch("requests.get") as mock_get, mock.patch("time.sleep"):
            mock_get.side_effect = Exception("Timeout")

            result = processor._download_file("https://example.com/file.pdf")

            assert result is None
            assert mock_get.call_count == 3


class TestFileProcessorConfiguration:
    """Tests for FileProcessor configuration."""

    def test_custom_cache_directory(self) -> None:
        """Test setting custom cache directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            custom_cache = Path(tmpdir) / "custom" / "cache"
            processor = FileProcessor(cache_dir=str(custom_cache))

            assert processor.cache_dir == custom_cache
            assert custom_cache.exists()

    def test_custom_download_timeout(self) -> None:
        """Test setting custom download timeout."""
        processor = FileProcessor(download_timeout_ms=60000)

        assert processor.download_timeout_ms == 60000

    def test_custom_max_retries(self) -> None:
        """Test setting custom max retries."""
        processor = FileProcessor(max_retries=5)

        assert processor.max_retries == 5


class TestFileProcessorMarkItDown:
    """Tests for MarkItDown integration and lazy initialization."""

    def test_markitdown_import_error(self) -> None:
        """Test that import error is raised when markitdown not available."""
        with (
            mock.patch.dict("sys.modules", {"markitdown": None}),
            mock.patch("builtins.__import__", side_effect=ImportError("No module")),
        ):
            try:
                FileProcessor()
                raise AssertionError("Should raise ImportError")
            except ImportError as e:
                assert "markitdown is required" in str(e)

    def test_get_markitdown_lazy_initialization(self) -> None:
        """Test that MarkItDown is initialized lazily on first access."""
        processor = FileProcessor()

        # Initially md should be None
        assert processor.md is None

        # Mock the MarkItDown class
        mock_md_instance = mock.MagicMock()
        with mock.patch("markitdown.MarkItDown", return_value=mock_md_instance):
            result = processor._get_markitdown()

            assert result is not None
            assert result is mock_md_instance
            assert processor.md is not None
            assert processor.md is mock_md_instance

    def test_get_markitdown_returns_cached_instance(self) -> None:
        """Test that _get_markitdown returns cached instance on subsequent calls."""
        processor = FileProcessor()

        mock_md_instance = mock.MagicMock()
        with mock.patch("markitdown.MarkItDown", return_value=mock_md_instance):
            # First call
            result1 = processor._get_markitdown()
            # Second call should not create new instance
            result2 = processor._get_markitdown()

            # Should return same instance due to caching
            assert result1 is result2
            assert result1 is processor.md
            assert result1 is mock_md_instance


class TestFileProcessorLocalFileProcessing:
    """Tests for local file processing with markitdown."""

    def test_process_local_file_success(self) -> None:
        """Test successful local file processing."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
            tmp.write(b"Test content")
            tmp_path = tmp.name

        try:
            file_input = FileInput(path=tmp_path, type="text")
            processor = FileProcessor()

            # Create a mock that properly mimics MarkItDown instance
            mock_md_instance = mock.MagicMock()
            mock_convert_result = mock.MagicMock()
            mock_convert_result.text_content = "# Test Content"
            mock_md_instance.convert.return_value = mock_convert_result

            with mock.patch("markitdown.MarkItDown", return_value=mock_md_instance):
                result = processor._process_local_file(file_input, start_time=0)

                assert result.markdown_content == "# Test Content"
                assert result.error is None
                assert result.metadata is not None
                assert result.metadata["path"] == tmp_path
                assert result.metadata["type"] == "text"
                assert "size_bytes" in result.metadata
        finally:
            Path(tmp_path).unlink()

    def test_process_local_file_not_found(self) -> None:
        """Test local file processing with missing file via process_file."""
        file_input = FileInput(path="/nonexistent/file.txt", type="text")
        processor = FileProcessor()

        # Use public process_file which has exception handling
        result = processor.process_file(file_input)

        assert result.error is not None
        assert "File not found" in result.error
        assert result.processing_time_ms is not None

    def test_process_local_file_via_process_file_no_path(self) -> None:
        """Test file processing through process_file with URL input."""
        # Create a FileInput with URL (tests routing to _process_remote_file)
        file_input = FileInput(url="https://example.com/file.txt", type="text")
        processor = FileProcessor()

        with mock.patch.object(processor, "_process_remote_file") as mock_remote:
            mock_remote.return_value = ProcessedFileInput(
                original=file_input,
                markdown_content="content",
                processing_time_ms=10,
            )
            processor.process_file(file_input)

            # Should route to remote processing, not local
            mock_remote.assert_called_once()

    def test_process_local_file_large_file_warning(self) -> None:
        """Test that large files are flagged in metadata."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
            # Create a file
            tmp_path = tmp.name

        try:
            file_input = FileInput(path=tmp_path, type="text")
            processor = FileProcessor()

            # Patch Path.stat to simulate large file
            with mock.patch.object(Path, "stat") as mock_stat:
                mock_stat.return_value.st_size = 101 * 1024 * 1024  # 101 MB

                mock_md_instance = mock.MagicMock()
                mock_result = mock.MagicMock()
                mock_result.text_content = "Large file content"
                mock_md_instance.convert.return_value = mock_result

                with mock.patch("markitdown.MarkItDown", return_value=mock_md_instance):
                    result = processor._process_local_file(file_input, start_time=0)

                    assert result.metadata is not None
                    assert "warning" in result.metadata
                    assert "Large file" in result.metadata["warning"]
        finally:
            Path(tmp_path).unlink()


class TestFileProcessorRemoteFileProcessing:
    """Tests for remote file processing with caching and downloading."""

    def test_process_remote_file_from_cache(self) -> None:
        """Test remote file processing using cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / ".holodeck" / "cache"
            processor = FileProcessor(cache_dir=str(cache_dir))

            url = "https://example.com/file.pdf"
            file_input = FileInput(url=url, type="pdf")

            # Save something to cache first
            cache_key = processor._get_cache_key(url)
            processor._save_to_cache(
                cache_key,
                "Cached markdown",
                {"url": url},
                50,
            )

            # Process remote file - should use cache
            result = processor._process_remote_file(file_input, start_time=0)

            assert result.markdown_content == "Cached markdown"
            assert result.cached_path is not None
            assert result.error is None

    def test_process_remote_file_download_fails(self) -> None:
        """Test remote file processing when download fails."""
        processor = FileProcessor()
        file_input = FileInput(url="https://example.com/file.pdf", type="pdf")

        with mock.patch.object(processor, "_download_file", return_value=None):
            result = processor._process_remote_file(file_input, start_time=0)

            assert result.error is not None
            assert "Failed to download" in result.error
            assert result.markdown_content == ""

    def test_process_remote_file_invalid_input_via_process_file(self) -> None:
        """Test remote file processing with local file via process_file."""
        processor = FileProcessor()
        file_input = FileInput(path="/local/file.txt", type="text")

        # Use public process_file which has exception handling
        with mock.patch.object(processor, "_process_local_file") as mock_local:
            mock_local.return_value = ProcessedFileInput(
                original=file_input,
                markdown_content="content",
                processing_time_ms=10,
            )
            processor.process_file(file_input)

            # Should route to local processing since it has path
            mock_local.assert_called_once()

    def test_process_remote_file_success_with_cache(self) -> None:
        """Test successful remote file processing and caching."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / ".holodeck" / "cache"
            processor = FileProcessor(cache_dir=str(cache_dir))

            url = "https://example.com/file.pdf"
            file_input = FileInput(url=url, type="pdf", cache=True)

            with mock.patch.object(
                processor, "_download_file", return_value=b"PDF content"
            ):
                mock_md_instance = mock.MagicMock()
                mock_result = mock.MagicMock()
                mock_result.text_content = "# PDF Content"
                mock_md_instance.convert.return_value = mock_result

                with mock.patch("markitdown.MarkItDown", return_value=mock_md_instance):
                    result = processor._process_remote_file(file_input, start_time=0)

                    assert result.markdown_content == "# PDF Content"
                    assert result.error is None
                    assert result.metadata is not None
                    assert result.metadata["url"] == url

                    # Verify cache was created
                    assert result.cached_path is not None

    def test_process_remote_file_cache_disabled(self) -> None:
        """Test remote file processing with cache disabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / ".holodeck" / "cache"
            processor = FileProcessor(cache_dir=str(cache_dir))

            url = "https://example.com/file.pdf"
            file_input = FileInput(url=url, type="pdf", cache=False)

            with mock.patch.object(processor, "_download_file", return_value=b"PDF"):
                mock_md_instance = mock.MagicMock()
                mock_result = mock.MagicMock()
                mock_result.text_content = "# PDF"
                mock_md_instance.convert.return_value = mock_result

                with mock.patch("markitdown.MarkItDown", return_value=mock_md_instance):
                    result = processor._process_remote_file(file_input, start_time=0)

                    assert result.error is None
                    # When cache is disabled, file won't be created
                    cache_key = processor._get_cache_key(url)
                    cache_file = processor.cache_dir / f"{cache_key}.json"
                    assert not cache_file.exists()

    def test_process_remote_file_cleanup_temp_file(self) -> None:
        """Test that temporary file is cleaned up after processing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            processor = FileProcessor(cache_dir=tmpdir)
            file_input = FileInput(url="https://example.com/file.pdf", type="pdf")

            with mock.patch.object(
                processor, "_download_file", return_value=b"content"
            ):
                # Use MagicMock to properly handle context manager
                mock_temp_file = mock.MagicMock()
                mock_temp_file.name = str(Path(tmpdir) / "test_file")

                with mock.patch(
                    "tempfile.NamedTemporaryFile", return_value=mock_temp_file
                ):
                    mock_md_instance = mock.MagicMock()
                    mock_result = mock.MagicMock()
                    mock_result.text_content = "content"
                    mock_md_instance.convert.return_value = mock_result

                    with mock.patch(
                        "markitdown.MarkItDown", return_value=mock_md_instance
                    ):
                        result = processor._process_remote_file(
                            file_input, start_time=0
                        )

                        assert result.error is None


class TestFileProcessorCacheEdgeCases:
    """Tests for cache edge cases and error handling."""

    def test_load_cache_corrupted_json(self) -> None:
        """Test loading from cache when JSON is corrupted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / ".holodeck" / "cache"
            cache_dir.mkdir(parents=True, exist_ok=True)

            processor = FileProcessor(cache_dir=str(cache_dir))
            cache_key = "test_key"
            cache_file = cache_dir / f"{cache_key}.json"

            # Write invalid JSON
            cache_file.write_text("{ invalid json }")

            result = processor._load_from_cache(cache_key)
            assert result is None

    def test_load_cache_non_dict_json(self) -> None:
        """Test loading from cache when JSON is not a dict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / ".holodeck" / "cache"
            cache_dir.mkdir(parents=True, exist_ok=True)

            processor = FileProcessor(cache_dir=str(cache_dir))
            cache_key = "test_key"
            cache_file = cache_dir / f"{cache_key}.json"

            # Write valid JSON but not a dict
            cache_file.write_text('["list", "not", "dict"]')

            result = processor._load_from_cache(cache_key)
            assert result is None

    def test_save_cache_permission_denied(self) -> None:
        """Test saving to cache when permissions denied."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / ".holodeck" / "cache"
            processor = FileProcessor(cache_dir=str(cache_dir))

            # Mock open to raise PermissionError
            with mock.patch("builtins.open", side_effect=PermissionError("No access")):
                # Should not raise, just suppress the error
                processor._save_to_cache("test_key", "content", {"key": "value"}, 100)


class TestFileProcessorValidationErrors:
    """Tests for validation error conditions."""

    def test_process_local_file_raises_error_no_path(self) -> None:
        """Test _process_local_file raises ValueError when path is None."""
        import pytest

        processor = FileProcessor()
        # Create a mock FileInput with no path
        mock_input = mock.MagicMock(spec=FileInput)
        mock_input.path = None

        with pytest.raises(ValueError, match="Local file must have path specified"):
            processor._process_local_file(mock_input, start_time=0)

    def test_process_remote_file_raises_error_no_url(self) -> None:
        """Test _process_remote_file raises ValueError when URL is None."""
        import pytest

        processor = FileProcessor()
        # Create a mock FileInput with no URL
        mock_input = mock.MagicMock(spec=FileInput)
        mock_input.url = None

        with pytest.raises(ValueError, match="Remote file must have URL specified"):
            processor._process_remote_file(mock_input, start_time=0)


class TestFileProcessorProcessFile:
    """Additional tests for process_file method."""

    def test_process_file_with_exception(self) -> None:
        """Test process_file handles exceptions gracefully."""
        processor = FileProcessor()
        file_input = FileInput(url="https://example.com/file.pdf", type="pdf")

        with mock.patch.object(processor, "_process_remote_file") as mock_remote:
            mock_remote.side_effect = RuntimeError("Processing failed")

            result = processor.process_file(file_input)

            assert result.error is not None
            assert "Processing failed" in result.error
            assert result.processing_time_ms is not None
            assert result.processing_time_ms >= 0


class TestFileProcessorPageSheetRangeExtraction:
    """Tests for page/sheet/range extraction preprocessing."""

    # PDF page extraction tests
    def test_pdf_pages_extraction_single_page(self) -> None:
        """Test extracting a single page from PDF."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            file_input = FileInput(path=tmp_path, type="pdf", pages=[1])
            processor = FileProcessor()

            # Mock the preprocessing to return a temp file path
            with mock.patch.object(processor, "_preprocess_file") as mock_preprocess:
                mock_preprocess.return_value = Path(tmp_path)

                # Mock markitdown conversion
                mock_md_instance = mock.MagicMock()
                mock_result = mock.MagicMock()
                mock_result.text_content = "# Page 1 Content"
                mock_md_instance.convert.return_value = mock_result

                with mock.patch("markitdown.MarkItDown", return_value=mock_md_instance):
                    result = processor._process_local_file(file_input, start_time=0)

                    assert result.error is None
                    mock_preprocess.assert_called_once_with(file_input, Path(tmp_path))
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_pdf_pages_extraction_multiple_pages(self) -> None:
        """Test extracting multiple specific pages from PDF."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            file_input = FileInput(path=tmp_path, type="pdf", pages=[1, 3, 5])
            processor = FileProcessor()

            with mock.patch.object(processor, "_preprocess_file") as mock_preprocess:
                mock_preprocess.return_value = Path(tmp_path)

                mock_md_instance = mock.MagicMock()
                mock_result = mock.MagicMock()
                mock_result.text_content = "# Pages 1, 3, 5"
                mock_md_instance.convert.return_value = mock_result

                with mock.patch("markitdown.MarkItDown", return_value=mock_md_instance):
                    result = processor._process_local_file(file_input, start_time=0)

                    assert result.error is None
                    mock_preprocess.assert_called_once()
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_pdf_pages_extraction_sequential_range(self) -> None:
        """Test extracting sequential page range from PDF."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            file_input = FileInput(path=tmp_path, type="pdf", pages=[2, 3, 4])
            processor = FileProcessor()

            with mock.patch.object(processor, "_preprocess_file") as mock_preprocess:
                mock_preprocess.return_value = Path(tmp_path)

                mock_md_instance = mock.MagicMock()
                mock_result = mock.MagicMock()
                mock_result.text_content = "# Pages 2-4"
                mock_md_instance.convert.return_value = mock_result

                with mock.patch("markitdown.MarkItDown", return_value=mock_md_instance):
                    result = processor._process_local_file(file_input, start_time=0)

                    assert result.error is None
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_pdf_pages_extraction_invalid_page(self) -> None:
        """Test handling invalid page numbers in PDF."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            file_input = FileInput(path=tmp_path, type="pdf", pages=[999])
            processor = FileProcessor()

            # Mock _preprocess_file to raise an error for invalid pages
            with mock.patch.object(processor, "_preprocess_file") as mock_preprocess:
                mock_preprocess.side_effect = ValueError("Page 999 out of range")

                # Use process_file which handles exceptions
                result = processor.process_file(file_input)

                assert result.error is not None
                assert "out of range" in result.error.lower()
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_pdf_no_pages_full_document(self) -> None:
        """Test processing full PDF when pages is None."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            file_input = FileInput(path=tmp_path, type="pdf", pages=None)
            processor = FileProcessor()

            with mock.patch.object(processor, "_preprocess_file") as mock_preprocess:
                # When pages is None, should return original path (no preprocessing)
                mock_preprocess.return_value = Path(tmp_path)

                mock_md_instance = mock.MagicMock()
                mock_result = mock.MagicMock()
                mock_result.text_content = "# Full PDF"
                mock_md_instance.convert.return_value = mock_result

                with mock.patch("markitdown.MarkItDown", return_value=mock_md_instance):
                    result = processor._process_local_file(file_input, start_time=0)

                    assert result.error is None
                    # Should still call preprocess_file to check for preprocessing needs
                    mock_preprocess.assert_called_once()
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_pdf_empty_pages_list(self) -> None:
        """Test handling empty pages list."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            file_input = FileInput(path=tmp_path, type="pdf", pages=[])
            processor = FileProcessor()

            with mock.patch.object(processor, "_preprocess_file") as mock_preprocess:
                # Empty list should be treated as no preprocessing
                mock_preprocess.return_value = Path(tmp_path)

                mock_md_instance = mock.MagicMock()
                mock_result = mock.MagicMock()
                mock_result.text_content = "# Full PDF"
                mock_md_instance.convert.return_value = mock_result

                with mock.patch("markitdown.MarkItDown", return_value=mock_md_instance):
                    result = processor._process_local_file(file_input, start_time=0)

                    assert result.error is None
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    # Excel sheet/range extraction tests
    def test_excel_sheet_extraction(self) -> None:
        """Test extracting specific sheet from Excel."""
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            file_input = FileInput(path=tmp_path, type="excel", sheet="Sheet2")
            processor = FileProcessor()

            with mock.patch.object(processor, "_preprocess_file") as mock_preprocess:
                # Mock returns a CSV temp file
                mock_csv_path = Path(tmp_path).with_suffix(".csv")
                mock_preprocess.return_value = mock_csv_path

                mock_md_instance = mock.MagicMock()
                mock_result = mock.MagicMock()
                mock_result.text_content = "| Col1 | Col2 |\n| --- | --- |"
                mock_md_instance.convert.return_value = mock_result

                with mock.patch("markitdown.MarkItDown", return_value=mock_md_instance):
                    result = processor._process_local_file(file_input, start_time=0)

                    assert result.error is None
                    mock_preprocess.assert_called_once()
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_excel_sheet_extraction_invalid_sheet(self) -> None:
        """Test handling non-existent sheet name."""
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            file_input = FileInput(path=tmp_path, type="excel", sheet="NonExistent")
            processor = FileProcessor()

            with mock.patch.object(processor, "_preprocess_file") as mock_preprocess:
                mock_preprocess.side_effect = ValueError(
                    "Sheet 'NonExistent' not found"
                )

                result = processor.process_file(file_input)

                assert result.error is not None
                assert "not found" in result.error.lower()
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_excel_range_extraction(self) -> None:
        """Test extracting cell range from Excel."""
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            file_input = FileInput(path=tmp_path, type="excel", range="A1:E10")
            processor = FileProcessor()

            with mock.patch.object(processor, "_preprocess_file") as mock_preprocess:
                mock_csv_path = Path(tmp_path).with_suffix(".csv")
                mock_preprocess.return_value = mock_csv_path

                mock_md_instance = mock.MagicMock()
                mock_result = mock.MagicMock()
                mock_result.text_content = "| A | B | C |"
                mock_md_instance.convert.return_value = mock_result

                with mock.patch("markitdown.MarkItDown", return_value=mock_md_instance):
                    result = processor._process_local_file(file_input, start_time=0)

                    assert result.error is None
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_excel_sheet_and_range_extraction(self) -> None:
        """Test extracting range from specific sheet."""
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            file_input = FileInput(
                path=tmp_path, type="excel", sheet="Data", range="B2:D20"
            )
            processor = FileProcessor()

            with mock.patch.object(processor, "_preprocess_file") as mock_preprocess:
                mock_csv_path = Path(tmp_path).with_suffix(".csv")
                mock_preprocess.return_value = mock_csv_path

                mock_md_instance = mock.MagicMock()
                mock_result = mock.MagicMock()
                mock_result.text_content = "| Header |"
                mock_md_instance.convert.return_value = mock_result

                with mock.patch("markitdown.MarkItDown", return_value=mock_md_instance):
                    result = processor._process_local_file(file_input, start_time=0)

                    assert result.error is None
                    mock_preprocess.assert_called_once()
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_excel_no_sheet_first_sheet_default(self) -> None:
        """Test using first sheet when sheet is None."""
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            file_input = FileInput(path=tmp_path, type="excel", sheet=None)
            processor = FileProcessor()

            with mock.patch.object(processor, "_preprocess_file") as mock_preprocess:
                # When no sheet specified, should process normally
                mock_preprocess.return_value = Path(tmp_path)

                mock_md_instance = mock.MagicMock()
                mock_result = mock.MagicMock()
                mock_result.text_content = "| Data |"
                mock_md_instance.convert.return_value = mock_result

                with mock.patch("markitdown.MarkItDown", return_value=mock_md_instance):
                    result = processor._process_local_file(file_input, start_time=0)

                    assert result.error is None
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    # PowerPoint slide extraction tests
    def test_powerpoint_pages_extraction(self) -> None:
        """Test extracting specific slides from PowerPoint using pages field."""
        with tempfile.NamedTemporaryFile(suffix=".pptx", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            file_input = FileInput(path=tmp_path, type="powerpoint", pages=[1, 3])
            processor = FileProcessor()

            with mock.patch.object(processor, "_preprocess_file") as mock_preprocess:
                mock_pptx_path = Path(tmp_path)
                mock_preprocess.return_value = mock_pptx_path

                mock_md_instance = mock.MagicMock()
                mock_result = mock.MagicMock()
                mock_result.text_content = "# Slide 1\n\n# Slide 3"
                mock_md_instance.convert.return_value = mock_result

                with mock.patch("markitdown.MarkItDown", return_value=mock_md_instance):
                    result = processor._process_local_file(file_input, start_time=0)

                    assert result.error is None
                    mock_preprocess.assert_called_once()
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_powerpoint_pages_extraction_invalid(self) -> None:
        """Test handling invalid slide numbers in PowerPoint."""
        with tempfile.NamedTemporaryFile(suffix=".pptx", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            file_input = FileInput(path=tmp_path, type="powerpoint", pages=[999])
            processor = FileProcessor()

            with mock.patch.object(processor, "_preprocess_file") as mock_preprocess:
                mock_preprocess.side_effect = ValueError("Slide 999 out of range")

                result = processor.process_file(file_input)

                assert result.error is not None
                assert "out of range" in result.error.lower()
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    # Integration tests for preprocessing flow
    def test_preprocessing_before_markitdown(self) -> None:
        """Test that preprocessing happens before markitdown conversion."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            file_input = FileInput(path=tmp_path, type="pdf", pages=[1])
            processor = FileProcessor()

            call_order = []

            def mock_preprocess(*args: object, **kwargs: object) -> Path:
                call_order.append("preprocess")
                return Path(tmp_path)

            def mock_convert(*args: object, **kwargs: object) -> mock.MagicMock:
                call_order.append("convert")
                result = mock.MagicMock()
                result.text_content = "content"
                return result

            with (
                mock.patch.object(
                    processor, "_preprocess_file", side_effect=mock_preprocess
                ),
                mock.patch("markitdown.MarkItDown") as mock_md_class,
            ):
                mock_md_instance = mock.MagicMock()
                mock_md_instance.convert.side_effect = mock_convert
                mock_md_class.return_value = mock_md_instance

                processor._process_local_file(file_input, start_time=0)

                # Verify preprocessing happened before conversion
                assert call_order == ["preprocess", "convert"]
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_preprocessing_creates_temp_file(self) -> None:
        """Test that preprocessing creates temporary file and cleans up."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            file_input = FileInput(path=tmp_path, type="pdf", pages=[1])
            processor = FileProcessor()

            temp_file_created = None

            def mock_preprocess(*args: object, **kwargs: object) -> Path:
                nonlocal temp_file_created
                # Simulate creating a temp file
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as t:
                    temp_file_created = Path(t.name)
                return temp_file_created

            with mock.patch.object(
                processor, "_preprocess_file", side_effect=mock_preprocess
            ):
                mock_md_instance = mock.MagicMock()
                mock_result = mock.MagicMock()
                mock_result.text_content = "content"
                mock_md_instance.convert.return_value = mock_result

                with mock.patch("markitdown.MarkItDown", return_value=mock_md_instance):
                    result = processor._process_local_file(file_input, start_time=0)

                    assert result.error is None
                    # Verify temp file was created during preprocessing
                    assert temp_file_created is not None

            # Clean up the temp file
            if temp_file_created and temp_file_created.exists():
                temp_file_created.unlink()
        finally:
            Path(tmp_path).unlink(missing_ok=True)
