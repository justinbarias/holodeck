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
