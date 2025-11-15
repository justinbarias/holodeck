"""File processor module for converting multimodal files to markdown using markitdown.

This module provides unified file processing for various file types including:
- Office documents (DOCX, XLSX, PPTX)
- Documents (PDF, TXT, HTML)
- Images (JPG, PNG with OCR)
- Data files (CSV, JSON)

Files are converted to markdown format for optimal LLM consumption.
"""

import contextlib
import hashlib
import json
import tempfile
import time
from pathlib import Path
from typing import Any

import requests

from holodeck.lib.logging_config import get_logger
from holodeck.lib.logging_utils import log_exception, log_retry
from holodeck.models.test_case import FileInput
from holodeck.models.test_result import ProcessedFileInput

logger = get_logger(__name__)


class FileProcessor:
    """Process files with markitdown for multimodal test inputs."""

    def __init__(
        self,
        cache_dir: str | None = None,
        download_timeout_ms: int = 30000,
        max_retries: int = 3,
    ) -> None:
        """Initialize file processor.

        Args:
            cache_dir: Directory for caching remote files. Defaults to .holodeck/cache/
            download_timeout_ms: Timeout for file downloads in milliseconds
            max_retries: Maximum number of retry attempts for downloads
        """
        self.cache_dir = Path(cache_dir or ".holodeck/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.download_timeout_ms = download_timeout_ms
        self.max_retries = max_retries
        self.md: Any = None  # Initialize lazily

        logger.debug(
            f"FileProcessor initialized: cache_dir={self.cache_dir}, "
            f"timeout={download_timeout_ms}ms, max_retries={max_retries}"
        )

        try:
            from markitdown import MarkItDown  # noqa: F401
        except ImportError as e:
            logger.error("markitdown package not found", exc_info=True)
            raise ImportError(
                "markitdown is required for file processing. "
                "Install with: pip install 'markitdown[all]'"
            ) from e

    def _get_markitdown(self) -> Any:
        """Get or create MarkItDown instance."""
        if self.md is None:
            from markitdown import MarkItDown

            self.md = MarkItDown()
        return self.md

    def _get_cache_key(self, url_or_path: str) -> str:
        """Generate cache key using MD5 hash of URL or path.

        Args:
            url_or_path: File URL or path

        Returns:
            MD5 hash of the input string
        """
        return hashlib.md5(url_or_path.encode(), usedforsecurity=False).hexdigest()

    def _load_from_cache(self, cache_key: str) -> dict[str, Any] | None:
        """Load processed file from cache.

        Args:
            cache_key: Cache key (MD5 hash)

        Returns:
            Cached data dict or None if not found
        """
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file) as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        logger.debug(f"Cache hit: {cache_key}")
                        return data
                    return None
            except Exception as e:
                logger.warning(f"Failed to load cache for {cache_key}: {e}")
                return None
        logger.debug(f"Cache miss: {cache_key}")
        return None

    def _save_to_cache(
        self,
        cache_key: str,
        markdown_content: str,
        metadata: dict,
        processing_time_ms: int,
    ) -> None:
        """Save processed file to cache.

        Args:
            cache_key: Cache key (MD5 hash)
            markdown_content: Converted markdown content
            metadata: File metadata
            processing_time_ms: Processing time in milliseconds
        """
        cache_file = self.cache_dir / f"{cache_key}.json"
        data = {
            "markdown_content": markdown_content,
            "metadata": metadata,
            "processing_time_ms": processing_time_ms,
        }
        try:
            with open(cache_file, "w") as f:
                json.dump(data, f)
            logger.debug(f"File cached: {cache_key} ({len(markdown_content)} bytes)")
        except Exception as e:
            logger.warning(f"Failed to save cache for {cache_key}: {e}")

    def _download_file(self, url: str) -> bytes | None:
        """Download file from URL with retry logic.

        Implements exponential backoff: 1s, 2s, 4s for retries.

        Args:
            url: Remote file URL

        Returns:
            File content bytes or None if download fails
        """
        timeout_sec = self.download_timeout_ms / 1000.0

        logger.debug(f"Downloading file from URL: {url}")

        for attempt in range(self.max_retries):
            try:
                response = requests.get(url, timeout=timeout_sec)
                response.raise_for_status()
                size_bytes = len(response.content)
                logger.debug(
                    f"File downloaded successfully: {url} ({size_bytes} bytes)"
                )
                return response.content
            except Exception as e:
                if attempt == self.max_retries - 1:
                    logger.error(
                        f"Failed to download file after {self.max_retries} "
                        f"attempts: {url}"
                    )
                    return None

                # Exponential backoff: 1s, 2s, 4s
                backoff_sec = 2**attempt
                log_retry(
                    logger,
                    f"Download {url}",
                    attempt=attempt + 1,
                    max_attempts=self.max_retries,
                    delay=backoff_sec,
                    error=e,
                )
                time.sleep(backoff_sec)

        return None

    def process_file(self, file_input: FileInput) -> ProcessedFileInput:
        """Process a single file input to markdown.

        Args:
            file_input: File input configuration with path or URL

        Returns:
            ProcessedFileInput with markdown content and metadata
        """
        start_time = time.time()
        file_location = file_input.url or file_input.path or "unknown"

        logger.debug(
            f"Processing file: {file_location} (type={file_input.type}, "
            f"cache={'enabled' if file_input.cache else 'disabled'})"
        )

        try:
            # Determine if file is local or remote
            if file_input.url:
                return self._process_remote_file(file_input, start_time)
            else:
                return self._process_local_file(file_input, start_time)

        except Exception as e:
            elapsed_ms = int((time.time() - start_time) * 1000)
            log_exception(logger, f"File processing failed: {file_location}", e)
            return ProcessedFileInput(
                original=file_input,
                markdown_content="",
                metadata={},
                processing_time_ms=elapsed_ms,
                cached_path=None,
                error=str(e),
            )

    def _process_local_file(
        self, file_input: FileInput, start_time: float
    ) -> ProcessedFileInput:
        """Process a local file.

        Args:
            file_input: File input with local path
            start_time: Processing start time

        Returns:
            ProcessedFileInput with markdown content
        """
        if not file_input.path:
            raise ValueError("Local file must have path specified")

        path = Path(file_input.path)

        # Validate file exists
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_input.path}")

        # Get file metadata
        file_size = path.stat().st_size
        metadata: dict = {
            "size_bytes": file_size,
            "path": str(path),
            "type": file_input.type,
        }

        # Warn if file is large
        size_mb = file_size / (1024 * 1024)
        if size_mb > 100:
            logger.warning(f"Large file detected: {file_input.path} ({size_mb:.2f}MB)")
            metadata["warning"] = f"Large file detected ({size_mb:.2f}MB)"

        # Convert file
        logger.debug(f"Converting local file to markdown: {file_input.path}")
        md = self._get_markitdown()
        result = md.convert(str(path))
        markdown_content = result.text_content

        elapsed_ms = int((time.time() - start_time) * 1000)

        logger.debug(
            f"Local file processed: {file_input.path} "
            f"({len(markdown_content)} bytes in {elapsed_ms}ms)"
        )

        return ProcessedFileInput(
            original=file_input,
            markdown_content=markdown_content,
            metadata=metadata,
            processing_time_ms=elapsed_ms,
            cached_path=None,
            error=None,
        )

    def _process_remote_file(
        self, file_input: FileInput, start_time: float
    ) -> ProcessedFileInput:
        """Process a remote file from URL.

        Args:
            file_input: File input with URL
            start_time: Processing start time

        Returns:
            ProcessedFileInput with markdown content
        """
        if not file_input.url:
            raise ValueError("Remote file must have URL specified")

        url = file_input.url

        # Check cache if enabled
        if file_input.cache is not False:
            cache_key = self._get_cache_key(url)
            cached_data = self._load_from_cache(cache_key)

            if cached_data:
                elapsed_ms = int((time.time() - start_time) * 1000)
                logger.debug(f"Using cached remote file: {url}")
                return ProcessedFileInput(
                    original=file_input,
                    markdown_content=cached_data["markdown_content"],
                    metadata=cached_data["metadata"],
                    cached_path=str(self.cache_dir / f"{cache_key}.json"),
                    processing_time_ms=elapsed_ms,
                    error=None,
                )

        # Download file
        file_content = self._download_file(url)
        if file_content is None:
            elapsed_ms = int((time.time() - start_time) * 1000)
            logger.error(f"Failed to download remote file: {url}")
            return ProcessedFileInput(
                original=file_input,
                markdown_content="",
                metadata={"url": url},
                processing_time_ms=elapsed_ms,
                cached_path=None,
                error="Failed to download file after max retries",
            )

        # Save to temporary file and process
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(file_content)
            tmp_path = tmp.name

        try:
            logger.debug(f"Converting remote file to markdown: {url}")
            md = self._get_markitdown()
            result = md.convert(tmp_path)
            markdown_content = result.text_content

            # Get metadata
            metadata: dict = {
                "url": url,
                "size_bytes": len(file_content),
                "type": file_input.type,
            }

            elapsed_ms = int((time.time() - start_time) * 1000)
            cached_path = None

            # Cache if enabled
            if file_input.cache is not False:
                cache_key = self._get_cache_key(url)
                self._save_to_cache(cache_key, markdown_content, metadata, elapsed_ms)
                cached_path = str(self.cache_dir / f"{cache_key}.json")

            logger.debug(
                f"Remote file processed: {url} "
                f"({len(markdown_content)} bytes in {elapsed_ms}ms, "
                f"cached={cached_path is not None})"
            )

            return ProcessedFileInput(
                original=file_input,
                markdown_content=markdown_content,
                metadata=metadata,
                cached_path=cached_path,
                processing_time_ms=elapsed_ms,
                error=None,
            )

        finally:
            # Clean up temporary file
            with contextlib.suppress(Exception):
                Path(tmp_path).unlink()
