"""Source resolution for tool initialization.

Resolves local paths and remote URIs (S3, Azure Blob, HTTP) to local
directories for tool ingestion. Provides guaranteed cleanup of temporary
files and security utilities for error sanitization.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)

# Supported remote URI schemes
_REMOTE_SCHEMES = {"s3", "az", "https", "http"}
_LOCAL_SCHEMES = {"file"}
_ALL_SCHEMES = _REMOTE_SCHEMES | _LOCAL_SCHEMES

# Default limits
DEFAULT_MAX_SOURCE_SIZE_BYTES = 500 * 1024 * 1024  # 500 MB
TEMP_DIR_PREFIX = "holodeck-init-"

# HTTP retry settings
_HTTP_MAX_RETRIES = 3
_HTTP_BACKOFF_SECONDS = [1, 2, 4]


class SourceError(Exception):
    """Raised when source resolution fails."""


@dataclass
class ResolvedSource:
    """Result of resolving a source URI to a local path."""

    local_path: Path
    is_remote: bool
    temp_dir: Path | None


def _import_boto3() -> ModuleType:
    """Lazy-import boto3 with a clear error message.

    Returns:
        The boto3 module.

    Raises:
        SourceError: If boto3 is not installed.
    """
    try:
        import boto3  # type: ignore[import-untyped]

        return boto3  # type: ignore[no-any-return]
    except ImportError as exc:
        raise SourceError(
            "S3 source requires boto3. Install with: pip install holodeck[s3]"
        ) from exc


def _import_azure_blob() -> ModuleType:
    """Lazy-import azure.storage.blob with a clear error message.

    Returns:
        The azure.storage.blob module.

    Raises:
        SourceError: If azure-storage-blob is not installed.
    """
    try:
        import azure.storage.blob  # type: ignore[import-untyped]

        return azure.storage.blob  # type: ignore[no-any-return]
    except ImportError as exc:
        raise SourceError(
            "Azure source requires azure-storage-blob. "
            "Install with: pip install holodeck[azure-blob]"
        ) from exc


class SourceResolver:
    """Resolves source URIs to local paths for tool initialization."""

    @staticmethod
    def _detect_scheme(source: str) -> str:
        """Detect the URI scheme of a source string.

        Args:
            source: Source path or URI.

        Returns:
            Scheme identifier: "local", "s3", "az", or "http".

        Raises:
            ValueError: If the scheme is not supported.
        """
        if "://" not in source:
            return "local"

        scheme = source.split("://", 1)[0].lower()

        if scheme in _LOCAL_SCHEMES:
            return "local"
        if scheme in _REMOTE_SCHEMES:
            # Normalize https/http to "http" for resolver dispatch
            if scheme in ("https", "http"):
                return "http"
            return scheme

        raise ValueError(
            f"Unsupported source scheme: '{scheme}'. "
            f"Supported schemes: {sorted(_ALL_SCHEMES | {'local'})}"
        )

    @staticmethod
    async def resolve(
        source: str,
        base_dir: str | None = None,
        max_source_size_bytes: int = DEFAULT_MAX_SOURCE_SIZE_BYTES,
    ) -> ResolvedSource:
        """Resolve a source URI to a local path.

        Args:
            source: Source path or URI.
            base_dir: Base directory for relative local paths.
            max_source_size_bytes: Maximum total download size for remote sources.

        Returns:
            ResolvedSource with local path and metadata.

        Raises:
            SourceError: If resolution fails.
            ValueError: If the scheme is unsupported.
        """
        scheme = SourceResolver._detect_scheme(source)

        if scheme == "local":
            return await _resolve_local(source, base_dir)
        if scheme == "s3":
            return await _resolve_s3(source, max_source_size_bytes)
        if scheme == "az":
            return await _resolve_azure(source, max_source_size_bytes)
        if scheme == "http":
            return await _resolve_http(source, max_source_size_bytes)

        raise NotImplementedError(
            f"Remote resolver for scheme '{scheme}' not yet implemented"
        )

    @staticmethod
    @asynccontextmanager
    async def resolve_context(
        source: str,
        base_dir: str | None = None,
        max_source_size_bytes: int = DEFAULT_MAX_SOURCE_SIZE_BYTES,
    ) -> AsyncIterator[ResolvedSource]:
        """Context manager for source resolution with guaranteed cleanup.

        Usage:
            async with SourceResolver.resolve_context("s3://bucket/data") as resolved:
                # Use resolved.local_path
            # Temp directory automatically cleaned up

        Args:
            source: Source path or URI.
            base_dir: Base directory for relative local paths.
            max_source_size_bytes: Maximum total download size.

        Yields:
            ResolvedSource with local path and metadata.
        """
        resolved = await SourceResolver.resolve(source, base_dir, max_source_size_bytes)
        try:
            yield resolved
        finally:
            if resolved.temp_dir is not None:
                await SourceResolver.cleanup(resolved.temp_dir)

    @staticmethod
    async def cleanup(temp_dir: Path) -> None:
        """Remove a temporary directory created during source resolution.

        Args:
            temp_dir: Path to the temporary directory.
        """
        if temp_dir.exists():
            await asyncio.to_thread(shutil.rmtree, str(temp_dir), ignore_errors=True)
            logger.debug("Cleaned up temp directory: %s", temp_dir)

    @staticmethod
    async def cleanup_orphans(max_age_hours: float = 1.0) -> int:
        """Remove stale temporary directories from previous runs.

        Scans the system temp directory for directories matching the
        holodeck-init-* prefix that are older than max_age_hours.

        Args:
            max_age_hours: Maximum age in hours before cleanup.

        Returns:
            Number of directories removed.
        """
        import time

        tmpdir = Path(tempfile.gettempdir())
        max_age_seconds = max_age_hours * 3600
        now = time.time()
        removed = 0

        for entry in tmpdir.iterdir():
            if not entry.is_dir() or not entry.name.startswith(TEMP_DIR_PREFIX):
                continue

            try:
                mtime = entry.stat().st_mtime
                if (now - mtime) > max_age_seconds:
                    await asyncio.to_thread(
                        shutil.rmtree, str(entry), ignore_errors=True
                    )
                    removed += 1
                    logger.info("Removed orphan temp dir: %s", entry)
            except OSError:
                continue

        return removed


async def _resolve_local(source: str, base_dir: str | None = None) -> ResolvedSource:
    """Resolve a local file path or file:// URI.

    Args:
        source: Local path or file:// URI.
        base_dir: Base directory for relative paths.

    Returns:
        ResolvedSource with local path.
    """
    from holodeck.tools.common import resolve_source_path

    # Strip file:// scheme if present
    if source.startswith("file://"):
        source = source[7:]

    resolved_path = resolve_source_path(source, base_dir)
    return ResolvedSource(local_path=resolved_path, is_remote=False, temp_dir=None)


async def _resolve_s3(
    source: str,
    max_source_size_bytes: int = DEFAULT_MAX_SOURCE_SIZE_BYTES,
) -> ResolvedSource:
    """Resolve an S3 URI by downloading objects to a temp directory.

    Args:
        source: S3 URI in the form s3://bucket/prefix/.
        max_source_size_bytes: Maximum total download size in bytes.

    Returns:
        ResolvedSource with local path to temp directory.

    Raises:
        SourceError: If download fails or size exceeds limit.
    """
    from holodeck.tools.common import SUPPORTED_EXTENSIONS

    boto3 = _import_boto3()

    # Parse bucket and prefix from s3://bucket/prefix/
    parsed = urlparse(source)
    bucket = parsed.netloc
    prefix = parsed.path.lstrip("/")

    s3_client = boto3.client("s3")

    # List objects
    response = await asyncio.to_thread(
        s3_client.list_objects_v2, Bucket=bucket, Prefix=prefix
    )
    contents = response.get("Contents", [])

    # Filter by supported extensions
    filtered = []
    for obj in contents:
        key = obj["Key"]
        ext = Path(key).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            logger.debug("Skipping unsupported extension: %s", key)
            continue
        filtered.append(obj)

    # Validate total size
    total_size = sum(obj["Size"] for obj in filtered)
    if total_size > max_source_size_bytes:
        raise SourceError(
            f"Total size of S3 objects ({total_size} bytes) exceeds "
            f"maximum allowed size ({max_source_size_bytes} bytes)"
        )

    # Create temp directory
    temp_dir = Path(tempfile.mkdtemp(prefix=TEMP_DIR_PREFIX))

    try:
        for obj in filtered:
            key = obj["Key"]
            # Validate object path for traversal
            try:
                target_path = validate_object_path(key, temp_dir)
            except ValueError:
                logger.warning("Skipping invalid object key: %s", key)
                continue

            # Ensure parent directories exist
            target_path.parent.mkdir(parents=True, exist_ok=True)

            # Download file
            await asyncio.to_thread(
                s3_client.download_file, bucket, key, str(target_path)
            )
            logger.debug("Downloaded s3://%s/%s -> %s", bucket, key, target_path)
    except Exception:
        # Clean up on failure
        await asyncio.to_thread(shutil.rmtree, str(temp_dir), ignore_errors=True)
        raise

    return ResolvedSource(local_path=temp_dir, is_remote=True, temp_dir=temp_dir)


async def _resolve_azure(
    source: str,
    max_source_size_bytes: int = DEFAULT_MAX_SOURCE_SIZE_BYTES,
) -> ResolvedSource:
    """Resolve an Azure Blob URI by downloading blobs to a temp directory.

    Args:
        source: Azure Blob URI in the form az://container/prefix/.
        max_source_size_bytes: Maximum total download size in bytes.

    Returns:
        ResolvedSource with local path to temp directory.

    Raises:
        SourceError: If download fails or size exceeds limit.
    """
    from holodeck.tools.common import SUPPORTED_EXTENSIONS

    azure_blob = _import_azure_blob()

    # Parse container and prefix from az://container/prefix/
    parsed = urlparse(source)
    container_name = parsed.netloc
    prefix = parsed.path.lstrip("/")

    # Create service client from connection string or DefaultAzureCredential
    conn_str = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
    if conn_str:
        service_client = azure_blob.BlobServiceClient.from_connection_string(conn_str)
    else:
        raise SourceError(
            "Azure Blob source requires AZURE_STORAGE_CONNECTION_STRING "
            "environment variable to be set"
        )

    container_client = service_client.get_container_client(container_name)

    # List blobs
    blobs = list(container_client.list_blobs(name_starts_with=prefix))

    # Filter by supported extensions
    filtered = []
    for blob in blobs:
        ext = Path(blob.name).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            logger.debug("Skipping unsupported extension: %s", blob.name)
            continue
        filtered.append(blob)

    # Validate total size
    total_size = sum(blob.size for blob in filtered)
    if total_size > max_source_size_bytes:
        raise SourceError(
            f"Total size of Azure blobs ({total_size} bytes) exceeds "
            f"maximum allowed size ({max_source_size_bytes} bytes)"
        )

    # Create temp directory
    temp_dir = Path(tempfile.mkdtemp(prefix=TEMP_DIR_PREFIX))

    try:
        for blob in filtered:
            # Validate object path for traversal
            try:
                target_path = validate_object_path(blob.name, temp_dir)
            except ValueError:
                logger.warning("Skipping invalid blob name: %s", blob.name)
                continue

            # Ensure parent directories exist
            target_path.parent.mkdir(parents=True, exist_ok=True)

            # Download blob
            blob_client = container_client.get_blob_client(blob.name)
            data = blob_client.download_blob().readall()
            target_path.write_bytes(data)
            logger.debug(
                "Downloaded az://%s/%s -> %s",
                container_name,
                blob.name,
                target_path,
            )
    except SourceError:
        await asyncio.to_thread(shutil.rmtree, str(temp_dir), ignore_errors=True)
        raise
    except Exception:
        await asyncio.to_thread(shutil.rmtree, str(temp_dir), ignore_errors=True)
        raise

    return ResolvedSource(local_path=temp_dir, is_remote=True, temp_dir=temp_dir)


async def _resolve_http(
    source: str,
    max_source_size_bytes: int = DEFAULT_MAX_SOURCE_SIZE_BYTES,
) -> ResolvedSource:
    """Resolve an HTTP/HTTPS URI by downloading to a temp directory.

    Args:
        source: HTTP or HTTPS URL.
        max_source_size_bytes: Maximum download size in bytes.

    Returns:
        ResolvedSource with local path to temp directory.

    Raises:
        SourceError: If download fails or size exceeds limit.
    """
    # Infer filename from URL path
    parsed = urlparse(source)
    filename = Path(parsed.path).name or "download"

    # Build headers
    headers: dict[str, str] = {}
    auth_header = os.environ.get("HOLODECK_HTTP_AUTH_HEADER")
    if auth_header:
        headers["Authorization"] = auth_header

    async with httpx.AsyncClient() as client:
        for attempt in range(_HTTP_MAX_RETRIES):
            try:
                async with client.stream("GET", source, headers=headers) as response:
                    response.raise_for_status()

                    # Check Content-Length before downloading
                    content_length_str = response.headers.get("content-length")
                    if content_length_str:
                        content_length = int(content_length_str)
                        if content_length > max_source_size_bytes:
                            raise SourceError(
                                f"Content-Length ({content_length} bytes) exceeds "
                                f"maximum allowed size ({max_source_size_bytes} bytes)"
                            )

                    # Create temp directory
                    temp_dir = Path(tempfile.mkdtemp(prefix=TEMP_DIR_PREFIX))
                    target_path = temp_dir / filename

                    try:
                        downloaded = 0
                        with open(target_path, "wb") as f:
                            async for chunk in response.aiter_bytes():
                                downloaded += len(chunk)
                                if downloaded > max_source_size_bytes:
                                    raise SourceError(
                                        f"Download size exceeds maximum "
                                        f"({max_source_size_bytes} bytes)"
                                    )
                                f.write(chunk)
                    except SourceError:
                        await asyncio.to_thread(
                            shutil.rmtree, str(temp_dir), ignore_errors=True
                        )
                        raise
                    except Exception:
                        await asyncio.to_thread(
                            shutil.rmtree, str(temp_dir), ignore_errors=True
                        )
                        raise

                    logger.debug("Downloaded %s -> %s", source, target_path)
                    return ResolvedSource(
                        local_path=temp_dir, is_remote=True, temp_dir=temp_dir
                    )

            except SourceError:
                raise
            except httpx.HTTPStatusError as exc:
                status_code = exc.response.status_code
                if 500 <= status_code < 600:
                    if attempt < _HTTP_MAX_RETRIES - 1:
                        backoff = _HTTP_BACKOFF_SECONDS[attempt]
                        logger.warning(
                            "HTTP %d from %s, retrying in %ds (attempt %d/%d)",
                            status_code,
                            source,
                            backoff,
                            attempt + 1,
                            _HTTP_MAX_RETRIES,
                        )
                        await asyncio.sleep(backoff)
                        continue
                    raise SourceError(
                        f"Failed to download {source} after {_HTTP_MAX_RETRIES} "
                        f"retries: HTTP {status_code}"
                    ) from exc
                else:
                    raise SourceError(
                        f"Failed to download {source}: HTTP {status_code}"
                    ) from exc
            except Exception as exc:
                raise SourceError(f"Failed to download {source}: {exc}") from exc

    # Should not reach here, but just in case
    raise SourceError(f"Failed to download {source} after {_HTTP_MAX_RETRIES} retries")


def sanitize_error_detail(detail: str) -> str:
    """Sanitize error messages by redacting sensitive information.

    Redacts:
    - AWS access key IDs (AKIA...)
    - Azure connection string account keys
    - Bearer tokens
    - Generic API keys, tokens, secrets, passwords
    - Temporary directory paths from init jobs

    Args:
        detail: Raw error detail string.

    Returns:
        Sanitized string safe for client-facing responses.
    """
    # AWS access key IDs
    detail = re.sub(r"AKIA[A-Z0-9]{16}", "<AWS_KEY_REDACTED>", detail)

    # Azure connection string account keys
    detail = re.sub(r"AccountKey=[^;]+", "AccountKey=<REDACTED>", detail)

    # Bearer tokens
    detail = re.sub(r"Bearer\s+[A-Za-z0-9._\-]+", "Bearer <REDACTED>", detail)

    # Generic sensitive key-value patterns (api_key, api-key, token, secret, password)
    detail = re.sub(
        r"(api[_-]?key|token|secret|password)\s*[=:]\s*\S+",
        r"\1=<REDACTED>",
        detail,
        flags=re.IGNORECASE,
    )

    # Temp directory paths
    detail = re.sub(
        r"/tmp/holodeck-init-[^\s/]+",  # noqa: S108  # nosec B108
        "<TEMP_DIR>",
        detail,
    )

    return detail


def validate_object_path(key: str, temp_dir: Path) -> Path:
    """Validate that an object key does not escape the temp directory.

    Prevents path traversal attacks when downloading from S3 or Azure Blob.

    Args:
        key: S3 object key or Azure blob name.
        temp_dir: Temporary directory that files must stay within.

    Returns:
        Validated absolute path within temp_dir.

    Raises:
        ValueError: If the key contains path traversal or resolves outside temp_dir.
    """
    # Check for explicit path traversal
    if ".." in key.split("/"):
        raise ValueError(f"Invalid path traversal in object key: '{key}'")

    # Resolve and verify containment
    target = (temp_dir / key).resolve()
    resolved_temp = temp_dir.resolve()

    if not target.is_relative_to(resolved_temp):
        raise ValueError(f"Object key resolves outside temp directory: '{key}'")

    return target
