"""Tests for source resolver infrastructure."""

from __future__ import annotations

import os
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from holodeck.lib.source_resolver import (
    ResolvedSource,
    SourceError,
    SourceResolver,
    sanitize_error_detail,
    validate_object_path,
)


class TestDetectScheme:
    """Tests for SourceResolver._detect_scheme()."""

    def test_local_relative_path(self) -> None:
        assert SourceResolver._detect_scheme("./data") == "local"

    def test_local_absolute_path(self) -> None:
        assert SourceResolver._detect_scheme("/abs/path") == "local"

    def test_s3_scheme(self) -> None:
        assert SourceResolver._detect_scheme("s3://bucket/key") == "s3"

    def test_az_scheme(self) -> None:
        assert SourceResolver._detect_scheme("az://container/blob") == "az"

    def test_https_scheme(self) -> None:
        assert SourceResolver._detect_scheme("https://host/file") == "http"

    def test_http_scheme(self) -> None:
        assert SourceResolver._detect_scheme("http://host/file") == "http"

    def test_file_scheme(self) -> None:
        assert SourceResolver._detect_scheme("file:///path") == "local"

    def test_unsupported_scheme_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported source scheme"):
            SourceResolver._detect_scheme("ftp://host")


class TestResolvedSource:
    """Tests for ResolvedSource dataclass."""

    def test_local_source(self) -> None:
        rs = ResolvedSource(local_path=Path("/data"), is_remote=False, temp_dir=None)
        assert rs.is_remote is False
        assert rs.temp_dir is None

    def test_remote_source(self) -> None:
        tmp = Path("/tmp/holodeck-init-test")  # noqa: S108
        rs = ResolvedSource(local_path=tmp / "data", is_remote=True, temp_dir=tmp)
        assert rs.is_remote is True
        assert rs.temp_dir == tmp


class TestSanitizeErrorDetail:
    """Tests for sanitize_error_detail()."""

    def test_redacts_aws_access_key(self) -> None:
        result = sanitize_error_detail("Key: AKIAIOSFODNN7EXAMPLE")
        assert "AKIAIOSFODNN7EXAMPLE" not in result
        assert "<AWS_KEY_REDACTED>" in result

    def test_redacts_azure_connection_string(self) -> None:
        result = sanitize_error_detail("AccountKey=abc123def456;stuff")
        assert "abc123def456" not in result
        assert "AccountKey=<REDACTED>" in result

    def test_redacts_bearer_token(self) -> None:
        result = sanitize_error_detail("Authorization: Bearer eyJhbGciOiJ.xyz")
        assert "eyJhbGciOiJ" not in result
        assert "Bearer <REDACTED>" in result

    def test_redacts_api_key_equals(self) -> None:
        result = sanitize_error_detail("api_key=secret123abc")
        assert "secret123abc" not in result

    def test_redacts_token_colon(self) -> None:
        result = sanitize_error_detail("token: mysecrettoken")
        assert "mysecrettoken" not in result

    def test_redacts_temp_paths(self) -> None:
        result = sanitize_error_detail(
            "/tmp/holodeck-init-my_tool-abc123/subdir/file.txt failed"  # noqa: S108
        )
        assert "holodeck-init-my_tool-abc123" not in result
        assert "<TEMP_DIR>" in result

    def test_preserves_safe_text(self) -> None:
        msg = "Connection timed out after 30 seconds"
        assert sanitize_error_detail(msg) == msg

    def test_redacts_multiple_patterns(self) -> None:
        msg = "Key AKIAIOSFODNN7EXAMPLE with Bearer eyJhbGci.xyz"
        result = sanitize_error_detail(msg)
        assert "AKIAIOSFODNN7EXAMPLE" not in result
        assert "eyJhbGci" not in result

    def test_redacts_secret_keyword(self) -> None:
        result = sanitize_error_detail("secret=mysupersecret")
        assert "mysupersecret" not in result

    def test_redacts_password_keyword(self) -> None:
        result = sanitize_error_detail("password=hunter2")
        assert "hunter2" not in result


class TestValidateObjectPath:
    """Tests for validate_object_path()."""

    def test_valid_path(self, tmp_path: Path) -> None:
        result = validate_object_path("docs/file.txt", tmp_path)
        assert result == (tmp_path / "docs" / "file.txt").resolve()

    def test_nested_path(self, tmp_path: Path) -> None:
        result = validate_object_path("a/b/c/file.md", tmp_path)
        assert result.is_relative_to(tmp_path.resolve())

    def test_dotdot_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="path traversal"):
            validate_object_path("../etc/passwd", tmp_path)

    def test_dotdot_middle_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="path traversal"):
            validate_object_path("docs/../../etc/passwd", tmp_path)

    def test_resolved_outside_temp_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            validate_object_path("/etc/passwd", tmp_path)


# ---------------------------------------------------------------------------
# T019 — LocalResolver
# ---------------------------------------------------------------------------


class TestLocalResolver:
    """Tests for _resolve_local() via SourceResolver.resolve()."""

    @pytest.mark.asyncio
    @patch("holodeck.tools.common.resolve_source_path")
    async def test_resolve_local_absolute_path(self, mock_resolve: MagicMock) -> None:
        """Resolve a relative local path with a base_dir."""
        mock_resolve.return_value = Path("/tmp/resolved/data")  # noqa: S108
        result = await SourceResolver.resolve("./data", base_dir="/tmp")  # noqa: S108
        assert result.is_remote is False
        assert result.temp_dir is None
        assert result.local_path == Path("/tmp/resolved/data")  # noqa: S108
        mock_resolve.assert_called_once_with("./data", "/tmp")  # noqa: S108

    @pytest.mark.asyncio
    @patch("holodeck.tools.common.resolve_source_path")
    async def test_resolve_local_file_scheme(self, mock_resolve: MagicMock) -> None:
        """Resolve a file:// URI — scheme is stripped before resolution."""
        mock_resolve.return_value = Path("/some/path")
        result = await SourceResolver.resolve("file:///some/path")
        assert result.is_remote is False
        assert result.temp_dir is None
        # file:// should be stripped; resolve_source_path sees "/some/path"
        mock_resolve.assert_called_once_with("/some/path", None)


# ---------------------------------------------------------------------------
# T020 — S3Resolver
# ---------------------------------------------------------------------------


class TestS3Resolver:
    """Tests for _resolve_s3()."""

    @pytest.mark.asyncio
    async def test_resolve_s3_downloads_to_temp_dir(self, tmp_path: Path) -> None:
        """Mock boto3 and verify files are downloaded to temp dir."""
        mock_s3 = MagicMock()
        # Simulate list_objects_v2 returning two valid files
        mock_s3.list_objects_v2.return_value = {
            "Contents": [
                {"Key": "prefix/doc.txt", "Size": 100},
                {"Key": "prefix/notes.md", "Size": 200},
            ],
            "IsTruncated": False,
        }

        temp_dir = tmp_path / "holodeck-init-test"
        temp_dir.mkdir()

        with (
            patch(
                "holodeck.lib.source_resolver._import_boto3",
                return_value=MagicMock(client=MagicMock(return_value=mock_s3)),
            ),
            patch("tempfile.mkdtemp", return_value=str(temp_dir)),
        ):
            result = await SourceResolver.resolve("s3://mybucket/prefix/")

        assert result.is_remote is True
        assert result.temp_dir == temp_dir
        assert result.local_path == temp_dir
        # Verify download_file was called for each object
        assert mock_s3.download_file.call_count == 2

    @pytest.mark.asyncio
    async def test_resolve_s3_missing_boto3_raises(self) -> None:
        """When boto3 import fails, a SourceError with install hint is raised."""
        with (
            patch(
                "holodeck.lib.source_resolver._import_boto3",
                side_effect=SourceError(
                    "S3 source requires boto3. Install with: pip install holodeck[s3]"
                ),
            ),
            pytest.raises(SourceError, match="boto3"),
        ):
            await SourceResolver.resolve("s3://mybucket/prefix/")

    @pytest.mark.asyncio
    async def test_resolve_s3_filters_unsupported_extensions(
        self, tmp_path: Path
    ) -> None:
        """Files with unsupported extensions (e.g. .exe) are skipped."""
        mock_s3 = MagicMock()
        mock_s3.list_objects_v2.return_value = {
            "Contents": [
                {"Key": "prefix/doc.txt", "Size": 100},
                {"Key": "prefix/malware.exe", "Size": 500},
            ],
            "IsTruncated": False,
        }

        temp_dir = tmp_path / "holodeck-init-test"
        temp_dir.mkdir()

        with (
            patch(
                "holodeck.lib.source_resolver._import_boto3",
                return_value=MagicMock(client=MagicMock(return_value=mock_s3)),
            ),
            patch("tempfile.mkdtemp", return_value=str(temp_dir)),
        ):
            await SourceResolver.resolve("s3://mybucket/prefix/")

        # Only the .txt file should be downloaded
        assert mock_s3.download_file.call_count == 1

    @pytest.mark.asyncio
    async def test_resolve_s3_validates_total_size(self) -> None:
        """Objects exceeding max_source_size_bytes raise SourceError."""
        mock_s3 = MagicMock()
        mock_s3.list_objects_v2.return_value = {
            "Contents": [
                {"Key": "prefix/big.txt", "Size": 1_000_000_000},
            ],
            "IsTruncated": False,
        }

        with (
            patch(
                "holodeck.lib.source_resolver._import_boto3",
                return_value=MagicMock(client=MagicMock(return_value=mock_s3)),
            ),
            pytest.raises(SourceError, match="[Ss]ize"),
        ):
            await SourceResolver.resolve(
                "s3://mybucket/prefix/", max_source_size_bytes=1024
            )

    @pytest.mark.asyncio
    async def test_resolve_s3_skips_traversal_keys(self, tmp_path: Path) -> None:
        """Keys with '..' are skipped (logged), not fatal."""
        mock_s3 = MagicMock()
        mock_s3.list_objects_v2.return_value = {
            "Contents": [
                {"Key": "prefix/../etc/passwd", "Size": 50},
                {"Key": "prefix/safe.txt", "Size": 100},
            ],
            "IsTruncated": False,
        }

        temp_dir = tmp_path / "holodeck-init-test"
        temp_dir.mkdir()

        with (
            patch(
                "holodeck.lib.source_resolver._import_boto3",
                return_value=MagicMock(client=MagicMock(return_value=mock_s3)),
            ),
            patch("tempfile.mkdtemp", return_value=str(temp_dir)),
        ):
            await SourceResolver.resolve("s3://mybucket/prefix/")

        # Only the safe file should be downloaded
        assert mock_s3.download_file.call_count == 1


# ---------------------------------------------------------------------------
# T021 — AzureBlobResolver
# ---------------------------------------------------------------------------


class TestAzureBlobResolver:
    """Tests for _resolve_azure()."""

    @pytest.mark.asyncio
    async def test_resolve_az_downloads_to_temp_dir(self, tmp_path: Path) -> None:
        """Mock Azure client and verify files are downloaded to temp dir."""
        mock_blob_client = MagicMock()
        mock_blob_client.download_blob.return_value.readall.return_value = b"content"

        mock_container_client = MagicMock()
        # list_blobs returns blob objects with name and size attributes
        blob1 = MagicMock()
        blob1.name = "prefix/doc.txt"
        blob1.size = 100
        blob2 = MagicMock()
        blob2.name = "prefix/notes.md"
        blob2.size = 200
        mock_container_client.list_blobs.return_value = [blob1, blob2]
        mock_container_client.get_blob_client.return_value = mock_blob_client

        mock_service = MagicMock()
        mock_service.get_container_client.return_value = mock_container_client

        temp_dir = tmp_path / "holodeck-init-test"
        temp_dir.mkdir()

        with (
            patch(
                "holodeck.lib.source_resolver._import_azure_blob",
                return_value=MagicMock(
                    BlobServiceClient=MagicMock(
                        from_connection_string=MagicMock(return_value=mock_service)
                    )
                ),
            ),
            patch("tempfile.mkdtemp", return_value=str(temp_dir)),
            patch.dict(os.environ, {"AZURE_STORAGE_CONNECTION_STRING": "fake-conn"}),
        ):
            result = await SourceResolver.resolve("az://mycontainer/prefix/")

        assert result.is_remote is True
        assert result.temp_dir == temp_dir
        assert result.local_path == temp_dir
        assert mock_blob_client.download_blob.call_count == 2

    @pytest.mark.asyncio
    async def test_resolve_az_missing_sdk_raises(self) -> None:
        """Missing azure.storage.blob raises SourceError."""
        with (
            patch(
                "holodeck.lib.source_resolver._import_azure_blob",
                side_effect=SourceError(
                    "Azure source requires azure-storage-blob. "
                    "Install with: pip install holodeck[azure-blob]"
                ),
            ),
            pytest.raises(SourceError, match="azure-storage-blob"),
        ):
            await SourceResolver.resolve("az://mycontainer/prefix/")

    @pytest.mark.asyncio
    async def test_resolve_az_validates_total_size(self) -> None:
        """Objects exceeding limit raise SourceError."""
        mock_container_client = MagicMock()
        blob1 = MagicMock()
        blob1.name = "prefix/huge.txt"
        blob1.size = 1_000_000_000
        mock_container_client.list_blobs.return_value = [blob1]

        mock_service = MagicMock()
        mock_service.get_container_client.return_value = mock_container_client

        with (
            patch(
                "holodeck.lib.source_resolver._import_azure_blob",
                return_value=MagicMock(
                    BlobServiceClient=MagicMock(
                        from_connection_string=MagicMock(return_value=mock_service)
                    )
                ),
            ),
            patch.dict(os.environ, {"AZURE_STORAGE_CONNECTION_STRING": "fake-conn"}),
            pytest.raises(SourceError, match="[Ss]ize"),
        ):
            await SourceResolver.resolve(
                "az://mycontainer/prefix/", max_source_size_bytes=1024
            )

    @pytest.mark.asyncio
    async def test_resolve_az_uses_connection_string_env(self, tmp_path: Path) -> None:
        """Uses AZURE_STORAGE_CONNECTION_STRING env var."""
        mock_blob_client = MagicMock()
        mock_blob_client.download_blob.return_value.readall.return_value = b"data"

        mock_container_client = MagicMock()
        mock_container_client.list_blobs.return_value = []  # no blobs

        mock_service = MagicMock()
        mock_service.get_container_client.return_value = mock_container_client

        mock_from_conn = MagicMock(return_value=mock_service)

        temp_dir = tmp_path / "holodeck-init-test"
        temp_dir.mkdir()

        with (
            patch(
                "holodeck.lib.source_resolver._import_azure_blob",
                return_value=MagicMock(
                    BlobServiceClient=MagicMock(from_connection_string=mock_from_conn)
                ),
            ),
            patch("tempfile.mkdtemp", return_value=str(temp_dir)),
            patch.dict(
                os.environ, {"AZURE_STORAGE_CONNECTION_STRING": "my-conn-string"}
            ),
        ):
            await SourceResolver.resolve("az://mycontainer/prefix/")

        mock_from_conn.assert_called_once_with("my-conn-string")


# ---------------------------------------------------------------------------
# T022 — HttpResolver
# ---------------------------------------------------------------------------


def _make_http_stream_context(response: MagicMock) -> MagicMock:
    """Create a mock that works as an async context manager for client.stream()."""
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=response)
    ctx.__aexit__ = AsyncMock(return_value=False)
    return ctx


class TestHttpResolver:
    """Tests for _resolve_http()."""

    @pytest.mark.asyncio
    async def test_resolve_http_downloads_single_file(self, tmp_path: Path) -> None:
        """Mock httpx and verify file downloaded to temp dir."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-length": "100"}
        mock_response.raise_for_status = MagicMock()

        async def _aiter_bytes() -> None:  # type: ignore[return]
            yield b"hello world"

        mock_response.aiter_bytes = _aiter_bytes

        mock_client = AsyncMock()
        mock_client.stream = MagicMock(
            return_value=_make_http_stream_context(mock_response)
        )

        temp_dir = tmp_path / "holodeck-init-test"
        temp_dir.mkdir()

        with (
            patch(
                "holodeck.lib.source_resolver.httpx.AsyncClient",
                return_value=mock_client,
            ),
            patch("tempfile.mkdtemp", return_value=str(temp_dir)),
        ):
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)

            result = await SourceResolver.resolve("https://example.com/data/report.txt")

        assert result.is_remote is True
        assert result.temp_dir == temp_dir

    @pytest.mark.asyncio
    async def test_resolve_http_content_length_check(self) -> None:
        """Content-Length exceeding max raises before download."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-length": "999999999"}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.stream = MagicMock(
            return_value=_make_http_stream_context(mock_response)
        )

        with patch(
            "holodeck.lib.source_resolver.httpx.AsyncClient",
            return_value=mock_client,
        ):
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)

            with pytest.raises(SourceError, match="[Ss]ize|[Ee]xceed|[Ll]arge"):
                await SourceResolver.resolve(
                    "https://example.com/big.txt",
                    max_source_size_bytes=1024,
                )

    @pytest.mark.asyncio
    async def test_resolve_http_5xx_retries(self) -> None:
        """500 error is retried up to 3 times."""
        import httpx

        error_response = MagicMock()
        error_response.status_code = 500

        mock_stream_response = MagicMock()
        mock_stream_response.raise_for_status = MagicMock(
            side_effect=httpx.HTTPStatusError(
                "500", request=MagicMock(), response=error_response
            )
        )

        mock_client = AsyncMock()
        mock_client.stream = MagicMock(
            return_value=_make_http_stream_context(mock_stream_response)
        )

        with (
            patch(
                "holodeck.lib.source_resolver.httpx.AsyncClient",
                return_value=mock_client,
            ),
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)

            with pytest.raises(SourceError, match="[Ff]ailed|[Rr]etry|5[0-9][0-9]"):
                await SourceResolver.resolve("https://example.com/data.txt")

    @pytest.mark.asyncio
    async def test_resolve_http_4xx_no_retry(self) -> None:
        """404 fails immediately without retrying."""
        import httpx

        error_response = MagicMock()
        error_response.status_code = 404

        mock_stream_response = MagicMock()
        mock_stream_response.raise_for_status = MagicMock(
            side_effect=httpx.HTTPStatusError(
                "404", request=MagicMock(), response=error_response
            )
        )

        mock_client = AsyncMock()
        mock_client.stream = MagicMock(
            return_value=_make_http_stream_context(mock_stream_response)
        )

        with patch(
            "holodeck.lib.source_resolver.httpx.AsyncClient",
            return_value=mock_client,
        ):
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)

            with pytest.raises(SourceError, match="[Ff]ailed|404"):
                await SourceResolver.resolve("https://example.com/missing.txt")

    @pytest.mark.asyncio
    async def test_resolve_http_auth_header_from_env(self, tmp_path: Path) -> None:
        """HOLODECK_HTTP_AUTH_HEADER env var is included in request."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-length": "5"}
        mock_response.raise_for_status = MagicMock()

        async def _aiter_bytes() -> None:  # type: ignore[return]
            yield b"hello"

        mock_response.aiter_bytes = _aiter_bytes

        mock_client = AsyncMock()
        mock_client.stream = MagicMock(
            return_value=_make_http_stream_context(mock_response)
        )

        temp_dir = tmp_path / "holodeck-init-test"
        temp_dir.mkdir()

        with (
            patch(
                "holodeck.lib.source_resolver.httpx.AsyncClient",
                return_value=mock_client,
            ),
            patch("tempfile.mkdtemp", return_value=str(temp_dir)),
            patch.dict(
                os.environ,
                {"HOLODECK_HTTP_AUTH_HEADER": "Bearer my-secret-token"},
            ),
        ):
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)

            await SourceResolver.resolve("https://example.com/data.txt")

        # Verify the auth header was passed in the stream call
        call_args = mock_client.stream.call_args
        assert call_args is not None
        # Check headers kwarg
        headers = call_args.kwargs.get("headers", call_args[1].get("headers", {}))
        assert headers.get("Authorization") == "Bearer my-secret-token"


# ---------------------------------------------------------------------------
# T025 — cleanup_orphans
# ---------------------------------------------------------------------------


class TestCleanupOrphans:
    """Tests for SourceResolver.cleanup_orphans()."""

    @pytest.mark.asyncio
    async def test_removes_stale_dirs(self, tmp_path: Path) -> None:
        """Old holodeck-init-* directories are removed."""
        stale_dir = tmp_path / "holodeck-init-old"
        stale_dir.mkdir()
        # Set mtime to 2 hours ago
        old_time = time.time() - 7200
        os.utime(str(stale_dir), (old_time, old_time))

        with patch("tempfile.gettempdir", return_value=str(tmp_path)):
            removed = await SourceResolver.cleanup_orphans(max_age_hours=1.0)

        assert removed == 1
        assert not stale_dir.exists()

    @pytest.mark.asyncio
    async def test_preserves_fresh_dirs(self, tmp_path: Path) -> None:
        """Fresh holodeck-init-* directories are preserved."""
        fresh_dir = tmp_path / "holodeck-init-fresh"
        fresh_dir.mkdir()
        # mtime is current, so it should be preserved

        with patch("tempfile.gettempdir", return_value=str(tmp_path)):
            removed = await SourceResolver.cleanup_orphans(max_age_hours=1.0)

        assert removed == 0
        assert fresh_dir.exists()

    @pytest.mark.asyncio
    async def test_ignores_non_matching_dirs(self, tmp_path: Path) -> None:
        """Directories without the prefix are untouched."""
        other_dir = tmp_path / "some-other-dir"
        other_dir.mkdir()
        old_time = time.time() - 7200
        os.utime(str(other_dir), (old_time, old_time))

        with patch("tempfile.gettempdir", return_value=str(tmp_path)):
            removed = await SourceResolver.cleanup_orphans(max_age_hours=1.0)

        assert removed == 0
        assert other_dir.exists()

    @pytest.mark.asyncio
    async def test_returns_count(self, tmp_path: Path) -> None:
        """Returns correct integer count of removed directories."""
        old_time = time.time() - 7200
        for i in range(3):
            d = tmp_path / f"holodeck-init-stale{i}"
            d.mkdir()
            os.utime(str(d), (old_time, old_time))

        # Also one fresh one
        fresh = tmp_path / "holodeck-init-fresh"
        fresh.mkdir()

        with patch("tempfile.gettempdir", return_value=str(tmp_path)):
            removed = await SourceResolver.cleanup_orphans(max_age_hours=1.0)

        assert removed == 3
        assert fresh.exists()
