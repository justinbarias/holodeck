# Design: Remote Source Resolution for Tool Initialization

**Feature Branch**: `025-tool-init-endpoints`
**Date**: 2026-03-23
**Status**: Approved
**Last Updated**: 2026-03-24 (verification pass #2)

## Context

The async tool initialization endpoints (`POST /tools/{tool-name}/init`) trigger ingestion/indexing of vectorstore and hierarchical_document tools. In container-based production deployments, data sources are not packaged with the agent image — they live in cloud storage (S3, Azure Blob) or are served via HTTP(S). The `source` field in tool config must support remote URIs in addition to local paths.

## Architecture

### Source Resolution Pipeline

```
tool.source (string)
  → SourceResolver.resolve_context(source, base_dir)  [async context manager]
    ├─ No scheme / file:// → LocalResolver → returns Path directly
    ├─ s3://              → S3Resolver → boto3 download → temp dir Path
    ├─ az://              → AzureBlobResolver → azure-storage-blob download → temp dir Path
    └─ https:// / http:// → HttpResolver → httpx download → temp dir Path
  → ResolvedSource(local_path, is_remote, temp_dir)
  → [cleanup guaranteed on exit, even on exception]
```

Everything downstream of `ResolvedSource.local_path` is unchanged — file discovery, processing, chunking, embedding, and storage all operate on local paths.

### ResolvedSource Dataclass

```python
@dataclass
class ResolvedSource:
    local_path: Path          # Directory or file to process
    is_remote: bool           # Whether cleanup is needed
    temp_dir: Path | None     # Temp dir to clean up (None for local)
```

### SourceResolver Class

```python
class SourceResolver:
    # Configurable limits
    max_source_size_bytes: int = 1_073_741_824  # 1 GB default

    @staticmethod
    async def resolve(source: str, base_dir: str | None = None) -> ResolvedSource:
        """Resolve a source string to a local path, downloading if remote."""
        scheme = _detect_scheme(source)
        resolver = _RESOLVERS[scheme]  # LocalResolver, S3Resolver, etc.
        return await resolver.resolve(source, base_dir)

    @staticmethod
    @asynccontextmanager
    async def resolve_context(
        source: str, base_dir: str | None = None
    ) -> AsyncGenerator[ResolvedSource, None]:
        """Context manager that resolves source and guarantees cleanup."""
        resolved = await SourceResolver.resolve(source, base_dir)
        try:
            yield resolved
        finally:
            await SourceResolver.cleanup(resolved)

    @staticmethod
    async def cleanup(resolved: ResolvedSource) -> None:
        """Remove temp directory if source was remote. Non-blocking."""
        if resolved.is_remote and resolved.temp_dir:
            await asyncio.to_thread(shutil.rmtree, resolved.temp_dir, True)

    @staticmethod
    async def cleanup_orphans(prefix: str = "holodeck-init-", max_age_hours: int = 1) -> int:
        """Remove stale temp dirs from previous runs. Called on server startup."""
        import tempfile
        tmp_root = Path(tempfile.gettempdir())
        cutoff = datetime.now() - timedelta(hours=max_age_hours)
        removed = 0
        for entry in tmp_root.iterdir():
            if entry.is_dir() and entry.name.startswith(prefix):
                if datetime.fromtimestamp(entry.stat().st_mtime) < cutoff:
                    await asyncio.to_thread(shutil.rmtree, entry, True)
                    removed += 1
        return removed
```

## Provider Implementations

### LocalResolver

Extracts and wraps existing `resolve_source_path()` logic from `tools/common.py`:
- Absolute paths used directly
- Relative paths resolved against `base_dir` or `agent_base_dir` context
- Returns `ResolvedSource(local_path=resolved, is_remote=False, temp_dir=None)`

### S3Resolver

- **SDK**: `boto3` (optional dependency, lazy import)
- **URI format**: `s3://bucket-name/prefix/path/`
- **Auth**: Standard boto3 credential chain (env vars → IAM role → config file → profile)
- **Behavior**:
  1. Parse bucket and prefix from URI
  2. `s3.list_objects_v2(Bucket=bucket, Prefix=prefix)` to enumerate objects
  3. Filter by supported extensions (`.txt`, `.md`, `.pdf`, `.csv`, `.json`)
  4. **Pre-download size check**: Sum object sizes from listing; reject if total exceeds `max_source_size_bytes`
  5. **Path validation**: For each object key, validate it does not contain `..` and that the resolved download path is within the temp dir (`Path.resolve().is_relative_to(temp_dir)`). Skip and log invalid keys.
  6. Download each valid object to temp dir preserving relative path structure
  7. Return `ResolvedSource(local_path=temp_dir, is_remote=True, temp_dir=temp_dir)`
- **Error handling**: Auth failures → immediate fail with sanitized message; network errors → retry 3x with backoff

### AzureBlobResolver

- **SDK**: `azure-storage-blob` (optional dependency, lazy import)
- **URI format**: `az://container-name/blob-prefix/`
- **Auth**: `AZURE_STORAGE_CONNECTION_STRING` env var or `DefaultAzureCredential` (managed identity)
- **Behavior**:
  1. Parse container and prefix from URI
  2. `ContainerClient.list_blobs(name_starts_with=prefix)` to enumerate blobs
  3. Filter by supported extensions
  4. **Pre-download size check**: Sum blob sizes from listing; reject if total exceeds `max_source_size_bytes`
  5. **Path validation**: Same as S3 — reject blobs with `..` in name, verify resolved path stays within temp dir
  6. Download each valid blob to temp dir preserving relative path structure
  7. Return `ResolvedSource(local_path=temp_dir, is_remote=True, temp_dir=temp_dir)`
- **Error handling**: Same pattern as S3

### HttpResolver

- **SDK**: `httpx` (core dependency, async-native)
- **URI format**: `https://example.com/path/to/file.md`
- **Auth**: No auth by default; optional `Authorization` header via env var `HOLODECK_HTTP_AUTH_HEADER`
- **Behavior**:
  - **Single file downloads only** (no archive extraction — dropped for security reasons)
  - Download to temp dir, infer filename from URL path
  - **Pre-download size check**: Read `Content-Length` header; reject if exceeds `max_source_size_bytes`. If no Content-Length, stream with running size check and abort if limit exceeded.
  - **Directory listing**: Not supported — HTTP sources must point to individual files
- **Error handling**: HTTP 4xx → fail immediately with sanitized message; HTTP 5xx / network errors → retry 3x with exponential backoff (1s, 2s, 4s)

**Note**: `httpx` is used instead of the existing `requests` library because it is async-native, fitting the asyncio architecture. The existing `requests` usage in FileProcessor remains unchanged — different concern (multimodal file processing vs. source resolution).

## Dependencies

```toml
# pyproject.toml - core dependency
"httpx>=0.27"

# pyproject.toml - optional extras
[project.optional-dependencies]
s3 = ["boto3>=1.42.0"]
azure-blob = ["azure-storage-blob>=12.19"]
all-sources = ["boto3>=1.42.0", "azure-storage-blob>=12.19"]
```

- `httpx` added to core dependencies (async-native HTTP client)
- `boto3>=1.42.0` aligns with existing `deploy-aws` extra version constraint
- Cloud SDKs are optional extras with lazy imports
- Missing SDK → `SourceError("S3 source requires boto3. Install with: pip install holodeck[s3]")`
- Note: `deploy-aws` extra already includes boto3 — installing it implicitly enables S3 source support

## Credentials

| Provider | Environment Variables | Fallback |
|----------|----------------------|----------|
| S3 | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION` | IAM role, AWS config file |
| Azure Blob | `AZURE_STORAGE_CONNECTION_STRING` | `DefaultAzureCredential` (managed identity) |
| HTTP(S) | `HOLODECK_HTTP_AUTH_HEADER` (optional) | None (public access) |

No credentials are stored in YAML config. All auth follows cloud-native environment variable conventions compatible with container orchestrators (K8s secrets, ECS task roles, etc.).

## Size Limits

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_source_size_bytes` | 1 GB (1,073,741,824) | Maximum total size of files to download per init job |

- **S3/Azure**: Computed from object listing before download begins. If total exceeds limit, init fails with a descriptive error before any download starts.
- **HTTP**: Read from `Content-Length` header. If absent, enforce limit via streaming byte counter — abort download if limit exceeded mid-stream.
- **Local paths**: No size limit (operator controls local filesystem).
- Configurable via `ToolInitManager` constructor parameter. Future: expose in agent YAML config.

## Security

### Credential Sanitization

All error messages returned to clients via `InitJobResponse.error_detail` MUST be sanitized. A `sanitize_error_detail()` utility handles this:

**Patterns redacted**:
- AWS access key IDs: `AKIA[A-Z0-9]{16}` → `<AWS_KEY_REDACTED>`
- Azure connection strings: `AccountKey=[^;]+` → `AccountKey=<REDACTED>`
- Bearer tokens: `Bearer [A-Za-z0-9._\-]+` → `Bearer <REDACTED>`
- Generic API keys: `(api[_-]?key|token|secret|password)\s*[=:]\s*\S+` → `<REDACTED>`
- Absolute file paths: `/tmp/holodeck-init-*/...` → `<TEMP_DIR>/...`

**Implementation**:
```python
def sanitize_error_detail(error: Exception) -> str:
    """Redact credentials and paths from error messages for client responses."""
    message = str(error)
    message = re.sub(r'AKIA[A-Z0-9]{16}', '<AWS_KEY_REDACTED>', message)
    message = re.sub(r'AccountKey=[^;]+', 'AccountKey=<REDACTED>', message)
    message = re.sub(r'Bearer [A-Za-z0-9._\-]+', 'Bearer <REDACTED>', message)
    message = re.sub(
        r'(api[_\-]?key|token|secret|password)\s*[=:]\s*\S+',
        r'\1=<REDACTED>', message, flags=re.IGNORECASE
    )
    message = re.sub(r'/tmp/holodeck-init-[^/\s]+', '<TEMP_DIR>', message)
    return message
```

Full unredacted errors are logged server-side at ERROR level for debugging.

### Path Traversal Prevention

All S3 object keys and Azure blob names are validated before download:

```python
def validate_object_path(key: str, temp_dir: Path) -> Path:
    """Validate and resolve download path within temp dir."""
    if ".." in key:
        raise ValueError(f"Invalid object key: path traversal detected in '{key}'")
    target = (temp_dir / key).resolve()
    if not target.is_relative_to(temp_dir.resolve()):
        raise ValueError(f"Invalid object key: resolved path escapes temp dir: '{key}'")
    return target
```

Invalid keys are logged and skipped — they do not fail the entire init job.

## Temp Directory Lifecycle

1. **Created**: When `SourceResolver.resolve()` starts downloading for a remote source
2. **Named**: `tempfile.mkdtemp(prefix=f"holodeck-init-{tool_name}-")`
3. **Used**: Passed as `local_path` to file discovery and processing pipeline
4. **Cleaned up** (guaranteed via context manager):
   - On init job completion (success or failure) — `async with SourceResolver.resolve_context()` ensures cleanup in `finally` block
   - On server shutdown via `ToolInitManager.shutdown()` (catches any missed cleanups)
   - On init job cancellation (graceful shutdown)
   - Cleanup runs via `asyncio.to_thread(shutil.rmtree, ...)` to avoid blocking the event loop
5. **Orphan recovery**: On server startup, `SourceResolver.cleanup_orphans()` scans TMPDIR for `holodeck-init-*` directories older than 1 hour and removes them. This handles SIGKILL and hard-crash scenarios.
6. **Not persisted**: No caching across init calls. Each init is a fresh download.

## Integration with Init Endpoints

The init job coroutine uses the context manager for guaranteed cleanup:

```python
async def _run_init_job(tool_name, tool_config, agent, force, progress_callback):
    async with SourceResolver.resolve_context(tool_config.source, base_dir) as resolved:
        progress_callback(message="Downloading files" if resolved.is_remote else "Processing")
        await initialize_single_tool(
            agent, tool_name, force, progress_callback,
            source_override=resolved.local_path,
        )
```

On `ToolInitManager` startup (called from `AgentServer.start()`):
```python
orphans_removed = await SourceResolver.cleanup_orphans()
if orphans_removed:
    logger.info(f"Cleaned up {orphans_removed} orphaned temp directories from previous runs")
```

## Source Field Validation

The `source` field validator in `tool.py` is updated to accept:
- Local paths (relative or absolute) — existing behavior
- `s3://bucket/prefix/` — validated as URI with bucket component
- `az://container/prefix/` — validated as URI with container component
- `https://` or `http://` — validated as URL
- Other schemes → `ValidationError`

## Testing Strategy

- **Unit tests for SourceResolver**: Mock boto3/azure-storage-blob/httpx; verify temp dir creation, file structure, cleanup
- **Unit tests for scheme detection**: Cover all URI formats and edge cases
- **Unit tests for path validation**: Traversal attempts (`../`, absolute paths in keys), oversized sources
- **Unit tests for credential sanitization**: Verify no AWS keys, Azure strings, or tokens in error output
- **Unit tests for context manager**: Exception during download → verify cleanup still runs
- **Unit tests for orphan cleanup**: Stale dirs removed, fresh dirs preserved
- **Integration tests**: Use localstack (S3) and azurite (Azure Blob) for real cloud SDK testing
- **Error case tests**: Missing SDK, auth failures, network errors, empty prefix, unsupported file types, size limit exceeded
