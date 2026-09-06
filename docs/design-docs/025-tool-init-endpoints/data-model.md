# Data Model: Async Tool Initialization Endpoints

**Feature Branch**: `025-tool-init-endpoints`
**Date**: 2026-03-23

## Entities

### InitJobState (Enum)

Represents the lifecycle state of a tool initialization job.

| Value | Description |
|-------|-------------|
| `pending` | Job created, waiting to start |
| `in_progress` | Initialization is actively running |
| `completed` | Initialization finished successfully |
| `failed` | Initialization encountered an error |

**State transitions**:
```
pending → in_progress → completed
                      → failed
```

Note: `pending` → `in_progress` happens immediately when the background task starts executing. There is no queue — if a slot is available, the job starts immediately; if not, the request is rejected with 429.

### InitJobProgress

Tracks granular progress of document ingestion.

| Field | Type | Description |
|-------|------|-------------|
| `documents_processed` | `int` | Number of documents processed so far |
| `total_documents` | `int \| None` | Total documents to process (null if not yet determined) |

### InitJob

Represents an asynchronous initialization task for a single tool.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `tool_name` | `str` | Yes | Tool identifier (matches agent config `tools[].name`) |
| `state` | `InitJobState` | Yes | Current lifecycle state |
| `created_at` | `datetime` | Yes | UTC timestamp when job was created |
| `started_at` | `datetime \| None` | No | UTC timestamp when initialization began |
| `completed_at` | `datetime \| None` | No | UTC timestamp when initialization finished (success or failure) |
| `message` | `str \| None` | No | Human-readable status message |
| `error_detail` | `str \| None` | No | Error description (only when state=failed) |
| `progress` | `InitJobProgress \| None` | No | Document processing progress (when available) |
| `force` | `bool` | Yes | Whether force re-ingestion was requested |

**Validation rules**:
- `tool_name` must match `^[a-zA-Z_][a-zA-Z0-9_]*$` (same as tool config name pattern)
- `started_at` must be null when state is `pending`
- `completed_at` must be null when state is `pending` or `in_progress`
- `error_detail` must be null when state is not `failed`
- `created_at <= started_at <= completed_at` when all are present

**Relationships**:
- References `Tool` by `tool_name` (lookup from `Agent.tools[]`)
- One active `InitJob` per tool at a time (enforced by `ToolInitManager`). Re-init POST returns 409 Conflict when the existing job is in `pending` OR `in_progress` state — both are considered "active" from the client's perspective.

### ToolInfo

Summary of a configured tool for the `GET /tools` listing.

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Tool name from agent config |
| `type` | `str` | Tool type (`vectorstore`, `hierarchical_document`, `function`, `mcp`, `prompt`) |
| `supports_init` | `bool` | Whether this tool type supports initialization |
| `init_status` | `InitJobState \| None` | Current init state (null if no init job exists and tool supports init) |

**Derivation rules**:
- `supports_init = type in ("vectorstore", "hierarchical_document")`
- `init_status` is derived from the latest `InitJob` for this tool (if any)
- For tools that don't support init, `init_status` is always null

### ToolInitManager (Service)

In-memory orchestrator for initialization jobs. Not a data model — included here to document its state management.

| Field | Type | Description |
|-------|------|-------------|
| `_jobs` | `dict[str, InitJob]` | Active/completed jobs keyed by tool name |
| `_tasks` | `dict[str, asyncio.Task]` | Background task handles keyed by tool name |
| `_active_count` | `int` | Number of currently running jobs |
| `max_concurrent` | `int` | Maximum concurrent init jobs (default: 3, configurable) |
| `_shutting_down` | `bool` | Whether shutdown has been initiated |

**Invariants**:
- `_active_count == len([j for j in _jobs.values() if j.state == InitJobState.IN_PROGRESS])`
- `_active_count <= max_concurrent`
- At most one `InitJob` per `tool_name` in `_jobs` (latest replaces previous on re-init)
- `_tasks` keys are a subset of `_jobs` keys (only for pending/in_progress jobs)

## Response Models (Pydantic)

### InitJobResponse

Returned by `POST /tools/{tool-name}/init` (201) and `GET /tools/{tool-name}/init` (200).

```python
class InitJobResponse(BaseModel):
    tool_name: str
    state: InitJobState
    href: str                          # "/tools/{tool-name}/init"
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None
    message: str | None
    error_detail: str | None
    progress: InitJobProgress | None
    force: bool
```

### ToolInfoResponse

Single tool entry in the `GET /tools` listing.

```python
class ToolInfoResponse(BaseModel):
    name: str
    type: str
    supports_init: bool
    init_status: InitJobState | None
```

### ToolListResponse

Returned by `GET /tools` (200).

```python
class ToolListResponse(BaseModel):
    tools: list[ToolInfoResponse]
    total: int
```

### ResolvedSource (Dataclass)

Result of resolving a tool's `source` field (local path or remote URI) to a local directory.

| Field | Type | Description |
|-------|------|-------------|
| `local_path` | `Path` | Local directory or file path to process |
| `is_remote` | `bool` | Whether the source was downloaded from a remote URI |
| `temp_dir` | `Path \| None` | Temp directory to clean up (None for local sources) |

**Lifecycle**:
- Created by `SourceResolver.resolve()` at the start of an init job
- `local_path` is passed to the existing file discovery and processing pipeline
- `temp_dir` is cleaned up via `SourceResolver.cleanup()` when the init job completes, fails, or is cancelled

**Supported source schemes**:
- (none) / `file://` — Local path, resolved against `base_dir`
- `s3://bucket/prefix/` — AWS S3, downloaded via boto3
- `az://container/prefix/` — Azure Blob Storage, downloaded via azure-storage-blob
- `https://` / `http://` — HTTP(S), downloaded via httpx (single file or archive)

## Storage

All state is held **in-memory** on the `ToolInitManager` instance:
- No database tables or persistent storage
- State is lost on server restart (consistent with existing `SessionStore` pattern)
- No migration or schema versioning needed
- Remote source temp directories are cleaned up on job completion or server shutdown
