# Research: Claude Backend Serve & Deploy Parity

**Feature**: 024-claude-serve-deploy
**Date**: 2026-03-20

## Decision 1: Pre-Flight Validation Injection Point

**Decision**: Inject validation in `AgentServer.start()` (server.py) before transitioning to RUNNING state.

**Rationale**: The server startup flow is: `__init__()` → `create_app()` → `start()` → uvicorn. Validating in `start()` ensures:
- FastAPI routes are registered (health endpoint works even on failure)
- Validation runs once at startup, not per-request
- Failure blocks RUNNING state transition (fail-fast)
- Clean error reporting before uvicorn takes over stdout

**Alternatives considered**:
- In `serve.py` between server creation and `create_app()` — too early, routes not yet registered
- In `create_app()` before `READY` state — conflates app creation with backend validation
- At first request time (current behavior) — unacceptable UX, spec requires fail-fast

**Implementation**: New `async def _validate_backend_prerequisites()` method on `AgentServer` that calls the existing `validate_nodejs()`, `validate_credentials()`, etc. from `validators.py`. Called from `start()` before `self.state = ServerState.RUNNING`.

## Decision 2: Dockerfile Node.js Strategy

**Decision**: Conditional Node.js installation in the Jinja2 Dockerfile template, controlled by a `needs_nodejs` boolean flag passed from `_generate_dockerfile_content()`.

**Rationale**: The existing `generate_dockerfile()` function accepts template variables. Adding a `needs_nodejs: bool` parameter is the minimal change that keeps the template self-contained. The deploy command detects `agent.model.provider == ProviderEnum.ANTHROPIC` and sets the flag.

**Alternatives considered**:
- Separate Claude-specific base image — increases maintenance burden, diverges from single-image pattern
- Always install Node.js in base image — adds ~60MB to all images including non-Claude agents
- Multi-stage build with Node.js layer — over-engineered for adding a runtime dependency

**Implementation**: Add conditional block in `HOLODECK_DOCKERFILE_TEMPLATE`:
```dockerfile
{% if needs_nodejs %}
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*
{% endif %}
```

## Decision 3: Entrypoint Fix Approach

**Decision**: Fix the inline entrypoint in `_prepare_build_context()` (deploy.py lines 641-651) — this is the entrypoint actually used in builds.

**Rationale**: Two entrypoints exist:
1. `docker/entrypoint.sh` — full validation, startup banner, `--config` bug (used by base image only)
2. Inline entrypoint in `_prepare_build_context()` — simpler, correct positional arg, used by `deploy build`

The inline entrypoint is what matters for `deploy build`. The `docker/entrypoint.sh` bug should also be fixed but is used only when running the base image directly. Both will be fixed.

**Alternatives considered**:
- Only fix `docker/entrypoint.sh` — wouldn't fix the actual deploy build path
- Merge into single entrypoint — breaks the build context isolation pattern

## Decision 4: Session Cap Configuration Model

**Decision**: Add `max_concurrent_sessions` field to `ClaudeConfig` rather than creating a new `ServeConfig` class.

**Rationale**: The concurrent session cap is Claude-specific (each session spawns a Node.js subprocess). Other backends don't have this constraint. Putting it in `ClaudeConfig` keeps it scoped to where it matters and follows the existing pattern where Claude-specific settings live in the `claude:` YAML section.

**Alternatives considered**:
- New `ServeConfig` class on Agent model — adds a generic config section for what is currently a Claude-specific concern
- CLI flag on `holodeck serve` — not declarative, doesn't follow no-code-first principle
- `SessionStore.max_sessions` override — already exists (default 1000) but is about total sessions, not concurrent Claude subprocesses

**Implementation**: Add `max_concurrent_sessions: int | None = Field(default=10, ge=1, le=100)` to `ClaudeConfig`.

## Decision 5: Health Check Enhancement Pattern

**Decision**: Add backend-specific health fields to `HealthResponse` and a `check_backend_health()` method to `AgentServer` that delegates to provider-specific validators.

**Rationale**: The existing `HealthResponse` model (serve/models.py) has `status`, `agent_ready`, `active_sessions`, `uptime_seconds`. Adding `backend_ready: bool` and `backend_diagnostics: list[str]` fields provides actionable information without changing the response structure for existing consumers.

**Alternatives considered**:
- Separate `/health/backend` endpoint — fragments health checking, doesn't integrate with existing orchestrator patterns
- Detailed per-check fields — over-specific, would need updating for each new backend

## Decision 6: Node.js Version Validation

**Decision**: Enhance `validate_nodejs()` to check version >= 18 by parsing `node --version` output.

**Rationale**: Claude Agent SDK 0.1.44 requires Node.js 18+. The current validator only checks binary existence. Adding version checking prevents cryptic runtime failures when an old Node.js is installed.

**Alternatives considered**:
- Hard-code minimum version constant — chosen, simple and correct
- Query Claude Agent SDK for its requirement — SDK doesn't expose this programmatically
- Skip version check — risk of confusing errors when old Node.js fails silently

## Decision 7: OTel Instrumentation in Serve Mode

**Decision**: No new code needed — GenAI instrumentation activates automatically via `ClaudeBackend.initialize()` → `_activate_instrumentation()` and works in serve mode by construction.

**Rationale**: The serve path uses `AgentExecutor` → `BackendSelector.select()` → `ClaudeBackend(...)` → `await backend.initialize()`. The `initialize()` method calls `_activate_instrumentation()` (claude_backend.py line 638) which instruments the SDK with the OTel tracer provider. Since initialization happens once per backend instance and the instrumentor monkey-patches at the module level, all subsequent `query()` calls emit GenAI-convention spans regardless of whether they're invoked from test, chat, or serve mode.

**Verification required**: Add an integration test confirming OTel spans are emitted when a Claude agent handles a request in serve mode. The `_patch_hooks_for_context_propagation()` function (claude_backend.py line 328) works around a ContextVar timing issue — verify this works correctly with the serve session lifecycle.

**Alternatives considered**:
- Explicit OTel setup in serve startup — unnecessary duplication of what `initialize()` already does
- Separate serve-specific instrumentor configuration — over-engineering; same instrumentor works for all modes
