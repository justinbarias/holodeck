# Implementation Plan: Claude Backend Serve & Deploy Parity

**Branch**: `024-claude-serve-deploy` | **Date**: 2026-03-20 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/024-claude-serve-deploy/spec.md`

## Summary

Bring the Claude Agent SDK backend to full serve/deploy parity with the Semantic Kernel backend. The protocol-level abstraction (AgentBackend/AgentSession) is already backend-agnostic — this feature addresses the practical gaps: serve pre-flight validation, Dockerfile Node.js injection, container entrypoint fixes, backend-aware health checks, subprocess lifecycle management, and integration test coverage.

## Technical Context

**Language/Version**: Python 3.10+
**Primary Dependencies**: claude-agent-sdk==0.1.44, FastAPI (serve), Docker SDK (deploy), Jinja2 (Dockerfile template)
**Storage**: N/A (in-memory session management)
**Testing**: pytest with async support, pytest-xdist for parallel execution
**Target Platform**: Linux server (containers), macOS/Linux (local serve)
**Project Type**: Single project (existing HoloDeck monorepo)
**Performance Goals**: Pre-flight validation < 5 seconds; 10 concurrent Claude sessions per instance (configurable)
**Constraints**: Node.js >= 18 required at runtime for Claude backend; container images must support `--cap-drop ALL --read-only`
**Scale/Scope**: 10 source files modified, 2 new test files, 1 new fixture, ~450 LOC additions

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| I. No-Code-First | PASS | Session cap configured via YAML (`claude.max_concurrent_sessions`), no code required |
| II. MCP for APIs | PASS | No new API integrations; Claude SDK uses subprocess protocol |
| III. Test-First | PASS | Integration tests for serve/deploy included as P3 story; unit tests for all new validators |
| IV. OTel-Native | PASS | FR-013 requires preservation of GenAI semantic convention instrumentation in serve mode |
| V. Evaluation Flexibility | N/A | No evaluation changes in this feature |
| Architecture: 3 Engines Decoupled | PASS | Changes span Agent Engine (validators) and Deployment Engine (Dockerfile) but use existing contracts |
| Code Quality | PASS | All new code follows existing patterns (Pydantic models, pytest markers, type hints) |

**Post-design re-check**: All gates still pass. `max_concurrent_sessions` on `ClaudeConfig` follows existing nested-config pattern (matches `SubagentConfig.max_parallel`). Health endpoint extension is backward-compatible.

## Project Structure

### Documentation (this feature)

```text
specs/024-claude-serve-deploy/
├── plan.md              # This file
├── spec.md              # Feature specification
├── research.md          # Phase 0: Research decisions
├── data-model.md        # Phase 1: Entity changes
├── quickstart.md        # Phase 1: Integration scenarios
├── contracts/
│   ├── health-endpoint.md   # Enhanced health response contract
│   └── session-cap.md       # Capacity error response contract
├── checklists/
│   └── requirements.md      # Spec quality checklist
└── tasks.md             # Phase 2 output (/speckit.tasks)
```

### Source Code (repository root)

```text
src/holodeck/
├── models/
│   └── claude_config.py         # MODIFY: Add max_concurrent_sessions field
├── lib/
│   └── backends/
│       └── validators.py        # MODIFY: Enhance validate_nodejs() with version check
├── serve/
│   ├── server.py                # MODIFY: Add _validate_backend_prerequisites(), enhance health endpoint
│   ├── models.py                # MODIFY: Add backend_ready + backend_diagnostics to HealthResponse
│   └── protocols/
│       ├── rest.py              # MODIFY: Map BackendSessionError → 502, capacity → 503
│       └── agui.py              # MODIFY: Map BackendSessionError → error SSE event
├── deploy/
│   └── dockerfile.py            # MODIFY: Add needs_nodejs template variable + conditional Node.js install
├── cli/
│   └── commands/
│       └── deploy.py            # MODIFY: Detect provider in _generate_dockerfile_content()
docker/
├── Dockerfile                   # MODIFY: Add conditional Node.js layer (base image)
└── entrypoint.sh                # MODIFY: Fix --config bug, add Claude-specific validation

tests/
├── unit/
│   ├── lib/backends/
│   │   └── test_validators.py   # MODIFY: Add Node.js version validation tests
│   ├── serve/
│   │   └── test_server.py       # MODIFY: Add backend validation + health check tests
│   └── deploy/
│       └── test_dockerfile.py   # MODIFY: Add needs_nodejs template tests
├── integration/
│   ├── serve/
│   │   └── test_server_claude.py    # NEW: Claude backend serve integration tests
│   └── deploy/
│       └── test_build_claude.py     # NEW: Claude backend deploy build tests
└── fixtures/
    └── deploy/
        └── claude_agent/
            └── agent.yaml           # NEW: Claude agent fixture for deploy tests
```

**Structure Decision**: Single project (existing monorepo). All changes fit within the established `src/holodeck/` and `tests/` layout. No new top-level directories needed.

## Architecture

### Phase 1: Serve Pre-Flight Validation (FR-001, FR-006, FR-008, FR-012, FR-013)

```
serve.py → _run_server() → AgentServer.start()
                              ↓
                    NEW: _validate_backend_prerequisites()
                              ↓
                    Detect provider from agent_config.model.provider
                              ↓
              ┌── ANTHROPIC ──────────────────────┐
              │  validate_nodejs()  ← enhanced    │
              │  validate_credentials(model)      │
              │  validate_embedding_provider()    │
              │  validate_response_format()       │
              └──────────────────────────────────┘
                              ↓
                    Pass → state = RUNNING → uvicorn starts
                    Fail → log error → sys.exit(1)
```

**Key decision**: Validation runs in `AgentServer.start()` before RUNNING state. On failure, the server exits with a clear error message — it never transitions to RUNNING.

**FR-006 (env var pass-through)**: Proxy and credential env vars (`ANTHROPIC_BASE_URL`, `HTTP_PROXY`, `HTTPS_PROXY`) are provided by the operator at `docker run` time and inherited by child processes. No explicit forwarding code is needed. However, verify that `ClaudeAgentOptions.env` (constructed in `build_options()` at claude_backend.py:272) **merges with** rather than **replaces** the inherited environment — add a unit test to confirm proxy env vars reach the SDK subprocess.

**FR-013 (OTel instrumentation)**: GenAI semantic convention instrumentation activates automatically via `ClaudeBackend.initialize()` → `_activate_instrumentation()`. This works in serve mode because `AgentExecutor` calls `BackendSelector.select()` which triggers `initialize()`. Add an integration test to verify OTel spans are emitted when serving Claude agents.

### Phase 2: Backend-Aware Health Checks (FR-005)

```
GET /health
    ↓
HealthResponse(
    status = "healthy" | "degraded" | "unhealthy",
    ...existing fields...,
    backend_ready = bool,           ← NEW
    backend_diagnostics = [str],    ← NEW
)
```

Health checks reuse the same validators called at startup. Results are cached for 30 seconds to avoid per-request overhead from `node --version` calls.

### Phase 3: Dockerfile Generation (FR-002, FR-003, FR-007)

```
deploy build agent.yaml
    ↓
_generate_dockerfile_content(agent, deployment_config, version)
    ↓
Detect: agent.model.provider == ProviderEnum.ANTHROPIC
    ↓
generate_dockerfile(..., needs_nodejs=True)
    ↓
Jinja2 template renders:
  {% if needs_nodejs %}
  RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
      && apt-get install -y --no-install-recommends nodejs \
      && rm -rf /var/lib/apt/lists/*
  {% endif %}
```

### Phase 4: Entrypoint Fixes (FR-004, FR-011)

Two entrypoints exist in the codebase — they serve different purposes:

**Inline entrypoint** (deploy.py `_prepare_build_context()`, lines 642-650):
- Used by `holodeck deploy build` output — this is what deployed containers actually run
- Already correct: uses positional arg (`holodeck serve /app/agent.yaml`)
- Add Claude validation: check `node --version` and credential env vars before `exec holodeck serve`

**docker/entrypoint.sh** (base image, line 111):
- Used only when running the base Docker image directly (not via `deploy build`)
- Has `--config` bug: passes `--config ${HOLODECK_AGENT_CONFIG}` but `holodeck serve` uses `@click.argument` (positional), not `--config` option
- Fix: change to positional arg `holodeck serve ${HOLODECK_AGENT_CONFIG} --port ... --protocol ...`
- Add `validate_claude_requirements()` function for Node.js + credentials

### Phase 5: Session Cap & Crash Handling (FR-014, EC-2)

```
AgentServer.__init__()
    ↓
Read claude.max_concurrent_sessions from agent_config
    ↓
Pass to SessionStore(max_sessions=cap)
    ↓
SessionStore.create() checks len(sessions) < max_sessions
    ↓
Exceeded → raise RuntimeError → endpoint returns 503
```

**Key detail**: The existing `SessionStore.max_sessions` (default 1000) is repurposed. When `provider: anthropic`, it's overridden by `claude.max_concurrent_sessions` (default 10). Non-Claude backends keep the 1000 default.

**Crash handling**: When `ClaudeSession.send()` raises `BackendSessionError` (subprocess crash), the protocol handlers (`serve/protocols/rest.py`, `serve/protocols/agui.py`) must catch this and return the contract-defined 502 response with `{"error": "backend_error", "retriable": true}`. The crashed session is removed from SessionStore, freeing the slot.

### Edge Case Mitigations

- **EC-3 (tmpfs fills up)**: Mitigated by operator-configurable tmpfs size at `docker run` time (e.g., `--tmpfs /tmp:size=500m`). Not a code concern — document in deployment guide.
- **EC-4 (ANTHROPIC_BASE_URL unreachable)**: The Claude SDK raises `CLIConnectionError` which propagates through existing `BackendSessionError` handling. No new code needed — the error surfaces through standard error paths.

## File Change Summary

| File | Change | FR Coverage |
|------|--------|-------------|
| `models/claude_config.py` | Add `max_concurrent_sessions` field | FR-014 |
| `lib/backends/validators.py` | Add Node.js version check to `validate_nodejs()` | FR-012 |
| `serve/server.py` | Add `_validate_backend_prerequisites()`, enhance health endpoint, pass session cap | FR-001, FR-005, FR-008, FR-014 |
| `serve/models.py` | Add `backend_ready`, `backend_diagnostics` to `HealthResponse` | FR-005 |
| `deploy/dockerfile.py` | Add `needs_nodejs` param + conditional template block | FR-002, FR-003 |
| `cli/commands/deploy.py` | Detect provider, pass `needs_nodejs` flag | FR-002, FR-007 |
| `docker/Dockerfile` | Add Node.js layer (base image) | FR-002 |
| `docker/entrypoint.sh` | Fix `--config` bug, add Claude validation | FR-004, FR-011 |
| `serve/protocols/rest.py` | Map BackendSessionError → 502, capacity → 503 | FR-014 |
| `serve/protocols/agui.py` | Map BackendSessionError → error SSE event | FR-014 |
| `tests/unit/lib/backends/test_validators.py` | Node.js version tests | FR-012 |
| `tests/unit/serve/test_server.py` | Pre-flight + health tests | FR-001, FR-005 |
| `tests/unit/deploy/test_dockerfile.py` | needs_nodejs template tests | FR-002 |
| `tests/integration/serve/test_server_claude.py` | **NEW**: Claude serve integration | FR-010 |
| `tests/integration/deploy/test_build_claude.py` | **NEW**: Claude deploy build integration | FR-010 |
| `tests/fixtures/deploy/claude_agent/agent.yaml` | **NEW**: Claude agent fixture | FR-010 |

## Complexity Tracking

> No constitution violations. All changes follow existing patterns.

| Concern | Mitigation |
|---------|------------|
| Node.js version parsing | Use simple `node --version` → strip `v` prefix → split on `.` → compare major. No semver library needed. |
| Health check caching | TTL-based cache (30s) prevents repeated `node --version` subprocess calls on frequent health polls |
| Backward compatibility | HealthResponse new fields are additive; existing consumers ignore unknown JSON keys |
