# Data Model: Claude Backend Serve & Deploy Parity

**Feature**: 024-claude-serve-deploy
**Date**: 2026-03-20

## Modified Entities

### ClaudeConfig (existing — `models/claude_config.py`)

**New field:**

| Field | Type | Default | Constraints | Purpose |
|-------|------|---------|-------------|---------|
| `max_concurrent_sessions` | `int \| None` | `10` | `ge=1, le=100` | Maximum concurrent Claude SDK subprocesses per serve instance |

**YAML representation:**
```yaml
claude:
  max_concurrent_sessions: 10  # New field
  permission_mode: acceptAll   # Existing
  max_turns: 25                # Existing
```

### HealthResponse (existing — `serve/models.py`)

**New fields:**

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `backend_ready` | `bool` | `True` | Whether backend prerequisites are satisfied |
| `backend_diagnostics` | `list[str]` | `[]` | List of diagnostic messages (empty when healthy) |

**Extended response example:**
```json
{
  "status": "healthy",
  "agent_name": "my-agent",
  "agent_ready": true,
  "active_sessions": 3,
  "uptime_seconds": 120.5,
  "backend_ready": true,
  "backend_diagnostics": []
}
```

**Degraded response example:**
```json
{
  "status": "degraded",
  "agent_name": "my-agent",
  "agent_ready": true,
  "active_sessions": 0,
  "uptime_seconds": 0.1,
  "backend_ready": false,
  "backend_diagnostics": [
    "Node.js not found on PATH (required for Claude Agent SDK)"
  ]
}
```

## Modified Functions

### generate_dockerfile() (existing — `deploy/dockerfile.py`)

**New parameter:**

| Parameter | Type | Default | Purpose |
|-----------|------|---------|---------|
| `needs_nodejs` | `bool` | `False` | Whether to include Node.js installation in Dockerfile |

### validate_nodejs() (existing — `lib/backends/validators.py`)

**Enhanced behavior:**
- Current: Checks `shutil.which("node")` — binary existence only
- New: Also runs `node --version` and parses semver to verify >= 18

## State Transitions

### ServerState (existing)

```
INITIALIZING → READY → RUNNING → SHUTTING_DOWN → STOPPED
                 ↑                      ↓
                 └── (no change) ←──────┘
```

**New validation gate**: Between `READY` and `RUNNING`, `_validate_backend_prerequisites()` runs. If validation fails, server logs the error and exits (does not transition to RUNNING).

### Session Lifecycle with Cap

```
Request arrives
    ↓
Check active_sessions < max_concurrent_sessions
    ├── Yes → Create AgentExecutor + ClaudeSession (subprocess spawned)
    └── No  → Return 503 Service Unavailable with capacity message

Session active
    ↓
Request completes OR TTL expires OR subprocess crashes
    ↓
Session closed → subprocess terminated → slot freed
```
