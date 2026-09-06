# Data Model: Subagent Definitions & Multi-Agent Orchestration

**Spec**: 029-subagent-orchestration
**Phase**: 1 (Design)
**Date**: 2026-04-29

This feature is configuration-only. No new persistence, no new runtime
state. The data model below describes Pydantic v2 models added to
`src/holodeck/models/claude_config.py` and how they translate to the
Claude Agent SDK's `AgentDefinition`.

---

## Entities

### 1. `SubagentSpec` (new)

A single subagent definition. Lives next to existing config models in
`claude_config.py`. Strict (`extra="forbid"`).

| Field | Type | Required | Default | Validation |
|---|---|---|---|---|
| `description` | `str` | yes | — | Non-empty after strip. The SDK uses this for routing decisions. |
| `prompt` | `str \| None` | conditionally | `None` | Mutually exclusive with `prompt_file`. At least one of `prompt` / `prompt_file` MUST be set. Non-empty after strip. |
| `prompt_file` | `str \| None` | conditionally | `None` | Mutually exclusive with `prompt`. Path resolved relative to `agent_base_dir`; file MUST exist at validation time. Inlined into `prompt` after load. |
| `tools` | `list[str] \| None` | no | `None` | When omitted (`None`), subagent inherits all parent tools. When provided, only those tools are available; entries should match a known built-in (see static set below), an MCP tool name (`mcp__<server>__<tool>`), or a HoloDeck-bridged tool — unknown entries produce a `UserWarning` at config-load time (not error). |
| `model` | `Literal["sonnet", "opus", "haiku", "inherit"] \| None` | no | `None` | Restricted to the SDK's literal set. Any other value produces a `ValidationError` at config-load time. |

**Validation rules** (implemented as `@model_validator(mode="after")`):

1. **Mutual exclusion**: if both `prompt` and `prompt_file` are set →
   `ValidationError("prompt and prompt_file are mutually exclusive")`.
2. **At-least-one**: if neither is set → `ValidationError("subagent
   requires either prompt or prompt_file")`.
3. **prompt_file resolution**: if `prompt_file` is set, resolve relative
   to `agent_base_dir.get()`; if the path doesn't exist →
   `ValidationError("prompt_file not found: {path}")`. If it exists,
   read the file and store the contents in `prompt`; clear `prompt_file`
   so downstream code only ever sees `prompt`.
4. **description non-empty**: `description.strip() == ""` →
   `ValidationError`.
5. **Tool-name typo warnings** (load time): for each entry in `tools`,
   emit a `UserWarning` if the entry is **not** in the known-built-ins
   set, **not** prefixed with `mcp__`, and **not** a HoloDeck-bridged
   tool name resolvable from the parent agent's top-level `tools` field.
   The known-built-ins set is a module constant in `claude_config.py`:
   `{"Read", "Write", "Edit", "Bash", "Glob", "Grep", "WebSearch",
   "WebFetch", "Task", "TodoWrite", "NotebookEdit"}`. The warning
   message includes the offending name and the three accepted patterns.

**State transitions**: none. Immutable after construction.

**Maps to**: `claude_agent_sdk.types.AgentDefinition` 1-to-1, with the
following correspondence:

| `SubagentSpec` | `AgentDefinition` |
|---|---|
| `description` | `description` |
| `prompt` (after `prompt_file` inlining) | `prompt` |
| `tools` | `tools` |
| `model` | `model` |

---

### 2. `ClaudeConfig.agents` (new field on existing model)

A new optional field on the existing `ClaudeConfig` model:

```python
agents: dict[str, SubagentSpec] | None = Field(
    default=None,
    description=(
        "Named subagent definitions. Each entry becomes an AgentDefinition "
        "on ClaudeAgentOptions.agents. Subagents share parent MCP servers "
        "and use their `tools` allowlist to scope MCP tool access."
    ),
)
```

**Validation rules**:

1. **Empty-map normalization** (`@model_validator(mode="after")` on
   `ClaudeConfig`): if `agents == {}` → set `agents = None`.
2. **Legacy-block migration error** (`@model_validator(mode="before")` on
   `ClaudeConfig`): if the input dict contains a `subagents` key, raise
   `ValidationError` with the exact message: *"`claude.subagents` is no
   longer supported; remove this block. Subagent forwarding is gated
   solely by the presence of `claude.agents`. To cap HoloDeck-side test
   concurrency, set `execution.parallel_test_cases` instead."* This runs
   before the `extra="forbid"` check so the user gets a targeted error
   rather than a generic "extra fields not permitted".

**Translation to SDK** (in
`src/holodeck/lib/backends/claude_backend.py:build_options()`):

```python
if claude and claude.agents:
    opts_kwargs["agents"] = {
        name: AgentDefinition(
            description=spec.description,
            prompt=spec.prompt,           # already inlined
            tools=spec.tools,             # None → inherit
            model=spec.model,             # None → inherit
        )
        for name, spec in claude.agents.items()
    }
```

The presence of `claude.agents` (after empty-map normalization) is the
sole gate. There is no separate `subagents.enabled` check.

Tool-name typo warnings are emitted at config-load time by the
`SubagentSpec` model validator (rule #5 above), not in `build_options()`.
This satisfies the spec edge-case requirement that warnings fire at
"config load time", and keeps `build_options()` focused on translation.

---

### 3. `SubagentConfig` (deleted)

The previously-existing `SubagentConfig` model and `ClaudeConfig.subagents`
field are removed. See research §4 for rationale. Migration: any
`agent.yaml` referencing `claude.subagents` produces a clear
`ValidationError` directing users at `claude.agents` and
`execution.parallel_test_cases`. Tests in
`tests/unit/models/test_claude_config.py` that exercised
`SubagentConfig` are removed; new tests verify the migration error
fires correctly.

---

## Relationships

```
ClaudeConfig
└── agents : dict[str, SubagentSpec] | None   (new — sole gate)

SubagentSpec  →  claude_agent_sdk.types.AgentDefinition  (1-to-1, lossless,
                                                          translated in
                                                          build_options())
```

No cross-entity foreign keys; the dict key (subagent name) is opaque to
the SDK and used only as the `agents` map key on `ClaudeAgentOptions`.

---

## Invariants

- `SubagentSpec.prompt` is always non-empty after construction (loader
  has inlined `prompt_file` contents).
- `SubagentSpec.prompt_file` is always `None` after construction.
- `ClaudeConfig.agents` is either `None` or a non-empty dict (empty maps
  normalized to `None`).
- `ClaudeConfig` has no `subagents` field. Any input dict containing a
  `subagents` key is rejected at validation time with a migration error.

---

## Backwards compatibility

- Existing `agent.yaml` files without a `claude.agents` and without a
  `claude.subagents` section behave identically. No new required fields
  are added at the top level.
- **Breaking change**: `claude.subagents` (with `enabled` and
  `max_parallel`) is removed. Any `agent.yaml` still using this block
  fails to load with a clear migration error. Per `git log` and a sweep
  of `sample/`, no shipped sample agent.yaml uses the field; the recent
  feature commits (`9c1dc8c`, `65e292e`) added it but did not publish
  user-facing examples that depend on it. Custom user configs require a
  one-line removal.
- Schema additions (`SubagentSpec` `$def`, `agents` property) are purely
  additive. The schema also drops `SubagentConfig` `$def` and the
  `subagents` property — also a breaking change with the same rationale.
