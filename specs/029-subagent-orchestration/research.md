# Research: Subagent Definitions & Multi-Agent Orchestration

**Spec**: 029-subagent-orchestration
**Phase**: 0 (Outline & Research)
**Date**: 2026-04-29

The Technical Context section of `plan.md` had no `NEEDS CLARIFICATION`
markers — every unknown was resolvable against the installed SDK and the
existing codebase. The decisions below document the design choices that
flow into Phase 1.

---

## 1. SDK surface for subagent definitions

**Decision**: Translate each YAML `claude.agents.<name>` entry into one
`claude_agent_sdk.types.AgentDefinition` and assemble them into the
`agents: dict[str, AgentDefinition] | None` field of `ClaudeAgentOptions`.

**Rationale**: Verified against the installed SDK (`claude-agent-sdk==0.1.44`):

```python
@dataclass
class AgentDefinition:
    description: str
    prompt: str
    tools: list[str] | None = None
    model: Literal["sonnet", "opus", "haiku", "inherit"] | None = None
```

`ClaudeAgentOptions` exposes `agents` as a top-level dataclass field. The
spec's Assumptions section already locks this in; this research step
confirmed it empirically.

**Alternatives considered**:

- *A separate Pydantic model per subagent type* (researcher, analyst, …) —
  rejected: the SDK doesn't differentiate, and HoloDeck's value is
  configuration not code generation.
- *Keep the existing `subagents` config and overload it with definitions* —
  rejected: `SubagentConfig` semantically describes HoloDeck-side
  concurrency. Mixing it with SDK-side specs would be confusing and lock
  out the future possibility of supporting both knobs together.

---

## 2. `prompt_file` resolution

**Decision**: Resolve `prompt_file` paths relative to the agent.yaml
directory using the existing `agent_base_dir` `ContextVar`
(`src/holodeck/config/context.py:22`). Read the file at config-load time
and inline the contents into the resulting `SubagentSpec.prompt`.

**Rationale**:

- The loader already sets `agent_base_dir` for the duration of YAML loading
  (`src/holodeck/config/loader.py:775-783`).
- The same pattern is already used for hierarchical-document tools
  (`src/holodeck/config/schema.py:104-106`). Reusing it keeps the
  resolution rules consistent and avoids a second context mechanism.
- Inlining at load time means the SDK never sees `prompt_file`; it always
  receives a non-empty `prompt` (matching FR-006).

**Alternatives considered**:

- *Lazy file read inside `build_options()`* — rejected: errors would
  surface at session start instead of config validation, breaking SC-004.
- *CWD-relative resolution* — rejected: fails when the user runs `holodeck`
  from a different directory. The agent.yaml directory is the only stable
  anchor.

**Edge cases covered**:

- `prompt_file` does not exist → `ValidationError` at load time (SC-004).
- Both `prompt` and `prompt_file` set → `ValidationError` (FR-006).
- Neither set → `ValidationError` (FR-006; SDK requires non-empty prompt).

---

## 3. Tool list validation strategy

**Decision**: At config-load time (`@model_validator(mode="after")` on
`SubagentSpec`), validate that each entry in a subagent's `tools` list
matches one of three patterns:

1. A name in a static set of known SDK built-in tools, kept as a module
   constant in `claude_config.py`:
   `{"Read", "Write", "Edit", "Bash", "Glob", "Grep", "WebSearch",
   "WebFetch", "Task", "TodoWrite", "NotebookEdit"}`. (This list is
   advisory and may lag SDK upgrades — that's acceptable because the
   penalty for a stale entry is a warning, not a hard error.)
2. A string starting with `mcp__` (parent-registered MCP tool — the
   actual server presence is verified later by the SDK at runtime).
3. A name listed in the parent agent's top-level `tools` field
   (HoloDeck-bridged tools from `tool_adapters.py`).

Unknown entries produce a `UserWarning` at config-load time, mirroring
the existing `_warn_effort_with_extended_thinking` validator on
`ClaudeConfig`. This satisfies the spec edge case verbatim ("Validation
warning at config load time") and catches typos like `WebSerach` or
`mcp__db_query` (single underscore) before any session starts.

**Rationale**: Earlier draft deferred this to `build_options()` (session
start). Verification surfaced the timing mismatch with the spec, which
explicitly promises load-time. Implementing the check in the model
validator keeps everything in `claude_config.py` and matches the
load-time guarantee. The static built-in set is small and SDK-stable;
when the SDK adds a new tool, HoloDeck users get a one-release lag of
spurious warnings — acceptable tradeoff vs. blocking valid configs.

**Alternatives considered**:

- *No validation* — rejected: doesn't help users catch typos.
- *Hard error on unknown* — rejected: too brittle; fails on legitimate
  built-ins not yet in HoloDeck's allowlist.
- *Defer to `build_options()`* — rejected at verification time: violates
  the spec's "config load time" promise and means warnings are tied to
  session start rather than `holodeck` config-validation paths.

---

## 4. Removal of `claude.subagents` block

**Decision**: Delete the entire `claude.subagents` block (`SubagentConfig`
model and `ClaudeConfig.subagents` field) instead of trying to give it
meaningful semantics under spec 029. The presence of `claude.agents`
becomes the sole gate for forwarding subagent definitions to the SDK.
HoloDeck-side test concurrency continues to be controlled exclusively by
the existing top-level `execution.parallel_test_cases`.

**Rationale**: Verification surfaced that both fields in the block are
vestigial under the new design:

- **`max_parallel` is not a real SDK control.** The Claude Agent SDK
  manages subagent dispatch internally — HoloDeck has no API to
  intercede. The earlier framing as "HoloDeck-side concurrency cap"
  amounted to clamping the test-runner semaphore via
  `min(execution.parallel_test_cases, max_parallel)`, which is pure
  duplication: a user who wants a lower cap can just set
  `execution.parallel_test_cases` lower in the same agent YAML. There
  is no scenario where `max_parallel` provides functionality
  `parallel_test_cases` doesn't already cover.

- **`enabled` is redundant with `agents` presence.** If a user defines
  `claude.agents`, they want subagent forwarding; if they don't, they
  don't. There is no third state. A separate boolean gate adds
  ceremony without value.

- **Avoids the bool|None tri-state hack.** With `enabled` removed, we no
  longer need the `bool | None` workaround the previous research draft
  introduced to satisfy FR-011 + FR-012. Both FRs are now obsolete and
  removed from the spec.

The deletion is acceptable because the `claude.subagents` block is
recently introduced (commits `9c1dc8c` and `65e292e`) and not yet
load-bearing in production agent.yaml configs. To make migration crisp,
we add a `@model_validator(mode="before")` on `ClaudeConfig` that
detects a legacy `subagents` key and raises a `ValidationError` with a
specific message: *"`claude.subagents` is no longer supported; remove
this block. Subagent forwarding is gated solely by the presence of
`claude.agents`. To cap HoloDeck-side test concurrency, set
`execution.parallel_test_cases` instead."* This is friendlier than the
generic Pydantic "extra fields not permitted" that `extra="forbid"`
would otherwise produce.

**Alternatives considered**:

- *Keep `subagents.enabled` only, drop `max_parallel`* — rejected:
  `enabled` is still redundant with `agents` presence. Half-measure.
- *Keep both with deprecation warnings (gradual sunset)* — rejected:
  the block is too new to warrant a deprecation cycle. A clean delete
  with a clear migration error is simpler and less code.
- *Keep `max_parallel` for backward-compat with shipped configs* —
  rejected: per `git log`, no shipped agent.yaml under `sample/` uses
  the field, and the user explicitly directed deletion. The migration
  cost is one-line removal in any custom configs.

**Verification**: SC-005 (revised) — loading an `agent.yaml` containing
`claude.subagents` produces a clear `ValidationError` pointing the user
at `claude.agents` and `execution.parallel_test_cases`.

---

## 5. Empty `agents` map handling

**Decision**: Treat `claude.agents: {}` (empty map) and a missing `agents`
field as equivalent: do not pass `agents` to the SDK at all. Per the spec:

> What happens when subagent definitions are empty (no agents listed)?
> Treated the same as not having the `agents` section at all.

Implemented by storing `agents` as `dict[str, SubagentSpec] | None` on
`ClaudeConfig`, with a `model_validator` that normalizes `{}` to `None`.

**Rationale**: Avoids passing an empty dict to the SDK, which would
suppress the "no subagents configured" code path in the SDK.

---

## 6. Schema strategy

**Decision**: Add a `SubagentSpec` `$def` to `schemas/agent.schema.json`
mirroring the Pydantic model's structural shape (closed object,
`additionalProperties: false`), and add `agents` as an optional property
of `ClaudeConfig` typed as `{"type": "object",
"additionalProperties": {"$ref": "#/$defs/SubagentSpec"}}`. The schema
declares field types and the `model` enum, but **does not** encode the
prompt/prompt_file mutual-exclusion rule — Pydantic owns that.

**Rationale**: HoloDeck's schema is the user-facing contract for IDE
autocompletion and pre-Pydantic validation. Mirroring the model keeps
the two in lockstep. Closed-object enforcement catches typos early. The
mutual-exclusion rule was originally drafted in the schema as a
`oneOf`/`allOf` block, but verification surfaced that JSON Schema treats
explicit-null as "present", so a config like `prompt: null, prompt_file:
"x.md"` would fail the `oneOf` even though the user's intent is
unambiguous. Pydantic's `model_validator` evaluates field truthiness
correctly and is already the authoritative validator. Keeping the rule
in one place (Pydantic) avoids divergence.

**Alternatives considered**:

- *Encode mutual exclusion via JSON Schema `oneOf`* — rejected at
  verification time: nullable-field semantics make the constraint produce
  false negatives; users get confusing schema errors instead of the clear
  Pydantic message.
- *Tighten `prompt` / `prompt_file` to `"string"` only (no null)* —
  rejected: would force YAML users to omit the field rather than write
  explicit null, a change in user-facing surface area for a small win.
- *Auto-generate the schema from the Pydantic model* — out of scope; the
  rest of the schema is hand-written and divergence would be jarring.
- *Encode `claude.subagents` as a deprecated/ignored block in the
  schema* — rejected: spec 029 deletes it outright (research §4).

---

## Open items

None. All FRs and edge cases in the spec map to a concrete decision above.
Phase 1 (data-model.md, contracts/, quickstart.md) can proceed.
