# Implementation Plan: Subagent Definitions & Multi-Agent Orchestration

**Branch**: `feature/029-claude-subagents` | **Date**: 2026-04-29 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/029-subagent-orchestration/spec.md`
**Spec ID**: 029-subagent-orchestration
**Dependencies**: 027-mcp-http-sse-transport (parent agent's MCP servers; subagents inherit access via their `tools` allowlist)

## Summary

Expose the Claude Agent SDK's multi-agent orchestration via YAML. Add a new
`claude.agents` map of named subagent specs (`description`, `prompt` |
`prompt_file`, `tools`, `model`) and translate each entry 1-to-1 into a
`claude_agent_sdk.types.AgentDefinition` on `ClaudeAgentOptions.agents`. No
per-subagent MCP server registration — subagents share the parent's
registrations and use their `tools` allowlist (including `mcp__<server>__<tool>`
identifiers) for capability scoping.

**Removal**: the existing `claude.subagents` block (`enabled`, `max_parallel`)
is deleted entirely. Both fields were vestigial — `enabled` is redundant
with `agents` presence, and `max_parallel` was never a real SDK control
(the SDK manages subagent dispatch internally) and only duplicates what
`execution.parallel_test_cases` already does at the test-runner level. The
presence of `claude.agents` is the only gate. HoloDeck-side test
concurrency continues to be controlled exclusively by
`execution.parallel_test_cases`.

The implementation is config-only: a new `SubagentSpec` Pydantic model, a
schema fragment, validators (prompt_file resolution, mutual-exclusion,
model literal, tool-name typo warnings), one `agents=` kwarg on
`ClaudeAgentOptions`, deletion of `SubagentConfig` + `ClaudeConfig.subagents`
+ the corresponding schema `$def`, and migration of any existing tests
that referenced them.

## Technical Context

**Language/Version**: Python 3.10+
**Primary Dependencies**: `claude-agent-sdk==0.1.44` (provides `AgentDefinition`, `ClaudeAgentOptions.agents`), Pydantic v2, PyYAML, Click
**Storage**: N/A (config-only feature; no persistent state)
**Testing**: pytest with `-n auto`; markers `@pytest.mark.unit` and `@pytest.mark.integration`
**Target Platform**: Linux/macOS dev workstations and CI runners (HoloDeck CLI)
**Project Type**: Single project (HoloDeck monolith with backend abstraction)
**Performance Goals**: No runtime hot path. Config load and validation must complete in <50ms for an agent.yaml with up to ~16 subagents (typical: 3–5).
**Constraints**:
- `AgentDefinition` exposes exactly 4 fields (`description`, `prompt`, `tools`, `model`); HoloDeck must not invent extras.
- `model` literal set is `{sonnet, opus, haiku, inherit}` — full model IDs are rejected by the SDK type.
- `prompt_file` paths must resolve relative to the agent.yaml directory via the existing `agent_base_dir` `ContextVar` (see `src/holodeck/config/context.py:22`).
- HoloDeck cannot intercede in SDK-internal subagent dispatch. The presence of `claude.agents` is the sole gate for forwarding subagent definitions.
- `claude.subagents` is deleted; loading an `agent.yaml` that still contains it produces a clear Pydantic `ValidationError` (since `ClaudeConfig` uses `extra="forbid"`).

**Scale/Scope**: Up to ~16 subagent specs per agent (typical: 3–5); single-agent YAML files; thousands of test cases per run.

### Concrete integration points (from codebase survey)

- `src/holodeck/models/claude_config.py:63-70,115-118` — **delete** `SubagentConfig` and the `ClaudeConfig.subagents` field. Add new `SubagentSpec` model and `ClaudeConfig.agents: dict[str, SubagentSpec] | None`.
- `src/holodeck/lib/backends/claude_backend.py:248-356` — `build_options()`. Populate `opts_kwargs["agents"]` from `claude.agents` when non-empty. No `subagents.enabled` check.
- `schemas/agent.schema.json:158-215` — **delete** `SubagentConfig` `$def` and the `subagents` property under `ClaudeConfig`. Add `SubagentSpec` `$def` and an `agents` property under `ClaudeConfig`.
- `src/holodeck/config/loader.py:775-783` — already sets `agent_base_dir`; reuse for `prompt_file` resolution at validation time. No other changes.
- `src/holodeck/lib/test_runner/executor.py:820-828` — **no changes**. The existing `execution.parallel_test_cases` semaphore is the only HoloDeck-side concurrency control.
- `tests/unit/models/test_claude_config.py:158-196,248-255` — remove the existing `SubagentConfig` test cases; add `SubagentSpec` and `ClaudeConfig.agents` tests.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

Reference: `.specify/memory/constitution.md` v1.0.1.

| Principle | Status | Notes |
|---|---|---|
| I. No-Code-First Agent Definition | ✅ Pass | The whole feature is YAML-driven. Users define subagent teams entirely via `claude.agents` without writing Python. |
| II. MCP for API Integrations | ✅ Pass | No new API integrations. Subagents reuse parent-level MCP server registrations; `tools` allowlist references `mcp__<server>__<tool>` identifiers. The spec explicitly forbids per-subagent MCP server registration (FR-009). |
| III. Test-First with Multimodal Support | ✅ Pass | Plan delivers unit tests for `SubagentSpec` validation and contract tests on `build_options()`'s `agents` kwarg (SC-002, SC-003, SC-004). Tests for the deleted `subagents` block are removed (SC-005). No multimodal surface area changes. |
| IV. OpenTelemetry-Native Observability | ✅ Pass | Subagent dispatch is instrumented by the Claude SDK itself; no new HoloDeck spans required. The OTel bridge (`otel_bridge.py`) continues to forward env vars unchanged. |
| V. Evaluation Flexibility with Model Overrides | ✅ Pass | Per-subagent `model` override aligns with this principle by extending model selection granularity from "global / per-run / per-metric" to "per-subagent" within a single agent run. |

**Architecture constraints**: Config-only change. No new engine, no cross-engine
contract changes. Backend abstraction (`AgentBackend` / `AgentSession`) is
untouched — only the Claude backend's options builder gains a new field, and
the model layer loses one redundant block.

**Code quality / testing discipline**: Adheres to existing conventions
(`extra="forbid"` Pydantic, Black/Ruff/MyPy strict, pytest markers, AAA tests).
No coverage regression expected — replaced tests cover an equal-or-greater
surface.

**Backwards compatibility note**: Removing `claude.subagents` is a breaking
config change. Acceptable because (a) `extra="forbid"` will produce a clear
error message, (b) the field is freshly introduced (recent commits 9c1dc8c
and 65e292e) and not yet load-bearing in shipped agent.yaml configs, and
(c) the migration is a one-line removal. We add a custom validator that
detects the legacy `subagents` key and emits a more helpful error than the
generic Pydantic "extra fields not permitted" — pointing users at the
`claude.agents`-only model.

**Result**: ✅ All gates pass. No Complexity Tracking entries needed.

### Post-Phase-1 re-evaluation

After producing `research.md`, `data-model.md`, `contracts/subagent-spec.schema.json`, and `quickstart.md`, the design surface is:

- One new Pydantic model (`SubagentSpec`), one new optional field (`ClaudeConfig.agents`), three new validators on `SubagentSpec` (mutual-exclusion + prompt_file resolution, tool-name typo warnings, model literal — the last enforced by Pydantic's `Literal` type), one new validator on `ClaudeConfig` (legacy `subagents` key migration error).
- One deleted Pydantic model (`SubagentConfig`) and one deleted field (`ClaudeConfig.subagents`).
- One new translation block in `build_options()` (≤ 8 lines).
- One additive JSON schema fragment (`SubagentSpec` `$def` + `agents` property).
- One subtractive JSON schema change (delete `SubagentConfig` `$def` + `subagents` property).
- Test-runner executor: **no changes**.
- Two unit test files updated (one extended with new cases, one with the deleted-block cases removed).

No new engines, no cross-engine contract changes, no new dependencies. All five constitution principles continue to hold; gate result remains ✅.

## Project Structure

### Documentation (this feature)

```text
specs/029-subagent-orchestration/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output
│   └── subagent-spec.schema.json   # JSON schema fragment for SubagentSpec
├── checklists/          # Existing
└── tasks.md             # Phase 2 output (/speckit.tasks)
```

### Source Code (repository root)

Only the files below change. No new packages or modules. Test-runner executor
is unchanged.

```text
src/holodeck/
├── models/
│   └── claude_config.py                  # -SubagentConfig; -ClaudeConfig.subagents;
│                                         # +SubagentSpec; +ClaudeConfig.agents;
│                                         # +legacy-key validator
├── lib/
│   └── backends/
│       └── claude_backend.py             # +translate ClaudeConfig.agents → ClaudeAgentOptions.agents
└── config/
    └── loader.py                         # (no changes — agent_base_dir already set)

schemas/
└── agent.schema.json                     # -SubagentConfig $def; -subagents property;
                                          # +SubagentSpec $def; +agents property

tests/
└── unit/
    ├── models/
    │   └── test_claude_config.py         # remove SubagentConfig cases;
    │                                     # add SubagentSpec + ClaudeConfig.agents cases
    └── backends/
        └── test_claude_backend_options.py  # +AgentDefinition translation cases

sample/                                   # +one example agent.yaml under sample/<scenario>/claude/
                                          #  showing claude.agents with 2-3 subagents (US1 demo)
```

**Structure Decision**: Single-project layout (`src/holodeck/`) preserved. No
new top-level directories. The change is a focused vertical slice: models →
schema → backend translation → tests/sample.

## Complexity Tracking

> Fill ONLY if Constitution Check has violations that must be justified.

No violations. Table intentionally empty.
