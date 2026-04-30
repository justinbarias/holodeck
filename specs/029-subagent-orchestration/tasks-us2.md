---
description: "Per-user-story tasks for US2 (Restrict Subagent Tool Access) — spec 029-subagent-orchestration"
---

# Tasks: Subagent Orchestration — User Story 2 (Restrict Subagent Tool Access)

**Input**: Design documents from `/specs/029-subagent-orchestration/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/subagent-spec.schema.json, quickstart.md
**Scope**: This file covers **only User Story 2 (P2)** — subagent `tools`-allowlist scoping, with emphasis on MCP tool isolation and the load-time tool-name typo warning. Foundational tasks below overlap with US1 (shared scaffolding) and are listed here so this file is self-contained.

**Tests**: Tests are REQUIRED for this feature. Spec SC-003 mandates per-`AgentDefinition.tools` assertions, and the data-model rule #5 (typo warning) is observable only via `pytest.warns(UserWarning)`.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: `US2` for tasks unique to this story; Setup/Foundational/Polish carry no story tag
- All file paths are absolute or repo-relative

## Path Conventions

Single-project layout per `plan.md`:

- Source: `src/holodeck/`
- Tests: `tests/unit/`
- Schema: `schemas/agent.schema.json`
- Samples: `sample/`

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Ensure the dev environment can run the targeted unit tests for this feature.

- [ ] T001 Verify the working tree is on `feature/029-claude-subagents` and that `make init && source .venv/bin/activate` produces a green `pytest tests/unit/models/test_claude_config.py tests/unit/lib/backends/test_claude_backend.py -n auto` baseline. No code changes.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Establish the `SubagentSpec` model surface, the `ClaudeConfig.agents` field, and the SDK translation block that US2 validates against. **Shared with US1's Foundational** — every task in this phase is canonically owned by US1's foundational pass (US1.T002–T011). If US1's Foundational PR is already merged on `feature/029-claude-subagents`, skip Phase 2 entirely and jump to Phase 3. This phase is reproduced here only so US2 is self-contained as a deliverable slice.

**TDD note**: foundational tests are authored first per US1.T002–T004; this file does not duplicate them. If US2 is being delivered standalone, run `pytest tests/unit/models/test_claude_config.py tests/unit/lib/backends/test_claude_backend.py -n auto` after every implementation task to verify the foundational test surface from US1 stays green.

**CRITICAL**: No US2 (Phase 3) work can begin until this phase is complete.

- [ ] T002 Add `SubagentSpec` Pydantic v2 model to `src/holodeck/models/claude_config.py` with `extra="forbid"` and the five fields per `data-model.md` §1: `description: str`, `prompt: str | None = None`, `prompt_file: str | None = None`, `tools: list[str] | None = None`, `model: Literal["sonnet", "opus", "haiku", "inherit"] | None = None`. (Skeleton only; the typo-warning validator is added in Phase 3 / T010.) **Shared with US1.T006** — first owner wins.
- [ ] T003 Add `agents: dict[str, SubagentSpec] | None = None` field to the existing `ClaudeConfig` model in `src/holodeck/models/claude_config.py`, with the `description` text from `data-model.md` §2. **Shared with US1.T007**.
- [ ] T004 Add the empty-map normalization `@model_validator(mode="after")` on `ClaudeConfig` (`if self.agents == {}: self.agents = None`) per `research.md` §5. **Shared with US1.T007**.
- [ ] T005 In `src/holodeck/lib/backends/claude_backend.py` (`build_options()`, around lines 346-355), add the translation block from `data-model.md` §2: when `claude.agents` is non-empty, populate `opts_kwargs["agents"]` with `AgentDefinition(description=..., prompt=..., tools=spec.tools, model=spec.model)` per entry. Add `AgentDefinition` to the existing `from claude_agent_sdk.types import (...)` block at lines 24-29. The `tools` field passes through 1:1 — the SDK enforces the allowlist at runtime (FR-009). **Shared with US1.T009** — if US1's Foundational has landed, this is a no-op verification; otherwise this is the canonical implementation site for both stories.
- [ ] T006 [P] Add the `SubagentSpec` `$def` and `ClaudeConfig.properties.agents` property to `schemas/agent.schema.json` per `contracts/subagent-spec.schema.json`. Closed object (`additionalProperties: false`). The `tools` field is `{"type": ["array", "null"], "items": {"type": "string"}}`. **Shared with US1.T010**.

**Checkpoint**: Foundation ready — US2 work can begin. Running `pytest tests/unit/models/test_claude_config.py -n auto` should still pass (no behavior change yet from US2's perspective).

---

## Phase 3: User Story 2 — Restrict Subagent Tool Access (Priority: P2)

**Story Goal**: Each subagent's `tools` allowlist scopes which built-in or MCP tools it can call. Two subagents on the same parent share parent-level MCP server registrations but see disjoint tool sets when their `tools` lists name disjoint entries. Unknown tool-name entries (typos) surface as `UserWarning` at config-load time, not as hard errors.

**Independent Test (per spec.md)**: Define two subagents whose `tools` lists name different MCP tool identifiers (`mcp__<server>__<tool>`); initialize the agent and verify each subagent's `AgentDefinition.tools` contains only the named entries (SC-003).

**Traceability**: FR-009, FR-007 (omitted `tools` → inherit), SC-003, data-model.md rule #5, research.md §3, quickstart.md §3.

### Tests for User Story 2 (write FIRST, ensure they FAIL before implementation) ⚠️

- [ ] T007 [P] [US2] Add a `KNOWN_BUILTIN_TOOLS` import-or-attribute test to `tests/unit/models/test_claude_config.py` asserting the constant in `claude_config.py` equals `{"Read", "Write", "Edit", "Bash", "Glob", "Grep", "WebSearch", "WebFetch", "Task", "TodoWrite", "NotebookEdit"}` (data-model.md rule #5; research.md §3).
- [ ] T008 [P] [US2] Add `pytest.warns(UserWarning)` tests in `tests/unit/models/test_claude_config.py` covering rule #5 of `SubagentSpec`:
  - **No warning** for entries in the built-ins set (e.g. `["Read", "WebSearch"]`).
  - **No warning** for entries prefixed with `mcp__` (e.g. `["mcp__db__query", "mcp__db__describe"]`).
  - **Warning** for an unknown bare name (e.g. `["WebSerach"]` — typo of WebSearch); assert the warning message includes the offending name and references the three accepted patterns.
  - **No warning** when `tools` is `None` (omitted) — confirms FR-007 inheritance path is silent.
  - **No warning** when `tools` is `[]` (explicit empty list — pure-reasoning subagent per quickstart §7).
- [ ] T009 [P] [US2] Add unit tests in `tests/unit/lib/backends/test_claude_backend.py` exercising `build_options()` translation for SC-003:
  - **AC-1**: parent agent with one MCP server registered + a subagent `db_analyst` whose `tools=["mcp__db__query", "mcp__db__describe"]` → `opts.agents["db_analyst"].tools == ["mcp__db__query", "mcp__db__describe"]`. Verify the value is a list (not `None`) and matches exactly.
  - **AC-2 (disjoint isolation)**: same parent, two subagents `db_analyst` (`tools=["mcp__db__query"]`) and `researcher` (`tools=["WebSearch", "WebFetch"]`) → `opts.agents["db_analyst"].tools == ["mcp__db__query"]` AND `opts.agents["researcher"].tools == ["WebSearch", "WebFetch"]` AND `"mcp__db__query" not in opts.agents["researcher"].tools`.
  - **FR-007 inheritance**: subagent with `tools` omitted → `opts.agents[name].tools is None` (signals SDK to inherit parent tools).
  - **Pure-reasoning**: subagent with `tools=[]` → `opts.agents[name].tools == []` (preserved verbatim, not normalized to `None`).

### Implementation for User Story 2

- [ ] T010 [US2] Add the `KNOWN_BUILTIN_TOOLS: frozenset[str]` module-level constant to `src/holodeck/models/claude_config.py` per data-model.md rule #5 / research.md §3: `frozenset({"Read", "Write", "Edit", "Bash", "Glob", "Grep", "WebSearch", "WebFetch", "Task", "TodoWrite", "NotebookEdit"})`.
- [ ] T011 [US2] Implement the tool-name typo warning validator on `SubagentSpec` in `src/holodeck/models/claude_config.py` as a `@model_validator(mode="after")`. Pattern after `_warn_effort_with_extended_thinking` (research.md §3 cites this as the reference). For each entry in `self.tools or []`:
  - skip if entry is in `KNOWN_BUILTIN_TOOLS`,
  - skip if entry starts with `mcp__`,
  - otherwise emit `warnings.warn(f"Subagent tool '{entry}' does not match a known built-in (one of {sorted(KNOWN_BUILTIN_TOOLS)}), an MCP tool name (mcp__<server>__<tool>), or a HoloDeck-bridged tool. This may be a typo.", UserWarning, stacklevel=2)`.
  - **Note**: cross-checking against the parent agent's top-level `tools` field (HoloDeck-bridged tools, the third accepted pattern in research.md §3) is **deferred** — the model validator runs in isolation and does not have parent-`Agent`-level context. Document this limitation in a comment; the warning message still names all three accepted patterns so users understand bridged names are also valid.
- [ ] T012 [US2] Verify the translation block from T005 already preserves the three states needed by T009: `None` (inherit), `[]` (pure reasoning), and a populated list (allowlist). No additional code expected — this is a verify-and-confirm task. If T009 tests fail because of unintended normalization, fix `build_options()` to pass `tools=spec.tools` verbatim.
- [ ] T013 [US2] Run `pytest tests/unit/models/test_claude_config.py tests/unit/lib/backends/test_claude_backend.py -n auto` and confirm T007–T009 now pass.

### Optional sample addition

- [ ] T014 [P] [US2] Add a sample `agent.yaml` at `sample/subagents-mcp-isolation/claude/agent.yaml` mirroring `quickstart.md` §3 (the `db_analyst` / `researcher` disjoint-MCP-tools pattern). Include a stub `mcp_servers.db` registration so the sample is loadable. Optional but recommended for demoability of SC-003.

**Checkpoint**: User Story 2 is fully functional and independently testable. SC-003 (per-`AgentDefinition.tools` value verification) and the load-time typo-warning behavior are both observable via the unit suite.

---

## Phase N: Polish & Cross-Cutting

**Purpose**: Lint, type-check, and quickstart validation specific to US2's surface area.

- [ ] T015 [P] Run `make format` and `make lint` against `src/holodeck/models/claude_config.py` and `src/holodeck/lib/backends/claude_backend.py`. Address any Black/Ruff/Bandit findings.
- [ ] T016 [P] Run `make type-check` and confirm MyPy is clean for the touched files. The `KNOWN_BUILTIN_TOOLS` constant must be typed `frozenset[str]`; the validator's `warnings.warn` call must satisfy MyPy strict.
- [ ] T017 Manually walk `quickstart.md` §3 (Restricting MCP tool access per subagent) end-to-end: load a YAML matching the snippet, confirm each subagent's resulting `AgentDefinition.tools` matches the YAML's enumeration. Use `holodeck chat` or a small Python harness; this is the human-loop check on top of T009.

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies.
- **Foundational (Phase 2)**: Depends on Setup. **Shared with US1** — if US1's foundational tasks (model + field + translation site + schema) are already merged on the branch, Phase 2 here is a no-op confirmation. Otherwise, the Phase 2 tasks below MUST land before any Phase 3 work begins.
- **User Story 2 (Phase 3)**: Depends on Phase 2 only. Independent of US1's prompt/prompt_file/model validation work.
- **Polish (Phase N)**: Depends on Phase 3 completion.

### Cross-File Dependencies a Developer Should Know

- T002, T003, T004, T010, T011 all touch `src/holodeck/models/claude_config.py` — they are **sequential**, not parallel.
- T005 and T012 both touch `src/holodeck/lib/backends/claude_backend.py` — sequential.
- T007, T008 both touch `tests/unit/models/test_claude_config.py` — can be drafted in one editor session; mark as parallel-safe only if split across two distinct test classes/functions.
- T009 lives in `tests/unit/lib/backends/test_claude_backend.py` (existing file) — independent of the test_claude_config.py edits and can run [P] with T007/T008.
- T006 (schema) is independent of the `.py` edits and can run [P] with the model work.
- T014 (sample YAML) is independent of every other task and can run [P] at any point after T002–T005.

### Within User Story 2

- All test tasks (T007, T008, T009) MUST be written and FAIL before T010–T012.
- T010 (constant) before T011 (validator that consumes it).
- T011 (validator) before T013 (test run) — since T008 asserts the warning behavior implemented in T011.
- T012 (translation verify) before T013.

---

## Parallel Execution Example: User Story 2

```bash
# Once Phase 2 is complete, launch all US2 tests together (different files):
Task: "T007 KNOWN_BUILTIN_TOOLS constant test in tests/unit/models/test_claude_config.py"
Task: "T008 SubagentSpec typo-warning tests in tests/unit/models/test_claude_config.py"   # same file as T007 — coordinate
Task: "T009 build_options() AgentDefinition.tools translation tests in tests/unit/lib/backends/test_claude_backend.py"

# Sample addition can run in parallel with the test/code work:
Task: "T014 sample/subagents-mcp-isolation/claude/agent.yaml"

# Polish tasks (after Phase 3) — file-independent:
Task: "T015 make format && make lint"
Task: "T016 make type-check"
```

---

## Implementation Strategy

### Recommended order (US2-only, single developer)

1. Confirm Phase 1 / T001 baseline is green.
2. Land Phase 2 (T002–T006) as one commit — model + field + translation skeleton + schema fragment. Confirm existing tests still pass.
3. Write the three test groups (T007, T008, T009). Confirm they FAIL.
4. Implement T010 → T011 → T012. Re-run the suite (T013).
5. Optional: T014 (sample YAML).
6. Polish: T015–T017.

### Parallel-team strategy

- One developer takes Phase 2 (foundational) — coordinated with US1's foundational owner, since the work overlaps.
- Once Phase 2 lands, US2 is a single self-contained slice of code (≤ 30 lines added to `claude_config.py`, ≤ 8 lines added to `claude_backend.py`) and is best owned by one developer end-to-end.

### Stopping points

- After T013: US2 is **mergeable** as an independent slice covering SC-003 + load-time typo warnings.
- After T014: US2 has a runnable demo sample.
- After T017: US2 is **release-ready** (lint, types, quickstart walked).

---

## Notes

- Every task above traces directly to a spec FR (FR-007, FR-009), success criterion (SC-003), data-model rule (rule #5), or research decision (§3). No invented scope.
- The Claude SDK enforces the `tools` allowlist at runtime; HoloDeck's job is purely (a) translation 1:1 and (b) load-time warnings for likely typos. Do not add runtime enforcement code.
- Cross-checking unknown tool names against the parent's HoloDeck-bridged tool list is intentionally deferred — it requires `Agent`-level context the `SubagentSpec` validator does not have. The warning message names the third accepted pattern so users understand bridged names are valid.
- **Migration**: this file does not own the `claude.subagents` legacy-block migration error (research §4 / SC-005). That belongs to US1's foundational work — do not duplicate it here.
- Commit after each task or logical group; do not bundle Phase 2 with Phase 3 in a single commit.
