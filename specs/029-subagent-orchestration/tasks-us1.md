---

description: "Per-user-story tasks for US1 (P1): Define a Multi-Agent Research Team"
---

# Tasks (US1): Subagent Definitions & Multi-Agent Orchestration — User Story 1

**Spec ID**: 029-subagent-orchestration
**Branch**: `feature/029-claude-subagents`
**Input**: Design documents from `/specs/029-subagent-orchestration/`
**Prerequisites**: plan.md, spec.md (US1 + FR-001..FR-011 + SC-001..SC-005), data-model.md, research.md §1/§5/§6, contracts/subagent-spec.schema.json

**Scope**: This file delivers **only User Story 1 (P1) — Define a Multi-Agent Research Team** as the MVP increment. US2 (per-subagent MCP tool scoping) and US3 (`prompt`/`prompt_file` mutual-exclusion + file resolution) live in their own per-story tasks files.

US1 specifically covers:
- Three named subagents (researcher / analyst / writer) round-trip from `claude.agents` YAML to SDK `AgentDefinition` objects.
- Per-subagent `model: haiku | sonnet | opus | inherit` override.
- Per-subagent `tools: [...]` allowlist.
- Omitted `tools` → subagent inherits all parent tools (None passed to SDK).

Foundational phase establishes the `SubagentSpec` model + `ClaudeConfig.agents` field skeleton and removes the legacy `claude.subagents` block. The legacy removal MUST land in foundational (not US1) because `ClaudeConfig` uses `extra="forbid"`, so the new `agents` field cannot be added next to a leftover `subagents` field without breaking config load, and US1's tests would otherwise fail before they could exercise the new translation.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies on other [P] tasks in the same phase).
- **[US1]**: Tagged on tasks scoped to User Story 1. Setup / Foundational / Polish tasks are NOT tagged with a story.
- File paths are absolute or repository-relative.

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Confirm the working environment is ready. No new dependencies are introduced by this feature (`claude-agent-sdk==0.1.44` already installed; `AgentDefinition` and `ClaudeAgentOptions.agents` are available).

- [ ] T001 Verify `claude-agent-sdk==0.1.44` is installed and `AgentDefinition` import works by running `source .venv/bin/activate && python -c "from claude_agent_sdk.types import AgentDefinition; from claude_agent_sdk import ClaudeAgentOptions; print(AgentDefinition.__dataclass_fields__.keys()); print('agents' in ClaudeAgentOptions.__dataclass_fields__)"`. Expect the four fields `description, prompt, tools, model` and `True` for `agents`. (Sanity gate per plan.md "Concrete integration points".)

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Stand up the SDK-translation surface and the validation surface needed by US1, US2, and US3. Foundational tests are authored FIRST and verified FAILING before implementation per the project's TDD discipline (constitution principle III).

**Foundational scope** (shared with US2 and US3 — skip whichever tasks have landed on the branch):
- Delete legacy `SubagentConfig` + `ClaudeConfig.subagents` (FR-011, SC-005).
- Add `SubagentSpec` model skeleton + `description` non-empty validator + `ClaudeConfig.agents` field + empty-map normalizer (FR-001, FR-002, FR-004, FR-008, FR-010).
- Add legacy-key migration validator + targeted error message (SC-005, FR-011).
- Add `build_options()` translation block (FR-001, FR-003, FR-007, FR-010, SC-002). **Shared with US2** — US2's T005 covers the same edit; first owner to land wins, the other deletes their copy.
- Sync `schemas/agent.schema.json` to the new shape.
- Remove legacy tests; add migration-error and translation tests.

**CRITICAL**: No US1 (Phase 3) work begins until Phase 2 is complete.

### Foundational tests (write FIRST — must FAIL before T009–T015)

- [ ] T002 [P] Add unit test `test_legacy_subagents_block_rejected` in `tests/unit/models/test_claude_config.py` that constructs `ClaudeConfig(**{"subagents": {"enabled": True}})` and asserts the resulting `ValidationError` message contains the substring `"claude.subagents is no longer supported"` and mentions both `claude.agents` and `execution.parallel_test_cases`. Verify it FAILS at this point (validator not yet added). Traces to SC-005, FR-011.
- [ ] T003 [P] Add unit test `test_claude_config_agents_empty_map_normalized_to_none` in `tests/unit/models/test_claude_config.py` asserting `ClaudeConfig(agents={}).agents is None`. Verify it FAILS (field/validator not yet added). Traces to FR-010, edge case "empty agents map", research.md §5.
- [ ] T004 [P] Add a foundational translation test `test_build_options_no_agents_omits_field` in the existing `tests/unit/lib/backends/test_claude_backend.py` asserting that when `claude.agents is None`, `ClaudeAgentOptions.agents` is `None` (or absent from `opts_kwargs`). Verify it FAILS or passes trivially today (translation block not yet added). Traces to FR-010.

### Foundational implementation

- [ ] T005 Delete the `SubagentConfig` Pydantic model (lines ~63-70) and the `ClaudeConfig.subagents` field (lines ~115-118) from `src/holodeck/models/claude_config.py`. Also remove `SubagentConfig` from any module-level `__all__` if present. Traces to FR-011, plan.md "Concrete integration points" line 1.
- [ ] T006 Add the new `SubagentSpec` Pydantic model in `src/holodeck/models/claude_config.py` next to the other Claude config models, with `model_config = ConfigDict(extra="forbid")` and these fields per data-model.md §1: `description: str`, `prompt: str | None = None`, `prompt_file: str | None = None`, `tools: list[str] | None = None`, `model: Literal["sonnet", "opus", "haiku", "inherit"] | None = None`. **Do not** add `prompt`/`prompt_file` mutual-exclusion / file-resolution validators yet — those belong to US3. Do add a `description.strip() != ""` check (data-model.md rule 4) since US1 acceptance scenario 1 asserts on `description`. Traces to FR-001, FR-002, FR-004, FR-008.
- [ ] T007 Add the `agents: dict[str, SubagentSpec] | None = None` field to `ClaudeConfig` in `src/holodeck/models/claude_config.py` with the description text from data-model.md §2. Add a `@model_validator(mode="after")` that normalizes `agents == {}` → `agents = None` (research.md §5 / data-model.md rule 1). Confirm T003 now PASSES. Traces to FR-001, FR-010, edge-case "empty agents map".
- [ ] T008 Add a `@model_validator(mode="before")` on `ClaudeConfig` in `src/holodeck/models/claude_config.py` that detects a legacy `subagents` key in the input dict and raises `ValueError` with the exact message from data-model.md §2 rule 2 (\"`claude.subagents` is no longer supported; remove this block. Subagent forwarding is gated solely by the presence of `claude.agents`. To cap HoloDeck-side test concurrency, set `execution.parallel_test_cases` instead.\"). Confirm T002 now PASSES. Traces to FR-011, SC-005.
- [ ] T009 In `src/holodeck/lib/backends/claude_backend.py` `build_options()` (around lines 346-355, immediately after the existing `if claude.disallowed_tools:` block), add the SDK translation block from data-model.md §2:

  ```python
  if claude.agents:
      opts_kwargs["agents"] = {
          name: AgentDefinition(
              description=spec.description,
              prompt=spec.prompt,
              tools=spec.tools,
              model=spec.model,
          )
          for name, spec in claude.agents.items()
      }
  ```

  Add `AgentDefinition` to the existing `from claude_agent_sdk.types import (...)` block at `src/holodeck/lib/backends/claude_backend.py:24-29` (alongside `HookContext`, `HookEvent`, `McpSdkServerConfig`, `SyncHookJSONOutput`). Confirm T004 now PASSES. **Shared with US2.T005** — first owner wins. Traces to FR-001, FR-003, FR-007, FR-010, US1 acceptance scenarios 1–4, SC-002.
- [ ] T010 [P] Update `schemas/agent.schema.json`: delete the `SubagentConfig` `$def` (lines ~158-176) and the `subagents` property under `ClaudeConfig.properties` (lines ~212-215). Add a new `SubagentSpec` `$def` and an `agents` property under `ClaudeConfig.properties` per `specs/029-subagent-orchestration/contracts/subagent-spec.schema.json` (the `$defs.SubagentSpec` block and the `patches.ClaudeConfig.properties.agents` block). Traces to FR-001, FR-002, FR-008, SC-005, research.md §6.
- [ ] T011 [P] Remove the legacy `TestSubagentConfig` test class (lines ~158-196) and the `test_with_subagents` case + the `subagents=...` line in `test_with_all_fields` (around lines 211, 248-255, 269) from `tests/unit/models/test_claude_config.py`. Also remove the `SubagentConfig` import. Traces to plan.md "Concrete integration points" final line; SC-005.

**Checkpoint**: `ClaudeConfig` accepts an `agents` map of `SubagentSpec`s, rejects the legacy `subagents` block with a targeted error, the SDK translation block populates `ClaudeAgentOptions.agents`, and the schema mirrors the model. US1 acceptance-scenario tests can now proceed.

---

## Phase 3: User Story 1 — Define a Multi-Agent Research Team (Priority: P1) MVP

**Story Goal**: A user can declare three named subagents (researcher, analyst, writer) under `claude.agents` in YAML — each with its own `description`, inline `prompt`, optional `tools` list, and optional `model` literal — and have all three round-trip into `ClaudeAgentOptions.agents` as `AgentDefinition` objects when the agent is initialized.

**Independent Test**: Run the new unit tests in `tests/unit/lib/backends/test_claude_backend.py` (added in this phase) — they construct an `Agent` with a `claude.agents` map of three subagents and assert that `build_options(...)` returns a `ClaudeAgentOptions` whose `agents` dict has three entries, each an `AgentDefinition` with the expected `description`, `prompt`, `tools`, and `model` values, plus the four US1 acceptance scenarios (round-trip, model override, tools allowlist, tools omitted → None).

### Tests for User Story 1 (write FIRST — must FAIL before T016/T017)

> Foundational tests T002–T004 cover `ClaudeAgentOptions.agents` baseline (None/empty), legacy migration error, and empty-map normalization. The US1 tests below cover the four acceptance scenarios on top of that scaffolding.

- [ ] T012 [P] [US1] Add unit test `test_subagent_spec_round_trip_minimal` in `tests/unit/models/test_claude_config.py`: construct a `SubagentSpec(description="x", prompt="y")` and assert `description == "x"`, `prompt == "y"`, `tools is None`, `model is None`. Verify it PASSES once T006 has landed (round-trip on the model itself, not the SDK translation). Traces to FR-001, FR-002, US1 acceptance scenario 1.
- [ ] T013 [P] [US1] Add unit test `test_subagent_spec_model_literal_accepts_allowed` (parametrized over `["sonnet", "opus", "haiku", "inherit"]`) and `test_subagent_spec_model_literal_rejects_other` (parametrized over `["claude-3-5-sonnet-20241022", "gpt-4", "haiku-3"]` expecting `ValidationError`) in `tests/unit/models/test_claude_config.py`. Traces to FR-008, US1 acceptance scenario 2, edge case "model outside allowed set".
- [ ] T014 [P] [US1] Add the four SDK-translation tests to the existing `tests/unit/lib/backends/test_claude_backend.py` (do NOT create a new file — match the existing layout convention used by every other backend test in that directory). All marked `@pytest.mark.unit`:
  - `test_build_options_translates_three_named_subagents`: three entries (researcher / analyst / writer) round-trip 1-to-1 into `AgentDefinition` objects with matching `description` and `prompt`. Traces to US1 acceptance scenario 1, SC-002.
  - `test_build_options_subagent_model_override_haiku`: a subagent with `model="haiku"` produces `AgentDefinition(model="haiku")` on the SDK side. Traces to US1 acceptance scenario 2, SC-002.
  - `test_build_options_subagent_tools_allowlist`: a subagent with `tools=["WebSearch", "WebFetch"]` produces `AgentDefinition(tools=["WebSearch", "WebFetch"])`. Traces to US1 acceptance scenario 3, SC-003.
  - `test_build_options_subagent_tools_omitted_inherits_all`: a subagent with no `tools` field produces `AgentDefinition(tools=None)`. Traces to US1 acceptance scenario 4, FR-007.

  All four tests must construct `Agent` via the project's existing helpers/fixtures (mirror the patterns in `tests/unit/lib/backends/test_claude_backend.py`) and call `build_options(...)`. **Verify all four tests FAIL** before T016/T017 if the foundational T009 translation block has not yet been merged on this branch.

### Implementation for User Story 1

- [ ] T015 [US1] **No-op verification**: confirm the foundational translation block (T009) is in place. If T009 was deferred (e.g., parallel branch), implement it here. The translation logic is identical; T009 is the canonical home.
- [ ] T016 [US1] If any of T012/T013/T014 fail because of missing model fields or translation gaps, surface the diff and fix it inside the foundational tasks (T006/T009) rather than adding US1-specific code. US1 should be a pure-test slice on top of the foundational scaffolding.
- [ ] T017 [US1] Run T012/T013/T014 to green: `source .venv/bin/activate && pytest tests/unit/models/test_claude_config.py tests/unit/lib/backends/test_claude_backend.py -n auto -v`. Confirm all four US1 backend cases plus the two US1 model cases pass.

**Checkpoint**: User Story 1 is fully functional and independently testable. A user can write a `claude.agents` block with three named subagents, optional per-subagent `model` literal, and optional `tools` allowlist; the SDK receives matching `AgentDefinition` objects.

---

## Phase N: Polish & Cross-Cutting Concerns

**Purpose**: Ship-quality polish for the US1 increment. Keep this phase narrow — extended validators (mutual exclusion, prompt_file resolution, tool-name typo warnings) belong to US3 / US2 task files, not here.

- [ ] T018 [P] Run `make format && make lint && make type-check` from the repo root and resolve any issues introduced by T005–T014. Traces to constitution principle (code quality).
- [ ] T019 [P] Add a US1 sample agent.yaml under `sample/research/claude/research-team.yaml` (or the closest existing scenario directory) demonstrating two or three subagents per the `quickstart.md §1` minimal example, **without** `prompt_file` (US3 territory) and **without** MCP-scoped tools (US2 territory) — keep it pure US1: inline `prompt`, optional `model`, optional `tools` allowlist, one subagent that omits `tools` to demonstrate inheritance. Traces to SC-001, plan.md project-structure tree (sample/ entry).
- [ ] T020 Verify the full project test suite still passes: `source .venv/bin/activate && make test-unit`. Confirm no regressions in the broader `tests/unit/models/` or `tests/unit/lib/backends/` trees from the legacy-block deletion. Traces to constitution principle (test discipline).

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1 (Setup)**: No dependencies — T001 can run immediately.
- **Phase 2 (Foundational)**: Depends on Phase 1. **Blocks Phase 3.** TDD-ordered: tests T002–T004 are authored first and verified failing; then implementation T005–T011 lands; foundational tests must be green before Phase 3 begins.
- **Phase 3 (US1)**: Depends on Phase 2. T012–T014 are pure tests (model + translation); T015–T017 verify-and-run.
- **Phase N (Polish)**: Depends on Phase 3 completion. T020 should run last (after T018 fixes any lint/type issues).

### Within Phase 2 (Foundational)

- Tests first: T002, T003, T004 are `[P]` — author in parallel, all verified FAILING.
- Implementation in order: T005 → T006 → T007 → T008 are sequential (all in `claude_config.py`, same file). T009 (translation in `claude_backend.py`) can run in parallel with T010 (`agent.schema.json`) and T011 (test cleanup) — all marked `[P]`.
- T002 turns green after T008. T003 turns green after T007. T004 turns green after T009.

### Within Phase 3 (US1)

- T012, T013, T014 are all `[P]` — different test files / functions; T012 and T013 share `test_claude_config.py` so coordinate on merge.
- T015 is a no-op verification IF foundational T009 has landed; otherwise it is the canonical implementation site.
- T017 depends on T012–T014.

### Cross-File Dependencies a Developer Should Know

- `src/holodeck/models/claude_config.py` is touched by T005, T006, T007, T008 — strictly sequential edits to the same file.
- `tests/unit/models/test_claude_config.py` is touched by T002, T003, T011, T012, T013 — these can be authored in parallel only if developers coordinate around merge conflicts (different test classes / functions). Recommend serializing if a single dev is working alone.
- `src/holodeck/lib/backends/claude_backend.py` (T009, T015) and `tests/unit/lib/backends/test_claude_backend.py` (T004, T014) depend on `SubagentSpec` and `ClaudeConfig.agents` existing — i.e. T006 and T007 must be merged first.
- The legacy-block deletion (T005, T010, T011) must land **before** T009 ships any agent.yaml fixture, because `extra="forbid"` on `ClaudeConfig` would otherwise reject any fixture that exercises the new `agents` key alongside a leftover `subagents` field.
- **US2 overlap**: foundational T009 (translation) is shared with US2's T005. First owner to land wins; the other deletes their copy at PR review.

---

## Parallel Execution Example: User Story 1

Once Phase 2 is green, the three US1 test-authoring tasks can be picked up in parallel:

```bash
# Author all three US1 test tasks in parallel — different files / different test functions:
T012 [US1] tests/unit/models/test_claude_config.py::test_subagent_spec_round_trip_minimal
T013 [US1] tests/unit/models/test_claude_config.py::test_subagent_spec_model_literal_*
T014 [US1] tests/unit/lib/backends/test_claude_backend.py  (four new translation tests)

# Run them to confirm they all PASS once foundational T006/T007/T009 are merged:
source .venv/bin/activate && pytest \
  tests/unit/models/test_claude_config.py::TestSubagentSpec \
  tests/unit/lib/backends/test_claude_backend.py \
  -n auto -v
```

If any of T012–T014 fail, fix the foundational task they depend on (T006 for model fields, T009 for translation) — do NOT add US1-specific code outside the foundational scaffolding.

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. **Phase 1: Setup** — T001 sanity-check the SDK surface.
2. **Phase 2: Foundational (TDD)** — Author T002–T004 first (foundational tests, all FAILING). Then land T005–T011 in topological order. Verify T002–T004 turn green at T008/T007/T009 respectively.
3. **Phase 3: User Story 1** — Author T012–T014 (acceptance-scenario tests). They should pass immediately if Phase 2 is sound. T015–T017 are verification.
4. **STOP and VALIDATE**: A user can hand-write the quickstart.md §1 YAML example and `holodeck chat` it without errors; all four US1 acceptance scenarios are covered by passing unit tests.
5. **Phase N: Polish** — T018 (format/lint/type), T019 (sample), T020 (full suite).
6. **Ship**.

### Out-of-Scope for US1 (deliberately deferred)

- Mutual exclusion of `prompt`/`prompt_file` and file resolution → **US3** (separate tasks file).
- Tool-name typo warnings (`UserWarning` for unknown built-ins / non-`mcp__` / non-bridged names) → **US2** (per data-model.md rule 5).
- Per-subagent MCP tool scoping demos and tests → **US2**.

If a developer is tempted to "just add the mutual-exclusion validator while editing `SubagentSpec`" during T006, **don't** — it changes the test surface and risks landing US3 work in the US1 PR. Keep the slice surgical.

---

## Notes

- Every task above traces to a concrete spec FR / SC / acceptance scenario or to a plan.md integration point. No invented scope.
- `[P]` is reserved for tasks in different files with no ordering dependency.
- TDD discipline applies to every phase: Foundational tests (T002–T004) and US1 tests (T012–T014) are authored before implementation and verified FAILING (or passing trivially) at the time they're written.
- Commit after each task or at the natural Phase 2 / Phase 3 / Phase N checkpoints.
- Do not delete pre-existing dead code while editing `claude_config.py` or `claude_backend.py` — clean up only the symbols this feature removes (`SubagentConfig`, `ClaudeConfig.subagents`).
