# Tasks: Subagent Definitions & Multi-Agent Orchestration — User Story 3

**Spec**: 029-subagent-orchestration
**User Story**: US3 — Define Subagent with Custom System Prompt (Priority: P1)
**Input**: Design documents from `/specs/029-subagent-orchestration/`
**Prerequisites**: spec.md, plan.md, research.md, data-model.md, contracts/subagent-spec.schema.json, quickstart.md
**Scope**: This file delivers ONLY US3. The Phase 2 (Foundational) tasks here are the minimum SubagentSpec/ClaudeConfig.agents scaffolding US3 needs to be testable; US1 and US2 reuse them but are out of scope here.

**Tests**: Included. SC-004 explicitly mandates validation-error tests at config-load time; without them, US3's acceptance scenarios cannot be verified.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: `[US3]` only on User Story 3 tasks. Setup / Foundational / Polish carry no story tag.
- All paths absolute from repo root `/Users/justinbarias/Documents/Git/python/agentlab/`.

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Confirm the working tree and tooling baseline before any model edits.

- [ ] T001 Verify branch is `feature/029-claude-subagents` and working tree is clean; run `make install-dev` if `.venv/` is missing so `pytest`, `mypy`, `ruff`, and `black` are available.
- [ ] T002 [P] Confirm `claude-agent-sdk==0.1.44` is installed in `.venv/` (it provides `AgentDefinition`); run `python -c "from claude_agent_sdk.types import AgentDefinition; print(AgentDefinition)"` to fail-fast if missing.

---

## Phase 2: Foundational (Blocking Prerequisites — shared with US1/US2)

**Purpose**: Stand up the `SubagentSpec` model skeleton, the `ClaudeConfig.agents` field with empty-map normalization, the `description` validator, and the legacy `claude.subagents` migration validator so US3's prompt-sourcing rules have a model to attach to AND so SC-005 holds regardless of which story ships first. These tasks are NOT US3-unique — US1 and US2 build on the same scaffolding — and so are tagged as foundational, not `[US3]`.

**TDD-ordered**: foundational tests (T003a, T003b, T003c) are authored FIRST and verified FAILING before implementation tasks (T004–T009).

**CRITICAL**: No US3 (Phase 3) work can begin until this phase is complete.

### Foundational tests (write FIRST — must FAIL before T004–T009)

- [ ] T003a [P] Add `test_subagent_description_required_non_empty` in `tests/unit/models/test_claude_config.py` asserting `SubagentSpec(description="", prompt="x")` and `SubagentSpec(description="   ", prompt="x")` both raise `ValidationError("subagent requires description")`. Verify it FAILS today (model not yet added). (FR-004, SC-004)
- [ ] T003b [P] Add `test_legacy_subagents_block_rejected` in `tests/unit/models/test_claude_config.py` asserting `ClaudeConfig(**{"subagents": {"enabled": True}})` raises `ValidationError` whose message contains `"claude.subagents is no longer supported"` and references both `claude.agents` and `execution.parallel_test_cases`. Verify it FAILS. **Shared with US1.T002** — first owner wins. (FR-011, SC-005)
- [ ] T003c [P] Add `test_claude_config_agents_empty_map_normalized_to_none` in `tests/unit/models/test_claude_config.py` asserting `ClaudeConfig(agents={}).agents is None`. Verify it FAILS. **Shared with US1.T003**. (FR-010, research §5)

### Foundational implementation

- [ ] T004 Delete the existing `SubagentConfig` model and the `ClaudeConfig.subagents` field in `src/holodeck/models/claude_config.py` (lines 63-70 and 115-118 per `plan.md`). Also remove the `SubagentConfig` import in `tests/unit/models/test_claude_config.py` and any test cases that reference it (lines ~158-196, 248-255, and the `subagents=...` line in `test_with_all_fields` around line 269). **Shared with US1.T005/T011**.
- [ ] T005 Add a `SubagentSpec(BaseModel)` skeleton in `src/holodeck/models/claude_config.py` with `model_config = ConfigDict(extra="forbid")` and the five fields per `data-model.md` §1: `description: str`, `prompt: str | None = None`, `prompt_file: str | None = None`, `tools: list[str] | None = None`, `model: Literal["sonnet", "opus", "haiku", "inherit"] | None = None`. No prompt/prompt_file validators yet — those land in Phase 3. **Shared with US1.T006** — first owner wins. (FR-002, FR-008)
- [ ] T006 Add the `description` non-empty-after-strip `@model_validator(mode="after")` on `SubagentSpec` in `src/holodeck/models/claude_config.py` that raises `ValueError("subagent requires description")` when `description.strip() == ""`. Confirm T003a now PASSES. (FR-004, SC-004)
- [ ] T007 Add `agents: dict[str, SubagentSpec] | None = Field(default=None, ...)` to `ClaudeConfig` in `src/holodeck/models/claude_config.py` (description per `data-model.md` §2). Add a `@model_validator(mode="after")` that normalizes `agents == {}` → `agents = None`. Confirm T003c now PASSES. **Shared with US1.T007**. (FR-001, FR-010, research §5)
- [ ] T008 Add a `@model_validator(mode="before")` on `ClaudeConfig` in `src/holodeck/models/claude_config.py` that detects a legacy `subagents` key in the input dict and raises `ValueError` with the exact message from data-model.md §2 rule 2: `"claude.subagents is no longer supported; remove this block. Subagent forwarding is gated solely by the presence of claude.agents. To cap HoloDeck-side test concurrency, set execution.parallel_test_cases instead."`. The `mode="before"` runs ahead of `extra="forbid"` so the targeted error wins. Confirm T003b now PASSES. **Shared with US1.T008**. (FR-011, SC-005)

**Checkpoint**: `SubagentSpec` exists with `description` validator, `ClaudeConfig.agents` exists with empty-map normalization, the legacy `subagents` block is rejected with the documented friendly error, and all three foundational tests are green. US3 prompt-validation work can now begin.

---

## Phase 3: User Story 3 — Define Subagent with Custom System Prompt (Priority: P1)

**Goal**: Allow users to give each subagent a distinct system prompt, sourced either inline (`prompt`) or from a file (`prompt_file`) resolved relative to the agent.yaml directory. Validation errors fire at config-load time.

**Independent Test**: Load three minimal `agent.yaml` fixtures (inline prompt, prompt_file, neither) through `holodeck.config.loader`. Verify:
1. Inline prompt → `SubagentSpec.prompt == "You are a financial analyst."`, `prompt_file is None`.
2. `prompt_file: ./prompts/analyst.md` → `SubagentSpec.prompt` contains the file's contents, `prompt_file is None`.
3. Neither → `ValidationError("subagent requires either prompt or prompt_file")` at load time.

### Tests for User Story 3 ⚠️ WRITE FIRST — must FAIL before T017–T020

> All test tasks add cases to `tests/unit/models/test_claude_config.py` (single file). They share that file, so NOT marked `[P]`. The fixture-creation task T009 is `[P]` because it touches a different path.

- [ ] T009 [P] [US3] Create test prompt fixture at `tests/unit/models/fixtures/subagent_prompts/analyst.md` containing the literal text `You are a data analyst. Analyze data only — do not write code.` (used by T012 and T015). Confirm parent directories exist; create `tests/unit/models/fixtures/` and `tests/unit/models/fixtures/subagent_prompts/` if missing.
- [ ] T010 [US3] Add `test_subagent_inline_prompt_loads` to `tests/unit/models/test_claude_config.py` — constructs a `SubagentSpec(description="x", prompt="You are a financial analyst.")`, asserts `.prompt == "You are a financial analyst."` and `.prompt_file is None`. (US3 acceptance scenario 1; FR-002)
- [ ] T011 [US3] Add `test_subagent_inline_prompt_empty_after_strip_rejected` to `tests/unit/models/test_claude_config.py` — `SubagentSpec(description="x", prompt="   ")` raises `ValidationError`. (data-model.md §1 rule "Non-empty after strip"; SC-004)
- [ ] T012 [US3] Add `test_subagent_prompt_file_inlined` to `tests/unit/models/test_claude_config.py` — sets `agent_base_dir` `ContextVar` (from `src/holodeck/config/context.py`) to `tests/unit/models/fixtures/subagent_prompts/`, constructs `SubagentSpec(description="x", prompt_file="./analyst.md")`, asserts `.prompt` matches the fixture's contents and `.prompt_file is None`. Use `agent_base_dir.set(...)` with a token and `agent_base_dir.reset(token)` in a `try/finally` (mirror the pattern in `src/holodeck/config/schema.py:104-106`). (US3 acceptance scenario 2; FR-005)
- [ ] T013 [US3] Add `test_subagent_prompt_file_resolves_relative_to_agent_base_dir` to `tests/unit/models/test_claude_config.py` — same as T012 but uses a nested path (`subdir/analyst.md`) under a temp directory created via `tmp_path`, and verifies absolute paths are also accepted (path resolution doesn't mangle absolute paths). (FR-005; quickstart §5 row "prompt_file resolves relative")
- [ ] T014 [US3] Add `test_subagent_prompt_file_not_found_rejected` to `tests/unit/models/test_claude_config.py` — `SubagentSpec(description="x", prompt_file="./does-not-exist.md")` raises `ValidationError` whose message contains `"prompt_file not found"` and the offending path. (Edge case: nonexistent prompt_file; SC-004; quickstart §5)
- [ ] T015 [US3] Add `test_subagent_prompt_and_prompt_file_mutually_exclusive` to `tests/unit/models/test_claude_config.py` — `SubagentSpec(description="x", prompt="hi", prompt_file="./analyst.md")` raises `ValidationError("prompt and prompt_file are mutually exclusive")` (exact message per quickstart §5). (FR-006; SC-004)
- [ ] T016 [US3] Add `test_subagent_neither_prompt_nor_prompt_file_rejected` to `tests/unit/models/test_claude_config.py` — `SubagentSpec(description="x")` raises `ValidationError("subagent requires either prompt or prompt_file")` (exact message per quickstart §5). (FR-006; US3 acceptance scenario 3; SC-004)

> Run `pytest tests/unit/models/test_claude_config.py -n auto -v` and confirm T010–T016 all FAIL with the expected error shapes before proceeding to implementation.

### Implementation for User Story 3

> All implementation tasks edit `src/holodeck/models/claude_config.py`. Single file, sequential.

- [ ] T017 [US3] In `src/holodeck/models/claude_config.py`, add a `@model_validator(mode="after")` on `SubagentSpec` that raises `ValueError("prompt and prompt_file are mutually exclusive")` when both `self.prompt` and `self.prompt_file` are non-`None` (truthy after the `None` check; explicit empty strings are caught separately by T018/T019). (FR-006; satisfies T015)
- [ ] T018 [US3] In the same `@model_validator(mode="after")` (or a sibling validator in the same file), raise `ValueError("subagent requires either prompt or prompt_file")` when both are `None`. Order this check after the mutual-exclusion check so the messages don't conflict. (FR-006; satisfies T016; US3 acceptance scenario 3)
- [ ] T019 [US3] In `src/holodeck/models/claude_config.py`, extend the `SubagentSpec` `@model_validator(mode="after")` to handle `prompt_file` resolution. Mirror the canonical pattern at `src/holodeck/config/schema.py:101-112`:

  ```python
  from pathlib import Path
  from holodeck.config.context import agent_base_dir   # ContextVar[str | None]

  if self.prompt_file is not None:
      base_dir_value = agent_base_dir.get()
      base_dir = Path.cwd() if base_dir_value is None else Path(base_dir_value)
      path = Path(self.prompt_file)
      if not path.is_absolute():
          path = base_dir / path
      if not path.exists():
          raise ValueError(f"prompt_file not found: {path}")
      self.prompt = path.read_text(encoding="utf-8")
      self.prompt_file = None
  ```

  Use `model_validator(mode="after")` so the assignments stick (Pydantic v2 returns `self`). Note the ContextVar is `ContextVar[str | None]` (not `Path | None`), so coerce only after the None check. (FR-005, FR-006; satisfies T012, T013, T014; data-model.md §1 rule 3; research.md §2)
- [ ] T020 [US3] In `src/holodeck/models/claude_config.py`, after `prompt_file` inlining, add a non-empty-after-strip check on the resulting `self.prompt`: raise `ValueError("subagent prompt must be non-empty")` when `self.prompt is not None and self.prompt.strip() == ""`. This catches both inline empty prompts and prompt_file paths that point at an empty file. (data-model.md §1 invariant "always non-empty after construction"; satisfies T011)

> Run `pytest tests/unit/models/test_claude_config.py -n auto -v` and confirm T010–T016 all PASS.

**Checkpoint**: US3 is fully functional and testable independently. A subagent's `prompt`/`prompt_file` is correctly sourced and validated at config-load time, with the exact error messages spec'd in `quickstart.md` §5.

---

## Phase 4: Polish & Cross-Cutting

- [ ] T021 Run `make format` and `make lint` from the repo root; confirm `src/holodeck/models/claude_config.py` and `tests/unit/models/test_claude_config.py` pass Black (88 cols) + Ruff with no warnings.
- [ ] T022 Run `make type-check` from the repo root; confirm `src/holodeck/models/claude_config.py` passes MyPy strict (especially the `agent_base_dir.get()` fallback path and the `Path` import).
- [ ] T023 [P] Run the full unit test suite — `pytest tests/unit/ -n auto -v` — to confirm no regressions in adjacent `ClaudeConfig` tests, in particular that `_warn_effort_with_extended_thinking` still fires.
- [ ] T024 Validate `quickstart.md` §5 row "prompt_file: ./does-not-exist.md" by writing a 5-line ad-hoc YAML fixture and loading it through `holodeck.config.loader.load_agent_config()` from a Python REPL; confirm the `ValidationError` matches the documented message verbatim. (SC-004 end-to-end smoke check)

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1 (Setup)**: No dependencies — start immediately.
- **Phase 2 (Foundational, TDD)**: Depends on Phase 1. BLOCKS Phase 3. Tests T003a/T003b/T003c authored FIRST and verified FAILING before T004–T008 implementations.
- **Phase 3 (US3)**: Depends on Phase 2 completion. Within Phase 3, tests T009–T016 must be written and FAIL before implementation T017–T020.
- **Phase 4 (Polish)**: Depends on Phase 3.

### Task-Level Dependencies (within US3)

- T009 (fixture file) → T012, T013, T015 (tests that read the fixture).
- T010–T016 (failing tests) → T017–T020 (implementations that make them pass).
- T017 (mutual-exclusion check) → T018 (at-least-one check; ordered second so the error messages don't collide).
- T018 → T019 (prompt_file resolution; runs only if at least one of prompt/prompt_file is set).
- T019 → T020 (non-empty post-strip check; runs after `prompt_file` has been inlined into `prompt`).

### Foundational → US3 Dependencies

- T005 (SubagentSpec skeleton) → all of T010–T020.
- T006 (description validator) → T010–T016 (tests construct `SubagentSpec(description="x", ...)` so the description rule must already permit non-empty strings).
- T004 (delete old `SubagentConfig`) → T005 (add new model) → T007 (add `agents` field) → T008 (add legacy migration validator) — strict but safe ordering since they all edit the same file.
- T003a/T003b/T003c (foundational failing tests) → T006/T008/T007 (the implementations that turn each green).

### Cross-File Dependencies a Developer Should Know

- `src/holodeck/config/context.py` defines `agent_base_dir: ContextVar[str | None]` (line 22 — note `str`, not `Path`). T019 imports it and coerces with `Path(...)` only after the None check; do **not** add a new ContextVar.
- `src/holodeck/config/loader.py` (lines 775-783) sets `agent_base_dir` for the duration of YAML loading. US3 does NOT modify the loader — the existing wiring is reused.
- `src/holodeck/config/schema.py` (lines 104-106) is the canonical example of `agent_base_dir.get()` usage with a fallback for the unset case. T019's resolution logic must match that pattern.
- `src/holodeck/lib/backends/claude_backend.py` `build_options()` is **out of scope for US3** — translation to `AgentDefinition` belongs to US1. US3 only guarantees that `SubagentSpec.prompt` is a non-empty string ready for downstream translation.
- T004 deletes `SubagentConfig`. Existing tests in `tests/unit/models/test_claude_config.py` (lines 158-196, 248-255 per plan.md) reference it and are removed in the same task. **Shared with US1.T005/T011** — coordinate ownership at PR time.

---

## Parallel Execution Example: User Story 3

The vast majority of US3 tasks edit one file (`tests/unit/models/test_claude_config.py` or `src/holodeck/models/claude_config.py`) and are sequential. Foundational tests T003a/T003b/T003c are `[P]`. In Phase 3, only T009 is genuinely parallel.

```bash
# Phase 1 setup — independent commands:
make install-dev                                                          # T001
python -c "from claude_agent_sdk.types import AgentDefinition; print(AgentDefinition)"   # T002

# Phase 3 — only T009 is parallelizable; the test- and impl-edits all share one file:
mkdir -p tests/unit/models/fixtures/subagent_prompts                       # T009 setup
# (then add the file's contents)

# After tests T010–T016 are written and confirmed failing, run the full file in parallel:
pytest tests/unit/models/test_claude_config.py -n auto -v                  # verify FAIL → implement → verify PASS
```

---

## Implementation Strategy

### Slice-First (US3 only — this file)

1. Complete Phase 1 (Setup) — environment confirmed.
2. Complete Phase 2 (Foundational, TDD) — author T003a/T003b/T003c first (failing); land T004 → T005 → T006 → T007 → T008. Each implementation turns the matching foundational test green. **STOP if there is friction here**: T004's deletion of `SubagentConfig` will break unrelated tests in `test_claude_config.py`; the deletion sweeps those legacy cases at the same time, but coordinate with US1 if a parallel branch is open.
3. Complete Phase 3 tests (T009–T016) — all FAIL.
4. Complete Phase 3 implementation (T017–T020) — all PASS.
5. Complete Phase 4 polish — format/lint/type-check/regression suite green.
6. **STOP and VALIDATE**: Run `pytest tests/unit/models/test_claude_config.py -n auto -v -k subagent` and confirm all foundational + US3 tests pass. Run `make ci` and confirm no regressions.

### Parallel Team Strategy (US3 + US1 + US2)

Phase 2 is the shared foundation. One developer should land Phases 1+2 alone, then US1, US2, and US3 can split:

- Developer A: US1 (build_options translation, schema, sample agent.yaml).
- Developer B: US2 (tools-list typo warnings, MCP scoping tests).
- Developer C (this file): US3 (prompt validation rules + fixture).

US3 has no runtime dependency on US1 or US2 once Phase 2 is in place — the model validators it adds are self-contained.

---

## Notes

- Every task line traces to at least one of: FR-002, FR-004, FR-005, FR-006, US3 acceptance scenario 1/2/3, an edge case in spec.md §"Edge Cases", or SC-004.
- `[P]` is used sparingly and only when the task touches a different file from sibling tasks.
- US3 does **not** touch `claude_backend.py`, `agent.schema.json`, or any sample YAML — those changes belong to US1.
- The exact validation error messages in T012, T013, T014 are copied verbatim from `quickstart.md` §5 so users see the documented strings.
- After Phase 3, `SubagentSpec.prompt_file` is always `None` post-construction (data-model.md §"Invariants"). Downstream code (US1's `build_options()`) can read `spec.prompt` directly without re-resolving paths.
- Do NOT add a JSON schema fragment in this slice — `contracts/subagent-spec.schema.json` is applied to `schemas/agent.schema.json` by US1.
