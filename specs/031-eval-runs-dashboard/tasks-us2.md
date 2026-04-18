---
description: "Task list — User Story 2: Prompt Versioning via Frontmatter"
---

# Tasks — US2: Prompt Versioning via Frontmatter (Priority: P1)

**Feature**: 031-eval-runs-dashboard
**Spec**: [spec.md](./spec.md) — User Story 2 (P1)
**Plan**: [plan.md](./plan.md)
**Data model**: [data-model.md](./data-model.md) — `PromptVersion`
**Research**: [research.md](./research.md) — R1

**Goal**: Parse optional YAML frontmatter from the instructions markdown file via `python-frontmatter`. Expose a `PromptVersion` Pydantic model capturing `version` (manual or `auto-<sha256[:8]>`), `author`, `description`, `tags`, `source`, `file_path`, `body_hash`, and `extra` (unknown keys). The prompt body presented to the LLM MUST be identical to today (frontmatter-stripped).

**Independent Test**: Create `instructions.md` with `version: "1.2"` frontmatter → run `holodeck test` → `EvalRun.metadata.prompt_version.version == "1.2"`. Remove the manual version → re-run → version becomes `"auto-" + first 8 hex chars of SHA-256(body)` and is stable across reruns with unchanged body.

**TDD discipline**: Every task marked "(TDD)" writes the failing test first. Do not implement ahead.

**Dependency**: US1 must have a `PromptVersion` stub in `src/holodeck/models/eval_run.py` (US1 T024). US2 replaces stub fields with full validation and provides the `resolve_prompt_version()` function US1 T032 calls.

---

## Phase 1: Setup

- [ ] T101 [US2] Add `python-frontmatter>=1.1,<2.0` to core dependencies in `pyproject.toml`; run `uv lock` and commit the updated lockfile
- [ ] T102 [US2] Verify `frontmatter` imports cleanly in `.venv` by running `python -c "import frontmatter; print(frontmatter.__version__)"`

---

## Phase 2: Foundational

None. US2 is self-contained once the dep is installed.

---

## Phase 3: US2 — PromptVersion model + resolver

### Tests first (TDD)

- [ ] T103 [P] [US2] (TDD) `tests/unit/models/test_prompt_version.py`: model field coverage — `version: str` required, `author: str | None`, `description: str | None`, `tags: list[str]` default `[]`, `source: Literal["file", "inline"]` required, `file_path: str | None`, `body_hash: str` required (matches `^[a-f0-9]{64}$`), `extra: dict[str, Any]` default `{}`; `extra="forbid"` on the model itself
- [ ] T104 [P] [US2] (TDD) `tests/unit/models/test_prompt_version.py`: validation — `source="inline"` requires `file_path is None`; `source="file"` requires `file_path` non-empty; invalid `body_hash` (wrong length, non-hex) fails validation
- [ ] T105 [P] [US2] (TDD) `tests/unit/lib/test_prompt_version.py`: `resolve_prompt_version(instructions, base_dir)` — when `instructions.inline` is set, returns `source="inline"`, `file_path=None`, `version=f"auto-{sha256(inline)[:8]}"`, `body_hash` is full 64-char SHA-256 of inline content, frontmatter parsing is skipped (FR-014)
- [ ] T106 [P] [US2] (TDD) `tests/unit/lib/test_prompt_version.py`: given a fixture `instructions.md` with frontmatter `version: "1.2"`, `author: "jane"`, `description: "d"`, `tags: [a, b]` — returns those values verbatim; `source="file"`, `file_path` is the resolved path, `body_hash` is SHA-256 of content AFTER frontmatter removal (post.content) (FR-011, AC1)
- [ ] T107 [P] [US2] (TDD) `tests/unit/lib/test_prompt_version.py`: given frontmatter without a `version:` key, `version == "auto-" + sha256(body)[:8]` and stays identical across two calls with the same file content (FR-013, AC2, SC-003)
- [ ] T108 [P] [US2] (TDD) `tests/unit/lib/test_prompt_version.py`: given a file with NO frontmatter at all, no error is raised; `version == auto-hash`, `author is None`, `description is None`, `tags == []`, `extra == {}` (FR-011, AC3)
- [ ] T109 [P] [US2] (TDD) `tests/unit/lib/test_prompt_version.py`: mutating one character of the prompt body changes `version` (SC-004, AC5)
- [ ] T110 [P] [US2] (TDD) `tests/unit/lib/test_prompt_version.py`: unknown frontmatter keys (`custom_field: value`, `another: 42`) are preserved in `PromptVersion.extra`, recognised keys are NOT duplicated into `extra` (FR-016, AC7)
- [ ] T111 [P] [US2] (TDD) `tests/unit/lib/test_prompt_version.py`: malformed YAML inside `---` fences (e.g., `tags: [unclosed`) raises `ConfigError` referencing the file path and YAML parse error (FR-017, edge case)
- [ ] T112 [P] [US2] (TDD) `tests/unit/lib/test_prompt_version.py`: relative `instructions.file` paths are resolved against the provided `base_dir` (matches existing `resolve_instructions` convention)
- [ ] T113 [P] [US2] (TDD) `tests/integration/cli/test_prompt_version_in_eval_run.py`: end-to-end — run `holodeck test` against a fixture with frontmatter-annotated instructions, assert the persisted `EvalRun.metadata.prompt_version.version == "1.2"` and `tags == ["support", "v1"]`
- [ ] T114 [P] [US2] (TDD) `tests/unit/lib/test_prompt_version.py`: assert the body returned by the existing `resolve_instructions()` is UNCHANGED by this feature — `resolve_instructions` returns the same full string it always did (including or excluding frontmatter per pre-existing behaviour). Document the invariant: `resolve_prompt_version` is a SEPARATE call; `resolve_instructions` signature is untouched (research.md R1, plan.md §"Key architecture decisions")
- [ ] T115 [P] [US2] (TDD) `tests/integration/cli/test_prompt_body_unchanged.py`: snapshot the exact string passed to the LLM (intercept via SK/Claude backend spy) before and after this feature lands — byte-equivalent (FR-015)

### Implementation

- [ ] T116 [US2] Replace the `PromptVersion` stub in `src/holodeck/models/eval_run.py` with the full model per data-model.md: all fields, validators (`body_hash` regex, `source`/`file_path` consistency), `extra="forbid"` at Pydantic level (unknown YAML keys land in the `extra` dict, not as extra model fields)
- [ ] T117 [US2] Create `src/holodeck/lib/prompt_version.py` exposing `resolve_prompt_version(instructions: Instructions, base_dir: Path | None) -> PromptVersion`
- [ ] T118 [US2] Inline branch in `resolve_prompt_version`: when `instructions.inline` is set, skip frontmatter entirely; compute SHA-256 of `instructions.inline`; return `PromptVersion(source="inline", file_path=None, version=f"auto-{hash[:8]}", body_hash=hash, tags=[], extra={})`
- [ ] T119 [US2] File branch in `resolve_prompt_version`: resolve `instructions.file` against `base_dir` (match `resolve_instructions` convention); call `frontmatter.load(resolved_path)`; wrap `yaml.YAMLError` → raise `ConfigError("instructions.file", f"Malformed YAML frontmatter in {path}: {e}")` (FR-017)
- [ ] T120 [US2] Partition parsed metadata in `resolve_prompt_version`: split `post.metadata` into recognised keys (`version`, `author`, `description`, `tags`) vs. the remainder (→ `extra`); compute `body_hash = sha256(post.content.encode("utf-8")).hexdigest()`; derive `version = metadata.get("version", f"auto-{body_hash[:8]}")`
- [ ] T121 [US2] Export `resolve_prompt_version` and `PromptVersion` from `src/holodeck/lib/prompt_version.py` and re-export `PromptVersion` from `src/holodeck/models/__init__.py` for discoverability
- [ ] T122 [US2] Update US1 T032 wiring in `src/holodeck/cli/commands/test.py` (or wherever US1 assembled `EvalRun`) to call `resolve_prompt_version(agent.instructions, agent_base_dir)` and pass the result into `build_eval_run_metadata(...)` — replacing the US1 stub
- [ ] T123 [US2] Verify `resolve_instructions` in `src/holodeck/lib/instruction_resolver.py` is UNCHANGED (no signature change); confirm no other call site imports or references `resolve_prompt_version` (additive module, per plan.md §"Key architecture decisions")

**Checkpoint**: All US2 tests green. Quickstart §3 reproducible: removing the manual `version:` yields a stable `auto-<hash>` across reruns; one-char body edit changes the hash.

---

## Dependencies

- T101 blocks T102–T123 (dep must be installed).
- T103–T115 (TDD tests) must be failing before T116–T123 implementation.
- T116 depends on US1 T024 (PromptVersion stub location exists).
- T117–T120 depend on T116 (need the full model).
- T122 depends on US1 T030–T031 (CLI wiring present) and on T117–T121.

### Parallel Opportunities

```bash
# TDD phase — all test files are independent:
Task: "tests/unit/models/test_prompt_version.py (T103, T104)"
Task: "tests/unit/lib/test_prompt_version.py (T105–T112, T114)"
Task: "tests/integration/cli/test_prompt_version_in_eval_run.py (T113)"
Task: "tests/integration/cli/test_prompt_body_unchanged.py (T115)"

# Implementation — leaf module independent of US1:
Task: "src/holodeck/lib/prompt_version.py (T117–T120)"
```

---

## Acceptance Scenario Traceability

| AC | Covered by |
|---|---|
| AC1 (`version: "1.2"` from frontmatter) | T106, T113 |
| AC2 (no manual `version:` → auto-hash) | T107 |
| AC3 (no frontmatter → no error, auto-hash) | T108 |
| AC4 (stable hash when body unchanged, SC-003) | T107 |
| AC5 (one-char edit changes hash, SC-004) | T109 |
| AC6 (inline instructions → auto-hash) | T105 |
| AC7 (unknown keys → `extra`) | T110 |
| FR-015 (body to LLM unchanged) | T114, T115 |
| FR-017 (malformed YAML → `ConfigError`) | T111 |

---

## Implementation Strategy

1. Install the dep first (T101).
2. Write every TDD test (T103–T115); confirm all fail.
3. Replace the US1 `PromptVersion` stub with the full model (T116).
4. Implement `resolve_prompt_version` leaf-first (inline branch → file branch → metadata partition → error wrapping).
5. Wire into CLI (T122); remove US1's stub call.
6. Verify quickstart §3 by hand.
