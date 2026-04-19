---
description: "Task list — User Story 3: EvalRun Captures Full Agent Configuration Snapshot"
---

# Tasks — US3: EvalRun Captures Full Agent Configuration Snapshot (Priority: P1)

**Feature**: 031-eval-runs-dashboard
**Spec**: [spec.md](./spec.md) — User Story 3 (P1)
**Plan**: [plan.md](./plan.md)
**Data model**: [data-model.md](./data-model.md) — `EvalRunMetadata.agent_config`

**Goal**: Guarantee that `EvalRun.metadata.agent_config` is a complete, faithful, frozen snapshot of the validated `Agent` model that produced the run — covering `model`, `embedding_provider`, `tools` (every tool type with its full config), `evaluations`, `claude` SDK block, and `instructions` metadata. Subsequent edits to `agent.yaml` MUST NOT affect previously-written run files.

**Independent Test**: Run `holodeck test agent.yaml` → edit `agent.yaml` (change `temperature` from 0.7 to 0.2) → load the previously written `EvalRun` → `metadata.agent_config.model.temperature == 0.7`.

**TDD discipline**: Every task marked "(TDD)" writes the failing test first.

**Dependency**: US1 must have persisted an `EvalRun` with `metadata.agent_config: Agent`. US3 hardens and verifies that snapshot fidelity.

---

## Phase 1: Setup

None. US3 operates on existing US1 infrastructure.

---

## Phase 2: Foundational

None.

---

## Phase 3: US3 — Snapshot fidelity & freeze

### Tests first (TDD) — config surface coverage

- [ ] T201 [P] [US3] (TDD) `tests/unit/models/test_eval_run_snapshot.py`: build an `Agent` with `model.provider=openai, model.name=gpt-4o, model.temperature=0.7, model.max_tokens=1024`, construct `EvalRun`, `model_dump_json()` → `model_validate_json()`, assert all model fields round-trip (AC1)
- [ ] T202 [P] [US3] (TDD) `tests/unit/models/test_eval_run_snapshot.py`: build an `Agent` with `embedding_provider.provider=azure_openai, name="text-embedding-3-large", endpoint=..., api_version=...`, round-trip, assert `endpoint`/`api_version` preserved (non-secret env-substituted values — AC2 positive case)
- [ ] T203 [P] [US3] (TDD) `tests/unit/models/test_eval_run_snapshot.py`: build an `Agent` with a `VectorStoreConfig` tool (full config: `collection`, `embedding_model`, `source_paths`, `chunk_size`, etc.), round-trip, assert every field preserved (AC3)
- [ ] T204 [P] [US3] (TDD) `tests/unit/models/test_eval_run_snapshot.py`: build an `Agent` with an `MCPConfig` tool (transport, command, args, env, headers), round-trip, assert every field preserved (AC3)
- [ ] T205 [P] [US3] (TDD) `tests/unit/models/test_eval_run_snapshot.py`: build an `Agent` with a `FunctionConfig` tool and a `PromptConfig` tool, round-trip, assert preserved (AC3)
- [ ] T206 [P] [US3] (TDD) `tests/unit/models/test_eval_run_snapshot.py`: build an `Agent` with `HierarchicalDocumentConfig` tool, round-trip, assert preserved (AC3)
- [ ] T207 [P] [US3] (TDD) `tests/unit/models/test_eval_run_snapshot.py`: build an `Agent` with a full `claude` block — `working_directory`, `permission_mode`, `max_turns`, `extended_thinking.enabled=true, budget_tokens=50000`, `web_search=true`, `bash.enabled=true, excluded_commands=[...]`, `file_system.{read,write,edit}`, `subagents.enabled=true, max_parallel=4`, `allowed_tools=[...]` — round-trip, assert the entire block under `metadata.agent_config.claude` (AC4)
- [ ] T208 [P] [US3] (TDD) `tests/unit/models/test_eval_run_snapshot.py`: build an `Agent` with `evaluations` containing one of each metric type (standard/geval/rag) plus per-metric `model` override, round-trip, assert preserved
- [ ] T209 [P] [US3] (TDD) `tests/unit/models/test_eval_run_snapshot.py`: build an `Agent` with `test_cases` referencing multimodal files (`files: [{path: ./x.png, type: image}]`), round-trip, assert only `path`, `type`, and inline metadata (`sheet`, `range`, `pages`, `description`) are preserved — NO file bytes, NO content hash, NO extracted text (FR-009a)
- [ ] T210 [US3] (TDD) `tests/integration/cli/test_snapshot_is_frozen.py`: run `holodeck test agent.yaml` with `temperature: 0.7`, capture the run file path, edit `agent.yaml` to `temperature: 0.2` (write-through), reload the captured run, assert `metadata.agent_config.model.temperature == 0.7` (AC5)
- [ ] T211 [US3] (TDD) `tests/integration/cli/test_snapshot_reproducibility.py`: load a persisted run, read `metadata.agent_config` only, assert the reader can reconstruct a valid `Agent` model via `Agent.model_validate(run.metadata.agent_config.model_dump())` without touching the live `agent.yaml` (AC6)
- [ ] T212 [US3] (TDD) `tests/unit/models/test_eval_run_snapshot.py`: secret redaction does NOT mask non-secret env-substituted values — e.g., `model.endpoint="https://my-azure.openai.azure.com"` (env-substituted from `${AZURE_ENDPOINT}`) stays as the resolved URL; only fields matching redactor rules are `"***"` (AC2 both sides)
- [ ] T213 [US3] (TDD) `tests/unit/models/test_eval_run_snapshot.py`: when `instructions.inline` is set on the snapshot, the full inline string is preserved in `metadata.agent_config.instructions.inline` (a critical part of reproducibility)
- [ ] T214 [US3] (TDD) `tests/unit/models/test_eval_run_snapshot.py`: when `instructions.file` is set, `metadata.agent_config.instructions.file` preserves the original path string (frontmatter metadata lives separately under `metadata.prompt_version`, not duplicated here)

### Implementation (mostly verification — US1 already persists `Agent`; these tasks ensure every nested discriminator and type survives serialization)

- [ ] T215 [US3] Run the full US3 test suite; for every failing test, identify which field's serialization broke and fix: typical causes are (a) missing `model_config = ConfigDict(extra="forbid")` on a nested model, (b) custom serializers stripping fields, (c) `SecretStr` round-trip needing `Field(..., repr=False)` hygiene
- [ ] T216 [US3] Verify the `tools` discriminated union round-trips — confirm `ToolUnion` uses Pydantic's `discriminator="type"` so each tool sub-type's full shape is preserved. If not, migrate to `Annotated[Union[...], Field(discriminator="type")]` in `src/holodeck/models/tool.py`
- [ ] T217 [US3] Verify the `evaluations.metrics` union round-trips the same way (standard/geval/rag discriminator)
- [ ] T218 [US3] Ensure the snapshot is built IMMEDIATELY at persistence time (not later) so no intermediate mutation can leak in — confirm `build_eval_run_metadata(agent, ...)` in `src/holodeck/lib/eval_run/metadata.py` invokes `agent.model_copy(deep=True)` before redaction to guard against shared-reference mutation (data-model.md §"Snapshot semantics")
- [ ] T219 [US3] Apply redaction (US1 T026) to the deep-copied `Agent` before attaching it to `EvalRunMetadata` — redaction MUST NOT mutate the original `Agent` instance held by the running test execution
- [ ] T220 [US3] Add a module-level docstring to `src/holodeck/lib/eval_run/metadata.py` documenting the snapshot invariants: "deep copy → redact → freeze into `EvalRunMetadata`; the on-disk artifact is authoritative"

**Checkpoint**: All US3 tests green. Quickstart §4 reproducible.

---

## Dependencies

- US1 must be fully implemented (T001–T032) before US3 tests can run meaningfully.
- T201–T214 (TDD tests) must all be written and failing before T215–T220 implementation/fixes.
- T218 depends on US1 T027 (metadata builder exists).

### Parallel Opportunities

```bash
# All TDD tests cover independent Agent configurations:
Task: "test_eval_run_snapshot.py::model_block (T201)"
Task: "test_eval_run_snapshot.py::embedding_provider (T202)"
Task: "test_eval_run_snapshot.py::vectorstore_tool (T203)"
Task: "test_eval_run_snapshot.py::mcp_tool (T204)"
Task: "test_eval_run_snapshot.py::function_prompt_tools (T205)"
Task: "test_eval_run_snapshot.py::hierarchical_document_tool (T206)"
Task: "test_eval_run_snapshot.py::claude_block (T207)"
Task: "test_eval_run_snapshot.py::evaluations (T208)"
Task: "test_eval_run_snapshot.py::multimodal_files_path_only (T209)"
```

---

## Acceptance Scenario Traceability

| AC | Covered by |
|---|---|
| AC1 (Agent → JSON deep copy on disk) | T201–T208, T218 |
| AC2 (secrets redacted, non-secret env values preserved) | T212, T219 |
| AC3 (every tool's full config preserved) | T203–T206 |
| AC4 (entire `claude` block captured) | T207 |
| AC5 (snapshot frozen; later `agent.yaml` edits ignored) | T210 |
| AC6 (reconstructable from JSON alone) | T211 |
| FR-009a (multimodal files: path-only, no bytes/hash/text) | T209 |

---

## Implementation Strategy

1. Write the full config-surface test battery first (T201–T214) — broad coverage is cheap because the models already exist.
2. Run tests; expect most to pass if Pydantic defaults are reasonable, but at least one nested discriminator will likely round-trip imperfectly.
3. Fix each failure in the underlying model (T215–T217); these fixes benefit the whole codebase, not just US3.
4. Harden the snapshot path with deep-copy + redaction ordering (T218–T219).
5. Validate quickstart §4 by hand.
