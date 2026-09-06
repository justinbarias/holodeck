# Implementation Plan: Decouple Dead Semantic Kernel Harness Leftovers

**Execution handoff:** Use the [completion execution plan](2026-09-06-complete-035.md) for current task order, acceptance gates, and progress.
This record retains the reconciled design and historical task evidence.

**Spec:** `docs/product-specs/035-openai-agents-backend/spec.md`
**Status:** Implementation complete; verification gates not re-established by this reconciliation.
**Reconciled:** 2026-09-06 against `b585e80` and implementation commit `3805e80` (#338).
**Scope:** Delete orphaned SK MCP factories, tool filtering, and dead stubs. Retain live retrieval dependencies.

## Evidence and status convention

Checked items below mean the implementation is present in the current tree, established by source,
configuration, test, and Git inspection. See the [reconciliation report](reconciliation.md) for
fresh focused test results; these do not establish every historical full-suite gate below.
Unchecked validation items identify evidence still needed to close the original gates.
The task IDs preserve the original plan's work breakdown.

The original draft described SK embedding/chat services as retained. The subsequent
[LiteLLM plan](plan-litellm-embeddings-contextgen.md) replaced those services. The current SK
runtime dependency is vector-store abstractions/connectors and text splitting, plus retained
telemetry configuration and logging/filter defaults. Agent execution already uses native backends.

## Decisions and migration

- Remove the entire dead SK MCP package; retain native backend MCP bridges and `MCPTool`.
- Remove the parsed-but-ignored `tool_filtering` field immediately. Agent configuration forbids
  extra fields, so users must delete that block from their YAML. This breaking change is recorded
  in `docs/CHANGELOG.md`; a deprecation flag was not implemented.
- Retain the `semantic-kernel` dependency until connectors and chunking are replaced.
- Retain SK observability hooks at this stage. Their former embedding/chat rationale is superseded
  by LiteLLM telemetry; reassess the remaining hooks during final decommission.

## Task reconciliation

### Task 1 — Delete the dead SK MCP package

- [x] `src/holodeck/tools/mcp/` and `tests/unit/tools/mcp/` deleted.
- [x] No `holodeck.tools.mcp`, `create_mcp_plugin`, or `MCPStdioPlugin` references in `src/` or `tests/`.
- [x] Native MCP implementation remains in `lib/backends/mcp_bridge.py` and the OpenAI backend.
- [ ] Re-establish the original full unit-suite and live-MCP regression gate.

**Checkpoint A:** Deletion and search criteria complete; original runtime validation not rerun.

### Task 2 — Delete the unused tool-filter module

- [x] `src/holodeck/lib/tool_filter/` and `tests/unit/lib/tool_filter/` deleted.
- [x] No `holodeck.lib.tool_filter`, `ToolFilterManager`, or `ToolIndex` references in source/tests.
- [ ] Re-establish the original type-check gate.

### Task 3 — Remove the tool-filter configuration surface

- [x] `ToolFilterConfig` import and `Agent.tool_filtering` field removed.
- [x] `validate_tool_filtering` and its Claude-backend invocation removed.
- [x] Schema definition/property and research-agent sample block removed.
- [x] Validator tests for the removed field removed.
- [x] Breaking migration note published in `docs/CHANGELOG.md`.
- [x] No `tool_filtering` or `ToolFilterConfig` references in `src/`, `tests/`, `schemas/`, or `sample/`.
- [ ] Re-establish schema validation, sample loading, and the original full-suite gate.

**Checkpoint B:** Configuration removal complete; runtime/schema validation not rerun.

### Task 4 — Remove the hierarchical-tool stub

- [x] `to_semantic_kernel_function` and its tests removed; source/test search is empty.
- [ ] Re-establish the hierarchical-document regression-test gate.

### Task 5 — Remove the unused observability field

- [x] `include_semantic_kernel_metrics` removed from the model and tests.
- [x] `enable_semantic_kernel_telemetry`, logger suppression, and bridge namespace defaults retained.
- [ ] Re-establish observability tests and the original type-check gate.

Searches for removed fields must exclude historical plans/design records, which intentionally
retain their names; the old requirement for zero matches across all documentation was incorrect.

### Task 6 — Documentation cleanup

- [x] `docs/api/tool-filter.md` deleted.
- [x] Tool-filter references removed from `docs/guides/tools.md` and `docs/api/backends.md`.
- [x] SK MCP factory references removed from `docs/api/tools.md`.
- [x] Current `AGENTS.md` has no tool-filter/SK-MCP guidance. Its former detailed vector-store
  description is superseded by the harness architecture map, not a missing restoration task.
- [x] Historical feature documents remain point-in-time records in the migrated product/design trees.

### Task 7 — Final verification and retained surface

- [x] Source search confirms deleted modules/stubs/configuration are absent.
- [x] Remaining SK references match the inventory below; `tool_initializer.py` and
  `llm_context_generator.py` now use LiteLLM, as established by the later plan.
- [ ] Re-establish format, lint, type, security, and full parallel test gates.
- [ ] Verify an installed `semantic_kernel` import and a real RAG ingest/grounded-answer run.

**Checkpoint C:** Implementation complete. The historical all-checks-green acceptance statement
is not established by this source reconciliation; keep the verification items distinct from code work.

## Remaining SK inventory and follow-ups

| Surface | Evidence | Remaining work |
| --- | --- | --- |
| Vector abstractions and 11 collection mappings | `src/holodeck/lib/vector_store.py` | Replace record decorators/fields, collection contracts, provider clients, and Postgres settings; preserve provider/search behavior. |
| Chunking | `src/holodeck/lib/text_chunker.py` imports `split_plaintext_paragraph` | Replace splitter and verify chunk-boundary/overlap behavior. |
| Observability | `lib/observability/{instrumentation,providers,__init__}.py` | Remove obsolete SK setup/exports after checking retained connector telemetry. |
| Logger/filter defaults | `lib/logging_config.py`, `lib/backends/otel_bridge.py`, `models/observability.py` | Reassess/remove SK defaults and update tests/configuration docs. |
| Test dependencies | SK vector fixtures/mocks; `tests/integration/test_structured_vectorstore.py` imports `OllamaTextEmbedding` | Migrate fixtures and the direct embedding client; remove obsolete SK mocks. |
| Packaging/docs | `pyproject.toml`, observability API and vector-store guide | Remove dependency constraints/lock entries and update docs after runtime/test imports are gone. |

Embedding and contextual-chat replacement is implemented in the
[LiteLLM plan](plan-litellm-embeddings-contextgen.md), with its own outstanding acceptance gaps.
Full SK decommission still needs a separate scoped implementation plan; this cleanup did not provide one.
