# Implementation Plan: Decouple Dead Semantic Kernel Harness Leftovers

**Spec:** `specs/035-openai-agents-backend/spec.md`
**Scope of this plan:** Remove Semantic Kernel (SK) usage from the surfaces that were
**orphaned when the SK agent backend (`SKBackend`/`AgentFactory`) was deleted** in feature 035 —
the SK MCP factory, the never-wired `tool_filter` module + its config surface, and a couple of
dead SK-era stubs. **SK is deliberately kept** as the backbone of the vectorstore / RAG layer.
**Status:** Draft for review

## Overview

After 035 removed the SK agent-execution path, several SK-coupled surfaces were left stranded with
**zero live callers**. They still pull `semantic_kernel` symbols (`Kernel`, `KernelFunction`,
`MCPStdioPlugin`, …) and carry test/sys.modules-mock weight. This plan deletes those dead
surfaces so the only remaining SK usage is the part we actually run.

**This is NOT "drop the `semantic-kernel` dependency."** SK remains load-bearing for the
vectorstore / embedding / context-generation / chunking layer (confirmed live via
`holodeck chat`/`test` → `ClaudeBackend` RAG path). The `semantic-kernel` pin in `pyproject.toml`
**stays**. Full removal of those RAG-layer SK abstractions is explicitly out of scope (large,
separate effort) and tracked as a follow-up.

## Decisions (confirmed with user)

1. **Full sweep.** Remove MCP **and** `tool_filter` **and** the dead SK-era stubs in one pass.
2. **Remove the `tool_filtering` config surface entirely** — accepted as a **breaking config
   change**. `ToolFilterConfig`, the `Agent.tool_filtering` field, the validator, the schema
   entry, and the sample that uses it all go. (Claude manages tool selection natively; the field
   was parsed-but-ignored.)
3. **SK stays for vectorstores/RAG.** `pyproject.toml` `semantic-kernel` pin is untouched.
4. **Keep observability SK telemetry hooks.** SK still executes for RAG embeddings + LLM context
   generation, so `enable_semantic_kernel_telemetry`, the `"semantic_kernel"` logger-suppression
   entry, and the `otel_bridge` default namespace remain meaningful and stay. Only the **unused**
   `include_semantic_kernel_metrics` config field is removed.

## Migration note (breaking change)

**Removed config field: `tool_filtering`.** As of this change the top-level `tool_filtering:`
block in `agent.yaml` is no longer a recognized field. Because the `Agent` model uses
`extra="forbid"`, any agent config that still sets `tool_filtering` will **fail validation** with
an "extra field" error. The field had no runtime effect (it was parsed and ignored — Claude
manages tool selection natively), so the fix is simply to **delete the `tool_filtering:` block**
from affected `agent.yaml` files. No behavior changes.

## What stays (do NOT touch — the live SK RAG surface)

| File | SK usage kept | Why |
|------|---------------|-----|
| `src/holodeck/lib/vector_store.py` | `data.vector` connectors, `@vectorstoremodel`, `VectorStoreField`, all provider collections, `PostgresSettings` | Core vector store abstraction (live) |
| `src/holodeck/lib/tool_initializer.py` | `create_embedding_service` (OpenAI/Azure/Ollama TextEmbedding); `_create_chat_service_from_config` (chat services) | Embeddings + context-gen factories (live) |
| `src/holodeck/lib/llm_context_generator.py` | `ChatHistory`, `ChatCompletionClientBase`, `PromptExecutionSettings`, `OpenAIChatPromptExecutionSettings` | Contextual-retrieval generation (live) |
| `src/holodeck/lib/text_chunker.py` | `split_plaintext_paragraph` | Chunking utility (live, RAG ingestion) |
| `src/holodeck/lib/observability/instrumentation.py` + `providers.py` | `enable_semantic_kernel_telemetry` | SK still runs for RAG → telemetry meaningful (Decision 4) |
| `src/holodeck/lib/logging_config.py` | `"semantic_kernel"` in `THIRD_PARTY_LOGGERS` | Benign log suppression for the live SK code |
| `src/holodeck/lib/backends/otel_bridge.py` | `_DEFAULT_FILTER_NAMESPACES = ["semantic_kernel"]` | Default still correct |

## Architecture decisions

- **Live MCP path is unaffected.** MCP tools run through `src/holodeck/lib/backends/mcp_bridge.py`
  (`build_claude_mcp_configs` → Claude SDK `McpStdioServerConfig`), driven from
  `claude_backend.py` via the `MCPTool` model in `holodeck.models.tool`. The SK MCP factory
  package (`holodeck/tools/mcp/`) has **zero importers** and is independent of `MCPTool`.
- **`holodeck/tools/mcp/` is entirely dead.** Package contents: `__init__.py`, `factory.py`
  (the only SK import), `errors.py`, `utils.py`. A repo-wide grep for `holodeck.tools.mcp` outside
  the package returns nothing — delete the whole package, not just `factory.py`.
- **`tool_filter` SK internals are never instantiated.** `index.py`/`manager.py` (SK `Kernel`-based
  semantic tool selection) have no live caller; only `models.py:ToolFilterConfig` is referenced
  (the ignored agent field). Per Decision 2 the entire module + config surface is removed.
- **Breaking change is contained.** The only in-repo consumer of the removed config is
  `sample/research-agent/agent.yaml`; it is updated in the same change. A migration note is added.

## Task List

### Phase 1 — Remove the dead SK MCP factory package

#### Task 1: Delete the `holodeck/tools/mcp/` package + tests
**Description:** Remove the dead SK-based MCP plugin factory wholesale. The live MCP path
(`mcp_bridge.py` → Claude SDK) and the `MCPTool` model are untouched.
**Acceptance criteria:**
- [ ] `src/holodeck/tools/mcp/` (`__init__.py`, `factory.py`, `errors.py`, `utils.py`) deleted.
- [ ] `tests/unit/tools/mcp/` deleted.
- [ ] No remaining import of `holodeck.tools.mcp` / `create_mcp_plugin` / `MCPStdioPlugin`
      anywhere in `src/` or `tests/`.
**Verification:** `grep -rn "holodeck.tools.mcp\|create_mcp_plugin\|MCPStdioPlugin" src tests`
returns nothing; `make test-unit` green; an `MCPTool` agent still builds Claude MCP configs
(existing `mcp_bridge` tests pass).
**Dependencies:** None
**Files (delete):** `src/holodeck/tools/mcp/{__init__,factory,errors,utils}.py`,
`tests/unit/tools/mcp/{__init__,test_config,test_errors,test_factory,test_utils}.py`
**Scope:** S

### Checkpoint A — MCP decoupled
- [ ] Full unit suite green; `mcp_bridge` (live MCP) tests unaffected; grep clean.

### Phase 2 — Remove the never-wired SK `tool_filter` module + config surface

#### Task 2: Delete the SK `tool_filter` module
**Description:** Remove the orphaned SK `Kernel`-based tool-filtering module (index + manager +
models + package init) and its tests.
**Acceptance criteria:**
- [ ] `src/holodeck/lib/tool_filter/` deleted (`__init__.py`, `index.py`, `manager.py`,
      `models.py`).
- [ ] `tests/unit/lib/tool_filter/` deleted.
- [ ] No import of `holodeck.lib.tool_filter` remains except the ones removed in Task 3.
**Verification:** `grep -rn "holodeck.lib.tool_filter\|ToolFilterManager\|ToolIndex" src tests`
returns only the Task-3 sites (then none after Task 3). `make type-check` clean.
**Dependencies:** None (the SK internals have no live caller)
**Files (delete):** `src/holodeck/lib/tool_filter/{__init__,index,manager,models}.py`,
`tests/unit/lib/tool_filter/{__init__,test_index,test_manager,test_models}.py`
**Scope:** S

#### Task 3: Remove the `tool_filtering` agent-config surface (breaking)
**Description:** Excise the parsed-but-ignored `tool_filtering` config end to end: model field,
validator, Claude-backend call, schema, sample, and its tests.
**Acceptance criteria:**
- [ ] `models/agent.py`: removed `from holodeck.lib.tool_filter.models import ToolFilterConfig`
      (line 18) and the `tool_filtering` field (lines 90–97).
- [ ] `backends/validators.py`: `validate_tool_filtering` (≈281–302) removed.
- [ ] `backends/claude_backend.py`: removed the `validate_tool_filtering` import (line 106) and
      its call (≈2034).
- [ ] `schemas/agent.schema.json`: `$defs.ToolFilterConfig` (≈3178–3232) and the `tool_filtering`
      property (≈3763–3766) removed; schema still validates.
- [ ] `sample/research-agent/agent.yaml`: `tool_filtering:` block (lines 113–119) removed; sample
      still loads.
- [ ] `tests/unit/lib/backends/test_validators.py`: `ToolFilterConfig` import (line 20) +
      `tool_filtering` tests (≈565, 574) removed.
- [ ] A migration note records the breaking change (CHANGELOG or the spec's follow-up section).
**Verification:** `grep -rn "tool_filtering\|ToolFilterConfig" src tests schemas sample` returns
nothing; `holodeck` loads `sample/research-agent/agent.yaml` without error; `make test` green.
**Dependencies:** Task 2
**Files (edit):** `src/holodeck/models/agent.py`, `src/holodeck/lib/backends/validators.py`,
`src/holodeck/lib/backends/claude_backend.py`, `schemas/agent.schema.json`,
`sample/research-agent/agent.yaml`, `tests/unit/lib/backends/test_validators.py`
**Scope:** M

### Checkpoint B — tool_filter gone
- [ ] Full suite green; schema validates; research-agent sample loads; grep clean.

### Phase 3 — Dead SK-era stub & observability orphan

#### Task 4: Remove the dead `to_semantic_kernel_function` stub
**Description:** Delete the unused SK-named stub on the hierarchical document tool (only its own
tests call it; it does not even import SK).
**Acceptance criteria:**
- [ ] `to_semantic_kernel_function` removed from
      `src/holodeck/tools/hierarchical_document_tool.py` (≈1402–1417).
- [ ] Its tests removed from `tests/unit/tools/test_hierarchical_document_tool.py` (≈1622–1688).
**Verification:** `grep -rn "to_semantic_kernel_function" src tests` returns nothing; the
hier-doc tool's remaining tests pass.
**Dependencies:** None
**Files (edit):** `src/holodeck/tools/hierarchical_document_tool.py`,
`tests/unit/tools/test_hierarchical_document_tool.py`
**Scope:** XS

#### Task 5: Remove the unused `include_semantic_kernel_metrics` config field
**Description:** Drop the orphaned metrics config field (defined but never read). Keep all other
observability SK hooks (Decision 4).
**Acceptance criteria:**
- [ ] `include_semantic_kernel_metrics` removed from `src/holodeck/models/observability.py`
      (≈121–124).
- [ ] Any test asserting that field removed/updated
      (`tests/unit/models/test_observability.py`).
- [ ] `enable_semantic_kernel_telemetry`, the `"semantic_kernel"` logger entry, and the
      `otel_bridge` default namespace are **left intact**.
**Verification:** `grep -rn "include_semantic_kernel_metrics" src tests docs` returns nothing;
`make type-check` clean; observability tests green.
**Dependencies:** None
**Files (edit):** `src/holodeck/models/observability.py`, `tests/unit/models/test_observability.py`
**Scope:** XS

### Phase 4 — Docs & final verification

#### Task 6: Documentation cleanup
**Description:** Remove user-facing docs for the deleted surfaces.
**Acceptance criteria:**
- [ ] `docs/api/tool-filter.md` deleted.
- [ ] `tool_filtering` references removed from `docs/guides/tools.md` and `docs/api/backends.md`.
- [ ] SK MCP factory references removed from `docs/api/tools.md`.
- [ ] `AGENTS.md` tool-filter / SK-MCP mentions updated (keep the "vectorstore (Semantic Kernel)"
      mention — that layer stays).
- [ ] Historical `specs/0xx/**` references left as-is (point-in-time records).
**Verification:** `grep -rn "tool_filtering\|tool-filter\|create_mcp_plugin" docs AGENTS.md`
returns nothing (excluding `specs/`).
**Dependencies:** Tasks 1–3
**Files (edit/delete):** `docs/api/tool-filter.md` (delete), `docs/guides/tools.md`,
`docs/api/backends.md`, `docs/api/tools.md`, `AGENTS.md`
**Scope:** S

#### Task 7: Full verification + SK-surface confirmation
**Description:** Prove the kept SK RAG surface is intact and the dead surfaces are gone.
**Acceptance criteria:**
- [ ] `make format lint type-check security` clean; `make test` (parallel) green.
- [ ] `python -c "import semantic_kernel"` still resolves (SK kept on purpose).
- [ ] `grep -rn "semantic_kernel" src/` lists **only** the "What stays" surfaces above (vector
      store, tool_initializer, llm_context_generator, text_chunker, observability instrumentation,
      logging_config, otel_bridge) — no `tool_filter`, no `tools/mcp`.
- [ ] A vectorstore/RAG agent still ingests + answers a grounded query (existing integration
      tests, creds-gated, unchanged).
**Verification:** Commands above; spot-check an existing RAG integration test.
**Dependencies:** Tasks 1–6
**Files:** none (verification only)
**Scope:** S

### Checkpoint C — Complete
- [ ] All acceptance criteria met; SK confined to the RAG layer; full suite green; docs consistent.

## Risks and mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Removing `tool_filtering` breaks existing user `agent.yaml` files | Med | Decision 2 accepts it; update the in-repo sample, add a migration/CHANGELOG note, call out in release notes |
| Accidentally deleting a live MCP path | High | Verified live MCP = `mcp_bridge.py` + `MCPTool` (in `holodeck.models.tool`); the deleted package has zero importers — gate on grep + `mcp_bridge` tests |
| Leftover sys.modules SK mocks in deleted tests cause xdist bleed | Low | Deleting the test files removes the mocks; full parallel suite is the gate (see prior `tests/unit/tools/test_hierarchical_document_tool.py` mock-leak fix) |
| Schema edit malforms `agent.schema.json` | Med | JSON-validate the schema + load the sample agents after the edit (Task 3 verification) |
| Hidden reader of `include_semantic_kernel_metrics` | Low | grep confirms zero readers before deletion (Task 5) |

## Out of scope (tracked follow-ups)

- **Full removal of SK from the vectorstore/embedding/context-gen/chunking layer** (would replace
  `data.vector` connectors, `TextEmbedding`/`ChatCompletion` services, `split_plaintext_paragraph`
  — likely with LiteLLM/direct SDKs + per-provider vector clients). Large; separate spec.
- **Dropping the `semantic-kernel` pin from `pyproject.toml`** — only possible after the above.
- **Replacing SK GenAI telemetry** with native OTel GenAI spans for the RAG calls.

## Open questions

- **Release/version bump for the breaking `tool_filtering` removal** — confirm whether this rides a
  minor bump + CHANGELOG entry, or needs a deprecation window. (Plan assumes immediate removal per
  Decision 2.)
