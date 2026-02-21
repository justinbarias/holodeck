# Tasks: Native Claude Agent SDK Integration — Phase 5: Tool Adapters

**Feature**: `021-claude-agent-sdk`
**Input**: Design documents from `/specs/021-claude-agent-sdk/`
**Scope**: Phase 5 only (Tool Adapters). Continues from `tasks-phase4.md` (T041).
**Approach**: TDD — test tasks are written and **must fail** before implementation tasks begin
**References**: Each task includes a document reference `(doc:L<line>)` pointing to the relevant design decision
**FRs**: FR-003, FR-004, FR-023 (plan.md:L307-309)

---

## Format: `[ID] [P?] [US?] Description (doc:Lline)`

- **[P]**: Task is parallelizable (different files, no unresolved dependencies)
- **[US2]**: Maps to User Story 2 (Vectorstore and Document Tools Work with Claude-Native Agents)
- **TDD rule**: Test tasks MUST precede their implementation tasks

---

## Phase 5 Overview

**Purpose**: Create tool adapters that wrap HoloDeck's existing `VectorStoreTool.search()` and `HierarchicalDocumentTool.search()` methods as Claude Agent SDK `@tool`-decorated functions, bundle them into an in-process MCP server via `create_sdk_mcp_server()`, and register that server in `ClaudeAgentOptions.mcp_servers`. After Phase 5, vectorstore and hierarchical document tools are callable from the Claude-native subprocess.

**User Story**: US2 — Vectorstore and Document Tools Work with Claude-Native Agents (spec.md:L44-58)

**Files created**:
- `src/holodeck/lib/backends/tool_adapters.py` (new — adapter classes + server builder)

**Files modified**:
- `src/holodeck/lib/backends/__init__.py` (updated — export new types)

**Files created (tests)**:
- `tests/unit/lib/backends/test_tool_adapters.py` (new — unit tests for all adapters)

**Key design decisions**:
- **Factory function pattern**: Python closures capture by reference; a factory function is mandatory to avoid all adapters sharing the last loop variable (plan.md:L333-343, research.md:L147-156)
- **In-process MCP server**: `create_sdk_mcp_server()` bundles `@tool` functions into an `McpSdkServerConfig` that the SDK subprocess communicates with transparently (research.md:L142-169)
- **Tool name convention**: `mcp__holodeck_tools__<tool_name>_search` — must match entries in `allowed_tools` (research.md:L165-171)
- **Embedding initialization**: Tools still use SK embedding services via `embedding_provider` credentials, initialized in the parent process before subprocess spawn (plan.md:L346-349)
- **Return format**: VectorStoreTool.search() returns `str`; HierarchicalDocumentTool.search() returns `list[SearchResult]` — each result is serialized via `SearchResult.format()` (the canonical per-result formatter) rather than custom serialization (vectorstore_tool.py:L751, hierarchical_document_tool.py:L947, hybrid_search.py:L146)
- **Empty result standardization**: Both adapters return `"No results found."` for empty search results (consistent behavior across tool types)
- **Defensive initialization guard**: Each adapter handler checks `tool.is_initialized` before calling `search()` — raises `BackendSessionError` if False (guards against contract violations from Phase 8)
- **Tool description truncation**: `@tool` description is `f"Search {config.name}: {config.description}"` truncated to 200 chars max
- **McpSdkServerConfig is a TypedDict**: `create_sdk_mcp_server()` returns a `TypedDict` (not a dataclass) — tests should use duck-type key checks, not `isinstance()`

---

## Phase 5A: Tests (write these first — they MUST fail before implementation)

> **NOTE**: Write ALL test tasks before implementing. Run `pytest tests/unit/lib/backends/test_tool_adapters.py -n auto` to confirm all fail with `ImportError` or `ModuleNotFoundError` before starting Phase 5B.

### Tests for `VectorStoreToolAdapter`

- [ ] T042 [P] [US2] Write unit tests for `VectorStoreToolAdapter` in `tests/unit/lib/backends/test_tool_adapters.py`: (plan.md:L312-320, research.md:L142-169, quickstart.md:L103-137)

  1. `VectorStoreToolAdapter.__init__()` accepts a `VectorstoreToolConfig` (from `holodeck.models.tool`) and a `VectorStoreTool` instance (from `holodeck.tools.vectorstore_tool`) — stores both as attributes
  2. `to_sdk_tool()` returns an `SdkMcpTool` (from `claude_agent_sdk`) with correct `name` matching `f"{config.name}_search"` pattern
  3. `to_sdk_tool()` produces a tool whose async handler calls `tool_instance.search(query)` and returns `{"content": [{"type": "text", "text": <search_result>}]}` when invoked with `{"query": "test query"}`
  4. When `tool_instance.search()` returns an empty string (no results), the adapter handler returns `{"content": [{"type": "text", "text": "No results found."}]}` (standardized empty-result message)
  5. When `tool_instance.search()` raises an exception, the adapter handler propagates the error (does NOT silently swallow it)
  6. Factory function closure: create two distinct mock `VectorStoreTool` instances with different return values, pass both configs + instances through `create_tool_adapters()`, get two adapters, invoke each adapter's SDK tool handler, and assert each called the correct mock instance (tests the factory function pattern from plan.md:L333-343)
  7. Defensive guard: when `tool_instance.is_initialized` is `False`, the adapter handler raises `BackendSessionError` with message identifying the uninitialized tool — does NOT call `search()`

  Mock `VectorStoreTool` — do NOT make real embedding/search calls. Mock `claude_agent_sdk.tool` decorator to verify it's called with correct arguments `(name, description, input_schema)`.

### Tests for `HierarchicalDocToolAdapter`

- [ ] T043 [P] [US2] Write unit tests for `HierarchicalDocToolAdapter` in `tests/unit/lib/backends/test_tool_adapters.py`: (plan.md:L321-326, research.md:L142-169)

  1. `HierarchicalDocToolAdapter.__init__()` accepts a `HierarchicalDocumentToolConfig` (from `holodeck.models.tool`) and a `HierarchicalDocumentTool` instance (from `holodeck.tools.hierarchical_document_tool`) — stores both as attributes
  2. `to_sdk_tool()` returns an `SdkMcpTool` with correct `name` matching `f"{config.name}_search"` pattern
  3. `to_sdk_tool()` produces a tool whose async handler calls `tool_instance.search(query)` and serializes each `SearchResult` using `result.format()` (the canonical formatter from `holodeck.lib.hybrid_search.SearchResult`), joining results with `"\n---\n"`. Returns `{"content": [{"type": "text", "text": <serialized_results>}]}`
  4. Serialization via `SearchResult.format()` includes: score, source path, location (parent_chain), content, and definitions context — the canonical human-readable format
  5. When `tool_instance.search()` returns an empty list (no results), the adapter handler returns `{"content": [{"type": "text", "text": "No results found."}]}` (standardized empty-result message, consistent with VectorStoreToolAdapter)
  6. When `tool_instance.search()` raises an exception, the adapter handler propagates the error
  7. Defensive guard: when `tool_instance._initialized` is `False`, the adapter handler raises `BackendSessionError` with message identifying the uninitialized tool — does NOT call `search()`

  Mock `HierarchicalDocumentTool` and `SearchResult` — do NOT make real embedding/search calls. For `SearchResult.format()`, create mock SearchResult dataclass instances with known field values and verify the adapter's output matches `"\n---\n".join(r.format() for r in results)`.

### Tests for `build_holodeck_sdk_server`

- [ ] T044 [P] [US2] Write unit tests for `build_holodeck_sdk_server()` in `tests/unit/lib/backends/test_tool_adapters.py`: (plan.md:L327-331, research.md:L157-168, quickstart.md:L110-137)

  1. Returns a tuple of `(server_config, list[str])` — server config (a `dict` with keys `"type"`, `"name"`, `"instance"` — `McpSdkServerConfig` is a TypedDict) + allowed tool names. Verify via duck-type key checks: `assert config["type"] == "sdk"` and `assert "name" in config` and `assert "instance" in config`
  2. With no adapters (empty list), returns a server with no tools and an empty allowed_tools list
  3. With one `VectorStoreToolAdapter`, returns a server named `"holodeck_tools"` and allowed_tools `["mcp__holodeck_tools__<name>_search"]`
  4. With one `HierarchicalDocToolAdapter`, returns a server named `"holodeck_tools"` and allowed_tools `["mcp__holodeck_tools__<name>_search"]`
  5. With mixed adapters (2 vectorstore + 1 hierarchical), returns a server with 3 tools and 3 allowed_tool entries — each with the correct `mcp__holodeck_tools__<name>_search` prefix
  6. Allowed tool name format follows convention: `mcp__holodeck_tools__<config.name>_search` (research.md:L165-171)
  7. Calls `create_sdk_mcp_server(name="holodeck_tools", tools=<list_of_sdk_tools>)` with the correct arguments

  Mock `claude_agent_sdk.create_sdk_mcp_server` to verify arguments.

### Tests for `create_tool_adapters`

- [ ] T045 [P] [US2] Write unit tests for `create_tool_adapters()` factory function in `tests/unit/lib/backends/test_tool_adapters.py`: (plan.md:L346-349, research.md:L244-250)

  1. Given a list of tool configs and corresponding initialized tool instances, returns `list[VectorStoreToolAdapter | HierarchicalDocToolAdapter]`
  2. Filters only `VectorstoreTool` and `HierarchicalDocumentToolConfig` types from the tool configs — ignores `MCPTool`, `FunctionTool`, `PromptTool`
  3. Matches each config to its initialized tool instance by `config.name`
  4. Raises `BackendInitError` with message `f"No initialized instance found for tool '{config.name}' (type: {config.type}). Ensure tool initialization completed before creating adapters."` if a vectorstore/hierarchical-document config has no matching initialized tool instance (e.g., tool initialization failed)
  5. Returns an empty list when no vectorstore or hierarchical document tools are configured

  Mock tool instances — do NOT create real tools.

**Checkpoint**: All test tasks written. Run `pytest tests/unit/lib/backends/test_tool_adapters.py -n auto` — all MUST fail.

---

## Phase 5B: Implementation

### Core: `tool_adapters.py`

- [ ] T046 [US2] Create `src/holodeck/lib/backends/tool_adapters.py` implementing: (plan.md:L311-353, research.md:L142-171, quickstart.md:L103-137, data-model.md:L362-376)

  **Class: `VectorStoreToolAdapter`**
  - `__init__(self, config: VectorstoreToolConfig, instance: VectorStoreTool)` — stores config and tool instance as attributes
  - `to_sdk_tool(self) -> SdkMcpTool` — uses the factory function pattern `_make_vectorstore_search_fn()` to create a `@tool`-decorated async function that:
    1. Checks `instance.is_initialized` — raises `BackendSessionError(f"Tool '{config.name}' is not initialized")` if `False`
    2. Calls `instance.search(args["query"])`
    3. If result is empty string: returns `{"content": [{"type": "text", "text": "No results found."}]}`
    4. Otherwise returns `{"content": [{"type": "text", "text": result}]}`
  - Tool registration: `@tool(f"{config.name}_search", _truncate_description(f"Search {config.name}: {config.description}"), {"query": str})` — third param is `input_schema` per confirmed SDK API (research.md:L57). Description truncated to 200 chars max via `_truncate_description()` helper.
  - Uses factory function to avoid closure capture bug (plan.md:L333-343)

  **Class: `HierarchicalDocToolAdapter`**
  - `__init__(self, config: HierarchicalDocumentToolConfig, instance: HierarchicalDocumentTool)` — stores config and tool instance
  - `to_sdk_tool(self) -> SdkMcpTool` — uses factory function pattern `_make_hierarchical_search_fn()` to create a `@tool`-decorated async function that:
    1. Checks `instance._initialized` — raises `BackendSessionError(f"Tool '{config.name}' is not initialized")` if `False`
    2. Calls `instance.search(args["query"])`
    3. Serializes `list[SearchResult]` using `SearchResult.format()`: `"\n---\n".join(r.format() for r in results)` — uses the canonical per-result formatter from `holodeck.lib.hybrid_search`
    4. If results list is empty: returns `{"content": [{"type": "text", "text": "No results found."}]}`
    5. Otherwise returns `{"content": [{"type": "text", "text": serialized_text}]}`
  - Tool registration: `@tool(f"{config.name}_search", _truncate_description(f"Search {config.name}: {config.description}"), {"query": str})`. Description truncated to 200 chars max.

  **Private factory functions** (module-level or static, NOT closures in loops):
  ```python
  def _make_vectorstore_search_fn(
      instance: VectorStoreTool, name: str, description: str
  ) -> SdkMcpTool: ...

  def _make_hierarchical_search_fn(
      instance: HierarchicalDocumentTool, name: str, description: str
  ) -> SdkMcpTool: ...
  ```

  **Function: `create_tool_adapters()`**
  ```python
  def create_tool_adapters(
      tool_configs: list[ToolUnion],
      tool_instances: dict[str, VectorStoreTool | HierarchicalDocumentTool],
  ) -> list[VectorStoreToolAdapter | HierarchicalDocToolAdapter]:
  ```
  - Iterates `tool_configs`, filters for `VectorstoreTool` and `HierarchicalDocumentToolConfig`
  - Matches each to its initialized instance by `config.name` lookup in `tool_instances` dict
  - Raises `BackendInitError` if a matching instance is not found
  - Returns list of adapter objects

  **Function: `build_holodeck_sdk_server()`**
  ```python
  def build_holodeck_sdk_server(
      adapters: list[VectorStoreToolAdapter | HierarchicalDocToolAdapter],
  ) -> tuple[McpSdkServerConfig, list[str]]:
  ```
  - Calls `adapter.to_sdk_tool()` on each adapter to collect SDK tools
  - Calls `create_sdk_mcp_server(name="holodeck_tools", tools=sdk_tools)` to create server config
  - Builds allowed_tools list: `[f"mcp__holodeck_tools__{adapter.config.name}_search" for adapter in adapters]`
  - Returns `(server_config, allowed_tool_names)`

  **Private helper**:
  ```python
  def _truncate_description(desc: str, max_len: int = 200) -> str:
      """Truncate tool description to max_len chars for SDK tool manifest."""
      if len(desc) <= max_len:
          return desc
      return desc[: max_len - 3] + "..."
  ```

  **Imports**:
  ```python
  from claude_agent_sdk import tool, create_sdk_mcp_server, SdkMcpTool
  from claude_agent_sdk import McpSdkServerConfig  # TypedDict returned by create_sdk_mcp_server
  from holodeck.models.tool import VectorstoreTool as VectorstoreToolConfig
  from holodeck.models.tool import HierarchicalDocumentToolConfig, ToolUnion
  from holodeck.tools.vectorstore_tool import VectorStoreTool
  from holodeck.tools.hierarchical_document_tool import HierarchicalDocumentTool
  from holodeck.lib.hybrid_search import SearchResult
  from holodeck.lib.backends.base import BackendInitError, BackendSessionError
  ```

### Update: `backends/__init__.py` exports

- [ ] T047 [P] [US2] Update `src/holodeck/lib/backends/__init__.py` to export new types: (plan.md:L311)
  - Export from `tool_adapters`: `VectorStoreToolAdapter`, `HierarchicalDocToolAdapter`, `build_holodeck_sdk_server`, `create_tool_adapters`
  - Add to `__all__` list
  - Verify existing exports remain unchanged

**Checkpoint**: All implementation tasks complete. Run `pytest tests/unit/lib/backends/test_tool_adapters.py -n auto` — all tests from T042–T045 MUST pass.

---

## Phase 5C: Verification & Quality

- [ ] T048 Run full existing unit test suite: `make test-unit` — zero regressions. New tests in `test_tool_adapters.py` must pass alongside all existing tests. Pay special attention to: (plan.md:L553-560)
  - `tests/unit/lib/backends/` — all backend tests (base, sk_backend, selector, validators, tool_adapters)
  - `tests/unit/models/` — model tests (claude_config, agent_extensions, llm_extensions)

- [ ] T049 Run code quality checks — all MUST pass: (plan.md:L553-560)
  ```bash
  make format         # Black + Ruff formatting
  make lint-fix       # Auto-fix linting issues
  make type-check     # MyPy strict mode — all new code must pass
  make security       # Bandit + Safety + detect-secrets
  ```

---

## Dependencies & Execution Order

### Prerequisites from Earlier Phases

Phase 5 depends on:
- `src/holodeck/lib/backends/__init__.py` exists (T007, Phase 1)
- `tests/unit/lib/backends/__init__.py` exists (T007b, Phase 1)
- `src/holodeck/lib/backends/base.py` exists with `BackendInitError` (T008, Phase 1)
- `src/holodeck/lib/backends/sk_backend.py` exists (T030, Phase 4)
- `src/holodeck/lib/backends/selector.py` exists (T031, Phase 4)
- Phase 4 Integration Gate passed (T041) — all existing tests green

### Within Phase 5: TDD Order

```
1. Write all test tasks (T042–T045) — confirm they FAIL
2. Implement tool_adapters.py (T046)
3. Update exports (T047) — can run in parallel with T046
4. Verify (T048–T049) — all tests pass, quality checks
```

### Task Dependencies (strict order)

```
T042, T043, T044, T045 ─────┐ (all test tasks can be written in parallel)
                              │
                              ▼ (confirm all tests fail)
                        T046 ◄─── tool_adapters.py (depends on: T042–T045 tests exist and fail)
                              │
                        T047 ◄─── __init__.py exports (can run in parallel with T046)
                              │
                        T048 ◄─── test suite (depends on: T046, T047)
                        T049 ◄─── quality checks (depends on: T048 passing)
```

### Parallel Opportunities

**Test writing** (T042–T045): All four test tasks target the same file (`test_tool_adapters.py`) but cover different test classes — they CAN be written in parallel by different developers, or sequentially by a single developer:
- T042: `TestVectorStoreToolAdapter` class
- T043: `TestHierarchicalDocToolAdapter` class
- T044: `TestBuildHolodeckSdkServer` class
- T045: `TestCreateToolAdapters` class

**Implementation** (T046–T047):
- T046 and T047 can run in parallel (different files) — but T047 imports from T046, so practically sequential

**Parallel with Phase 6 and 7**:
- Phase 5 (tool adapters), Phase 6 (MCP bridge), and Phase 7 (OTel bridge) can proceed in parallel after Phase 4's Integration Gate (T041) passes — they touch different files and have no interdependencies (plan.md:L578-581)

### Parallel Example

```bash
# Phase 5A — Write all tests in a single file (sequential for single dev):
# tests/unit/lib/backends/test_tool_adapters.py
# T042: TestVectorStoreToolAdapter
# T043: TestHierarchicalDocToolAdapter
# T044: TestBuildHolodeckSdkServer
# T045: TestCreateToolAdapters

# Confirm all tests fail:
pytest tests/unit/lib/backends/test_tool_adapters.py -n auto

# Phase 5B — Implementation:
# T046 → T047 (sequential — exports import from tool_adapters)

# Phase 5C — Sequential verification:
# T048 → T049
```

---

## Task Summary

| Group | Tasks | Tests | Implementations | Parallel |
|-------|-------|-------|-----------------|---------|
| Phase 5A (Tests) | T042–T045 | 4 | 0 | T042∥T043∥T044∥T045 (different classes) |
| Phase 5B (Implementation) | T046–T047 | 0 | 2 | T046∥T047 (different files) |
| Phase 5C (Verification) | T048–T049 | 0 | 0 | Sequential |
| **Total** | **8 tasks** | **4** | **2** | |

---

## Continuity from Phase 4

Phase 4 ended with T041 (Integration Gate). Phase 5 starts at T042.

**Prerequisites from earlier phases**:
- `src/holodeck/lib/backends/__init__.py` exists with base types exported (T007 + T038)
- `tests/unit/lib/backends/__init__.py` exists for pytest discovery (T007b)
- `src/holodeck/lib/backends/base.py` exists with `ExecutionResult`, `AgentBackend`, `BackendInitError` (T008)
- `src/holodeck/models/claude_config.py` exists with `ClaudeConfig` (T014, Phase 2)
- `src/holodeck/lib/backends/validators.py` exists with embedding_provider validation (T023, Phase 3)
- `src/holodeck/lib/backends/sk_backend.py` exists (T030, Phase 4)
- Phase 4 Integration Gate passed — all existing tests green (T041)

---

## Design Notes

### Why factory functions, not closures in loops

Python closures capture variables **by reference**, not by value. If adapters are created in a `for` loop using a lambda or closure, all closures share the same loop variable and will all call the **last** tool instance when invoked. The factory function pattern creates a new scope for each iteration, binding the correct tool instance:

```python
# BAD — all closures share `t` reference:
for t in tools:
    @tool(t.name, ...)
    async def search(args): return await t.search(args["query"])  # Always calls last `t`!

# GOOD — factory function creates new scope:
def _make_search_fn(t, name):
    @tool(name, ...)
    async def search(args): return await t.search(args["query"])  # Correctly bound
    return search
```

This is tested explicitly in T042 test case 6.

(plan.md:L333-343, research.md:L147-156)

### Why `SearchResult.format()` instead of custom serialization

`VectorStoreTool.search()` returns a `str` directly — it can be passed through to the SDK as-is. However, `HierarchicalDocumentTool.search()` returns `list[SearchResult]` — structured Python dataclass objects that must be serialized to text for the Claude subprocess to consume them.

Rather than custom serialization (`f"[Score: {r.fused_score:.3f}] {r.source_path}\n{r.content}\n"`), the adapter uses `SearchResult.format()` — the canonical per-result formatter defined in `holodeck.lib.hybrid_search`. This avoids format drift if the upstream `SearchResult` structure changes, and keeps serialization logic in one place.

The adapter also has a `get_context(query)` alternative available (returns pre-formatted `str` directly), but we chose `search()` + `format()` to maintain consistent access to individual results and their scores, which may be useful for debugging and logging.

(vectorstore_tool.py:L751-808, hierarchical_document_tool.py:L947-992, hybrid_search.py:L146)

### Why `create_tool_adapters` takes a `dict[str, ...]` for instances

Tool instances are created and initialized by the backend (either `ClaudeBackend` or `SKBackend`) before adapters are built. The dict is keyed by `config.name` for O(1) lookup. This pattern matches the existing `AgentFactory._register_vectorstore_tools()` pattern where tools are initialized by iterating configs and the instance is associated with the config's name.

(agent_factory.py:L842-903, agent_factory.py:L991-1054)

### Consumer contract for `create_tool_adapters()`

Phase 8's `ClaudeBackend.initialize()` is the primary consumer of `create_tool_adapters()`. It MUST provide:

1. **`tool_instances` dict keyed by `config.name`**: Each key must match the `name` field of a `VectorstoreTool` or `HierarchicalDocumentToolConfig` in the agent's tool list
2. **Initialized tools**: Each instance must have `is_initialized == True` (for VectorStoreTool) or `_initialized == True` (for HierarchicalDocumentTool) — embedding services injected and indexes built
3. **Embedding services injected**: `tool.set_embedding_service()` must have been called using `agent.embedding_provider` credentials (NOT `agent.model` — plan.md:L346-349)

If any of these contracts are violated, `create_tool_adapters()` raises `BackendInitError` (missing instance) or the adapter handler raises `BackendSessionError` at invocation time (uninitialized tool).

### Embedding provider usage in Phase 5

Phase 5 does NOT implement embedding initialization — that is Phase 8's `ClaudeBackend.initialize()` responsibility. Phase 5 assumes tool instances are **already initialized** with their embedding services injected. The adapters wrap initialized instances; they do not manage the initialization lifecycle.

For Claude-native agents, `ClaudeBackend.initialize()` will:
1. Call `validate_embedding_provider(agent)` (Phase 3, T023)
2. Create an SK embedding service using `agent.embedding_provider` credentials (NOT `agent.model` — plan.md:L346-349)
3. Initialize each tool instance with `tool.set_embedding_service()` and `tool.initialize()`
4. Pass initialized instances to `create_tool_adapters()`

(plan.md:L346-349, research.md:L244-250)

---

## What Follows Phase 5

Phase 5 delivers the tool adapter layer. Phase 8 (Claude Backend Core) consumes it:

```
Phase 4 (SK backend refactor) ← confirmed green
    │
    ├── Phase 5 (tool adapters — THIS FILE)   ─┐
    ├── Phase 6 (MCP bridge)                    ├── parallel after Phase 4 confirmed green
    └── Phase 7 (OTel bridge)                  ─┘
                                                  │
                                              Phase 8 (Claude backend — uses tool_adapters.py,
                                                       mcp_bridge.py, otel_bridge.py)
```

Phase 8's `ClaudeBackend.initialize()` calls:
1. `create_tool_adapters(agent.tools, initialized_instances)` — from Phase 5
2. `build_holodeck_sdk_server(adapters)` — from Phase 5
3. `build_claude_mcp_configs(mcp_tools)` — from Phase 6
4. `translate_observability(agent.observability)` — from Phase 7
5. `build_options(agent, tool_server, mcp_configs)` — Phase 8 internal

---

## Code Quality Gates

Run after Phase 5 implementation:

```bash
make format         # Black + Ruff formatting
make lint-fix       # Auto-fix linting issues
make type-check     # MyPy strict mode (all new code must pass)
make test-unit      # pytest tests/unit/ -n auto
make security       # Bandit + Safety + detect-secrets
```
