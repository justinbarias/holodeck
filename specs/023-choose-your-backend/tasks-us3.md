# Tasks: Seamless Tool Portability Across Backends (US3)

**Input**: Design documents from `/specs/023-choose-your-backend/`
**Prerequisites**: plan.md (required), spec.md (required), research.md, data-model.md
**Tests**: TDD approach — write tests FIRST, verify they FAIL, then implement.

**Organization**: Tasks are grouped by phase. US3 is primarily an integration testing and validation story that verifies the tool adapters built in US1 (ADK) and US2 (AF) produce semantically identical results across all four backends.

## Format: `[ID] [P?] [US3] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[US3]**: All tasks belong to User Story 3
- Include exact file paths in descriptions

## Path Conventions

- **Single project**: `src/holodeck/`, `tests/` at repository root

---

## Phase 1: Setup — Test Fixtures and Shared Test Infrastructure

**Purpose**: Create shared test fixtures and helpers that provide identical tool configurations across all backends, enabling direct portability comparison.

- [ ] T301 [P] [US3] Create shared tool fixture module `tests/fixtures/tool_portability_fixtures.py` with factory functions that return canonical HoloDeck tool configurations for each tool type: `make_vectorstore_config()` returning a `VectorstoreTool` config dict, `make_function_tool_config()` returning a function tool config dict, `make_mcp_stdio_config()` returning an MCP stdio tool config dict, `make_mcp_sse_config()` returning an MCP SSE/HTTP tool config dict, and `make_hierarchical_doc_config()` returning a `HierarchicalDocumentToolConfig` dict. Each factory must return the same canonical config regardless of which backend test imports it.
- [ ] T302 [P] [US3] Create shared mock helpers in `tests/fixtures/tool_portability_fixtures.py`: `make_mock_vectorstore_instance()` returning a mock `VectorStoreTool` with a `.search()` async method that returns deterministic results (3 fixed documents with scores), and `make_mock_hierarchical_doc_instance()` returning a mock `HierarchicalDocumentTool` with identical `.search()` behavior. These mocks simulate the initialized tool instances that adapters wrap.
- [ ] T303 [P] [US3] Create shared mock helpers in `tests/fixtures/tool_portability_fixtures.py`: `make_mock_embedding_service()` returning a mock `EmbeddingService` that returns deterministic embedding vectors (fixed 3-dimensional vectors for reproducibility). This ensures embedding behavior is identical across backend tests.
- [ ] T304 [US3] Register the fixture module in `tests/conftest.py` by adding `pytest_plugins` entry or importing fixtures so they are available to both unit and integration test files. Verify fixtures are discoverable by running `pytest --collect-only tests/fixtures/tool_portability_fixtures.py`.

**Checkpoint**: Shared fixtures ready — portability tests can reference canonical tool configurations.

---

## Phase 2: Foundational — Tool Adapter Interface Consistency

**Purpose**: Ensure all four backend tool adapters expose a consistent interface pattern so portability tests can compare outputs uniformly. Refactor if needed.

- [ ] T305 [P] [US3] Add a `normalize_tool_output(raw_output: Any) -> dict[str, Any]` helper function to `tests/fixtures/tool_portability_fixtures.py` that takes the raw output from any backend's tool invocation and normalizes it to a common schema: `{"results": list[dict], "query": str, "tool_name": str}`. This enables direct comparison of tool outputs across backends regardless of native format differences.
- [ ] T306 [P] [US3] Review and document (as code comments in `tests/fixtures/tool_portability_fixtures.py`) the tool adapter entry points for each backend: (1) `src/holodeck/lib/backends/tool_adapters.py` — Claude: `adapt_tools()` returns list of `SdkMcpTool`, (2) `src/holodeck/lib/backends/adk_tool_adapters.py` — ADK: `adapt_tools_for_adk()` returns list of ADK-native tools, (3) `src/holodeck/lib/backends/af_tool_adapters.py` — AF: `adapt_tools_for_af()` returns list of AF `FunctionTool`/MCP tools, (4) SK backend uses `tool_initializer.py` directly. Document the expected input/output contract for each.
- [ ] T307 [US3] Verify that all four tool adapter modules (`tool_adapters.py`, `adk_tool_adapters.py`, `af_tool_adapters.py`, and the SK path in `sk_backend.py`) accept the same `list[ToolUnion]` input type along with initialized tool instances. If any adapter deviates, add an adapter shim or refactor the entry point to accept `(tool_configs: list[ToolUnion], tool_instances: dict[str, Any])` consistently. File paths: `src/holodeck/lib/backends/tool_adapters.py`, `src/holodeck/lib/backends/adk_tool_adapters.py`, `src/holodeck/lib/backends/af_tool_adapters.py`, `src/holodeck/lib/backends/sk_backend.py`.

**Checkpoint**: All adapters have consistent interfaces — portability comparison tests can proceed.

---

## Phase 3: User Story 3 — Cross-Backend Validation Tests

**Purpose**: Validate that switching `backend` preserves tool behavior. Tests are organized by tool type, then by backend pair.

### 3a. VectorStore Tool Portability

- [ ] T308 [P] [US3] Write unit test `test_vectorstore_adk_adapter_output_matches_claude` in `tests/unit/lib/backends/test_adk_tool_adapters.py` — adapt the canonical vectorstore config from T301 using `adapt_tools_for_adk()`, invoke the adapted tool's search with query `"test query"`, verify the response contains the same 3 documents with identical content and scores as the mock from T302. Compare against Claude adapter output (adapt same config via `tool_adapters.py`, invoke, normalize both outputs via T305).
- [ ] T309 [P] [US3] Write unit test `test_vectorstore_af_adapter_output_matches_claude` in `tests/unit/lib/backends/test_af_tool_adapters.py` — same pattern as T308 but adapt via `adapt_tools_for_af()` and compare AF output against Claude adapter output. Verify `FunctionTool` wrapping preserves search semantics.
- [ ] T310 [P] [US3] Write unit test `test_vectorstore_adk_adapter_output_matches_af` in `tests/unit/lib/backends/test_adk_tool_adapters.py` — adapt canonical vectorstore config via both ADK and AF adapters, invoke both with same query, normalize outputs, assert results are semantically identical (same documents, same scores, same ordering).
- [ ] T311 [P] [US3] Write unit test `test_hierarchical_doc_adk_adapter_output_matches_claude` in `tests/unit/lib/backends/test_adk_tool_adapters.py` — same pattern as T308 but using `make_hierarchical_doc_config()` and `make_mock_hierarchical_doc_instance()` from T301/T302.
- [ ] T312 [P] [US3] Write unit test `test_hierarchical_doc_af_adapter_output_matches_claude` in `tests/unit/lib/backends/test_af_tool_adapters.py` — same pattern as T309 but for hierarchical document tools.

### 3b. Function Tool Portability

- [ ] T313 [P] [US3] Write unit test `test_function_tool_adk_adapter_callable` in `tests/unit/lib/backends/test_adk_tool_adapters.py` — adapt the canonical function tool config from T301 using `adapt_tools_for_adk()`, verify the adapted tool is a plain callable (ADK auto-wraps), invoke it with test arguments, assert return value matches expected output. The function under test should be a simple deterministic function (e.g., `def add(a: int, b: int) -> int: return a + b`).
- [ ] T314 [P] [US3] Write unit test `test_function_tool_af_adapter_callable` in `tests/unit/lib/backends/test_af_tool_adapters.py` — adapt the same canonical function tool config using `adapt_tools_for_af()`, verify result is an AF `FunctionTool`, invoke it with the same test arguments, assert return value is identical to T313.
- [ ] T315 [P] [US3] Write unit test `test_function_tool_cross_backend_identical_output` in `tests/unit/lib/backends/test_adk_tool_adapters.py` — adapt the canonical function tool via ADK, AF, and Claude adapters, invoke all three with identical arguments, assert all three return the same result. This validates FR-006 (tool adapters translate definitions to native format) for function tools.

### 3c. MCP Tool Portability

- [ ] T316 [P] [US3] Write unit test `test_mcp_stdio_adk_adapter_creates_toolset` in `tests/unit/lib/backends/test_adk_tool_adapters.py` — adapt the canonical MCP stdio config from T301 using `adapt_tools_for_adk()`, verify the result is an `McpToolset` instance with `StdioConnectionParams` containing the correct `command` and `args`. Mock the MCP process to avoid spawning real subprocesses.
- [ ] T317 [P] [US3] Write unit test `test_mcp_stdio_af_adapter_creates_mcp_tool` in `tests/unit/lib/backends/test_af_tool_adapters.py` — adapt the same canonical MCP stdio config using `adapt_tools_for_af()`, verify the result is an `MCPStdioTool` instance with matching `command` and `args`. Mock the MCP process.
- [ ] T318 [P] [US3] Write unit test `test_mcp_sse_adk_adapter_creates_toolset` in `tests/unit/lib/backends/test_adk_tool_adapters.py` — adapt the canonical MCP SSE/HTTP config from T301, verify `McpToolset` with `SseConnectionParams` or `StreamableHTTPConnectionParams` containing the correct URL.
- [ ] T319 [P] [US3] Write unit test `test_mcp_sse_af_adapter_creates_http_tool` in `tests/unit/lib/backends/test_af_tool_adapters.py` — adapt the same MCP SSE/HTTP config, verify `MCPStreamableHTTPTool` with the correct URL.

### 3d. End-to-End Portability (Integration)

- [ ] T320 [US3] Create integration test file `tests/integration/test_tool_portability.py` with module docstring explaining this file validates SC-001 (switching backends preserves tool behavior). Add `pytest.importorskip()` guards for `google.adk` and `agent_framework` packages. Import shared fixtures from T301-T303.
- [ ] T321 [P] [US3] Write integration test `test_vectorstore_portability_across_all_backends` in `tests/integration/test_tool_portability.py` — for each backend (SK, Claude, ADK, AF): create an agent config with `backend` set explicitly and a vectorstore tool, initialize the backend with mocked LLM responses, invoke `invoke_once()` with a query that triggers the vectorstore tool, capture the `ExecutionResult.tool_results`, normalize outputs, assert all four backends return semantically identical search results. Skip backends whose packages are not installed.
- [ ] T322 [P] [US3] Write integration test `test_function_tool_portability_across_all_backends` in `tests/integration/test_tool_portability.py` — same pattern as T321 but with a function tool. For each backend, invoke with the same query, assert the function is called with identical arguments and returns the same result in `ExecutionResult.tool_results`.
- [ ] T323 [P] [US3] Write integration test `test_mcp_stdio_portability_across_backends` in `tests/integration/test_tool_portability.py` — same pattern as T321 but with MCP stdio tools. Mock the MCP subprocess to return deterministic tool results. Verify all backends (that support MCP) produce identical `ExecutionResult.tool_results`. Note: SK does not support MCP tools natively, so skip SK for this test.
- [ ] T324 [P] [US3] Write integration test `test_backend_switch_preserves_tool_config` in `tests/integration/test_tool_portability.py` — load a single agent.yaml fixture with multiple tool types (vectorstore + function + MCP), run it with `backend: google_adk`, then reload the same config with `backend: agent_framework`, then with `backend: claude`. For each run, assert that all tools are initialized without errors and the agent responds without tool-related failures.

### 3e. Edge Cases and Error Handling

- [ ] T325 [P] [US3] Write unit test `test_unsupported_tool_type_raises_error` in `tests/unit/lib/backends/test_adk_tool_adapters.py` — pass a tool config with an unsupported type to the ADK adapter, verify it raises a clear error message listing supported tool types. Repeat for AF adapter in `tests/unit/lib/backends/test_af_tool_adapters.py`.
- [ ] T326 [P] [US3] Write unit test `test_empty_tool_list_adapts_cleanly` in `tests/unit/lib/backends/test_adk_tool_adapters.py` — pass an empty tool list to ADK and AF adapters, verify both return empty lists without errors. This covers agents configured with no tools switching backends.
- [ ] T327 [P] [US3] Write unit test `test_tool_adapter_preserves_tool_name_and_description` in `tests/unit/lib/backends/test_adk_tool_adapters.py` — for each tool type (vectorstore, function, MCP), adapt via ADK adapter, extract the tool name and description from the adapted object, assert they match the original HoloDeck config. Repeat for AF adapter in `tests/unit/lib/backends/test_af_tool_adapters.py`.
- [ ] T328 [P] [US3] Write unit test `test_vectorstore_tool_requires_embedding_service` in `tests/unit/lib/backends/test_adk_tool_adapters.py` — attempt to adapt a vectorstore tool without providing an `EmbeddingService` instance, verify the adapter raises `ToolInitializerError` or equivalent with a message mentioning `embedding_provider`. Repeat for AF adapter.
- [ ] T329 [US3] Run all US3 unit tests: `pytest tests/unit/lib/backends/test_adk_tool_adapters.py tests/unit/lib/backends/test_af_tool_adapters.py -n auto -v -k "portability or cross_backend or matches"` and verify they PASS.
- [ ] T330 [US3] Run all US3 integration tests: `pytest tests/integration/test_tool_portability.py -n auto -v` and verify they PASS (skipping backends whose packages are not installed).

**Checkpoint**: All cross-backend tool portability tests pass — US3 acceptance scenarios validated.

### 3f. Skill Tool Portability

**Blocked by**: US7 (SkillTool model + backend adapters T730-T738)

- [ ] T338 [P] [US3] Add `make_skill_config()` fixture (inline skill with `instructions` + `allowed_tools`) in `tests/fixtures/tool_portability_fixtures.py`. The factory should return a canonical `SkillTool` config dict with a deterministic instruction string and a fixed list of allowed tools, consistent with the pattern established in T301.
- [ ] T339 [P] [US3] Write unit test `test_skill_tool_adk_adapter_output_matches_claude` in `tests/unit/lib/backends/test_adk_tool_adapters.py` — adapt the canonical skill config from T338 using `adapt_tools_for_adk()`, compare the adapted tool's structure and invocation output against Claude adapter output (adapt same config via `tool_adapters.py`). Normalize both outputs via T305 and assert semantic equivalence.
- [ ] T340 [P] [US3] Write unit test `test_skill_tool_af_adapter_output_matches_claude` in `tests/unit/lib/backends/test_af_tool_adapters.py` — same pattern as T339 but adapt via `adapt_tools_for_af()` and compare AF output against Claude adapter output. Verify `FunctionTool` wrapping preserves skill invocation semantics.
- [ ] T341 [US3] Write unit test `test_skill_tool_adk_adapter_output_matches_af` in `tests/unit/lib/backends/test_cross_backend_skill.py` — adapt canonical skill config via both ADK and AF adapters, invoke both with the same input, normalize outputs, assert results are semantically identical.
- [ ] T342 [US3] Write integration test `test_skill_tool_portability_across_backends` in `tests/integration/test_tool_portability.py` — configure an agent with a skill tool, run it with `backend: google_adk`, then `backend: agent_framework`, then `backend: claude`. For each backend, verify skill invocation works without errors and produces semantically consistent results. Skip backends whose packages are not installed.

**Checkpoint**: Skill tool portability validated across all three backends.

---

## Phase 4: Polish — Documentation and Final Validation

**Purpose**: Document tool portability guarantees and ensure code quality.

- [ ] T331 [P] [US3] Add a docstring block to `tests/integration/test_tool_portability.py` documenting the portability guarantee: "HoloDeck guarantees that switching the `backend` field while keeping tool configurations unchanged produces semantically identical tool behavior. This file validates that guarantee per SC-001 and FR-006."
- [ ] T332 [P] [US3] Add inline comments to `src/holodeck/lib/backends/adk_tool_adapters.py` at each tool type handler documenting the portability contract: input format (HoloDeck ToolUnion), output format (ADK-native), and semantic equivalence expectation.
- [ ] T333 [P] [US3] Add inline comments to `src/holodeck/lib/backends/af_tool_adapters.py` at each tool type handler documenting the same portability contract.
- [ ] T334 [US3] Run `make format` to format all new and modified files with Black + Ruff.
- [ ] T335 [US3] Run `make lint` and fix any Ruff + Bandit violations in US3 files.
- [ ] T336 [US3] Run `make type-check` and fix any MyPy errors in US3 files.
- [ ] T337 [US3] Run full test suite `make test` to verify no regressions from US3 changes.

**Checkpoint**: US3 complete — tool portability is validated, documented, and code quality verified.

---

## Dependencies & Execution Order

### External Dependencies (BLOCKING)

- **US1 (ADK Backend)**: Must be complete before US3. US3 validates `adk_tool_adapters.py` built in US1.
- **US2 (AF Backend)**: Must be complete before US3. US3 validates `af_tool_adapters.py` built in US2.
- **US6 (EmbeddingService)**: Must be complete before US3. Vectorstore/hierarchical_document tool portability tests depend on `EmbeddingService` protocol and `LiteLLMEmbeddingAdapter`.
- **US7 (SkillTool)**: Must be complete before Phase 3f only. Skill tool portability tests depend on `SkillTool` model and backend adapters (T730-T738). Other US3 phases can proceed without US7.

### Phase Dependencies

- **Phase 1 (Setup)**: No internal dependencies — can start immediately after US1 + US2 + US6 are complete
- **Phase 2 (Foundational)**: Depends on Phase 1 completion (needs fixtures)
- **Phase 3 (Validation Tests)**: Depends on Phase 2 completion (needs consistent adapter interfaces)
- **Phase 4 (Polish)**: Depends on Phase 3 completion (all tests must pass first)

### Within Phase 3

- **3a (VectorStore)**, **3b (Function)**, **3c (MCP)** can run in parallel — they test independent tool types
- **3d (Integration)** depends on 3a, 3b, 3c being complete — integration tests exercise all tool types together
- **3e (Edge Cases)** can run in parallel with 3a-3c
- **3f (Skill Tool Portability)** depends on US7 completion — can run in parallel with 3a-3e once US7 is done

---

## Parallel Example: Phase 3a + 3b + 3c

```bash
# All tool-type-specific portability tests can run in parallel:
Task: "Write test_vectorstore_adk_adapter_output_matches_claude in tests/unit/lib/backends/test_adk_tool_adapters.py"
Task: "Write test_vectorstore_af_adapter_output_matches_claude in tests/unit/lib/backends/test_af_tool_adapters.py"
Task: "Write test_function_tool_adk_adapter_callable in tests/unit/lib/backends/test_adk_tool_adapters.py"
Task: "Write test_function_tool_af_adapter_callable in tests/unit/lib/backends/test_af_tool_adapters.py"
Task: "Write test_mcp_stdio_adk_adapter_creates_toolset in tests/unit/lib/backends/test_adk_tool_adapters.py"
Task: "Write test_mcp_stdio_af_adapter_creates_mcp_tool in tests/unit/lib/backends/test_af_tool_adapters.py"
```

---

## Implementation Strategy

### Test-Driven Validation (US3 is primarily a testing story)

Unlike US1 and US2 which build new adapters, US3 creates the **cross-backend validation layer** that proves portability works. The strategy is:

1. **Phase 1**: Build shared fixtures that represent the "ground truth" tool configurations and expected outputs
2. **Phase 2**: Verify all four backends' tool adapters have consistent entry points (refactor if needed)
3. **Phase 3**: Write exhaustive cross-backend comparison tests:
   - Unit tests compare adapter outputs pairwise (ADK vs Claude, AF vs Claude, ADK vs AF)
   - Integration tests run the same tool config through all backends end-to-end
   - Edge case tests validate error paths and boundary conditions
4. **Phase 4**: Document the portability guarantee and run full quality checks

### Key Files Modified

| File | Phase | Changes |
|------|-------|---------|
| `tests/fixtures/tool_portability_fixtures.py` | Phase 1 | NEW: Shared fixtures, mocks, normalization helpers |
| `tests/conftest.py` | Phase 1 | MODIFY: Register fixture module |
| `tests/unit/lib/backends/test_adk_tool_adapters.py` | Phase 3 | MODIFY: Add portability-focused comparison tests |
| `tests/unit/lib/backends/test_af_tool_adapters.py` | Phase 3 | MODIFY: Add portability-focused comparison tests |
| `tests/integration/test_tool_portability.py` | Phase 3 | NEW: End-to-end cross-backend portability tests (incl. skill tool T342) |
| `tests/unit/lib/backends/test_cross_backend_skill.py` | Phase 3f | NEW: ADK vs AF skill tool portability comparison |
| `src/holodeck/lib/backends/adk_tool_adapters.py` | Phase 2, 4 | MODIFY: Interface consistency + portability docs |
| `src/holodeck/lib/backends/af_tool_adapters.py` | Phase 2, 4 | MODIFY: Interface consistency + portability docs |

### Acceptance Scenario Traceability

| Acceptance Scenario | Validating Tasks |
|---------------------|-----------------|
| AS-1: Vectorstore tools adapted to ADK format, identical results | T308, T310, T321 |
| AS-2: MCP tools adapted to AF format, function correctly | T317, T319, T323 |
| AS-3: Switch AF to SK, tool behavior identical | T321, T322, T324 |
| AS-4: Function tools loadable and callable across backends | T313, T314, T315, T322 |
| AS-4 (extended): Skill tools portable across any two supported backends | T338, T339, T340, T341, T342 |

---

## Notes

- [P] tasks = different files or independent test functions, no dependencies
- [US3] label maps all tasks to User Story 3 for traceability
- US3 is a validation/testing story — minimal production code changes expected
- Mock LLM responses in integration tests to avoid real API calls
- Use `pytest.importorskip()` for backends that may not be installed
- Commit after each phase or logical group of tasks
- Stop at any checkpoint to validate progress independently
- Tool portability is validated by comparing normalized outputs, not internal representations
