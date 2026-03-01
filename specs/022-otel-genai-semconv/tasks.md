# Tasks: Integrate OTel GenAI Instrumentation into Claude Backend

**Input**: Design documents from `/specs/022-otel-genai-semconv/`
**Prerequisites**: plan.md (required), spec.md (required), research.md, data-model.md, quickstart.md
**Tests**: TDD approach — write tests FIRST, verify they FAIL, then implement.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3, US4)
- Include exact file paths in descriptions

## Path Conventions

- **Single project**: `src/holodeck/`, `tests/` at repository root

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Add optional dependency and create the observability accessor needed by all user stories.

- [ ] T001 Add `otel-instrumentation-claude-agent-sdk>=0.0.3,<0.1.0` to `[claude-otel]` optional extras group in `pyproject.toml` and run `uv lock` to regenerate the lock file
- [ ] T002 Add `get_observability_context() -> ObservabilityContext | None` accessor function to `src/holodeck/lib/observability/providers.py` that returns the module-level `_observability_context` variable (with thread-safety docstring per data-model.md). Also add `"get_observability_context"` to `providers.py`'s own `__all__` list (L395)
- [ ] T003 Export `get_observability_context` from `src/holodeck/lib/observability/__init__.py` by adding it to `__all__` and the import list
- [ ] T004 Add `_instrumentor` instance attribute (initialized to `None`) in `ClaudeBackend.__init__()` in `src/holodeck/lib/backends/claude_backend.py`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Create integration test file. MUST complete before user story implementation.

**CRITICAL**: No user story work can begin until this phase is complete.

- [ ] T005 Create integration test file `tests/integration/test_claude_instrumentation.py` with imports, fixtures, and placeholder test class (to be filled in US4). Directory `tests/integration/` already exists — no new subdirectories needed

**Checkpoint**: Foundation ready — user story implementation can now begin.

---

## Phase 3: User Story 1 — Activate Instrumentation During Backend Initialization (Priority: P1)

**Goal**: When observability is enabled and the instrumentation package is installed, `ClaudeBackend.initialize()` activates the external instrumentation package with the correct `TracerProvider`, `MeterProvider`, `agent_name`, and `capture_content` parameters.

**Independent Test**: Initialize a `ClaudeBackend` with observability enabled and mock the instrumentation package. Verify `instrument()` is called with correct parameters.

### Tests for User Story 1 (TDD — write FIRST, verify they FAIL)

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation.**
> All tests calling `initialize()` MUST mock the validators (`validate_nodejs`, `validate_credentials`, `validate_embedding_provider`, `validate_tool_filtering`, `validate_working_directory`, `validate_response_format`) and `_initialize_tools()` to isolate instrumentation logic. Follow existing patterns in `test_claude_backend.py`.

- [ ] T006 [P] [US1] Write unit test `test_instrument_called_with_tracer_and_meter_providers` in `tests/unit/lib/backends/test_claude_backend.py` — mock `ClaudeAgentSdkInstrumentor` import and `get_observability_context()`, create agent with `observability.enabled=True`, `traces.enabled=True`, `metrics.enabled=True`, call `initialize()`, assert `instrument()` called with `tracer_provider`, `meter_provider`, `agent_name="test-agent"`, `capture_content` from config. Assert the exact `TracerProvider` and `MeterProvider` instances passed match those returned by the mock `get_observability_context()` (not the global OTel providers)
- [ ] T007 [P] [US1] Write unit test `test_instrument_called_without_meter_provider_when_metrics_disabled` in `tests/unit/lib/backends/test_claude_backend.py` — same setup but `metrics.enabled=False`, assert `instrument()` called with `meter_provider=None`
- [ ] T008 [P] [US1] Write unit test `test_instrument_not_called_when_observability_disabled` in `tests/unit/lib/backends/test_claude_backend.py` — create agent with `observability.enabled=False` or `observability=None`, call `initialize()`, assert `instrument()` NOT called
- [ ] T009 [P] [US1] Write unit test `test_instrument_not_called_when_traces_disabled` in `tests/unit/lib/backends/test_claude_backend.py` — create agent with `observability.enabled=True` but `traces.enabled=False`, call `initialize()`, assert `instrument()` NOT called

### Implementation for User Story 1

- [ ] T010 [US1] Implement `_activate_instrumentation(self) -> None` private method in `ClaudeBackend` in `src/holodeck/lib/backends/claude_backend.py` — check `agent.observability` is enabled and `traces.enabled`, try-import `ClaudeAgentSdkInstrumentor`, get `ObservabilityContext` via `get_observability_context()`, call `instrument()` with `tracer_provider`, `meter_provider` (only if `metrics.enabled`), `agent_name`, `capture_content`. **Important**: set `self._instrumentor` only AFTER `instrument()` returns successfully. Wrap entire body in try/except to catch all errors and log warning
- [ ] T011 [US1] Add call to `self._activate_instrumentation()` in `ClaudeBackend.initialize()` in `src/holodeck/lib/backends/claude_backend.py` — place after `validate_response_format()` call and before `self._initialized = True`
- [ ] T012 [US1] Run tests T006–T009 and verify they PASS after implementation

**Checkpoint**: User Story 1 complete — instrumentation activates correctly when observability is enabled.

---

## Phase 4: User Story 2 — Graceful Degradation When Package Not Installed (Priority: P1)

**Goal**: When the `otel-instrumentation-claude-agent-sdk` package is not installed, HoloDeck logs a warning and continues without crashing.

**Independent Test**: Mock the import of the instrumentation package to raise `ImportError`, run `initialize()`, verify no exception raised and warning logged.

### Tests for User Story 2 (TDD — write FIRST, verify they FAIL)

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation.**
> All tests calling `initialize()` MUST mock the validators and `_initialize_tools()` as noted in US1.

- [ ] T013 [P] [US2] Write unit test `test_graceful_degradation_when_package_not_installed` in `tests/unit/lib/backends/test_claude_backend.py` — patch the import of `opentelemetry.instrumentation.claude_agent_sdk` to raise `ImportError`, create agent with observability enabled, call `initialize()`, assert no exception raised, assert `self._instrumentor is None`, assert warning logged (use `caplog` fixture)
- [ ] T014 [P] [US2] Write unit test `test_graceful_degradation_when_instrument_raises` in `tests/unit/lib/backends/test_claude_backend.py` — mock `ClaudeAgentSdkInstrumentor` but make `instrument()` raise `RuntimeError("version mismatch")`, call `initialize()`, assert no exception raised, assert `self._instrumentor is None` (instrumentor is never stored because `instrument()` failed before assignment), assert warning logged
- [ ] T015 [P] [US2] Write unit test `test_backend_functional_without_instrumentation` in `tests/unit/lib/backends/test_claude_backend.py` — patch import to raise `ImportError`, initialize backend, call `invoke_once()` (with mocked `query()`), assert `ExecutionResult` returned successfully

### Implementation for User Story 2

- [ ] T016 [US2] Run tests T013–T015 and verify they PASS with the `_activate_instrumentation()` implementation from US1 (ImportError handling and general Exception catching should already be in place from T010)

**Checkpoint**: User Story 2 complete — backend remains functional when package is missing or broken.

---

## Phase 5: User Story 3 — Clean Deactivation on Teardown (Priority: P2)

**Goal**: When `teardown()` is called, `uninstrument()` is called to cleanly remove instrumentation hooks and prevent state leakage.

**Independent Test**: Initialize backend with instrumentation active, call `teardown()`, verify `uninstrument()` was called and `self._instrumentor` is `None`.

### Tests for User Story 3 (TDD — write FIRST, verify they FAIL)

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation.**
> All tests calling `initialize()` MUST mock the validators and `_initialize_tools()` as noted in US1.

- [ ] T017 [P] [US3] Write unit test `test_teardown_calls_uninstrument` in `tests/unit/lib/backends/test_claude_backend.py` — initialize backend with mocked instrumentation active (`self._instrumentor` set to a mock), call `teardown()`, assert `uninstrument()` called on the mock instrumentor, assert `self._instrumentor is None`
- [ ] T018 [P] [US3] Write unit test `test_teardown_safe_without_instrumentation` in `tests/unit/lib/backends/test_claude_backend.py` — initialize backend without instrumentation (`self._instrumentor is None`), call `teardown()`, assert no exception raised
- [ ] T019 [P] [US3] Write unit test `test_sequential_init_teardown_no_leakage` in `tests/unit/lib/backends/test_claude_backend.py` — initialize backend with instrumentation, teardown, initialize again, teardown again; verify `instrument()` called twice and `uninstrument()` called twice (no leakage between runs)
- [ ] T020 [P] [US3] Write unit test `test_teardown_handles_uninstrument_exception` in `tests/unit/lib/backends/test_claude_backend.py` — mock `uninstrument()` to raise `RuntimeError`, call `teardown()`, assert no exception raised, assert warning logged via `caplog`, assert `self._instrumentor is None` after teardown (FR-008: deactivation MUST NOT raise)

### Implementation for User Story 3

- [ ] T021 [US3] Add uninstrument logic to `ClaudeBackend.teardown()` in `src/holodeck/lib/backends/claude_backend.py` — if `self._instrumentor is not None`, call `self._instrumentor.uninstrument()` wrapped in try/except (log warning on error), set `self._instrumentor = None`; place before existing cleanup code
- [ ] T022 [US3] Run tests T017–T020 and verify they PASS after implementation

**Checkpoint**: User Story 3 complete — clean lifecycle management with no state leakage.

---

## Phase 6: User Story 4 — Span Hierarchy Alignment (Priority: P2)

**Goal**: GenAI `invoke_agent` spans produced by the instrumentation package nest correctly under HoloDeck's existing parent spans via OTel context propagation.

**Independent Test**: Use `InMemorySpanExporter` with a real `TracerProvider` and a real `ClaudeAgentSdkInstrumentor` to verify that spans created by the instrumentation package are children of the HoloDeck parent span.

### Integration Test for User Story 4

- [ ] T023 [US4] Write integration test `test_span_hierarchy_parent_child` in `tests/integration/test_claude_instrumentation.py` — use `pytest.importorskip("opentelemetry.instrumentation.claude_agent_sdk")` to skip when package not installed. Set up a real `InMemorySpanExporter` and `TracerProvider`, create a real `ClaudeAgentSdkInstrumentor`, call `instrument()` with the test `TracerProvider`. Create a parent span (simulating `holodeck.cli.test`), then within that span context, invoke `ClaudeBackend.initialize()` and `invoke_once()` (mock SDK `query()` to return a minimal response, mock all validators). Verify that exported spans include the `invoke_agent` span as a child of the parent span. Call `uninstrument()` in test teardown. This exercises the real instrumentation package with real OTel providers

**Checkpoint**: All user stories are independently functional.

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Code quality, documentation, and final validation.

- [ ] T024 Run `make format` to format all modified files with Black + Ruff
- [ ] T025 Run `make lint` and fix any Ruff + Bandit violations
- [ ] T026 Run `make type-check` and fix any MyPy errors (ensure `_instrumentor` type annotation is correct)
- [ ] T027 Run full test suite `make test` to verify no regressions
- [ ] T028 Run quickstart.md validation — verify install and usage instructions match implementation

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion — BLOCKS all user stories
- **User Stories (Phase 3–6)**: All depend on Foundational phase completion
  - US1 and US2 can proceed in parallel (US2 tests the error paths of US1's implementation)
  - US3 depends on US1 completion (needs `_instrumentor` to be set)
  - US4 depends on US1 completion (needs `_activate_instrumentation()` to exist)
- **Polish (Phase 7)**: Depends on all user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2) — No dependencies on other stories
- **User Story 2 (P1)**: Can start after Foundational (Phase 2) — Tests error paths of US1 implementation, but US2 tests can be written in parallel; US2 verification (T016) confirms US1's error handling is correct
- **User Story 3 (P2)**: Can start after US1 (needs `self._instrumentor` attribute and `_activate_instrumentation()`)
- **User Story 4 (P2)**: Can start after US1 (needs `_activate_instrumentation()` to exist for integration test)

### Within Each User Story

- Tests MUST be written and FAIL before implementation
- Implementation after tests
- Verify tests PASS after implementation
- Story complete before moving to next priority

### Parallel Opportunities

- T001, T002, T003 can run in parallel (different files)
- T006, T007, T008, T009 can run in parallel (same file, independent test functions)
- T013, T014, T015 can run in parallel (same file, independent test functions)
- T017, T018, T019, T020 can run in parallel (same file, independent test functions)
- T024, T025, T026 can run in parallel (independent quality checks)

---

## Parallel Example: User Story 1

```bash
# Launch all tests for User Story 1 together:
Task: "Write test_instrument_called_with_tracer_and_meter_providers in tests/unit/lib/backends/test_claude_backend.py"
Task: "Write test_instrument_called_without_meter_provider_when_metrics_disabled in tests/unit/lib/backends/test_claude_backend.py"
Task: "Write test_instrument_not_called_when_observability_disabled in tests/unit/lib/backends/test_claude_backend.py"
Task: "Write test_instrument_not_called_when_traces_disabled in tests/unit/lib/backends/test_claude_backend.py"
```

---

## Implementation Strategy

### MVP First (User Stories 1 + 2 Only)

1. Complete Phase 1: Setup (T001–T004)
2. Complete Phase 2: Foundational (T005)
3. Complete Phase 3: User Story 1 (T006–T012) — instrumentation activates
4. Complete Phase 4: User Story 2 (T013–T016) — graceful degradation
5. **STOP and VALIDATE**: Run `make test` — core integration works
6. Deploy/demo if ready

### Incremental Delivery

1. Complete Setup + Foundational → Foundation ready
2. Add User Story 1 → Test independently → Instrumentation works (MVP!)
3. Add User Story 2 → Test independently → Graceful degradation works
4. Add User Story 3 → Test independently → Clean lifecycle
5. Add User Story 4 → Test independently → Span hierarchy verified
6. Polish → Code quality validated

### Key Files Modified

| File | Phase | Changes |
|------|-------|---------|
| `pyproject.toml` | Setup | `[claude-otel]` optional group |
| `src/holodeck/lib/observability/providers.py` | Setup | `get_observability_context()` accessor + `__all__` |
| `src/holodeck/lib/observability/__init__.py` | Setup | Export new accessor |
| `src/holodeck/lib/backends/claude_backend.py` | US1, US3 | `_instrumentor` attr, `_activate_instrumentation()`, teardown logic |
| `tests/unit/lib/backends/test_claude_backend.py` | US1–US3 | Instrumentation lifecycle unit tests |
| `tests/integration/test_claude_instrumentation.py` | US4 | Span hierarchy integration test (real instrumentation package) |

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story is independently completable and testable
- TDD: Verify tests fail before implementing
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- All GenAI semconv logic owned by external package — HoloDeck only calls `instrument()`/`uninstrument()`
- Integration test (T023) requires `otel-instrumentation-claude-agent-sdk` to be installed; uses `pytest.importorskip()` to skip gracefully when not available
