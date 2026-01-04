# Tasks: Agent Local Server

**Input**: Design documents from `/specs/017-agent-local-server/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/openapi.yaml, contracts/sse-events.md

**Tests**: TDD approach - write tests FIRST, ensure they FAIL, then implement.

**Organization**: Tasks grouped by user story to enable independent implementation and testing.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and serve module structure

- [x] T001 Add FastAPI, Uvicorn, ag-ui-protocol, python-ulid dependencies to pyproject.toml
- [x] T002 Create serve module structure: src/holodeck/serve/__init__.py
- [x] T003 [P] Create protocols submodule: src/holodeck/serve/protocols/__init__.py
- [x] T004 [P] Create test directory structure: tests/unit/serve/, tests/integration/serve/, tests/contract/serve/

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**CRITICAL**: No user story work can begin until this phase is complete

### Tests for Foundational (TDD)

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [x] T005 [P] Unit tests for Pydantic models (ChatRequest, ChatResponse, FileContent validation) in tests/unit/serve/test_models.py
- [x] T006 [P] Unit tests for SessionStore (create, get, delete, touch, cleanup_expired) in tests/unit/serve/test_session_store.py
- [x] T007 [P] Unit tests for ServerSession dataclass in tests/unit/serve/test_session_store.py

### Implementation for Foundational

- [x] T008 Implement Pydantic models (ProtocolType, FileContent, ChatRequest, ChatResponse, ToolCallInfo, TokenUsage, HealthResponse, ProblemDetail) in src/holodeck/serve/models.py
- [x] T009 [P] Implement ServerSession dataclass with session_id, agent_executor, timestamps, message_count in src/holodeck/serve/session_store.py
- [x] T010 Implement SessionStore class with get, create, delete, touch, cleanup_expired methods in src/holodeck/serve/session_store.py
- [x] T011 [P] Implement Protocol ABC with abstract methods for request handling in src/holodeck/serve/protocols/base.py
- [x] T012 [P] Implement logging middleware for request metadata capture in src/holodeck/serve/middleware.py
- [x] T013 [P] Implement error handling middleware with RFC 7807 ProblemDetail responses in src/holodeck/serve/middleware.py
- [x] T014 Create AgentServer class skeleton with FastAPI app factory in src/holodeck/serve/server.py

**Checkpoint**: Foundation ready - all foundational tests pass - user story implementation can now begin

---

## Phase 3: User Story 1 - Start Agent Server with AG-UI Protocol (Priority: P1) MVP

**Goal**: Developer runs `holodeck serve agent.yaml` to expose agent via AG-UI protocol on configurable port

**Independent Test**: Start server with AG-UI protocol, connect with AG-UI compatible client, verify agent responds correctly

### Tests for User Story 1 (TDD)

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [x] T015 [P] [US1] Unit tests for AG-UI event mapping (lifecycle, text message, tool call events) in tests/unit/serve/test_protocols_agui.py
- [x] T016 [P] [US1] Unit tests for RunAgentInput to HoloDeck request mapping in tests/unit/serve/test_protocols_agui.py
- [x] T017 [P] [US1] Integration test for AG-UI protocol endpoint in tests/integration/serve/test_server_agui.py
- [x] T018 [P] [US1] Integration test for AG-UI streaming response in tests/integration/serve/test_server_agui.py

### Implementation for User Story 1

- [x] T019 [US1] Implement RunAgentInput to HoloDeck request mapping in src/holodeck/serve/protocols/agui.py
- [x] T020 [US1] Implement AG-UI EventEncoder wrapper for streaming responses in src/holodeck/serve/protocols/agui.py
- [x] T021 [US1] Implement AG-UI lifecycle events (RunStartedEvent, RunFinishedEvent, RunErrorEvent) in src/holodeck/serve/protocols/agui.py
- [x] T022 [US1] Implement AG-UI text message events (TextMessageStartEvent, TextMessageContentEvent, TextMessageEndEvent) in src/holodeck/serve/protocols/agui.py
- [x] T023 [US1] Implement AG-UI tool call events (ToolCallStartEvent, ToolCallArgsEvent, ToolCallEndEvent, ToolCallResultEvent) in src/holodeck/serve/protocols/agui.py
- [x] T024 [US1] Implement AGUIProtocol class extending Protocol ABC in src/holodeck/serve/protocols/agui.py
- [x] T025 [US1] Integrate AgentExecutor with AG-UI event stream in src/holodeck/serve/protocols/agui.py
- [x] T026 [US1] Wire AG-UI protocol endpoints into AgentServer in src/holodeck/serve/server.py
- [x] T027 [US1] Implement CLI serve command with --port option in src/holodeck/cli/commands/serve.py
- [x] T028 [US1] Add serve command to CLI entry point in src/holodeck/cli/main.py
- [x] T029 [US1] Implement server startup display with agent name, protocol, and URL in src/holodeck/cli/commands/serve.py

### AG-UI Multimodal Support (TDD)

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [x] T029a [P] [US1] Unit tests for AG-UI BinaryInputContent parsing in tests/unit/serve/test_protocols_agui.py
- [x] T029b [P] [US1] Unit tests for AG-UI multimodal content to FileInput conversion in tests/unit/serve/test_protocols_agui.py
- [x] T029c [P] [US1] Integration test for AG-UI with image content in tests/integration/serve/test_server_agui.py
- [x] T029d [P] [US1] Integration test for AG-UI with PDF/document content in tests/integration/serve/test_server_agui.py

### AG-UI Multimodal Implementation

- [x] T029e [US1] Implement BinaryInputContent parsing from AG-UI message content in src/holodeck/serve/protocols/agui.py
- [x] T029f [US1] Implement conversion of AG-UI binary content to FileInput for FileProcessor in src/holodeck/serve/protocols/agui.py
- [x] T029g [US1] Integrate FileProcessor output into agent message context in src/holodeck/serve/protocols/agui.py
- [x] T029h [US1] Handle inline base64, URL references, and file ID references in src/holodeck/serve/protocols/agui.py

**Checkpoint**: At this point, `holodeck serve agent.yaml` works with AG-UI protocol including multimodal - all US1 tests pass ✅

---

## Phase 4: User Story 2 - Start Agent Server with REST Protocol (Priority: P1)

**Goal**: Developer runs `holodeck serve agent.yaml --protocol rest` to expose agent at `/agent/{name}/chat` endpoints

**Independent Test**: Start server with REST protocol, make POST requests to chat endpoints, verify JSON/SSE responses

### Tests for User Story 2 (TDD)

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [x] T030 [P] [US2] Contract tests for OpenAPI spec compliance in tests/contract/serve/test_openapi_contract.py
- [x] T031 [P] [US2] Unit tests for REST protocol handlers in tests/unit/serve/test_protocols_rest.py
- [x] T032 [P] [US2] Unit tests for SSE event serialization (stream_start, message_delta, tool_call_*, stream_end, error) in tests/unit/serve/test_protocols_rest.py
- [x] T033 [P] [US2] Integration test for synchronous chat endpoint in tests/integration/serve/test_server_rest.py
- [x] T034 [P] [US2] Integration test for streaming chat endpoint (SSE) in tests/integration/serve/test_server_rest.py
- [x] T035 [P] [US2] Integration test for multimodal file upload (base64 JSON) in tests/integration/serve/test_server_rest.py
- [x] T036 [P] [US2] Integration test for multimodal file upload (multipart form-data) in tests/integration/serve/test_server_rest.py

### Implementation for User Story 2

- [x] T037 [US2] Implement RESTProtocol class extending Protocol ABC in src/holodeck/serve/protocols/rest.py
- [x] T038 [US2] Implement synchronous chat endpoint handler (/agent/{agent_name}/chat) in src/holodeck/serve/protocols/rest.py
- [x] T039 [US2] Implement SSE streaming chat endpoint handler (/agent/{agent_name}/chat/stream) in src/holodeck/serve/protocols/rest.py
- [x] T040 [US2] Implement SSE event serialization per sse-events.md contract (stream_start, message_delta, tool_call_*, stream_end, error) in src/holodeck/serve/protocols/rest.py
- [x] T041 [US2] Implement keepalive comments for long-running SSE streams in src/holodeck/serve/protocols/rest.py
- [x] T042 [US2] Implement multimodal file processing integration with FileProcessor in src/holodeck/serve/protocols/rest.py
- [x] T043 [US2] Implement multipart form-data file upload handling in src/holodeck/serve/protocols/rest.py
- [x] T044 [US2] Wire REST protocol endpoints into AgentServer in src/holodeck/serve/server.py
- [x] T045 [US2] Add --protocol flag to serve CLI command in src/holodeck/cli/commands/serve.py
- [x] T046 [US2] Enable FastAPI OpenAPI docs at root URL for REST protocol in src/holodeck/serve/server.py

**Checkpoint**: At this point, `holodeck serve agent.yaml --protocol rest` works - all US2 tests pass ✅

---

## Phase 5: User Story 3 - Health Check and Monitoring Endpoints (Priority: P2)

**Goal**: Server provides /health, /health/agent, and /ready endpoints for monitoring

**Independent Test**: Start server, request health endpoints, verify status responses

### Tests for User Story 3 (TDD)

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [ ] T047 [P] [US3] Integration test for /health endpoint in tests/integration/serve/test_health_endpoints.py
- [ ] T048 [P] [US3] Integration test for /health/agent endpoint in tests/integration/serve/test_health_endpoints.py
- [ ] T049 [P] [US3] Integration test for /ready endpoint in tests/integration/serve/test_health_endpoints.py
- [ ] T050 [P] [US3] Integration test for health response when agent not ready (503) in tests/integration/serve/test_health_endpoints.py

### Implementation for User Story 3

- [ ] T051 [US3] Implement /health endpoint returning HealthResponse in src/holodeck/serve/server.py
- [ ] T052 [P] [US3] Implement /health/agent endpoint with agent-specific health in src/holodeck/serve/server.py
- [ ] T053 [P] [US3] Implement /ready endpoint for readiness checks in src/holodeck/serve/server.py
- [ ] T054 [US3] Track server uptime and active session count for health responses in src/holodeck/serve/server.py

### Documentation Updates

- [ ] T054a [P] [US3] Create serve command guide at docs/guides/serve.md (usage, protocols, options, examples)
- [ ] T054b [P] [US3] Add serve API reference at docs/api/serve.md (endpoints, models, events)
- [ ] T054c [US3] Update README.md to include serve capability in Quick Start section
- [ ] T054d [US3] Update README.md Architecture section (remove "(Planned)" from Deployment, add serve details)
- [ ] T054e [US3] Add serve to docs/index.md navigation and documentation links

**Checkpoint**: Health endpoints working, documentation complete - all US3 tests pass

---

## Phase 6: User Story 4 - Session Management (Priority: P2)

**Goal**: Server maintains conversation context via session_id across multiple requests with 30-minute TTL

**Independent Test**: Send multiple messages with same session_id, verify context is maintained, verify session expiration

### Tests for User Story 4 (TDD)

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [ ] T055 [P] [US4] Integration test for automatic session creation in tests/integration/serve/test_session_management.py
- [ ] T056 [P] [US4] Integration test for session continuity across requests in tests/integration/serve/test_session_management.py
- [ ] T057 [P] [US4] Integration test for DELETE /sessions/{session_id} endpoint in tests/integration/serve/test_session_management.py
- [ ] T058 [P] [US4] Unit test for TTL cleanup background task in tests/unit/serve/test_session_store.py

### Implementation for User Story 4

- [ ] T059 [US4] Implement automatic session creation when no session_id provided in src/holodeck/serve/server.py
- [ ] T060 [US4] Implement session lookup and activity touch on each request in src/holodeck/serve/server.py
- [ ] T061 [US4] Implement DELETE /sessions/{session_id} endpoint in src/holodeck/serve/server.py
- [ ] T062 [US4] Implement background asyncio task for TTL cleanup (30-minute expiration) in src/holodeck/serve/session_store.py
- [ ] T063 [US4] Start cleanup task on server startup, stop on shutdown in src/holodeck/serve/server.py

**Checkpoint**: Session management fully functional - all US4 tests pass

---

## Phase 7: User Story 5 - Interactive Server Startup Feedback (Priority: P3)

**Goal**: Server displays clear startup information and supports --open flag to open browser

**Independent Test**: Start server, verify console output contains URL, endpoints, and agent info

### Tests for User Story 5 (TDD)

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [ ] T064 [P] [US5] Unit test for startup display output formatting in tests/unit/serve/test_cli_serve.py
- [ ] T065 [P] [US5] Unit test for --cors-origins flag parsing in tests/unit/serve/test_cli_serve.py
- [ ] T066 [P] [US5] Unit test for --debug flag enabling verbose logging in tests/unit/serve/test_cli_serve.py

### Implementation for User Story 5

- [ ] T067 [US5] Enhance startup display with available endpoints list in src/holodeck/cli/commands/serve.py
- [ ] T068 [P] [US5] Implement --open flag to open browser on startup in src/holodeck/cli/commands/serve.py
- [ ] T069 [P] [US5] Implement --cors-origins flag for CORS configuration in src/holodeck/cli/commands/serve.py
- [ ] T070 [US5] Implement --debug flag for full request/response logging in src/holodeck/cli/commands/serve.py
- [ ] T071 [US5] Add CORS middleware with configurable origins in src/holodeck/serve/middleware.py

**Checkpoint**: Full startup feedback and CLI options complete - all US5 tests pass

---

## Phase 8: Edge Cases and Error Handling

**Purpose**: Handle edge cases from spec.md

### Tests for Edge Cases (TDD)

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [ ] T072 [P] Unit test for invalid configuration rejection in tests/unit/serve/test_cli_serve.py
- [ ] T073 [P] Unit test for port-in-use error handling in tests/unit/serve/test_cli_serve.py
- [ ] T074 [P] Integration test for graceful error responses (503 when not ready) in tests/integration/serve/test_error_handling.py
- [ ] T075 [P] Integration test for graceful shutdown with in-flight requests in tests/integration/serve/test_error_handling.py

### Implementation for Edge Cases

- [ ] T076 Implement configuration validation before server start with clear error messages in src/holodeck/cli/commands/serve.py
- [ ] T077 [P] Implement port-in-use detection with helpful error message in src/holodeck/cli/commands/serve.py
- [ ] T078 [P] Implement graceful error responses during agent processing (503 when not ready) in src/holodeck/serve/server.py
- [ ] T079 Implement graceful shutdown on SIGINT/SIGTERM with in-flight request completion in src/holodeck/serve/server.py
- [ ] T080 Implement server state tracking (INITIALIZING, READY, RUNNING, SHUTTING_DOWN, STOPPED) in src/holodeck/serve/server.py

---

## Phase 9: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [ ] T081 Add debug logging with full request/response content in src/holodeck/serve/middleware.py
- [ ] T082 Add structured logging with timestamp, session_id, endpoint, latency in src/holodeck/serve/middleware.py
- [ ] T083 Validate all file size limits (10MB base64, 50MB per file, 100MB total) in src/holodeck/serve/models.py
- [ ] T084 Run quickstart.md validation scenarios manually
- [ ] T085 Verify OpenAPI contract matches implementation via contract tests

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Story 1 (Phase 3)**: Depends on Foundational - AG-UI protocol (MVP)
- **User Story 2 (Phase 4)**: Depends on Foundational - REST protocol (can run parallel to US1)
- **User Story 3 (Phase 5)**: Depends on Foundational - Health endpoints (can run parallel to US1/US2)
- **User Story 4 (Phase 6)**: Depends on US1 or US2 completion - Session management
- **User Story 5 (Phase 7)**: Depends on US1 or US2 completion - Startup feedback
- **Edge Cases (Phase 8)**: Depends on US1-US4 completion
- **Polish (Phase 9)**: Depends on all phases complete

### TDD Workflow Within Each Phase

1. Write all tests for the phase FIRST
2. Run tests - verify they FAIL
3. Implement code to make tests pass
4. Refactor if needed (tests still pass)
5. Move to next phase

### User Story Dependencies

- **User Story 1 (P1)**: AG-UI Protocol - No dependencies on other stories, MVP
- **User Story 2 (P1)**: REST Protocol - Can run parallel to US1
- **User Story 3 (P2)**: Health Endpoints - Can run parallel to US1/US2
- **User Story 4 (P2)**: Session Management - Builds on US1/US2 foundation
- **User Story 5 (P3)**: Startup Feedback - Enhancement to US1/US2

### Within Each User Story (TDD)

1. Write tests FIRST - ensure they FAIL
2. Models before services
3. Services before endpoints
4. Core implementation before integration
5. All tests pass before story complete

### Parallel Opportunities

- T003, T004 can run in parallel (Setup phase)
- T005, T006, T007 can run in parallel (Foundational tests)
- T009, T011, T012, T013 can run in parallel (Foundational implementation)
- US1, US2, US3 can start in parallel after Foundational phase
- All tests within a user story marked [P] can run in parallel
- T047, T048, T049, T050 can run in parallel (Health tests)
- T055, T056, T057, T058 can run in parallel (Session tests)
- T072, T073, T074, T075 can run in parallel (Edge case tests)

---

## Parallel Example: User Story 2 Tests (TDD)

```bash
# Launch ALL tests for User Story 2 together BEFORE implementation:
Task: "Contract tests for OpenAPI spec compliance in tests/contract/serve/test_openapi_contract.py"
Task: "Unit tests for REST protocol handlers in tests/unit/serve/test_protocols_rest.py"
Task: "Unit tests for SSE event serialization in tests/unit/serve/test_protocols_rest.py"
Task: "Integration test for synchronous chat endpoint in tests/integration/serve/test_server_rest.py"
Task: "Integration test for streaming chat endpoint in tests/integration/serve/test_server_rest.py"
Task: "Integration test for multimodal file upload (base64) in tests/integration/serve/test_server_rest.py"
Task: "Integration test for multimodal file upload (multipart) in tests/integration/serve/test_server_rest.py"

# Verify all tests FAIL, then implement...
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (tests first, then implementation)
3. Complete Phase 3: User Story 1 (tests first, then implementation)
4. **STOP and VALIDATE**: All tests pass, test with AG-UI compatible client
5. Deploy/demo if ready

### Incremental Delivery (TDD)

1. Setup + Foundational (tests → implementation) → Foundation ready
2. Add User Story 1 (tests → implementation) → MVP ready
3. Add User Story 2 (tests → implementation) → Dual-protocol support
4. Add User Story 3 (tests → implementation) → Production ready
5. Add User Story 4 (tests → implementation) → Stateful conversations
6. Add User Story 5 (tests → implementation) → Developer experience

### Parallel Team Strategy

With multiple developers:

1. Team completes Setup + Foundational together (TDD)
2. Once Foundational is done:
   - Developer A: User Story 1 (tests → implementation)
   - Developer B: User Story 2 (tests → implementation)
   - Developer C: User Story 3 (tests → implementation)
3. Stories complete and integrate independently - all tests pass

---

## Test Summary

| Phase | Test Tasks | Implementation Tasks | Doc Tasks | Status |
|-------|------------|---------------------|-----------|--------|
| Foundational | 3 | 7 | 0 | ✅ Complete |
| User Story 1 (text) | 4 | 11 | 0 | ✅ Complete |
| User Story 1 (multimodal) | 4 | 4 | 0 | ⏳ Pending |
| User Story 2 | 7 | 10 | 0 | ✅ Complete |
| User Story 3 | 4 | 4 | 5 | ⏳ Pending |
| User Story 4 | 4 | 5 | 0 | ⏳ Pending |
| User Story 5 | 3 | 5 | 0 | ⏳ Pending |
| Edge Cases | 4 | 5 | 0 | ⏳ Pending |
| **Total** | **33** | **51** | **5** | |

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- **TDD**: Write tests FIRST, verify they FAIL, then implement
- Each user story is independently completable and testable
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- All tests must pass before moving to next phase
- File paths follow plan.md structure: `src/holodeck/serve/`, `tests/*/serve/`
