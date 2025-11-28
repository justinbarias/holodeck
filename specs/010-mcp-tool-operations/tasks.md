# Tasks: MCP Tool Operations

**Input**: Design documents from `/specs/010-mcp-tool-operations/`
**Prerequisites**: plan.md ✅, spec.md ✅, research.md ✅, data-model.md ✅, contracts/mcp-config.md ✅, quickstart.md ✅

**Tests**: Not explicitly requested in the feature specification. Unit tests will be added as part of implementation for code quality assurance.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

**Key Simplification**: Semantic Kernel's MCP plugins (`MCPStdioPlugin`, `MCPSsePlugin`, etc.) handle full server lifecycle management via async context managers (`__aenter__`/`__aexit__`). HoloDeck does NOT need custom lifecycle wrappers - we use SK plugins directly with thin adapter classes for configuration translation only.

## Format: `[ID] [P?] [Story] Description`
- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions
- **Single project**: `src/holodeck/`, `tests/` at repository root (per plan.md)

---

## Phase 1: Setup (Shared Infrastructure) ✅

**Purpose**: Create project structure for MCP tool module

- [x] T001 Create MCP module directory structure: `src/holodeck/tools/mcp/__init__.py`
- [x] T002 [P] Create MCP errors module in `src/holodeck/tools/mcp/errors.py` with MCPError, MCPConfigError, MCPConnectionError, MCPTimeoutError, MCPProtocolError, MCPToolNotFoundError
- [x] T003 [P] Create TransportType and CommandType enums in `src/holodeck/models/tool.py`

---

## Phase 2: Foundational (Blocking Prerequisites) ✅

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

**Architecture**: MCP tools are registered as SK plugins directly on the kernel. SK handles:
- Lifecycle via async context managers (`__aenter__`/`__aexit__`)
- Tool discovery via `list_tools()`
- Tool invocation via the kernel's automatic function calling
- No custom wrappers or adapters needed

- [x] T004 Enhance MCPToolConfig model in `src/holodeck/models/tool.py` with all transport fields (transport, command, args, env, env_file, encoding, url, headers, timeout, sse_read_timeout, terminate_on_close, config, load_tools, load_prompts, request_timeout)
- [x] T005 [P] Add transport-specific Pydantic validators in `src/holodeck/models/tool.py` (validate command for stdio, url for sse/websocket/http, allowed commands only npx/uvx/docker)
- [x] T006 Create factory function in `src/holodeck/tools/mcp/factory.py` with `create_mcp_plugin(config: MCPToolConfig)` that returns the appropriate SK plugin instance (MCPStdioPlugin, MCPSsePlugin, MCPWebsocketPlugin, MCPStreamableHttpPlugin)
- [x] T007 [P] Add environment variable resolution in factory using existing `substitute_env_vars()` before passing to SK plugin constructors

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel ✅

---

## Phase 3: User Story 1 - Standard MCP Server Integration (Priority: P1) 🎯 MVP

**Goal**: Enable developers to define and invoke standard MCP servers via stdio transport in agent.yaml

**Independent Test**: Configure a simple MCP tool (e.g., filesystem server), call it through the agent, and verify the tool executes and returns results

**Architecture**: SK's `MCPStdioPlugin` is registered directly on the kernel. The kernel handles tool invocation automatically.

### Implementation for User Story 1

- [ ] T008 [US1] Implement stdio transport support in factory - translate MCPToolConfig to MCPStdioPlugin constructor args (command, args, env, encoding)
- [ ] T009 [US1] Integrate MCP plugin loading into AgentFactory - load MCP tools from agent.yaml and register as SK plugins on kernel
- [ ] T010 [US1] Add MCP plugin lifecycle management in AgentFactory - use async context manager to connect/disconnect MCP servers with kernel lifecycle
- [ ] T011 [P] [US1] Add unit tests for MCPToolConfig validation in `tests/unit/tools/mcp/test_config.py`
- [ ] T012 [P] [US1] Add unit tests for factory create_mcp_plugin() in `tests/unit/tools/mcp/test_factory.py`

**Checkpoint**: At this point, User Story 1 should be fully functional - agents can invoke stdio MCP servers

---

## Phase 4: User Story 2 - MCP Server Configuration (Priority: P1)

**Goal**: Support custom configuration including environment variables, config objects, and envFile loading

**Independent Test**: Configure an MCP tool with specific environment variables or configuration options and verify the MCP server receives and uses them

**Note**: Environment resolution happens in factory before creating SK plugin instance.

### Implementation for User Story 2

- [ ] T013 [US2] Implement env_file loading using existing `load_env_file()` in factory before env var resolution
- [ ] T014 [US2] Implement config passthrough to MCP server initialization (SK plugin's `env` parameter)
- [ ] T015 [P] [US2] Add unit tests for environment variable resolution in `tests/unit/tools/mcp/test_factory.py`
- [ ] T016 [P] [US2] Add unit tests for env_file loading in `tests/unit/tools/mcp/test_factory.py`

**Checkpoint**: At this point, User Stories 1 AND 2 should both work independently - full stdio MCP configuration support

---

## Phase 5: User Story 3 & 4 - Response Processing & Tool Discovery (Priority: P1)

**Goal**: MCP server responses rendered into agent context; tools automatically discovered

**Note**: SK handles both response processing AND tool discovery automatically:
- Response rendering into agent context
- Tool discovery via `load_tools=True` (default)
- Prompt discovery via `load_prompts=True`
- Notification handling for `tools/list_changed` and `prompts/list_changed`

### Implementation for User Stories 3 & 4

- [ ] T017 [US3/4] Pass `load_tools` and `load_prompts` config options to SK plugin constructors in factory
- [ ] T018 [P] [US3/4] Add integration test verifying tool discovery and invocation works end-to-end

**Checkpoint**: At this point, all P1 user stories complete - full stdio MCP functionality

---

## Phase 6: User Story 5 - MCP Error Handling (Priority: P2)

**Goal**: Provide clear, actionable error messages for MCP server failures

**Independent Test**: Intentionally cause MCP server failures and verify appropriate error messages are returned

**Note**: SK plugins raise exceptions for MCP errors. We wrap/translate these to HoloDeck error types for consistent error handling.

### Implementation for User Story 5

- [ ] T019 [US5] Add try/catch in AgentFactory MCP plugin initialization to translate SK exceptions to MCPConfigError/MCPConnectionError
- [ ] T020 [US5] Implement request_timeout passthrough to SK plugin constructors
- [ ] T021 [P] [US5] Add unit tests for error handling scenarios in `tests/unit/tools/mcp/test_factory.py`

**Checkpoint**: At this point, User Story 5 complete - robust error handling for stdio MCP

---

## Phase 7: User Story 6 - HTTP/SSE MCP Servers (Priority: P2)

**Goal**: Support remote MCP servers via HTTP/SSE transport with authentication headers

**Independent Test**: Configure an MCP tool with SSE transport and verify connection, authentication, and request/response flow

### Implementation for User Story 6

- [ ] T022 [US6] Add SSE transport support in factory - translate MCPToolConfig to MCPSsePlugin constructor args (url, headers, timeout, sse_read_timeout)
- [ ] T023 [US6] Implement header environment variable resolution using `substitute_env_vars()` in factory
- [ ] T024 [P] [US6] Add unit tests for SSE transport in `tests/unit/tools/mcp/test_factory.py`

**Checkpoint**: At this point, User Story 6 complete - SSE transport support

---

## Phase 8: User Story 7 - WebSocket MCP Servers (Priority: P3)

**Goal**: Support bidirectional WebSocket MCP communication

**Independent Test**: Configure an MCP tool with WebSocket transport and verify bidirectional communication

### Implementation for User Story 7

- [ ] T025 [US7] Add WebSocket transport support in factory - translate MCPToolConfig to MCPWebsocketPlugin constructor args (url)
- [ ] T026 [P] [US7] Add unit tests for WebSocket transport in `tests/unit/tools/mcp/test_factory.py`

**Checkpoint**: At this point, User Story 7 complete - WebSocket transport support

---

## Phase 9: User Story 8 - Streamable HTTP Transport (Priority: P3)

**Goal**: Support HTTP with streaming response and terminate_on_close option

**Independent Test**: Configure streamable HTTP transport and verify streaming responses are handled correctly

### Implementation for User Story 8

- [ ] T027 [US8] Add HTTP transport support in factory - translate MCPToolConfig to MCPStreamableHttpPlugin constructor args (url, headers, timeout, sse_read_timeout, terminate_on_close)
- [ ] T028 [P] [US8] Add unit tests for HTTP transport in `tests/unit/tools/mcp/test_factory.py`

**Checkpoint**: At this point, all user stories complete - full MCP transport support

---

## Phase 10: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [ ] T029 [P] Export public API from `src/holodeck/tools/mcp/__init__.py` (create_mcp_plugin, MCP errors)
- [ ] T030 [P] Update `src/holodeck/tools/__init__.py` to export MCP module
- [ ] T031 [P] Add integration tests for stdio MCP server in `tests/integration/tools/test_mcp_integration.py`
- [ ] T032 Run quickstart.md validation with actual MCP server
- [ ] T033 Run `make format && make lint && make type-check && make security` to ensure code quality

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3-9)**: All depend on Foundational phase completion
  - P1 stories (US1-US4) form the MVP - Phase 3, 4, 5
  - P2 stories (US5-US6) add robustness - Phase 6, 7
  - P3 stories (US7-US8) add advanced transports - Phase 8, 9
- **Polish (Phase 10)**: Depends on all desired user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2) - Core stdio support
- **User Story 2 (P1)**: Can start after US1 (adds env/config to factory)
- **User Story 3 & 4 (P1)**: Can start after US1 (SK handles automatically, just config passthrough)
- **User Story 5 (P2)**: Can start after US1 (error wrapping in factory)
- **User Story 6 (P2)**: Can start after Foundational - Independent transport
- **User Story 7 (P3)**: Can start after Foundational - Independent transport
- **User Story 8 (P3)**: Can start after Foundational - Independent transport

### Parallel Opportunities

- All Setup tasks marked [P] can run in parallel
- All Foundational tasks marked [P] can run in parallel (within Phase 2)
- Once Foundational phase completes, transport implementations (US6, US7, US8) can start in parallel
- All unit tests marked [P] can run in parallel

---

## Implementation Strategy

### MVP First (User Stories 1-4 Only)

1. Complete Phase 1: Setup (T001-T003)
2. Complete Phase 2: Foundational (T004-T007) - CRITICAL
3. Complete Phase 3: User Story 1 (T008-T012) - Basic stdio MCP
4. Complete Phase 4: User Story 2 (T013-T016) - Environment/config support
5. Complete Phase 5: User Stories 3 & 4 (T017-T018) - SK handles automatically
6. **STOP and VALIDATE**: Test with real MCP servers (filesystem, memory)
7. Deploy/demo if ready - MVP complete!

### Incremental Delivery

1. Complete Setup + Foundational → Foundation ready
2. Add User Story 1 → Test independently → Basic stdio MCP works
3. Add User Story 2 → Test independently → Configuration support works
4. Add User Stories 3 & 4 → Test independently → SK handles response/discovery (MVP!)
5. Add User Story 5 → Test independently → Error handling robust
6. Add User Story 6 → Test independently → SSE transport works
7. Add User Story 7 → Test independently → WebSocket transport works
8. Add User Story 8 → Test independently → HTTP transport works
9. Each story adds value without breaking previous stories

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- DRY: Reuse existing `substitute_env_vars()`, `load_env_file()` from `env_loader.py`
- DRY: Extend existing error hierarchy from `holodeck.lib.errors`
- **KEY SIMPLIFICATION**: SK plugins handle lifecycle, tool discovery, response rendering - no custom wrappers needed
- Avoid: vague tasks, same file conflicts, cross-story dependencies that break independence

## Task Summary

| Phase | Tasks | Description |
|-------|-------|-------------|
| 1. Setup | T001-T003 | Module structure, errors, enums |
| 2. Foundational | T004-T007 | MCPToolConfig model, validators, factory |
| 3. US1 (Stdio) | T008-T012 | Stdio transport, AgentFactory integration |
| 4. US2 (Config) | T013-T016 | Env vars, env_file, config passthrough |
| 5. US3&4 (Response/Discovery) | T017-T018 | SK handles automatically, config only |
| 6. US5 (Errors) | T019-T021 | Error translation, timeout |
| 7. US6 (SSE) | T022-T024 | SSE transport support |
| 8. US7 (WebSocket) | T025-T026 | WebSocket transport support |
| 9. US8 (HTTP) | T027-T028 | Streamable HTTP transport |
| 10. Polish | T029-T033 | Exports, integration tests, validation |

**Total: 33 tasks** (reduced from 59 by leveraging SK's built-in capabilities)
