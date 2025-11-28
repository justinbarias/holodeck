# Tasks: MCP Tool Operations

**Input**: Design documents from `/specs/010-mcp-tool-operations/`
**Prerequisites**: plan.md ✅, spec.md ✅, research.md ✅, data-model.md ✅, contracts/mcp-config.md ✅, quickstart.md ✅

**Tests**: Not explicitly requested in the feature specification. Unit tests will be added as part of implementation for code quality assurance.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`
- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions
- **Single project**: `src/holodeck/`, `tests/` at repository root (per plan.md)

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Create project structure for MCP tool module

- [ ] T001 Create MCP module directory structure: `src/holodeck/tools/mcp/__init__.py`
- [ ] T002 [P] Create MCP errors module in `src/holodeck/tools/mcp/errors.py` with MCPError, MCPConfigError, MCPConnectionError, MCPTimeoutError, MCPProtocolError, MCPToolNotFoundError
- [ ] T003 [P] Create TransportType and CommandType enums in `src/holodeck/models/tool.py`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [ ] T004 Enhance MCPToolConfig model in `src/holodeck/models/tool.py` with all transport fields (transport, command, args, env, env_file, encoding, url, headers, timeout, sse_read_timeout, terminate_on_close, config, load_tools, load_prompts, request_timeout)
- [ ] T005 [P] Add transport-specific Pydantic validators in `src/holodeck/models/tool.py` (validate command for stdio, url for sse/websocket/http, allowed commands only npx/uvx/docker)
- [ ] T006 Create ContentBlock models (TextContent, ImageContent, AudioContent, BinaryContent) in `src/holodeck/tools/mcp/content.py`
- [ ] T007 [P] Create MCPToolResult model in `src/holodeck/tools/mcp/content.py` with success, content, error, metadata fields
- [ ] T008 Create MCPPluginWrapper base class in `src/holodeck/tools/mcp/plugin.py` with connect(), disconnect(), call_tool(), get_prompt(), list_tools(), list_prompts() interface
- [ ] T009 Create plugin factory in `src/holodeck/tools/mcp/factory.py` with create_mcp_plugin(config: MCPToolConfig) function that returns appropriate SK plugin wrapper

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Standard MCP Server Integration (Priority: P1) 🎯 MVP

**Goal**: Enable developers to define and invoke standard MCP servers via stdio transport in agent.yaml

**Independent Test**: Configure a simple MCP tool (e.g., filesystem server), call it through the agent, and verify the tool executes and returns results

### Implementation for User Story 1

- [ ] T010 [US1] Implement MCPStdioPluginWrapper in `src/holodeck/tools/mcp/plugin.py` wrapping SK MCPStdioPlugin with command, args, env, encoding support
- [ ] T011 [US1] Implement async context manager (`__aenter__`, `__aexit__`) in MCPPluginWrapper for proper lifecycle management in `src/holodeck/tools/mcp/plugin.py`
- [ ] T012 [US1] Implement call_tool() method in MCPPluginWrapper that invokes MCP server tools and returns MCPToolResult in `src/holodeck/tools/mcp/plugin.py`
- [ ] T013 [US1] Add tool name normalization (replace invalid chars with "-") in MCPPluginWrapper in `src/holodeck/tools/mcp/plugin.py`
- [ ] T014 [US1] Register stdio transport in factory create_mcp_plugin() in `src/holodeck/tools/mcp/factory.py`
- [ ] T015 [P] [US1] Add unit tests for MCPToolConfig validation in `tests/unit/tools/mcp/test_config.py`
- [ ] T016 [P] [US1] Add unit tests for MCPStdioPluginWrapper in `tests/unit/tools/mcp/test_plugin.py`

**Checkpoint**: At this point, User Story 1 should be fully functional - agents can invoke stdio MCP servers

---

## Phase 4: User Story 2 - MCP Server Configuration (Priority: P1)

**Goal**: Support custom configuration including environment variables, config objects, and envFile loading

**Independent Test**: Configure an MCP tool with specific environment variables or configuration options and verify the MCP server receives and uses them

### Implementation for User Story 2

- [ ] T017 [US2] Implement environment variable resolution using existing `substitute_env_vars()` in MCPPluginWrapper in `src/holodeck/tools/mcp/plugin.py`
- [ ] T018 [US2] Implement env_file loading using existing `load_env_file()` in MCPPluginWrapper in `src/holodeck/tools/mcp/plugin.py`
- [ ] T019 [US2] Implement config passthrough to MCP server initialization in MCPPluginWrapper in `src/holodeck/tools/mcp/plugin.py`
- [ ] T020 [US2] Implement encoding parameter for stdin/stdout streams in MCPStdioPluginWrapper in `src/holodeck/tools/mcp/plugin.py`
- [ ] T021 [P] [US2] Add unit tests for environment variable resolution in `tests/unit/tools/mcp/test_config.py`
- [ ] T022 [P] [US2] Add unit tests for env_file loading in `tests/unit/tools/mcp/test_config.py`

**Checkpoint**: At this point, User Stories 1 AND 2 should both work independently - full stdio MCP configuration support

---

## Phase 5: User Story 3 - MCP Response Processing (Priority: P1)

**Goal**: Parse MCP server responses and convert content blocks to HoloDeck internal representation

**Independent Test**: Call an MCP tool that returns structured data and verify the response is correctly parsed and converted

### Implementation for User Story 3

- [ ] T023 [US3] Implement content type conversion from MCP TextContent to internal TextContent in `src/holodeck/tools/mcp/content.py`
- [ ] T024 [P] [US3] Implement content type conversion from MCP ImageContent to internal ImageContent in `src/holodeck/tools/mcp/content.py`
- [ ] T025 [P] [US3] Implement content type conversion from MCP AudioContent to internal AudioContent in `src/holodeck/tools/mcp/content.py`
- [ ] T026 [P] [US3] Implement content type conversion from MCP EmbeddedResource/ResourceLink to internal BinaryContent in `src/holodeck/tools/mcp/content.py`
- [ ] T027 [US3] Implement convert_mcp_content() function that handles all MCP content types in `src/holodeck/tools/mcp/content.py`
- [ ] T028 [US3] Integrate content conversion into call_tool() response processing in `src/holodeck/tools/mcp/plugin.py`
- [ ] T029 [P] [US3] Add unit tests for content type conversion in `tests/unit/tools/mcp/test_content.py`

**Checkpoint**: At this point, User Stories 1, 2, AND 3 should work - full stdio MCP with response processing

---

## Phase 6: User Story 4 - MCP Tool Discovery (Priority: P1)

**Goal**: Automatically discover and register tools and prompts from MCP servers

**Independent Test**: Connect to an MCP server and verify all server-provided tools are discovered and callable

### Implementation for User Story 4

- [ ] T030 [US4] Implement list_tools() method that returns discovered tool definitions in `src/holodeck/tools/mcp/plugin.py`
- [ ] T031 [US4] Implement list_prompts() method that returns discovered prompt definitions in `src/holodeck/tools/mcp/plugin.py`
- [ ] T032 [US4] Implement get_prompt() method to retrieve MCP prompts in `src/holodeck/tools/mcp/plugin.py`
- [ ] T033 [US4] Implement tool list refresh on `notifications/tools/list_changed` notification in `src/holodeck/tools/mcp/plugin.py`
- [ ] T034 [US4] Implement prompt list refresh on `notifications/prompts/list_changed` notification in `src/holodeck/tools/mcp/plugin.py`
- [ ] T035 [US4] Implement load_tools and load_prompts configuration options in MCPPluginWrapper in `src/holodeck/tools/mcp/plugin.py`
- [ ] T036 [P] [US4] Add unit tests for tool/prompt discovery in `tests/unit/tools/mcp/test_plugin.py`

**Checkpoint**: At this point, all P1 user stories complete - full stdio MCP functionality

---

## Phase 7: User Story 5 - MCP Error Handling (Priority: P2)

**Goal**: Provide clear, actionable error messages for MCP server failures

**Independent Test**: Intentionally cause MCP server failures and verify appropriate error messages are returned

### Implementation for User Story 5

- [ ] T037 [US5] Implement connection error handling with clear command path in error message in `src/holodeck/tools/mcp/plugin.py`
- [ ] T038 [US5] Implement server crash detection and MCPConnectionError in `src/holodeck/tools/mcp/plugin.py`
- [ ] T039 [US5] Implement MCP protocol error extraction and MCPProtocolError in `src/holodeck/tools/mcp/plugin.py`
- [ ] T040 [US5] Implement request_timeout handling with MCPTimeoutError in `src/holodeck/tools/mcp/plugin.py`
- [ ] T041 [US5] Implement MCPToolNotFoundError when tool not found on server in `src/holodeck/tools/mcp/plugin.py`
- [ ] T042 [P] [US5] Add unit tests for error handling scenarios in `tests/unit/tools/mcp/test_plugin.py`

**Checkpoint**: At this point, User Story 5 complete - robust error handling for stdio MCP

---

## Phase 8: User Story 6 - HTTP/SSE MCP Servers (Priority: P2)

**Goal**: Support remote MCP servers via HTTP/SSE transport with authentication headers

**Independent Test**: Configure an MCP tool with SSE transport and verify connection, authentication, and request/response flow

### Implementation for User Story 6

- [ ] T043 [US6] Implement MCPSsePluginWrapper in `src/holodeck/tools/mcp/plugin.py` wrapping SK MCPSsePlugin with url, headers, timeout, sse_read_timeout
- [ ] T044 [US6] Implement header environment variable resolution in MCPSsePluginWrapper in `src/holodeck/tools/mcp/plugin.py`
- [ ] T045 [US6] Register SSE transport in factory create_mcp_plugin() in `src/holodeck/tools/mcp/factory.py`
- [ ] T046 [P] [US6] Add unit tests for MCPSsePluginWrapper in `tests/unit/tools/mcp/test_plugin.py`

**Checkpoint**: At this point, User Story 6 complete - SSE transport support

---

## Phase 9: User Story 7 - WebSocket MCP Servers (Priority: P3)

**Goal**: Support bidirectional WebSocket MCP communication

**Independent Test**: Configure an MCP tool with WebSocket transport and verify bidirectional communication

### Implementation for User Story 7

- [ ] T047 [US7] Implement MCPWebsocketPluginWrapper in `src/holodeck/tools/mcp/plugin.py` wrapping SK MCPWebsocketPlugin with url
- [ ] T048 [US7] Register WebSocket transport in factory create_mcp_plugin() in `src/holodeck/tools/mcp/factory.py`
- [ ] T049 [P] [US7] Add unit tests for MCPWebsocketPluginWrapper in `tests/unit/tools/mcp/test_plugin.py`

**Checkpoint**: At this point, User Story 7 complete - WebSocket transport support

---

## Phase 10: User Story 8 - Streamable HTTP Transport (Priority: P3)

**Goal**: Support HTTP with streaming response and terminate_on_close option

**Independent Test**: Configure streamable HTTP transport and verify streaming responses are handled correctly

### Implementation for User Story 8

- [ ] T050 [US8] Implement MCPStreamableHttpPluginWrapper in `src/holodeck/tools/mcp/plugin.py` wrapping SK MCPStreamableHttpPlugin with url, headers, timeout, sse_read_timeout, terminate_on_close
- [ ] T051 [US8] Register HTTP transport in factory create_mcp_plugin() in `src/holodeck/tools/mcp/factory.py`
- [ ] T052 [P] [US8] Add unit tests for MCPStreamableHttpPluginWrapper in `tests/unit/tools/mcp/test_plugin.py`

**Checkpoint**: At this point, all user stories complete - full MCP transport support

---

## Phase 11: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [ ] T053 [P] Export public API from `src/holodeck/tools/mcp/__init__.py` (MCPPluginWrapper, MCPToolResult, ContentBlock, create_mcp_plugin, MCP errors)
- [ ] T054 [P] Add MCP logging callbacks with log level mapping in `src/holodeck/tools/mcp/plugin.py`
- [ ] T055 [P] Add message handler callbacks for server exceptions in `src/holodeck/tools/mcp/plugin.py`
- [ ] T056 Update `src/holodeck/tools/__init__.py` to export MCP module
- [ ] T057 [P] Add integration tests for stdio MCP server in `tests/integration/tools/test_mcp_integration.py`
- [ ] T058 Run quickstart.md validation with actual MCP server
- [ ] T059 Run `make format && make lint && make type-check && make security` to ensure code quality

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3-10)**: All depend on Foundational phase completion
  - P1 stories (US1-US4) form the MVP
  - P2 stories (US5-US6) add robustness
  - P3 stories (US7-US8) add advanced transports
- **Polish (Phase 11)**: Depends on all desired user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 2 (P1)**: Can start after US1 (builds on plugin structure)
- **User Story 3 (P1)**: Can start after US1 (builds on plugin structure)
- **User Story 4 (P1)**: Can start after US1 (builds on plugin structure)
- **User Story 5 (P2)**: Can start after US1-US4 (error handling for all core features)
- **User Story 6 (P2)**: Can start after Foundational (Phase 2) - Independent transport
- **User Story 7 (P3)**: Can start after Foundational (Phase 2) - Independent transport
- **User Story 8 (P3)**: Can start after Foundational (Phase 2) - Independent transport

### Within Each User Story

- Models before services
- Core implementation before integration
- Unit tests with each feature
- Story complete before moving to next priority

### Parallel Opportunities

- All Setup tasks marked [P] can run in parallel
- All Foundational tasks marked [P] can run in parallel (within Phase 2)
- Once Foundational phase completes, transport implementations (US6, US7, US8) can start in parallel
- All unit tests for a user story marked [P] can run in parallel

---

## Parallel Example: Phase 2 (Foundational)

```bash
# Launch all parallel foundational tasks together:
Task: "Add transport-specific Pydantic validators in src/holodeck/models/tool.py"
Task: "Create MCPToolResult model in src/holodeck/tools/mcp/content.py"
```

## Parallel Example: User Story 3 (Content Conversion)

```bash
# Launch all content type conversions together:
Task: "Implement content type conversion from MCP ImageContent in src/holodeck/tools/mcp/content.py"
Task: "Implement content type conversion from MCP AudioContent in src/holodeck/tools/mcp/content.py"
Task: "Implement content type conversion from MCP EmbeddedResource/ResourceLink in src/holodeck/tools/mcp/content.py"
```

---

## Implementation Strategy

### MVP First (User Stories 1-4 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - blocks all stories)
3. Complete Phase 3: User Story 1 (Standard MCP Server Integration)
4. Complete Phase 4: User Story 2 (MCP Server Configuration)
5. Complete Phase 5: User Story 3 (MCP Response Processing)
6. Complete Phase 6: User Story 4 (MCP Tool Discovery)
7. **STOP and VALIDATE**: Test with real MCP servers (filesystem, memory)
8. Deploy/demo if ready - MVP complete!

### Incremental Delivery

1. Complete Setup + Foundational → Foundation ready
2. Add User Story 1 → Test independently → Basic stdio MCP works
3. Add User Story 2 → Test independently → Configuration support works
4. Add User Story 3 → Test independently → Response processing works
5. Add User Story 4 → Test independently → Tool discovery works (MVP!)
6. Add User Story 5 → Test independently → Error handling robust
7. Add User Story 6 → Test independently → SSE transport works
8. Add User Story 7 → Test independently → WebSocket transport works
9. Add User Story 8 → Test independently → HTTP transport works
10. Each story adds value without breaking previous stories

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- DRY: Reuse existing `substitute_env_vars()`, `load_env_file()` from `env_loader.py`
- DRY: Extend existing error hierarchy from `holodeck.lib.errors`
- Avoid: vague tasks, same file conflicts, cross-story dependencies that break independence
