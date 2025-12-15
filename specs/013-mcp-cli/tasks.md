# Tasks: MCP CLI Command Group

**Input**: Design documents from `/specs/013-mcp-cli/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Tests**: Test tasks are included within each phase to ensure incremental validation.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3, US4)
- Include exact file paths in descriptions

## Path Conventions

- **Single project**: `src/holodeck/`, `tests/` at repository root

---

## Phase 1: Setup (Shared Infrastructure) ✅

**Purpose**: Project initialization and data models for registry API

- [x] T001 [P] Create RegistryServer and related Pydantic models in src/holodeck/models/registry.py
- [x] T002 [P] Add mcp_servers field to GlobalConfig model in src/holodeck/models/config.py
- [x] T002a [P] Add registry_name field to MCPTool model in src/holodeck/models/tool.py for duplicate detection
- [x] T003 [P] Create custom exceptions (RegistryConnectionError, RegistryAPIError, ServerNotFoundError) in src/holodeck/lib/errors.py

---

## Phase 2: Foundational (Blocking Prerequisites) ✅

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**CRITICAL**: No user story work can begin until this phase is complete

- [x] T004 Implement MCPRegistryClient class in src/holodeck/services/mcp_registry.py with search(), get_server(), list_versions() methods
- [x] T005 Implement registry_to_mcp_tool() transformation function in src/holodeck/services/mcp_registry.py
- [x] T006 Create mcp command group skeleton in src/holodeck/cli/commands/mcp.py with Click group and help text
- [x] T007 Register mcp command group in src/holodeck/cli/main.py

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Search MCP Registry (Priority: P1)

**Goal**: Enable developers to discover MCP servers from the official registry

**Independent Test**: Run `holodeck mcp search <query>` and verify results are returned with name, description, and transports

### Implementation for User Story 1

- [x] T008 [US1] Implement `holodeck mcp search` command in src/holodeck/cli/commands/mcp.py with QUERY argument
- [x] T009 [US1] Add --limit and --json options to search command in src/holodeck/cli/commands/mcp.py
- [x] T010 [US1] Implement table output formatter for search results in src/holodeck/cli/commands/mcp.py
- [x] T011 [US1] Implement JSON output formatter for search results in src/holodeck/cli/commands/mcp.py
- [x] T012 [US1] Add error handling for network timeout, API errors, and empty results in search command
- [x] T013 [US1] Add pagination support (next_cursor handling) to search command

### Tests for User Story 1

- [x] T013a [US1] [P] Add unit tests for search command basic functionality in tests/unit/cli/commands/test_mcp_search.py
- [x] T013b [US1] [P] Add unit tests for --limit and --json options in tests/unit/cli/commands/test_mcp_search.py
- [x] T013c [US1] [P] Add unit tests for table output formatter in tests/unit/cli/commands/test_mcp_search.py
- [x] T013d [US1] [P] Add unit tests for JSON output formatter in tests/unit/cli/commands/test_mcp_search.py
- [x] T013e [US1] [P] Add unit tests for error handling (network timeout, API errors, empty results) in tests/unit/cli/commands/test_mcp_search.py
- [x] T013f [US1] [P] Add unit tests for pagination support (next_cursor handling) in tests/unit/cli/commands/test_mcp_search.py

**Checkpoint**: At this point, User Story 1 should be fully functional and tested - users can search the MCP registry

---

## Phase 4: User Story 2 - Add MCP Server to Agent (Priority: P1) ✅

**Goal**: Enable developers to add MCP servers to their agent configuration

**Independent Test**: Run `holodeck mcp add <server-name>` and verify agent.yaml is updated with correct MCPTool configuration

### Implementation for User Story 2

- [x] T014 [US2] Implement add_mcp_server_to_agent() helper function in src/holodeck/config/loader.py
- [x] T015 [US2] Implement add_mcp_server_to_global() helper function in src/holodeck/config/loader.py
- [x] T016 [US2] Implement save_global_config() function in src/holodeck/config/loader.py (creates ~/.holodeck/ if needed)
- [x] T017 [US2] Implement `holodeck mcp add` command in src/holodeck/cli/commands/mcp.py with SERVER argument
- [x] T018 [US2] Add --agent, -g/--global, --version, --transport, --name options to add command
- [x] T019 [US2] Implement duplicate detection logic in add command using registry_name for registry servers, name for manual servers, with short-name conflict warnings
- [x] T020 [US2] Display required environment variables after successful add
- [x] T021 [US2] Add error handling for no agent.yaml found, registry errors, and file write errors

### Tests for User Story 2

- [x] T021a [US2] [P] Add unit tests for add_mcp_server_to_agent() helper in tests/unit/config/test_mcp_helpers.py
- [x] T021b [US2] [P] Add unit tests for add_mcp_server_to_global() helper in tests/unit/config/test_mcp_helpers.py
- [x] T021c [US2] [P] Add unit tests for save_global_config() in tests/unit/config/test_mcp_helpers.py
- [x] T021d [US2] [P] Add unit tests for add command basic functionality in tests/unit/cli/commands/test_mcp_add.py
- [x] T021e [US2] [P] Add unit tests for --agent, --global, --version, --transport, --name options in tests/unit/cli/commands/test_mcp_add.py
- [x] T021f [US2] [P] Add unit tests for duplicate detection logic in tests/unit/cli/commands/test_mcp_add.py
- [x] T021g [US2] [P] Add unit tests for error handling (no agent.yaml, registry errors, file write errors) in tests/unit/cli/commands/test_mcp_add.py

**Checkpoint**: At this point, User Stories 1 AND 2 work and are tested - users can search and add MCP servers

---

## Phase 5: User Story 3 - List Installed MCP Servers (Priority: P2) ✅

**Goal**: Enable developers to view all MCP servers configured in agent and/or global config

**Independent Test**: Run `holodeck mcp list` after adding servers and verify output shows all configured servers

### Implementation for User Story 3

- [x] T022 [US3] Implement get_mcp_servers_from_agent() helper function in src/holodeck/config/loader.py
- [x] T023 [US3] Implement get_mcp_servers_from_global() helper function in src/holodeck/config/loader.py
- [x] T024 [US3] Implement `holodeck mcp list` command in src/holodeck/cli/commands/mcp.py
- [x] T025 [US3] Add --agent, -g/--global, --all, --json options to list command
- [x] T026 [US3] Implement table output with SOURCE column showing agent/global origin
- [x] T027 [US3] Add empty state message "No MCP servers configured. Use 'holodeck mcp search' to find available servers."

### Tests for User Story 3

- [x] T027a [US3] [P] Add unit tests for get_mcp_servers_from_agent() helper in tests/unit/config/test_mcp_helpers.py
- [x] T027b [US3] [P] Add unit tests for get_mcp_servers_from_global() helper in tests/unit/config/test_mcp_helpers.py
- [x] T027c [US3] [P] Add unit tests for list command basic functionality in tests/unit/cli/commands/test_mcp_list.py
- [x] T027d [US3] [P] Add unit tests for --agent, --global, --all, --json options in tests/unit/cli/commands/test_mcp_list.py
- [x] T027e [US3] [P] Add unit tests for table output with SOURCE column in tests/unit/cli/commands/test_mcp_list.py
- [x] T027f [US3] [P] Add unit tests for empty state message in tests/unit/cli/commands/test_mcp_list.py

**Checkpoint**: At this point, User Stories 1, 2, AND 3 work and are tested - users can search, add, and list servers

---

## Phase 6: User Story 4 - Remove MCP Server (Priority: P2)

**Goal**: Enable developers to remove MCP servers from their configuration

**Independent Test**: Run `holodeck mcp remove <server-name>` and verify the server is removed from configuration file

### Implementation for User Story 4

- [x] T028 [US4] Implement remove_mcp_server_from_agent() helper function in src/holodeck/config/loader.py
- [x] T029 [US4] Implement remove_mcp_server_from_global() helper function in src/holodeck/config/loader.py
- [x] T030 [US4] Implement `holodeck mcp remove` command in src/holodeck/cli/commands/mcp.py with SERVER argument
- [x] T031 [US4] Add --agent, -g/--global options to remove command
- [x] T032 [US4] Add error handling for server not found and file write errors

### Tests for User Story 4

- [x] T032a [US4] [P] Add unit tests for remove_mcp_server_from_agent() helper in tests/unit/config/test_mcp_helpers.py
- [x] T032b [US4] [P] Add unit tests for remove_mcp_server_from_global() helper in tests/unit/config/test_mcp_helpers.py
- [x] T032c [US4] [P] Add unit tests for remove command basic functionality in tests/unit/cli/commands/test_mcp_remove.py
- [x] T032d [US4] [P] Add unit tests for --agent, --global options in tests/unit/cli/commands/test_mcp_remove.py
- [x] T032e [US4] [P] Add unit tests for error handling (server not found, file write errors) in tests/unit/cli/commands/test_mcp_remove.py

**Checkpoint**: All user stories should now be independently functional and tested

---

## Phase 7: Config Merge Integration ✅

**Purpose**: Integrate global MCP servers into agent config loading

### Implementation for Config Merge

- [x] T033 Implement _merge_mcp_servers() method in src/holodeck/config/loader.py
- [x] T034 Extend merge_configs() to call _merge_mcp_servers() for global MCP server merging
- [x] T035 Add merge precedence logic (agent-level MCP tools override global with same name)

### Tests for Config Merge

- [x] T035a [P] Add unit tests for _merge_mcp_servers() method in tests/unit/config/test_config_merge.py
- [x] T035b [P] Add unit tests for merge_configs() with MCP server merging in tests/unit/config/test_config_merge.py
- [x] T035c [P] Add unit tests for merge precedence logic (agent overrides global) in tests/unit/config/test_config_merge.py

**Checkpoint**: Config merge integration complete and tested - global MCP servers merge into agent config at runtime

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: Integration tests, foundational unit tests, and final validation

### Foundational Tests (Registry Client)

- [ ] T036 [P] Add unit tests for MCPRegistryClient.search() in tests/unit/services/test_mcp_registry.py
- [ ] T036a [P] Add unit tests for MCPRegistryClient.get_server() in tests/unit/services/test_mcp_registry.py
- [ ] T036b [P] Add unit tests for MCPRegistryClient.list_versions() in tests/unit/services/test_mcp_registry.py
- [ ] T036c [P] Add unit tests for registry_to_mcp_tool() transformation in tests/unit/services/test_mcp_registry.py

### Integration Tests

- [ ] T037 [P] Add integration test for search → add workflow in tests/integration/cli/test_mcp_integration.py
- [ ] T037a [P] Add integration test for add → list → remove workflow in tests/integration/cli/test_mcp_integration.py
- [ ] T037b [P] Add integration test for global vs agent config interactions in tests/integration/cli/test_mcp_integration.py
- [ ] T037c [P] Add integration test for config merge with global MCP servers in tests/integration/cli/test_mcp_integration.py

### Final Validation

- [ ] T038 Run quickstart.md validation - verify all documented commands work
- [ ] T039 Run code quality checks (make format, make lint, make type-check, make security)

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phases 3-6)**: All depend on Foundational phase completion
  - US1 (Search) and US2 (Add) are P1 priority
  - US3 (List) and US4 (Remove) are P2 priority
  - User stories can proceed in parallel (if staffed) or sequentially by priority
- **Config Merge (Phase 7)**: Can start after Phase 2, independent of user stories
- **Polish (Phase 8)**: Depends on all user stories being complete

### User Story Dependencies

- **User Story 1 (Search - P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 2 (Add - P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 3 (List - P2)**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 4 (Remove - P2)**: Can start after Foundational (Phase 2) - No dependencies on other stories

### Within Each Phase

- Setup tasks (T001-T003) can all run in parallel (different files)
- Foundational tasks must complete sequentially: T004 → T005 → T006 → T007
- Within each user story: helper functions → command implementation → options → error handling

### Parallel Opportunities

**Phase 1 (Setup)**: T001, T002, T003 can all run in parallel

**Phase 3-6 (User Stories)**: After Phase 2 completes, all user stories can start in parallel:
```bash
# Team can split work:
Developer A: T008-T013f (US1 Search + tests)
Developer B: T014-T021g (US2 Add + tests)
Developer C: T022-T027f (US3 List + tests)
Developer D: T028-T032e (US4 Remove + tests)
```

**Phase 7 (Config Merge)**: T033-T035c can run in parallel with user stories

**Phase 8 (Polish)**: T036-T036c, T037-T037c can run in parallel (different test files)

---

## Parallel Example: Phase 1 Setup

```bash
# Launch all setup tasks together:
Task: "Create RegistryServer models in src/holodeck/models/registry.py"
Task: "Add mcp_servers field to GlobalConfig in src/holodeck/models/config.py"
Task: "Add registry_name field to MCPTool in src/holodeck/models/tool.py"
Task: "Create registry exceptions in src/holodeck/lib/errors.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 + 2 Only)

1. Complete Phase 1: Setup (T001-T003)
2. Complete Phase 2: Foundational (T004-T007)
3. Complete Phase 3: User Story 1 - Search (T008-T013f, including tests)
4. Complete Phase 4: User Story 2 - Add (T014-T021g, including tests)
5. **STOP and VALIDATE**: Run all tests, verify search and add workflows
6. Deploy/demo if ready - users can discover and add servers

### Incremental Delivery

1. Complete Setup + Foundational -> Foundation ready
2. Add User Story 1 (Search) -> Users can discover servers (Partial MVP)
3. Add User Story 2 (Add) -> Users can add servers (Full MVP!)
4. Add User Story 3 (List) -> Users can view installed servers
5. Add User Story 4 (Remove) -> Users can clean up servers
6. Add Phase 7 (Merge) -> Global servers integrate at runtime
7. Each story adds value without breaking previous stories

### File Change Summary

| File | Action | Stories |
|------|--------|---------|
| src/holodeck/models/registry.py | NEW | Setup |
| src/holodeck/models/config.py | MODIFY | Setup |
| src/holodeck/models/tool.py | MODIFY | Setup (add registry_name field) |
| src/holodeck/lib/errors.py | MODIFY | Setup |
| src/holodeck/services/mcp_registry.py | NEW | Foundational, US1 |
| src/holodeck/cli/commands/mcp.py | NEW | US1, US2, US3, US4 |
| src/holodeck/cli/main.py | MODIFY | Foundational |
| src/holodeck/config/loader.py | MODIFY | US2, US3, US4, Merge |

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- Avoid: vague tasks, same file conflicts, cross-story dependencies that break independence
- All CLI commands use Click framework following existing patterns in src/holodeck/cli/commands/
- Registry API uses 5-second timeout with fail-fast behavior (no retries)
- YAML modifications use PyYAML (comments may not be preserved - documented behavior)
