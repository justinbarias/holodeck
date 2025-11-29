# Tasks: Interactive Init Wizard

**Input**: Design documents from `/specs/011-interactive-init-wizard/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/

**Tests**: Tests ARE included as plan.md indicates 80%+ coverage requirement.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`
- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions
- **Single project**: `src/holodeck/`, `tests/` at repository root (per CLAUDE.md)

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Add new dependency and create base project structure for wizard feature

- [ ] T001 Add `inquirerpy>=0.3.4,<0.4.0` dependency to pyproject.toml
- [ ] T002 [P] Create test fixture file for MCP registry mock response in tests/fixtures/mcp_registry_response.json
- [ ] T003 [P] Create empty module files: src/holodeck/models/wizard_config.py, src/holodeck/lib/mcp_registry.py, src/holodeck/cli/utils/wizard.py

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core models and MCP registry client that ALL user stories depend on

**CRITICAL**: No user story work can begin until this phase is complete

### Models (Shared by All Stories)

- [ ] T004 [P] Create WizardStep enum, WizardState model, and WizardResult model in src/holodeck/models/wizard_config.py per data-model.md
- [ ] T005 [P] Create LLMProviderChoice model with LLM_PROVIDER_CHOICES list in src/holodeck/models/wizard_config.py
- [ ] T006 [P] Create VectorStoreChoice model with VECTOR_STORE_CHOICES list in src/holodeck/models/wizard_config.py
- [ ] T007 Extend ProjectInitInput model with llm_provider, vector_store, mcp_servers fields in src/holodeck/models/project_config.py

### MCP Registry Client

- [ ] T008 [P] Create MCPRegistryError exception hierarchy (MCPRegistryError, MCPRegistryNetworkError, MCPRegistryTimeoutError, MCPRegistryResponseError) in src/holodeck/lib/mcp_registry.py
- [ ] T009 [P] Create MCPPackage and MCPServerInfo models in src/holodeck/lib/mcp_registry.py per data-model.md
- [ ] T010 [P] Create MCPRegistryMetadata and MCPRegistryResponse models in src/holodeck/lib/mcp_registry.py
- [ ] T011 [P] Create MCPServerChoice model and DEFAULT_MCP_SERVERS constant in src/holodeck/lib/mcp_registry.py
- [ ] T012 Implement MCPRegistryClient class with __init__, list_servers, get_server_choices methods in src/holodeck/lib/mcp_registry.py per contracts/mcp-registry-client.md

### Unit Tests for Foundation

- [ ] T013 [P] Create unit tests for WizardState, WizardResult, LLMProviderChoice, VectorStoreChoice in tests/unit/test_wizard_config.py
- [ ] T014 [P] Create unit tests for MCPRegistryClient with mocked requests in tests/unit/test_mcp_registry.py

**Checkpoint**: Foundation ready - MCP registry client and all models tested. User story implementation can now begin.

---

## Phase 3: User Story 1 - Quick Start with Defaults (Priority: P1) MVP

**Goal**: User can run `holodeck init` and press Enter at each prompt to create a project with all defaults (Ollama, ChromaDB, 3 default MCP servers)

**Independent Test**: Run `holodeck init test-project`, press Enter at all prompts, verify agent.yaml contains defaults

### Implementation for User Story 1

- [ ] T015 Create is_interactive() function in src/holodeck/cli/utils/wizard.py that checks sys.stdin.isatty() and sys.stdout.isatty()
- [ ] T016 [P] [US1] Create _prompt_llm_provider() internal function using InquirerPy select in src/holodeck/cli/utils/wizard.py
- [ ] T017 [P] [US1] Create _prompt_vectorstore() internal function using InquirerPy select in src/holodeck/cli/utils/wizard.py
- [ ] T018 [P] [US1] Create _prompt_mcp_servers() internal function using InquirerPy checkbox in src/holodeck/cli/utils/wizard.py
- [ ] T019 [US1] Create _fetch_mcp_servers() function that uses MCPRegistryClient.get_server_choices() in src/holodeck/cli/utils/wizard.py
- [ ] T020 [US1] Implement run_wizard() public function orchestrating all prompts in src/holodeck/cli/utils/wizard.py per contracts/wizard-module.md
- [ ] T021 [US1] Create WizardCancelledError exception in src/holodeck/cli/utils/wizard.py
- [ ] T022 [US1] Update ProjectInitializer.initialize() to use llm_provider, vector_store, mcp_servers from ProjectInitInput in src/holodeck/cli/utils/project_init.py
- [ ] T023 [US1] Update agent.yaml.j2 template to include wizard selection variables in src/holodeck/templates/conversational/agent.yaml.j2
- [ ] T024 [US1] Update init command to call run_wizard() when interactive, pass result to ProjectInitInput in src/holodeck/cli/commands/init.py

### Unit Tests for User Story 1

- [ ] T025 [P] [US1] Create unit tests for is_interactive(), _prompt_llm_provider, _prompt_vectorstore, _prompt_mcp_servers with mocked InquirerPy in tests/unit/test_wizard.py
- [ ] T026 [P] [US1] Create unit test for run_wizard() with all prompts mocked in tests/unit/test_wizard.py

**Checkpoint**: User Story 1 complete - users can run wizard with defaults and get a working project.

---

## Phase 4: User Story 2 - Custom LLM Provider Selection (Priority: P1)

**Goal**: User can select OpenAI, Azure OpenAI, or Anthropic and get provider-specific config stubs in generated files

**Independent Test**: Run `holodeck init test-openai`, select OpenAI, verify agent.yaml has OpenAI settings and OPENAI_API_KEY reference

### Implementation for User Story 2

- [ ] T027 [US2] Add provider-specific configuration generation logic to ProjectInitializer for OpenAI (api_key_env_var, model defaults) in src/holodeck/cli/utils/project_init.py
- [ ] T028 [US2] Add provider-specific configuration generation logic to ProjectInitializer for Azure OpenAI (endpoint placeholder, deployment name) in src/holodeck/cli/utils/project_init.py
- [ ] T029 [US2] Add provider-specific configuration generation logic to ProjectInitializer for Anthropic (api_key_env_var, model defaults) in src/holodeck/cli/utils/project_init.py
- [ ] T030 [US2] Update agent.yaml.j2 template with conditional sections for each LLM provider in src/holodeck/templates/conversational/agent.yaml.j2
- [ ] T031 [US2] Update .env.example template with provider-specific environment variable stubs in src/holodeck/templates/conversational/.env.example.j2

### Unit Tests for User Story 2

- [ ] T032 [P] [US2] Create unit tests for OpenAI provider config generation in tests/unit/test_project_init.py
- [ ] T033 [P] [US2] Create unit tests for Azure OpenAI provider config generation in tests/unit/test_project_init.py
- [ ] T034 [P] [US2] Create unit tests for Anthropic provider config generation in tests/unit/test_project_init.py

**Checkpoint**: User Story 2 complete - users can select any LLM provider and get correct configuration.

---

## Phase 5: User Story 3 - Custom Vector Store Selection (Priority: P2)

**Goal**: User can select Redis or In-Memory and get appropriate config stubs, including warning for ephemeral storage

**Independent Test**: Run `holodeck init test-redis`, select Redis, verify agent.yaml has Redis connection settings

### Implementation for User Story 3

- [ ] T035 [US3] Add vector store-specific configuration generation for Redis (connection_string env var, redis provider) in src/holodeck/cli/utils/project_init.py
- [ ] T036 [US3] Add vector store-specific configuration generation for In-Memory (ephemeral warning comment) in src/holodeck/cli/utils/project_init.py
- [ ] T037 [US3] Update agent.yaml.j2 template with conditional sections for each vector store in src/holodeck/templates/conversational/agent.yaml.j2
- [ ] T038 [US3] Add warning display when In-Memory is selected in _prompt_vectorstore() in src/holodeck/cli/utils/wizard.py

### Unit Tests for User Story 3

- [ ] T039 [P] [US3] Create unit tests for Redis vectorstore config generation in tests/unit/test_project_init.py
- [ ] T040 [P] [US3] Create unit tests for In-Memory vectorstore config generation in tests/unit/test_project_init.py
- [ ] T041 [P] [US3] Create unit test for In-Memory warning display in tests/unit/test_wizard.py

**Checkpoint**: User Story 3 complete - users can select any vector store and get correct configuration.

---

## Phase 6: User Story 4 - MCP Server Selection from Registry (Priority: P2)

**Goal**: User sees list of MCP servers from registry with descriptions, can multi-select, defaults are pre-selected

**Independent Test**: Run `holodeck init test-mcp`, modify default MCP selection (add GitHub, remove memory), verify agent.yaml reflects changes

### Implementation for User Story 4

- [ ] T042 [US4] Add MCP server configuration generation to ProjectInitializer for selected servers in src/holodeck/cli/utils/project_init.py
- [ ] T043 [US4] Update agent.yaml.j2 template with MCP tools section using selected servers in src/holodeck/templates/conversational/agent.yaml.j2
- [ ] T044 [US4] Generate MCP tool configuration stubs for each selected server (command, args for npm packages) in src/holodeck/cli/utils/project_init.py

### Unit Tests for User Story 4

- [ ] T045 [P] [US4] Create unit tests for MCP server config generation in tests/unit/test_project_init.py
- [ ] T046 [P] [US4] Create integration test for MCP registry fetch with mock server in tests/integration/test_init_wizard.py

**Checkpoint**: User Story 4 complete - users can select MCP servers and get proper tool configuration.

---

## Phase 7: User Story 5 - Non-Interactive Mode (Priority: P3)

**Goal**: User can run init with --llm, --vectorstore, --mcp, --non-interactive flags for CI/CD usage

**Independent Test**: Run `holodeck init test-ci --llm openai --vectorstore redis --mcp filesystem,github --non-interactive`, verify no prompts and correct config

### Implementation for User Story 5

- [ ] T047 [US5] Add --llm option with click.Choice validator to init command in src/holodeck/cli/commands/init.py
- [ ] T048 [US5] Add --vectorstore option with click.Choice validator to init command in src/holodeck/cli/commands/init.py
- [ ] T049 [US5] Add --mcp option (comma-separated string) to init command in src/holodeck/cli/commands/init.py
- [ ] T050 [US5] Add --non-interactive flag to init command in src/holodeck/cli/commands/init.py
- [ ] T051 [US5] Implement _parse_mcp_arg() helper to parse comma-separated MCP servers in src/holodeck/cli/commands/init.py
- [ ] T052 [US5] Add logic to skip wizard when non-interactive or flags provided in src/holodeck/cli/commands/init.py
- [ ] T053 [US5] Add validation error messages for invalid --llm, --vectorstore values in src/holodeck/cli/commands/init.py
- [ ] T054 [US5] Add warning messages for invalid MCP server names in --mcp (skip invalid, continue) in src/holodeck/cli/commands/init.py
- [ ] T055 [US5] Add TTY detection fallback: use defaults when not interactive and no flags in src/holodeck/cli/commands/init.py

### Unit Tests for User Story 5

- [ ] T056 [P] [US5] Create unit tests for _parse_mcp_arg() helper in tests/unit/test_init_command.py
- [ ] T057 [P] [US5] Create integration test for non-interactive mode with all flags in tests/integration/test_init_wizard.py
- [ ] T058 [P] [US5] Create integration test for invalid flag error messages in tests/integration/test_init_wizard.py

**Checkpoint**: User Story 5 complete - full non-interactive mode support for CI/CD.

---

## Phase 8: Edge Cases & Error Handling

**Purpose**: Handle edge cases defined in spec.md

- [ ] T059 [P] Implement Ctrl+C cancellation handling with clean exit (no partial files) in src/holodeck/cli/commands/init.py
- [ ] T060 [P] Implement MCP registry network error handling with clear message per FR-014 in src/holodeck/cli/commands/init.py
- [ ] T061 [P] Implement TTY detection fallback to non-interactive with defaults per FR-013 in src/holodeck/cli/commands/init.py
- [ ] T062 [P] Update output format to include LLM Provider, Vector Store, MCP Servers in success message in src/holodeck/cli/commands/init.py
- [ ] T063 Create integration test for Ctrl+C cancellation (verify no partial files) in tests/integration/test_init_wizard.py
- [ ] T064 Create integration test for MCP registry network error in tests/integration/test_init_wizard.py

---

## Phase 9: Polish & Cross-Cutting Concerns

**Purpose**: Final cleanup and template updates for other project templates

- [ ] T065 [P] Update research template agent.yaml.j2 with wizard selection variables in src/holodeck/templates/research/agent.yaml.j2
- [ ] T066 [P] Update customer-support template agent.yaml.j2 with wizard selection variables in src/holodeck/templates/customer-support/agent.yaml.j2
- [ ] T067 [P] Update .env.example templates for research and customer-support templates
- [ ] T068 Run `make format` to format all new code
- [ ] T069 Run `make lint` and fix any linting issues
- [ ] T070 Run `make type-check` and fix any type errors
- [ ] T071 Run `make security` and fix any security issues
- [ ] T072 Run `make test-coverage` and ensure 80%+ coverage on new files
- [ ] T073 Validate quickstart.md scenarios work end-to-end

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phases 3-7)**: All depend on Foundational phase completion
  - US1 (Phase 3): Foundation only
  - US2 (Phase 4): Foundation only (parallel with US1)
  - US3 (Phase 5): Foundation only (parallel with US1, US2)
  - US4 (Phase 6): Foundation only (parallel with US1, US2, US3)
  - US5 (Phase 7): Depends on US1 (needs run_wizard to exist)
- **Edge Cases (Phase 8)**: Depends on US1, US5 complete
- **Polish (Phase 9)**: Depends on all user stories complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Phase 2 - No dependencies on other stories
- **User Story 2 (P1)**: Can start after Phase 2 - No dependencies on other stories
- **User Story 3 (P2)**: Can start after Phase 2 - No dependencies on other stories
- **User Story 4 (P2)**: Can start after Phase 2 - No dependencies on other stories
- **User Story 5 (P3)**: Depends on US1 (run_wizard function must exist)

### Within Each User Story

- Models before services
- Services before endpoints
- Core implementation before integration
- Tests can be written before or with implementation
- Story complete before moving to next priority

### Parallel Opportunities

- T002, T003 can run in parallel (Setup phase)
- T004, T005, T006 can run in parallel (Foundation models)
- T008, T009, T010, T011 can run in parallel (MCP registry models)
- T013, T014 can run in parallel (Foundation tests)
- T016, T017, T018 can run in parallel (US1 prompt functions)
- T025, T026 can run in parallel (US1 unit tests)
- T032, T033, T034 can run in parallel (US2 tests)
- T039, T040, T041 can run in parallel (US3 tests)
- T045, T046 can run in parallel (US4 tests)
- T056, T057, T058 can run in parallel (US5 tests)
- US1, US2, US3, US4 can be developed in parallel (different files)
- T065, T066, T067 can run in parallel (template updates)

---

## Parallel Example: User Story 1

```bash
# Launch all prompt functions together (different functions, same file but no conflicts):
Task: "_prompt_llm_provider() in src/holodeck/cli/utils/wizard.py"
Task: "_prompt_vectorstore() in src/holodeck/cli/utils/wizard.py"
Task: "_prompt_mcp_servers() in src/holodeck/cli/utils/wizard.py"

# Then implement orchestrating function:
Task: "run_wizard() in src/holodeck/cli/utils/wizard.py"

# Launch all unit tests together:
Task: "tests/unit/test_wizard.py (all test functions)"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (T001-T003)
2. Complete Phase 2: Foundational (T004-T014)
3. Complete Phase 3: User Story 1 (T015-T026)
4. **STOP and VALIDATE**: Run `holodeck init test-mvp` and press Enter at all prompts
5. Deploy/demo if ready - users can create projects with defaults!

### Incremental Delivery

1. Complete Setup + Foundational → Foundation ready
2. Add User Story 1 → Test independently → MVP ready!
3. Add User Story 2 → Test independently → Custom LLM providers work
4. Add User Story 3 → Test independently → Custom vector stores work
5. Add User Story 4 → Test independently → MCP server selection works
6. Add User Story 5 → Test independently → CI/CD support ready
7. Complete Edge Cases + Polish → Production ready

### Parallel Team Strategy

With multiple developers:

1. Team completes Setup + Foundational together
2. Once Foundational is done:
   - Developer A: User Story 1 + User Story 5 (dependent)
   - Developer B: User Story 2
   - Developer C: User Story 3 + User Story 4
3. Stories complete and integrate independently

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- Run code quality commands (`make format lint type-check security`) after each phase
