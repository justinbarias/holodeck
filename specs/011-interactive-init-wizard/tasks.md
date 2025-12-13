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

- [x] T001 Add `inquirerpy>=0.3.4,<0.4.0` dependency to pyproject.toml
- [x] T002 [P] Create empty module files: src/holodeck/models/wizard_config.py, src/holodeck/cli/utils/wizard.py
- [x] T003 [P] Create test fixture file for wizard defaults in tests/fixtures/wizard_defaults.py

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core models that ALL user stories depend on

**CRITICAL**: No user story work can begin until this phase is complete

### Models (Shared by All Stories)

- [x] T004 [P] Create WizardStep enum (AGENT_NAME, TEMPLATE, LLM_PROVIDER, VECTOR_STORE, EVALS, MCP_SERVERS, COMPLETE), WizardState model, and WizardResult model in src/holodeck/models/wizard_config.py per data-model.md
- [x] T005 [P] Create LLMProviderChoice model with LLM_PROVIDER_CHOICES list (ollama default with gpt-oss:20b, openai, azure_openai, anthropic) in src/holodeck/models/wizard_config.py
- [x] T006 [P] Create VectorStoreChoice model with VECTOR_STORE_CHOICES list (chromadb default with http://localhost:8000, ChromaDb, in-memory) in src/holodeck/models/wizard_config.py
- [x] T007 [P] Create EvalChoice model with EVAL_CHOICES list (rag-faithfulness, rag-answer_relevancy default, rag-context_precision, rag-context_recall) in src/holodeck/models/wizard_config.py
- [x] T008 [P] Create MCPServerChoice model with MCP_SERVER_CHOICES list (brave-search, memory, sequentialthinking default, filesystem, github, postgres) in src/holodeck/models/wizard_config.py
- [x] T008a [P] Create TemplateChoice model with get_template_choices() helper that dynamically loads templates from manifest files in src/holodeck/models/wizard_config.py
- [x] T008b [P] Add get_available_templates() method to TemplateRenderer that returns template metadata (value, display_name, description) in src/holodeck/lib/template_engine.py
- [x] T009 Extend ProjectInitInput model with agent_name, llm_provider, vector_store, evals, mcp_servers fields in src/holodeck/models/project_config.py

### Unit Tests for Foundation

- [x] T010 [P] Create unit tests for WizardState, WizardResult, LLMProviderChoice, VectorStoreChoice, EvalChoice, MCPServerChoice in tests/unit/test_wizard_config.py

**Checkpoint**: Foundation ready - all models tested. User story implementation can now begin.

---

## Phase 3: User Story 1 - Quick Start with Defaults (Priority: P1) MVP

**Goal**: User can run `holodeck init`, enter agent name, and press Enter at each prompt to create a project with all defaults (Ollama gpt-oss:20b, ChromaDB http://localhost:8000, default evals, default MCP servers)

**Independent Test**: Run `holodeck init`, enter agent name, press Enter at all prompts, verify agent.yaml contains defaults

### Implementation for User Story 1

- [x] T011 Create is_interactive() function in src/holodeck/cli/utils/wizard.py that checks sys.stdin.isatty() and sys.stdout.isatty()
- [x] T012 [P] [US1] Create \_prompt_agent_name() internal function using InquirerPy text with validation in src/holodeck/cli/utils/wizard.py
- [x] T012a [P] [US1] Create \_prompt_template() internal function using InquirerPy select with dynamic template loading in src/holodeck/cli/utils/wizard.py
- [x] T013 [P] [US1] Create \_prompt_llm_provider() internal function using InquirerPy select in src/holodeck/cli/utils/wizard.py
- [x] T014 [P] [US1] Create \_prompt_vectorstore() internal function using InquirerPy select in src/holodeck/cli/utils/wizard.py
- [x] T015 [P] [US1] Create \_prompt_evals() internal function using InquirerPy checkbox in src/holodeck/cli/utils/wizard.py
- [x] T016 [P] [US1] Create \_prompt_mcp_servers() internal function using InquirerPy checkbox in src/holodeck/cli/utils/wizard.py
- [x] T017 [US1] Implement run_wizard() public function orchestrating all prompts in src/holodeck/cli/utils/wizard.py per contracts/wizard-module.md
- [x] T018 [US1] Create WizardCancelledError exception in src/holodeck/cli/utils/wizard.py
- [x] T019 [US1] Update ProjectInitializer.initialize() to use agent_name, llm_provider, vector_store, evals, mcp_servers from ProjectInitInput in src/holodeck/cli/utils/project_init.py
- [x] T020 [US1] Update agent.yaml.j2 template to include wizard selection variables in src/holodeck/templates/conversational/agent.yaml.j2
- [x] T021 [US1] Update init command to call run_wizard() when interactive, pass result to ProjectInitInput in src/holodeck/cli/commands/init.py

### Unit Tests for User Story 1

- [x] T022 [P] [US1] Create unit tests for is_interactive(), \_prompt_agent_name, \_prompt_llm_provider, \_prompt_vectorstore, \_prompt_evals, \_prompt_mcp_servers with mocked InquirerPy in tests/unit/test_wizard.py
- [x] T023 [P] [US1] Create unit test for run_wizard() with all prompts mocked in tests/unit/test_wizard.py

**Checkpoint**: User Story 1 complete - users can run wizard with defaults and get a working project.

---

## Phase 4: User Story 2 - Custom LLM Provider Selection (Priority: P1)

**Goal**: User can select OpenAI, Azure OpenAI, or Anthropic and get provider-specific config stubs in generated files

**Independent Test**: Run `holodeck init`, select OpenAI, verify agent.yaml has OpenAI settings and OPENAI_API_KEY reference

### Implementation for User Story 2

- [ ] T024 [US2] Add provider-specific configuration generation logic to ProjectInitializer for OpenAI (api_key_env_var, model defaults) in src/holodeck/cli/utils/project_init.py
- [ ] T025 [US2] Add provider-specific configuration generation logic to ProjectInitializer for Azure OpenAI (endpoint placeholder, deployment name) in src/holodeck/cli/utils/project_init.py
- [ ] T026 [US2] Add provider-specific configuration generation logic to ProjectInitializer for Anthropic (api_key_env_var, model defaults) in src/holodeck/cli/utils/project_init.py
- [ ] T027 [US2] Update agent.yaml.j2 template with conditional sections for each LLM provider in src/holodeck/templates/conversational/agent.yaml.j2
- [ ] T028 [US2] Update .env.example template with provider-specific environment variable stubs in src/holodeck/templates/conversational/.env.example.j2

### Unit Tests for User Story 2

- [ ] T029 [P] [US2] Create unit tests for OpenAI provider config generation in tests/unit/test_project_init.py
- [ ] T030 [P] [US2] Create unit tests for Azure OpenAI provider config generation in tests/unit/test_project_init.py
- [ ] T031 [P] [US2] Create unit tests for Anthropic provider config generation in tests/unit/test_project_init.py

**Checkpoint**: User Story 2 complete - users can select any LLM provider and get correct configuration.

---

## Phase 5: User Story 3 - Custom Vector Store Selection (Priority: P2)

**Goal**: User can select ChromaDb or In-Memory and get appropriate config stubs, including warning for ephemeral storage

**Independent Test**: Run `holodeck init`, select ChromaDb, verify agent.yaml has ChromaDb connection settings

### Implementation for User Story 3

- [ ] T032 [US3] Add vector store-specific configuration generation for ChromaDb (connection_string env var, ChromaDb provider) in src/holodeck/cli/utils/project_init.py
- [ ] T033 [US3] Add vector store-specific configuration generation for In-Memory (ephemeral warning comment) in src/holodeck/cli/utils/project_init.py
- [ ] T034 [US3] Update agent.yaml.j2 template with conditional sections for each vector store in src/holodeck/templates/conversational/agent.yaml.j2
- [x] T035 [US3] Add warning display when In-Memory is selected in \_prompt_vectorstore() in src/holodeck/cli/utils/wizard.py

### Unit Tests for User Story 3

- [ ] T036 [P] [US3] Create unit tests for ChromaDb vectorstore config generation in tests/unit/test_project_init.py
- [ ] T037 [P] [US3] Create unit tests for In-Memory vectorstore config generation in tests/unit/test_project_init.py
- [ ] T038 [P] [US3] Create unit test for In-Memory warning display in tests/unit/test_wizard.py

**Checkpoint**: User Story 3 complete - users can select any vector store and get correct configuration.

---

## Phase 6: User Story 4 - Evaluation Metrics Selection (Priority: P2)

**Goal**: User sees list of evaluation metrics, can multi-select, with rag-faithfulness and rag-answer_relevancy pre-selected

**Independent Test**: Run `holodeck init`, modify default eval selection (add rag-context_precision), verify agent.yaml reflects changes

### Implementation for User Story 4

- [ ] T039 [US4] Add eval configuration generation to ProjectInitializer for selected metrics in src/holodeck/cli/utils/project_init.py
- [ ] T040 [US4] Update agent.yaml.j2 template with evaluations section using selected evals in src/holodeck/templates/conversational/agent.yaml.j2
- [ ] T041 [US4] Generate eval configuration stubs for each selected metric in src/holodeck/cli/utils/project_init.py

### Unit Tests for User Story 4

- [ ] T042 [P] [US4] Create unit tests for eval config generation in tests/unit/test_project_init.py
- [ ] T043 [P] [US4] Create unit tests for eval prompt with defaults in tests/unit/test_wizard.py

**Checkpoint**: User Story 4 complete - users can select evals and get proper configuration.

---

## Phase 7: User Story 5 - MCP Server Selection (Priority: P2)

**Goal**: User sees list of MCP servers with descriptions, can multi-select, with brave-search, memory, sequentialthinking pre-selected

**Independent Test**: Run `holodeck init`, modify default MCP selection (add filesystem, remove memory), verify agent.yaml reflects changes

### Implementation for User Story 5

- [ ] T044 [US5] Add MCP server configuration generation to ProjectInitializer for selected servers in src/holodeck/cli/utils/project_init.py
- [ ] T045 [US5] Update agent.yaml.j2 template with MCP tools section using selected servers in src/holodeck/templates/conversational/agent.yaml.j2
- [ ] T046 [US5] Generate MCP tool configuration stubs for each selected server (command, args for npm packages) in src/holodeck/cli/utils/project_init.py

### Unit Tests for User Story 5

- [ ] T047 [P] [US5] Create unit tests for MCP server config generation in tests/unit/test_project_init.py
- [ ] T048 [P] [US5] Create integration test for MCP selection flow in tests/integration/test_init_wizard.py

**Checkpoint**: User Story 5 complete - users can select MCP servers and get proper tool configuration.

---

## Phase 8: User Story 6 - Non-Interactive Mode (Priority: P3)

**Goal**: User can run init with --name, --llm, --vectorstore, --evals, --mcp, --non-interactive flags for CI/CD usage

**Independent Test**: Run `holodeck init --name test-agent --llm openai --vectorstore ChromaDb --evals rag-faithfulness --mcp filesystem,brave-search --non-interactive`, verify no prompts and correct config

### Implementation for User Story 6

- [x] T049 [US6] Add --name option (required for non-interactive) to init command in src/holodeck/cli/commands/init.py
- [x] T050 [US6] Add --llm option with click.Choice validator to init command in src/holodeck/cli/commands/init.py
- [x] T051 [US6] Add --vectorstore option with click.Choice validator to init command in src/holodeck/cli/commands/init.py
- [x] T052 [US6] Add --evals option (comma-separated string) to init command in src/holodeck/cli/commands/init.py
- [x] T053 [US6] Add --mcp option (comma-separated string) to init command in src/holodeck/cli/commands/init.py
- [x] T054 [US6] Add --non-interactive flag to init command in src/holodeck/cli/commands/init.py
- [x] T055 [US6] Implement \_parse_comma_arg() helper to parse comma-separated values in src/holodeck/cli/commands/init.py
- [x] T056 [US6] Add logic to skip wizard when non-interactive or flags provided in src/holodeck/cli/commands/init.py
- [x] T057 [US6] Add validation error messages for invalid flag values in src/holodeck/cli/commands/init.py
- [x] T058 [US6] Add warning messages for invalid evals/MCP server names (skip invalid, continue) in src/holodeck/cli/commands/init.py
- [x] T059 [US6] Add TTY detection fallback: use defaults when not interactive and no flags in src/holodeck/cli/commands/init.py

### Unit Tests for User Story 6

- [ ] T060 [P] [US6] Create unit tests for \_parse_comma_arg() helper in tests/unit/test_init_command.py
- [ ] T061 [P] [US6] Create integration test for non-interactive mode with all flags in tests/integration/test_init_wizard.py
- [ ] T062 [P] [US6] Create integration test for invalid flag error messages in tests/integration/test_init_wizard.py

**Checkpoint**: User Story 6 complete - full non-interactive mode support for CI/CD.

---

## Phase 9: Edge Cases & Error Handling

**Purpose**: Handle edge cases defined in spec.md

- [x] T063 [P] Implement Ctrl+C cancellation handling with clean exit (no partial files) in src/holodeck/cli/commands/init.py
- [x] T064 [P] Implement TTY detection fallback to non-interactive with defaults per FR-015 in src/holodeck/cli/commands/init.py
- [x] T065 [P] Update output format to include Agent Name, LLM Provider, Vector Store, Evals, MCP Servers in success message in src/holodeck/cli/commands/init.py
- [ ] T066 Create integration test for Ctrl+C cancellation (verify no partial files) in tests/integration/test_init_wizard.py

---

## Phase 10: Polish & Cross-Cutting Concerns

**Purpose**: Final cleanup and template updates for other project templates

- [x] T067 [P] Update research template agent.yaml.j2 with wizard selection variables in src/holodeck/templates/research/agent.yaml.j2
- [x] T068 [P] Update customer-support template agent.yaml.j2 with wizard selection variables in src/holodeck/templates/customer-support/agent.yaml.j2
- [ ] T069 [P] Update .env.example templates for research and customer-support templates
- [x] T070 Run `make format` to format all new code
- [x] T071 Run `make lint` and fix any linting issues
- [x] T072 Run `make type-check` and fix any type errors
- [x] T073 Run `make security` and fix any security issues
- [ ] T074 Run `make test-coverage` and ensure 80%+ coverage on new files
- [ ] T075 Validate quickstart.md scenarios work end-to-end

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phases 3-8)**: All depend on Foundational phase completion
  - US1 (Phase 3): Foundation only
  - US2 (Phase 4): Foundation only (parallel with US1)
  - US3 (Phase 5): Foundation only (parallel with US1, US2)
  - US4 (Phase 6): Foundation only (parallel with US1, US2, US3)
  - US5 (Phase 7): Foundation only (parallel with US1, US2, US3, US4)
  - US6 (Phase 8): Depends on US1 (needs run_wizard to exist)
- **Edge Cases (Phase 9)**: Depends on US1, US6 complete
- **Polish (Phase 10)**: Depends on all user stories complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Phase 2 - No dependencies on other stories
- **User Story 2 (P1)**: Can start after Phase 2 - No dependencies on other stories
- **User Story 3 (P2)**: Can start after Phase 2 - No dependencies on other stories
- **User Story 4 (P2)**: Can start after Phase 2 - No dependencies on other stories
- **User Story 5 (P2)**: Can start after Phase 2 - No dependencies on other stories
- **User Story 6 (P3)**: Depends on US1 (run_wizard function must exist)

### Within Each User Story

- Models before services
- Services before endpoints
- Core implementation before integration
- Tests can be written before or with implementation
- Story complete before moving to next priority

### Parallel Opportunities

- T002, T003 can run in parallel (Setup phase)
- T004, T005, T006, T007, T008 can run in parallel (Foundation models)
- T012, T013, T014, T015, T016 can run in parallel (US1 prompt functions)
- T022, T023 can run in parallel (US1 unit tests)
- T029, T030, T031 can run in parallel (US2 tests)
- T036, T037, T038 can run in parallel (US3 tests)
- T042, T043 can run in parallel (US4 tests)
- T047, T048 can run in parallel (US5 tests)
- T060, T061, T062 can run in parallel (US6 tests)
- US1, US2, US3, US4, US5 can be developed in parallel (different files)
- T067, T068, T069 can run in parallel (template updates)

---

## Parallel Example: User Story 1

```bash
# Launch all prompt functions together (different functions, same file but no conflicts):
Task: "_prompt_agent_name() in src/holodeck/cli/utils/wizard.py"
Task: "_prompt_llm_provider() in src/holodeck/cli/utils/wizard.py"
Task: "_prompt_vectorstore() in src/holodeck/cli/utils/wizard.py"
Task: "_prompt_evals() in src/holodeck/cli/utils/wizard.py"
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
2. Complete Phase 2: Foundational (T004-T010)
3. Complete Phase 3: User Story 1 (T011-T023)
4. **STOP and VALIDATE**: Run `holodeck init` and enter agent name, press Enter at all prompts
5. Deploy/demo if ready - users can create projects with defaults!

### Incremental Delivery

1. Complete Setup + Foundational → Foundation ready
2. Add User Story 1 → Test independently → MVP ready!
3. Add User Story 2 → Test independently → Custom LLM providers work
4. Add User Story 3 → Test independently → Custom vector stores work
5. Add User Story 4 → Test independently → Custom evals work
6. Add User Story 5 → Test independently → MCP server selection works
7. Add User Story 6 → Test independently → CI/CD support ready
8. Complete Edge Cases + Polish → Production ready

### Parallel Team Strategy

With multiple developers:

1. Team completes Setup + Foundational together
2. Once Foundational is done:
   - Developer A: User Story 1 + User Story 6 (dependent)
   - Developer B: User Story 2 + User Story 3
   - Developer C: User Story 4 + User Story 5
3. Stories complete and integrate independently

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- Run code quality commands (`make format lint type-check security`) after each phase
