# Tasks: Ollama Endpoint Support

**Input**: Design documents from `/specs/009-ollama-endpoint-support/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/, quickstart.md

**Tests**: Unit tests included (mocked); integration tests optional (require Ollama installation)

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `- [ ] [ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3, US4)
- Include exact file paths in descriptions

## Path Conventions

- Single project structure at repository root
- Source code: `src/holodeck/`
- Tests: `tests/unit/`, `tests/integration/`

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and infrastructure updates

- [x] T001 Review existing LLM provider implementation in src/holodeck/models/llm.py (validate OLLAMA enum exists at line 18)
- [x] T002 Review existing endpoint validation in src/holodeck/models/llm.py (validate check_endpoint_required exists at line 74-80)
- [x] T003 Review existing AgentFactory implementation in src/holodeck/lib/test_runner/agent_factory.py (understand \_create_kernel pattern)

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [x] T004 Add Ollama defaults to src/holodeck/config/defaults.py (OLLAMA_DEFAULTS with endpoint, temperature, max_tokens, top_p, api_key)
- [x] T005 [P] Add OllamaConnectionError exception class to src/holodeck/lib/errors.py (inherits from AgentFactoryError)
- [x] T006 [P] Add OllamaModelNotFoundError exception class to src/holodeck/lib/errors.py (inherits from AgentFactoryError)
- [x] T007 Create test fixtures directory tests/fixtures/ollama/
- [x] T008 [P] Create sample Ollama agent config tests/fixtures/ollama/agent_ollama_local.yaml (local endpoint)
- [x] T009 [P] Create sample Ollama agent config tests/fixtures/ollama/agent_ollama_remote.yaml (remote endpoint with env vars)

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Configure Agent with Local Ollama Model (Priority: P1) 🎯 MVP

**Goal**: Enable users to develop and test AI agents using locally-hosted Ollama models

**Independent Test**: Create agent.yaml with Ollama configuration (localhost:11434), verify agent loads without errors, and deliver working chat interactions

### Unit Tests for User Story 1

**NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [x] T010 [P] [US1] Unit test for Ollama provider validation in tests/unit/models/test_llm_ollama.py (test endpoint required validation)
- [x] T011 [P] [US1] Unit test for Ollama parameter validation in tests/unit/models/test_llm_ollama.py (test temperature, max_tokens, top_p ranges)
- [x] T012 [P] [US1] Unit test for Ollama config with defaults in tests/unit/models/test_llm_ollama.py (test default values applied)
- [x] T013 [P] [US1] Unit test for Ollama config with environment variables in tests/unit/models/test_llm_ollama.py (test env var substitution)

### Implementation for User Story 1

- [x] T014 [US1] Add Ollama case to AgentFactory.\_create_kernel() in src/holodeck/lib/test_runner/agent_factory.py (import OllamaChatCompletion from semantic_kernel.connectors.ai.ollama)
- [x] T015 [US1] Implement Ollama service initialization in AgentFactory.\_create_kernel() (create OllamaChatCompletion with ai_model_id, url, api_key)
- [x] T016 [US1] Add error handling for Ollama connection failures in AgentFactory.\_create_kernel() (catch ConnectionError, raise OllamaConnectionError)
- [x] T017 [US1] Add error handling for Ollama model not found in AgentFactory.\_create_kernel() (detect model not found error, raise OllamaModelNotFoundError)
- [x] T018 [P] [US1] Unit test for AgentFactory Ollama kernel creation in tests/unit/lib/test_agent_factory_ollama.py (test \_create_kernel with mocked OllamaChatCompletion)
- [x] T019 [P] [US1] Unit test for Ollama connection error handling in tests/unit/lib/test_agent_factory_ollama.py (test OllamaConnectionError raised when endpoint unreachable)
- [x] T020 [P] [US1] Unit test for Ollama model not found error handling in tests/unit/lib/test_agent_factory_ollama.py (test OllamaModelNotFoundError raised when model missing)

**Checkpoint**: At this point, User Story 1 should be fully functional - agents can load with Ollama configuration and basic chat works

---

## Phase 4: User Story 2 - Test Agent with Ollama Models (Priority: P2)

**Goal**: Enable users to run test cases against agents using Ollama models to validate behavior and response quality

**Independent Test**: Define test cases in agent.yaml, run `holodeck test`, verify tests execute against Ollama endpoint and produce pass/fail results

### Integration Tests for User Story 2

- [x] T021 [P] [US2] Integration test for Ollama config loading in tests/integration/test_ollama_config_loading.py (load agent_ollama_local.yaml fixture)
- [x] T022 [P] [US2] Integration test for Ollama agent initialization in tests/integration/test_ollama_config_loading.py (test AgentFactory initializes with Ollama config)
- [x] T023 [P] [US2] Integration test for Ollama with test cases in tests/integration/test_ollama_test_execution.py (requires local Ollama - mark with @pytest.mark.integration @pytest.mark.skip_if_no_ollama)

### Implementation for User Story 2

- [x] T024 [US2] Verify test runner supports Ollama agents in src/holodeck/lib/test_runner/executor.py (review existing implementation, confirm no changes needed)
- [x] T025 [US2] Verify evaluation framework supports Ollama models in src/holodeck/lib/test_runner/executor.py (confirm evaluations work with Ollama provider)
- [x] T026 [US2] Add documentation for test execution with Ollama in quickstart.md (if not already documented)

**Checkpoint**: At this point, User Stories 1 AND 2 should both work independently - agents load, chat works, and tests execute

---

## Phase 5: User Story 3 - Switch Between Multiple Ollama Models (Priority: P3)

**Goal**: Enable users to compare agent performance across different Ollama models by changing model name in configuration

**Independent Test**: Create multiple agent configs with different Ollama model names, run chat/test with each, verify each uses its specified model

### Unit Tests for User Story 3

- [ ] T027 [P] [US3] Unit test for switching Ollama models in tests/unit/models/test_llm_ollama.py (test different model names: llama3, phi3, mistral, codellama, gemma)
- [ ] T028 [P] [US3] Create additional test fixtures in tests/fixtures/ollama/ for multiple models (agent_ollama_llama3.yaml, agent_ollama_phi3.yaml, agent_ollama_mistral.yaml)

### Implementation for User Story 3

- [ ] T029 [US3] Verify model name is correctly passed to OllamaChatCompletion in AgentFactory.\_create_kernel() (confirm ai_model_id parameter uses config.name)
- [ ] T030 [US3] Add example configurations for multiple Ollama models in quickstart.md (add examples for llama3, phi3, mistral, codellama, gemma)
- [ ] T031 [P] [US3] Integration test for model switching in tests/integration/test_ollama_model_switching.py (requires local Ollama with multiple models - mark with @pytest.mark.integration @pytest.mark.skip_if_no_ollama)

**Checkpoint**: All three user stories should now be independently functional - configuration loading, chat, tests, and model switching all work

---

## Phase 6: User Story 4 - Remote Ollama Server Configuration (Priority: P3)

**Goal**: Enable users to connect to Ollama servers running on remote machines or network endpoints

**Independent Test**: Configure agent.yaml with remote Ollama URL, ensure network connectivity, verify chat/test commands successfully connect to remote endpoint

### Unit Tests for User Story 4

- [ ] T032 [P] [US4] Unit test for remote Ollama endpoint validation in tests/unit/models/test_llm_ollama.py (test remote URL formats: http://192.168.1.100:11434)
- [ ] T033 [P] [US4] Unit test for Ollama with API key authentication in tests/unit/models/test_llm_ollama.py (test api_key parameter passed correctly)
- [ ] T034 [P] [US4] Unit test for environment variable resolution in tests/unit/config/test_config_loader_ollama.py (test OLLAMA_ENDPOINT and OLLAMA_API_KEY substitution)

### Implementation for User Story 4

- [ ] T035 [US4] Verify API key is correctly passed to OllamaChatCompletion in AgentFactory.\_create_kernel() (confirm api_key parameter uses config.api_key)
- [ ] T036 [US4] Add error handling for authentication failures in AgentFactory.\_create_kernel() (catch auth errors, provide clear message referencing OLLAMA_API_KEY)
- [ ] T037 [P] [US4] Unit test for Ollama authentication error handling in tests/unit/lib/test_agent_factory_ollama.py (test auth failure error message)
- [ ] T038 [US4] Add remote endpoint configuration examples in quickstart.md (document environment variable usage for OLLAMA_ENDPOINT and OLLAMA_API_KEY)
- [ ] T039 [P] [US4] Integration test for remote Ollama with authentication in tests/integration/test_ollama_remote_auth.py (requires remote Ollama setup - mark with @pytest.mark.integration @pytest.mark.skip_if_no_ollama)

**Checkpoint**: All four user stories should now be independently functional - local config, testing, model switching, and remote endpoints all work

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories and final validation

- [ ] T040 [P] Add URL format validation to LLMProvider model in src/holodeck/models/llm.py (validate endpoint starts with http:// or https:// for Ollama provider)
- [ ] T041 [P] Add logging for Ollama connection attempts in AgentFactory.\_create_kernel() (log endpoint, model name, connection status)
- [ ] T042 [P] Add logging for Ollama response times in AgentFactory.invoke() (log latency metrics for observability)
- [ ] T043 Update README.md with Ollama configuration examples (add Ollama to list of supported providers)
- [ ] T044 Validate quickstart.md examples are complete and accurate (review all configuration examples)
- [ ] T045 Run code quality checks: make format && make format-check && make lint && make lint-fix && make type-check && make security
- [ ] T046 Run test suite: make test-unit (ensure 80% coverage minimum)
- [ ] T047 Verify integration tests are properly marked and skippable (check @pytest.mark.integration and @pytest.mark.skip_if_no_ollama decorators)

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Story 1 (Phase 3)**: Depends on Foundational phase completion - No dependencies on other stories
- **User Story 2 (Phase 4)**: Depends on User Story 1 completion (requires working Ollama agent)
- **User Story 3 (Phase 5)**: Depends on User Story 1 completion (builds on basic Ollama support)
- **User Story 4 (Phase 6)**: Depends on User Story 1 completion (extends basic Ollama with remote endpoints)
- **Polish (Phase 7)**: Depends on all desired user stories being complete

### User Story Dependencies

- **User Story 1 (P1) - MVP**: Core functionality - MUST be implemented first
- **User Story 2 (P2)**: Builds on US1 - requires working Ollama agent
- **User Story 3 (P3)**: Independent of US2 - can be implemented in parallel with US2
- **User Story 4 (P3)**: Independent of US2 and US3 - can be implemented in parallel

### Within Each User Story

- Tests MUST be written and FAIL before implementation
- Error handling classes (Phase 2) before agent factory modifications
- Agent factory modifications before integration tests
- Unit tests can run in parallel (marked with [P])
- Integration tests require local/remote Ollama setup

### Parallel Opportunities

- **Setup (Phase 1)**: All review tasks (T001-T003) can run in parallel
- **Foundational (Phase 2)**: T005-T006 (error classes) can run in parallel; T008-T009 (fixtures) can run in parallel
- **User Story 1**: All unit tests (T010-T013, T018-T020) can run in parallel
- **User Story 2**: Integration tests (T021-T023) can run in parallel
- **User Story 3**: Unit tests (T027-T028) and T031 can run in parallel
- **User Story 4**: Unit tests (T032-T034, T037) can run in parallel
- **After US1 completes**: US2, US3, and US4 can be worked on in parallel by different team members
- **Polish (Phase 7)**: Tasks T040-T042 (code improvements) can run in parallel

---

## Parallel Example: User Story 1

```bash
# Launch all unit tests for User Story 1 together:
# - Unit test for Ollama provider validation in tests/unit/models/test_llm_ollama.py
# - Unit test for Ollama parameter validation in tests/unit/models/test_llm_ollama.py
# - Unit test for Ollama config with defaults in tests/unit/models/test_llm_ollama.py
# - Unit test for Ollama config with environment variables in tests/unit/models/test_llm_ollama.py

# After implementation:
# - Unit test for AgentFactory Ollama kernel creation in tests/unit/lib/test_agent_factory_ollama.py
# - Unit test for Ollama connection error handling in tests/unit/lib/test_agent_factory_ollama.py
# - Unit test for Ollama model not found error handling in tests/unit/lib/test_agent_factory_ollama.py
```

---

## Parallel Example: After Foundational Phase

```bash
# Once Foundational (Phase 2) is complete, these user stories can proceed in parallel:
# Team Member A: User Story 1 (T010-T020) - Core Ollama support
# Team Member B: Prepare User Story 2 fixtures and tests (after US1 core is done)
# Team Member C: Prepare User Story 3 fixtures (after US1 core is done)

# After US1 is complete:
# Team Member A: User Story 2 (T021-T026) - Test execution
# Team Member B: User Story 3 (T027-T031) - Model switching
# Team Member C: User Story 4 (T032-T039) - Remote endpoints
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (T001-T003)
2. Complete Phase 2: Foundational (T004-T009) - CRITICAL, blocks all stories
3. Complete Phase 3: User Story 1 (T010-T020)
4. **STOP and VALIDATE**: Test User Story 1 independently
   - Create agent.yaml with Ollama config
   - Run `holodeck chat` and verify responses
   - Verify error messages are clear
5. Run Phase 7 quality checks (T045-T047)
6. Deploy/demo if ready

### Incremental Delivery

1. Complete Setup + Foundational → Foundation ready
2. Add User Story 1 → Test independently → **MVP COMPLETE!**
3. Add User Story 2 → Test execution works → Deploy/Demo
4. Add User Story 3 → Model switching works → Deploy/Demo (optional)
5. Add User Story 4 → Remote endpoints work → Deploy/Demo (optional)
6. Complete Polish → Production ready

### Parallel Team Strategy

With multiple developers:

1. Team completes Setup + Foundational together (T001-T009)
2. Once Foundational is done:
   - Developer A: Complete User Story 1 (T010-T020) - BLOCKING for others
3. Once User Story 1 is complete:
   - Developer A: User Story 2 (T021-T026)
   - Developer B: User Story 3 (T027-T031)
   - Developer C: User Story 4 (T032-T039)
4. Developers integrate and test independently
5. Final polish together (T040-T047)

---

## Notes

- [P] tasks = different files, no dependencies within the story
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Verify tests fail before implementing (TDD approach)
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- Run `make format lint type-check security` after each significant change
- Integration tests require Ollama installation - mark with @pytest.mark.integration
- User Story 1 is the MVP - prioritize completion before other stories
- User Stories 3 and 4 are lower priority and can be deferred if needed

---

## Success Criteria Validation

After completing tasks, verify these success criteria are met:

- **SC-001**: Users can configure an agent and start a chat session in under 2 minutes ✓ (US1)
- **SC-002**: Chat interactions respond within Ollama endpoint latency + <50ms overhead ✓ (US1)
- **SC-003**: Test cases execute successfully with 100% success rate when Ollama is running ✓ (US2)
- **SC-004**: Error messages are clear enough for 90% of users to self-resolve ✓ (US1, error handling)
- **SC-005**: Users can switch models by changing one configuration value ✓ (US3)
- **SC-006**: System supports 5+ Ollama models (llama3, phi3, mistral, codellama, gemma) ✓ (US3)
- **SC-007**: Configuration validation catches 100% of invalid formats and missing fields ✓ (US1, validation tests)
- **SC-008**: Users can run evaluations using Ollama models identically to cloud providers ✓ (US2)

---

## Total Task Count

- **Setup**: 3 tasks
- **Foundational**: 6 tasks
- **User Story 1**: 11 tasks (4 unit tests, 7 implementation)
- **User Story 2**: 6 tasks (3 integration tests, 3 implementation)
- **User Story 3**: 5 tasks (2 unit tests, 3 implementation)
- **User Story 4**: 8 tasks (3 unit tests, 5 implementation)
- **Polish**: 8 tasks
- **TOTAL**: 47 tasks

---

## Parallel Opportunities Identified

- Phase 1: 3 tasks can run in parallel
- Phase 2: 4 tasks can run in parallel (T005-T006, T008-T009)
- User Story 1: 7 unit tests can run in parallel
- User Story 2: 2 integration tests can run in parallel
- User Story 3: 2 tasks can run in parallel
- User Story 4: 4 unit tests can run in parallel
- Polish: 3 tasks can run in parallel

**Total parallelizable tasks**: 25 out of 47 (53%)

---

## Independent Test Criteria Summary

- **User Story 1**: Create agent.yaml with Ollama config, run `holodeck chat`, verify responses work
- **User Story 2**: Add test cases to agent.yaml, run `holodeck test`, verify pass/fail results
- **User Story 3**: Create multiple configs with different models, verify each uses correct model
- **User Story 4**: Configure remote endpoint, verify successful connection and authentication

---

## Suggested MVP Scope

**Minimum Viable Product**: User Story 1 ONLY

- Phase 1: Setup (T001-T003)
- Phase 2: Foundational (T004-T009)
- Phase 3: User Story 1 (T010-T020)
- Phase 7: Essential polish (T045-T047)

**Total MVP Tasks**: 20 tasks

This delivers the core value proposition: users can configure and use local Ollama models with HoloDeck.
