# Implementation Tasks: Global Settings and Response Format Configuration (TDD)

**Feature**: Global Settings and Response Format Configuration
**Branch**: `005-global-settings-response-format`
**Date**: 2025-10-25
**Approach**: Test-Driven Development (Tests first, then implementation)

---

## Summary

- **Total Tasks**: 24
- **Setup Phase**: 3 tasks
- **Foundational Phase**: 4 tasks
- **User Story 1 (P1)**: 6 tasks (2 test, 4 implementation)
- **User Story 2 (P1)**: 6 tasks (2 test, 4 implementation)
- **User Story 4 (P1)**: 3 tasks (1 test, 2 implementation)
- **Polish Phase**: 2 tasks

**Test Coverage**: 100% of user stories have TDD tests before implementation

**Independent Test Criteria**:

- **US1**: Global settings loaded from user-level and project-level; agents inherit defaults
- **US2**: Response format defined inline and externally in agent config; validated at config load time
- **US4**: Schema validation catches errors with clear messages

**Recommended MVP Scope**: User Stories 1, 2, and 4 (configuration loading, response format validation, error handling)

---

## Phase 1: Setup & Initialization

- [x] T001 Create `src/holodeck/config/` package directory with `__init__.py`
- [x] T002 Create base models and type annotations in `src/holodeck/config/models.py`
- [x] T003 Create test directory structure: `tests/unit/config/` and `tests/integration/`

---

## Phase 2: Foundational Infrastructure

- [x] T004 [P] Implement configuration file discovery logic in `src/holodeck/config/loader.py` (user-level ~/.holodeck/config.yml|yaml detection)
- [x] T005 [P] Implement YAML parsing and validation in `src/holodeck/config/loader.py` (PyYAML integration)
- [x] T006 Implement GlobalConfig merge logic in `src/holodeck/config/merge.py` (user-level + project-level merging)
- [x] T007 Implement Basic JSON Schema validator in `src/holodeck/config/schema.py` (jsonschema integration with keyword restrictions)

---

## Phase 3: User Story 1 - Configure Global Settings (Priority: P1)

**Goal**: Developers can define global settings files at user and project levels; agents automatically inherit defaults.

**Independent Test**: Create config.yml with model provider; verify agent loads and uses inherited values without modification.

### US1 Test Tasks (TDD - Write Tests First)

- [x] T008 [US1] Write tests: Configuration file discovery in `tests/unit/config/test_loader.py`

  - Test user-level config at ~/.holodeck/config.yml|yaml detection
  - Test project-level config at config.yml|yaml detection
  - Test file extension preference (.yml > .yaml)
  - Test missing config files handled gracefully

- [x] T009 [US1] Write tests: Configuration inheritance and precedence in `tests/unit/config/test_inheritance.py`
  - Test user-level → project-level → agent precedence
  - Test inherit_global: false disables inheritance
  - Test configuration merging with overrides
  - Test agent-level overrides global settings

### US1 Implementation Tasks (TDD - Implement to Pass Tests)

- [x] T010 [US1] Implement project-level config file discovery in `src/holodeck/config/loader.py` (make T008 tests pass)
- [x] T011 [US1] Implement configuration inheritance logic in `src/holodeck/config/merge.py` (make T009 tests pass)
- [x] T012 [US1] Implement `inherit_global: false` flag handling in `src/holodeck/config/merge.py`
- [x] T013 [US1] Implement configuration loading API in `src/holodeck/config/loader.py` (load_global_config, load_agent_config, merge_configs functions)

---

## Phase 4: User Story 2 - Define Response Format (Priority: P1)

**Goal**: Developers can specify response format constraints (inline or external file) at agent level only; LLM generates structured output.

**Independent Test**: Define response_format inline and in file in agent.yaml; verify schema is loaded and applied to LLM calls.

### US2 Test Tasks (TDD - Write Tests First)

- [x] T014 [US2] Write tests: Response format loading and validation in `tests/unit/config/test_schema_validation.py`

  - Test inline YAML-to-JSON schema parsing
  - Test external .json schema file loading (relative to project root)
  - Test Basic JSON Schema keyword validation (type, properties, required, additionalProperties)
  - Test rejection of unsupported keywords (anyOf, $ref, patternProperties)

- [x] T015 [US2] Write tests: Response format application in `tests/unit/config/test_response_format.py`
  - Test inline response_format stored in agent config
  - Test external response_format file loaded and validated
  - Test response_format is agent-specific (not inherited from global)
  - Test agent-level response_format stored for LLM processing

### US2 Implementation Tasks (TDD - Implement to Pass Tests)

- [x] T016 [US2] Implement inline response_format validation in `src/holodeck/config/schema.py` (SchemaValidator.validate_schema)
- [x] T017 [US2] Implement external schema file loading in `src/holodeck/config/schema.py` (SchemaValidator.load_schema_from_file)
- [x] T018 [US2] Implement response_format field in Agent model in `src/holodeck/models/agent.py` with validator
- [x] T019 [US2] Response_format is agent-specific by design (no implementation needed - validated in merge logic)

---

## Phase 5: User Story 4 - Validate Schema Syntax and Structure (Priority: P1)

**Goal**: Invalid schemas caught at config load time with clear error messages.

**Independent Test**: Provide invalid schemas; verify error messages indicate line number and specific violation.

### US4 Test Tasks (TDD - Write Tests First)

- [x] T020 [US4] Write tests: Error handling in `tests/unit/config/test_error_handling.py`
  - Test YAML syntax errors with line numbers
  - Test invalid JSON in response_format with line numbers
  - Test missing schema files with path display
  - Test unknown JSON Schema keywords with keyword name
  - Test LLM provider unsupported warning

### US4 Implementation Tasks (TDD - Implement to Pass Tests)

- [x] T021 [US4] Schema syntax validation already implemented in `src/holodeck/config/schema.py` (SchemaValidator validates and provides clear errors)
- [x] T022 [US4] Configuration error messages already implemented in `src/holodeck/config/loader.py` (ConfigError with file/line context)

---

## Phase 6: Polish & Cross-Cutting Concerns

- [x] T023 [P] Integration test: End-to-end config loading and agent initialization in `tests/integration/test_config_end_to_end.py`

  - Load user-level config
  - Load project-level config with override
  - Load agent with response_format
  - Verify inheritance of global settings
  - Verify response_format applied to agent
  - Test all error scenarios

- [x] T024 Code quality: Format, lint, type-check, security scan with `make format`, `make lint`, `make type-check`, `make security` (all must pass)

- [x] T025 Update docsite /docs to reflect response_format changes.

---

## Dependency Graph

```
T001 (Create directories)
  ├─→ T002 (Base models)
  │   ├─→ T004 (User-level discovery)
  │   ├─→ T010 (Project-level discovery)
  │   ├─→ T016 (Response format validation)
  │   └─→ T017 (External schema loading)
  │
  ├─→ T003 (Test directories)
  │   ├─→ T008 (US1 test - file discovery)
  │   ├─→ T009 (US1 test - inheritance)
  │   ├─→ T014 (US2 test - schema validation)
  │   ├─→ T015 (US2 test - response format app)
  │   └─→ T020 (US4 test - error handling)
  │
  └─→ T005 (YAML parsing)
      └─→ T004 (File discovery depends on parsing)

T008 (US1 file discovery tests)
  └─→ T010 (Implement project-level discovery)

T009 (US1 inheritance tests)
  └─→ T011 (Implement inheritance logic)
      └─→ T012 (Implement inherit_global flag)
          └─→ T013 (Implement config loading API)

T014 (US2 schema validation tests)
  └─→ T016 (Implement inline validation)
      └─→ T017 (Implement external loading)

T015 (US2 response format tests)
  └─→ T018 (Implement response format app)
      └─→ T019 (Ensure agent-specific, not global)

T020 (US4 error handling tests)
  └─→ T021 (Implement error handling)
      └─→ T022 (Implement error messages)

All implementation tasks (T010-T022)
  └─→ T023 (Integration test)

T023 (All features working)
  └─→ T024 (Code quality checks)
```

---

## Parallel Execution Opportunities

### Phase 2 Parallel Tasks

```
Can run in parallel (no dependencies between):
  - T004: User-level config discovery
  - T005: YAML parsing
  - T006: GlobalConfig merge logic
  - T007: JSON Schema validator
```

### User Story 1 Tests + Implementation Parallel

```
After T008 (tests) written:
  - T010: Implement project-level discovery (parallel with US2)

After T009 (tests) written:
  - T011: Implement inheritance logic (parallel with US2)
  - T012: Implement inherit_global flag
```

### User Story 2 Tests + Implementation Parallel

```
After T014 (tests) written:
  - T016: Implement inline validation (parallel with US1)
  - T017: Implement external loading (parallel with US1)

After T015 (tests) written:
  - T018: Implement response format app
  - T019: Ensure agent-specific
```

### Cross-Story Implementation Parallelization

```
After all Phase 2 complete + tests written (T008, T009, T014, T015, T020):
  - US1 implementation: T010-T013 (parallel with US2 + US4)
  - US2 implementation: T016-T019 (parallel with US1 + US4)
  - US4 implementation: T021-T022 (parallel with US1 + US2)
```

---

## TDD Workflow

**For each User Story:**

1. **Write Tests First** (T008, T009, T014, T015, T020)

   - Define all test cases
   - Tests should FAIL initially (red)
   - Include assertions for all acceptance criteria

2. **Implement Feature** (T010-T013, T016-T019, T021-T022)

   - Write minimal code to pass tests
   - Tests should PASS (green)
   - Refactor as needed while keeping tests passing

3. **Verify Integration** (T023)

   - End-to-end test combining all stories
   - All tests pass together

4. **Code Quality** (T024)
   - Format, lint, type-check, security checks
   - All checks pass

---

## Implementation Strategy

### MVP (Minimal Viable Product)

**Scope**: User Stories 1, 2, 4
**Tasks**: T001-T007, T008-T022, T023-T024 (24 tasks)
**Duration**: ~5-7 days (2 developers with TDD discipline)

**Deliverables**:

- Global settings loading (user and project levels)
- Configuration inheritance and override (no response_format inheritance)
- Response format definition and validation (agent-level only)
- Clear error messages for configuration issues
- 100% test coverage for config module (TDD enforced)

**Acceptance**:

- All P1 user stories pass acceptance criteria
- All unit tests pass (TDD verification)
- All integration tests pass
- Code quality checks pass (format, lint, type-check, security)

---

## Test-First Checklist

Each TDD cycle follows:

```
WRITE TESTS (Red)
  - [ ] T### [US#] Write test cases for feature
         [ ] Test 1: [description] in [test_file.py]
         [ ] Test 2: [description] in [test_file.py]
         [ ] Test 3: [description] in [test_file.py]
         [ ] All tests FAIL (expected, red phase)
         [ ] Commit: "WIP: [US#] Add test cases"

IMPLEMENT FEATURE (Green)
  - [ ] T### [US#] Implement feature to pass tests
         [ ] Feature in [src_file.py]
         [ ] All tests PASS (green phase)
         [ ] Code coverage ≥ 80%
         [ ] No new test failures
         [ ] Commit: "[US#] Implement feature (pass tests)"

REFACTOR (Refactor)
  - [ ] Run make format
  - [ ] Run make lint-fix
  - [ ] Run make type-check
  - [ ] All tests still PASS
  - [ ] Commit: "[US#] Refactor and code quality"
```

---

## Code Quality Standards

**Must pass before commit**:

```bash
make format              # Black + Ruff formatting
make lint              # Ruff linting
make type-check        # MyPy strict mode
make security          # Bandit + Safety + detect-secrets
make test-unit         # Run unit tests with coverage
```

**Coverage Requirement**: 100% for new code (TDD enforced), 80%+ overall config module

**Style Guide**: Google Python Style Guide (enforced by Black, Ruff)

---

## Testing Strategy (TDD)

### Unit Tests (tests/unit/config/)

1. **test_loader.py** (T008, T015)

   - User-level config discovery (T008)
   - Project-level config discovery (T008)
   - File extension preference (.yml > .yaml) (T008)
   - Inline response_format storage (T015)
   - External schema file loading (T015)
   - response_format NOT inherited from global (T015)

2. **test_inheritance.py** (T009)

   - User → project → agent precedence
   - inherit_global: false disables inheritance
   - Configuration merging with overrides
   - Agent-level overrides global settings
   - response_format is agent-specific (not inherited)

3. **test_schema_validation.py** (T014)

   - Inline YAML-to-JSON schema parsing
   - External .json schema file loading
   - Basic JSON Schema keyword validation
   - Rejection of unsupported keywords

4. **test_error_handling.py** (T020)
   - YAML syntax errors with line numbers
   - Invalid JSON in response_format with line numbers
   - Missing schema files with path
   - Unknown JSON Schema keywords
   - LLM provider unsupported warnings

### Integration Tests (tests/integration/)

1. **test_config_end_to_end.py** (T023)
   - Load user-level config
   - Load project-level config with override
   - Load agent with response_format (agent-level)
   - Verify inheritance of global settings
   - Verify response_format applied to agent
   - Test all error scenarios together

---

## Git Workflow

**Branch**: `005-global-settings-response-format`

**TDD Commit Message Format**:

```
[Task ID] [User Story] [Test|Implement|Refactor] Brief description

Longer description if needed.
- Point 1
- Point 2

🤖 Generated with Claude Code
```

**Examples**:

```
[T008] [US1] Test File discovery for global settings

Write test cases for user-level and project-level config discovery.
- Test ~/.holodeck/config.yml detection
- Test config.yml at project root detection
- Test .yml preference over .yaml
- Tests in RED phase (expected to fail)

🤖 Generated with Claude Code
```

```
[T010] [US1] Implement Project-level config discovery

Implement project-level config.yml|yaml detection.
- Detect config.yml first, then config.yaml
- Prefer .yml if both exist
- Tests pass (GREEN phase)

🤖 Generated with Claude Code
```

```
[T010] [US1] Refactor and code quality

Format, lint, type-check code. Tests still pass.
- make format passed
- make lint passed
- make type-check passed
- All tests passing

🤖 Generated with Claude Code
```

---

## Success Criteria per User Story

### User Story 1: Configure Global Settings

- ✅ Tests written first: test_loader.py (T008), test_inheritance.py (T009)
- ✅ User-level config at ~/.holodeck/config.yml|yaml loaded
- ✅ Project-level config at config.yml|yaml loaded with precedence
- ✅ Agents inherit defaults (model, api_keys, vectorstores, deployment)
- ✅ Agent config overrides global settings
- ✅ inherit_global: false disables all inheritance
- ✅ All tests pass (RED → GREEN → REFACTOR cycle complete)

### User Story 2: Define Response Format (Agent-Level Only)

- ✅ Tests written first: test_schema_validation.py (T014), test_loader.py (T015)
- ✅ Inline response_format specified in agent.yaml
- ✅ External response_format file loaded from schemas/
- ✅ Response format applied to LLM calls
- ✅ Schema validated at config load time
- ✅ response_format NOT inherited from global config
- ✅ All tests pass (RED → GREEN → REFACTOR cycle complete)

### User Story 4: Schema Validation

- ✅ Tests written first: test_error_handling.py (T020)
- ✅ Invalid JSON syntax errors include line number
- ✅ Unknown schema keywords rejected with keyword name
- ✅ Missing files reported with path
- ✅ All errors include file location and remediation
- ✅ All tests pass (RED → GREEN → REFACTOR cycle complete)

---

## Risk Mitigation

| Risk                                | Mitigation                                        | Status   |
| ----------------------------------- | ------------------------------------------------- | -------- |
| Configuration precedence confusion  | TDD tests verify precedence before implementation | T009     |
| Response format not enforced at LLM | Integration test verifies constraint passing      | T023     |
| Error messages unclear to users     | TDD error handling tests validate clarity         | T020     |
| Performance: Config loading slow    | Profile with ~100 agents; optimize if needed      | Post-MVP |
| File permission issues              | TDD error handling covers edge cases              | T020     |

---

## Acceptance Testing

**Manual acceptance testing** (after all automated tests pass - TDD complete):

1. **US1 Acceptance** (after T008-T013 + tests pass):

   ```bash
   # Create ~/.holodeck/config.yml with model provider
   # Create project/config.yml with temperature override
   # Create agent.yaml with inherit_global: true
   # Run: pytest tests/unit/config/test_inheritance.py -v
   # Verify: Agent uses project temperature (from project config)
   # Verify: Agent uses model provider (from user config)
   # Verify: All tests pass
   ```

2. **US2 Acceptance** (after T016-T019 + tests pass):

   ```bash
   # Define inline response_format in agent.yaml
   # Run: pytest tests/unit/config/test_schema_validation.py -v
   # Define external response_format file
   # Run: pytest tests/unit/config/test_loader.py -v
   # Verify: Schema loaded and validated
   # Verify: All tests pass
   ```

3. **US4 Acceptance** (after T021-T022 + tests pass):
   ```bash
   # Run: pytest tests/unit/config/test_error_handling.py -v
   # Provide invalid JSON in response_format
   # Verify: Error with line number
   # Provide unknown keyword
   # Verify: Error with keyword name
   # Verify: All error tests pass
   ```

---

## Documentation & Examples

**Auto-generated from quickstart.md and spec.md:**

- Configuration examples (user-level, project-level)
- Response format examples (inline, external, agent-level only)
- Inheritance scenarios (model settings, not response_format)
- Troubleshooting guide
- API reference

**No additional documentation tasks needed** - design docs already complete (quickstart.md, data-model.md, contracts/).
