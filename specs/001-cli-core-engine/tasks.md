# Implementation Tasks: CLI & Core Agent Engine (v0.1)

**Feature**: CLI & Core Agent Engine (v0.1)
**Focus**: User Story 1 - Define Agent Configuration (Priority: P1)
**Branch**: `001-cli-core-engine`
**Date Generated**: 2025-10-19
**Planning**: `/speckit.plan` from `/specs/001-cli-core-engine/plan.md`
**Approach**: Test-Driven Development (TDD) - Tests First

---

## Executive Summary

This task list implements **User Story 1: Define Agent Configuration** using TDD methodology. Tests are written first, then implementation follows. All tasks are organized by dependency and parallelizability to enable efficient implementation.

- **Total Tasks**: 28
- **Setup Phase**: 3 tasks
- **Foundational Phase**: 5 tasks (includes test setup)
- **US1 Phase**: 20 tasks (tests written first, then implementation)
- **Parallel Opportunities**: 12 tasks marked [P]
- **Independent Test Criteria**: Full config loading/validation workflow testable end-to-end

**MVP Scope**: Complete Phase 1 (Setup) + Phase 2 (Foundational) + Phase 3 (US1) for standalone agent.yaml validation capability.

**Estimated Timeline**: ~45-55 hours development (includes TDD overhead, ~50% test code)

---

## Phase 1: Setup & Project Initialization

*Initialize project structure, dependencies, and development environment.*

### Phase 1 Goals
- Project scaffold ready
- Dependencies installed
- Development tools configured
- Basic module structure in place

### Phase 1 Independent Test Criteria
- `poetry install` completes without errors
- `pytest --version` shows pytest available
- `python -m agentlab --version` outputs version (placeholder OK)
- Project structure matches plan.md specification

---

### Tasks

- [x] T001 Create project structure per plan.md: `src/agentlab/{config,models,cli,lib}/`, `tests/{unit,integration,fixtures}/`
- [x] T002 Create `pyproject.toml` with core dependencies: Pydantic v2, PyYAML, python-dotenv, pytest, pytest-cov, ruff, black, mypy
- [x] T003 Create `src/agentlab/__init__.py` with version export: `__version__ = "0.1.0"`

---

## Phase 2: Foundational - Core Configuration Infrastructure

*Implement shared error handling, utilities, and data model foundations needed by all user stories.*

### Phase 2 Goals
- Custom exception hierarchy for configuration errors
- Base validation utilities
- Environment variable support
- Configuration defaults and templates
- Test fixtures and conftest setup

### Phase 2 Independent Test Criteria
- Custom exceptions raise with clear messages
- Environment variable substitution works (${VAR_NAME} pattern)
- Default configurations load successfully
- Validation utilities produce actionable error messages
- Test fixtures available and reusable

---

### Phase 2 - Tests First

- [x] T004 [P] Write exception tests in `tests/unit/test_errors.py`: test ConfigError message formatting, test ValidationError with field details, test FileNotFoundError with path suggestion
- [x] T005 [P] Write env loader tests in `tests/unit/test_env_loader.py`: test substitute_env_vars() with existing vars, test with missing vars, test ${VAR_NAME} pattern parsing
- [x] T006 [P] Write defaults tests in `tests/unit/test_defaults.py`: test default model config loads, test default tool config, test default evaluation config
- [x] T007 [P] Write validator utility tests in `tests/unit/test_validator.py`: test normalize_errors() returns human-readable format, test flatten_pydantic_errors() with nested errors
- [x] T008 Create conftest.py in `tests/`: setup pytest fixtures, configure coverage thresholds (min 80%), setup temp directories for file-based tests

### Phase 2 - Implementation

- [x] T009 [P] Create custom exception hierarchy in `src/agentlab/lib/errors.py`: ConfigError, ValidationError, FileNotFoundError subtypes with clear messages (satisfy T004)
- [x] T010 [P] Create environment variable utilities in `src/agentlab/config/env_loader.py`: substitute_env_vars() function supporting ${VAR_NAME} pattern (satisfy T005)
- [x] T011 [P] Create default configurations in `src/agentlab/config/defaults.py`: default model config, default tool config, default evaluation config (satisfy T006)
- [x] T012 Create validation utilities in `src/agentlab/config/validator.py`: normalize_errors(), flatten_pydantic_errors() for human-readable messages (satisfy T007)
- [x] T013 Create test fixtures in `tests/fixtures/agents/`: minimal_agent.yaml, valid_agent.yaml, invalid_agent.yaml for test reuse (satisfy T008)

---

## Phase 3: User Story 1 - Define Agent Configuration

*Implement core configuration schema, loading, and validation for agent.yaml files.*

### US1 Goals
Enable developers to define AI agents entirely through YAML configuration. Parse agent.yaml, validate against schema, support model providers (OpenAI, Azure, Anthropic), instructions (file/inline), tools (vectorstore, function, MCP, prompt), evaluations with flexible model configuration.

### US1 Independent Test Criteria
- Valid agent.yaml parses without errors
- Invalid agent.yaml shows clear validation messages
- Configuration composition (file + inline references) works
- Tool type validation correctly discriminates tool types
- Global config precedence works: agent.yaml > env vars > global config
- All Pydantic models validate against defined constraints
- All unit & integration tests pass with ≥80% coverage

### US1 TDD Cycle: Tests → Implementation

---

### US1 Phase 3a: Tool Model Tests & Implementation

- [x] T014 [P] Write tests for Tool base model in `tests/unit/test_tool_models.py`: test type field is required, test type discriminator validates tool type, test abstract fields not allowed
- [x] T015 [P] Write tests for VectorstoreTool in `tests/unit/test_tool_models.py`: test source required, test vector_field XOR vector_fields validation, test chunk_size/chunk_overlap optional with defaults, test embedding_model optional
- [x] T016 [P] Write tests for FunctionTool in `tests/unit/test_tool_models.py`: test file and function required, test parameters schema optional, test description optional
- [x] T017 [P] Write tests for MCPTool in `tests/unit/test_tool_models.py`: test server required, test config dict optional, test description optional
- [x] T018 [P] Write tests for PromptTool in `tests/unit/test_tool_models.py`: test template (inline XOR file), test parameters schema validation, test model config optional, test description optional

- [x] T019 [P] [US1] Create Tool base model in `src/agentlab/models/tool.py`: type field (vectorstore|function|mcp|prompt), discriminated union for subtypes (satisfy T014)
- [x] T020 [P] [US1] Create VectorstoreTool in `src/agentlab/models/tool.py`: implement all fields with validation (satisfy T015)
- [x] T021 [P] [US1] Create FunctionTool in `src/agentlab/models/tool.py`: implement all fields with validation (satisfy T016)
- [x] T022 [P] [US1] Create MCPTool in `src/agentlab/models/tool.py`: implement all fields with validation (satisfy T017)
- [x] T023 [P] [US1] Create PromptTool in `src/agentlab/models/tool.py`: implement all fields with validation (satisfy T018)

---

### US1 Phase 3b: LLM Provider & Evaluation Models Tests & Implementation

- [x] T024 [P] Write tests for LLMProvider in `tests/unit/test_llm_models.py`: test provider field required (enum), test name required, test temperature range 0-2, test max_tokens > 0, test endpoint required for azure_openai
- [x] T025 [P] Write tests for EvaluationMetric in `tests/unit/test_evaluation_models.py`: test metric name required, test threshold required and numeric, test enabled optional (default true), test model optional for per-metric override

- [x] T026 [P] [US1] Create LLMProvider model in `src/agentlab/models/llm.py`: provider enum (openai|azure_openai|anthropic), name, temperature, max_tokens, endpoint (satisfy T024)
- [x] T027 [P] [US1] Create EvaluationMetric model in `src/agentlab/models/evaluation.py`: metric name, threshold, enabled, model optional (satisfy T025)

---

### US1 Phase 3c: Test Case & Agent Models Tests & Implementation

- [x] T028 [P] Write tests for TestCase in `tests/unit/test_testcase_models.py`: test input required (string), test expected_tools optional (list), test ground_truth optional (string), test evaluations optional (list)
- [x] T029 Write tests for Agent in `tests/unit/test_agent_models.py`: test name required, test description optional, test model required (LLMProvider), test instructions file XOR inline, test tools optional (list), test evaluations optional (list), test test_cases optional (list)
- [x] T030 Write tests for GlobalConfig in `tests/unit/test_globalconfig_models.py`: test providers dict, test vectorstores dict, test deployment dict, test all sections optional

- [x] T031 [P] [US1] Create TestCase model in `src/agentlab/models/test_case.py`: implement with all field validations (satisfy T028)
- [x] T032 [US1] Create Agent model in `src/agentlab/models/agent.py`: implement with all field validations (satisfy T029)
- [x] T033 [US1] Create GlobalConfig model in `src/agentlab/models/config.py`: implement with all field validations (satisfy T030)

---

### US1 Phase 3d: ConfigLoader Tests & Implementation

- [x] T034 Write loader tests in `tests/unit/test_config_loader.py`: test load_agent_yaml() with valid YAML, test load_agent_yaml() with invalid YAML, test load_agent_yaml() with missing required fields, test parse_yaml() returns dict
- [x] T035 Write global config tests in `tests/unit/test_config_loader.py`: test load_global_config() reads ~/.agentlab/config.yaml, test load_global_config() returns empty config if file missing, test load_global_config() applies env var substitution
- [x] T036 Write precedence tests in `tests/unit/test_config_loader.py`: test merge_configs() agent.yaml overrides env vars, test merge_configs() env vars override global config, test merge_configs() respects precedence hierarchy
- [x] T037 Write file resolution tests in `tests/unit/test_config_loader.py`: test resolve_file_path() resolves relative to agent.yaml directory, test resolve_file_path() raises error if file not found, test resolve_file_path() with absolute paths
- [x] T038 Write error handling tests in `tests/unit/test_config_loader.py`: test validation errors converted to ConfigError, test error messages include field name and expected type, test file not found errors include full path

- [x] T039 [US1] Create ConfigLoader class in `src/agentlab/config/loader.py`: load_agent_yaml(path), parse_yaml(), return Agent instance (satisfy T034)
- [x] T040 [US1] Implement global config loading in ConfigLoader: load_global_config() method, env var substitution, merge with agent.yaml (satisfy T035, T036)
- [x] T041 [US1] Implement file reference resolution in ConfigLoader: resolve_file_path() for instructions and tool files, validation of file existence (satisfy T037)
- [x] T042 [US1] Implement error handling in ConfigLoader: catch Pydantic errors, convert to ConfigError with human messages (satisfy T038)

---

### US1 Phase 3e: Integration Tests

- [x] T043 Write integration test in `tests/integration/test_config_end_to_end.py`: full workflow (load agent.yaml with tools, merge global config, resolve file references, return Agent instance)
- [x] T044 Write integration test in `tests/integration/test_config_end_to_end.py`: error scenario (missing required field, invalid YAML, missing file reference, should raise ConfigError with actionable message)
- [x] T045 Write integration test in `tests/integration/test_config_end_to_end.py`: precedence scenario (create agent.yaml, global config, env vars; verify correct precedence)

- [x] T046 [US1] Execute integration tests: `pytest tests/integration/test_config_end_to_end.py -v`, all tests pass
- [x] T047 [US1] Execute all unit tests: `pytest tests/unit/ -v`, all tests pass, coverage ≥80%
- [x] T048 [US1] Generate coverage report: `pytest --cov=src/agentlab tests/`, verify ≥80% coverage on all modules

---

## Phase 4: Documentation & Polish

*Documentation, examples, and cross-cutting quality improvements.*

### Phase 4 Goals
- API documentation clear and comprehensive
- Example agent.yaml files available
- Type hints complete and mypy-compliant
- Code formatting and linting pass

---

### Tasks

- [ ] T049 [P] Create docstrings for all public classes in `src/agentlab/config/` and `src/agentlab/models/`: module docstring, class docstring, method docstring with examples
- [ ] T050 [P] Add type hints to all functions in `src/agentlab/config/loader.py` and `src/agentlab/models/`: full type coverage, no `Any` types where avoidable
- [ ] T051 Create example agent files in `docs/examples/`: basic_agent.yaml, with_tools.yaml, with_evaluations.yaml, with_global_config.yaml showing all major options
- [ ] T052 Run linting & formatting: `ruff check src/` and `black --check src/`, fix any violations
- [ ] T053 Run type checking: `mypy src/agentlab/ --strict`, fix any type errors

---

## Dependency Graph & Completion Order

```
Phase 1 (Setup)
    ├─ T001, T002, T003
    └─→ Phase 2 (Foundational)
         ├─ T004-T008 (Tests First)
         ├─ T009-T013 (Implementation)
         └─→ Phase 3 (US1 Implementation - TDD)
              ├─ T014-T018 (Tool Model Tests) [P]
              ├─ T019-T023 (Tool Model Implementation) [P]
              ├─ T024-T025 (Provider/Eval Tests) [P]
              ├─ T026-T027 (Provider/Eval Implementation) [P]
              ├─ T028-T030 (TestCase/Agent/Config Tests) [P]
              ├─ T031-T033 (TestCase/Agent/Config Implementation) [P]
              ├─ T034-T038 (Loader Tests)
              ├─ T039-T042 (Loader Implementation)
              ├─ T043-T045 (Integration Tests)
              ├─ T046-T048 (Test Execution & Coverage)
              └─→ Phase 4 (Polish)
                   ├─ T049, T050, T051 [P]
                   └─ T052, T053
```

### Critical Path (Longest Dependency Chain)
T001 → T002 → T003 → T004-T008 → T009-T013 → T034-T038 → T039-T042 → T043-T048 → T052-T053
**Estimated Critical Path**: ~40 hours

### Parallelizable Opportunities (TDD Batches)

**After Phase 2 Foundational complete (T013)**:
- **Batch 1 (Tests)**: T014-T018, T024-T025, T028-T030 write tests in parallel [P]
- **Batch 2 (Implementation)**: T019-T023, T026-T027, T031-T033 implement in parallel after tests pass [P]

**After Loader tests pass (T038)**:
- **Batch 3 (Loader Impl)**: T039-T042 sequential (depends on test feedback)

**Late Phase (Polish)**:
- T049, T050, T051 can run in parallel [P]

**Example Parallel Execution (TDD)**:
```
Sequential (setup):       T001-T003
Sequential (foundation):  T004-T013
Parallel (model tests):   [T014, T015, T016, T017, T018, T024, T025, T028, T029, T030]
Parallel (model impl):    [T019, T020, T021, T022, T023, T026, T027, T031, T032, T033]
Sequential (loader):      T034-T038 → T039-T042
Sequential (integration): T043-T045 → T046-T048
Parallel (polish):        [T049, T050, T051]
Sequential (final):       T052-T053
```

**Estimated Speedup**: 3-4x on model development with parallel TDD cycles.

---

## TDD Workflow

Each feature (Tool, LLMProvider, Evaluation, TestCase, Agent, GlobalConfig, Loader) follows this cycle:

### TDD Cycle Steps (Example: Tool Models)

1. **Write Tests First** (T014-T018)
   ```bash
   pytest tests/unit/test_tool_models.py -v
   # Tests fail (RED phase)
   ```

2. **Implement Minimal Code** (T019-T023)
   ```bash
   # Implement models to satisfy tests
   pytest tests/unit/test_tool_models.py -v
   # Tests pass (GREEN phase)
   ```

3. **Refactor & Validate** (During T049-T050)
   ```bash
   mypy src/agentlab/models/tool.py --strict
   ruff check src/agentlab/models/tool.py
   # Code quality passes (REFACTOR phase)
   ```

### Benefits of TDD for This Feature
- Tests document expected behavior (configuration validation rules)
- Catches edge cases early (tool type discrimination, file validation)
- Validation errors are explicit and testable
- Refactoring is safe (tests prevent regression)
- Coverage naturally ≥80% (tests write first)

---

## Implementation Notes

### Configuration Loading Flow
```python
1. User runs agent with agent.yaml path
2. ConfigLoader.load_agent_yaml(path) called
3. Read ~/.agentlab/config.yaml (if exists) → load GlobalConfig
4. Read agent.yaml → parse YAML
5. Apply precedence: agent.yaml > env vars > global config
6. Resolve file references (instructions, tool files)
7. Validate against Pydantic schema
8. Return Agent instance or raise ConfigError with human message
```

### Error Handling Strategy
- All Pydantic ValidationErrors caught and converted to ConfigError
- Error messages include: field name, expected type, actual value, file path (if applicable)
- File not found errors show full path and suggestion to create file
- For nested validation errors: flatten hierarchy and show user-friendly paths

### Testing Strategy (TDD)
1. **Unit Tests** (RED → GREEN → REFACTOR)
   - Each Pydantic model: test valid data, invalid data, edge cases
   - Loader: test YAML parsing, file resolution, env var substitution
   - Validation: test error messages readable, contains actionable information
   - **Tests written first**, implementation follows

2. **Integration Tests** (After all models & loader complete)
   - Full workflow: create agent.yaml, create global config, load both, verify precedence
   - File reference resolution: instructions file, tool file
   - Error scenarios: missing files, invalid YAML, missing required fields

3. **Coverage** (Measured after each phase)
   - Target: ≥80% coverage on all modules
   - Run `pytest --cov=src/agentlab tests/` after T048

### Performance Targets
- Config parsing: <100ms per agent.yaml
- Global config loading: <50ms
- Full loading + validation: <200ms
- Measured in test suite (pytest with timing)

### Global Configuration Precedence
As clarified in spec.md session 2025-10-19:
```
Priority 1 (Highest): agent.yaml explicit settings
Priority 2 (Medium):  Environment variables (e.g., OPENAI_API_KEY)
Priority 3 (Lowest):  ~/.agentlab/config.yaml global settings
```

---

## Success Criteria for Phase 3 (US1) Completion

All items must be satisfied:

1. ✅ All unit tests written first, then pass (T014-T038)
2. ✅ All integration tests pass (T043-T045)
3. ✅ Code coverage ≥80% (T048)
4. ✅ AgentConfig loads from valid agent.yaml without errors
5. ✅ Invalid agent.yaml raises ConfigError with human-readable message
6. ✅ Instruction files loaded from path (relative to agent.yaml)
7. ✅ Tool type validation works: tool type field correctly discriminates tool subtypes
8. ✅ EvaluationMetric supports per-metric model override
9. ✅ Global config loads from ~/.agentlab/config.yaml and merges correctly
10. ✅ Environment variable substitution works (${VAR_NAME} pattern)
11. ✅ Configuration precedence correct: agent.yaml > env vars > global config
12. ✅ File references validated (file existence checked before loading)
13. ✅ Type checking passes (mypy --strict)
14. ✅ Linting passes (ruff, black)

---

## Next Steps After US1

After all Phase 1-4 tasks complete and tests pass:

1. **Run `/speckit.plan plan-us2`** to plan User Story 2 (Initialize New Agent Project)
2. **US2 Focus**: `agentlab init` command, project template generation, default agent.yaml creation
3. **Integration**: US2 will use ConfigLoader from US1, add CLI layer

---

## Appendix: Task ID Reference

| ID Range | Phase | Purpose | TDD Stage |
|----------|-------|---------|-----------|
| T001-T003 | 1 | Setup & project initialization | - |
| T004-T013 | 2 | Foundational infrastructure | Tests (T004-T008) → Impl (T009-T013) |
| T014-T048 | 3 | US1 implementation (TDD cycle) | Tests → Impl → Test Execution |
| T049-T053 | 4 | Polish & documentation | - |

---

## Appendix: Test File Checklist

**Unit Test Files** (Write First):
- [ ] `tests/unit/test_errors.py`
- [ ] `tests/unit/test_env_loader.py`
- [ ] `tests/unit/test_defaults.py`
- [ ] `tests/unit/test_validator.py`
- [ ] `tests/unit/test_tool_models.py`
- [ ] `tests/unit/test_llm_models.py`
- [ ] `tests/unit/test_evaluation_models.py`
- [ ] `tests/unit/test_testcase_models.py`
- [ ] `tests/unit/test_agent_models.py`
- [ ] `tests/unit/test_globalconfig_models.py`
- [ ] `tests/unit/test_config_loader.py`

**Integration Test Files**:
- [ ] `tests/integration/test_config_end_to_end.py`

---

## Appendix: Source Code File Checklist

**Source Code Files** (Implement After Tests):
- [ ] `src/agentlab/__init__.py`
- [ ] `src/agentlab/lib/__init__.py`
- [ ] `src/agentlab/lib/errors.py`
- [ ] `src/agentlab/config/__init__.py`
- [ ] `src/agentlab/config/env_loader.py`
- [ ] `src/agentlab/config/validator.py`
- [ ] `src/agentlab/config/defaults.py`
- [ ] `src/agentlab/config/loader.py`
- [ ] `src/agentlab/models/__init__.py`
- [ ] `src/agentlab/models/tool.py`
- [ ] `src/agentlab/models/llm.py`
- [ ] `src/agentlab/models/evaluation.py`
- [ ] `src/agentlab/models/test_case.py`
- [ ] `src/agentlab/models/agent.py`
- [ ] `src/agentlab/models/config.py`

**Test Fixtures**:
- [ ] `tests/fixtures/agents/minimal_agent.yaml`
- [ ] `tests/fixtures/agents/valid_agent.yaml`
- [ ] `tests/fixtures/agents/invalid_agent.yaml`
- [ ] `tests/fixtures/agents/with_tools.yaml`
- [ ] `tests/fixtures/agents/with_evaluations.yaml`

**Documentation**:
- [ ] `docs/examples/basic_agent.yaml`
- [ ] `docs/examples/with_tools.yaml`
- [ ] `docs/examples/with_evaluations.yaml`
- [ ] `docs/examples/with_global_config.yaml`

---

**Generated**: 2025-10-19 | **Command**: `/speckit.tasks plan-us1` | **Approach**: Test-Driven Development (TDD)
