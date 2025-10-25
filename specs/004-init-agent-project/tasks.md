# Implementation Tasks: Initialize New Agent Project

**Feature**: Initialize New Agent Project (004-init-agent-project)
**Branch**: `004-init-agent-project`
**Created**: 2025-10-22
**Spec**: [spec.md](./spec.md) | **Plan**: [plan.md](./plan.md)

---

## Overview

This document breaks down the feature into independently testable phases using **Test-Driven Development (TDD)** approach. Each phase follows this pattern: write tests first, then implement to pass tests. Each user story can be implemented and tested in isolation.

**Estimated effort with TDD**: 55-70 development hours (1.5 weeks full-time for single developer). TDD adds ~20-30% overhead for test writing, but significantly reduces debugging and integration issues.

**Key Principle**: Tests are written and passing BEFORE implementation code is written. This ensures tests are independent of implementation details and capture true behavioral requirements.

### User Stories Summary

| Story | Priority | Title | Complexity |
|-------|----------|-------|------------|
| US1 | P1 | Create Basic Agent Project | High |
| US2 | P1 | Select Project Templates | High |
| US3 | P1 | Generate Sample Files and Examples | High |
| US4 | P1 | Validate Project Structure | Medium |
| US5 | P2 | Specify Project Metadata | Low |

### Task Summary (TDD-Driven)

| Phase | User Stories | Task Range | Count | Focus | Est. Hours |
|-------|--------------|-----------|-------|-------|-----------|
| 1 | Setup | T001-T011 | 11 | Infrastructure, tests first | 5 |
| 2 | Core Engine | T012-T026 | 15 | Unit tests (T012-T017), then implementation | 12 |
| 3 | US1: Basic Creation | T027-T039 | 13 | Integration tests first (T027-T031), then templates + CLI | 10 |
| 4 | US2: Templates | T040-T055 | 16 | Integration tests first (T040-T044), then 2 templates + management | 10 |
| 5 | US3: Examples | T056-T072 | 17 | Integration tests first (T056-T060), then 9 example files | 8 |
| 6 | US4: Validation | T073-T083 | 11 | Validation tests first (T073-T077), then error handling | 6 |
| 7 | US5: Metadata | T084-T093 | 10 | Metadata tests first (T084-T087), then CLI + templates | 5 |
| 8 | Polish & QA | T094-T139 | 46 | Edge cases, performance, docs, QA (all TDD) | 10 |
| **TOTAL** | **All** | **T001-T139** | **139** | **Complete feature, production-ready** | **66 hours** |

**TDD Note**: All 139 tasks follow test-first approach. Tests are written and passing BEFORE implementation. This ensures high code quality and low integration issues at the cost of ~20-30% additional development time.

### Implementation Strategy

**MVP Scope (Phase 1-3)**: US1 + US4 (basic project creation with validation)
- Smallest viable feature: create project directory with agent.yaml
- Time: 1-2 weeks (with TDD)
- Enables users to bootstrap projects and verify structure
- Tasks: T001-T039 (39 tasks including tests)

**Phase 2 (US2 + Templates)**: Add template system with 3 templates
- Adds template selection and manifest system
- Time: 2-3 weeks (with TDD)
- Major UX improvement, enables use case customization
- Tasks: T040-T055 (16 tasks including integration tests)

**Phase 3 (US3 + Examples)**: Generate example files
- Adds instructional content and sample configurations
- Time: 1-2 weeks (with TDD)
- Critical for self-serve learning
- Tasks: T056-T072 (17 tasks including integration tests)

**Phase 4 (US5 + Polish)**: Metadata support + production readiness
- Adds optional metadata fields, validation, quality assurance
- Time: 1-2 weeks (with TDD)
- Polish feature with comprehensive testing and documentation
- Tasks: T084-T139 (56 tasks including QA, performance, documentation tests)

---

## Phase 1: Setup & Infrastructure

**Goal**: Establish project structure, dependencies, and core utilities

### TDD Approach

All implementation is test-first. Each task includes:
1. Test definition (what behavior we expect)
2. Test implementation (write failing test)
3. Implementation (make test pass)

### Core Tasks

- [x] T001 Create Python package structure in `src/holodeck/` with `__init__.py` files and CLI entry point
- [x] T002 [P] Add Click, Pyyaml, Jinja2 dependencies to `pyproject.toml` with version pins
- [x] T003 [P] Write test for Click CLI group functionality in `tests/unit/test_cli_setup.py`
- [x] T004 [P] Implement Click CLI group in `src/holodeck/cli/__init__.py` and register init command (test-driven from T003)
- [x] T005 Write tests for exception classes in `tests/unit/test_exceptions.py` (ValidationError, InitError, TemplateError)
- [x] T006 Implement base error/exception classes in `src/holodeck/lib/exceptions.py` (test-driven from T005)
- [x] T007 [P] Write tests for Pydantic models in `tests/unit/test_models.py`: ProjectInitInput, ProjectInitResult, TemplateManifest
- [x] T008 [P] Implement Pydantic models in `src/holodeck/models/project_config.py` and `src/holodeck/models/template_manifest.py` (test-driven from T007)
- [x] T009 Setup pytest test structure with conftest.py in `tests/conftest.py` with markers (@pytest.mark.unit, @pytest.mark.integration)
- [x] T010 [P] Create `.gitignore` template in `src/holodeck/templates/_static/.gitignore` for all projects
- [x] T011 Create test fixtures directory `tests/fixtures/` and `tests/fixtures/conftest.py` with temp project helpers

**Acceptance Criteria**:
- ✓ All imports work without errors
- ✓ Click CLI responds to `--help`
- ✓ Pydantic models validate without errors
- ✓ Test runner (pytest) executes successfully with 100% of setup tests passing
- ✓ No linting/type-checking errors (ruff, mypy)

---

## Phase 2: Core Initialization Engine

**Goal**: Implement project creation logic and validation framework (blocking prerequisite for all user stories) - TDD-driven

### Unit Tests First (TDD)

- [x] T012 Write unit tests for `ProjectInitializer.validate_inputs()` in `tests/unit/test_project_init.py` covering: valid names, invalid names (special chars, leading digits), missing template, directory permissions check
- [x] T013 Write unit tests for `ProjectInitializer.load_template()` in `tests/unit/test_project_init.py` covering: valid manifest, missing manifest, malformed YAML
- [x] T014 [P] Write unit tests for `ProjectInitializer.initialize()` in `tests/unit/test_project_init.py` covering: successful creation, directory exists (no --force), partial cleanup on error
- [x] T015 Write unit tests for `TemplateRenderer.render_template()` in `tests/unit/test_template_engine.py` covering: valid Jinja2 templates, template variable substitution, missing variables
- [x] T016 [P] Write unit tests for `TemplateRenderer.validate_agent_config()` in `tests/unit/test_template_engine.py` covering: valid YAML, invalid YAML, schema validation pass/fail
- [x] T017 Write unit tests for error handling in `tests/unit/test_template_engine.py` covering: rendering failures, validation failures, clear error messages

### Core Logic Implementation (TDD)

- [x] T018 Implement `ProjectInitializer` class in `src/holodeck/cli/utils/project_init.py` with methods: `validate_inputs()`, `load_template()`, `initialize()` to pass T012-T014 tests
- [x] T019 Implement `ProjectInitializer.validate_inputs()`: name pattern validation, template existence check, directory permissions check (test-driven from T012)
- [x] T020 [P] Implement `ProjectInitializer.load_template()`: load TemplateManifest from YAML, validate manifest schema (test-driven from T013)
- [x] T021 Implement `ProjectInitializer.initialize()`: directory creation, file writing, all-or-nothing cleanup semantics (test-driven from T014)
- [x] T022 [P] Create `TemplateRenderer` class in `src/holodeck/lib/template_engine.py` with methods: `render_template()`, `validate_agent_config()`, `render_and_validate()` (test-driven from T015-T017)
- [x] T023 Implement Jinja2 environment setup with restricted filters for safety in `TemplateRenderer.__init__()` (test-driven from T015)
- [x] T024 [P] Implement `validate_agent_config()` method that validates rendered YAML against AgentConfig schema (test-driven from T016)
- [x] T025 Implement error handling for template rendering failures with clear messages (test-driven from T017)
- [x] T026 [P] Ensure `AgentConfig` model is available: import from core models package or create in `src/holodeck/models/agent_config.py`

**Acceptance Criteria**:
- ✓ All unit tests pass (T012-T017)
- ✓ `ProjectInitializer` creates directories with correct structure
- ✓ Input validation rejects invalid names, templates, permissions
- ✓ TemplateRenderer renders Jinja2 without errors
- ✓ AgentConfig validation passes for valid YAML, fails for invalid
- ✓ All-or-nothing cleanup on failure (no partial directories)
- ✓ Clear error messages for all failure modes
- ✓ 80%+ unit test coverage for core logic

---

## Phase 3: User Story 1 - Create Basic Agent Project

**Goal**: Developers can run `holodeck init <name>` and get a working project structure - TDD-driven

### Integration Tests First (TDD)

- [x] T027 [US1] Write integration test for basic project creation in `tests/integration/test_init_basic.py`: verify `holodeck init test-project` creates directory, agent.yaml, and all folders
- [x] T028 [US1] Write integration test for default template selection in `tests/integration/test_init_basic.py`: verify conversational is default when --template omitted
- [x] T029 [US1] [P] Write integration test for overwrite behavior in `tests/integration/test_init_basic.py`: prompt user when dir exists, accept --force flag
- [x] T030 [US1] Write integration test for success message in `tests/integration/test_init_basic.py`: verify output shows location and next steps
- [x] T031 [US1] Write integration test for Ctrl+C handling in `tests/integration/test_init_basic.py`: verify cleanup on interrupt

### Template Creation (TDD)

- [x] T032 [US1] Create default `conversational` template directory structure in `src/holodeck/templates/conversational/` (with tests from T027-T031)
- [x] T033 [US1] Create `conversational/manifest.yaml` with template metadata, variables, and file list in `src/holodeck/templates/conversational/manifest.yaml`
- [x] T034 [US1] Create `conversational/agent.yaml.j2` Jinja2 template with default model (OpenAI), instructions placeholder, and tools section in `src/holodeck/templates/conversational/agent.yaml.j2`
- [x] T035 [US1] Create `conversational/instructions/system-prompt.md.j2` with sample conversational system prompt in `src/holodeck/templates/conversational/instructions/system-prompt.md.j2`

### CLI Implementation (TDD)

- [x] T036 [US1] Implement Click command in `src/holodeck/cli/commands/init.py` with arguments (project_name) and options (--template, --description, --author, --force) to pass T027-T031 tests
- [x] T037 [US1] [P] Implement command handler that calls `ProjectInitializer.initialize()` and formats result messages in `src/holodeck/cli/commands/init.py` (test-driven from T030)
- [x] T038 [US1] Implement overwrite prompt when directory exists (unless --force) in `src/holodeck/cli/commands/init.py` (test-driven from T029)
- [x] T039 [US1] Handle Ctrl+C gracefully with cleanup in `src/holodeck/cli/commands/init.py` (test-driven from T031)

**Acceptance Criteria (US1)**:
- ✓ All integration tests pass (T027-T031)
- ✓ `holodeck init my-project` creates directory with all required folders
- ✓ Generated agent.yaml is valid YAML and parses without errors
- ✓ Default template is conversational
- ✓ Project structure matches spec requirements (agent.yaml, instructions/, tools/, data/, tests/)
- ✓ Success message displays project location and next steps
- ✓ Overwrite behavior works correctly (prompt and --force)
- ✓ < 30 seconds initialization time

---

## Phase 4: User Story 2 - Select Project Templates

**Goal**: Developers can choose from 3 domain-specific templates (conversational, research, customer-support) - TDD-driven

### Integration Tests First (TDD)

- [x] T040 [US2] Write integration test for research template in `tests/integration/test_init_templates.py`: verify `holodeck init <name> --template research` creates research project
- [x] T041 [US2] Write integration test for customer-support template in `tests/integration/test_init_templates.py`: verify `holodeck init <name> --template customer-support` creates support project
- [x] T042 [US2] [P] Write test for invalid template selection error handling in `tests/integration/test_init_templates.py`: show list of templates on error
- [x] T043 [US2] Write test that all 3 templates produce valid agent.yaml files in `tests/integration/test_init_templates.py`: schema validation
- [x] T044 [US2] Write test for template-specific instructions in `tests/integration/test_init_templates.py`: research has vector search, support has functions

### Template Development (TDD)

- [x] T045 [US2] Create `research` template directory in `src/holodeck/templates/research/` (driven by T040)
- [x] T046 [US2] Create `research/manifest.yaml` with research-focused variables and defaults in `src/holodeck/templates/research/manifest.yaml`
- [x] T047 [US2] Create `research/agent.yaml.j2` with research instructions and vector search tool example in `src/holodeck/templates/research/agent.yaml.j2`
- [x] T048 [US2] Create `research/instructions/system-prompt.md.j2` with research analysis system prompt in `src/holodeck/templates/research/instructions/system-prompt.md.j2`
- [x] T049 [US2] [P] Create `customer-support` template directory in `src/holodeck/templates/customer-support/` (driven by T041)
- [x] T050 [US2] Create `customer-support/manifest.yaml` with customer-support variables and defaults in `src/holodeck/templates/customer-support/manifest.yaml`
- [x] T051 [US2] Create `customer-support/agent.yaml.j2` with support instructions and function tool examples in `src/holodeck/templates/customer-support/agent.yaml.j2`
- [x] T052 [US2] Create `customer-support/instructions/system-prompt.md.j2` with support agent system prompt in `src/holodeck/templates/customer-support/instructions/system-prompt.md.j2`

### Template Management (TDD)

- [x] T053 [US2] [P] Create template discovery function in `src/holodeck/lib/template_engine.py` that lists available templates (driven by T042)
- [x] T054 [US2] Update `ProjectInitializer.load_template()` to validate template choice against available templates in `src/holodeck/cli/utils/project_init.py` (driven by T042)
- [x] T055 [US2] Update help text and error messages to list available templates in `src/holodeck/cli/commands/init.py` (driven by T042)

**Acceptance Criteria (US2)**:
- ✓ All integration tests pass (T040-T044)
- ✓ `holodeck init <name> --template research` creates research project
- ✓ `holodeck init <name> --template customer-support` creates support project
- ✓ Each template has appropriate default instructions
- ✓ Template selection is case-insensitive (friendly error for typos)
- ✓ Unknown template shows list of available templates
- ✓ All 3 templates generate valid agent.yaml per AgentConfig schema

---

## Phase 5: User Story 3 - Generate Sample Files and Examples

**Goal**: Generated projects include working example files for learning and reference - TDD-driven

### Integration Tests First (TDD)

- [x] T056 [US3] [P] Write test that all template files are generated in `tests/integration/test_init_examples.py`: verify instructions, tools/README, data, tests folders
- [x] T057 [US3] Write test that example test cases YAML is valid in `tests/integration/test_init_examples.py`: 2-3 valid examples per template
- [x] T058 [US3] Write test that instructions are present and non-empty in `tests/integration/test_init_examples.py`: template-specific content
- [x] T059 [US3] Write test that data files are present with proper formatting in `tests/integration/test_init_examples.py`: CSV, JSON, Markdown valid
- [x] T060 [US3] Write test for learning experience: examples discoverable and understandable in `tests/integration/test_init_examples.py`

### Example Files & Templates (TDD)

- [x] T061 [US3] Create `conversational/instructions/system-prompt.md.j2` with detailed conversational agent instructions in `src/holodeck/templates/conversational/instructions/system-prompt.md.j2` (test-driven from T058)
- [x] T062 [US3] Create `conversational/tools/README.md.j2` with instructions for adding custom functions in `src/holodeck/templates/conversational/tools/README.md.j2` (test-driven from T056)
- [x] T063 [US3] Create `conversational/data/faqs.md` with sample FAQ data for vector search in `src/holodeck/templates/conversational/data/faqs.md` (test-driven from T059)
- [x] T064 [US3] Update `conversational/agent.yaml.j2` to include 2-3 sample test cases in `test_cases` field in `src/holodeck/templates/conversational/agent.yaml.j2` (test-driven from T057)
- [x] T065 [US3] [P] Create `research/tools/README.md.j2` in `src/holodeck/templates/research/tools/README.md.j2` (test-driven from T056)
- [x] T066 [US3] Create `research/data/papers_index.json` with sample research papers index in `src/holodeck/templates/research/data/papers_index.json` (test-driven from T059)
- [x] T067 [US3] Update `research/agent.yaml.j2` to include research-focused test cases in `test_cases` field in `src/holodeck/templates/research/agent.yaml.j2` (test-driven from T057)
- [x] T068 [US3] Create `customer-support/tools/README.md.j2` in `src/holodeck/templates/customer-support/tools/README.md.j2` (test-driven from T056)
- [x] T069 [US3] Create `customer-support/data/sample_issues.csv` with sample customer issues in `src/holodeck/templates/customer-support/data/sample_issues.csv` (test-driven from T059)
- [x] T070 [US3] Update `customer-support/agent.yaml.j2` to include support ticket test cases in `test_cases` field in `src/holodeck/templates/customer-support/agent.yaml.j2` (test-driven from T057)

### Template Manifest Updates (TDD)

- [x] T071 [US3] Update all template manifests to include file list with template/static flags in manifest.yaml files (test-driven from T056)
- [x] T072 [US3] Ensure all `.j2` files have proper variable substitution for project_name, description, etc. (test-driven from T056)

**Acceptance Criteria (US3)**:
- ✓ All integration tests pass (T056-T060)
- ✓ All template files (instructions, tools/README, data, tests) are generated
- ✓ Example test cases follow HoloDeck test case schema
- ✓ Instructions are specific to template type (conversational/research/support)
- ✓ Data files are present and properly formatted (CSV/JSON/Markdown)
- ✓ Users can learn from generated examples without external docs
- ✓ All generated files validate against their respective schemas

---

## Phase 6: User Story 4 - Validate Project Structure - TDD-driven

**Goal**: Initialization provides clear feedback on success/failure with helpful error messages

### Validation Tests First (TDD)

- [x] T073 [US4] [P] Write test for valid project structure verification in `tests/integration/test_init_validation.py`: all directories and files exist
- [x] T074 [US4] Write test for agent.yaml YAML syntax validation rejection in `tests/integration/test_init_validation.py`: invalid YAML shows errors with line numbers
- [x] T075 [US4] Write test for AgentConfig schema validation with invalid YAML rejection in `tests/integration/test_init_validation.py`: missing required fields caught
- [x] T076 [US4] Write test for error message clarity and actionability in `tests/integration/test_init_validation.py`: helpful next steps provided
- [x] T077 [US4] Write test for partial cleanup on failure in `tests/integration/test_init_validation.py`: partial directories removed after error

### Validation Implementation (TDD)

- [x] T078 [US4] Implement directory validation in `ProjectInitializer.initialize()` in `src/holodeck/cli/utils/project_init.py` (test-driven from T073)
- [x] T079 [US4] Implement YAML syntax validation for agent.yaml before write in `TemplateRenderer.validate_agent_config()` in `src/holodeck/lib/template_engine.py` (test-driven from T074)
- [x] T080 [US4] Implement AgentConfig schema validation after YAML parse in `TemplateRenderer.validate_agent_config()` in `src/holodeck/lib/template_engine.py` (test-driven from T075)
- [x] T081 [US4] Create detailed error messages for validation failures (schema errors with line numbers) in `src/holodeck/lib/exceptions.py` (test-driven from T076)
- [x] T082 [US4] Implement success message showing all created files and paths in `src/holodeck/cli/commands/init.py` (test-driven from T073)
- [x] T083 [US4] Implement failure cleanup (remove partial directories) in `ProjectInitializer.initialize()` in `src/holodeck/cli/utils/project_init.py` (test-driven from T077)

**Acceptance Criteria (US4)**:
- ✓ All required directories are created
- ✓ Generated agent.yaml is syntactically valid YAML
- ✓ Generated agent.yaml validates against AgentConfig schema
- ✓ Success message shows project location and next steps
- ✓ Error messages are clear, actionable, and include line numbers for YAML errors
- ✓ No partial projects left on disk after failures
- ✓ Validation happens before file write (no invalid projects created)

---

## Phase 7: User Story 5 - Specify Project Metadata - TDD-driven

**Goal**: Developers can provide optional metadata (description, author) during initialization

### Metadata Tests First (TDD)

- [x] T084 [US5] [P] Write test for --description flag in generated agent.yaml in `tests/integration/test_init_metadata.py`: metadata persisted correctly
- [x] T085 [US5] Write test for --author flag in generated agent.yaml in `tests/integration/test_init_metadata.py`: author field populated
- [x] T086 [US5] Write test for metadata with special characters and escaping in `tests/integration/test_init_metadata.py`: quotes, newlines handled safely
- [x] T087 [US5] Write test for missing metadata defaults (placeholder text) in `tests/integration/test_init_metadata.py`: sensible defaults provided

### Metadata Implementation (TDD)

- [x] T088 [US5] Update `ProjectInitInput` model to include optional description and author fields in `src/holodeck/models/project_config.py` (test-driven from T084-T085)
- [x] T089 [US5] Update `ProjectInitializer.validate_inputs()` to validate metadata fields (max length, valid characters) in `src/holodeck/cli/utils/project_init.py` (test-driven from T086)
- [x] T090 [US5] Update CLI command to accept --description and --author flags in `src/holodeck/cli/commands/init.py` (test-driven from T084-T085)
- [x] T091 [US5] Pass metadata to template renderer in `ProjectInitializer.initialize()` in `src/holodeck/cli/utils/project_init.py` (test-driven from T084-T085)
- [x] T092 [US5] Update all template manifests to include description and author variables in manifest.yaml files (test-driven from T087)
- [x] T093 [US5] Update all `agent.yaml.j2` templates to include description and author fields in agent.yaml.j2 files (test-driven from T084-T085)

**Acceptance Criteria (US5)**:
- ✓ `holodeck init <name> --description "text"` stores description in agent.yaml
- ✓ `holodeck init <name> --author "name"` stores author in agent.yaml
- ✓ Metadata is preserved in agent.yaml structure
- ✓ Missing metadata shows placeholder text (e.g., "TODO: Add description")
- ✓ Metadata validation prevents invalid characters
- ✓ Metadata appears in generated agent.yaml

---

## Phase 8: Polish & Cross-Cutting Concerns - TDD-driven

**Goal**: Production-ready feature with comprehensive testing, documentation, and quality assurance

### Edge Case & Error Scenario Tests First (TDD)

- [x] T094 [P] Write tests for special characters in project name in `tests/integration/test_init_edge_cases.py`: should fail gracefully with clear message
- [x] T095 Write tests for very long project names in `tests/integration/test_init_edge_cases.py`: should truncate or reject
- [x] T096 [P] Write tests for read-only filesystem in `tests/integration/test_init_edge_cases.py`: permission error shown
- [x] T097 Write tests for disk full scenario in `tests/integration/test_init_edge_cases.py`: cleanup on failure
- [x] T098 [P] Write tests for corrupted template manifest in `tests/integration/test_init_edge_cases.py`: helpful error message
- [x] T099 Write tests for rapid consecutive init commands in `tests/integration/test_init_edge_cases.py`: no race conditions

### Edge Case & Error Handling Implementation (TDD)

- [ ] T100 Implement handling for special characters in project name (test-driven from T094)
- [ ] T101 Implement handling for very long project names (test-driven from T095)
- [ ] T102 Implement handling for read-only filesystem (test-driven from T096)
- [ ] T103 Implement handling for disk full scenario (test-driven from T097)
- [ ] T104 Implement handling for corrupted template manifest (test-driven from T098)
- [ ] T105 Implement race condition protection (test-driven from T099)

### Performance Tests First (TDD)

- [x] T106 [P] Write test for initialization time < 30 seconds per SC-001 in `tests/integration/test_init_performance.py`: profile and verify
- [x] T107 Write test for template rendering < 5 seconds per file in `tests/integration/test_init_performance.py`
- [x] T108 Write test for file I/O < 5 seconds total in `tests/integration/test_init_performance.py`

### Performance Optimization (TDD)

- [x] T109 Optimize template rendering if test T107 fails (test-driven from T107)
- [x] T110 Optimize file I/O if test T108 fails (test-driven from T108)

### Documentation Tests & Help (TDD)

- [x] T111 [P] Write test for CLI help text completeness in `tests/unit/test_cli_help.py`: examples present and clear
- [x] T112 Write test for version flag: `holodeck init --version` in `tests/unit/test_cli_help.py`
- [x] T113 Write test that README.md documents `holodeck init` command in `tests/unit/test_documentation.py`
- [x] T114 Write test for QUICKSTART.md accuracy in `tests/unit/test_documentation.py`: all examples work

### Documentation Implementation (TDD)

- [x] T115 Update CLI help text with examples in `src/holodeck/cli/commands/init.py` (test-driven from T111)
- [x] T116 Add version flag support: `holodeck --version` in `src/holodeck/cli/__init__.py` (test-driven from T112)
- [x] T117 Update README.md with `holodeck init` command documentation (test-driven from T113)
- [x] T118 Create QUICKSTART.md in repo root with user-facing getting started guide (test-driven from T114)
- [x] T118a Update `docs/guides/agent-configuration.md` to document author field and metadata
- [x] T118b Update `docs/getting-started/quickstart.md` with --description and --author examples
- [x] T118c Update `docs/api/models.md` to reflect new Agent.author field
- [x] T118d Ensure all `src/holodeck/models/__init__.py` exports latest models (Agent with author field)

### Quality Assurance Tests & Checks (TDD)

- [x] T119 [P] Write test that verifies 80%+ code coverage in `tests/unit/test_coverage.py`: coverage report generated
- [x] T120 Write test to verify all linting passes in `tests/unit/test_linting.py`: runs `make lint` successfully
- [x] T121 Write test to verify all type checking passes in `tests/unit/test_typing.py`: runs `make type-check` successfully
- [x] T122 Write test to verify all security checks pass in `tests/unit/test_security.py`: runs `make security` successfully
- [x] T123 [P] Write test to verify formatting compliance in `tests/unit/test_formatting.py`: runs `make format-check` successfully

### Quality Assurance Implementation & Verification (TDD)

- [x] T124 Run full test suite and verify 80%+ coverage (test-driven from T119): `pytest tests/ --cov`
- [x] T125 Run linting checks and fix violations (test-driven from T120): `make lint-fix`
- [x] T126 Run type checking for strict compliance (test-driven from T121): `make type-check`
- [x] T127 Run security checks and verify no issues (test-driven from T122): `make security`
- [x] T128 [P] Run formatting and ensure compliance (test-driven from T123): `make format`
- [x] T129 Run integration tests end-to-end with fresh environment

### Pre-Release Verification Tests (TDD)

- [x] T130 [P] Write test verifying all user story acceptance criteria met in `tests/integration/test_acceptance_criteria.py`: US1-US5 all pass
- [x] T131 Write test that all templates generate valid projects in `tests/integration/test_template_validation.py`: each template creates usable project
- [x] T132 Write test that generated projects work with `holodeck test` command in `tests/integration/test_generated_project_usability.py`

### Pre-Release Verification (TDD)

- [x] T133 Verify all user story acceptance criteria are met (test-driven from T130)
- [x] T134 Create test project with each template (test-driven from T131)
- [x] T135 [P] Verify generated projects are usable with `holodeck test` (test-driven from T132)
- [x] T136 Verify error messages are user-friendly (manual user testing)
- [x] T137 Update CHANGELOG.md with feature description and CLI usage
- [x] T138 Verify documentation is complete and links work
- [x] T139 Run `make ci` locally and verify all checks pass

**Acceptance Criteria (Phase 8)**:
- ✓ 80%+ test coverage across all modules
- ✓ All linting, type-checking, and security checks pass
- ✓ < 30 seconds initialization time
- ✓ No unhandled exceptions (all errors caught and user-friendly)
- ✓ Documentation complete and examples working
- ✓ Ready for `pip install` and public use

---

## Task Dependencies & Parallel Execution

### Dependency Graph

```
Phase 1 (Setup & Infrastructure)
  ├─ T001-T011: TDD setup (tests first)
  └─ Blocks: All subsequent phases

Phase 2 (Core Engine)
  ├─ T012-T017: Unit tests first [P]
  ├─ T018-T026: Implementation (test-driven)
  └─ Blocks: All user story phases

Phase 3 (US1): Basic Creation
  ├─ T027-T031: Integration tests first [P]
  ├─ T032-T035: Template creation (test-driven)
  ├─ T036-T039: CLI implementation (test-driven)
  └─ No blocks (independent from US2-5)

Phase 4 (US2): Templates
  ├─ T040-T044: Integration tests first [P]
  ├─ T045-T052: Template development (test-driven)
  ├─ T053-T055: Template management (test-driven)
  └─ Depends on: US1 complete

Phase 5 (US3): Examples
  ├─ T056-T060: Integration tests first [P]
  ├─ T061-T072: Example files (test-driven)
  └─ Depends on: US2 complete

Phase 6 (US4): Validation
  ├─ T073-T077: Validation tests first [P]
  ├─ T078-T083: Implementation (test-driven)
  └─ Depends on: Phase 2 complete

Phase 7 (US5): Metadata
  ├─ T084-T087: Metadata tests first [P]
  ├─ T088-T093: Implementation (test-driven)
  └─ Depends on: US1 complete

Phase 8 (Polish)
  ├─ T094-T099: Edge case tests first [P]
  ├─ T100-T105: Edge case handling (test-driven)
  ├─ T106-T108: Performance tests first
  ├─ T109-T110: Performance optimization (test-driven)
  ├─ T111-T114: Documentation tests first [P]
  ├─ T115-T118: Documentation implementation (test-driven)
  ├─ T119-T123: QA tests first [P]
  ├─ T124-T129: QA verification (test-driven)
  ├─ T130-T132: Acceptance tests first [P]
  ├─ T133-T139: Pre-release verification (test-driven)
  └─ Depends on: US1-US5 complete
```

### Parallel Execution Opportunities

**Setup Phase (T001-T011)**:
- T002, T004, T006, T008, T010 can run in parallel (different files)
- Sequential: T001 → then others
- Estimated parallel time: ~2 days instead of 3 days

**Core Engine (T012-T026)**:
- Unit tests T012-T017 first (5 days)
- Then T018-T026 implementation runs after tests pass
- 2 developers: one on ProjectInitializer tests (T012-T017), one on TemplateRenderer tests (T012-T017)
- Estimated parallel time: ~7 days instead of 10 days

**Templates (T040-T055)**:
- Integration tests T040-T044 first (parallel execution)
- Template development T045-T052 can start after tests define requirements
- Research and customer-support templates T045-T052 can be done in parallel
- 2 developers: one on research, one on support (conversational baseline)
- Estimated parallel time: ~5 days instead of 7 days

**Examples (T056-T072)**:
- Integration tests T056-T060 first
- Example file creation T061-T072 for all 3 templates in parallel
- Tests can run in parallel
- Estimated parallel time: ~4 days instead of 5 days

**Validation & Metadata (T073-T093)**:
- Validation tests T073-T077 and metadata tests T084-T087 can run in parallel
- Implementation T078-T083 and T088-T093 can proceed in parallel
- 2 developers: one on validation, one on metadata
- Estimated parallel time: ~4 days instead of 5 days

**Polish & QA (T094-T139)**:
- Test-first approach: all test suites can be written in parallel (T094-T132)
- Implementation and QA can proceed in parallel (T100-T105, T109-T110, T115-T129)
- 2-3 developers: edge cases, performance, documentation, QA
- Estimated parallel time: ~6 days instead of 10 days

### Recommended Team Structure

**Solo Developer** (55-70 hours with TDD):
1. Do Phase 1 (Setup) - 5 hours
2. Do Phase 2 (Core Engine - tests first) - 12 hours
3. Do Phase 3 (US1 Basic - tests first) - 10 hours
4. Do Phase 4 (US2 Templates - tests first) - 10 hours
5. Do Phase 5 (US3 Examples - tests first) - 8 hours
6. Do Phase 6 (US4 Validation - tests first) - 6 hours
7. Do Phase 7 (US5 Metadata - tests first) - 5 hours
8. Do Phase 8 (Polish - tests first) - 10 hours
Total: 66 hours (1.5 weeks full-time)

**TDD Note**: Test-first approach adds ~20-30% overhead (writing tests before implementation), but significantly reduces debugging and integration issues. The investment in tests pays off in code quality and maintainability.

**2 Developers** (parallel, 30-35 hours each with TDD):
- Dev 1: Phase 1, Phase 2 tests (ProjectInitializer T012-T017), US1 (T027-T039), US2 (T040-T055), Phase 8 tests/docs
- Dev 2: Phase 2 tests (TemplateRenderer T012-T017), US3 (T056-T072), US4 (T073-T083), US5 (T084-T093), Phase 8 QA/verification
- Coordinate on Phase 1 setup and Phase 2 test definitions
- Estimated: 2.5-3 weeks

---

## Success Metrics

- **SC-001**: Projects initialize in < 30 seconds ✓ (verify with T101-T103)
- **SC-002**: Generated agent.yaml validates with 0 errors ✓ (verify with T070-T072)
- **SC-003**: All template files present and formatted ✓ (verify with T060-T063)
- **SC-004**: 80%+ first-time user success ✓ (verify with user testing)
- **SC-005**: Customization learnable in < 2 minutes ✓ (verify with example files)
- **SC-006**: Test cases validate perfectly ✓ (verify with T060-T063)
- **SC-007**: Can run `holodeck test` immediately ✓ (verify with T083)

---

## Testing Strategy (TDD-Driven)

### Unit Tests First (Foundation Layer)
- **Phase 1 setup tests**: T003, T005, T007 (before implementation)
- **Phase 2 core tests**: T012-T017 (ProjectInitializer, TemplateRenderer logic before implementation)
- **Phase 8 quality tests**: T119-T123 (coverage, linting, typing, security, formatting before QA)
- Coverage target: 80%+ (verified with T119)
- Run: `pytest tests/unit/ -v`

### Integration Tests First (Feature Layer)
- **Phase 3 (US1)**: T027-T031 (basic creation tests before template/CLI implementation)
- **Phase 4 (US2)**: T040-T044 (template selection tests before template development)
- **Phase 5 (US3)**: T056-T060 (example file tests before example creation)
- **Phase 6 (US4)**: T073-T077 (validation tests before implementation)
- **Phase 7 (US5)**: T084-T087 (metadata tests before implementation)
- **Phase 8 edge cases**: T094-T099 (error scenario tests before handling)
- **Phase 8 performance**: T106-T108 (performance tests before optimization)
- **Phase 8 documentation**: T111-T114 (documentation tests before writing docs)
- **Phase 8 acceptance**: T130-T132 (acceptance criteria tests before verification)
- Coverage target: 100% of user stories
- Run: `pytest tests/integration/ -v`

### Manual Tests (User Experience)
- T136: Verify error messages are user-friendly with actual users
- Run: `holodeck init <name>` with various inputs
- Verify: Clear error messages, helpful next steps

---

## Definition of Done (for each task)

- [ ] Code written and compiles/runs without errors
- [ ] All tests pass (unit + integration)
- [ ] Linting passes (ruff)
- [ ] Type checking passes (mypy --strict)
- [ ] Security checks pass (bandit)
- [ ] Code review approved
- [ ] Acceptance criteria met
- [ ] Documentation updated
- [ ] Commit message follows convention

---

## Notes for Implementation

1. **AgentConfig Model**: If not available in core package, create in `src/holodeck/models/agent_config.py` based on spec
2. **Template Variables**: Keep manifest.yaml `variables` whitelist to avoid Jinja2 injection risks
3. **All-or-Nothing Semantics**: Use try/except in `ProjectInitializer.initialize()` to cleanup on any error
4. **Template Validation**: Call `AgentConfig.model_validate()` immediately after Jinja2 render for YAML files
5. **User Messaging**: Use color output (green ✓, red ✗) for success/failure (via Click)
6. **Concurrency**: Don't worry about concurrent init in same directory for v0.1 (assume single-user)
