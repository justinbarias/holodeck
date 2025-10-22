# Implementation Tasks: Initialize New Agent Project

**Feature**: Initialize New Agent Project (004-init-agent-project)
**Branch**: `004-init-agent-project`
**Created**: 2025-10-22
**Spec**: [spec.md](./spec.md) | **Plan**: [plan.md](./plan.md)

---

## Overview

This document breaks down the feature into independently testable phases. Each user story can be implemented and tested in isolation. Estimated effort: 40-50 development hours (3-4 weeks for single developer).

### User Stories Summary

| Story | Priority | Title | Complexity |
|-------|----------|-------|------------|
| US1 | P1 | Create Basic Agent Project | High |
| US2 | P1 | Select Project Templates | High |
| US3 | P1 | Generate Sample Files and Examples | High |
| US4 | P1 | Validate Project Structure | Medium |
| US5 | P2 | Specify Project Metadata | Low |

### Implementation Strategy

**MVP Scope (Phase 1-3)**: US1 + US4 (basic project creation with validation)
- Smallest viable feature: create project directory with agent.yaml
- Time: 1-2 weeks
- Enables users to bootstrap projects and verify structure

**Phase 2 (US2 + Templates)**: Add template system with 3 templates
- Adds template selection and manifest system
- Time: 2-3 weeks
- Major UX improvement, enables use case customization

**Phase 3 (US3 + Examples)**: Generate example files
- Adds instructional content and sample configurations
- Time: 1-2 weeks
- Critical for self-serve learning

**Phase 4 (US5)**: Metadata support
- Adds optional metadata fields
- Time: 0.5 weeks
- Polish feature for documentation

---

## Phase 1: Setup & Infrastructure

**Goal**: Establish project structure, dependencies, and core utilities

### Core Tasks

- [ ] T001 Create Python package structure in `src/holodeck/` with `__init__.py` files and CLI entry point
- [ ] T002 [P] Add Click, Pyyaml, Jinja2 dependencies to `pyproject.toml` with version pins
- [ ] T003 [P] Create Click CLI group in `src/holodeck/cli/__init__.py` and register init command
- [ ] T004 Create base error/exception classes in `src/holodeck/lib/exceptions.py` (ValidationError, InitError, TemplateError)
- [ ] T005 [P] Create Pydantic models in `src/holodeck/models/project_config.py`: ProjectInitInput, ProjectInitResult
- [ ] T006 [P] Create Pydantic model in `src/holodeck/models/template_manifest.py`: TemplateManifest
- [ ] T007 Setup pytest test structure with conftest.py and test markers (@pytest.mark.unit, @pytest.mark.integration)
- [ ] T008 [P] Create `.gitignore` template in `src/holodeck/templates/_static/.gitignore` for all projects
- [ ] T009 Create test fixtures directory `tests/fixtures/` for temporary projects and template examples

**Acceptance Criteria**:
- ✓ All imports work without errors
- ✓ Click CLI responds to `--help`
- ✓ Pydantic models validate without errors
- ✓ Test runner (pytest) executes successfully
- ✓ No linting/type-checking errors (ruff, mypy)

---

## Phase 2: Core Initialization Engine

**Goal**: Implement project creation logic and validation framework (blocking prerequisite for all user stories)

### Core Logic Tasks

- [ ] T010 Implement `ProjectInitializer` class in `src/holodeck/cli/utils/project_init.py` with methods: `validate_inputs()`, `load_template()`, `initialize()`
- [ ] T011 Implement input validation in `ProjectInitializer.validate_inputs()`: name pattern, template existence, directory permissions check in `src/holodeck/cli/utils/project_init.py`
- [ ] T012 [P] Implement template loading logic in `ProjectInitializer.load_template()` to load TemplateManifest from YAML in `src/holodeck/cli/utils/project_init.py`
- [ ] T013 Implement directory creation logic in `ProjectInitializer.initialize()` with all-or-nothing semantics in `src/holodeck/cli/utils/project_init.py`
- [ ] T014 [P] Create `TemplateRenderer` class in `src/holodeck/lib/template_engine.py` with methods: `render_template()`, `validate_agent_config()`, `render_and_validate()`
- [ ] T015 Implement Jinja2 environment setup in `TemplateRenderer.__init__()` with restricted filters for safety in `src/holodeck/lib/template_engine.py`
- [ ] T016 Implement `validate_agent_config()` method that validates rendered YAML against AgentConfig Pydantic model in `src/holodeck/lib/template_engine.py`
- [ ] T017 Implement error handling for template rendering failures with clear messages in `src/holodeck/lib/template_engine.py`
- [ ] T018 [P] Implement `AgentConfig` model import from core models package (or create if missing) in `src/holodeck/models/agent_config.py`

### Unit Tests (Optional - Foundation Layer)

- [ ] T019 [P] Write unit tests for `ProjectInitializer.validate_inputs()` covering all edge cases (invalid names, missing template, permissions) in `tests/unit/test_project_init.py`
- [ ] T020 Write unit tests for `TemplateRenderer.render_template()` and `validate_agent_config()` with sample templates in `tests/unit/test_template_engine.py`

**Acceptance Criteria**:
- ✓ `ProjectInitializer` creates directories with correct structure
- ✓ Input validation rejects invalid names, templates, permissions
- ✓ TemplateRenderer renders Jinja2 without errors
- ✓ AgentConfig validation passes for valid YAML, fails for invalid
- ✓ All-or-nothing cleanup on failure (no partial directories)
- ✓ Clear error messages for all failure modes
- ✓ 80%+ test coverage for core logic

---

## Phase 3: User Story 1 - Create Basic Agent Project

**Goal**: Developers can run `holodeck init <name>` and get a working project structure

### Default Template & Core CLI

- [ ] T021 [US1] Create default `conversational` template directory structure in `src/holodeck/templates/conversational/`
- [ ] T022 [US1] Create `conversational/manifest.yaml` with template metadata, variables, and file list in `src/holodeck/templates/conversational/manifest.yaml`
- [ ] T023 [US1] Create `conversational/agent.yaml.j2` Jinja2 template with default model (OpenAI), instructions placeholder, and tools section in `src/holodeck/templates/conversational/agent.yaml.j2`
- [ ] T024 [US1] Create `conversational/instructions/system-prompt.md.j2` with sample conversational system prompt in `src/holodeck/templates/conversational/instructions/system-prompt.md.j2`
- [ ] T025 [US1] Implement Click command in `src/holodeck/cli/commands/init.py` with arguments (project_name) and options (--template, --description, --author, --force)
- [ ] T026 [US1] Implement command handler that calls `ProjectInitializer.initialize()` and formats result messages in `src/holodeck/cli/commands/init.py`
- [ ] T027 [US1] Implement success message showing project name, location, and next steps in `src/holodeck/cli/commands/init.py`
- [ ] T028 [US1] Implement overwrite prompt when directory exists (unless --force) in `src/holodeck/cli/commands/init.py`
- [ ] T029 [US1] Handle Ctrl+C gracefully with cleanup in `src/holodeck/cli/commands/init.py`

### Integration Tests (US1)

- [ ] T030 [US1] [P] Write integration test for basic project creation in `tests/integration/test_init_basic.py`: verify directory, agent.yaml, and folder structure
- [ ] T031 [US1] Write integration test for default template selection in `tests/integration/test_init_basic.py`
- [ ] T032 [US1] Write integration test for overwrite behavior (prompt vs --force) in `tests/integration/test_init_basic.py`

**Acceptance Criteria (US1)**:
- ✓ `holodeck init my-project` creates directory with all required folders
- ✓ Generated agent.yaml is valid YAML and parses without errors
- ✓ Default template is conversational
- ✓ Project structure matches spec requirements (agent.yaml, instructions/, tools/, data/, tests/)
- ✓ Success message displays project location and next steps
- ✓ Overwrite behavior works correctly (prompt and --force)
- ✓ < 30 seconds initialization time

---

## Phase 4: User Story 2 - Select Project Templates

**Goal**: Developers can choose from 3 domain-specific templates (conversational, research, customer-support)

### Template Development

- [ ] T033 [US2] Create `research` template directory in `src/holodeck/templates/research/`
- [ ] T034 [US2] Create `research/manifest.yaml` with research-focused variables and defaults in `src/holodeck/templates/research/manifest.yaml`
- [ ] T035 [US2] Create `research/agent.yaml.j2` with research instructions and vector search tool example in `src/holodeck/templates/research/agent.yaml.j2`
- [ ] T036 [US2] Create `research/instructions/system-prompt.md.j2` with research analysis system prompt in `src/holodeck/templates/research/instructions/system-prompt.md.j2`
- [ ] T037 [US2] Create `customer-support` template directory in `src/holodeck/templates/customer-support/`
- [ ] T038 [US2] Create `customer-support/manifest.yaml` with customer-support variables and defaults in `src/holodeck/templates/customer-support/manifest.yaml`
- [ ] T039 [US2] Create `customer-support/agent.yaml.j2` with support instructions and function tool examples in `src/holodeck/templates/customer-support/agent.yaml.j2`
- [ ] T040 [US2] Create `customer-support/instructions/system-prompt.md.j2` with support agent system prompt in `src/holodeck/templates/customer-support/instructions/system-prompt.md.j2`

### Template Management

- [ ] T041 [US2] [P] Create template discovery function in `src/holodeck/lib/template_engine.py` that lists available templates
- [ ] T042 [US2] Update `ProjectInitializer.load_template()` to validate template choice against available templates in `src/holodeck/cli/utils/project_init.py`
- [ ] T043 [US2] Update help text and error messages to list available templates in `src/holodeck/cli/commands/init.py`

### Integration Tests (US2)

- [ ] T044 [US2] [P] Write integration test for research template in `tests/integration/test_init_templates.py`
- [ ] T045 [US2] Write integration test for customer-support template in `tests/integration/test_init_templates.py`
- [ ] T046 [US2] Write test for invalid template selection error handling in `tests/integration/test_init_templates.py`
- [ ] T047 [US2] Write test that all 3 templates produce valid agent.yaml files in `tests/integration/test_init_templates.py`

**Acceptance Criteria (US2)**:
- ✓ `holodeck init <name> --template research` creates research project
- ✓ `holodeck init <name> --template customer-support` creates support project
- ✓ Each template has appropriate default instructions
- ✓ Template selection is case-insensitive (friendly error for typos)
- ✓ Unknown template shows list of available templates
- ✓ All 3 templates generate valid agent.yaml per AgentConfig schema

---

## Phase 5: User Story 3 - Generate Sample Files and Examples

**Goal**: Generated projects include working example files for learning and reference

### Example Files & Templates

- [ ] T048 [US3] Create `conversational/instructions/system-prompt.md.j2` with detailed conversational agent instructions in `src/holodeck/templates/conversational/instructions/system-prompt.md.j2`
- [ ] T049 [US3] Create `conversational/tools/README.md.j2` with instructions for adding custom functions in `src/holodeck/templates/conversational/tools/README.md.j2`
- [ ] T050 [US3] Create `conversational/data/faqs.md` with sample FAQ data for vector search in `src/holodeck/templates/conversational/data/faqs.md`
- [ ] T051 [US3] Create `conversational/tests/example_test_cases.yaml.j2` with 2-3 sample test cases in `src/holodeck/templates/conversational/tests/example_test_cases.yaml.j2`
- [ ] T052 [US3] [P] Create `research/tools/README.md.j2` in `src/holodeck/templates/research/tools/README.md.j2`
- [ ] T053 [US3] Create `research/data/papers_index.json` with sample research papers index in `src/holodeck/templates/research/data/papers_index.json`
- [ ] T054 [US3] Create `research/tests/example_test_cases.yaml.j2` with research-focused test cases in `src/holodeck/templates/research/tests/example_test_cases.yaml.j2`
- [ ] T055 [US3] Create `customer-support/tools/README.md.j2` in `src/holodeck/templates/customer-support/tools/README.md.j2`
- [ ] T056 [US3] Create `customer-support/data/sample_issues.csv` with sample customer issues in `src/holodeck/templates/customer-support/data/sample_issues.csv`
- [ ] T057 [US3] Create `customer-support/tests/example_test_cases.yaml.j2` with support ticket test cases in `src/holodeck/templates/customer-support/tests/example_test_cases.yaml.j2`

### Template Manifest Updates

- [ ] T058 [US3] Update all template manifests to include file list with template/static flags in manifest.yaml files
- [ ] T059 [US3] Ensure all `.j2` files have proper variable substitution for project_name, description, etc.

### Integration Tests (US3)

- [ ] T060 [US3] [P] Write test that all template files are generated in `tests/integration/test_init_examples.py`
- [ ] T061 [US3] Write test that example test cases YAML is valid in `tests/integration/test_init_examples.py`
- [ ] T062 [US3] Write test that instructions are present and non-empty in `tests/integration/test_init_examples.py`
- [ ] T063 [US3] Write test that data files are present with proper formatting in `tests/integration/test_init_examples.py`

**Acceptance Criteria (US3)**:
- ✓ All template files (instructions, tools/README, data, tests) are generated
- ✓ Example test cases follow HoloDeck test case schema
- ✓ Instructions are specific to template type (conversational/research/support)
- ✓ Data files are present and properly formatted (CSV/JSON/Markdown)
- ✓ Users can learn from generated examples without external docs
- ✓ All generated files validate against their respective schemas

---

## Phase 6: User Story 4 - Validate Project Structure

**Goal**: Initialization provides clear feedback on success/failure with helpful error messages

### Validation & Error Handling

- [ ] T064 [US4] Implement validation for all required directories exist in `ProjectInitializer.initialize()` in `src/holodeck/cli/utils/project_init.py`
- [ ] T065 [US4] Implement YAML syntax validation for agent.yaml before write in `TemplateRenderer.validate_agent_config()` in `src/holodeck/lib/template_engine.py`
- [ ] T066 [US4] Implement AgentConfig schema validation after YAML parse in `TemplateRenderer.validate_agent_config()` in `src/holodeck/lib/template_engine.py`
- [ ] T067 [US4] Create detailed error messages for validation failures (schema errors with line numbers) in `src/holodeck/lib/exceptions.py`
- [ ] T068 [US4] Implement success message showing all created files and paths in `src/holodeck/cli/commands/init.py`
- [ ] T069 [US4] Implement failure cleanup (remove partial directories) in `ProjectInitializer.initialize()` in `src/holodeck/cli/utils/project_init.py`

### Validation Tests (US4)

- [ ] T070 [US4] [P] Write test for valid project structure verification in `tests/integration/test_init_validation.py`
- [ ] T071 [US4] Write test for agent.yaml YAML syntax validation in `tests/integration/test_init_validation.py`
- [ ] T072 [US4] Write test for AgentConfig schema validation with invalid YAML rejection in `tests/integration/test_init_validation.py`
- [ ] T073 [US4] Write test for error message clarity and actionability in `tests/integration/test_init_validation.py`
- [ ] T074 [US4] Write test for partial cleanup on failure in `tests/integration/test_init_validation.py`

**Acceptance Criteria (US4)**:
- ✓ All required directories are created
- ✓ Generated agent.yaml is syntactically valid YAML
- ✓ Generated agent.yaml validates against AgentConfig schema
- ✓ Success message shows project location and next steps
- ✓ Error messages are clear, actionable, and include line numbers for YAML errors
- ✓ No partial projects left on disk after failures
- ✓ Validation happens before file write (no invalid projects created)

---

## Phase 7: User Story 5 - Specify Project Metadata

**Goal**: Developers can provide optional metadata (description, author) during initialization

### Metadata Support

- [ ] T075 [US5] Update `ProjectInitInput` model to include optional description and author fields in `src/holodeck/models/project_config.py`
- [ ] T076 [US5] Update `ProjectInitializer.validate_inputs()` to validate metadata fields (max length, valid characters) in `src/holodeck/cli/utils/project_init.py`
- [ ] T077 [US5] Update all template manifests to include description and author variables in manifest.yaml files
- [ ] T078 [US5] Update all `agent.yaml.j2` templates to include description and author fields in agent.yaml.j2 files
- [ ] T079 [US5] Update CLI command to accept --description and --author flags in `src/holodeck/cli/commands/init.py`
- [ ] T080 [US5] Pass metadata to template renderer in `ProjectInitializer.initialize()` in `src/holodeck/cli/utils/project_init.py`

### Metadata Tests (US5)

- [ ] T081 [US5] [P] Write test for --description flag in generated agent.yaml in `tests/integration/test_init_metadata.py`
- [ ] T082 [US5] Write test for --author flag in generated agent.yaml in `tests/integration/test_init_metadata.py`
- [ ] T083 [US5] Write test for metadata with special characters and escaping in `tests/integration/test_init_metadata.py`
- [ ] T084 [US5] Write test for missing metadata defaults (placeholder text) in `tests/integration/test_init_metadata.py`

**Acceptance Criteria (US5)**:
- ✓ `holodeck init <name> --description "text"` stores description in agent.yaml
- ✓ `holodeck init <name> --author "name"` stores author in agent.yaml
- ✓ Metadata is preserved in agent.yaml structure
- ✓ Missing metadata shows placeholder text (e.g., "TODO: Add description")
- ✓ Metadata validation prevents invalid characters
- ✓ Metadata appears in generated agent.yaml

---

## Phase 8: Polish & Cross-Cutting Concerns

**Goal**: Production-ready feature with comprehensive testing, documentation, and quality assurance

### Documentation & Help

- [ ] T085 Update CLI help text with examples: `holodeck init --help` in `src/holodeck/cli/commands/init.py`
- [ ] T086 Add version flag support: `holodeck --version` in `src/holodeck/cli/__init__.py`
- [ ] T087 Create QUICKSTART.md in repo root with user-facing getting started guide
- [ ] T088 Update README.md with `holodeck init` command documentation

### Quality Assurance

- [ ] T089 [P] Run full test suite and verify 80%+ coverage in `tests/`
- [ ] T090 Run linting checks: `make lint` (ruff, mypy, bandit) and fix all violations
- [ ] T091 Run type checking: `make type-check` and ensure all new code passes mypy strict mode
- [ ] T092 Run security checks: `make security` (bandit, safety) and verify no issues
- [ ] T093 [P] Run formatting check: `make format-check` and auto-format: `make format`
- [ ] T094 Run integration tests end-to-end with fresh Python environment

### Edge Cases & Error Scenarios

- [ ] T095 [P] Test with special characters in project name (should fail gracefully)
- [ ] T096 Test with very long project names (should truncate or reject)
- [ ] T097 [P] Test with read-only filesystem (should show permission error)
- [ ] T098 Test disk full scenario (should cleanup and show error)
- [ ] T099 [P] Test with corrupted template manifest (should show helpful error)
- [ ] T100 Test rapid consecutive init commands (concurrency/race conditions)

### Performance Validation

- [ ] T101 [P] Profile initialization time: target < 30 seconds per SC-001
- [ ] T102 Optimize template rendering if > 5 seconds per file
- [ ] T103 [P] Optimize file I/O if > 5 seconds total

### Pre-Release Checklist

- [ ] T104 Verify all user story acceptance criteria are met (US1-US5)
- [ ] T105 Run `make ci` locally and verify all checks pass
- [ ] T106 Create test project with each template and verify they're usable with `holodeck test`
- [ ] T107 Verify error messages are user-friendly (test with actual users if possible)
- [ ] T108 Update CHANGELOG.md with feature description and CLI usage
- [ ] T109 Verify documentation is complete and links work

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
  ├─ T001-T009: Sequential setup
  └─ Blocks: All subsequent phases

Phase 2 (Core Engine)
  ├─ T010-T018: Core logic (some parallelizable)
  ├─ T019-T020: Unit tests [P]
  └─ Blocks: All user story phases

Phase 3 (US1): Basic Creation
  ├─ T021-T029: CLI + default template
  ├─ T030-T032: Integration tests [P]
  └─ No blocks (independent from US2-5)

Phase 4 (US2): Templates
  ├─ T033-T046: Template development [P]
  └─ Depends on: US1 complete

Phase 5 (US3): Examples
  ├─ T048-T063: Example files [P]
  └─ Depends on: US2 complete

Phase 6 (US4): Validation
  ├─ T064-T074: Error handling [P]
  └─ Depends on: Phase 2 complete

Phase 7 (US5): Metadata
  ├─ T075-T084: Metadata support
  └─ Depends on: US1 complete

Phase 8 (Polish)
  ├─ T085-T109: Documentation, QA, testing
  └─ Depends on: US1-US5 complete
```

### Parallel Execution Opportunities

**Setup Phase (T001-T009)**:
- T002, T003, T005, T006, T008 can run in parallel (different files)
- Sequential: T001 → then others
- Estimated parallel time: ~2 days instead of 3 days

**Core Engine (T010-T020)**:
- T014-T017 can start after T010 (TemplateRenderer)
- T019-T020 can run after implementation (unit tests)
- 2 developers: one on ProjectInitializer, one on TemplateRenderer
- Estimated parallel time: ~5 days instead of 7 days

**Templates (T033-T046)**:
- Create 3 templates in parallel: research and customer-support simultaneously
- All 3 templates can be tested in parallel
- 2 developers: one on research, one on support (conversational baseline)
- Estimated parallel time: ~4 days instead of 6 days

**Examples (T048-T063)**:
- Generate examples for all 3 templates in parallel
- Tests can run in parallel
- Estimated parallel time: ~3 days instead of 4 days

### Recommended Team Structure

**Solo Developer** (40-50 hours):
1. Do Phase 1 (Setup) - 4 hours
2. Do Phase 2 (Core Engine) - 8 hours
3. Do Phase 3 (US1 Basic) - 8 hours
4. Do Phase 4 (US2 Templates) - 8 hours
5. Do Phase 5 (US3 Examples) - 6 hours
6. Do Phase 6 (US4 Validation) - 4 hours
7. Do Phase 7 (US5 Metadata) - 2 hours
8. Do Phase 8 (Polish) - 4 hours
Total: 44 hours (1 week full-time)

**2 Developers** (parallel, 25-30 hours each):
- Dev 1: Phase 1, 2 core (ProjectInitializer), 3, 4, 8
- Dev 2: Phase 2 core (TemplateRenderer), 5, 6, 7, templates
- Coordinate on Phase 1 setup and Phase 2 integration points
- Estimated: 2-3 weeks

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

## Testing Strategy

### Unit Tests (Foundation Layer)
- T019-T020: ProjectInitializer, TemplateRenderer logic
- Coverage target: 80%+
- Run: `pytest tests/unit/ -v`

### Integration Tests (Feature Layer)
- T030-T032: US1 basic creation
- T044-T047: US2 template selection
- T060-T063: US3 example files
- T070-T074: US4 validation
- T081-T084: US5 metadata
- Coverage target: 100% of user stories
- Run: `pytest tests/integration/ -v`

### Manual Tests (User Experience)
- T095-T100: Edge cases and error scenarios
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
