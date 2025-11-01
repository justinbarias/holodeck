# Implementation Tasks: Execute Agent Against Test Cases

**Branch**: `006-agent-test-execution`
**Spec**: [spec.md](./spec.md)
**Plan**: [plan.md](./plan.md)

## TDD Approach Overview

This task list follows **Test-Driven Development (TDD)** methodology. Each feature is implemented using the Red-Green-Refactor cycle:

### TDD Cycle

1. **[TEST] Red**: Write a failing test that defines the expected behavior
2. **[CODE] Green**: Write the minimal code to make the test pass
3. **[VERIFY] Refactor**: Run code quality checks (make format, make lint, make type-check)
4. **[VERIFY] Validate**: Run tests to ensure everything still passes

### Task Labels

- **[TEST]**: Write unit or integration tests (must fail initially)
- **[CODE]**: Implement code to make tests pass
- **[VERIFY]**: Run automated checks (make test-unit, make format, make lint, etc.)

### Benefits of TDD for This Project

- **Clear Requirements**: Tests document expected behavior before implementation
- **Regression Prevention**: Tests catch breaking changes immediately
- **Better Design**: Writing tests first encourages modular, testable code
- **Confidence**: High test coverage from day one (targeting 80%+)
- **Faster Debugging**: Failing tests pinpoint exact issues

### Workflow Example

```bash
# Step 1: Write test (should fail)
# Edit tests/unit/models/test_config.py - write test for ExecutionConfig

# Step 2: Run test (verify it fails - RED)
make test-unit

# Step 3: Implement code (minimal to pass)
# Edit src/holodeck/models/config.py - implement ExecutionConfig

# Step 4: Run test again (should pass - GREEN)
make test-unit

# Step 5: Refactor and verify
make format
make lint
make type-check
make test-unit  # Ensure still passing
```

## Task Summary

- **Total Tasks**: 148 (TDD approach doubles task count with TEST + CODE tasks)
- **Setup Tasks**: 10
- **Foundational Tasks**: 22 (11 TEST + 11 CODE)
- **User Story Tasks**: 95
  - US1 (P1): 38 tasks (19 TEST + 19 CODE/VERIFY)
  - US2 (P2): 16 tasks (8 TEST + 8 CODE/VERIFY)
  - US3 (P3): 9 tasks (4 TEST + 5 CODE/VERIFY)
  - US4 (P2): 16 tasks (8 TEST + 8 CODE/VERIFY)
  - US5 (P3): 16 tasks (8 TEST + 8 CODE/VERIFY)
- **Polish Tasks**: 21 (6 TEST + 15 CODE/VERIFY/DOC)

## Implementation Strategy

**MVP = User Story 1 (P1: Execute Basic Text-Only Test Cases)**

**TDD Workflow**: For each feature, follow the Red-Green-Refactor cycle:

1. **Red**: Write failing tests that define expected behavior
2. **Green**: Write minimal code to make tests pass
3. **Refactor**: Clean up code while keeping tests green
4. **Verify**: Run make format, make lint, make type-check

Each user story is independently testable and deliverable. User stories follow priority order (P1, P2, P3) and dependency constraints.

**Key TDD Principles Applied**:

- Write tests first, code second
- Each [TEST] task followed by corresponding [CODE] task
- Frequent verification with make test-unit and make test-integration
- Code quality checks after each implementation group

## Dependency Graph

```
Phase 1: Setup (T001-T010)
    ↓
Phase 2: Foundational (T011-T020) ← [BLOCKS ALL USER STORIES]
    ↓
    ├─→ Phase 3: US1 - Basic Text Execution (T021-T036) [P1 - MVP]
    │       ↓
    │       ├─→ Phase 4: US2 - Multimodal Files (T037-T043) [P2]
    │       ├─→ Phase 5: US3 - Per-Test Metrics (T044-T047) [P3]
    │       └─→ Phase 6: US4 - Progress Indicators (T048-T055) [P2]
    │
    └─→ Phase 7: US5 - Report Generation (T056-T062) [P3]
         ↓
Phase 8: Polish & QA (T063-T074)
```

**Critical Path**: Setup → Foundational → US1 → Polish

**Parallel Opportunities**:

- US2, US3, US4 can start after US1 completes (independent of each other)
- US5 requires only Foundational models (can run parallel to US1)

---

## Phase 1: Setup & Infrastructure

**Goal**: Initialize project structure, install dependencies, configure development environment

**TDD Approach**: Set up testing infrastructure before implementation

### Tasks

- [ ] T001 Install new dependencies in pyproject.toml (semantic-kernel[azure]>=1.37.0, markitdown[all]>=0.1.0, azure-ai-evaluation>=1.0.0, evaluate>=0.4.0, sacrebleu>=2.3.0, aiofiles>=23.0.0)
- [ ] T002 Create test fixtures directory structure (tests/fixtures/agents/, tests/fixtures/files/, tests/fixtures/expected_reports/)
- [ ] T003 Create lib/test_runner/ directory structure with **init**.py (stub files: executor.py, agent_bridge.py, progress.py, reporter.py)
- [ ] T004 Create lib/evaluators/ directory structure with **init**.py (stub files: base.py, azure_ai.py, nlp_metrics.py)
- [ ] T005 Create lib/file_processor.py stub module
- [ ] T006 Create cli/commands/test.py stub CLI command
- [ ] T007 Register test command in cli/main.py
- [ ] T008 Create models/test_result.py stub file for future models
- [ ] T009 Create stub for ExecutionConfig in models/config.py
- [ ] T010 Run make format && make lint to verify structure

---

## Phase 2: Foundational Components

**Goal**: Implement shared infrastructure needed by all user stories (blocking prerequisites)

**TDD Approach**: Write tests first for all models and core utilities, then implement to pass tests

**IMPORTANT**: All Phase 2 tasks must complete before ANY user story implementation begins.

### Tasks

#### T011-T016: Data Models (Test-First)

- [ ] T011 [TEST] Write unit tests for ExecutionConfig model in tests/unit/models/test_config.py (test field validation: file_timeout 1-300s, llm_timeout 1-600s, download_timeout 1-300s, cache_enabled, cache_dir, verbose, quiet constraints)
- [ ] T012 [CODE] Implement ExecutionConfig model in src/holodeck/models/config.py to pass T011 tests
- [ ] T013 [TEST] Write unit tests for ProcessedFileInput model in tests/unit/models/test_test_result.py (test fields: original, markdown_content, metadata, cached_path, processing_time_ms, error)
- [ ] T014 [CODE] Implement ProcessedFileInput model in src/holodeck/models/test_result.py to pass T013 tests
- [ ] T015 [TEST] Write unit tests for MetricResult model in tests/unit/models/test_test_result.py (test fields: metric_name, score, threshold, passed, scale, error, retry_count, evaluation_time_ms, model_used)
- [ ] T016 [CODE] Implement MetricResult model in src/holodeck/models/test_result.py to pass T015 tests
- [ ] T017 [TEST] Write unit tests for TestResult model in tests/unit/models/test_test_result.py (test fields: test_name, test_input, processed_files, agent_response, tool_calls, expected_tools, tools_matched, metric_results, ground_truth, passed, execution_time_ms, errors, timestamp)
- [ ] T018 [CODE] Implement TestResult model in src/holodeck/models/test_result.py to pass T017 tests
- [ ] T019 [TEST] Write unit tests for ReportSummary model in tests/unit/models/test_test_result.py (test fields: total_tests, passed, failed, pass_rate, total_duration_ms, metrics_evaluated, average_scores)
- [ ] T020 [CODE] Implement ReportSummary model in src/holodeck/models/test_result.py to pass T019 tests
- [ ] T021 [TEST] Write unit tests for TestReport model in tests/unit/models/test_test_result.py (test to_json() and to_file() methods, fields: agent_name, agent_config_path, results, summary, timestamp, holodeck_version, environment)
- [ ] T022 [CODE] Implement TestReport model in src/holodeck/models/test_result.py to pass T021 tests
- [ ] T023 [TEST] Write unit tests for Agent model update in tests/unit/models/test_agent.py (test execution: ExecutionConfig | None field)
- [ ] T024 [CODE] Update Agent model in src/holodeck/models/agent.py to pass T023 tests

#### T025-T028: File Processor (Test-First)

- [ ] T025 [TEST] Write unit tests for file processor in tests/unit/lib/test_file_processor.py (test markitdown integration for PDF, images, Excel, Word, PowerPoint, CSV, HTML)
- [ ] T026 [CODE] Implement file processor using markitdown in src/holodeck/lib/file_processor.py to pass T025 tests
- [ ] T027 [TEST] Write unit tests for file caching in tests/unit/lib/test_file_processor.py (test .holodeck/cache/ directory creation, hash-based cache keys, cache hit/miss)
- [ ] T028 [CODE] Implement file caching logic in src/holodeck/lib/file_processor.py to pass T027 tests
- [ ] T029 [TEST] Write unit tests for remote URL download in tests/unit/lib/test_file_processor.py (test timeout, 3 retries with exponential backoff, error handling)
- [ ] T030 [CODE] Implement remote URL download with timeout in src/holodeck/lib/file_processor.py to pass T029 tests
- [ ] T031 [VERIFY] Run make test-unit to verify all Phase 2 tests pass
- [ ] T032 [VERIFY] Run make format && make lint && make type-check

---

## Phase 3: User Story 1 - Execute Basic Text-Only Test Cases (P1)

**Story Goal**: Execute text-based test cases against agents and display pass/fail status with metric scores

**TDD Approach**: Write tests for each component before implementation, following Red-Green-Refactor cycle

**Independent Test**: Create agent.yaml with 3 simple text test cases, run `holodeck test agent.yaml`, verify:

- Sequential execution
- Agent responses captured
- Evaluation metrics calculated
- Pass/fail results displayed

**Dependencies**: Requires Phase 2 (Foundational) completion

### Tasks

#### T033-T042: Evaluators (Test-First)

- [ ] T033 [TEST] Write unit tests for base evaluator in tests/unit/lib/evaluators/test_base.py (test abstract evaluate() method, timeout handling, retry logic)
- [ ] T034 [CODE] Implement base evaluator interface in src/holodeck/lib/evaluators/base.py to pass T033 tests
- [ ] T035 [TEST] Write unit tests for Azure AI evaluators in tests/unit/lib/evaluators/test_azure_ai.py (test GroundednessEvaluator, RelevanceEvaluator, CoherenceEvaluator, FluencyEvaluator, SimilarityEvaluator with per-metric model config)
- [ ] T036 [CODE] Implement Azure AI Evaluation SDK integration in src/holodeck/lib/evaluators/azure_ai.py to pass T035 tests
- [ ] T037 [TEST] Write unit tests for retry logic in tests/unit/lib/evaluators/test_azure_ai.py (test 3 attempts with exponential backoff 2s/4s/8s, LLM API errors, timeouts)
- [ ] T038 [CODE] Implement metric evaluation retry logic in src/holodeck/lib/evaluators/azure_ai.py to pass T037 tests
- [ ] T039 [TEST] Write unit tests for NLP metrics in tests/unit/lib/evaluators/test_nlp_metrics.py (test BLEU, ROUGE, METEOR, F1 using evaluate.load() from Hugging Face)
- [ ] T040 [CODE] Implement NLP metrics in src/holodeck/lib/evaluators/nlp_metrics.py to pass T039 tests
- [ ] T041 [VERIFY] Run make test-unit to verify evaluator tests pass
- [ ] T042 [VERIFY] Run make format && make lint

#### T043-T050: Agent Bridge (Test-First)

- [ ] T043 [TEST] Write unit tests for agent bridge in tests/unit/lib/test_runner/test_agent_bridge.py (test Semantic Kernel integration, Kernel creation, agent config loading, ChatHistory invocation, response capture, tool_calls capture)
- [ ] T044 [CODE] Implement Semantic Kernel agent bridge in src/holodeck/lib/test_runner/agent_bridge.py to pass T043 tests
- [ ] T045 [VERIFY] Run make test-unit to verify agent bridge tests pass
- [ ] T046 [VERIFY] Run make format && make lint

#### T047-T056: Test Executor (Test-First)

- [ ] T047 [TEST] Write unit tests for configuration resolution in tests/unit/lib/test_runner/test_executor.py (test CLI > agent.yaml > env > defaults priority, ExecutionConfig merge)
- [ ] T048 [CODE] Implement configuration resolution in src/holodeck/lib/test_runner/executor.py to pass T047 tests
- [ ] T049 [TEST] Write unit tests for tool call validation in tests/unit/lib/test_runner/test_executor.py (test expected_tools matching, TestResult.tool_calls vs TestCaseModel.expected_tools)
- [ ] T050 [CODE] Implement tool call validation in src/holodeck/lib/test_runner/executor.py to pass T049 tests
- [ ] T051 [TEST] Write unit tests for timeout handling in tests/unit/lib/test_runner/test_executor.py (test file: 30s, LLM: 60s, download: 30s defaults using asyncio.timeout or threading.Timer)
- [ ] T052 [CODE] Implement timeout handling in src/holodeck/lib/test_runner/executor.py to pass T051 tests
- [ ] T053 [TEST] Write unit tests for test executor main flow in tests/unit/lib/test_runner/test_executor.py (test load AgentConfig, execute tests sequentially, collect TestResult instances, generate TestReport)
- [ ] T054 [CODE] Implement test executor in src/holodeck/lib/test_runner/executor.py to pass T053 tests
- [ ] T055 [VERIFY] Run make test-unit to verify executor tests pass
- [ ] T056 [VERIFY] Run make format && make lint

#### T057-T062: Progress Indicators (Test-First)

- [ ] T057 [TEST] Write unit tests for progress indicators in tests/unit/lib/test_runner/test_progress.py (test TTY detection with sys.stdout.isatty(), "Test X/Y" display, checkmarks/X marks, CI/CD plain text mode)
- [ ] T058 [CODE] Implement progress indicators in src/holodeck/lib/test_runner/progress.py to pass T057 tests
- [ ] T059 [VERIFY] Run make test-unit to verify progress tests pass
- [ ] T060 [VERIFY] Run make format && make lint

#### T061-T068: CLI Command (Test-First)

- [ ] T061 [TEST] Write unit tests for CLI command in tests/unit/cli/commands/test_test.py (test argument parsing for AGENT_CONFIG, option handling for --output, --format, --verbose, --quiet, --timeout flags)
- [ ] T062 [CODE] Implement CLI command in src/holodeck/cli/commands/test.py to pass T061 tests
- [ ] T063 [TEST] Write unit tests for exit code logic in tests/unit/cli/commands/test_test.py (test 0=success, 1=test failure, 2=config error, 3=execution error, 4=evaluation error)
- [ ] T064 [CODE] Implement exit code logic in src/holodeck/cli/commands/test.py to pass T063 tests
- [ ] T065 [VERIFY] Run make test-unit to verify CLI tests pass
- [ ] T066 [VERIFY] Run make format && make lint

#### T067-T070: Integration Testing

- [ ] T067 [TEST] Create sample test agent.yaml in tests/fixtures/agents/test_agent.yaml with 3 simple text test cases
- [ ] T068 [TEST] Write integration test for basic text execution in tests/integration/test_basic_execution.py (verify end-to-end test run with mocked LLM responses)
- [ ] T069 [VERIFY] Run make test-integration to verify integration tests pass
- [ ] T070 [VERIFY] Run make test to verify all tests pass

---

## Phase 4: User Story 2 - Execute Multimodal Test Cases with Files (P2)

**Story Goal**: Execute test cases with attached files (PDF, images, Office documents) and provide file content to agent

**TDD Approach**: Write tests for file processing features, then implement to pass tests

**Independent Test**: Create test cases with PDF/image/Excel files, run tests, verify:

- Files processed via markitdown
- Content extracted and provided to agent
- Agent responses reference file content
- Tests execute successfully

**Dependencies**: Requires US1 (basic test execution infrastructure)

### Tasks

#### T071-T080: File Processing Extensions (Test-First)

- [ ] T071 [TEST] Write unit tests for page/sheet/range extraction in tests/unit/lib/test_file_processor.py (test FileInput.pages for PDF, FileInput.sheet for Excel, FileInput.range for PowerPoint, preprocessing before markitdown)
- [ ] T072 [CODE] Implement page/sheet/range extraction in src/holodeck/lib/file_processor.py to pass T071 tests
- [ ] T073 [TEST] Write unit tests for file size warnings in tests/unit/lib/test_file_processor.py (test >100MB file warning message, verify processing continues)
- [ ] T074 [CODE] Implement file size warning logic in src/holodeck/lib/file_processor.py to pass T073 tests
- [ ] T075 [TEST] Write unit tests for file processing error handling in tests/unit/lib/test_file_processor.py (test timeout, malformed files, ProcessedFileInput.error population, test continuation)
- [ ] T076 [CODE] Implement file processing error handling in src/holodeck/lib/file_processor.py to pass T075 tests
- [ ] T077 [VERIFY] Run make test-unit to verify file processor tests pass
- [ ] T078 [VERIFY] Run make format && make lint

#### T079-T084: Executor Integration (Test-First)

- [ ] T079 [TEST] Write unit tests for file processor integration in tests/unit/lib/test_runner/test_executor.py (test process files before agent invocation, verify ProcessedFileInput.markdown_content in agent context)
- [ ] T080 [CODE] Integrate file processor with test executor in src/holodeck/lib/test_runner/executor.py to pass T079 tests
- [ ] T081 [VERIFY] Run make test-unit to verify executor integration tests pass
- [ ] T082 [VERIFY] Run make format && make lint

#### T083-T086: Integration Testing

- [ ] T083 [TEST] Create sample multimodal test files in tests/fixtures/files/ (sample.pdf, sample.jpg, sample.xlsx, sample.docx, sample.pptx with known content)
- [ ] T084 [TEST] Write integration test for multimodal execution in tests/integration/test_multimodal_execution.py (verify file processing and agent receives markdown content)
- [ ] T085 [VERIFY] Run make test-integration to verify multimodal tests pass
- [ ] T086 [VERIFY] Run make test to verify all tests pass

---

## Phase 5: User Story 3 - Execute Tests with Per-Test Metric Configuration (P3)

**Story Goal**: Allow test cases to specify their own evaluation metrics, overriding global defaults

**TDD Approach**: Write tests for metric resolution logic before implementation

**Independent Test**: Create test cases with varying metric configurations, verify:

- Per-test metrics override global metrics
- Tests without metrics use global defaults
- Invalid metric references raise errors

**Dependencies**: Requires US1 (evaluation infrastructure)

### Tasks

#### T087-T092: Per-Test Metrics (Test-First)

- [ ] T087 [TEST] Write unit tests for per-test metric resolution in tests/unit/lib/test_runner/test_executor.py (test TestCaseModel.evaluations override, test fallback to AgentConfig.evaluations.metrics)
- [ ] T088 [CODE] Implement per-test metric resolution logic in src/holodeck/lib/test_runner/executor.py to pass T087 tests
- [ ] T089 [TEST] Write unit tests for metric validation in tests/unit/lib/test_runner/test_executor.py (test undefined metric raises ConfigError, test valid metrics pass)
- [ ] T090 [CODE] Implement metric validation in src/holodeck/lib/test_runner/executor.py to pass T089 tests
- [ ] T091 [VERIFY] Run make test-unit to verify metric resolution tests pass
- [ ] T092 [VERIFY] Run make format && make lint

#### T093-T096: Integration Testing

- [ ] T093 [TEST] Write integration test for per-test metrics in tests/integration/test_evaluation_metrics.py (verify metric override behavior, test global defaults)
- [ ] T094 [VERIFY] Run make test-integration to verify metric tests pass
- [ ] T095 [VERIFY] Run make test to verify all tests pass

---

## Phase 6: User Story 4 - Display Test Results with Progress Indicators (P2)

**Story Goal**: Show real-time progress with visual indicators during test execution

**TDD Approach**: Write tests for display features before implementation

**Independent Test**: Run 10 test cases, observe console output, verify:

- Progress indicators update in real-time
- Checkmarks/X marks appear for pass/fail
- Final summary displayed
- CI/CD compatibility (no interactive elements in non-TTY)

**Dependencies**: Requires US1 (basic execution)

### Tasks

#### T096-T114: Enhanced Progress Display (Test-First)

- [ ] T096 [TEST] Write unit tests for TTY detection in tests/unit/lib/test_runner/test_progress.py (test sys.stdout.isatty() detection, test TTY vs non-TTY behavior)
- [ ] T097 [CODE] Implement TTY detection in src/holodeck/lib/test_runner/progress.py to pass T096 tests
- [ ] T098 [TEST] Write unit tests for progress indicator display in tests/unit/lib/test_runner/test_progress.py (test "Test X/Y" format, test update on each test start)
- [ ] T099 [CODE] Implement progress indicator display in src/holodeck/lib/test_runner/progress.py to pass T098 tests
- [ ] T100 [TEST] Write unit tests for pass/fail symbols in tests/unit/lib/test_runner/test_progress.py (test ✅/❌ symbols, test ANSI color codes for TTY, test plain text for non-TTY)
- [ ] T101 [CODE] Implement pass/fail symbols with color support in src/holodeck/lib/test_runner/progress.py to pass T100 tests
- [ ] T102 [TEST] Write unit tests for spinner in tests/unit/lib/test_runner/test_progress.py (test spinner for long-running tests >5s, test spinner char rotation during execution)
- [ ] T103 [CODE] Implement spinner for long-running tests in src/holodeck/lib/test_runner/progress.py to pass T102 tests
- [ ] T104 [TEST] Write unit tests for summary display in tests/unit/lib/test_runner/test_progress.py (test total/passed/failed/pass rate display after all tests complete)
- [ ] T105 [CODE] Implement summary display in src/holodeck/lib/test_runner/progress.py to pass T104 tests
- [ ] T106 [TEST] Write unit tests for quiet mode in tests/unit/lib/test_runner/test_progress.py (test --quiet flag suppresses progress, test only final summary shown)
- [ ] T107 [CODE] Implement quiet mode output in src/holodeck/lib/test_runner/progress.py to pass T106 tests
- [ ] T108 [TEST] Write unit tests for verbose mode in tests/unit/lib/test_runner/test_progress.py (test --verbose flag shows debug info, stack traces, timing)
- [ ] T109 [CODE] Implement verbose mode output in src/holodeck/lib/test_runner/progress.py to pass T108 tests
- [ ] T110 [VERIFY] Run make test-unit to verify progress tests pass
- [ ] T111 [VERIFY] Run make format && make lint

---

## Phase 7: User Story 5 - Generate Test Report Files (P3)

**Story Goal**: Generate detailed test reports in JSON or Markdown format

**TDD Approach**: Write tests for report generation before implementation

**Independent Test**: Run tests with --output flag, verify:

- JSON file created with complete test data
- Markdown file created with human-readable format
- Report structure matches expected schema

**Dependencies**: Requires US1 (test results data structure)

### Tasks

#### T112-T122: Report Generation (Test-First)

- [ ] T112 [TEST] Create expected report fixtures in tests/fixtures/expected_reports/ (sample_json_report.json, sample_markdown_report.md with known test data)
- [ ] T113 [TEST] Write unit tests for JSON report generation in tests/unit/lib/test_runner/test_reporter.py (test TestReport.to_json() method, verify structure matches schema)
- [ ] T114 [CODE] Implement JSON report generation in src/holodeck/lib/test_runner/reporter.py to pass T113 tests
- [ ] T115 [TEST] Write unit tests for Markdown report generation in tests/unit/lib/test_runner/test_reporter.py (test TestReport.to_markdown() with formatted tables, summary, test details)
- [ ] T116 [CODE] Implement Markdown report generation in src/holodeck/lib/test_runner/reporter.py to pass T115 tests
- [ ] T117 [TEST] Write unit tests for report file writing in tests/unit/lib/test_runner/test_reporter.py (test TestReport.to_file() method)
- [ ] T118 [CODE] Implement report file writing logic in src/holodeck/lib/test_runner/reporter.py to pass T117 tests
- [ ] T119 [VERIFY] Run make test-unit to verify reporter tests pass
- [ ] T120 [VERIFY] Run make format && make lint

#### T121-T126: CLI Format Detection (Test-First)

- [ ] T121 [TEST] Write unit tests for format auto-detection in tests/unit/cli/commands/test_test.py (test .json vs .md extension detection, test --format flag override)
- [ ] T122 [CODE] Implement format auto-detection in src/holodeck/cli/commands/test.py to pass T121 tests
- [ ] T123 [VERIFY] Run make test-unit to verify format detection tests pass
- [ ] T124 [VERIFY] Run make format && make lint

#### T125-T128: Integration Testing

- [ ] T125 [TEST] Write integration test for report generation in tests/integration/test_report_generation.py (verify JSON and Markdown output match expected format)
- [ ] T126 [VERIFY] Run make test-integration to verify report generation tests pass
- [ ] T127 [VERIFY] Run make test to verify all tests pass

---

## Phase 8: Polish & Cross-Cutting Concerns

**Goal**: Final refinements, error handling, documentation, and production readiness

**TDD Approach**: Write tests for error handling and edge cases, then implement

### Tasks

#### T128-T136: Error Handling (Test-First)

- [ ] T128 [TEST] Write unit tests for structured error messages in tests/unit/lib/test_runner/test_executor.py (test error format: "ERROR: {summary}\n Cause: {cause}\n Suggestion: {action}")
- [ ] T129 [CODE] Implement structured error messages in src/holodeck/lib/test_runner/executor.py to pass T128 tests
- [ ] T130 [TEST] Write unit tests for large file warnings in tests/unit/lib/test_file_processor.py (test warning when file >100MB before processing)
- [ ] T131 [CODE] Implement warning messages for large files in src/holodeck/lib/file_processor.py to pass T130 tests
- [ ] T132 [VERIFY] Run make test-unit to verify error handling tests pass
- [ ] T133 [VERIFY] Run make format && make lint

#### T134-T138: Configuration Defaults (Test-First)

- [ ] T134 [TEST] Write unit tests for configuration defaults in tests/unit/config/test_defaults.py (test file_timeout=30, llm_timeout=60, download_timeout=30, cache_dir=".holodeck/cache", cache_enabled=True)
- [ ] T135 [CODE] Add configuration defaults to src/holodeck/config/defaults.py to pass T134 tests
- [ ] T136 [VERIFY] Run make test-unit to verify defaults tests pass
- [ ] T137 [VERIFY] Run make format && make lint

#### T138-T148: Documentation & Quality Assurance

- [ ] T138 Create comprehensive docstrings for all new modules (file_processor.py, executor.py, agent_bridge.py, progress.py, reporter.py, azure_ai.py, nlp_metrics.py)
- [ ] T139 Update docsite (/docs) with new dependencies and modules (document test execution framework, new lib/ structure, ExecutionConfig model)
- [ ] T140 Run make format to format all new code
- [ ] T141 Run make lint to check code quality
- [ ] T142 Run make type-check to verify type hints
- [ ] T143 Run make security to scan for vulnerabilities
- [ ] T144 Run make test-coverage to verify 80% coverage minimum
- [ ] T145 Review coverage report and add missing tests if needed
- [ ] T146 Update pyproject.toml with version bump (0.1.0)
- [ ] T147 Create usage examples in quickstart.md documentation (already exists, verify completeness)
- [ ] T148 Run full CI pipeline: make ci

---

## Parallel Execution Opportunities (TDD Context)

**Note**: In TDD, [TEST] tasks must complete before their corresponding [CODE] tasks, but different test/code pairs can run in parallel.

### Phase 1 (Setup)

**All tasks can run in parallel** - different files being created

**Suggested groups**:

- Group A: T001-T005 (dependencies and directory structure)
- Group B: T006-T009 (CLI and model stubs)
- Group C: T010 (verification)

### Phase 2 (Foundational)

**Test/code pairs for different models can run in parallel**:

- Group A [P]: T011-T024 (model test/code pairs, ensure T011→T012, T013→T014, etc.)
- Group B [P]: T025-T030 (file processor test/code pairs, ensure T025→T026, T027→T028, etc.)
- Sequential: T031-T032 (verification after all implementations)

**TDD Workflow**: Write all model tests first (T011, T013, T015, T017, T019, T021, T023), then implement models (T012, T014, T016, T018, T020, T022, T024)

### Phase 3 (US1)

**Test/code pairs for different components can run in parallel**:

- Group A [P]: T033-T042 (evaluator test/code pairs)
- Group B [P]: T043-T046 (agent bridge test/code pairs)
- Group C [P]: T047-T056 (executor test/code pairs)
- Group D [P]: T057-T060 (progress test/code pairs)
- Group E [P]: T061-T066 (CLI test/code pairs)
- Sequential: T067-T070 (integration tests and final verification)

**TDD Workflow**: Within each group, write tests before implementation (e.g., T033→T034, T035→T036)

### Phase 4 (US2)

**Test/code pairs for file processing**:

- Group A [P]: T071-T078 (file processing test/code pairs)
- Group B [P]: T079-T082 (executor integration test/code pairs)
- Sequential: T083-T086 (integration tests and verification)

### Phase 5 (US3)

**Test/code pairs for metric resolution**:

- Group A [P]: T087-T092 (metric resolution test/code pairs)
- Sequential: T093-T095 (integration tests and verification)

### Phase 6 (US4)

**Test/code pairs for progress display**:

- Group A [P]: T096-T111 (all progress feature test/code pairs)

**TDD Workflow**: Write all progress tests first (T096, T098, T100, T102, T104, T106, T108), then implement (T097, T099, T101, T103, T105, T107, T109)

### Phase 7 (US5)

**Test/code pairs for report generation**:

- Group A [P]: T112-T120 (reporter test/code pairs)
- Group B [P]: T121-T124 (CLI format detection test/code pairs)
- Sequential: T125-T127 (integration tests and verification)

### Phase 8 (Polish)

**Test/code pairs for error handling and defaults**:

- Group A [P]: T128-T133 (error handling test/code pairs)
- Group B [P]: T134-T137 (configuration defaults test/code pairs)
- Sequential: T138-T148 (documentation and quality assurance)

---

## Validation Checklist

For each user story phase:

- [x] Story goal clearly stated
- [x] Independent test criteria defined
- [x] All required components identified (models, services, CLI, tests)
- [x] Dependencies on other stories documented
- [x] Tasks include exact file paths
- [x] Parallel opportunities marked with [P]
- [x] Story labels ([US1], [US2], etc.) applied correctly
- [x] Total task count and breakdown by phase provided
- [x] Exit codes defined (0=success, 1=test failure, 2=config error, 3=execution error, 4=evaluation error)

---

## Testing Strategy

### Unit Tests (34 test files expected)

**Models**:

- tests/unit/models/test_test_result.py (ProcessedFileInput, MetricResult, TestResult, ReportSummary, TestReport)
- tests/unit/models/test_config.py (ExecutionConfig validation)

**Lib Components**:

- tests/unit/lib/test_file_processor.py (markitdown integration, caching, range extraction)
- tests/unit/lib/test_runner/test_executor.py (config resolution, test execution flow, timeout handling)
- tests/unit/lib/test_runner/test_agent_bridge.py (Semantic Kernel integration)
- tests/unit/lib/test_runner/test_progress.py (TTY detection, output formatting)
- tests/unit/lib/test_runner/test_reporter.py (JSON/Markdown generation)
- tests/unit/lib/evaluators/test_azure_ai.py (Azure AI Evaluation SDK)
- tests/unit/lib/evaluators/test_nlp_metrics.py (BLEU, ROUGE, METEOR, F1)

**CLI**:

- tests/unit/cli/commands/test_test.py (CLI argument parsing, option handling)

### Integration Tests (4 test files)

- tests/integration/test_basic_execution.py (US1: end-to-end text test execution)
- tests/integration/test_multimodal_execution.py (US2: file processing and agent context)
- tests/integration/test_evaluation_metrics.py (US3: per-test metric resolution)
- tests/integration/test_report_generation.py (US5: JSON and Markdown output)

---

## Success Criteria

**From spec.md**:

- [ ] SC-001: Execute 10 test cases in <30s (excluding LLM latency)
- [ ] SC-002: 100% test case pass/fail reporting
- [ ] SC-003: 100% accuracy in expected_tools validation
- [ ] SC-004: Support 5+ file types (PDF, image, Excel, Word, PowerPoint)
- [ ] SC-005: 100% of test results include all required fields
- [ ] SC-006: Graceful metric failure handling (continue execution)
- [ ] SC-007: Real-time progress indicators update per test
- [ ] SC-008: Valid JSON/Markdown reports parseable by standard tools
- [ ] SC-009: Correct exit codes for CI/CD (0=success, 1=failure, 2=config error, 3=execution error, 4=evaluation error)
- [ ] SC-010: 90% of errors include structured, actionable messages

---

## Configuration Hierarchy Reminder

```
CLI Flags  >  agent.yaml execution  >  Environment Variables  >  Built-in Defaults
(highest priority)                                              (lowest priority)
```

**Example**:

- Built-in default: `file_timeout=30s`
- Environment variable: `HOLODECK_FILE_TIMEOUT=45`
- agent.yaml: `execution.file_timeout: 60`
- CLI flag: `--file-timeout 90`
- **Resolved**: `90s` (CLI wins)

---

## File Paths Reference

**New Files**:

- `src/holodeck/models/test_result.py` (T002)
- `src/holodeck/models/config.py` (T003, T011)
- `src/holodeck/lib/file_processor.py` (T005, T018-T020, T037-T040)
- `src/holodeck/lib/test_runner/executor.py` (T006, T025, T027-T030, T039, T044-T045, T063)
- `src/holodeck/lib/test_runner/agent_bridge.py` (T006, T021)
- `src/holodeck/lib/test_runner/progress.py` (T006, T026, T048-T054)
- `src/holodeck/lib/test_runner/reporter.py` (T006, T056-T058)
- `src/holodeck/lib/evaluators/base.py` (T007, T024)
- `src/holodeck/lib/evaluators/azure_ai.py` (T007, T022, T029)
- `src/holodeck/lib/evaluators/nlp_metrics.py` (T007, T023)
- `src/holodeck/cli/commands/test.py` (T008, T031-T032, T059)

**Modified Files**:

- `pyproject.toml` (T001, T073)
- `src/holodeck/models/agent.py` (T004, T017)
- `src/holodeck/cli/main.py` (T009)
- `src/holodeck/config/defaults.py` (T065)
- `CLAUDE.md` (T067)

**Test Files**:

- 34+ test files across unit and integration directories

---

## Notes

- **Semantic Kernel**: Agent execution uses `semantic_kernel.agents.Agent` with `ChatHistory` for context
- **markitdown**: Unified file processor handles all file types with `.convert()` method
- **Azure AI Evaluation**: Per-metric model configuration for cost optimization (GPT-4o for critical metrics, GPT-4o-mini for general)
- **Retry Logic**: 3 attempts with exponential backoff (2s/4s/8s for LLM, 1s/2s/4s for downloads)
- **Caching**: Remote files cached in `.holodeck/cache/` with hash-based keys
- **Exit Codes**: 0=success, 1=test failure, 2=config error, 3=execution error, 4=evaluation error
- **TTY Detection**: Use `sys.stdout.isatty()` for progress indicator behavior
- **Configuration Merge**: CLI > agent.yaml > env vars > defaults

---

**End of Tasks Document**
