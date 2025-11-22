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

- **Total Tasks**: 164 (TDD approach with TEST + CODE + VERIFY tasks)
- **Setup Tasks**: 10 (T001-T010)
- **Foundational Tasks**: 20 (T011-T030)
- **User Story 1 (P1 - Execute Basic Text-Only Tests)**: 49 tasks
  - T057-T060: Progress Indicators (4 tasks) ✅ **COMPLETED**
  - T061-T066: Executor Integration with Callback Pattern (6 NEW tasks)
  - T067-T074: CLI Command (8 tasks)
  - T075-T078: Integration Testing (4 tasks)
  - T043-T056, T031-T042: Evaluators, Agent Bridge, Test Executor (24 tasks) ✅ **COMPLETED**
  - T104-T122: Enhanced Progress Display & Callback Integration (19 tasks)
- **User Story 2 (P2 - Multimodal Files)**: 14 tasks (T079-T094)
- **User Story 3 (P3 - Per-Test Metrics)**: 9 tasks (T095-T103)
- **User Story 4 (P2 - Progress Display)**: 22 tasks (T104-T122, overlaps with US1)
- **User Story 5 (P3 - Report Files)**: 16 tasks (T123-T139)
- **Polish & QA Tasks**: 16 tasks (T140-T164)

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

- [x] T001 Install new dependencies in pyproject.toml (semantic-kernel[azure]>=1.37.0, markitdown[all]>=0.1.0, azure-ai-evaluation>=1.0.0, evaluate>=0.4.0, ~sacrebleu>=2.3.0~, aiofiles>=23.0.0)
- [x] T002 Create test fixtures directory structure (tests/fixtures/agents/, tests/fixtures/files/, tests/fixtures/expected_reports/)
- [x] T003 Create lib/test_runner/ directory structure with **init**.py (stub files: executor.py, agent_bridge.py, progress.py, reporter.py)
- [x] T004 Create lib/evaluators/ directory structure with **init**.py (stub files: base.py, azure_ai.py, nlp_metrics.py)
- [x] T005 Create lib/file_processor.py stub module
- [x] T006 Create cli/commands/test.py stub CLI command
- [x] T007 Register test command in cli/main.py
- [x] T008 Create models/test_result.py stub file for future models
- [x] T009 Create stub for ExecutionConfig in models/config.py
- [x] T010 Run make format && make lint to verify structure

---

## Phase 2: Foundational Components

**Goal**: Implement shared infrastructure needed by all user stories (blocking prerequisites)

**TDD Approach**: Write tests first for all models and core utilities, then implement to pass tests

**IMPORTANT**: All Phase 2 tasks must complete before ANY user story implementation begins.

### Tasks

#### T011-T016: Data Models (Test-First)

- [x] T011 [TEST] Write unit tests for ExecutionConfig model in tests/unit/models/test_config.py (test field validation: file_timeout 1-300s, llm_timeout 1-600s, download_timeout 1-300s, cache_enabled, cache_dir, verbose, quiet constraints)
- [x] T012 [CODE] Implement ExecutionConfig model in src/holodeck/models/config.py to pass T011 tests
- [x] T013 [TEST] Write unit tests for ProcessedFileInput model in tests/unit/models/test_test_result.py (test fields: original, markdown_content, metadata, cached_path, processing_time_ms, error)
- [x] T014 [CODE] Implement ProcessedFileInput model in src/holodeck/models/test_result.py to pass T013 tests
- [x] T015 [TEST] Write unit tests for MetricResult model in tests/unit/models/test_test_result.py (test fields: metric_name, score, threshold, passed, scale, error, retry_count, evaluation_time_ms, model_used)
- [x] T016 [CODE] Implement MetricResult model in src/holodeck/models/test_result.py to pass T015 tests
- [x] T017 [TEST] Write unit tests for TestResult model in tests/unit/models/test_test_result.py (test fields: test_name, test_input, processed_files, agent_response, tool_calls, expected_tools, tools_matched, metric_results, ground_truth, passed, execution_time_ms, errors, timestamp)
- [x] T018 [CODE] Implement TestResult model in src/holodeck/models/test_result.py to pass T017 tests
- [x] T019 [TEST] Write unit tests for ReportSummary model in tests/unit/models/test_test_result.py (test fields: total_tests, passed, failed, pass_rate, total_duration_ms, metrics_evaluated, average_scores)
- [x] T020 [CODE] Implement ReportSummary model in src/holodeck/models/test_result.py to pass T019 tests
- [x] T021 [TEST] Write unit tests for TestReport model in tests/unit/models/test_test_result.py (test to_json() and to_file() methods, fields: agent_name, agent_config_path, results, summary, timestamp, holodeck_version, environment)
- [x] T022 [CODE] Implement TestReport model in src/holodeck/models/test_result.py to pass T021 tests
- [x] T023 [TEST] Write unit tests for Agent model update in tests/unit/models/test_agent.py (test execution: ExecutionConfig | None field)
- [x] T024 [CODE] Update Agent model in src/holodeck/models/agent.py to pass T023 tests

#### T025-T028: File Processor (Test-First)

- [x] T025 [TEST] Write unit tests for file processor in tests/unit/lib/test_file_processor.py (test markitdown integration for PDF, images, Excel, Word, PowerPoint, CSV, HTML)
- [x] T026 [CODE] Implement file processor using markitdown in src/holodeck/lib/file_processor.py to pass T025 tests
- [x] T027 [TEST] Write unit tests for file caching in tests/unit/lib/test_file_processor.py (test .holodeck/cache/ directory creation, hash-based cache keys, cache hit/miss)
- [x] T028 [CODE] Implement file caching logic in src/holodeck/lib/file_processor.py to pass T027 tests
- [x] T029 [TEST] Write unit tests for remote URL download in tests/unit/lib/test_file_processor.py (test timeout, 3 retries with exponential backoff, error handling)
- [x] T030 [CODE] Implement remote URL download with timeout in src/holodeck/lib/file_processor.py to pass T029 tests
- [x] T031 [VERIFY] Run make test-unit to verify all Phase 2 tests pass
- [x] T032 [VERIFY] Run make format && make lint && make type-check

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

- [x] T033 [TEST] Write unit tests for base evaluator in tests/unit/lib/evaluators/test_base.py (test abstract evaluate() method, timeout handling, retry logic)
- [x] T034 [CODE] Implement base evaluator interface in src/holodeck/lib/evaluators/base.py to pass T033 tests
- [x] T035 [TEST] Write unit tests for Azure AI evaluators in tests/unit/lib/evaluators/test_azure_ai.py (test GroundednessEvaluator, RelevanceEvaluator, CoherenceEvaluator, FluencyEvaluator, SimilarityEvaluator with per-metric model config)
- [x] T036 [CODE] Implement Azure AI Evaluation SDK integration in src/holodeck/lib/evaluators/azure_ai.py to pass T035 tests
- [x] T037 [TEST] Write unit tests for retry logic in tests/unit/lib/evaluators/test_azure_ai.py (test 3 attempts with exponential backoff 2s/4s/8s, LLM API errors, timeouts)
- [x] T038 [CODE] Implement metric evaluation retry logic in src/holodeck/lib/evaluators/azure_ai.py to pass T037 tests
- [x] T039 [TEST] Write unit tests for NLP metrics in tests/unit/lib/evaluators/test_nlp_metrics.py (test BLEU, ROUGE, METEOR, F1 using evaluate.load() from Hugging Face)
- [x] T040 [CODE] Implement NLP metrics in src/holodeck/lib/evaluators/nlp_metrics.py to pass T039 tests
- [x] T041 [VERIFY] Run make test-unit to verify evaluator tests pass
- [x] T042 [VERIFY] Run make format && make lint

#### T043-T050: Agent Bridge (Test-First)

- [x] T043 [TEST] Write unit tests for agent bridge in tests/unit/lib/test_runner/test_agent_bridge.py (test Semantic Kernel integration, Kernel creation, agent config loading, ChatHistory invocation, response capture, tool_calls capture)
- [x] T044 [CODE] Implement Semantic Kernel agent bridge in src/holodeck/lib/test_runner/agent_bridge.py to pass T043 tests
- [x] T045 [VERIFY] Run make test-unit to verify agent bridge tests pass
- [x] T046 [VERIFY] Run make format && make lint

#### T047-T056: Test Executor (Test-First)

- [x] T047 [TEST] Write unit tests for configuration resolution in tests/unit/lib/test_runner/test_executor.py (test CLI > agent.yaml > env > defaults priority, ExecutionConfig merge)
- [x] T048 [CODE] Implement configuration resolution in src/holodeck/lib/test_runner/executor.py to pass T047 tests
- [x] T049 [TEST] Write unit tests for tool call validation in tests/unit/lib/test_runner/test_executor.py (test expected_tools matching, TestResult.tool_calls vs TestCaseModel.expected_tools)
- [x] T050 [CODE] Implement tool call validation in src/holodeck/lib/test_runner/executor.py to pass T049 tests
- [x] T051 [TEST] Write unit tests for timeout handling in tests/unit/lib/test_runner/test_executor.py (test file: 30s, LLM: 60s, download: 30s defaults using asyncio.timeout or threading.Timer)
- [x] T052 [CODE] Implement timeout handling in src/holodeck/lib/test_runner/executor.py to pass T051 tests
- [x] T053 [TEST] Write unit tests for test executor main flow in tests/unit/lib/test_runner/test_executor.py (test load AgentConfig, execute tests sequentially, collect TestResult instances, generate TestReport)
- [x] T054 [CODE] Implement test executor in src/holodeck/lib/test_runner/executor.py to pass T053 tests
- [x] T055 [VERIFY] Run make test-unit to verify executor tests pass
- [x] T056 [VERIFY] Run make format && make lint

#### T057-T060: Progress Indicators (Test-First)

- [x] T057 [TEST] Write unit tests for progress indicators in tests/unit/lib/test_runner/test_progress.py (test TTY detection with sys.stdout.isatty(), "Test X/Y" display, checkmarks/X marks, CI/CD plain text mode)
- [x] T058 [CODE] Implement progress indicators in src/holodeck/lib/test_runner/progress.py to pass T057 tests
- [x] T059 [VERIFY] Run make test-unit to verify progress tests pass
- [x] T060 [VERIFY] Run make format && make lint

#### T061-T066: Executor Integration with Callback Pattern (Test-First)

**Design Pattern**: Callback-based progress reporting

The progress indicator integrates with TestExecutor via a callback pattern for clean separation of concerns:

```python
# CLI passes callback to executor
def progress_callback(result: TestResult) -> None:
    indicator.update(result)
    print(indicator.get_progress_line())

executor = TestExecutor(
    agent_config_path=config_path,
    progress_callback=progress_callback
)

# Executor calls callback after each test
for test_case in test_cases:
    result = await self._execute_single_test(test_case)
    test_results.append(result)
    if self.progress_callback:
        self.progress_callback(result)  # Notify CLI of completion
```

**Benefits**:

- ✅ Executor stays focused on orchestration (no display logic)
- ✅ CLI controls display (quiet, verbose, progress suppression)
- ✅ Testable without side effects (mock the callback)
- ✅ Flexible for future use cases (webhooks, logging, etc.)

**Tasks**:

- [x] T061 [TEST] Write unit tests for progress callback integration in tests/unit/lib/test_runner/test_executor.py (test callback invocation after each test, callback with None handling, multiple test execution flow)
- [x] T062 [CODE] Add callback support to TestExecutor in src/holodeck/lib/test_runner/executor.py (add progress_callback parameter to **init**, call callback after each test in execute_tests, handle None gracefully)
- [x] T063 [TEST] Write unit tests for CLI progress display in tests/unit/cli/commands/test_test.py (test ProgressIndicator initialization, callback function execution, progress line printing, final summary display)
- [x] T064 [CODE] Integrate ProgressIndicator in CLI command src/holodeck/cli/commands/test.py (create indicator, pass callback to executor, display summary after tests complete, respect --quiet/--verbose flags)
- [x] T065 [VERIFY] Run make test-unit to verify executor and CLI integration tests pass
- [x] T066 [VERIFY] Run make format && make lint && make type-check

#### T067-T074: CLI Command (Test-First)

- [x] T067 [TEST] Write unit tests for CLI command in tests/unit/cli/commands/test_test.py (test argument parsing for AGENT_CONFIG, option handling for --output, --format, --verbose, --quiet, --timeout flags)
- [x] T068 [CODE] Implement CLI command in src/holodeck/cli/commands/test.py to pass T067 tests
- [x] T069 [TEST] Write unit tests for exit code logic in tests/unit/cli/commands/test_test.py (test 0=success, 1=test failure, 2=config error, 3=execution error, 4=evaluation error)
- [x] T070 [CODE] Implement exit code logic in src/holodeck/cli/commands/test.py to pass T069 tests
- [x] T071 [VERIFY] Run make test-unit to verify CLI tests pass
- [x] T072 [VERIFY] Run make format && make lint
- [x] T073 [TEST] Write unit tests for report generation integration in tests/unit/cli/commands/test_test.py (test --output flag, JSON/Markdown format)
- [x] T074 [CODE] Implement report file generation in CLI command

#### T075-T078: Integration Testing

- [x] T075 [TEST] Create sample test agent.yaml in tests/fixtures/agents/test_agent.yaml with 3 simple text test cases
- [x] T076 [TEST] Write integration test for basic text execution in tests/integration/test_basic_execution.py (verify end-to-end test run with mocked LLM responses)
- [x] T077 [VERIFY] Run make test-integration to verify integration tests pass
- [x] T078 [VERIFY] Run make test to verify all tests pass

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

#### T079-T088: File Processing Extensions (Test-First)

- [x] T079 [TEST] Write unit tests for page/sheet/range extraction in tests/unit/lib/test_file_processor.py (test FileInput.pages for PDF, FileInput.sheet for Excel, FileInput.range for PowerPoint, preprocessing before markitdown)
- [x] T080 [CODE] Implement page/sheet/range extraction in src/holodeck/lib/file_processor.py to pass T079 tests
- [x] T081 [TEST] Write unit tests for file size warnings in tests/unit/lib/test_file_processor.py (test >100MB file warning message, verify processing continues)
- [x] T082 [CODE] Implement file size warning logic in src/holodeck/lib/file_processor.py to pass T081 tests
- [x] T083 [TEST] Write unit tests for file processing error handling in tests/unit/lib/test_file_processor.py (test timeout, malformed files, ProcessedFileInput.error population, test continuation)
- [x] T084 [CODE] Implement file processing error handling in src/holodeck/lib/file_processor.py to pass T083 tests
- [x] T085 [VERIFY] Run make test-unit to verify file processor tests pass
- [x] T086 [VERIFY] Run make format && make lint

#### T087-T092: Executor Integration (Test-First)

- [ ] T087 [TEST] Write unit tests for file processor integration in tests/unit/lib/test_runner/test_executor.py (test process files before agent invocation, verify ProcessedFileInput.markdown_content in agent context)
- [ ] T088 [CODE] Integrate file processor with test executor in src/holodeck/lib/test_runner/executor.py to pass T087 tests
- [ ] T089 [VERIFY] Run make test-unit to verify executor integration tests pass
- [ ] T090 [VERIFY] Run make format && make lint

#### T091-T094: Integration Testing

- [x] T091 [TEST] Create sample multimodal test files in tests/fixtures/files/ (sample.pdf, sample.jpg, sample.xlsx, sample.docx, sample.pptx with known content)
- [x] T092 [TEST] Write integration test for multimodal execution in tests/integration/test_multimodal_execution.py (verify file processing and agent receives markdown content)
- [x] T093 [VERIFY] Run make test-integration to verify multimodal tests pass
- [x] T094 [VERIFY] Run make test to verify all tests pass

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

#### T095-T100: Per-Test Metrics (Test-First)

- [x] T095 [TEST] Write unit tests for per-test metric resolution in tests/unit/lib/test_runner/test_executor.py (test TestCaseModel.evaluations override, test fallback to AgentConfig.evaluations.metrics)
- [x] T096 [CODE] Implement per-test metric resolution logic in src/holodeck/lib/test_runner/executor.py to pass T095 tests
- [x] T097 [TEST] Write unit tests for metric validation in tests/unit/lib/test_runner/test_executor.py (test undefined metric raises ConfigError, test valid metrics pass)
- [x] T098 [CODE] Implement metric validation in src/holodeck/lib/test_runner/executor.py to pass T097 tests
- [x] T099 [VERIFY] Run make test-unit to verify metric resolution tests pass
- [x] T100 [VERIFY] Run make format && make lint

#### T101-T104: Integration Testing

- [x] T101 [TEST] Write integration test for per-test metrics in tests/integration/test_evaluation_metrics.py (verify metric override behavior, test global defaults)
- [x] T102 [VERIFY] Run make test-integration to verify metric tests pass
- [x] T103 [VERIFY] Run make test to verify all tests pass

---

## Phase 6: User Story 4 - Display Test Results with Enhanced Progress Indicators (P2)

**Story Goal**: Show real-time progress with visual indicators during test execution

**TDD Approach**: Write tests for display features before implementation

**Independent Test**: Run 10 test cases, observe console output, verify:

- Progress indicators update in real-time
- Checkmarks/X marks appear for pass/fail
- Final summary displayed
- CI/CD compatibility (no interactive elements in non-TTY)

**Dependencies**: Requires US1 (basic execution)

### Tasks

#### T104-T122: Enhanced Progress Display and Spinner (Test-First)

**Note**: Core progress indicator (T057-T058) is already implemented. These tasks add optional spinner animation and color enhancements built on top of the callback integration.

- [x] T104 [TEST] Write unit tests for optional spinner in tests/unit/lib/test_runner/test_progress.py (test spinner for long-running tests >5s, test spinner char rotation during execution, test disable in non-TTY)
- [x] T105 [CODE] Implement optional spinner for long-running tests in src/holodeck/lib/test_runner/progress.py to pass T104 tests
- [x] T106 [TEST] Write unit tests for ANSI color codes in tests/unit/lib/test_runner/test_progress.py (test color output in TTY, test no colors in non-TTY)
- [x] T107 [CODE] Add optional ANSI color support to symbols in src/holodeck/lib/test_runner/progress.py to pass T106 tests
- [x] T108 [TEST] Write unit tests for elapsed time display in tests/unit/lib/test_runner/test_progress.py (test elapsed time for long tests, test hide for quick tests)
- [x] T109 [CODE] Add elapsed time display to progress.py to pass T108 tests
- [x] T110 [TEST] Write unit tests for progress callback integration in CLI in tests/unit/cli/commands/test_test.py (test callback receives results, test progress display updates)
- [x] T111 [CODE] Integrate progress callback in CLI display logic in src/holodeck/cli/commands/test.py to pass T110 tests
- [x] T112 [TEST] Write unit tests for quiet mode suppression in tests/unit/lib/test_runner/test_progress.py (test --quiet flag suppresses progress lines, test summary still shown)
- [x] T113 [CODE] Implement quiet mode in CLI to pass T112 tests
- [x] T114 [TEST] Write unit tests for verbose mode in tests/unit/lib/test_runner/test_progress.py (test --verbose flag shows debug info, timing details)
- [x] T115 [CODE] Implement verbose mode in CLI to pass T114 tests
- [x] T116 [VERIFY] Run make test-unit to verify progress enhancements pass
- [x] T117 [VERIFY] Run make format && make lint && make type-check
- [x] T118 [TEST] Write integration test for progress display in tests/integration/test_basic_execution.py (verify progress output during live test execution)
- [x] T119 [VERIFY] Run make test-integration to verify progress integration tests pass
- [x] T120 [VERIFY] Run make test to verify all tests pass
- [x] T121 [VERIFY] Verify progress indicator demo script works
- [x] T122 [VERIFY] Update documentation with progress indicator usage examples

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

#### T123-T133: Report Generation (Test-First)

- [x] T123 [TEST] Create expected report fixtures in tests/fixtures/expected_reports/ (sample_json_report.json, sample_markdown_report.md with known test data)
- [x] T124 [TEST] Write unit tests for JSON report generation in tests/unit/lib/test_runner/test_reporter.py (test TestReport.to_json() method, verify structure matches schema)
- [x] T125 [CODE] Implement JSON report generation in src/holodeck/lib/test_runner/reporter.py to pass T124 tests
- [x] T126 [TEST] Write unit tests for Markdown report generation in tests/unit/lib/test_runner/test_reporter.py (test TestReport.to_markdown() with formatted tables, summary, test details)
- [x] T127 [CODE] Implement Markdown report generation in src/holodeck/lib/test_runner/reporter.py to pass T126 tests
- [x] T128 [TEST] Write unit tests for report file writing in tests/unit/lib/test_runner/test_reporter.py (test TestReport.to_file() method)
- [x] T129 [CODE] Implement report file writing logic in src/holodeck/lib/test_runner/reporter.py to pass T128 tests
- [x] T130 [VERIFY] Run make test-unit to verify reporter tests pass
- [x] T131 [VERIFY] Run make format && make lint
- [x] T132 [TEST] Write integration test for report generation in tests/integration/test_report_generation.py (verify JSON and Markdown output match expected format)
- [x] T133 [VERIFY] Run make test-integration to verify report generation tests pass

#### T134-T139: CLI Format Detection & Report Output (Test-First)

- [x] T134 [TEST] Write unit tests for format auto-detection in tests/unit/cli/commands/test_test.py (test .json vs .md extension detection, test --format flag override)
- [x] T135 [CODE] Implement format auto-detection in src/holodeck/cli/commands/test.py to pass T134 tests
- [x] T136 [TEST] Write unit tests for report file output in CLI in tests/unit/cli/commands/test_test.py (test --output flag, verify file creation)
- [x] T137 [CODE] Implement report file output in CLI to pass T136 tests
- [x] T138 [VERIFY] Run make test-unit to verify format detection and output tests pass
- [x] T139 [VERIFY] Run make test to verify all tests pass

---

## Phase 8: Polish & Cross-Cutting Concerns

**Goal**: Final refinements, error handling, documentation, and production readiness

**TDD Approach**: Write tests for error handling and edge cases, then implement

### Tasks

#### T140-T148: Error Handling (Test-First)

- [ ] T140 [TEST] Write unit tests for structured error messages in tests/unit/lib/test_runner/test_executor.py (test error format: "ERROR: {summary}\n Cause: {cause}\n Suggestion: {action}")
- [ ] T141 [CODE] Implement structured error messages in src/holodeck/lib/test_runner/executor.py to pass T140 tests
- [ ] T142 [TEST] Write unit tests for large file warnings in tests/unit/lib/test_file_processor.py (test warning when file >100MB before processing)
- [ ] T143 [CODE] Implement warning messages for large files in src/holodeck/lib/file_processor.py to pass T142 tests
- [ ] T144 [TEST] Write unit tests for timeout error handling in tests/unit/lib/test_runner/test_executor.py (test graceful timeout handling, test error messages)
- [ ] T145 [CODE] Improve timeout error handling in executor to pass T144 tests
- [ ] T146 [VERIFY] Run make test-unit to verify error handling tests pass
- [ ] T147 [VERIFY] Run make format && make lint
- [ ] T148 [VERIFY] Run make security to verify no security issues

#### T149-T152: Configuration Defaults (Test-First)

- [ ] T149 [TEST] Write unit tests for configuration defaults in tests/unit/config/test_defaults.py (test file_timeout=30, llm_timeout=60, download_timeout=30, cache_dir=".holodeck/cache", cache_enabled=True)
- [ ] T150 [CODE] Add configuration defaults to src/holodeck/config/defaults.py to pass T149 tests
- [ ] T151 [VERIFY] Run make test-unit to verify defaults tests pass
- [ ] T152 [VERIFY] Run make format && make lint

#### T153-T164: Documentation & Quality Assurance

- [ ] T153 Create comprehensive docstrings for all new modules (file_processor.py, executor.py, agent_bridge.py, progress.py, reporter.py, azure_ai.py, nlp_metrics.py)
- [ ] T154 Update docsite (/docs) with new dependencies and modules (document test execution framework, new lib/ structure, ExecutionConfig model, progress callback pattern)
- [ ] T155 Update README.md with `holodeck test` command usage and examples
- [ ] T156 Create PROGRESS_INDICATORS.md documentation with callback pattern explanation
- [ ] T157 Run make format to format all new code
- [ ] T158 Run make lint to check code quality
- [ ] T159 Run make type-check to verify type hints
- [ ] T160 Run make security to scan for vulnerabilities
- [ ] T161 Run make test-coverage to verify 80%+ coverage minimum
- [ ] T162 Review coverage report and add missing tests if needed
- [ ] T163 Update pyproject.toml with version bump (0.2.0)
- [ ] T164 Run full CI pipeline: make ci

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

**New Files (Core)** - Phase 1 & 2:

- `src/holodeck/models/test_result.py` (T002)
- `src/holodeck/models/config.py` (T003, T011)
- `src/holodeck/lib/file_processor.py` (T005, T025-T030)
- `src/holodeck/lib/test_runner/executor.py` (T006, T047-T056, T062)
- `src/holodeck/lib/test_runner/agent_bridge.py` (T006, T043-T045)
- `src/holodeck/lib/evaluators/base.py` (T007, T033-T034)
- `src/holodeck/lib/evaluators/azure_ai.py` (T007, T035-T038)
- `src/holodeck/lib/evaluators/nlp_metrics.py` (T007, T039-T040)

**New Files (Progress & CLI)** - Phase 3A & 3B:

- `src/holodeck/lib/test_runner/progress.py` (T057-T058, T104-T122)
- `src/holodeck/lib/test_runner/reporter.py` (T123-T139)
- `src/holodeck/cli/commands/test.py` (T061-T066 callback integration, T067-T074 CLI, T110-T111 callback display)

**Modified Files**:

- `pyproject.toml` (T001, dependencies)
- `src/holodeck/models/agent.py` (T023-T024 execution field)
- `src/holodeck/cli/main.py` (T007 register test command)
- `src/holodeck/config/defaults.py` (T149-T150)
- `CLAUDE.md` (documentation updates)

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
- **Callback Pattern** (T061-T066, T110-T111): Progress display is decoupled from executor via callback function. CLI creates ProgressIndicator, passes callback to TestExecutor, receives TestResult updates after each test. Keeps executor focused on orchestration, allows CLI to handle display modes (quiet/verbose).
- **Progress Indicator** (T057-T058): TTY-aware progress display with real-time updates, pass/fail symbols (✓/✗ for TTY, PASS/FAIL for CI/CD), summary statistics, and execution timing. Built-in support for quiet and verbose modes.

---

**End of Tasks Document**
