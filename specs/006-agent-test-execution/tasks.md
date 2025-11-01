# Implementation Tasks: Execute Agent Against Test Cases

**Branch**: `006-agent-test-execution`
**Spec**: [spec.md](./spec.md)
**Plan**: [plan.md](./plan.md)

## Task Summary

- **Total Tasks**: 74
- **Setup Tasks**: 10
- **Foundational Tasks**: 10
- **User Story Tasks**: 46
  - US1 (P1): 16 tasks
  - US2 (P2): 7 tasks
  - US3 (P3): 4 tasks
  - US4 (P2): 8 tasks
  - US5 (P3): 7 tasks
- **Polish Tasks**: 14

## Implementation Strategy

**MVP = User Story 1 (P1: Execute Basic Text-Only Test Cases)**

Each user story is independently testable and deliverable. User stories follow priority order (P1, P2, P3) and dependency constraints.

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

### Tasks

- [ ] T001 Install new dependencies in pyproject.toml (semantic-kernel[azure]>=1.37.0, markitdown[all]>=0.1.0, azure-ai-evaluation>=1.0.0, evaluate>=0.4.0, sacrebleu>=2.3.0, aiofiles>=23.0.0)
- [ ] T002 Create new models/test_result.py with ProcessedFileInput, MetricResult, TestResult, ReportSummary, TestReport models
- [ ] T003 Add ExecutionConfig model to models/config.py with timeout, cache, and output settings validation
- [ ] T004 Update Agent model in models/agent.py to include execution: ExecutionConfig | None field
- [ ] T005 Create lib/file_processor.py module structure with markitdown integration placeholder
- [ ] T006 Create lib/test_runner/ directory with __init__.py, executor.py, agent_bridge.py, progress.py, reporter.py
- [ ] T007 Create lib/evaluators/ directory with __init__.py, base.py, azure_ai.py, nlp_metrics.py
- [ ] T008 Create cli/commands/test.py CLI command structure with Click framework
- [ ] T009 Register test command in cli/main.py
- [ ] T010 Create test fixtures directory structure (tests/fixtures/agents/, tests/fixtures/files/, tests/fixtures/expected_reports/)

---

## Phase 2: Foundational Components

**Goal**: Implement shared infrastructure needed by all user stories (blocking prerequisites)

**IMPORTANT**: All Phase 2 tasks must complete before ANY user story implementation begins.

### Tasks

- [ ] T011 [P] Implement ExecutionConfig model with validation in src/holodeck/models/config.py (file_timeout 1-300s, llm_timeout 1-600s, download_timeout 1-300s, cache_enabled, cache_dir, verbose, quiet)
- [ ] T012 [P] Implement ProcessedFileInput model in src/holodeck/models/test_result.py (original, markdown_content, metadata, cached_path, processing_time_ms, error)
- [ ] T013 [P] Implement MetricResult model in src/holodeck/models/test_result.py (metric_name, score, threshold, passed, scale, error, retry_count, evaluation_time_ms, model_used)
- [ ] T014 [P] Implement TestResult model in src/holodeck/models/test_result.py (test_name, test_input, processed_files, agent_response, tool_calls, expected_tools, tools_matched, metric_results, ground_truth, passed, execution_time_ms, errors, timestamp)
- [ ] T015 [P] Implement ReportSummary model in src/holodeck/models/test_result.py (total_tests, passed, failed, pass_rate, total_duration_ms, metrics_evaluated, average_scores)
- [ ] T016 [P] Implement TestReport model in src/holodeck/models/test_result.py with to_json() and to_file() methods (agent_name, agent_config_path, results, summary, timestamp, holodeck_version, environment)
- [ ] T017 Update Agent model to include execution: ExecutionConfig | None field in src/holodeck/models/agent.py
- [ ] T018 [P] Implement file processor using markitdown in src/holodeck/lib/file_processor.py (handle PDF, images, Excel, Word, PowerPoint, CSV, HTML with MarkItDown().convert())
- [ ] T019 [P] Implement file caching logic in src/holodeck/lib/file_processor.py (.holodeck/cache/ directory with hash-based cache keys)
- [ ] T020 [P] Implement remote URL download with timeout in src/holodeck/lib/file_processor.py (requests library with configurable timeout, 3 retries with exponential backoff)

---

## Phase 3: User Story 1 - Execute Basic Text-Only Test Cases (P1)

**Story Goal**: Execute text-based test cases against agents and display pass/fail status with metric scores

**Independent Test**: Create agent.yaml with 3 simple text test cases, run `holodeck test agent.yaml`, verify:
- Sequential execution
- Agent responses captured
- Evaluation metrics calculated
- Pass/fail results displayed

**Dependencies**: Requires Phase 2 (Foundational) completion

### Tasks

- [ ] T021 [US1] Implement Semantic Kernel agent bridge in src/holodeck/lib/test_runner/agent_bridge.py (create Kernel, load agent config, invoke with ChatHistory, capture response and tool_calls)
- [ ] T022 [US1] Implement Azure AI Evaluation SDK integration in src/holodeck/lib/evaluators/azure_ai.py (GroundednessEvaluator, RelevanceEvaluator, CoherenceEvaluator, FluencyEvaluator, SimilarityEvaluator with per-metric model config)
- [ ] T023 [US1] Implement NLP metrics in src/holodeck/lib/evaluators/nlp_metrics.py (BLEU, ROUGE, METEOR, F1 using evaluate.load() from Hugging Face)
- [ ] T024 [US1] Implement base evaluator interface in src/holodeck/lib/evaluators/base.py (abstract evaluate() method, timeout handling, retry logic)
- [ ] T025 [US1] Implement test executor in src/holodeck/lib/test_runner/executor.py (load AgentConfig, execute tests sequentially, collect TestResult instances, generate TestReport)
- [ ] T026 [US1] Implement progress indicators in src/holodeck/lib/test_runner/progress.py (TTY detection with sys.stdout.isatty(), real-time "Test X/Y" display, checkmarks/X marks, CI/CD plain text mode)
- [ ] T027 [US1] Implement configuration resolution (CLI > agent.yaml > env > defaults) in src/holodeck/lib/test_runner/executor.py (merge ExecutionConfig from multiple sources)
- [ ] T028 [US1] Implement tool call validation (expected_tools matching) in src/holodeck/lib/test_runner/executor.py (compare TestResult.tool_calls with TestCaseModel.expected_tools)
- [ ] T029 [US1] Implement metric evaluation with retry logic (3 attempts, exponential backoff 2s/4s/8s) in src/holodeck/lib/evaluators/azure_ai.py (handle LLM API errors, timeouts)
- [ ] T030 [US1] Implement timeout handling (file: 30s, LLM: 60s, download: 30s defaults) in src/holodeck/lib/test_runner/executor.py (use asyncio.timeout or threading.Timer)
- [ ] T031 [US1] Implement CLI command in src/holodeck/cli/commands/test.py (argument parsing for AGENT_CONFIG, option handling for --output, --format, --verbose, --quiet, --timeout flags)
- [ ] T032 [US1] Implement exit code logic (0=success, 1=test failure, 2=config error, 3=execution error, 4=evaluation error) in src/holodeck/cli/commands/test.py
- [ ] T033 [US1] Create sample test agent.yaml in tests/fixtures/agents/test_agent.yaml with 3 simple text test cases
- [ ] T034 [US1] Create integration test for basic text execution in tests/integration/test_basic_execution.py (verify end-to-end test run with mocked LLM responses)
- [ ] T035 [US1] Create unit tests for agent_bridge in tests/unit/lib/test_runner/test_agent_bridge.py (test Semantic Kernel integration)
- [ ] T036 [US1] Create unit tests for executor in tests/unit/lib/test_runner/test_executor.py (test configuration resolution, test execution flow)

---

## Phase 4: User Story 2 - Execute Multimodal Test Cases with Files (P2)

**Story Goal**: Execute test cases with attached files (PDF, images, Office documents) and provide file content to agent

**Independent Test**: Create test cases with PDF/image/Excel files, run tests, verify:
- Files processed via markitdown
- Content extracted and provided to agent
- Agent responses reference file content
- Tests execute successfully

**Dependencies**: Requires US1 (basic test execution infrastructure)

### Tasks

- [ ] T037 [US2] Implement page/sheet/range extraction for PDF/Excel/PowerPoint in src/holodeck/lib/file_processor.py (use FileInput.pages, FileInput.sheet, FileInput.range to preprocess files before markitdown)
- [ ] T038 [US2] Implement file size warning logic (>100MB files) in src/holodeck/lib/file_processor.py (log warning message, continue processing)
- [ ] T039 [US2] Integrate file processor with test executor in src/holodeck/lib/test_runner/executor.py (process files before agent invocation, include ProcessedFileInput.markdown_content in agent context)
- [ ] T040 [US2] Implement file processing error handling (timeout, malformed files) in src/holodeck/lib/file_processor.py (catch exceptions, populate ProcessedFileInput.error, allow test to continue)
- [ ] T041 [US2] Create sample multimodal test files in tests/fixtures/files/ (sample.pdf, sample.jpg, sample.xlsx, sample.docx, sample.pptx with known content)
- [ ] T042 [US2] Create integration test for multimodal execution in tests/integration/test_multimodal_execution.py (verify file processing and agent receives markdown content)
- [ ] T043 [US2] Create unit tests for file_processor in tests/unit/lib/test_file_processor.py (test markitdown integration, caching, range extraction)

---

## Phase 5: User Story 3 - Execute Tests with Per-Test Metric Configuration (P3)

**Story Goal**: Allow test cases to specify their own evaluation metrics, overriding global defaults

**Independent Test**: Create test cases with varying metric configurations, verify:
- Per-test metrics override global metrics
- Tests without metrics use global defaults
- Invalid metric references raise errors

**Dependencies**: Requires US1 (evaluation infrastructure)

### Tasks

- [ ] T044 [US3] Implement per-test metric resolution logic in src/holodeck/lib/test_runner/executor.py (if TestCaseModel.evaluations is set, use those metrics; otherwise use AgentConfig.evaluations.metrics)
- [ ] T045 [US3] Implement metric validation (ensure test metrics exist in global config) in src/holodeck/lib/test_runner/executor.py (raise ConfigError if test references undefined metric)
- [ ] T046 [US3] Create integration test for per-test metrics in tests/integration/test_evaluation_metrics.py (verify metric override behavior)
- [ ] T047 [US3] Create unit test for metric resolution in tests/unit/lib/test_runner/test_executor.py (test per-test vs global metric selection)

---

## Phase 6: User Story 4 - Display Test Results with Progress Indicators (P2)

**Story Goal**: Show real-time progress with visual indicators during test execution

**Independent Test**: Run 10 test cases, observe console output, verify:
- Progress indicators update in real-time
- Checkmarks/X marks appear for pass/fail
- Final summary displayed
- CI/CD compatibility (no interactive elements in non-TTY)

**Dependencies**: Requires US1 (basic execution)

### Tasks

- [ ] T048 [US4] Implement TTY detection in src/holodeck/lib/test_runner/progress.py (use sys.stdout.isatty() to detect interactive terminal)
- [ ] T049 [US4] Implement progress indicator display (Test X/Y format) in src/holodeck/lib/test_runner/progress.py (update on each test start)
- [ ] T050 [US4] Implement pass/fail symbols (✅/❌) with color support in src/holodeck/lib/test_runner/progress.py (use ANSI color codes for TTY, plain text for non-TTY)
- [ ] T051 [US4] Implement spinner for long-running tests (>5s) in src/holodeck/lib/test_runner/progress.py (rotate spinner chars during execution)
- [ ] T052 [US4] Implement summary display (total/passed/failed/pass rate) in src/holodeck/lib/test_runner/progress.py (print summary after all tests complete)
- [ ] T053 [US4] Implement quiet mode output (--quiet flag) in src/holodeck/lib/test_runner/progress.py (suppress progress indicators, show only final summary)
- [ ] T054 [US4] Implement verbose mode output (--verbose flag) in src/holodeck/lib/test_runner/progress.py (show detailed debug info, stack traces, timing)
- [ ] T055 [US4] Create unit tests for progress indicators in tests/unit/lib/test_runner/test_progress.py (test TTY detection, format output, quiet/verbose modes)

---

## Phase 7: User Story 5 - Generate Test Report Files (P3)

**Story Goal**: Generate detailed test reports in JSON or Markdown format

**Independent Test**: Run tests with --output flag, verify:
- JSON file created with complete test data
- Markdown file created with human-readable format
- Report structure matches expected schema

**Dependencies**: Requires US1 (test results data structure)

### Tasks

- [ ] T056 [US5] Implement JSON report generation in src/holodeck/lib/test_runner/reporter.py (use TestReport.to_json() method)
- [ ] T057 [US5] Implement Markdown report generation in src/holodeck/lib/test_runner/reporter.py (implement TestReport.to_markdown() with formatted tables, summary, test details)
- [ ] T058 [US5] Implement report file writing logic in src/holodeck/lib/test_runner/reporter.py (use TestReport.to_file() method)
- [ ] T059 [US5] Implement format auto-detection from file extension in src/holodeck/cli/commands/test.py (detect .json vs .md extension, use --format flag as override)
- [ ] T060 [US5] Create expected report fixtures in tests/fixtures/expected_reports/ (sample_json_report.json, sample_markdown_report.md with known test data)
- [ ] T061 [US5] Create integration test for report generation in tests/integration/test_report_generation.py (verify JSON and Markdown output match expected format)
- [ ] T062 [US5] Create unit tests for reporter in tests/unit/lib/test_runner/test_reporter.py (test to_json(), to_markdown(), to_file() methods)

---

## Phase 8: Polish & Cross-Cutting Concerns

**Goal**: Final refinements, error handling, documentation, and production readiness

### Tasks

- [ ] T063 [P] Implement structured error messages with context and suggestions in src/holodeck/lib/test_runner/executor.py (format: "ERROR: {summary}\n  Cause: {cause}\n  Suggestion: {action}")
- [ ] T064 [P] Implement warning messages for large files in src/holodeck/lib/file_processor.py (warn when file >100MB before processing)
- [ ] T065 [P] Add configuration defaults to src/holodeck/config/defaults.py (file_timeout=30, llm_timeout=60, download_timeout=30, cache_dir=".holodeck/cache", cache_enabled=True)
- [ ] T066 [P] Create comprehensive docstrings for all new modules (file_processor.py, executor.py, agent_bridge.py, progress.py, reporter.py, azure_ai.py, nlp_metrics.py)
- [ ] T067 [P] Update CLAUDE.md with new dependencies and modules (document test execution framework, new lib/ structure, ExecutionConfig model)
- [ ] T068 Run make format to format all new code
- [ ] T069 Run make lint to check code quality
- [ ] T070 Run make type-check to verify type hints
- [ ] T071 Run make security to scan for vulnerabilities
- [ ] T072 Run make test-coverage to verify 80% coverage minimum
- [ ] T073 Update pyproject.toml with version bump (0.1.0)
- [ ] T074 Create usage examples in quickstart.md documentation (already exists, verify completeness)

---

## Parallel Execution Opportunities

### Phase 1 (Setup)
**All tasks can run in parallel** - different files being created

**Suggested groups**:
- Group A: T001-T004 (dependencies and models)
- Group B: T005-T007 (lib structure)
- Group C: T008-T010 (CLI and tests)

### Phase 2 (Foundational)
**Model implementations can run in parallel**:
- Group A [P]: T011-T016 (different Pydantic models)
- Group B [P]: T018-T020 (file processor components)
- Sequential: T017 (depends on T011)

### Phase 3 (US1)
**Different evaluator implementations**:
- Group A [P]: T021, T022, T023, T024 (different evaluator/bridge modules)
- Sequential: T025-T032 (executor and CLI depend on evaluators)
- Group B [P]: T033-T036 (tests - different test files)

### Phase 4 (US2)
**Sequential**: T037-T040 (same file modifications)
**Parallel**: T041-T043 (fixtures and tests)

### Phase 5 (US3)
**Sequential**: T044-T045 (same file modifications)
**Parallel**: T046-T047 (different test files)

### Phase 6 (US4)
**Sequential**: T048-T054 (same file - progress.py)
**Independent**: T055 (test file)

### Phase 7 (US5)
**Parallel**:
- Group A [P]: T056-T058 (different reporter methods)
- Sequential: T059 (depends on reporter)
- Group B [P]: T060-T062 (fixtures and tests)

### Phase 8 (Polish)
**Parallel**: T063-T067 (different files)
**Sequential**: T068-T072 (quality checks with dependencies)
**Independent**: T073-T074 (final tasks)

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
