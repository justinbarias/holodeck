# Feature Specification: Execute Agent Against Test Cases

**Feature Branch**: `006-agent-test-execution`
**Created**: 2025-11-01
**Status**: Draft
**Input**: User description: "Execute Agent Against Test Cases"

## Clarifications

### Session 2025-11-01

- Q: When a test case references a file that exceeds reasonable memory limits (e.g., 100MB+ PDF), how should the system respond? → A: Warn user but attempt processing (with potential timeout/failure)
- Q: What timeout values should be used for different operations during test execution? → A: File download: 30s, LLM API: 60s, File processing: 30s (user-configurable defaults)
- Q: When an evaluation metric fails (e.g., LLM API error, timeout), should the system automatically retry the metric evaluation? → A: Retry up to 3 times with exponential backoff
- Q: Where should cached remote files (downloaded via URL) be stored on the local filesystem? → A: Project directory: `.holodeck/cache/`
- Q: How verbose should error messages be when tests fail or encounter errors? → A: Structured errors with context and actionable suggestions (verbose mode available via flag)

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Execute Basic Text-Only Test Cases (Priority: P1)

A developer configures an agent with test cases in `agent.yaml` and runs `holodeck test agent.yaml` to validate agent responses against expected outputs using evaluation metrics.

**Why this priority**: This is the foundational capability - without basic test execution, no testing framework exists. It delivers immediate value by enabling developers to validate agent behavior programmatically.

**Independent Test**: Can be fully tested by creating an agent.yaml with simple text test cases, running `holodeck test agent.yaml`, and verifying that:
- Test cases execute sequentially
- Agent responses are captured
- Evaluation metrics are calculated
- Pass/fail results are reported

**Acceptance Scenarios**:

1. **Given** an agent.yaml with 3 text-based test cases and groundedness/relevance metrics configured, **When** user runs `holodeck test agent.yaml`, **Then** all 3 test cases execute and display pass/fail status with metric scores
2. **Given** a test case with `ground_truth` defined, **When** the test executes, **Then** the agent's response is compared against the ground truth using configured metrics
3. **Given** a test case with `expected_tools` specified, **When** the test executes, **Then** the system validates that the agent called the expected tools
4. **Given** multiple test cases with varying complexity, **When** tests execute, **Then** each test result includes test name, input, agent response, metric scores, and pass/fail status
5. **Given** a test case fails to meet threshold, **When** the test completes, **Then** the failure is logged with specific metric score and threshold comparison

---

### User Story 2 - Execute Multimodal Test Cases with Files (Priority: P2)

A developer creates test cases with attached files (images, PDFs, documents) and runs tests where the agent processes both text input and file content to generate responses.

**Why this priority**: Multimodal testing is a key differentiator for HoloDeck. It enables validation of agents that process documents, images, and mixed media - essential for real-world use cases like document analysis and customer support.

**Independent Test**: Can be tested independently by:
- Creating test cases with file inputs (PDF, image, Excel)
- Running `holodeck test agent.yaml`
- Verifying that file content is loaded and passed to the agent
- Confirming agent responses reference file content

**Acceptance Scenarios**:

1. **Given** a test case with a PDF file input, **When** the test executes, **Then** the PDF content is extracted and provided to the agent as context
2. **Given** a test case with image files, **When** the test executes, **Then** images are processed (OCR if applicable) and included in agent context
3. **Given** a test case with Excel file specifying sheet and range, **When** the test executes, **Then** only the specified data range is extracted and provided to the agent
4. **Given** a test case with PowerPoint file specifying slides, **When** the test executes, **Then** only the specified slides are extracted as context
5. **Given** a test case with multiple files (PDF + images), **When** the test executes, **Then** all files are processed and combined into the agent's context
6. **Given** a test case with remote URL file input and caching enabled, **When** the test executes for the first time, **Then** the file is downloaded, cached in `.holodeck/cache/` directory, and reused in subsequent runs

---

### User Story 3 - Execute Tests with Per-Test Metric Configuration (Priority: P3)

A developer configures specific evaluation metrics for individual test cases, allowing different tests to be validated using different criteria (e.g., groundedness for factual questions, relevance for general queries).

**Why this priority**: Provides flexibility in testing strategy - different test cases may require different validation approaches. This is an optimization over global metrics and can be added after core functionality works.

**Independent Test**: Can be tested by:
- Creating test cases with varying metric configurations
- Running tests and verifying each uses its specified metrics
- Confirming that per-test metrics override global defaults

**Acceptance Scenarios**:

1. **Given** a test case with `evaluations: [groundedness, relevance]` specified, **When** the test executes, **Then** only groundedness and relevance metrics are calculated for that test
2. **Given** multiple test cases with different metric configurations, **When** tests execute, **Then** each test uses its specific metrics (not global defaults)
3. **Given** a test case without specific metrics defined, **When** the test executes, **Then** the global evaluation metrics from agent.yaml are used
4. **Given** a test case specifying a metric not in the global config, **When** the test executes, **Then** an error is raised indicating the metric is not configured

---

### User Story 4 - Display Test Results with Progress Indicators (Priority: P2)

A developer runs tests and sees real-time progress with a visual indicator showing current test execution, completed tests, and overall progress.

**Why this priority**: User experience enhancement that makes testing feel responsive and professional. Important for CI/CD adoption where developers monitor test output in logs.

**Independent Test**: Can be tested by:
- Running tests and observing console output
- Verifying progress indicators appear and update
- Confirming final summary is displayed

**Acceptance Scenarios**:

1. **Given** 10 test cases to execute, **When** tests start running, **Then** a progress indicator shows "Running test 1/10", "Running test 2/10", etc.
2. **Given** tests are executing, **When** each test completes, **Then** a checkmark (✅) or X (❌) appears next to the test name with pass/fail status
3. **Given** all tests complete, **When** execution finishes, **Then** a summary is displayed showing total tests, passed, failed, and overall pass rate
4. **Given** a test takes longer than 5 seconds, **When** the test is running, **Then** a spinner or elapsed time indicator shows progress
5. **Given** tests are running in CI/CD environment, **When** output is displayed, **Then** format is compatible with standard CI log formatting (no interactive elements that break in non-TTY environments)

---

### User Story 5 - Generate Test Report Files (Priority: P3)

A developer runs tests and receives a detailed test report in JSON or Markdown format that can be saved, shared, or archived for audit purposes.

**Why this priority**: Important for CI/CD integration and historical tracking, but not essential for initial testing capability. Can be added after core execution works.

**Independent Test**: Can be tested by:
- Running tests with `--output report.json` flag
- Verifying report file is created with complete test data
- Validating report structure matches expected schema

**Acceptance Scenarios**:

1. **Given** user runs `holodeck test agent.yaml --output report.json`, **When** tests complete, **Then** a JSON file is created with all test results, metrics, and metadata
2. **Given** user runs `holodeck test agent.yaml --format markdown`, **When** tests complete, **Then** a human-readable Markdown report is generated
3. **Given** a report file is generated, **When** the file is opened, **Then** it contains: test case name, input, agent response, all metric scores, pass/fail status, and timestamp
4. **Given** tests include multimodal files, **When** report is generated, **Then** file paths and metadata are included in the report

---

### Edge Cases

- What happens when an agent.yaml file doesn't exist at the specified path?
- How does the system handle test cases with malformed file paths or missing files?
- **Timeout handling**: Remote URL file downloads timeout after 30s (configurable), LLM API calls timeout after 60s (configurable), and file processing operations timeout after 30s (configurable)
- **Metric retry strategy**: When evaluation metric fails (LLM API error, timeout), the system automatically retries up to 3 times with exponential backoff before marking the metric evaluation as failed
- What happens when a test case has `expected_tools` but the agent doesn't use tools?
- **Large file handling**: When a test case references a very large file (e.g., 100MB+ PDF), the system warns the user about potential memory/timeout issues but attempts to process the file, respecting LLM token context window limits
- What occurs when Excel file references a non-existent sheet or invalid range?
- How are concurrent test executions prevented (or handled) if user runs multiple `holodeck test` commands?
- What happens when a test case input is empty or contains only whitespace?
- How does the system handle evaluation metrics that require models not configured in agent.yaml?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST load agent configuration from the specified YAML file path
- **FR-002**: System MUST validate agent configuration schema before executing tests
- **FR-003**: System MUST execute test cases sequentially in the order they appear in configuration
- **FR-004**: System MUST pass test case input to the agent and capture the agent's response
- **FR-005**: System MUST evaluate agent responses using configured evaluation metrics
- **FR-006**: System MUST compare metric scores against thresholds to determine pass/fail status
- **FR-007**: System MUST validate that expected tools were called if `expected_tools` is specified
- **FR-008**: System MUST load and process multimodal file inputs (PDF, images, Excel, Word, PowerPoint) with warnings for files exceeding typical size thresholds
- **FR-009**: System MUST extract specific pages/sheets/ranges from files when specified in test configuration
- **FR-010**: System MUST download and cache remote URL files in `.holodeck/cache/` directory when cache is enabled with configurable timeout defaults (30s for downloads, 60s for LLM calls, 30s for file processing)
- **FR-011**: System MUST use per-test-case evaluation metrics when specified, falling back to global metrics
- **FR-012**: System MUST display real-time progress indicators showing current test execution status
- **FR-013**: System MUST display final summary with total tests, passed, failed, and pass rate
- **FR-014**: System MUST generate test report in JSON or Markdown format when output flag is provided
- **FR-015**: System MUST handle evaluation metric errors gracefully by retrying up to 3 times with exponential backoff before marking as failed, without stopping all test execution
- **FR-016**: System MUST return non-zero exit code when any test fails (for CI/CD integration)
- **FR-017**: System MUST validate file types match allowed types (image, pdf, text, excel, word, powerpoint, csv)
- **FR-018**: System MUST enforce maximum file count limit per test case (10 files)
- **FR-019**: System MUST provide structured error messages with context and actionable suggestions for all failures, with optional verbose mode (--verbose flag) for detailed stack traces
- **FR-020**: System MUST support both inline test cases and test cases loaded from external files

### Key Entities

- **TestCase**: Represents a single test scenario with input, expected output, expected tool usage, file inputs, and metric configuration
- **FileInput**: Represents a file reference (local or remote URL) with extraction parameters (pages, sheets, ranges); remote files are cached in `.holodeck/cache/` when caching is enabled
- **EvaluationMetric**: Represents a metric configuration with name, threshold, model override, and retry settings
- **TestResult**: Represents the outcome of a single test execution including agent response, metric scores, pass/fail status, and execution time
- **TestReport**: Aggregate of all test results with summary statistics and metadata
- **AgentConfig**: Complete agent configuration including LLM settings, tools, instructions, and evaluation framework

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Users can execute a complete test suite of 10 test cases in under 30 seconds (excluding LLM API latency)
- **SC-002**: Test execution reports pass/fail status for 100% of configured test cases
- **SC-003**: System correctly validates expected tool usage with 100% accuracy when `expected_tools` is specified
- **SC-004**: Multimodal file processing supports at least 5 file types (PDF, image, Excel, Word, PowerPoint) with correct content extraction
- **SC-005**: Test reports include all required information (input, response, metrics, status) for 100% of executed tests
- **SC-006**: System handles evaluation metric failures gracefully, allowing remaining tests to continue execution
- **SC-007**: Progress indicators update in real-time for each test execution, providing immediate feedback
- **SC-008**: Generated test reports are valid JSON/Markdown that can be parsed and read by standard tools
- **SC-009**: Exit codes correctly indicate test success (0) or failure (non-zero) for CI/CD integration
- **SC-010**: 90% of test execution errors include structured, actionable error messages with context and suggestions pointing to the specific configuration issue

## Assumptions

- Agent execution engine exists and can process inputs to generate responses (if not, execution will be mocked for testing)
- Evaluation metrics framework exists or will be implemented as part of this feature
- LLM provider integrations are functional for agent execution
- File processing libraries (PDF parsing, image OCR, Office document extraction) are available or will be integrated
- Configuration validation logic from existing `ConfigValidator` can be reused
- Test cases are defined in agent.yaml following the existing `TestCaseModel` schema
- Users have proper API keys configured for LLM providers used by agents and evaluation metrics

## Out of Scope

- Parallel test execution (tests run sequentially in this version)
- Web-based UI for viewing test results
- Automated test case generation from user conversations
- Test case parametrization or data-driven testing (running one test with multiple inputs)
- Performance benchmarking or load testing capabilities
- Test coverage analysis or code coverage reporting
- Integration with third-party test frameworks (pytest, unittest)
- Real-time streaming of agent responses during test execution
- Test result comparison across multiple test runs (historical trending)
- Custom evaluation metric plugin development
