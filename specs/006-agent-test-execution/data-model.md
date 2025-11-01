# Data Model: Test Execution Framework

**Date**: 2025-11-01
**Feature**: Execute Agent Against Test Cases
**Phase**: 1 - Design

## Overview

This document defines the data models for the test execution framework. Models follow Pydantic v2 patterns for validation and serialization. All models support JSON/YAML serialization for configuration and reporting.

## Entity Relationship Diagram

```
┌──────────────────┐
│   AgentConfig    │  (Existing)
│                  │
│ - name           │
│ - model          │
│ - instructions   │
│ - tools          │
│ - evaluations───┼──────┐
│ - test_cases────┼────┐ │
└──────────────────┘    │ │
                        │ │
        ┌───────────────┘ │
        │                 │
        ▼                 ▼
┌──────────────────┐  ┌──────────────────┐
│  TestCaseModel   │  │ EvaluationConfig │  (Existing)
│                  │  │                  │
│ - name           │  │ - model          │
│ - input          │  │ - metrics[]──────┼─────┐
│ - files[]────────┼─┐└──────────────────┘     │
│ - expected_tools │ │                         │
│ - ground_truth   │ │                         ▼
│ - evaluations    │ │                  ┌──────────────────┐
└──────────────────┘ │                  │ EvaluationMetric │  (Existing)
                     │                  │                  │
                     ▼                  │ - metric         │
             ┌──────────────────┐      │ - threshold      │
             │    FileInput     │      │ - model          │
             │                  │      │ - retry_on_fail  │
             │ - path/url       │      └──────────────────┘
             │ - type           │
             │ - pages          │
             │ - sheet/range    │
             │ - cache          │
             └──────────────────┘
                     │
                     │  During Execution
                     ▼
             ┌──────────────────┐
             │ ProcessedFileInput│  (NEW)
             │                  │
             │ - original       │
             │ - markdown       │
             │ - metadata       │
             └──────────────────┘

        Test Execution
             │
             ▼
    ┌──────────────────┐
    │   TestResult     │  (NEW)
    │                  │
    │ - test_case      │
    │ - agent_response │
    │ - tool_calls[]   │
    │ - metrics[]──────┼───┐
    │ - passed         │   │
    │ - execution_time │   │
    │ - errors[]       │   │
    └──────────────────┘   │
             │             │
             │             ▼
             │      ┌──────────────────┐
             │      │  MetricResult    │  (NEW)
             │      │                  │
             │      │ - metric_name    │
             │      │ - score          │
             │      │ - threshold      │
             │      │ - passed         │
             │      │ - error          │
             │      └──────────────────┘
             ▼
    ┌──────────────────┐
    │   TestReport     │  (NEW)
    │                  │
    │ - agent_name     │
    │ - results[]      │
    │ - summary        │
    │ - timestamp      │
    │ - duration       │
    └──────────────────┘
             │
             ▼
    ┌──────────────────┐
    │  ReportSummary   │  (NEW)
    │                  │
    │ - total_tests    │
    │ - passed         │
    │ - failed         │
    │ - pass_rate      │
    └──────────────────┘
```

## Existing Models (Reference)

### TestCaseModel

**Location**: `src/holodeck/models/test_case.py`

**Purpose**: Represents a single test scenario

**Fields**:
- `name: str | None` - Test case identifier
- `input: str` - User query or prompt (required)
- `expected_tools: list[str] | None` - Tools expected to be called
- `ground_truth: str | None` - Expected output for comparison
- `files: list[FileInput] | None` - Multimodal file inputs
- `evaluations: list[str] | None` - Specific metrics for this test

**Validation**:
- input must be non-empty
- Maximum 10 files per test case
- name must be non-empty if provided

### FileInput

**Location**: `src/holodeck/models/test_case.py`

**Purpose**: File reference for multimodal test inputs

**Fields**:
- `path: str | None` - Local file path
- `url: str | None` - Remote URL
- `type: str` - File type (image, pdf, text, excel, word, powerpoint, csv)
- `description: str | None` - File description
- `pages: list[int] | None` - Specific pages/slides to extract
- `sheet: str | None` - Excel sheet name
- `range: str | None` - Excel cell range (e.g., A1:E100)
- `cache: bool | None` - Cache remote files (default true for URLs)

**Validation**:
- Exactly one of `path` or `url` must be provided
- type must be in valid set
- pages must be positive integers

### EvaluationConfig

**Location**: `src/holodeck/models/evaluation.py`

**Purpose**: Container for evaluation metrics

**Fields**:
- `model: LLMProvider | None` - Default LLM model for all metrics
- `metrics: list[EvaluationMetric]` - List of metrics to evaluate

**Validation**:
- metrics must have at least one metric

### EvaluationMetric

**Location**: `src/holodeck/models/evaluation.py`

**Purpose**: Single evaluation metric configuration

**Fields**:
- `metric: str` - Metric name (e.g., "groundedness")
- `threshold: float | None` - Minimum passing score
- `enabled: bool` - Whether metric is enabled (default: True)
- `scale: int | None` - Score scale (e.g., 5 for 1-5 scale)
- `model: LLMProvider | None` - LLM model override for this metric
- `fail_on_error: bool` - Fail test if metric evaluation fails (default: False)
- `retry_on_failure: int | None` - Number of retries on failure (1-3)
- `timeout_ms: int | None` - Timeout in milliseconds for LLM calls
- `custom_prompt: str | None` - Custom evaluation prompt

**Validation**:
- metric must be non-empty
- threshold must be numeric
- retry_on_failure must be between 1 and 3
- timeout_ms must be positive

---

## New Models

### ProcessedFileInput

**Location**: `src/holodeck/models/test_result.py` (NEW)

**Purpose**: Represents a processed file with extracted content

**Fields**:
```python
class ProcessedFileInput(BaseModel):
    """Processed file with extracted markdown content."""

    model_config = ConfigDict(extra="forbid")

    original: FileInput = Field(..., description="Original file input configuration")
    markdown_content: str = Field(..., description="Extracted markdown content from markitdown")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="File metadata (size, pages, format, etc.)"
    )
    cached_path: str | None = Field(
        None,
        description="Path to cached file if downloaded from URL"
    )
    processing_time_ms: int = Field(..., description="Time taken to process file in milliseconds")
    error: str | None = Field(None, description="Error message if processing failed")
```

**Validation**:
- markdown_content must be non-empty if no error
- processing_time_ms must be non-negative

**Usage**:
```python
# File processor creates this after markitdown conversion
processed = ProcessedFileInput(
    original=file_input,
    markdown_content=md_result.text_content,
    metadata={"size_bytes": 1024, "format": "pdf"},
    processing_time_ms=150
)
```

---

### MetricResult

**Location**: `src/holodeck/models/test_result.py` (NEW)

**Purpose**: Result of a single metric evaluation

**Fields**:
```python
class MetricResult(BaseModel):
    """Result of evaluating a single metric."""

    model_config = ConfigDict(extra="forbid")

    metric_name: str = Field(..., description="Name of the metric")
    score: float | None = Field(None, description="Metric score (None if error)")
    threshold: float | None = Field(None, description="Threshold for passing")
    passed: bool = Field(..., description="Whether metric passed threshold")
    scale: int | None = Field(None, description="Score scale (e.g., 5)")
    error: str | None = Field(None, description="Error message if evaluation failed")
    retry_count: int = Field(default=0, description="Number of retries attempted")
    evaluation_time_ms: int = Field(..., description="Time taken to evaluate in milliseconds")
    model_used: str | None = Field(None, description="Model used for evaluation (for AI metrics)")
```

**Validation**:
- metric_name must be non-empty
- score must be within scale range if both provided
- retry_count must be non-negative (max 3)
- evaluation_time_ms must be non-negative

**Usage**:
```python
# Azure AI Evaluation creates this
metric_result = MetricResult(
    metric_name="groundedness",
    score=0.85,
    threshold=0.7,
    passed=True,
    scale=1,
    evaluation_time_ms=1200,
    model_used="gpt-4o"
)
```

---

### TestResult

**Location**: `src/holodeck/models/test_result.py` (NEW)

**Purpose**: Complete result of executing a single test case

**Fields**:
```python
class TestResult(BaseModel):
    """Result of executing a single test case."""

    model_config = ConfigDict(extra="forbid")

    test_name: str | None = Field(None, description="Test case name")
    test_input: str = Field(..., description="Original test input")
    processed_files: list[ProcessedFileInput] = Field(
        default_factory=list,
        description="Processed file inputs with extracted content"
    )

    # Agent execution results
    agent_response: str | None = Field(None, description="Agent's response")
    tool_calls: list[str] = Field(
        default_factory=list,
        description="List of tool names called by agent"
    )
    expected_tools: list[str] | None = Field(
        None,
        description="Expected tools from test case"
    )
    tools_matched: bool | None = Field(
        None,
        description="Whether tool calls matched expected_tools"
    )

    # Evaluation results
    metric_results: list[MetricResult] = Field(
        default_factory=list,
        description="Results for each evaluated metric"
    )
    ground_truth: str | None = Field(None, description="Expected output for comparison")

    # Overall status
    passed: bool = Field(..., description="Whether test passed all thresholds")
    execution_time_ms: int = Field(..., description="Total test execution time")
    errors: list[str] = Field(
        default_factory=list,
        description="List of errors encountered during execution"
    )

    timestamp: str = Field(..., description="ISO 8601 timestamp of execution")
```

**Validation**:
- test_input must be non-empty
- execution_time_ms must be non-negative
- If expected_tools provided, tools_matched must not be None
- passed is False if any metric failed or errors exist

**Computed Fields**:
```python
@property
def metrics_passed(self) -> int:
    """Number of metrics that passed."""
    return sum(1 for m in self.metric_results if m.passed)

@property
def metrics_total(self) -> int:
    """Total number of metrics evaluated."""
    return len(self.metric_results)
```

---

### ReportSummary

**Location**: `src/holodeck/models/test_result.py` (NEW)

**Purpose**: Aggregate statistics for test execution

**Fields**:
```python
class ReportSummary(BaseModel):
    """Summary statistics for test execution."""

    model_config = ConfigDict(extra="forbid")

    total_tests: int = Field(..., description="Total number of tests executed")
    passed: int = Field(..., description="Number of tests that passed")
    failed: int = Field(..., description="Number of tests that failed")
    pass_rate: float = Field(..., description="Percentage of tests passed (0-100)")
    total_duration_ms: int = Field(..., description="Total execution time in milliseconds")

    # Metric-specific stats
    metrics_evaluated: dict[str, int] = Field(
        default_factory=dict,
        description="Count of each metric evaluated across all tests"
    )
    average_scores: dict[str, float] = Field(
        default_factory=dict,
        description="Average score for each metric across all tests"
    )
```

**Validation**:
- total_tests must equal passed + failed
- pass_rate must be between 0 and 100
- total_duration_ms must be non-negative

**Usage**:
```python
summary = ReportSummary(
    total_tests=10,
    passed=8,
    failed=2,
    pass_rate=80.0,
    total_duration_ms=45000,
    metrics_evaluated={"groundedness": 10, "relevance": 10},
    average_scores={"groundedness": 0.82, "relevance": 0.75}
)
```

---

### TestReport

**Location**: `src/holodeck/models/test_result.py` (NEW)

**Purpose**: Complete test execution report with all results

**Fields**:
```python
class TestReport(BaseModel):
    """Complete test execution report."""

    model_config = ConfigDict(extra="forbid")

    agent_name: str = Field(..., description="Name of agent being tested")
    agent_config_path: str = Field(..., description="Path to agent.yaml file")

    results: list[TestResult] = Field(..., description="Individual test results")
    summary: ReportSummary = Field(..., description="Aggregate summary statistics")

    timestamp: str = Field(..., description="ISO 8601 timestamp of report generation")
    holodeck_version: str = Field(..., description="HoloDeck CLI version")

    # Optional metadata
    environment: dict[str, str] = Field(
        default_factory=dict,
        description="Environment information (Python version, OS, etc.)"
    )
```

**Validation**:
- results must not be empty
- summary.total_tests must equal len(results)
- timestamp must be valid ISO 8601 format

**JSON Serialization**:
```python
def to_json(self, indent: int = 2) -> str:
    """Serialize to JSON string."""
    return self.model_dump_json(indent=indent, exclude_none=True)

def to_file(self, file_path: str) -> None:
    """Write report to JSON file."""
    with open(file_path, "w") as f:
        f.write(self.to_json())
```

**Markdown Serialization**:
```python
def to_markdown(self) -> str:
    """Generate markdown report."""
    # See reporter.py for implementation
    pass
```

---

## Model Relationships

### Configuration → Execution Flow

1. **AgentConfig** is loaded from `agent.yaml`
2. **TestCaseModel** instances are extracted from `AgentConfig.test_cases`
3. **FileInput** instances are extracted from each `TestCaseModel.files`
4. **EvaluationConfig** and **EvaluationMetric** instances define evaluation strategy

### Execution → Result Flow

1. **FileInput** → **ProcessedFileInput** (via markitdown)
2. **TestCaseModel** + **ProcessedFileInput** → Agent execution → **TestResult**
3. **EvaluationMetric** + Agent response → **MetricResult**
4. **MetricResult** instances → **TestResult.metric_results**
5. **TestResult** instances → **TestReport.results**
6. **TestReport** → **ReportSummary** (computed)

---

## State Transitions

### Test Execution State Machine

```
PENDING
   │
   ├─ Load files ──> PROCESSING_FILES
   │                      │
   │                      ├─ Success ──> FILES_READY
   │                      └─ Error ────> FILES_FAILED
   │
   ├─ Execute agent ──> EXECUTING_AGENT
   │                      │
   │                      ├─ Success ──> AGENT_COMPLETE
   │                      └─ Error ────> AGENT_FAILED
   │
   └─ Evaluate metrics ──> EVALUATING
                              │
                              ├─ Success ──> COMPLETE (passed/failed)
                              └─ Error ────> EVALUATION_FAILED
```

### Metric Evaluation State Machine

```
PENDING
   │
   ├─ Call evaluator ──> EVALUATING
                            │
                            ├─ Success ──> COMPLETE (passed/failed)
                            ├─ Timeout ──> RETRY (attempt 1-3)
                            └─ Error ────> RETRY (attempt 1-3)
                                             │
                                             └─ Max retries ──> FAILED
```

---

## Validation Rules

### Cross-Model Validation

1. **Tool Call Validation**:
   - If `TestCaseModel.expected_tools` is provided
   - Then `TestResult.tool_calls` must match expected_tools
   - `TestResult.tools_matched` reflects this validation

2. **Metric Threshold Validation**:
   - If `EvaluationMetric.threshold` is provided
   - Then `MetricResult.score >= threshold` determines `MetricResult.passed`

3. **Test Pass/Fail Logic**:
   ```python
   test_passed = (
       len(errors) == 0 and
       all(m.passed for m in metric_results) and
       (tools_matched is None or tools_matched is True)
   )
   ```

4. **Summary Calculation**:
   ```python
   summary = ReportSummary(
       total_tests=len(results),
       passed=sum(1 for r in results if r.passed),
       failed=sum(1 for r in results if not r.passed),
       pass_rate=(passed / total_tests * 100),
       total_duration_ms=sum(r.execution_time_ms for r in results)
   )
   ```

---

## Serialization Formats

### JSON Schema

All models support JSON serialization via Pydantic's `model_dump_json()`:

```json
{
  "agent_name": "Customer Support Agent",
  "results": [
    {
      "test_name": "Business hours query",
      "test_input": "What are your business hours?",
      "agent_response": "We're open Monday-Friday 9AM-5PM EST",
      "tool_calls": ["get_hours"],
      "expected_tools": ["get_hours"],
      "tools_matched": true,
      "metric_results": [
        {
          "metric_name": "groundedness",
          "score": 0.95,
          "threshold": 0.7,
          "passed": true,
          "evaluation_time_ms": 1200,
          "model_used": "gpt-4o"
        }
      ],
      "passed": true,
      "execution_time_ms": 3500,
      "timestamp": "2025-11-01T14:30:00Z"
    }
  ],
  "summary": {
    "total_tests": 1,
    "passed": 1,
    "failed": 0,
    "pass_rate": 100.0
  }
}
```

### Markdown Format

Generated by `reporter.py`:

```markdown
# Test Report: Customer Support Agent

**Date**: 2025-11-01 14:30:00 UTC
**HoloDeck Version**: 0.1.0
**Tests**: 10 passed, 0 failed (100% pass rate)
**Duration**: 45.0s

## Summary

| Metric | Avg Score | Pass Rate |
|--------|-----------|-----------|
| groundedness | 0.85 | 90% |
| relevance | 0.78 | 100% |

## Test Results

### ✅ Test 1: Business hours query (PASSED)

**Input**: What are your business hours?
**Response**: We're open Monday-Friday 9AM-5PM EST
**Tools Called**: get_hours ✅ (matched expected)
**Execution Time**: 3.5s

**Metrics**:
- Groundedness: 0.95 / 0.70 ✅
- Relevance: 0.82 / 0.70 ✅
```

---

## File Locations

**New files to create**:
- `src/holodeck/models/test_result.py` - All new models (ProcessedFileInput, MetricResult, TestResult, ReportSummary, TestReport)

**Existing files (no changes)**:
- `src/holodeck/models/test_case.py` - TestCaseModel, FileInput
- `src/holodeck/models/evaluation.py` - EvaluationConfig, EvaluationMetric

---

## Usage Examples

### Creating Test Results

```python
from holodeck.models.test_result import TestResult, MetricResult

# After agent execution and evaluation
test_result = TestResult(
    test_name="Business hours query",
    test_input="What are your business hours?",
    agent_response="We're open Monday-Friday 9AM-5PM EST",
    tool_calls=["get_hours"],
    expected_tools=["get_hours"],
    tools_matched=True,
    metric_results=[
        MetricResult(
            metric_name="groundedness",
            score=0.95,
            threshold=0.7,
            passed=True,
            evaluation_time_ms=1200,
            model_used="gpt-4o"
        )
    ],
    passed=True,
    execution_time_ms=3500,
    timestamp="2025-11-01T14:30:00Z"
)
```

### Generating Report

```python
from holodeck.models.test_result import TestReport, ReportSummary

# After all tests complete
report = TestReport(
    agent_name="Customer Support Agent",
    agent_config_path="./agent.yaml",
    results=[test_result1, test_result2, ...],
    summary=ReportSummary(
        total_tests=10,
        passed=9,
        failed=1,
        pass_rate=90.0,
        total_duration_ms=45000
    ),
    timestamp="2025-11-01T14:35:00Z",
    holodeck_version="0.1.0"
)

# Export to JSON
report.to_file("test-report.json")

# Export to Markdown
markdown = report.to_markdown()
with open("test-report.md", "w") as f:
    f.write(markdown)
```
