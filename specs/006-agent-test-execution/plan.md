# Implementation Plan: Execute Agent Against Test Cases

**Branch**: `006-agent-test-execution` | **Date**: 2025-11-01 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/006-agent-test-execution/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Implement the `holodeck test` CLI command to execute agent test cases with evaluation metrics. This feature enables developers to validate agent behavior by running configured test cases (text-based and multimodal) against agents, evaluating responses using AI-powered and NLP metrics, and generating test reports. The implementation uses Semantic Kernel Agents for LLM interactions, markitdown for unified file processing, and Azure AI Evaluation SDK for metrics calculation.

## Technical Context

**Language/Version**: Python 3.10+

**Primary Dependencies**:

- Click (CLI framework - already in use)
- Pydantic (configuration models - already in use)
- PyYAML (YAML parsing - already in use)
- **Semantic Kernel**: Agent execution and LLM provider integrations
- **Azure AI Evaluation SDK**: AI-powered evaluation metrics (groundedness, relevance, coherence, etc.)
- **markitdown** (Microsoft): Unified file processing for PDF, images, Office documents (converts to markdown for LLM consumption)
- NLP metrics: nltk or evaluate library (for F1, BLEU, ROUGE, METEOR)
- HTTP client: requests (already in use)

**Storage**:

- File cache: `.holodeck/cache/` directory for remote URL downloads
- Test reports: JSON/Markdown files in user-specified output paths
- No database required for this feature

**Testing**: pytest with markers (@pytest.mark.unit, @pytest.mark.integration)

**Target Platform**: Cross-platform CLI (Linux, macOS, Windows)

**Project Type**: Single Python project with CLI interface

**Performance Goals**:

- Execute 10 test cases in <30 seconds (excluding LLM API latency)
- File processing timeout: 30s per file (configurable)
- LLM API timeout: 60s per call (configurable)
- File download timeout: 30s per URL (configurable)

**Constraints**:

- Sequential test execution (no parallelization in v1)
- Respect LLM token context window limits for large files
- Graceful degradation when evaluation metrics fail (retry up to 3 times with exponential backoff)
- Non-zero exit codes for CI/CD integration
- CI/CD-compatible output (no interactive elements in non-TTY environments)

**Scale/Scope**:

- Support up to 10 files per test case (enforced by existing TestCaseModel)
- Support 5+ file types via markitdown (PDF, image, Excel, Word, PowerPoint, CSV, HTML, etc.)
- Handle test suites with 100+ test cases
- Process files up to 100MB+ with warnings

## Constitution Check

_GATE: Must pass before Phase 0 research. Re-check after Phase 1 design._

### Principle I: No-Code-First Agent Definition

✅ **PASS** - Test execution reads agent configuration from YAML files. No code required from users to define test cases or evaluation metrics. All test configuration is declarative in agent.yaml.

### Principle II: MCP for API Integrations

✅ **PASS** - This feature integrates with agents that may use MCP tools, but test execution itself doesn't introduce new API integrations. LLM provider integrations use Semantic Kernel (which supports MCP protocol).

### Principle III: Test-First with Multimodal Support

✅ **PASS** - This feature IS the multimodal test execution engine. Implements support for images, PDFs, Office documents (via markitdown), validates expected_tools usage, and supports ground_truth comparison.

### Principle IV: OpenTelemetry-Native Observability

⚠️ **DEFERRED** - OpenTelemetry instrumentation is listed in Out of Scope for initial implementation. This aligns with project phase (v0.1 focusing on core functionality). Observability will be added in future iterations per VISION.md roadmap (v0.3+).

**Justification**: Core test execution capability takes priority for v0.1. Observability hooks can be added later without breaking test execution API.

### Principle V: Evaluation Flexibility with Model Overrides

✅ **PASS** - Implements three-level model configuration (global, per-evaluation, per-metric) as specified in EvaluationConfig and EvaluationMetric models. Uses Azure AI Evaluation SDK which follows Azure AI Evaluation patterns per constitution requirement.

### Architecture Constraints

✅ **PASS** - Implements Evaluation Framework engine component. Maintains separation from Agent Engine (receives agent responses via Semantic Kernel) and Deployment Engine (not involved in testing).

### Code Quality & Testing Discipline

✅ **PASS** - Follows Python 3.10+, Google Python Style Guide, MyPy strict mode, pytest with markers, 80% coverage minimum, Bandit/Safety/detect-secrets security scanning.

**Gate Status**: ✅ PASS (1 principle deferred with justification)

## Project Structure

### Documentation (this feature)

```
specs/006-agent-test-execution/
├── spec.md              # Feature specification
├── plan.md              # This file (/speckit.plan command output)
├── research.md          # Phase 0 output (/speckit.plan command)
├── data-model.md        # Phase 1 output (/speckit.plan command)
├── quickstart.md        # Phase 1 output (/speckit.plan command)
├── contracts/           # Phase 1 output (/speckit.plan command)
│   └── cli-api.md       # CLI command interface specification
├── tasks.md             # Phase 2 output (/speckit.tasks command - NOT created by /speckit.plan)
└── checklists/
    └── requirements.md  # Quality validation checklist
```

### Source Code (repository root)

```
src/holodeck/
├── cli/
│   ├── commands/
│   │   ├── init.py          # Existing: project initialization
│   │   └── test.py          # NEW: holodeck test command
│   └── main.py              # Update: register test command
│
├── models/
│   ├── test_case.py         # Existing: TestCaseModel, FileInput
│   ├── evaluation.py        # Existing: EvaluationMetric, EvaluationConfig
│   └── test_result.py       # NEW: TestResult, TestReport models
│
├── lib/
│   ├── file_processor.py    # NEW: markitdown integration for multimodal files
│   │
│   ├── test_runner/         # NEW: test execution engine
│   │   ├── __init__.py
│   │   ├── executor.py      # Main test execution orchestrator
│   │   ├── agent_bridge.py  # Semantic Kernel agent integration
│   │   ├── progress.py      # Progress indicators and output formatting
│   │   └── reporter.py      # Report generation (JSON/Markdown)
│   │
│   └── evaluators/          # NEW: evaluation metrics
│       ├── __init__.py
│       ├── base.py          # Abstract evaluator interface
│       ├── azure_ai.py      # Azure AI Evaluation SDK integration
│       └── nlp_metrics.py   # NLP metrics (F1, BLEU, ROUGE, METEOR)
│
└── config/
    └── defaults.py          # Update: add test execution defaults (timeouts, cache location)

tests/
├── unit/
│   ├── cli/
│   │   └── commands/
│   │       └── test_test.py              # NEW: test command unit tests
│   ├── lib/
│   │   ├── test_file_processor.py        # NEW: markitdown integration unit tests
│   │   ├── test_runner/                  # NEW: test runner unit tests
│   │   │   ├── test_executor.py
│   │   │   ├── test_agent_bridge.py
│   │   │   ├── test_progress.py
│   │   │   └── test_reporter.py
│   │   └── evaluators/                   # NEW: evaluator unit tests
│   │       ├── test_azure_ai.py
│   │       └── test_nlp_metrics.py
│   └── models/
│       └── test_test_result.py           # NEW: test result models unit tests
│
├── integration/
│   ├── test_basic_execution.py           # NEW: basic text test execution
│   ├── test_multimodal_execution.py      # NEW: multimodal file test execution
│   ├── test_evaluation_metrics.py        # NEW: end-to-end metric evaluation
│   └── test_report_generation.py         # NEW: report generation integration
│
└── fixtures/
    ├── agents/
    │   └── test_agent.yaml               # NEW: sample agent for testing
    ├── files/                            # NEW: multimodal test files
    │   ├── sample.pdf
    │   ├── sample.jpg
    │   ├── sample.xlsx
    │   ├── sample.docx
    │   └── sample.pptx
    └── expected_reports/                 # NEW: expected report outputs
        ├── sample_json_report.json
        └── sample_markdown_report.md
```

**Structure Decision**: Single Python project structure. This feature extends the existing HoloDeck CLI and library components. The test execution engine is organized into three main modules:

1. `file_processor.py` - Unified file processing via markitdown (handles all file types)
2. `test_runner/` - Orchestrates test execution (via Semantic Kernel), progress display, and reporting
3. `evaluators/` - Implements evaluation metrics (Azure AI Evaluation SDK + NLP libraries)

This structure is simplified compared to initial plans thanks to markitdown's unified file processing approach.

## Complexity Tracking

_No violations requiring justification. Principle IV (OpenTelemetry) is deferred, not violated - observability is out of scope for v0.1 per project roadmap._
