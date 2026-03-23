# Test Execution Framework API

The test runner orchestrates the complete test execution pipeline for HoloDeck agents,
from configuration resolution through agent invocation, evaluation, and result reporting.

The framework follows a sequential flow:

1. Load agent configuration from YAML
2. Resolve execution configuration (CLI > YAML > env > defaults)
3. Initialize components (FileProcessor, AgentFactory/Backend, Evaluators)
4. Execute each test case (file processing, agent invocation, tool validation, evaluation)
5. Generate a `TestReport` with summary statistics

---

## Executor

The executor module coordinates all stages of test execution. It owns configuration
resolution, evaluator creation, the agent invocation dispatch (backend or legacy
factory), and report generation.

### TestExecutor

::: holodeck.lib.test_runner.executor.TestExecutor
    options:
      docstring_style: google
      show_source: true

### validate_tool_calls

Standalone helper that checks actual tool calls against expected tool names using
substring matching. Returns `True`, `False`, or `None` (when validation is skipped).

::: holodeck.lib.test_runner.executor.validate_tool_calls
    options:
      docstring_style: google
      show_source: true

### RAGEvaluatorConstructor

Protocol that defines the common constructor signature shared by all RAG evaluator
classes (`FaithfulnessEvaluator`, `ContextualRelevancyEvaluator`, etc.). Used as the
value type in `RAG_EVALUATOR_MAP`.

::: holodeck.lib.test_runner.executor.RAGEvaluatorConstructor
    options:
      docstring_style: google
      show_source: true

### RAG_EVALUATOR_MAP

Module-level dictionary mapping `RAGMetricType` enum members to their evaluator
constructor. Eliminates repetitive `if/elif` chains when creating RAG evaluators.

::: holodeck.lib.test_runner.executor.RAG_EVALUATOR_MAP
    options:
      docstring_style: google
      show_source: true

---

## Agent Factory

The agent factory module provides Semantic Kernel-based agent creation, invocation
with timeout/retry logic, and response/tool-call extraction.

### AgentFactory

::: holodeck.lib.test_runner.agent_factory.AgentFactory
    options:
      docstring_style: google
      show_source: true

### AgentThreadRun

Encapsulates a single agent execution thread with an isolated `ChatHistory`.
Created by `AgentFactory.create_thread_run()` to ensure test-case isolation.

::: holodeck.lib.test_runner.agent_factory.AgentThreadRun
    options:
      docstring_style: google
      show_source: true

### AgentExecutionResult

Dataclass returned by `AgentThreadRun.invoke()` containing tool calls, tool results,
the full conversation history, optional token usage, and the extracted response text.

::: holodeck.lib.test_runner.agent_factory.AgentExecutionResult
    options:
      docstring_style: google
      show_source: true

---

## Reporter

Generates comprehensive Markdown reports from `TestReport` objects, including summary
tables, per-test sections, metric details, tool-usage validation, and file metadata.

### generate_markdown_report

::: holodeck.lib.test_runner.reporter.generate_markdown_report
    options:
      docstring_style: google
      show_source: true

---

## Progress

Real-time progress display with TTY detection. Interactive terminals get colored
symbols and spinners; CI/CD environments get plain-text output compatible with log
aggregation systems.

### ProgressIndicator

::: holodeck.lib.test_runner.progress.ProgressIndicator
    options:
      docstring_style: google
      show_source: true

---

## Eval Kwargs Builder

Type-safe construction of evaluation keyword arguments based on each evaluator's
`ParamSpec`. Handles the parameter-name divergence between evaluator families
(Azure AI/NLP use `response`/`query`; DeepEval uses `actual_output`/`input`).

### EvalKwargsBuilder

::: holodeck.lib.test_runner.eval_kwargs_builder.EvalKwargsBuilder
    options:
      docstring_style: google
      show_source: true

### build_retrieval_context_from_tools

Extracts retrieval context strings from tool results, filtering to only those tools
marked as retrieval tools.

::: holodeck.lib.test_runner.eval_kwargs_builder.build_retrieval_context_from_tools
    options:
      docstring_style: google
      show_source: true

---

## Example Usage

```python
from holodeck.lib.test_runner.executor import TestExecutor
from holodeck.lib.test_runner.reporter import generate_markdown_report
from holodeck.lib.test_runner.progress import ProgressIndicator
from holodeck.config.loader import ConfigLoader

# Load agent configuration
loader = ConfigLoader()
agent = loader.load_agent_yaml("agent.yaml")

# Set up progress tracking
progress = ProgressIndicator(total_tests=len(agent.test_cases or []))

# Create executor with progress callback
executor = TestExecutor(
    agent_config_path="agent.yaml",
    progress_callback=progress.update,
    on_test_start=lambda tc: progress.start_test(tc.name or "unnamed"),
)

# Run all test cases
report = await executor.execute_tests()

# Display summary
print(progress.get_summary())

# Generate markdown report
markdown = generate_markdown_report(report)
with open("report.md", "w") as f:
    f.write(markdown)
```

### Using EvalKwargsBuilder directly

```python
from holodeck.lib.test_runner.eval_kwargs_builder import (
    EvalKwargsBuilder,
    build_retrieval_context_from_tools,
)

# Build retrieval context from tool results
retrieval_ctx = build_retrieval_context_from_tools(
    tool_results=[
        {"name": "search_kb", "result": "Refund policy allows 30-day returns."},
        {"name": "get_user", "result": "User: Alice"},
    ],
    retrieval_tool_names={"search_kb"},
)

# Build kwargs for an evaluator
builder = EvalKwargsBuilder(
    agent_response="We offer 30-day returns.",
    input_query="What is your refund policy?",
    ground_truth="30-day money-back guarantee on all products.",
    retrieval_context=retrieval_ctx,
)
kwargs = builder.build_for(evaluator)
result = await evaluator.evaluate(**kwargs)
```

---

## Related Documentation

- [Data Models](models.md) -- Test case and result Pydantic models
- [Evaluation Framework](evaluators.md) -- Metrics, evaluators, and `ParamSpec`
- [Configuration Loading](config-loader.md) -- `ConfigLoader` and resolution hierarchy
- [Backend Abstraction](backends.md) -- `AgentBackend`, `BackendSelector`, and `ExecutionResult`
