# Test Execution Framework API

The test runner orchestrates the complete test execution pipeline for HoloDeck agents,
from configuration resolution through evaluation and result reporting.

## Overview

The test execution pipeline consists of four main components:

1. **Executor**: Main orchestrator coordinating the entire test flow
2. **Agent Factory**: Creates agent instances from configuration
3. **Progress Indicator**: Provides real-time feedback during test execution
4. **Reporter**: Generates structured test reports

## Test Executor

Main orchestrator that coordinates test execution, file processing, agent invocation,
evaluation, and report generation.

::: holodeck.lib.test_runner.executor.TestExecutor
    options:
      docstring_style: google
      show_source: true
      members:
        - run_tests
        - execute_test_case
        - validate_configuration

## Agent Factory

Creates and instantiates agents from configuration, resolving LLM providers and tools.

::: holodeck.lib.test_runner.agent_factory.AgentFactory
    options:
      docstring_style: google
      show_source: true
      members:
        - create_agent
        - create_from_config

## Progress Indicator

Real-time progress tracking and feedback during batch test execution.

::: holodeck.lib.test_runner.progress.ProgressIndicator
    options:
      docstring_style: google

## Test Reporter

Formats and generates test reports in multiple output formats.

::: holodeck.lib.test_runner.reporter.TestReporter
    options:
      docstring_style: google
      show_source: true
      members:
        - generate_report
        - format_results
        - save_report

## Example Usage

```python
from holodeck.lib.test_runner.executor import TestExecutor
from holodeck.config.loader import ConfigLoader

# Load agent configuration
loader = ConfigLoader()
config = loader.load("agent.yaml")

# Create executor and run tests
executor = TestExecutor()
results = executor.run_tests(config)

# Access results
for test_result in results.test_results:
    print(f"Test {test_result.test_name}: {test_result.status}")
    print(f"Metrics: {test_result.metrics}")
```

## Multimodal File Support

The test runner integrates with the file processor to handle:

- **Images**: JPG, PNG with OCR support
- **Documents**: PDF (full or page ranges), Word, PowerPoint
- **Data**: Excel (sheet/range selection), CSV, text files
- **Remote Files**: URL-based inputs with caching

Files are automatically processed before agent invocation and included in test context.

## Integration with Evaluation Framework

Test results automatically pass through the evaluation framework:

1. **NLP Metrics**: Computed on all test outputs (F1, BLEU, ROUGE, METEOR)
2. **AI-powered Metrics**: Optional evaluation by Azure AI models
3. **Custom Metrics**: User-defined evaluation functions

Evaluation configuration comes from the agent's `evaluations` section.

## Related Documentation

- [Data Models](models.md): Test case and result models
- [Evaluation Framework](evaluators.md): Metrics and evaluation system
- [Configuration Loading](config-loader.md): Loading agent configurations
