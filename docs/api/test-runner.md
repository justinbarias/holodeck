# Test Execution Framework API

The test runner orchestrates the complete test execution pipeline for HoloDeck agents,
from configuration resolution through evaluation and result reporting.

## Test Case Configuration

Configuration models for defining test cases with multimodal file support.

::: holodeck.models.test_case.TestCaseModel
    options:
      docstring_style: google
      show_source: true

::: holodeck.models.test_case.FileInput
    options:
      docstring_style: google
      show_source: true

## Test Results

Data models for test execution results and metrics.

::: holodeck.models.test_result.TestResult
    options:
      docstring_style: google
      show_source: true

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
