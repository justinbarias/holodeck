# Data Models API Reference

HoloDeck uses Pydantic models for all configuration validation. This section documents
the complete data model hierarchy used throughout the platform.

## Agent Models

Core agent configuration and instruction models.

::: holodeck.models.agent.Agent
    options:
      docstring_style: google
      show_source: true
      members:
        - name
        - description
        - instructions
        - model
        - tools
        - evaluations
        - test_cases

::: holodeck.models.agent.Instructions
    options:
      docstring_style: google

## LLM Provider Models

Language model provider configuration for OpenAI, Azure OpenAI, and Anthropic.

::: holodeck.models.llm.ProviderEnum
    options:
      docstring_style: google

::: holodeck.models.llm.LLMProvider
    options:
      docstring_style: google
      show_source: true
      members:
        - provider
        - model
        - temperature
        - max_tokens
        - top_p

## Tool Models

Five types of tools are supported: vectorstore, function, MCP, prompt, and plugins.

::: holodeck.models.tool.Tool
    options:
      docstring_style: google

::: holodeck.models.tool.VectorstoreTool
    options:
      docstring_style: google
      show_source: true

::: holodeck.models.tool.FunctionTool
    options:
      docstring_style: google
      show_source: true

::: holodeck.models.tool.MCPTool
    options:
      docstring_style: google
      show_source: true

::: holodeck.models.tool.PromptTool
    options:
      docstring_style: google
      show_source: true

## Evaluation Models

Metrics and evaluation framework configuration.

::: holodeck.models.evaluation.EvaluationMetric
    options:
      docstring_style: google

::: holodeck.models.evaluation.EvaluationConfig
    options:
      docstring_style: google
      show_source: true
      members:
        - model
        - metrics
        - thresholds

## Test Case Models

Test case definitions with multimodal file input support.

::: holodeck.models.test_case.FileInput
    options:
      docstring_style: google
      show_source: true

::: holodeck.models.test_case.TestCaseModel
    options:
      docstring_style: google

::: holodeck.models.test_case.TestCase
    options:
      docstring_style: google
      show_source: true
      members:
        - name
        - input
        - files
        - expected_tools
        - ground_truth
        - metadata

## Global Configuration Models

Project-wide settings for vectorstore, deployment, and execution.

::: holodeck.models.config.VectorstoreConfig
    options:
      docstring_style: google

::: holodeck.models.config.DeploymentConfig
    options:
      docstring_style: google

::: holodeck.models.config.GlobalConfig
    options:
      docstring_style: google
      show_source: true

## Test Result Models

Models for representing test execution results and reports.

::: holodeck.models.test_result.TestResult
    options:
      docstring_style: google

::: holodeck.models.test_result.TestReport
    options:
      docstring_style: google

## Error Models

HoloDeck exception hierarchy for error handling.

::: holodeck.lib.errors.HoloDeckError
    options:
      docstring_style: google

::: holodeck.lib.errors.ConfigError
    options:
      docstring_style: google

::: holodeck.lib.errors.ValidationError
    options:
      docstring_style: google

::: holodeck.lib.errors.FileNotFoundError
    options:
      docstring_style: google

## Related Documentation

- [Configuration Loading](config-loader.md): How to load and validate configurations
- [Test Runner](test-runner.md): Test execution framework using these models
- [Evaluation Framework](evaluators.md): Evaluation system using these models
