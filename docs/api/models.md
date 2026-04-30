# Data Models API Reference

HoloDeck uses [Pydantic v2](https://docs.pydantic.dev/) models for all configuration
validation. This page documents the complete data model hierarchy, backend protocols,
and exception classes used throughout the platform.

---

## Agent Configuration

Root-level models that define an AI agent instance.

::: holodeck.models.agent.Agent
    options:
      docstring_style: google
      show_source: true

::: holodeck.models.agent.Instructions
    options:
      docstring_style: google
      show_source: true

---

## LLM Provider

Language model provider configuration supporting OpenAI, Azure OpenAI, Anthropic, and Ollama.

::: holodeck.models.llm.ProviderEnum
    options:
      docstring_style: google

::: holodeck.models.llm.LLMProvider
    options:
      docstring_style: google
      show_source: true

---

## Claude Agent SDK

Configuration models for the Claude Agent SDK integration.
All capabilities default to disabled (least-privilege).

::: holodeck.models.claude_config.AuthProvider
    options:
      docstring_style: google

::: holodeck.models.claude_config.PermissionMode
    options:
      docstring_style: google

::: holodeck.models.claude_config.ClaudeConfig
    options:
      docstring_style: google
      show_source: true

::: holodeck.models.claude_config.ExtendedThinkingConfig
    options:
      docstring_style: google
      show_source: true

::: holodeck.models.claude_config.BashConfig
    options:
      docstring_style: google
      show_source: true

::: holodeck.models.claude_config.FileSystemConfig
    options:
      docstring_style: google
      show_source: true

::: holodeck.models.claude_config.SubagentSpec
    options:
      docstring_style: google
      show_source: true

---

## Backend Abstraction Layer

Provider-agnostic protocols and data classes for agent execution are documented in the
[Backends API Reference](backends.md). Key types: `AgentBackend`, `AgentSession`,
`ContextGenerator`, `ExecutionResult`, `ToolEvent`.

---

## Token Usage

::: holodeck.models.token_usage.TokenUsage
    options:
      docstring_style: google
      show_source: true

---

## Tool Models

Six tool types are supported via a discriminated union (`ToolUnion`):
vectorstore, hierarchical document, function, MCP, prompt, and plugin.

### Base and Union

::: holodeck.models.tool.Tool
    options:
      docstring_style: google
      show_source: true

### Vectorstore Tool

::: holodeck.models.tool.VectorstoreTool
    options:
      docstring_style: google
      show_source: true

### Function Tool

::: holodeck.models.tool.FunctionTool
    options:
      docstring_style: google
      show_source: true

### MCP Tool

::: holodeck.models.tool.MCPTool
    options:
      docstring_style: google
      show_source: true

### Prompt Tool

::: holodeck.models.tool.PromptTool
    options:
      docstring_style: google
      show_source: true

### Hierarchical Document Tool

::: holodeck.models.tool.HierarchicalDocumentToolConfig
    options:
      docstring_style: google
      show_source: true

### Supporting Enums

::: holodeck.models.tool.TransportType
    options:
      docstring_style: google

::: holodeck.models.tool.CommandType
    options:
      docstring_style: google

::: holodeck.models.tool.SearchMode
    options:
      docstring_style: google

::: holodeck.models.tool.ChunkingStrategy
    options:
      docstring_style: google

::: holodeck.models.tool.DocumentDomain
    options:
      docstring_style: google

::: holodeck.models.tool.KeywordIndexProvider
    options:
      docstring_style: google

### Database and Index Configuration

::: holodeck.models.tool.DatabaseConfig
    options:
      docstring_style: google
      show_source: true

::: holodeck.models.tool.KeywordIndexConfig
    options:
      docstring_style: google
      show_source: true

---

## Tool Execution and Events

Runtime models for tracking tool execution status and streaming events.

::: holodeck.models.tool_execution.ToolStatus
    options:
      docstring_style: google

::: holodeck.models.tool_execution.ToolExecution
    options:
      docstring_style: google
      show_source: true

::: holodeck.models.tool_event.ToolEventType
    options:
      docstring_style: google

::: holodeck.models.tool_event.ToolEvent
    options:
      docstring_style: google
      show_source: true

---

## Evaluation Models

Metrics and evaluation framework configuration. Three metric families are
supported via a discriminated union (`MetricType`): standard NLP, G-Eval
custom criteria, and RAG pipeline metrics.

::: holodeck.models.evaluation.EvaluationConfig
    options:
      docstring_style: google
      show_source: true

::: holodeck.models.evaluation.EvaluationMetric
    options:
      docstring_style: google
      show_source: true

::: holodeck.models.evaluation.GEvalMetric
    options:
      docstring_style: google
      show_source: true

::: holodeck.models.evaluation.RAGMetric
    options:
      docstring_style: google
      show_source: true

::: holodeck.models.evaluation.RAGMetricType
    options:
      docstring_style: google

::: holodeck.models.evaluation.MetricType
    options:
      docstring_style: google

---

## Test Case Models

Test case definitions with multimodal file input support.

::: holodeck.models.test_case.TestCaseModel
    options:
      docstring_style: google
      show_source: true

::: holodeck.models.test_case.FileInput
    options:
      docstring_style: google
      show_source: true

---

## Test Result Models

Models for representing test execution results, metric outcomes, and reports.

::: holodeck.models.test_result.ProcessedFileInput
    options:
      docstring_style: google
      show_source: true

::: holodeck.models.test_result.MetricResult
    options:
      docstring_style: google
      show_source: true

::: holodeck.models.test_result.TestResult
    options:
      docstring_style: google
      show_source: true

::: holodeck.models.test_result.ReportSummary
    options:
      docstring_style: google
      show_source: true

::: holodeck.models.test_result.TestReport
    options:
      docstring_style: google
      show_source: true

---

## Chat Models

Interactive chat session and message models.

::: holodeck.models.chat.SessionState
    options:
      docstring_style: google

::: holodeck.models.chat.MessageRole
    options:
      docstring_style: google

::: holodeck.models.chat.Message
    options:
      docstring_style: google
      show_source: true

::: holodeck.models.chat.ChatSession
    options:
      docstring_style: google
      show_source: true

::: holodeck.models.chat.ChatConfig
    options:
      docstring_style: google
      show_source: true

---

## Global Configuration

Project-wide settings stored in `~/.holodeck/config.yaml` for sharing
defaults across multiple agents.

::: holodeck.models.config.ExecutionConfig
    options:
      docstring_style: google
      show_source: true

::: holodeck.models.config.VectorstoreConfig
    options:
      docstring_style: google
      show_source: true

::: holodeck.models.config.GlobalConfig
    options:
      docstring_style: google
      show_source: true

---

## Deployment Models

Configuration and result models for containerized agent deployment.
The canonical deployment config lives at
[`holodeck.models.deployment.DeploymentConfig`](#holodeck.models.deployment.DeploymentConfig).

### Enums

::: holodeck.models.deployment.TagStrategy
    options:
      docstring_style: google

::: holodeck.models.deployment.CloudProvider
    options:
      docstring_style: google

::: holodeck.models.deployment.RuntimeType
    options:
      docstring_style: google

::: holodeck.models.deployment.ProtocolType
    options:
      docstring_style: google

### Registry

::: holodeck.models.deployment.RegistryConfig
    options:
      docstring_style: google
      show_source: true

### Cloud Provider Targets

::: holodeck.models.deployment.AWSAppRunnerConfig
    options:
      docstring_style: google
      show_source: true

::: holodeck.models.deployment.GCPCloudRunConfig
    options:
      docstring_style: google
      show_source: true

::: holodeck.models.deployment.AzureContainerAppsConfig
    options:
      docstring_style: google
      show_source: true

::: holodeck.models.deployment.CloudTargetConfig
    options:
      docstring_style: google
      show_source: true

### Main Config and Results

::: holodeck.models.deployment.DeploymentConfig
    options:
      docstring_style: google
      show_source: true

::: holodeck.models.deployment.DeployResult
    options:
      docstring_style: google
      show_source: true

::: holodeck.models.deployment.StatusResult
    options:
      docstring_style: google
      show_source: true

---

## Deployment State

Persisted deployment records stored on disk for tracking active deployments.

::: holodeck.models.deployment_state.DeploymentRecord
    options:
      docstring_style: google
      show_source: true

::: holodeck.models.deployment_state.DeploymentState
    options:
      docstring_style: google
      show_source: true

---

## Observability Models

OpenTelemetry observability configuration following no-code-first principles.

::: holodeck.models.observability.ObservabilityConfig
    options:
      docstring_style: google
      show_source: true

::: holodeck.models.observability.TracingConfig
    options:
      docstring_style: google
      show_source: true

::: holodeck.models.observability.MetricsConfig
    options:
      docstring_style: google
      show_source: true

::: holodeck.models.observability.LogsConfig
    options:
      docstring_style: google
      show_source: true

::: holodeck.models.observability.LogLevel
    options:
      docstring_style: google

::: holodeck.models.observability.OTLPProtocol
    options:
      docstring_style: google

### Exporter Configuration

::: holodeck.models.observability.ExportersConfig
    options:
      docstring_style: google
      show_source: true

::: holodeck.models.observability.ConsoleExporterConfig
    options:
      docstring_style: google
      show_source: true

::: holodeck.models.observability.OTLPExporterConfig
    options:
      docstring_style: google
      show_source: true

::: holodeck.models.observability.PrometheusExporterConfig
    options:
      docstring_style: google
      show_source: true

::: holodeck.models.observability.AzureMonitorExporterConfig
    options:
      docstring_style: google
      show_source: true

---

## MCP Registry Models

Data models for the MCP Registry API at `registry.modelcontextprotocol.io`.

::: holodeck.models.registry.RegistryServer
    options:
      docstring_style: google
      show_source: true

::: holodeck.models.registry.RegistryServerPackage
    options:
      docstring_style: google
      show_source: true

::: holodeck.models.registry.RegistryServerMeta
    options:
      docstring_style: google
      show_source: true

::: holodeck.models.registry.ServerVersion
    options:
      docstring_style: google
      show_source: true

::: holodeck.models.registry.TransportConfig
    options:
      docstring_style: google

::: holodeck.models.registry.EnvVarConfig
    options:
      docstring_style: google

::: holodeck.models.registry.RepositoryInfo
    options:
      docstring_style: google

::: holodeck.models.registry.SearchResult
    options:
      docstring_style: google
      show_source: true

---

## Template Models

Models for template management, manifests, and project scaffolding.

::: holodeck.models.template_manifest.TemplateManifest
    options:
      docstring_style: google
      show_source: true

::: holodeck.models.template_manifest.VariableSchema
    options:
      docstring_style: google
      show_source: true

::: holodeck.models.template_manifest.FileMetadata
    options:
      docstring_style: google
      show_source: true

---

## Project Initialization Models

Models for the `holodeck init` command input and output.

::: holodeck.models.project_config.ProjectInitInput
    options:
      docstring_style: google
      show_source: true

::: holodeck.models.project_config.ProjectInitResult
    options:
      docstring_style: google
      show_source: true

---

## Wizard Models

Interactive initialization wizard state, choices, and results.

### State and Results

::: holodeck.models.wizard_config.WizardStep
    options:
      docstring_style: google

::: holodeck.models.wizard_config.WizardState
    options:
      docstring_style: google
      show_source: true

::: holodeck.models.wizard_config.WizardResult
    options:
      docstring_style: google
      show_source: true

::: holodeck.models.wizard_config.ProviderConfig
    options:
      docstring_style: google

### Choice Models

::: holodeck.models.wizard_config.TemplateChoice
    options:
      docstring_style: google

::: holodeck.models.wizard_config.LLMProviderChoice
    options:
      docstring_style: google
      show_source: true

::: holodeck.models.wizard_config.VectorStoreChoice
    options:
      docstring_style: google
      show_source: true

::: holodeck.models.wizard_config.EvalChoice
    options:
      docstring_style: google

::: holodeck.models.wizard_config.MCPServerChoice
    options:
      docstring_style: google

---

## Error Hierarchy

Custom exception classes from `holodeck.lib.errors` and
`holodeck.lib.backends.base`. All exceptions inherit from `HoloDeckError`.

### Base Exception

::: holodeck.lib.errors.HoloDeckError
    options:
      docstring_style: google
      show_source: true

### Configuration and Validation Errors

::: holodeck.lib.errors.ConfigError
    options:
      docstring_style: google
      show_source: true

::: holodeck.lib.errors.ValidationError
    options:
      docstring_style: google
      show_source: true

::: holodeck.lib.errors.FileNotFoundError
    options:
      docstring_style: google
      show_source: true

### Execution Errors

::: holodeck.lib.errors.ExecutionError
    options:
      docstring_style: google

::: holodeck.lib.errors.EvaluationError
    options:
      docstring_style: google

### Agent Errors

::: holodeck.lib.errors.AgentInitializationError
    options:
      docstring_style: google
      show_source: true

::: holodeck.lib.errors.AgentFactoryError
    options:
      docstring_style: google

### Chat Errors

::: holodeck.lib.errors.ChatValidationError
    options:
      docstring_style: google

::: holodeck.lib.errors.ChatSessionError
    options:
      docstring_style: google

### Ollama Errors

::: holodeck.lib.errors.OllamaConnectionError
    options:
      docstring_style: google
      show_source: true

::: holodeck.lib.errors.OllamaModelNotFoundError
    options:
      docstring_style: google
      show_source: true

### MCP Registry Errors

::: holodeck.lib.errors.RegistryConnectionError
    options:
      docstring_style: google
      show_source: true

::: holodeck.lib.errors.RegistryAPIError
    options:
      docstring_style: google
      show_source: true

::: holodeck.lib.errors.ServerNotFoundError
    options:
      docstring_style: google
      show_source: true

::: holodeck.lib.errors.DuplicateServerError
    options:
      docstring_style: google
      show_source: true

::: holodeck.lib.errors.RecordPathError
    options:
      docstring_style: google
      show_source: true

### Deployment Errors

::: holodeck.lib.errors.DeploymentError
    options:
      docstring_style: google
      show_source: true

::: holodeck.lib.errors.DockerNotAvailableError
    options:
      docstring_style: google
      show_source: true

::: holodeck.lib.errors.RegistryAuthError
    options:
      docstring_style: google
      show_source: true

::: holodeck.lib.errors.CloudSDKNotInstalledError
    options:
      docstring_style: google
      show_source: true

### Backend Errors

Backend-specific errors (`BackendError`, `BackendInitError`, `BackendSessionError`,
`BackendTimeoutError`) are documented in the [Backends API Reference](backends.md#exceptions).

---

## Related Documentation

- [Configuration Loading](config-loader.md): How to load and validate configurations
- [Test Runner](test-runner.md): Test execution framework using these models
- [Evaluation Framework](evaluators.md): Evaluation system using these models
