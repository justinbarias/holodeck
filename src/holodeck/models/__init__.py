"""Data models and entities for HoloDeck configuration.

This package contains Pydantic models for all HoloDeck configuration entities,
including agents, tools, evaluations, test cases, and LLM providers.

All models enforce validation constraints and provide clear error messages
when configuration is invalid.
"""

from holodeck.models.agent import Agent, Instructions
from holodeck.models.chat import (
    ChatConfig,
    ChatSession,
    Message,
    MessageRole,
    SessionState,
)
from holodeck.models.config import DeploymentConfig, GlobalConfig, VectorstoreConfig
from holodeck.models.evaluation import EvaluationConfig, EvaluationMetric
from holodeck.models.llm import LLMProvider, ProviderEnum
from holodeck.models.observability import (
    AzureMonitorExporterConfig,
    ConsoleExporterConfig,
    ExportersConfig,
    LogLevel,
    LogsConfig,
    MetricsConfig,
    ObservabilityConfig,
    OTLPExporterConfig,
    OTLPProtocol,
    PrometheusExporterConfig,
    TracingConfig,
)
from holodeck.models.test_case import FileInput, TestCase, TestCaseModel
from holodeck.models.token_usage import TokenUsage
from holodeck.models.tool import (
    FunctionTool,
    MCPTool,
    PromptTool,
    Tool,
    ToolUnion,
    VectorstoreTool,
)
from holodeck.models.tool_execution import ToolExecution, ToolStatus

__all__: list[str] = [
    # Agent models
    "Agent",
    "Instructions",
    # Chat models
    "ChatConfig",
    "ChatSession",
    "Message",
    "MessageRole",
    "SessionState",
    # Config models
    "GlobalConfig",
    "VectorstoreConfig",
    "DeploymentConfig",
    # LLM models
    "LLMProvider",
    "ProviderEnum",
    # Token usage
    "TokenUsage",
    # Evaluation models
    "EvaluationConfig",
    "EvaluationMetric",
    # Test case models
    "TestCaseModel",
    "TestCase",
    "FileInput",
    # Tool models
    "Tool",
    "ToolUnion",
    "VectorstoreTool",
    "FunctionTool",
    "MCPTool",
    "PromptTool",
    # Tool execution
    "ToolExecution",
    "ToolStatus",
    # Observability models
    "ObservabilityConfig",
    "TracingConfig",
    "MetricsConfig",
    "LogsConfig",
    "LogLevel",
    "ExportersConfig",
    "ConsoleExporterConfig",
    "OTLPExporterConfig",
    "OTLPProtocol",
    "PrometheusExporterConfig",
    "AzureMonitorExporterConfig",
]
