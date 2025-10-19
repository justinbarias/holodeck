"""Data models and entities for HoloDeck configuration.

This package contains Pydantic models for all HoloDeck configuration entities,
including agents, tools, evaluations, test cases, and LLM providers.

All models enforce validation constraints and provide clear error messages
when configuration is invalid.
"""

from holodeck.models.agent import Agent, Instructions
from holodeck.models.config import DeploymentConfig, GlobalConfig, VectorstoreConfig
from holodeck.models.evaluation import EvaluationConfig, EvaluationMetric
from holodeck.models.llm import LLMProvider, ProviderEnum
from holodeck.models.test_case import FileInput, TestCase, TestCaseModel
from holodeck.models.tool import (
    FunctionTool,
    MCPTool,
    PromptTool,
    Tool,
    VectorstoreTool,
)

__all__: list[str] = [
    # Agent models
    "Agent",
    "Instructions",
    # Config models
    "GlobalConfig",
    "VectorstoreConfig",
    "DeploymentConfig",
    # LLM models
    "LLMProvider",
    "ProviderEnum",
    # Evaluation models
    "EvaluationConfig",
    "EvaluationMetric",
    # Test case models
    "TestCaseModel",
    "TestCase",
    "FileInput",
    # Tool models
    "Tool",
    "VectorstoreTool",
    "FunctionTool",
    "MCPTool",
    "PromptTool",
]
