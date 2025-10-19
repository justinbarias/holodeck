"""Agent model for agent configuration.

This module defines the main Agent model and related configuration used
in agent.yaml files.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from holodeck.models.evaluation import EvaluationConfig
from holodeck.models.llm import LLMProvider
from holodeck.models.test_case import TestCaseModel


class Instructions(BaseModel):
    """System instructions specification (file or inline).

    Represents the system prompt for an agent, supporting both
    file references and inline text.
    """

    model_config = ConfigDict(extra="forbid")

    file: str | None = Field(None, description="Path to instruction file")
    inline: str | None = Field(None, description="Inline instruction text")

    @field_validator("file")
    @classmethod
    def validate_file(cls, v: str | None) -> str | None:
        """Validate file is not empty if provided."""
        if v is not None and (not v or not v.strip()):
            raise ValueError("file must be non-empty if provided")
        return v

    @field_validator("inline")
    @classmethod
    def validate_inline(cls, v: str | None) -> str | None:
        """Validate inline is not empty if provided."""
        if v is not None and (not v or not v.strip()):
            raise ValueError("inline must be non-empty if provided")
        return v

    def model_post_init(self, __context: Any) -> None:
        """Validate file and inline mutual exclusivity."""
        if self.file and self.inline:
            raise ValueError("Cannot provide both 'file' and 'inline'")
        if not self.file and not self.inline:
            raise ValueError("Must provide either 'file' or 'inline'")


class Agent(BaseModel):
    """Agent configuration entity.

    Root configuration for a single AI agent instance, defining model,
    instructions, tools, evaluations, and test cases.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Agent identifier")
    description: str | None = Field(None, description="Human-readable description")
    model: LLMProvider = Field(..., description="LLM provider configuration")
    instructions: Instructions = Field(
        ..., description="System instructions (file or inline)"
    )
    tools: list[Any] | None = Field(
        None, description="Agent tools (vectorstore, function, mcp, prompt)"
    )
    evaluations: EvaluationConfig | None = Field(
        None, description="Evaluation configuration"
    )
    test_cases: list[TestCaseModel] | None = Field(None, description="Test scenarios")

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate name is not empty."""
        if not v or not v.strip():
            raise ValueError("name must be a non-empty string")
        return v

    @field_validator("description")
    @classmethod
    def validate_description(cls, v: str | None) -> str | None:
        """Validate description is not empty if provided."""
        if v is not None and (not v or not v.strip()):
            raise ValueError("description must be non-empty if provided")
        return v

    @field_validator("tools")
    @classmethod
    def validate_tools(cls, v: list[Any] | None) -> list[Any] | None:
        """Validate tools list."""
        if v is not None and len(v) > 50:
            raise ValueError("Maximum 50 tools per agent")
        return v

    @field_validator("test_cases")
    @classmethod
    def validate_test_cases(
        cls, v: list[TestCaseModel] | None
    ) -> list[TestCaseModel] | None:
        """Validate test cases list."""
        if v is not None and len(v) > 100:
            raise ValueError("Maximum 100 test cases per agent")
        return v
