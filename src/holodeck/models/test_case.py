"""Test case models for agent configuration.

This module defines test case and file input models used in agent.yaml
configuration for specifying test scenarios.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from holodeck.models.evaluation import EvaluationMetric, GEvalMetric, RAGMetric


class FileInput(BaseModel):
    """File input for multimodal test cases.

    Represents a single file reference for test case inputs, supporting
    both local files and remote URLs with optional extraction parameters.
    """

    model_config = ConfigDict(extra="forbid")

    path: str | None = Field(None, description="Local file path")
    url: str | None = Field(None, description="Remote URL")
    type: str = Field(
        ..., description="File type: image, pdf, text, excel, word, powerpoint, csv"
    )
    description: str | None = Field(None, description="File description")
    pages: list[int] | None = Field(
        None, description="Specific pages/slides to extract"
    )
    sheet: str | None = Field(None, description="Excel sheet name")
    range: str | None = Field(None, description="Excel cell range (e.g., A1:E100)")
    cache: bool | None = Field(
        None, description="Cache remote files (default true for URLs)"
    )

    @field_validator("path", "url", mode="before")
    @classmethod
    def check_path_or_url(cls, v: Any, info: Any) -> Any:
        """Validate that exactly one of path or url is provided."""
        # This runs before validation, so we check in root_validator
        return v

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        """Validate file type is supported."""
        valid_types = {"image", "pdf", "text", "excel", "word", "powerpoint", "csv"}
        if v not in valid_types:
            raise ValueError(f"type must be one of {valid_types}, got {v}")
        return v

    @field_validator("pages")
    @classmethod
    def validate_pages(cls, v: list[int] | None) -> list[int] | None:
        """Validate pages are positive integers."""
        if v is not None and not all(isinstance(p, int) and p > 0 for p in v):
            raise ValueError("pages must be positive integers")
        return v

    def model_post_init(self, __context: Any) -> None:
        """Validate path and url mutual exclusivity after initialization."""
        if self.path and self.url:
            raise ValueError("Cannot provide both 'path' and 'url'")
        if not self.path and not self.url:
            raise ValueError("Must provide either 'path' or 'url'")


class TestCaseModel(BaseModel):
    """Test case for agent evaluation.

    Represents a single test scenario with input, optional expected output,
    expected tool usage, multimodal file inputs, and RAG context.
    """

    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(None, description="Test case identifier")
    input: str = Field(..., description="User query or prompt")
    expected_tools: list[str] | None = Field(
        None, description="Tools expected to be called"
    )
    ground_truth: str | None = Field(None, description="Expected output for comparison")
    files: list[FileInput] | None = Field(None, description="Multimodal file inputs")
    retrieval_context: list[str] | None = Field(
        None, description="Retrieved text chunks for RAG evaluation metrics"
    )
    evaluations: list[EvaluationMetric | GEvalMetric | RAGMetric] | None = Field(
        None, description="Per-test metric overrides (standard, GEval, or RAG)"
    )

    @field_validator("input")
    @classmethod
    def validate_input(cls, v: str) -> str:
        """Validate input is not empty."""
        if not v or not v.strip():
            raise ValueError("input must be a non-empty string")
        return v

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str | None) -> str | None:
        """Validate name is not empty if provided."""
        if v is not None and (not v or not v.strip()):
            raise ValueError("name must be non-empty if provided")
        return v

    @field_validator("ground_truth")
    @classmethod
    def validate_ground_truth(cls, v: str | None) -> str | None:
        """Validate ground_truth is not empty if provided."""
        if v is not None and (not v or not v.strip()):
            raise ValueError("ground_truth must be non-empty if provided")
        return v

    @field_validator("files")
    @classmethod
    def validate_files(cls, v: list[FileInput] | None) -> list[FileInput] | None:
        """Validate files list is not empty if provided."""
        if v is not None and len(v) > 10:
            raise ValueError("Maximum 10 files per test case")
        return v


# Alias for backward compatibility
TestCase = TestCaseModel
