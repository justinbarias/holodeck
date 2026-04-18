"""Pydantic models for persisted EvalRun artifacts.

An :class:`EvalRun` wraps the existing :class:`TestReport` and adds an
:class:`EvalRunMetadata` block that captures a redacted snapshot of the agent
configuration plus run provenance (prompt version, holodeck version, sanitized
CLI args, git commit). One ``EvalRun`` is persisted per ``holodeck test``
invocation.

See ``specs/031-eval-runs-dashboard/data-model.md`` for the full field
inventory.
"""

from __future__ import annotations

import re
import sys
from typing import Any, Literal

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from holodeck.models.agent import Agent
from holodeck.models.test_result import TestReport

_SHA256_HEX = re.compile(r"^[a-f0-9]{64}$")


class PromptVersion(BaseModel):
    """Identity of the prompt body at run time.

    US1 ships a stub of this model; US2 owns derivation from frontmatter.
    The stub still validates shape so that :class:`EvalRun` can reference it.
    """

    model_config = ConfigDict(extra="forbid")

    version: str = Field(
        ...,
        description=(
            "Manual value from frontmatter ``version:`` key, or auto-derived "
            "``auto-<sha256[:8]>`` when no manual value exists."
        ),
    )
    author: str | None = Field(
        default=None, description="From frontmatter ``author:`` key."
    )
    description: str | None = Field(
        default=None, description="From frontmatter ``description:`` key."
    )
    tags: list[str] = Field(
        default_factory=list, description="From frontmatter ``tags:`` key."
    )
    source: Literal["file", "inline"] = Field(
        ...,
        description=(
            "``file`` when derived from ``instructions.file``; ``inline`` otherwise."
        ),
    )
    file_path: str | None = Field(
        default=None,
        description="Absolute or agent-relative path; ``None`` when inline.",
    )
    body_hash: str = Field(
        ...,
        description=(
            "Full 64-char SHA-256 hex of the prompt body (frontmatter stripped)."
        ),
    )
    extra: dict[str, Any] = Field(
        default_factory=dict,
        description="Frontmatter keys outside the documented schema (FR-016).",
    )

    @field_validator("body_hash")
    @classmethod
    def _validate_body_hash(cls, v: str) -> str:
        if not _SHA256_HEX.match(v):
            raise ValueError("body_hash must be a 64-char lowercase hex SHA-256 digest")
        return v

    @model_validator(mode="after")
    def _validate_source_path_pairing(self) -> Self:
        if self.source == "inline" and self.file_path is not None:
            raise ValueError("file_path must be None when source='inline'")
        if self.source == "file" and not self.file_path:
            raise ValueError("file_path must be non-empty when source='file'")
        return self


class EvalRunMetadata(BaseModel):
    """Snapshot + provenance block paired with a :class:`TestReport`."""

    model_config = ConfigDict(extra="forbid")

    agent_config: Agent = Field(
        ...,
        description="Deep snapshot of the validated Agent model, post-redaction.",
    )
    prompt_version: PromptVersion = Field(
        ..., description="Prompt identity at run time."
    )
    holodeck_version: str = Field(
        ...,
        description="``importlib.metadata.version('holodeck-ai')`` or sentinel.",
    )
    cli_args: list[str] = Field(
        ...,
        description="Faithful echo of ``sys.argv[1:]`` (sanitisation reserved).",
    )
    git_commit: str | None = Field(
        default=None,
        description="Best-effort ``git rev-parse HEAD``; ``None`` when unavailable.",
    )


class EvalRun(BaseModel):
    """Persisted artifact for a single ``holodeck test`` invocation."""

    model_config = ConfigDict(extra="forbid")

    report: TestReport = Field(
        ...,
        description=(
            "Existing TestReport — additive TestResult fields apply transitively."
        ),
    )
    metadata: EvalRunMetadata = Field(
        ..., description="Snapshot of the agent config plus run provenance."
    )

    @model_validator(mode="after")
    def _validate_consistency(self) -> Self:
        if self.report.agent_name != self.metadata.agent_config.name:
            raise ValueError(
                "EvalRun consistency error: report.agent_name "
                f"({self.report.agent_name!r}) must equal "
                f"metadata.agent_config.name ({self.metadata.agent_config.name!r})"
            )
        return self
