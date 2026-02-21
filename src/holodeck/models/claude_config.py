"""Claude Agent SDK-specific configuration models.

This module defines configuration models for the Claude Agent SDK integration,
including authentication, permissions, and capability settings.
All capabilities default to disabled (least-privilege).
"""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class AuthProvider(str, Enum):
    """Authentication method for Anthropic provider."""

    api_key = "api_key"
    oauth_token = "oauth_token"  # noqa: S105  # nosec B105
    bedrock = "bedrock"
    vertex = "vertex"
    foundry = "foundry"


class PermissionMode(str, Enum):
    """Level of autonomous action for Claude Agent SDK."""

    manual = "manual"
    acceptEdits = "acceptEdits"  # noqa: N815
    acceptAll = "acceptAll"  # noqa: N815


class ExtendedThinkingConfig(BaseModel):
    """Extended reasoning (deep thinking) configuration."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    budget_tokens: int = Field(default=10_000, ge=1_000, le=100_000)


class BashConfig(BaseModel):
    """Shell command execution settings."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    excluded_commands: list[str] = Field(default_factory=list)
    allow_unsafe: bool = False


class FileSystemConfig(BaseModel):
    """File read/write/edit access settings."""

    model_config = ConfigDict(extra="forbid")

    read: bool = False
    write: bool = False
    edit: bool = False


class SubagentConfig(BaseModel):
    """Parallel sub-agent execution settings."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    max_parallel: int = Field(default=4, ge=1, le=16)


class ClaudeConfig(BaseModel):
    """Claude Agent SDK-specific settings.

    All fields optional. All capabilities default to disabled (least-privilege).
    """

    model_config = ConfigDict(extra="forbid")

    working_directory: str | None = Field(
        default=None,
        description="Scope agent file access to this path. Subprocess cwd.",
    )
    permission_mode: PermissionMode = Field(
        default=PermissionMode.manual,
        description="Level of autonomous action. Defaults to manual (safest).",
    )
    max_turns: int | None = Field(
        default=None,
        ge=1,
        description="Maximum agent loop iterations. None = SDK default.",
    )
    extended_thinking: ExtendedThinkingConfig | None = Field(
        default=None,
        description="Extended reasoning (deep thinking) configuration.",
    )
    web_search: bool = Field(
        default=False,
        description="Enable built-in web search capability.",
    )
    bash: BashConfig | None = Field(
        default=None,
        description="Shell command execution settings.",
    )
    file_system: FileSystemConfig | None = Field(
        default=None,
        description="File read/write/edit access settings.",
    )
    subagents: SubagentConfig | None = Field(
        default=None,
        description="Parallel sub-agent execution settings.",
    )
    allowed_tools: list[str] | None = Field(
        default=None,
        description="Explicit tool allowlist. None = all configured tools.",
    )
