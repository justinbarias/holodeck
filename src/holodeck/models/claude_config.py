"""Claude Agent SDK-specific configuration models.

This module defines configuration models for the Claude Agent SDK integration,
including authentication, permissions, and capability settings.
All capabilities default to disabled (least-privilege).
"""

import warnings
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from holodeck.config.context import agent_base_dir

# Known built-in SDK tool names used for tool-name typo warnings (data-model.md rule 5).
KNOWN_BUILTIN_TOOLS: frozenset[str] = frozenset(
    {
        "Read",
        "Write",
        "Edit",
        "Bash",
        "Glob",
        "Grep",
        "WebSearch",
        "WebFetch",
        "Task",
        "TodoWrite",
        "NotebookEdit",
    }
)


class AuthProvider(str, Enum):
    """Authentication method for Anthropic provider."""

    api_key = "api_key"
    oauth_token = "oauth_token"  # noqa: S105  # nosec B105
    bedrock = "bedrock"
    vertex = "vertex"
    foundry = "foundry"
    custom = "custom"  # ANTHROPIC_AUTH_TOKEN for third-party endpoints


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


class SubagentSpec(BaseModel):
    """A single subagent definition for multi-agent orchestration.

    Each entry under ``claude.agents`` becomes an
    ``claude_agent_sdk.types.AgentDefinition`` on
    ``ClaudeAgentOptions.agents``. The SDK uses ``description`` for routing.

    US3 validators (prompt/prompt_file mutual exclusion and file resolution)
    are deliberately omitted here — they belong to the US3 task slice.
    """

    model_config = ConfigDict(extra="forbid")

    description: str = Field(
        description=(
            "Human-readable description used by the parent agent for routing "
            "decisions. Required by the SDK."
        )
    )
    prompt: str | None = Field(
        default=None,
        description=(
            "Inline system prompt for the subagent. "
            "Mutually exclusive with prompt_file."
        ),
    )
    prompt_file: str | None = Field(
        default=None,
        description=(
            "Path to a file containing the subagent's system prompt. "
            "Resolved relative to the agent.yaml directory. "
            "Mutually exclusive with prompt. "
            "File contents are inlined at config-load time."
        ),
    )
    tools: list[str] | None = Field(
        default=None,
        description=(
            "Allowlist of tool names. When null/omitted, the subagent "
            "inherits all parent tools."
        ),
    )
    model: Literal["sonnet", "opus", "haiku", "inherit"] | None = Field(
        default=None,
        description=(
            "Model override for this subagent. Must be one of the SDK's "
            "literal aliases: 'sonnet', 'opus', 'haiku', 'inherit'. Full "
            "model IDs (e.g. 'claude-haiku-4-5') are rejected — the Claude "
            "CLI silently drops AgentDefinitions whose `model` is not in "
            "this set, so the validator catches it at config-load time."
        ),
    )

    @model_validator(mode="after")
    def _validate_description_non_empty(self) -> "SubagentSpec":
        """Validate description is non-empty after strip (data-model.md rule 4)."""
        if not self.description.strip():
            raise ValueError("subagent requires description")
        return self

    @model_validator(mode="after")
    def _resolve_prompt_sources(self) -> "SubagentSpec":
        """Enforce prompt/prompt_file rules and inline prompt_file contents.

        Order matters: mutual-exclusion → at-least-one → file resolution →
        post-strip non-empty check. After this validator returns, ``prompt`` is
        always a non-empty string and ``prompt_file`` is always ``None``
        (data-model.md §1 invariants).
        """
        if self.prompt is not None and self.prompt_file is not None:
            raise ValueError("prompt and prompt_file are mutually exclusive")
        if self.prompt is None and self.prompt_file is None:
            raise ValueError("subagent requires either prompt or prompt_file")

        if self.prompt_file is not None:
            base_dir_value = agent_base_dir.get()
            base_dir = Path.cwd() if base_dir_value is None else Path(base_dir_value)
            path = Path(self.prompt_file)
            if not path.is_absolute():
                path = base_dir / path
            if not path.exists():
                raise ValueError(f"prompt_file not found: {path}")
            self.prompt = path.read_text(encoding="utf-8")
            self.prompt_file = None

        if self.prompt is not None and not self.prompt.strip():
            raise ValueError("subagent prompt must be non-empty")
        return self

    @model_validator(mode="after")
    def _warn_unknown_tool_names(self) -> "SubagentSpec":
        """Emit UserWarning for tool names that don't match a known pattern.

        Three accepted patterns (research.md §3):
          1. Known SDK built-ins — names in ``KNOWN_BUILTIN_TOOLS``.
          2. MCP tool names — entries prefixed with ``mcp__``.
          3. HoloDeck-bridged tool names — entries registered in the parent
             agent's top-level ``tools`` field.

        NOTE: Pattern 3 (HoloDeck-bridged tools) cannot be checked here because
        this model validator runs in isolation and does not have access to the
        parent ``Agent`` context.  The warning message names all three accepted
        patterns so users understand bridged names are valid and won't trigger
        a hard error.  Cross-checking against the parent ``Agent.tools`` list is
        intentionally deferred — it would require ``Agent``-level context.

        Traces to data-model.md rule #5, research.md §3.
        """
        for entry in self.tools or []:
            if entry in KNOWN_BUILTIN_TOOLS:
                continue
            if entry.startswith("mcp__"):
                continue
            warnings.warn(
                f"Subagent tool '{entry}' does not match a known built-in "
                f"(one of {sorted(KNOWN_BUILTIN_TOOLS)}), an MCP tool name "
                f"(mcp__<server>__<tool>), or a HoloDeck-bridged tool. "
                f"This may be a typo.",
                UserWarning,
                stacklevel=2,
            )
        return self


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
    max_concurrent_sessions: int | None = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum concurrent Claude SDK subprocesses per serve instance",
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
    agents: dict[str, SubagentSpec] | None = Field(
        default=None,
        description=(
            "Named subagent definitions. Each entry becomes an AgentDefinition "
            "on ClaudeAgentOptions.agents. Subagents share parent MCP servers "
            "and use their `tools` allowlist to scope MCP tool access."
        ),
    )
    allowed_tools: list[str] | None = Field(
        default=None,
        description="Explicit tool allowlist. None = all configured tools.",
    )
    effort: Literal["low", "medium", "high", "max"] | None = Field(
        default=None,
        description="Thinking effort level. SDK accepts low|medium|high|max.",
    )
    max_budget_usd: float | None = Field(
        default=None,
        gt=0,
        description="Hard cap on session spend in USD. Must be > 0.",
    )
    fallback_model: str | None = Field(
        default=None,
        description="Model used when the primary model is unavailable.",
    )
    disallowed_tools: list[str] | None = Field(
        default=None,
        description=(
            "Tools that must never be used; takes precedence over allowed_tools."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _reject_legacy_subagents_block(cls, data: Any) -> Any:
        """Raise a targeted error when the legacy claude.subagents block is present.

        Runs before extra="forbid" so the user gets a helpful message instead of
        a generic "extra fields not permitted" error. Traces to FR-011, SC-005.
        """
        if isinstance(data, dict) and "subagents" in data:
            raise ValueError(
                "`claude.subagents` is no longer supported; remove this block. "
                "Subagent forwarding is gated solely by the presence of "
                "`claude.agents`. To cap HoloDeck-side test concurrency, set "
                "`execution.parallel_test_cases` instead."
            )
        return data

    @model_validator(mode="after")
    def _normalize_agents_empty_map(self) -> "ClaudeConfig":
        """Normalize agents={} to agents=None (data-model.md rule 1)."""
        if self.agents == {}:
            self.agents = None
        return self

    @model_validator(mode="after")
    def _warn_effort_with_extended_thinking(self) -> "ClaudeConfig":
        if (
            self.effort is not None
            and self.extended_thinking is not None
            and self.extended_thinking.enabled
        ):
            warnings.warn(
                "Both `effort` and `extended_thinking` are set on ClaudeConfig. "
                "The Claude Agent SDK does not document precedence between "
                "`effort` and `thinking.budget_tokens`; pick one to avoid "
                "ambiguity.",
                UserWarning,
                stacklevel=2,
            )
        return self
