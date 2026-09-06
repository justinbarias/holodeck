"""OpenAI Agents SDK-specific configuration models.

This module defines the configuration models for the OpenAI Agents SDK
backend (``model.provider: openai`` / ``azure_openai``). ``OpenAIConfig`` is
the sibling of ``ClaudeConfig`` — it carries serve sizing, the spec-026 config
mappings (``effort``, ``max_budget_usd``, ``fallback_model``, ``disallowed_tools``),
the safety gate, and hook/redaction opt-outs.

This module must NOT import the ``agents`` / ``openai`` SDK at any level: the
lazy-import gate (SC-005) requires every SDK import to live inside the backend
modules. ``agents`` (subagents / handoff targets, spec 035 FR-060 to FR-063)
lives here; the ``hooks`` sub-block is added in a later phase (E).
"""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, model_validator

from holodeck.config.context import agent_base_dir

# Claude model aliases (``claude.agents.<name>.model``) that are not portable
# to this backend (FR-062).
CLAUDE_MODEL_LITERALS: frozenset[str] = frozenset({"sonnet", "opus", "haiku"})


class OpenAISubagentSpec(BaseModel):
    """One ``openai.agents`` entry: a handoff-target sub-agent (FR-060).

    Each entry becomes an SDK ``Agent`` on the parent's ``handoffs`` with
    ``name=<key>``, ``instructions=<prompt>``, ``handoff_description=
    <description>``. The SDK exposes it to the parent model as a
    ``transfer_to_<name>`` tool.
    """

    model_config = ConfigDict(extra="forbid")

    description: str = Field(
        description=(
            "Handoff description shown to the parent model for routing "
            "decisions (SDK handoff_description)."
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
            "Path to a file containing the subagent's system prompt, resolved "
            "relative to the agent.yaml directory and inlined at config-load "
            "time. Mutually exclusive with prompt."
        ),
    )
    tools: list[str] | None = Field(
        default=None,
        description=(
            "Allowlist of parent tool names (as written in the parent's "
            "`tools:`) the subagent may use. When null/omitted the subagent "
            "inherits every parent tool and MCP server (FR-061). Names that "
            "match no parent tool fail load."
        ),
    )
    model: str | None = Field(
        default=None,
        description=(
            "Model for this subagent: 'inherit' (or omitted) uses the parent's "
            "model; any other string is passed to the SDK as a model "
            "identifier (an Azure deployment name for azure_openai). The "
            "Claude literals 'sonnet', 'opus', and 'haiku' are rejected."
        ),
    )
    skip_recommended_prefix: StrictBool = Field(
        default=False,
        description=(
            "When true, the SDK's RECOMMENDED_PROMPT_PREFIX is not prepended "
            "to this subagent's instructions (FR-063)."
        ),
    )

    @model_validator(mode="after")
    def _validate_description_non_empty(self) -> "OpenAISubagentSpec":
        """Reject a blank description."""
        if not self.description.strip():
            raise ValueError("subagent requires description")
        return self

    @model_validator(mode="after")
    def _resolve_prompt_sources(self) -> "OpenAISubagentSpec":
        """Enforce prompt/prompt_file rules and inline ``prompt_file`` contents.

        After this validator ``prompt`` is always a non-empty string and
        ``prompt_file`` is ``None`` (same invariant as ``claude.agents``).
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
    def _reject_claude_model_literals(self) -> "OpenAISubagentSpec":
        """Fail load on Claude model aliases (FR-062)."""
        if self.model is None:
            return self
        value = self.model.strip()
        if not value:
            raise ValueError("subagent model must be non-empty when set")
        if value in CLAUDE_MODEL_LITERALS:
            raise ValueError(
                f"subagent model '{value}' is a Claude model literal and is not "
                "portable to the openai_agents backend; use 'inherit' or an "
                "OpenAI model identifier / Azure deployment name"
            )
        self.model = value
        return self


class OpenAIPermissionsConfig(BaseModel):
    """Tool permission lists for the OpenAI Agents backend."""

    model_config = ConfigDict(extra="forbid")

    allowed_tools: list[str] | None = Field(
        default=None,
        description="Explicit tool allowlist. None = all configured tools.",
    )
    disallowed_tools: list[str] | None = Field(
        default=None,
        description=(
            "Tools that must never be used; takes precedence over allowed_tools."
        ),
    )


class OpenAIConfig(BaseModel):
    """OpenAI Agents SDK-specific settings.

    All fields optional. Applicable only when ``model.provider`` is ``openai``
    or ``azure_openai``.
    """

    model_config = ConfigDict(extra="forbid")

    max_concurrent_sessions: int | None = Field(
        default=None,
        ge=1,
        le=500,
        description=(
            "Maximum concurrent active turns per serve instance. When unset, "
            "the serve layer derives the cap from the replica's memory limit "
            "divided by `session_memory_estimate_mib`."
        ),
    )
    session_memory_estimate_mib: int = Field(
        default=100,
        ge=50,
        le=2000,
        description=(
            "Estimated peak resident memory (MiB) per concurrent active turn. "
            "Used by the serve layer to derive `max_concurrent_sessions` from "
            "the replica's memory limit. The openai_agents backend runs "
            "in-process (no per-turn subprocess), so the default is lower than "
            "the Claude backend's."
        ),
    )
    max_turns: int = Field(
        default=20,
        ge=1,
        description="Maximum agent loop iterations passed to Runner.run.",
    )
    i_understand_this_is_unsafe: bool = Field(
        default=False,
        description=(
            "Acknowledge that enabling unsafe hosted tools (CodeInterpreterTool) "
            "permits server-side code execution. Required to load such tools."
        ),
    )
    disable_default_hooks: StrictBool = Field(
        default=False,
        description=(
            "Disable HoloDeck-provided default guardrails (credential redaction "
            "output guardrail). When True, the agent runs with ONLY user-defined "
            "hooks. Loud warning emitted at load time. Note: OTel attribute "
            "redaction runs independently and is NOT disabled by this flag."
        ),
    )
    disable_subprocess_env_scrub: StrictBool = Field(
        default=False,
        description=(
            "Disable HoloDeck's default-on subprocess env scrubbing. When True, "
            "stdio MCP servers and shelling-out function tools inherit the full "
            "agent container env including provider credentials."
        ),
    )
    permissions: OpenAIPermissionsConfig | None = Field(
        default=None,
        description="Tool allow/deny lists for this backend.",
    )
    effort: Literal["low", "medium", "high", "max"] | None = Field(
        default=None,
        description=(
            "Reasoning effort level for reasoning models. Mapped to the SDK's "
            "ReasoningEffort (`max` → `xhigh`)."
        ),
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
            "Tools that must never be used; removed from the resolved agent at "
            "build time."
        ),
    )
    agents: dict[str, OpenAISubagentSpec] | None = Field(
        default=None,
        description=(
            "Named subagents that become SDK handoff targets on the parent "
            "agent (FR-060). Keys are the subagent names."
        ),
    )
