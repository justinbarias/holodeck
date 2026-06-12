"""OpenAI Agents SDK-specific configuration models.

This module defines the configuration models for the OpenAI Agents SDK
backend (``model.provider: openai`` / ``azure_openai``). ``OpenAIConfig`` is
the sibling of ``ClaudeConfig`` — it carries serve sizing, the spec-026 config
mappings (``effort``, ``max_budget_usd``, ``fallback_model``, ``disallowed_tools``),
the safety gate, and hook/redaction opt-outs.

This module must NOT import the ``agents`` / ``openai`` SDK at any level: the
lazy-import gate (SC-005) requires every SDK import to live inside the backend
modules. The ``hooks`` and ``agents`` sub-blocks are added in later phases
(E and D respectively).
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool


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
