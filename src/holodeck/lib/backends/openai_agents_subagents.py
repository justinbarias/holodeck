"""Subagent and skill handoff targets for the OpenAI Agents backend.

Translates two HoloDeck surfaces into SDK ``Agent`` objects placed on the
parent agent's ``handoffs`` list (spec 035 FR-060 to FR-063, FR-070):

* ``openai.agents.<name>`` entries (:class:`OpenAISubagentSpec`) become
  ``Agent(name=<key>, instructions=<prompt>, handoff_description=
  <description>, tools=<resolved>, model=<resolved>, model_settings=
  <parent or per-model>, output_type=<parent>)``. The SDK's
  ``RECOMMENDED_PROMPT_PREFIX`` is prepended to the instructions once unless
  the entry sets ``skip_recommended_prefix: true``.
* ``type: skill`` tools (:class:`SkillTool`) become ``Agent(name=<skill
  name>, instructions=<inline instructions | SKILL.md body>,
  handoff_description=<description>, tools=<allowed_tools>, model=<parent>,
  model_settings=<parent>, output_type=<parent>)``.
  Skill instructions are used verbatim — no prefix — so the inline and
  file-based forms produce identical agents for identical content.

Targets carry the parent's ``output_type`` because the SDK takes the final
output from whichever agent finishes the run; without it a handed-off turn
would silently drop ``response_format``.

Tool resolution is by the parent's *config* names (as written under the
parent's ``tools:``), the same portable naming ``disallowed_tools`` uses.
A subagent with ``tools: null`` inherits every parent SDK tool and MCP
server (FR-061); an explicit list restricts it to the named entries and a
name that matches no parent tool fails load. A skill with ``allowed_tools``
unset or empty gets no tools.

Handoff-history shaping (``handoff_input_filter``, ``nest_handoff_history``)
stays at SDK defaults in v1.

Every ``import agents`` is performed lazily inside functions so importing this
module never pulls the SDK (SC-005).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from holodeck.lib.backends.openai_agents_tool_adapters import sdk_tool_name_for
from holodeck.lib.errors import ConfigError
from holodeck.lib.skills import SkillLoadError, load_skill_definition, resolve_skill_dir
from holodeck.models.tool import MCPTool, SkillTool, ToolUnion

if TYPE_CHECKING:  # pragma: no cover - typing only, no runtime SDK import
    from agents import Agent as SDKAgent
    from agents import Tool as SDKTool
    from agents.agent_output import AgentOutputSchemaBase
    from agents.mcp import MCPServer
    from agents.model_settings import ModelSettings
    from agents.models.interface import Model

    from holodeck.models.agent import Agent
    from holodeck.models.openai_config import OpenAISubagentSpec

logger = logging.getLogger(__name__)

ModelResolver = Callable[[str], "str | Model"]
ModelSettingsResolver = Callable[[str], "ModelSettings"]


@dataclass
class ParentToolSurface:
    """The parent agent's built tools, indexed by HoloDeck config name.

    Attributes:
        tools: Every SDK tool built for the parent, in build order.
        mcp_servers: Every connected SDK MCP server, in build order.
        tools_by_name: Config name to SDK tool, for tools that produced one.
        mcp_by_name: Config name to MCP server.
    """

    tools: list[SDKTool] = field(default_factory=list)
    mcp_servers: list[MCPServer] = field(default_factory=list)
    tools_by_name: dict[str, SDKTool] = field(default_factory=dict)
    mcp_by_name: dict[str, MCPServer] = field(default_factory=dict)

    @property
    def names(self) -> set[str]:
        """Config names that can be granted to a subagent or skill."""
        return set(self.tools_by_name) | set(self.mcp_by_name)


def index_parent_tools(
    tool_configs: list[ToolUnion] | None,
    sdk_tools: list[SDKTool],
    mcp_servers: list[MCPServer],
) -> ParentToolSurface:
    """Index the parent's built SDK tools and MCP servers by config name.

    Args:
        tool_configs: The parent's ``tools:`` list (may be ``None``).
        sdk_tools: The SDK tools ``build_sdk_tools`` produced (disallowed
            tools already omitted).
        mcp_servers: The connected SDK MCP servers (disallowed already
            omitted); each carries the config ``name``.

    Returns:
        A :class:`ParentToolSurface`.
    """
    sdk_by_name = {tool.name: tool for tool in sdk_tools}
    servers_by_name = {server.name: server for server in mcp_servers}
    surface = ParentToolSurface(tools=list(sdk_tools), mcp_servers=list(mcp_servers))
    for cfg in tool_configs or []:
        if isinstance(cfg, MCPTool):
            server = servers_by_name.get(cfg.name)
            if server is not None:
                surface.mcp_by_name[cfg.name] = server
            continue
        sdk_name = sdk_tool_name_for(cfg)
        if sdk_name is None:
            continue
        tool = sdk_by_name.get(sdk_name)
        if tool is None:
            # Not built (disallowed upstream) — leave it ungrantable.
            continue
        if cfg.name in surface.tools_by_name:  # pragma: no cover - guarded upstream
            raise ConfigError(
                f"tools.{cfg.name}",
                f"duplicate tool config name '{cfg.name}' while indexing the "
                "parent tool surface.",
            )
        surface.tools_by_name[cfg.name] = tool
    return surface


def _select_tools(
    surface: ParentToolSurface,
    names: list[str],
    *,
    owner: str,
) -> tuple[list[SDKTool], list[MCPServer]]:
    """Resolve an explicit tool-name list against the parent surface.

    Args:
        surface: The parent's indexed tools.
        names: Config names requested by the subagent / skill.
        owner: Config path used in the error message.

    Returns:
        ``(sdk_tools, mcp_servers)`` in the parent's build order.

    Raises:
        ConfigError: If any name matches no parent tool or MCP server.
    """
    unknown = sorted(name for name in names if name not in surface.names)
    if unknown:
        known = ", ".join(sorted(surface.names)) or "(none)"
        raise ConfigError(
            owner,
            f"tool name(s) not declared on the parent agent: {', '.join(unknown)}. "
            f"Available: {known}.",
        )
    wanted = set(names)
    tools = [tool for name, tool in surface.tools_by_name.items() if name in wanted]
    servers = [server for name, server in surface.mcp_by_name.items() if name in wanted]
    return tools, servers


def subagent_instructions(spec: OpenAISubagentSpec) -> str:
    """Return the subagent's instructions with the recommended prefix applied.

    The SDK's ``RECOMMENDED_PROMPT_PREFIX`` is prepended exactly once unless
    ``skip_recommended_prefix`` is true (FR-063). ``prompt`` is guaranteed
    non-empty by the config validator.

    Args:
        spec: The validated ``openai.agents`` entry.

    Returns:
        The instructions string to pass to the SDK ``Agent``.
    """
    prompt = spec.prompt or ""
    if spec.skip_recommended_prefix:
        return prompt
    from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX

    if prompt.startswith(RECOMMENDED_PROMPT_PREFIX):
        return prompt
    return f"{RECOMMENDED_PROMPT_PREFIX}\n\n{prompt}"


def build_handoff_agents(
    agent_config: Agent,
    *,
    parent_model: str | Model,
    parent_model_settings: ModelSettings,
    surface: ParentToolSurface,
    base_dir: Path | None,
    resolve_model: ModelResolver,
    resolve_model_settings: ModelSettingsResolver,
    output_type: AgentOutputSchemaBase | type[object] | None = None,
    disallowed: set[str] | None = None,
) -> list[SDKAgent]:
    """Build the parent's handoff-target ``Agent`` list.

    Subagents from ``openai.agents`` come first (declaration order), then
    skills from ``tools`` (declaration order, minus disallowed names). Every
    target receives the parent's ``output_type`` so a run that ends inside a
    handoff still honours ``response_format``, and the parent's
    ``model_settings`` (``model: inherit`` and skills) or settings rebuilt for
    the explicit model (so reasoning-model sampling rules apply per model).

    Args:
        agent_config: The HoloDeck agent config.
        parent_model: The parent's resolved SDK ``model=`` value; used for
            ``model: inherit`` subagents and for every skill.
        parent_model_settings: The parent's SDK ``ModelSettings``.
        surface: The parent's indexed tools (see :func:`index_parent_tools`).
        base_dir: Directory for resolving relative skill ``path`` values.
        resolve_model: Maps an explicit subagent model identifier to the SDK
            ``model=`` value for this provider (a plain string for OpenAI, a
            client-bound ``Model`` for Azure deployments).
        resolve_model_settings: Builds ``ModelSettings`` for an explicit
            subagent model identifier.
        output_type: The parent's SDK output type (structured output), or
            ``None``.
        disallowed: Config names to omit (skills listed here are dropped).

    Returns:
        SDK ``Agent`` objects ready to pass as ``Agent(handoffs=...)``.

    Raises:
        ConfigError: If a subagent or skill references an undeclared tool, a
            skill directory is invalid, or two targets would surface as the
            same ``transfer_to_`` tool (the SDK normalises names, so
            ``research_assistant`` and ``research-assistant`` collide).
    """
    from agents import Agent as SDKAgent

    blocked = disallowed or set()
    handoffs: list[SDKAgent] = []
    owners: list[str] = []

    openai_cfg = agent_config.openai
    entries = (openai_cfg.agents if openai_cfg is not None else None) or {}
    for name, spec in entries.items():
        if spec.tools is None:
            tools, servers = list(surface.tools), list(surface.mcp_servers)
        else:
            tools, servers = _select_tools(
                surface, spec.tools, owner=f"openai.agents.{name}.tools"
            )
        model: str | Model
        settings: ModelSettings
        if spec.model is None or spec.model == "inherit":
            model = parent_model
            settings = parent_model_settings
        else:
            model = resolve_model(spec.model)
            settings = resolve_model_settings(spec.model)
        handoffs.append(
            SDKAgent(
                name=name,
                instructions=subagent_instructions(spec),
                handoff_description=spec.description,
                tools=tools,
                mcp_servers=servers,
                model=model,
                model_settings=settings,
                output_type=output_type,
            )
        )
        owners.append(f"openai.agents.{name}")

    for cfg in agent_config.tools or []:
        if not isinstance(cfg, SkillTool):
            continue
        if cfg.name in blocked:
            logger.debug("Skipping disallowed skill '%s'.", cfg.name)
            continue
        owner = f"tools.{cfg.name}"
        instructions, description = _skill_content(cfg, base_dir)
        tools, servers = _select_tools(
            surface, cfg.allowed_tools or [], owner=f"{owner}.allowed_tools"
        )
        handoffs.append(
            SDKAgent(
                name=cfg.name,
                instructions=instructions,
                handoff_description=description,
                tools=tools,
                mcp_servers=servers,
                model=parent_model,
                model_settings=parent_model_settings,
                output_type=output_type,
            )
        )
        owners.append(owner)

    _reject_handoff_name_collisions(
        handoffs, owners, parent_tool_names={tool.name for tool in surface.tools}
    )
    return handoffs


def _reject_handoff_name_collisions(
    handoffs: list[SDKAgent],
    owners: list[str],
    parent_tool_names: set[str] | None = None,
) -> None:
    """Fail when a handoff tool name collides with another tool name.

    Two targets that normalise to the same SDK handoff tool name collide, and
    so does a handoff whose ``transfer_to_<name>`` shadows a tool the parent
    already exposes (a function tool literally named ``transfer_to_researcher``
    would otherwise be silently replaced by the handoff).

    Args:
        handoffs: The built targets.
        owners: Config path of each target, parallel to *handoffs*.
        parent_tool_names: SDK names of the parent's own tools.

    Raises:
        ConfigError: Naming both config entries and the shared tool name.
    """
    from agents.handoffs import Handoff

    seen: dict[str, str] = {}
    for target, owner in zip(handoffs, owners, strict=True):
        tool_name = str(Handoff.default_tool_name(target))
        if parent_tool_names and tool_name in parent_tool_names:
            raise ConfigError(
                owner,
                f"handoff target '{target.name}' surfaces as SDK tool "
                f"'{tool_name}', which the parent agent already declares as a "
                "tool; rename the tool or the handoff target.",
            )
        if tool_name in seen:
            raise ConfigError(
                owner,
                f"handoff target '{target.name}' and {seen[tool_name]} both "
                f"surface as SDK tool '{tool_name}'; handoff names must be "
                "unique after the SDK's function-style normalisation.",
            )
        seen[tool_name] = owner


def _skill_content(cfg: SkillTool, base_dir: Path | None) -> tuple[str, str]:
    """Return ``(instructions, description)`` for a skill config.

    Args:
        cfg: The validated skill tool.
        base_dir: Directory for resolving a relative ``path``.

    Returns:
        The instructions (inline or SKILL.md body) and the description (YAML
        value, falling back to the SKILL.md frontmatter).

    Raises:
        ConfigError: If the skill directory or SKILL.md is invalid.
    """
    if cfg.instructions is not None:
        return cfg.instructions, cfg.description or ""
    if cfg.path is None:  # pragma: no cover - the model validator enforces one form
        raise ConfigError(
            f"tools.{cfg.name}", "skill has neither instructions nor path"
        )
    try:
        definition = load_skill_definition(resolve_skill_dir(cfg.path, base_dir))
    except SkillLoadError as exc:
        raise ConfigError(f"tools.{cfg.name}.path", str(exc)) from exc
    description = cfg.description or definition.description
    return definition.instructions, description
