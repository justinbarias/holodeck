"""Tests for ``openai_agents_subagents`` — handoff targets (FR-060–063, FR-070).

Builder-level tests construct real SDK ``Agent`` / ``FunctionTool`` objects
(no network) and assert the exact handoff list the parent receives:
inheritance, explicit restriction, model resolution, prefix handling, skill
inline/file equivalence, and load failures for undeclared tools.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from holodeck.config.context import agent_base_dir
from holodeck.lib.backends.openai_agents_subagents import (
    ParentToolSurface,
    build_handoff_agents,
    index_parent_tools,
    subagent_instructions,
)
from holodeck.lib.backends.openai_agents_tool_adapters import sdk_tool_name_for
from holodeck.lib.errors import ConfigError
from holodeck.models.agent import Agent, Instructions
from holodeck.models.llm import LLMProvider, ProviderEnum
from holodeck.models.openai_config import OpenAIConfig, OpenAISubagentSpec

SKILL_MD = """---
name: research-assistant
description: File description.
---
Search then summarise.
"""


def _sdk_tool(name: str) -> Any:
    from agents import FunctionTool as SDKFunctionTool

    async def _invoke(_ctx: Any, _input: str) -> str:
        return "ok"

    return SDKFunctionTool(
        name=name,
        description=name,
        params_json_schema={"type": "object", "properties": {}},
        on_invoke_tool=_invoke,
        strict_json_schema=False,
    )


def _sdk_model_base() -> type:
    from agents.models.interface import Model

    return Model


def _function_cfg(name: str) -> dict[str, Any]:
    return {
        "name": name,
        "type": "function",
        "description": name,
        "file": "tools.py",
        "function": name,
    }


def _vectorstore_cfg(name: str) -> dict[str, Any]:
    return {
        "name": name,
        "type": "vectorstore",
        "description": name,
        "source": "data/",
    }


def _mcp_cfg(name: str) -> dict[str, Any]:
    return {
        "name": name,
        "type": "mcp",
        "description": name,
        "transport": "stdio",
        "command": "npx",
        "args": ["server"],
    }


def _agent(
    *,
    tools: list[dict[str, Any]] | None = None,
    agents: dict[str, dict[str, Any]] | None = None,
) -> Agent:
    return Agent(
        name="parent",
        model=LLMProvider(provider=ProviderEnum.OPENAI, name="gpt-4o"),
        instructions=Instructions(inline="parent"),
        tools=tools,
        openai=OpenAIConfig(agents=agents) if agents is not None else None,
    )


def _surface(agent: Agent, *mcp_names: str) -> ParentToolSurface:
    """Build the parent surface the backend would produce for *agent*."""
    sdk_tools = []
    for cfg in agent.tools or []:
        sdk_name = sdk_tool_name_for(cfg)
        if sdk_name is not None:
            sdk_tools.append(_sdk_tool(sdk_name))
    servers = [MagicMock(name=n) for n in mcp_names]
    for server, name in zip(servers, mcp_names, strict=True):
        server.name = name
    return index_parent_tools(agent.tools, sdk_tools, servers)


def _settings(**kw: Any) -> Any:
    from agents.model_settings import ModelSettings

    return ModelSettings(**kw)


def _build(agent: Agent, surface: ParentToolSurface | None = None, **kw: Any) -> list:
    kw.setdefault("parent_model", "gpt-4o")
    kw.setdefault("parent_model_settings", _settings(temperature=0.2))
    kw.setdefault("base_dir", None)
    kw.setdefault("resolve_model", lambda name: f"resolved:{name}")
    kw.setdefault("resolve_model_settings", lambda name: _settings(max_tokens=7))
    return build_handoff_agents(
        agent, surface=surface if surface is not None else _surface(agent), **kw
    )


@pytest.mark.unit
class TestSdkToolNames:
    def test_function_keeps_name_rag_gets_search_suffix(self) -> None:
        agent = _agent(tools=[_function_cfg("lookup"), _vectorstore_cfg("kb")])
        assert agent.tools is not None
        assert sdk_tool_name_for(agent.tools[0]) == "lookup"
        assert sdk_tool_name_for(agent.tools[1]) == "kb_search"

    def test_mcp_and_skill_have_no_function_tool(self) -> None:
        agent = _agent(
            tools=[
                _mcp_cfg("srv"),
                {"name": "s", "type": "skill", "description": "d", "instructions": "x"},
            ]
        )
        assert agent.tools is not None
        assert sdk_tool_name_for(agent.tools[0]) is None
        assert sdk_tool_name_for(agent.tools[1]) is None


@pytest.mark.unit
class TestIndexParentTools:
    def test_indexes_by_config_name(self) -> None:
        agent = _agent(tools=[_function_cfg("lookup"), _vectorstore_cfg("kb")])
        surface = _surface(agent)
        assert set(surface.tools_by_name) == {"lookup", "kb"}
        assert surface.tools_by_name["kb"].name == "kb_search"
        assert surface.names == {"lookup", "kb"}

    def test_disallowed_tool_absent_from_index(self) -> None:
        agent = _agent(tools=[_function_cfg("lookup"), _function_cfg("secret")])
        # Only 'lookup' was built (secret was disallowed upstream).
        surface = index_parent_tools(agent.tools, [_sdk_tool("lookup")], [])
        assert surface.names == {"lookup"}

    def test_mcp_servers_indexed(self) -> None:
        agent = _agent(tools=[_mcp_cfg("srv")])
        surface = _surface(agent, "srv")
        assert set(surface.mcp_by_name) == {"srv"}


@pytest.mark.unit
class TestSubagentInstructions:
    def test_prefix_prepended_once(self) -> None:
        from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX

        spec = OpenAISubagentSpec(description="d", prompt="Do research.")
        text = subagent_instructions(spec)
        assert text.startswith(RECOMMENDED_PROMPT_PREFIX)
        assert text.endswith("Do research.")
        assert text.count(RECOMMENDED_PROMPT_PREFIX) == 1
        # Idempotent: a prompt that already carries the prefix is not doubled.
        again = OpenAISubagentSpec(description="d", prompt=text)
        assert subagent_instructions(again).count(RECOMMENDED_PROMPT_PREFIX) == 1

    def test_skip_flag_leaves_prompt_verbatim(self) -> None:
        spec = OpenAISubagentSpec(
            description="d", prompt="Do research.", skip_recommended_prefix=True
        )
        assert subagent_instructions(spec) == "Do research."


@pytest.mark.unit
class TestBuildSubagents:
    def test_three_subagents_become_three_handoffs_in_order(self) -> None:
        from agents import Agent as SDKAgent
        from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX

        agent = _agent(
            agents={
                "researcher": {"description": "Finds", "prompt": "R"},
                "analyst": {"description": "Analyses", "prompt": "A"},
                "writer": {"description": "Writes", "prompt": "W"},
            }
        )
        handoffs = _build(agent)
        assert [h.name for h in handoffs] == ["researcher", "analyst", "writer"]
        assert all(isinstance(h, SDKAgent) for h in handoffs)
        assert [h.handoff_description for h in handoffs] == [
            "Finds",
            "Analyses",
            "Writes",
        ]
        assert all(
            str(h.instructions).startswith(RECOMMENDED_PROMPT_PREFIX) for h in handoffs
        )
        assert str(handoffs[0].instructions).endswith("R")

    def test_no_tools_inherits_parent_tools_and_mcp(self) -> None:
        agent = _agent(
            tools=[_function_cfg("lookup"), _vectorstore_cfg("kb"), _mcp_cfg("srv")],
            agents={"r": {"description": "d", "prompt": "p"}},
        )
        surface = _surface(agent, "srv")
        (handoff,) = _build(agent, surface)
        assert [t.name for t in handoff.tools] == ["lookup", "kb_search"]
        assert [s.name for s in handoff.mcp_servers] == ["srv"]
        # Same SDK objects, not copies.
        assert handoff.tools[0] is surface.tools_by_name["lookup"]

    def test_explicit_tools_restrict_by_config_name(self) -> None:
        agent = _agent(
            tools=[_function_cfg("lookup"), _vectorstore_cfg("kb"), _mcp_cfg("srv")],
            agents={"r": {"description": "d", "prompt": "p", "tools": ["kb"]}},
        )
        (handoff,) = _build(agent, _surface(agent, "srv"))
        assert [t.name for t in handoff.tools] == ["kb_search"]
        assert handoff.mcp_servers == []

    def test_explicit_tools_can_grant_mcp_server(self) -> None:
        agent = _agent(
            tools=[_function_cfg("lookup"), _mcp_cfg("srv")],
            agents={"r": {"description": "d", "prompt": "p", "tools": ["srv"]}},
        )
        (handoff,) = _build(agent, _surface(agent, "srv"))
        assert handoff.tools == []
        assert [s.name for s in handoff.mcp_servers] == ["srv"]

    def test_empty_tools_list_grants_nothing(self) -> None:
        agent = _agent(
            tools=[_function_cfg("lookup")],
            agents={"r": {"description": "d", "prompt": "p", "tools": []}},
        )
        (handoff,) = _build(agent)
        assert handoff.tools == []

    def test_unknown_tool_name_fails_load(self) -> None:
        agent = _agent(
            tools=[_function_cfg("lookup")],
            agents={"r": {"description": "d", "prompt": "p", "tools": ["ghost"]}},
        )
        with pytest.raises(ConfigError) as exc:
            _build(agent)
        assert "openai.agents.r.tools" in str(exc.value)
        assert "ghost" in str(exc.value)
        assert "Available: lookup" in str(exc.value)

    def test_disallowed_parent_tool_cannot_be_granted(self) -> None:
        agent = _agent(
            tools=[_function_cfg("lookup"), _function_cfg("secret")],
            agents={"r": {"description": "d", "prompt": "p", "tools": ["secret"]}},
        )
        surface = index_parent_tools(agent.tools, [_sdk_tool("lookup")], [])
        with pytest.raises(ConfigError, match="secret"):
            _build(agent, surface)

    def test_model_inherit_and_omitted_use_parent_model(self) -> None:
        parent_model = MagicMock(spec=_sdk_model_base(), name="parent-model")
        agent = _agent(
            agents={
                "a": {"description": "d", "prompt": "p", "model": "inherit"},
                "b": {"description": "d", "prompt": "p"},
            }
        )
        a, b = _build(agent, parent_model=parent_model)
        assert a.model is parent_model
        assert b.model is parent_model

    def test_explicit_model_goes_through_resolver(self) -> None:
        calls: list[str] = []

        def resolver(name: str) -> str:
            calls.append(name)
            return f"resolved:{name}"

        agent = _agent(
            agents={"a": {"description": "d", "prompt": "p", "model": "gpt-4o-mini"}}
        )
        (handoff,) = _build(agent, resolve_model=resolver)
        assert calls == ["gpt-4o-mini"]
        assert handoff.model == "resolved:gpt-4o-mini"
        # Explicit models get settings rebuilt for that model, not the parent's.
        assert handoff.model_settings.max_tokens == 7
        assert handoff.model_settings.temperature is None

    def test_inherit_carries_parent_model_settings_and_output_type(self) -> None:
        from agents.agent_output import AgentOutputSchema

        agent = _agent(
            tools=[
                {"name": "s", "type": "skill", "description": "d", "instructions": "X"}
            ],
            agents={"r": {"description": "d", "prompt": "p"}},
        )
        output_type = AgentOutputSchema(dict, strict_json_schema=False)
        subagent, skill = _build(agent, output_type=output_type)
        assert subagent.model_settings.temperature == 0.2
        assert skill.model_settings.temperature == 0.2
        assert subagent.output_type is output_type
        assert skill.output_type is output_type

    def test_normalised_handoff_names_must_be_unique(self) -> None:
        agent = _agent(
            agents={
                "research_assistant": {"description": "d", "prompt": "p"},
                "research assistant": {"description": "d", "prompt": "p"},
            }
        )
        with pytest.raises(ConfigError, match="transfer_to_research_assistant"):
            _build(agent)

    def test_no_agents_block_yields_no_handoffs(self) -> None:
        assert _build(_agent()) == []


@pytest.mark.unit
class TestBuildSkills:
    def test_inline_and_file_skills_are_equivalent(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "skills" / "research-assistant"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(SKILL_MD, encoding="utf-8")
        token = agent_base_dir.set(str(tmp_path))
        try:
            agent = _agent(
                tools=[
                    _function_cfg("lookup"),
                    {
                        "name": "inline-skill",
                        "type": "skill",
                        "description": "File description.",
                        "instructions": "Search then summarise.",
                        "allowed_tools": ["lookup"],
                    },
                    {
                        "name": "file-skill",
                        "type": "skill",
                        "path": "skills/research-assistant",
                        "allowed_tools": ["lookup"],
                    },
                ]
            )
        finally:
            agent_base_dir.reset(token)
        inline, file_based = _build(agent, base_dir=tmp_path, parent_model="pm")
        assert (
            inline.instructions == file_based.instructions == "Search then summarise."
        )
        assert (
            inline.handoff_description
            == file_based.handoff_description
            == "File description."
        )
        assert [t.name for t in inline.tools] == [t.name for t in file_based.tools]
        assert inline.model == file_based.model == "pm"
        assert (inline.name, file_based.name) == ("inline-skill", "file-skill")

    def test_skill_instructions_carry_no_prefix(self) -> None:
        from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX

        agent = _agent(
            tools=[
                {"name": "s", "type": "skill", "description": "d", "instructions": "X"}
            ]
        )
        (skill,) = _build(agent)
        assert skill.instructions == "X"
        assert RECOMMENDED_PROMPT_PREFIX not in str(skill.instructions)

    def test_allowed_tools_scopes_skill(self) -> None:
        agent = _agent(
            tools=[
                _function_cfg("lookup"),
                _function_cfg("other"),
                {
                    "name": "s",
                    "type": "skill",
                    "description": "d",
                    "instructions": "X",
                    "allowed_tools": ["other"],
                },
            ]
        )
        (skill,) = _build(agent)
        assert [t.name for t in skill.tools] == ["other"]

    def test_no_allowed_tools_means_no_tools(self) -> None:
        agent = _agent(
            tools=[
                _function_cfg("lookup"),
                {"name": "s", "type": "skill", "description": "d", "instructions": "X"},
            ]
        )
        (skill,) = _build(agent)
        assert skill.tools == []
        assert skill.mcp_servers == []

    def test_disallowed_skill_is_dropped(self) -> None:
        agent = _agent(
            tools=[
                {"name": "s", "type": "skill", "description": "d", "instructions": "X"}
            ]
        )
        assert _build(agent, disallowed={"s"}) == []

    def test_skill_name_colliding_with_subagent_fails(self) -> None:
        agent = _agent(
            tools=[
                {"name": "r", "type": "skill", "description": "d", "instructions": "X"}
            ],
            agents={"r": {"description": "d", "prompt": "p"}},
        )
        with pytest.raises(ConfigError, match="openai.agents.r"):
            _build(agent)

    def test_skill_name_colliding_after_normalisation_fails(self) -> None:
        agent = _agent(
            tools=[
                {
                    "name": "research-assistant",
                    "type": "skill",
                    "description": "d",
                    "instructions": "X",
                }
            ],
            agents={"research_assistant": {"description": "d", "prompt": "p"}},
        )
        with pytest.raises(ConfigError, match="transfer_to_research_assistant"):
            _build(agent)

    def test_skill_path_invalid_at_build_time_fails_clearly(
        self, tmp_path: Path
    ) -> None:
        skill_dir = tmp_path / "skills" / "s"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(SKILL_MD, encoding="utf-8")
        agent = _agent(
            tools=[{"name": "s", "type": "skill", "path": str(skill_dir)}],
        )
        # Simulate the directory disappearing between config load and build.
        (skill_dir / "SKILL.md").unlink()
        with pytest.raises(ConfigError, match="tools.s.path"):
            _build(agent, base_dir=tmp_path)

    def test_subagents_precede_skills(self) -> None:
        agent = _agent(
            tools=[
                {"name": "s", "type": "skill", "description": "d", "instructions": "X"}
            ],
            agents={"r": {"description": "d", "prompt": "p"}},
        )
        assert [h.name for h in _build(agent)] == ["r", "s"]


@pytest.mark.unit
class TestHandoffShadowsParentTool:
    def test_handoff_name_colliding_with_parent_tool_fails_load(self) -> None:
        from holodeck.lib.errors import ConfigError

        agent = _agent(
            tools=[_function_cfg("transfer_to_researcher")],
            agents={"researcher": {"description": "d", "prompt": "p"}},
        )
        with pytest.raises(ConfigError, match="already declares as a tool"):
            _build(agent)
