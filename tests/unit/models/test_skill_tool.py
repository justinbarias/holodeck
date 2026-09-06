"""Tests for ``SkillTool`` (spec 023 FR-022 to FR-024, spec 035 FR-070)."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from holodeck.config.context import agent_base_dir
from holodeck.models.agent import Agent
from holodeck.models.llm import LLMProvider, ProviderEnum
from holodeck.models.tool import FunctionTool, SkillTool

SKILL_MD = """---
name: research-assistant
description: Frontmatter description.
---
Body instructions.
"""


def _skill_dir(root: Path, text: str = SKILL_MD) -> Path:
    skill_dir = root / "skills" / "research-assistant"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(text, encoding="utf-8")
    return skill_dir


def _agent(tools: list[dict]) -> Agent:
    return Agent(
        name="a",
        model=LLMProvider(provider=ProviderEnum.OPENAI, name="gpt-4o"),
        instructions={"inline": "hi"},
        tools=tools,
    )


@pytest.mark.unit
class TestInlineSkill:
    def test_valid_inline(self) -> None:
        tool = SkillTool(name="summarise", description="d", instructions="Do it.")
        assert tool.type == "skill"
        assert tool.allowed_tools is None

    def test_inline_requires_description(self) -> None:
        with pytest.raises(ValidationError, match="requires a non-empty description"):
            SkillTool(name="summarise", instructions="Do it.")

    def test_blank_instructions_rejected(self) -> None:
        with pytest.raises(ValidationError, match="instructions must be non-empty"):
            SkillTool(name="summarise", description="d", instructions="  ")

    def test_instructions_and_path_exclusive(self, tmp_path: Path) -> None:
        with pytest.raises(ValidationError, match="mutually exclusive"):
            SkillTool(name="s", description="d", instructions="x", path=str(tmp_path))

    def test_neither_form_rejected(self) -> None:
        with pytest.raises(ValidationError, match="either 'instructions' or 'path'"):
            SkillTool(name="s", description="d")

    @pytest.mark.parametrize("name", ["Upper", "has_underscore", "-lead", "a--b", "x-"])
    def test_name_pattern_follows_agent_skills_spec(self, name: str) -> None:
        with pytest.raises(ValidationError):
            SkillTool(name=name, description="d", instructions="x")

    def test_unknown_key_rejected(self) -> None:
        with pytest.raises(ValidationError, match="extra_forbidden|Extra inputs"):
            SkillTool(name="s", description="d", instructions="x", bogus=1)


@pytest.mark.unit
class TestFileSkill:
    def test_description_falls_back_to_frontmatter(self, tmp_path: Path) -> None:
        skill_dir = _skill_dir(tmp_path)
        tool = SkillTool(name="research-assistant", path=str(skill_dir))
        assert tool.description == "Frontmatter description."

    def test_yaml_description_wins(self, tmp_path: Path) -> None:
        skill_dir = _skill_dir(tmp_path)
        tool = SkillTool(name="ra", description="YAML wins", path=str(skill_dir))
        assert tool.description == "YAML wins"

    def test_relative_path_resolves_against_agent_base_dir(
        self, tmp_path: Path
    ) -> None:
        _skill_dir(tmp_path)
        token = agent_base_dir.set(str(tmp_path))
        try:
            tool = SkillTool(name="ra", path="skills/research-assistant")
        finally:
            agent_base_dir.reset(token)
        assert tool.description == "Frontmatter description."

    def test_missing_directory_fails_load(self, tmp_path: Path) -> None:
        with pytest.raises(ValidationError, match="not a directory"):
            SkillTool(name="ra", path=str(tmp_path / "missing"))

    def test_missing_frontmatter_fields_fail_load(self, tmp_path: Path) -> None:
        skill_dir = _skill_dir(tmp_path, "---\nname: only\n---\nbody")
        with pytest.raises(ValidationError, match="missing required field"):
            SkillTool(name="ra", path=str(skill_dir))


@pytest.mark.unit
class TestAgentAllowedTools:
    def test_union_discriminates_skill(self) -> None:
        agent = _agent(
            [
                {
                    "name": "summarise",
                    "type": "skill",
                    "description": "d",
                    "instructions": "x",
                }
            ]
        )
        assert agent.tools is not None
        assert isinstance(agent.tools[0], SkillTool)

    def test_allowed_tools_must_name_parent_tools(self) -> None:
        with pytest.raises(ValidationError, match="unknown: skill 'summarise': ghost"):
            _agent(
                [
                    {
                        "name": "summarise",
                        "type": "skill",
                        "description": "d",
                        "instructions": "x",
                        "allowed_tools": ["ghost"],
                    }
                ]
            )

    def test_allowed_tools_cannot_reference_another_skill(self) -> None:
        with pytest.raises(ValidationError, match="unknown: skill 'b': a"):
            _agent(
                [
                    {
                        "name": "a",
                        "type": "skill",
                        "description": "d",
                        "instructions": "x",
                    },
                    {
                        "name": "b",
                        "type": "skill",
                        "description": "d",
                        "instructions": "x",
                        "allowed_tools": ["a"],
                    },
                ]
            )

    def test_allowed_tools_accepts_declared_function_tool(self) -> None:
        agent = _agent(
            [
                {
                    "name": "lookup",
                    "type": "function",
                    "description": "d",
                    "file": "tools.py",
                    "function": "lookup",
                },
                {
                    "name": "summarise",
                    "type": "skill",
                    "description": "d",
                    "instructions": "x",
                    "allowed_tools": ["lookup"],
                },
            ]
        )
        assert agent.tools is not None
        assert isinstance(agent.tools[0], FunctionTool)
        assert isinstance(agent.tools[1], SkillTool)


@pytest.mark.unit
class TestLoaderBaseDir:
    """``load_agent_yaml`` sets ``agent_base_dir`` before validation."""

    def test_relative_skill_path_resolves_against_yaml_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from holodeck.config.loader import ConfigLoader

        project = tmp_path / "project"
        project.mkdir()
        _skill_dir(project)
        (project / "agent.yaml").write_text(
            "name: a\n"
            "model:\n  provider: openai\n  name: gpt-4o\n"
            "instructions:\n  inline: hi\n"
            "tools:\n"
            "  - name: ra\n    type: skill\n    path: skills/research-assistant\n",
            encoding="utf-8",
        )
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        monkeypatch.chdir(elsewhere)
        token = agent_base_dir.set(str(elsewhere))  # stale value from a prior load
        try:
            agent = ConfigLoader().load_agent_yaml(str(project / "agent.yaml"))
        finally:
            agent_base_dir.reset(token)
        assert agent.tools is not None
        assert isinstance(agent.tools[0], SkillTool)
        assert agent.tools[0].description == "Frontmatter description."
