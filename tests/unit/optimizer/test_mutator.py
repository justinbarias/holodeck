"""Unit tests for mutator axis application (T2)."""

import pytest

from holodeck.lib.errors import OptimizerError
from holodeck.models.agent import Agent, Instructions
from holodeck.models.llm import LLMProvider, ProviderEnum
from holodeck.models.tool import VectorstoreTool
from holodeck.optimizer.mutator import apply_axes, apply_textual_edit


def _agent() -> Agent:
    return Agent(
        name="opt-agent",
        model=LLMProvider(
            provider=ProviderEnum.OPENAI, name="gpt-4o-mini", temperature=0.3
        ),
        instructions=Instructions(inline="You are helpful."),
        tools=[
            VectorstoreTool(name="kb", description="kb tool", source="./data", top_k=5)
        ],
    )


class TestApplyAxes:
    """apply_axes mutates declared numeric axes on a fresh Agent instance."""

    def test_mutates_model_temperature(self) -> None:
        agent = _agent()
        new_agent = apply_axes(agent, {"model.temperature": 0.9})
        assert new_agent.model.temperature == 0.9
        # Original untouched.
        assert agent.model.temperature == 0.3
        assert new_agent is not agent

    def test_mutates_tool_field_by_name_selector(self) -> None:
        agent = _agent()
        new_agent = apply_axes(agent, {"tools[name=kb].top_k": 9})
        assert new_agent.tools is not None
        assert new_agent.tools[0].top_k == 9
        # Original untouched.
        assert agent.tools is not None
        assert agent.tools[0].top_k == 5

    def test_multiple_axes_applied_together(self) -> None:
        agent = _agent()
        new_agent = apply_axes(
            agent, {"model.temperature": 0.7, "tools[name=kb].top_k": 3}
        )
        assert new_agent.model.temperature == 0.7
        assert new_agent.tools is not None
        assert new_agent.tools[0].top_k == 3

    def test_unknown_attribute_path_raises(self) -> None:
        with pytest.raises(OptimizerError):
            apply_axes(_agent(), {"model.bogus": 1.0})

    def test_unknown_root_path_raises(self) -> None:
        with pytest.raises(OptimizerError):
            apply_axes(_agent(), {"nonexistent.field": 1.0})

    def test_missing_selector_target_raises(self) -> None:
        with pytest.raises(OptimizerError):
            apply_axes(_agent(), {"tools[name=missing].top_k": 3})


class TestApplyTextualEdit:
    """apply_textual_edit rewrites only the targeted instruction text."""

    def test_rewrites_inline_instructions(self) -> None:
        agent = _agent()
        new_agent = apply_textual_edit(
            agent, "instructions.inline", "You are a concise expert."
        )
        assert new_agent.instructions.inline == "You are a concise expert."
        # Original untouched; other fields unchanged.
        assert agent.instructions.inline == "You are helpful."
        assert new_agent.model.temperature == agent.model.temperature
        assert new_agent.name == agent.name

    def test_unknown_textual_path_raises(self) -> None:
        with pytest.raises(OptimizerError):
            apply_textual_edit(_agent(), "instructions.bogus", "x")
