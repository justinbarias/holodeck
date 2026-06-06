"""Unit tests for mutator axis application (T2)."""

import pytest

from holodeck.lib.errors import OptimizerError
from holodeck.models.agent import Agent, Instructions
from holodeck.models.llm import LLMProvider, ProviderEnum
from holodeck.models.tool import VectorstoreTool
from holodeck.optimizer.mutator import (
    apply_axes,
    apply_textual_edit,
    get_path,
    overlay_axes,
)


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


class TestGetPath:
    """get_path reads values at dotted/selector paths."""

    def test_reads_instruction_text(self) -> None:
        assert get_path(_agent(), "instructions.inline") == "You are helpful."

    def test_reads_tool_field_by_selector(self) -> None:
        assert get_path(_agent(), "tools[name=kb].top_k") == 5

    def test_unknown_path_raises(self) -> None:
        with pytest.raises(OptimizerError):
            get_path(_agent(), "instructions.bogus")


class TestOverlayAxes:
    """overlay_axes copies only axis values, preserving other (templated) fields."""

    def test_overlays_axes_and_preserves_other_fields(self) -> None:
        # `template` mimics the unsubstituted source: a ${VAR} placeholder survives.
        template = Agent(
            name="opt-agent",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI, name="${MODEL_NAME}", temperature=0.1
            ),
            instructions=Instructions(inline="original prompt"),
        )
        # `source` mimics the env-resolved, optimized agent.
        source = Agent(
            name="opt-agent",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI, name="gpt-4o-mini", temperature=0.9
            ),
            instructions=Instructions(inline="TUNED PROMPT"),
        )

        result = overlay_axes(
            template, source, ["instructions.inline", "model.temperature"]
        )

        # Axis values come from the optimized source...
        assert result.instructions.inline == "TUNED PROMPT"
        assert result.model.temperature == 0.9
        # ...but the non-axis templated field is preserved (no secret leak).
        assert result.model.name == "${MODEL_NAME}"
        # The template itself is never mutated.
        assert template.instructions.inline == "original prompt"
