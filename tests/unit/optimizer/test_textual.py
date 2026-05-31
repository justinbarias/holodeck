"""Unit tests for the Critic/Applier textual proposer (T6)."""

import pytest

from holodeck.lib.backends.base import ExecutionResult
from holodeck.models.agent import Agent, Instructions
from holodeck.models.llm import LLMProvider, ProviderEnum
from holodeck.optimizer.config import TextualAxis
from holodeck.optimizer.mutator import apply_textual_edit
from holodeck.optimizer.proposers.textual import (
    TextualProposer,
    load_critic_applier,
)


def _agent(instruction: str = "You are helpful.") -> Agent:
    return Agent(
        name="opt-agent",
        model=LLMProvider(provider=ProviderEnum.OPENAI, name="gpt-4o-mini"),
        instructions=Instructions(inline=instruction),
    )


def _subagent(name: str) -> Agent:
    return Agent(
        name=name,
        model=LLMProvider(provider=ProviderEnum.OPENAI, name="gpt-4o-mini"),
        instructions=Instructions(inline="subagent prompt"),
    )


def _make_invoker(critic_response: str, applier_response: str):
    """Return a stub invoker keyed on the subagent's role in its name."""

    async def invoker(agent: Agent, prompt: str) -> ExecutionResult:
        if "critic" in agent.name:
            return ExecutionResult(response=critic_response)
        return ExecutionResult(response=applier_response)

    return invoker


def _proposer(invoker) -> TextualProposer:
    return TextualProposer(
        axes=[TextualAxis(path="instructions.inline", max_chars=4000)],
        critic_agent=_subagent("optimizer-critic"),
        applier_agent=_subagent("optimizer-applier"),
        invoker=invoker,
    )


class TestTextualProposal:
    """The proposer produces an edited prompt from canned subagent JSON."""

    @pytest.mark.asyncio
    async def test_produces_edited_prompt(self) -> None:
        invoker = _make_invoker(
            critic_response='{"gradient": "be more concise", "issues": ["verbose"]}',
            applier_response='{"new_text": "You are a concise expert.", '
            '"summary": "tightened"}',
        )
        proposer = _proposer(invoker)
        proposer.begin(_agent(), None)

        proposal = await proposer.ask()

        assert proposal is not None
        assert proposal.error is None
        assert proposal.textual_axis == "instructions.inline"
        assert proposal.new_text == "You are a concise expert."
        assert proposal.edit_summary == "tightened"

    @pytest.mark.asyncio
    async def test_applied_edit_changes_only_instructions(self) -> None:
        invoker = _make_invoker(
            critic_response='{"gradient": "tighten"}',
            applier_response='{"new_text": "Concise expert.", "summary": "s"}',
        )
        proposer = _proposer(invoker)
        agent = _agent()
        proposer.begin(agent, None)

        proposal = await proposer.ask()
        assert proposal is not None and proposal.new_text is not None
        edited = apply_textual_edit(agent, proposal.textual_axis, proposal.new_text)

        assert edited.instructions.inline == "Concise expert."
        assert edited.name == agent.name
        assert edited.model.name == agent.model.name
        assert agent.instructions.inline == "You are helpful."

    @pytest.mark.asyncio
    async def test_reads_structured_output_when_present(self) -> None:
        async def invoker(agent: Agent, prompt: str) -> ExecutionResult:
            if "critic" in agent.name:
                return ExecutionResult(
                    response="", structured_output={"gradient": "tighten"}
                )
            return ExecutionResult(
                response="",
                structured_output={"new_text": "Be brief.", "summary": "ok"},
            )

        proposer = _proposer(invoker)
        proposer.begin(_agent(), None)

        proposal = await proposer.ask()

        assert proposal is not None
        assert proposal.error is None
        assert proposal.new_text == "Be brief."


class TestParseFailure:
    """Unparseable subagent output skips the trial without crashing."""

    @pytest.mark.asyncio
    async def test_critic_garbage_skips_trial(self) -> None:
        invoker = _make_invoker(
            critic_response="not json at all",
            applier_response='{"new_text": "x", "summary": "y"}',
        )
        proposer = _proposer(invoker)
        proposer.begin(_agent(), None)

        proposal = await proposer.ask()

        assert proposal is not None
        assert proposal.error is not None
        assert proposal.new_text is None
        assert proposal.textual_axis == "instructions.inline"

    @pytest.mark.asyncio
    async def test_applier_missing_new_text_skips_trial(self) -> None:
        invoker = _make_invoker(
            critic_response='{"gradient": "tighten"}',
            applier_response='{"summary": "no new_text here"}',
        )
        proposer = _proposer(invoker)
        proposer.begin(_agent(), None)

        proposal = await proposer.ask()

        assert proposal is not None
        assert proposal.error is not None
        assert proposal.new_text is None


class TestExhaustion:
    """The proposer reports exhaustion when axes are consumed."""

    @pytest.mark.asyncio
    async def test_no_axes_returns_none(self) -> None:
        proposer = TextualProposer(
            axes=[],
            critic_agent=_subagent("optimizer-critic"),
            applier_agent=_subagent("optimizer-applier"),
            invoker=_make_invoker("{}", "{}"),
        )
        proposer.begin(_agent(), None)
        assert await proposer.ask() is None


class TestLoadCriticApplier:
    """The Critic/Applier templates load and adopt the injected model."""

    def test_templates_load_with_model_override(self) -> None:
        model = LLMProvider(provider=ProviderEnum.ANTHROPIC, name="claude-haiku-4-5")
        critic, applier = load_critic_applier(model)

        assert critic.name == "optimizer-critic"
        assert applier.name == "optimizer-applier"
        assert critic.model.provider == ProviderEnum.ANTHROPIC
        assert applier.model.name == "claude-haiku-4-5"
