"""Unit tests for the Critic/Applier textual proposer (T6)."""

import logging

import pytest

from holodeck.lib.backends.base import ExecutionResult
from holodeck.models.agent import Agent, Instructions
from holodeck.models.llm import LLMProvider, ProviderEnum
from holodeck.models.test_result import ReportSummary, TestReport, TestResult
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


def _report_failing(test_input: str) -> TestReport:
    """A report with a single failing case carrying ``test_input``."""
    return TestReport(
        agent_name="opt-agent",
        agent_config_path="agent.yaml",
        results=[
            TestResult(
                test_name="t0",
                test_input=test_input,
                agent_response="a wrong answer",
                passed=False,
                execution_time_ms=1,
                timestamp="2026-05-31T00:00:00Z",
            )
        ],
        summary=ReportSummary(
            total_tests=1,
            passed=0,
            failed=1,
            pass_rate=0.0,
            total_duration_ms=1,
            metrics_evaluated={"groundedness": 1},
            average_scores={"groundedness": 0.2},
        ),
        timestamp="2026-05-31T00:00:00Z",
        holodeck_version="test",
    )


class _RecordingInvoker:
    """Scripted invoker that records every prompt it is asked, by role."""

    def __init__(self, critic_response: str, applier_response: str) -> None:
        self._critic_response = critic_response
        self._applier_response = applier_response
        self.critic_prompts: list[str] = []
        self.applier_prompts: list[str] = []

    async def __call__(self, agent: Agent, prompt: str) -> ExecutionResult:
        if "critic" in agent.name:
            self.critic_prompts.append(prompt)
            return ExecutionResult(response=self._critic_response)
        self.applier_prompts.append(prompt)
        return ExecutionResult(response=self._applier_response)


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


class TestIterativeRefinement:
    """Single-axis textual refinement chains across multiple steps."""

    @pytest.mark.asyncio
    async def test_ask_keeps_proposing_across_iterations(self) -> None:
        # The phase no longer ends after one rewrite — ask() keeps producing
        # refinements; the loop's budget (not the proposer) bounds the phase.
        invoker = _RecordingInvoker(
            critic_response='{"gradient": "tighten"}',
            applier_response='{"new_text": "v2", "summary": "s"}',
        )
        proposer = _proposer(invoker)
        proposer.begin(_agent(), None)

        first = await proposer.ask()
        proposer.tell(first, 0.5, accepted=False, report=None)
        second = await proposer.ask()

        assert first is not None and first.new_text is not None
        assert second is not None and second.new_text is not None

    @pytest.mark.asyncio
    async def test_step_two_chains_from_last_attempt_not_best(self) -> None:
        # Momentum: step 2 must rewrite attempt 1's text, not the original best.
        invoker = _RecordingInvoker(
            critic_response='{"gradient": "g"}',
            applier_response='{"new_text": "ATTEMPT_ONE_TEXT", "summary": "s"}',
        )
        proposer = _proposer(invoker)
        proposer.begin(_agent(instruction="ORIGINAL_BEST_TEXT"), _report_failing("Q0"))

        first = await proposer.ask()
        proposer.tell(first, 0.5, accepted=False, report=_report_failing("Q1"))
        await proposer.ask()

        # The critic's 2nd prompt is built from the last attempt's text…
        second_critic_prompt = invoker.critic_prompts[1]
        assert "ATTEMPT_ONE_TEXT" in second_critic_prompt
        assert "ORIGINAL_BEST_TEXT" not in second_critic_prompt

    @pytest.mark.asyncio
    async def test_step_two_uses_last_attempts_report_for_context(self) -> None:
        # The gradient for step 2 is re-derived from attempt 1's own report.
        invoker = _RecordingInvoker(
            critic_response='{"gradient": "g"}',
            applier_response='{"new_text": "v2", "summary": "s"}',
        )
        proposer = _proposer(invoker)
        proposer.begin(_agent(), _report_failing("BASELINE_FAIL"))

        first = await proposer.ask()
        proposer.tell(
            first, 0.5, accepted=False, report=_report_failing("ATTEMPT_FAIL")
        )
        await proposer.ask()

        second_critic_prompt = invoker.critic_prompts[1]
        assert "ATTEMPT_FAIL" in second_critic_prompt
        assert "BASELINE_FAIL" not in second_critic_prompt

    @pytest.mark.asyncio
    async def test_errored_attempt_does_not_advance_momentum(self) -> None:
        # A skipped (errored) trial was never scored, so the chain must not adopt
        # its (absent) text — the next step still chains from the prior good text.
        invoker = _RecordingInvoker(
            critic_response='{"gradient": "g"}',
            applier_response='{"new_text": "GOOD_ONE", "summary": "s"}',
        )
        proposer = _proposer(invoker)
        proposer.begin(_agent(instruction="ORIGINAL_BEST_TEXT"), None)

        first = await proposer.ask()
        proposer.tell(first, 0.5, accepted=False, report=None)
        # Simulate the loop telling us about an errored proposal (new_text None).
        proposer.tell(
            type(first)(textual_axis="instructions.inline", error="bad JSON"),
            0.5,
            accepted=False,
            report=None,
        )
        await proposer.ask()

        third_critic_prompt = invoker.critic_prompts[-1]
        assert "GOOD_ONE" in third_critic_prompt

    @pytest.mark.asyncio
    async def test_no_secret_bearing_field_reaches_the_prompt(self) -> None:
        # Only the axis path's text + failing-case context feed the gradient.
        # A sentinel placed in a *non-axis* field must never appear in a prompt.
        invoker = _RecordingInvoker(
            critic_response='{"gradient": "g"}',
            applier_response='{"new_text": "v2", "summary": "s"}',
        )
        agent = _agent(instruction="SAFE_INSTRUCTION")
        agent.description = "SENTINEL_SECRET_XYZ"
        proposer = _proposer(invoker)
        proposer.begin(agent, _report_failing("Q0"))

        await proposer.ask()

        all_prompts = invoker.critic_prompts + invoker.applier_prompts
        assert all_prompts
        assert all("SENTINEL_SECRET_XYZ" not in p for p in all_prompts)


class TestMultiAxisFallback:
    """With >1 textual axis, fall back to one rewrite per axis (no iteration)."""

    def _two_axis_proposer(self, invoker) -> TextualProposer:
        return TextualProposer(
            axes=[
                TextualAxis(path="instructions.inline", max_chars=4000),
                TextualAxis(path="description", max_chars=4000),
            ],
            critic_agent=_subagent("optimizer-critic"),
            applier_agent=_subagent("optimizer-applier"),
            invoker=invoker,
        )

    @pytest.mark.asyncio
    async def test_one_pass_per_axis_then_exhausted(self) -> None:
        invoker = _RecordingInvoker(
            critic_response='{"gradient": "g"}',
            applier_response='{"new_text": "v2", "summary": "s"}',
        )
        proposer = self._two_axis_proposer(invoker)
        agent = _agent()
        agent.description = "A described agent."
        proposer.begin(agent, None)

        first = await proposer.ask()
        proposer.tell(first, 0.5, accepted=False, report=None)
        second = await proposer.ask()
        proposer.tell(second, 0.5, accepted=False, report=None)
        third = await proposer.ask()

        assert first is not None and first.textual_axis == "instructions.inline"
        assert second is not None and second.textual_axis == "description"
        assert third is None  # exhausted after one pass per axis

    @pytest.mark.asyncio
    async def test_logs_multi_axis_fallback(self, caplog) -> None:
        invoker = _RecordingInvoker(
            critic_response='{"gradient": "g"}',
            applier_response='{"new_text": "v2", "summary": "s"}',
        )
        proposer = self._two_axis_proposer(invoker)
        agent = _agent()
        agent.description = "A described agent."

        with caplog.at_level(
            logging.INFO, logger="holodeck.optimizer.proposers.textual"
        ):
            proposer.begin(agent, None)

        assert any("single axis" in r.message for r in caplog.records)


class TestLoadCriticApplier:
    """The Critic/Applier templates load and adopt the injected model."""

    def test_templates_load_with_model_override(self) -> None:
        model = LLMProvider(provider=ProviderEnum.ANTHROPIC, name="claude-haiku-4-5")
        critic, applier = load_critic_applier(model)

        assert critic.name == "optimizer-critic"
        assert applier.name == "optimizer-applier"
        assert critic.model.provider == ProviderEnum.ANTHROPIC
        assert applier.model.name == "claude-haiku-4-5"
