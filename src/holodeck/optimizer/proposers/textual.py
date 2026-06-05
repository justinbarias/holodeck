"""Critic/Applier proposer for the textual phase.

Implements a TextGrad/OPRO-style natural-language "gradient" step: a Critic
subagent reads the current instructions plus failing-case context and emits a
critique; an Applier subagent rewrites the instructions accordingly. Each
textual axis is proposed once per phase. Unparseable subagent output skips the
trial (counts toward the phase's patience) rather than crashing.
"""

import json
import logging
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, Literal

from holodeck.config.loader import ConfigLoader
from holodeck.lib.backends.base import ExecutionResult
from holodeck.lib.backends.selector import BackendSelector
from holodeck.models.agent import Agent
from holodeck.models.llm import LLMProvider
from holodeck.models.test_result import TestReport
from holodeck.optimizer.config import TextualAxis
from holodeck.optimizer.mutator import get_path
from holodeck.optimizer.proposers.base import Proposal

logger = logging.getLogger(__name__)

# Directory holding the Critic/Applier subagent templates.
_AGENTS_DIR = Path(__file__).resolve().parent.parent / "agents"

# An invoker runs a single stateless turn of a subagent and returns its result.
InvokerFn = Callable[[Agent, str], Awaitable[ExecutionResult]]

# Cap how many failing cases are fed to the Critic to keep the prompt bounded.
_MAX_FAILING_CASES = 5
_MAX_RESPONSE_CHARS = 600

# Cap how many prior attempts are shown to the Critic as gradient history.
_MAX_HISTORY = 3


def load_critic_applier(model: LLMProvider) -> tuple[Agent, Agent]:
    """Load the Critic/Applier subagent templates with ``model`` injected.

    Args:
        model: The LLM provider to run the subagents on (typically the
            optimized agent's own model, so existing credentials apply).

    Returns:
        A ``(critic_agent, applier_agent)`` tuple.
    """
    loader = ConfigLoader()
    critic = loader.load_agent_yaml(str(_AGENTS_DIR / "critic.yaml"))
    applier = loader.load_agent_yaml(str(_AGENTS_DIR / "applier.yaml"))
    return (
        critic.model_copy(update={"model": model}),
        applier.model_copy(update={"model": model}),
    )


async def _default_invoker(agent: Agent, prompt: str) -> ExecutionResult:
    """Invoke a subagent once via the auto-selected backend."""
    backend = await BackendSelector.select(agent)
    try:
        return await backend.invoke_once(prompt)
    finally:
        await backend.teardown()


def _strip_code_fence(text: str) -> str:
    """Remove a leading/trailing Markdown code fence if present."""
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        # Drop the opening fence (``` or ```json) and any closing fence.
        lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    return stripped


def _parse_json_result(result: ExecutionResult) -> dict[str, Any]:
    """Parse a subagent result into a dict, preferring structured output.

    Raises:
        ValueError: If neither structured output nor the response text yields a
            JSON object.
    """
    if isinstance(result.structured_output, dict):
        return result.structured_output
    text = _strip_code_fence(result.response or "")
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("subagent returned JSON that is not an object")
    return data


def build_failing_context(report: TestReport | None) -> str:
    """Summarize failing cases from a report for the Critic prompt.

    Args:
        report: The latest baseline report, or None.

    Returns:
        A bounded, human-readable summary of failing cases (empty if none).
    """
    if report is None:
        return ""
    failing = [r for r in report.results if not r.passed][:_MAX_FAILING_CASES]
    if not failing:
        return ""
    blocks: list[str] = []
    for case in failing:
        response = (case.agent_response or "")[:_MAX_RESPONSE_CHARS]
        metric_notes = [
            f"{m.metric_name}={m.score:.3f}" + (f" error={m.error}" if m.error else "")
            for m in case.metric_results
        ]
        blocks.append(
            f"- input: {case.test_input}\n"
            f"  response: {response}\n"
            f"  metrics: {', '.join(metric_notes) or 'n/a'}"
        )
    return "\n".join(blocks)


class TextualProposer:
    """Proposes instruction rewrites via Critic → Applier subagents."""

    phase: Literal["numeric", "textual"] = "textual"

    def __init__(
        self,
        axes: list[TextualAxis],
        critic_agent: Agent,
        applier_agent: Agent,
        invoker: InvokerFn | None = None,
    ) -> None:
        """Initialize the proposer.

        Args:
            axes: Declared textual axes to rewrite (one proposal each per phase).
            critic_agent: Subagent that emits the natural-language gradient.
            applier_agent: Subagent that rewrites the instructions.
            invoker: Async callable invoking a subagent; defaults to the
                backend-selected real invoker.
        """
        self._axes = axes
        self._critic = critic_agent
        self._applier = applier_agent
        self._invoke = invoker or _default_invoker
        self._index = 0
        self._best_agent: Agent | None = None
        self._best_report: TestReport | None = None
        # Momentum state for single-axis iterative refinement: the chain follows
        # the *last attempt* (TextGrad's in-place step), not a frozen best.
        self._last_text: str | None = None
        self._last_loss: float | None = None
        self._last_report: TestReport | None = None
        self._history: list[tuple[str | None, float, bool]] = []

    def begin(self, best_agent: Agent, best_report: TestReport | None) -> None:
        """Re-initialize for a new textual phase against the current best."""
        self._best_agent = best_agent
        self._best_report = best_report
        self._index = 0
        self._last_text = None
        self._last_loss = None
        self._last_report = None
        self._history = []
        if len(self._axes) > 1:
            logger.info(
                "Textual phase declares %d axes; iterative refinement supports a "
                "single axis. Falling back to one rewrite per axis.",
                len(self._axes),
            )

    async def ask(self) -> Proposal | None:
        """Propose the next instruction rewrite.

        Single axis: refine iteratively, chaining from the last attempt; the
        loop's ``max_trials``/``patience`` budget (not the proposer) bounds the
        phase. Multiple axes: fall back to one rewrite per axis, then exhausted.
        """
        if self._best_agent is None or not self._axes:
            return None
        if len(self._axes) > 1:
            return await self._ask_single_pass()
        return await self._ask_iterative(self._axes[0])

    async def _ask_single_pass(self) -> Proposal | None:
        """Multi-axis fallback: rewrite each axis once from the current best."""
        if self._index >= len(self._axes):
            return None
        axis = self._axes[self._index]
        self._index += 1
        return await self._propose(
            axis, source_text=None, context_report=self._best_report
        )

    async def _ask_iterative(self, axis: TextualAxis) -> Proposal | None:
        """Single-axis refinement: chain from the last attempt (momentum)."""
        if self._last_text is not None:
            return await self._propose(
                axis, source_text=self._last_text, context_report=self._last_report
            )
        return await self._propose(
            axis, source_text=None, context_report=self._best_report
        )

    async def _propose(
        self,
        axis: TextualAxis,
        source_text: str | None,
        context_report: TestReport | None,
    ) -> Proposal:
        """Run one Critic→Applier step against ``source_text``.

        ``source_text`` is the momentum source (the last attempt's text); when
        ``None`` it is resolved from the current best agent. Unparseable subagent
        output yields an errored proposal (a skipped trial), never a crash.
        ``ask`` guarantees ``self._best_agent`` is set before this is reached.
        """
        try:
            if source_text is None:
                source_text = str(get_path(self._best_agent, axis.path))
            context = build_failing_context(context_report)
            gradient = await self._critique(source_text, context)
            new_text, summary = await self._apply_edit(
                source_text, gradient, axis.max_chars
            )
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            logger.warning("Textual proposal for '%s' failed: %s", axis.path, exc)
            return Proposal(
                textual_axis=axis.path, error=f"textual proposal failed: {exc}"
            )

        return Proposal(textual_axis=axis.path, new_text=new_text, edit_summary=summary)

    async def _critique(self, current_text: str, context: str) -> str:
        """Ask the Critic for a natural-language gradient.

        On a refinement step the prompt also carries the prior attempt's loss and
        a bounded history of recent edits — the natural-language gradient — so the
        Critic can build on what was already tried. Only instruction text,
        failing-case context, losses, and edit summaries are included; never
        credentials.
        """
        step = len(self._history) + 1
        parts = [
            f"Current agent instructions (refinement step {step}):",
            current_text,
            "",
            "Failing test cases:",
            context or "(none provided)",
            "",
        ]
        if self._last_loss is not None:
            parts.append(
                f"Your previous rewrite scored a loss of {self._last_loss:.3f} "
                "(lower is better)."
            )
        if self._history:
            parts.append("Recent attempts (most recent last):")
            for summary, loss, accepted in self._history:
                status = "accepted" if accepted else "rejected"
                parts.append(
                    f"- {summary or '(no summary)'} → loss {loss:.3f} ({status})"
                )
            parts.append("")
        parts.append(
            "Return JSON with a 'gradient' field describing the single most "
            "impactful next improvement to the instructions."
        )
        result = await self._invoke(self._critic, "\n".join(parts))
        data = _parse_json_result(result)
        gradient = data.get("gradient")
        if not isinstance(gradient, str) or not gradient.strip():
            raise KeyError("gradient")
        return gradient

    async def _apply_edit(
        self, current_text: str, gradient: str, max_chars: int
    ) -> tuple[str, str | None]:
        """Ask the Applier to rewrite the instructions per the gradient."""
        prompt = (
            "Current agent instructions:\n"
            f"{current_text}\n\n"
            "Improvement to apply:\n"
            f"{gradient}\n\n"
            f"Rewrite the instructions (at most {max_chars} characters). "
            "Return JSON with 'new_text' (the rewritten instructions) and an "
            "optional 'summary'."
        )
        result = await self._invoke(self._applier, prompt)
        data = _parse_json_result(result)
        new_text = data.get("new_text")
        if not isinstance(new_text, str) or not new_text.strip():
            raise KeyError("new_text")
        summary = data.get("summary")
        return new_text[:max_chars], summary if isinstance(summary, str) else None

    def tell(
        self,
        proposal: Proposal,
        loss: float,
        accepted: bool,
        report: TestReport | None = None,
    ) -> None:
        """Advance the momentum chain with the scored attempt.

        The next refinement chains from this attempt's text (whether or not it
        was accepted — in-place TextGrad step); the loop's accept gate, not the
        proposer, decides what becomes ``best_agent``. An errored proposal was
        never scored, so it does not advance the chain.
        """
        if proposal.new_text is None:
            return
        self._last_text = proposal.new_text
        self._last_loss = loss
        self._last_report = report
        self._history.append((proposal.edit_summary, loss, accepted))
        if len(self._history) > _MAX_HISTORY:
            self._history = self._history[-_MAX_HISTORY:]
