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

    def begin(self, best_agent: Agent, best_report: TestReport | None) -> None:
        """Re-initialize for a new textual phase against the current best."""
        self._best_agent = best_agent
        self._best_report = best_report
        self._index = 0

    async def ask(self) -> Proposal | None:
        """Propose the next instruction rewrite, or None when axes are consumed."""
        if self._index >= len(self._axes) or self._best_agent is None:
            return None
        axis = self._axes[self._index]
        self._index += 1

        try:
            current_text = str(get_path(self._best_agent, axis.path))
            context = build_failing_context(self._best_report)
            gradient = await self._critique(current_text, context)
            new_text, summary = await self._apply_edit(
                current_text, gradient, axis.max_chars
            )
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            logger.warning("Textual proposal for '%s' failed: %s", axis.path, exc)
            return Proposal(
                textual_axis=axis.path, error=f"textual proposal failed: {exc}"
            )

        return Proposal(textual_axis=axis.path, new_text=new_text, edit_summary=summary)

    async def _critique(self, current_text: str, context: str) -> str:
        """Ask the Critic for a natural-language gradient."""
        prompt = (
            "Current agent instructions:\n"
            f"{current_text}\n\n"
            "Failing test cases:\n"
            f"{context or '(none provided)'}\n\n"
            "Return JSON with a 'gradient' field describing the single most "
            "impactful improvement to the instructions."
        )
        result = await self._invoke(self._critic, prompt)
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

    def tell(self, proposal: Proposal, loss: float, accepted: bool) -> None:
        """No-op: the Critic/Applier are stateless across trials."""
