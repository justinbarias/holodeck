"""Run the real worker CLI with a deterministic backend in this subprocess.

This module is the process boundary for spec 040's AC-4 test. It patches the
same deferred ``BackendSelector`` module attribute as the T10 acceptance suite,
then delegates argument parsing and worker startup to the real Click command.
No pytest hook or ``sitecustomize`` participates, so ``python -m`` remains a
separate OS process exercising the production CLI path.
"""

from __future__ import annotations

from typing import Any

from holodeck.lib.backends import selector as selector_module
from holodeck.lib.backends.base import ExecutionResult
from holodeck.models.agent import Agent

EVIDENCE_AGENT = "hardship-evidence-extractor"
LETTER_AGENT = "hardship-letter-writer"

EVIDENCE_OUTPUT: dict[str, Any] = {
    "income": {"net": 5000, "expenses": 3500},
    "residency": {"status": "verified"},
}

LETTER_OUTPUT: dict[str, Any] = {
    "letter": "Dear applicant, we have completed our review of your case.",
    "tone": "neutral",
}

_SCRIPT: dict[str, dict[str, Any]] = {
    EVIDENCE_AGENT: EVIDENCE_OUTPUT,
    LETTER_AGENT: LETTER_OUTPUT,
}


class _ScriptedBackend:
    """Return one canned structured result without contacting a model."""

    def __init__(self, output: dict[str, Any]) -> None:
        self._output = output

    async def invoke_once(
        self, message: str, context: list[dict[str, Any]] | None = None
    ) -> ExecutionResult:
        """Return the output bound to this agent.

        Args:
            message: Unused prompt from the activity.
            context: Unused prior-turn context.

        Returns:
            A successful result carrying the canned structured object.
        """
        return ExecutionResult(response="", structured_output=self._output)

    async def teardown(self) -> None:
        """Release the fake backend's empty resource set."""


class _ScriptedSelector:
    """Route fixture agent names to canned backends."""

    @classmethod
    async def select(
        cls,
        agent: Agent,
        tool_instances: dict[str, Any] | None = None,
        mode: str = "test",
    ) -> _ScriptedBackend:
        """Build the canned backend for ``agent``.

        Args:
            agent: Loaded hardship agent configuration.
            tool_instances: Unused initialized tools.
            mode: Unused execution mode.

        Returns:
            A backend that returns the agent's deterministic fixture object.
        """
        return _ScriptedBackend(_SCRIPT[agent.name])


def main() -> None:
    """Install the scripted selector and invoke the real worker CLI."""
    selector_module.BackendSelector = _ScriptedSelector

    # Imported after patching so every CLI import observes the same selector
    # module object. The activity itself resolves this binding lazily per turn.
    from holodeck.cli.main import main as cli_main

    cli_main()


if __name__ == "__main__":
    main()
