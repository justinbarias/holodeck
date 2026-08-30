"""The sample hardship workflow: extract → decide → write (spec 040, T9).

This is the shape a HoloDeck consumer authors. HoloDeck ships the two agents
as activities; the *control flow* — what runs, in what order, and what the
policy decides — is this file, and it is ordinary Temporal workflow code.

Three properties the sample exists to demonstrate:

* **Timeouts and retries are the caller's.** Each ``execute_activity`` carries
  its own :class:`ActivityParameters` (decision 10). Extraction is retried:
  a gate rejection is evidence about the model, and a second attempt is
  usually cheaper than failing the run. Letter writing is not: by then the
  decision is already made, and a repeated call would only re-bill it.
* **The policy decision never touches the model.** The verdict comes from
  :func:`evaluate` over a versioned decision table, inside workflow code, on
  the gate-validated object. No LLM is asked what the answer is.
* **The table loads at module import time** (decision 7), never during a run:
  a table is policy-as-code deployed with the workflow, so it must be the same
  table on every replay.

The evidence gate's shape *is* the table's named-input shape, so the gated
object is handed to ``evaluate`` with no mapping step in between — the gate is
what makes that safe.

Sandbox note: the HoloDeck imports sit inside
``workflow.unsafe.imports_passed_through()`` because the D3 surface is
deterministic and need not be re-imported by the sandbox. This module itself
stays fully under sandbox validation — only the one import that touches the
disk (the table in ``policy.py``) is lifted out.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    # The table is read at import time, once, in the worker process — never on
    # a replay. It lives in `policy.py` so that this import is served from the
    # already-loaded module when the sandbox re-imports *this* file; see that
    # module's docstring.
    from policy import TABLE

    from holodeck.temporal.deterministic import ActivityParameters, evaluate
    from holodeck.temporal.models import AgentActivityInput, AgentActivityResult

# Extraction may be retried — a gate rejection is evidence about the model, and
# the next attempt often lands. `GateSchemaError`/`ConfigError` never reach here
# retryably (the activity classifies authoring faults as non-retryable), so the
# attempts are spent only on faults a retry could actually fix.
EVIDENCE_PARAMETERS = ActivityParameters(
    start_to_close=timedelta(minutes=3),
    maximum_attempts=3,
    initial_interval=timedelta(seconds=2),
)

# Letter writing is not retried: the decision is already made by then, and a
# second call only re-bills the same letter.
LETTER_PARAMETERS = ActivityParameters(
    start_to_close=timedelta(minutes=3),
    maximum_attempts=1,
)

LETTER_BRIEF = (
    "Write the hardship decision letter for this applicant. The affordability "
    "decision has already been made; it is in the context object."
)


@workflow.defn
class HardshipWorkflow:
    """Extracts evidence, decides affordability by table, then writes back."""

    @workflow.run
    async def run(self, statement: str) -> dict[str, Any]:
        """Run one hardship application end to end.

        Args:
            statement: The applicant's free-text hardship statement.

        Returns:
            The gate-validated letter object: ``letter`` and ``tone``.
        """
        evidence: AgentActivityResult = await workflow.execute_activity(
            "evidence",
            AgentActivityInput(message=statement),
            result_type=AgentActivityResult,
            **EVIDENCE_PARAMETERS.to_activity_kwargs(),
        )

        # Deterministic policy, in workflow code, over the gated object. The
        # gate's shape is the table's input shape, so nothing is mapped here.
        verdict = evaluate(TABLE, evidence.output)

        letter: AgentActivityResult = await workflow.execute_activity(
            "letter",
            AgentActivityInput(
                message=LETTER_BRIEF,
                context={
                    "affordability": verdict.outputs["affordability"],
                    "policy": verdict.table_id,
                    "policy_version": verdict.table_version,
                    "evidence": evidence.output,
                },
            ),
            result_type=AgentActivityResult,
            **LETTER_PARAMETERS.to_activity_kwargs(),
        )
        return letter.output
