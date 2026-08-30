"""User-authored workflow for the live phase-1 e2e test.

Lives in its own module — exactly as a spec-040 consumer would structure it —
because the workflow sandbox re-imports the defining module for validation:
only sandbox-safe imports belong here, with the HoloDeck D3 surface marked as
pass-through (it is deterministic; the sandbox need not re-import it).
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from holodeck.temporal.models import (
        ActivityParameters,
        AgentActivityInput,
        AgentActivityResult,
    )


@workflow.defn
class EvidenceWorkflow:
    """Calls the agent activity by name and returns its gated output."""

    @workflow.run
    async def run(self, statement: str) -> dict[str, Any]:
        """Run one evidence-extraction turn through the agent activity.

        Args:
            statement: The applicant statement to extract evidence from.

        Returns:
            The gate-validated output dict from the activity envelope.
        """
        params = ActivityParameters(
            start_to_close=timedelta(minutes=3),
            maximum_attempts=1,
        )
        result = await workflow.execute_activity(
            "evidence",
            AgentActivityInput(message=statement),
            result_type=AgentActivityResult,
            **params.to_activity_kwargs(),
        )
        return result.output
