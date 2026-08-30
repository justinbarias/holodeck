"""Workflow modules used by the sandbox-safety test (spec 040, T5).

These live in their own module rather than inside the test because Temporal's
workflow sandbox **re-imports the module that defines the workflow** in order
to validate it. A workflow defined inside a test module would drag pytest, the
test's fixtures, and every other import of that file through the sandbox, and
the test would measure those instead of the D3 surface.

``DeterministicWorkflow`` deliberately imports
:mod:`holodeck.temporal.deterministic` *without*
``workflow.unsafe.imports_passed_through()``, so the sandbox really does
re-import the HoloDeck helper modules and check every access they make.
``PassedThroughWorkflow`` shows the shape a consumer actually writes (and the
one ``tests/integration/temporal/workflows.py`` uses): the surface is
deterministic, so it may simply be passed through.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from temporalio import workflow

# No imports_passed_through(): the point of this module is to make the sandbox
# re-import and validate the HoloDeck helpers themselves.
from holodeck.temporal.deterministic import (  # noqa: E402
    ActivityParameters,
    check_gate,
    evaluate,
    load_decision_table,
)

GATE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"approved": {"type": "boolean"}},
    "required": ["approved"],
}


@workflow.defn
class DeterministicWorkflow:
    """Uses the D3 surface with nothing passed through."""

    @workflow.run
    async def run(self, candidate: dict[str, Any]) -> dict[str, Any]:
        """Gate the candidate object and return it.

        Args:
            candidate: The object to put through the gate.

        Returns:
            The gate-validated object.
        """
        # Named so the surface is exercised, not merely imported.
        assert callable(evaluate)
        assert callable(load_decision_table)
        params = ActivityParameters(start_to_close=timedelta(minutes=1))
        assert "start_to_close_timeout" in params.to_activity_kwargs()
        return check_gate(candidate, GATE_SCHEMA, node_id="sandbox")


with workflow.unsafe.imports_passed_through():
    from holodeck.temporal.deterministic import check_gate as passed_through_check_gate


@workflow.defn
class PassedThroughWorkflow:
    """The shape a consumer writes: the D3 surface marked as pass-through."""

    @workflow.run
    async def run(self, candidate: dict[str, Any]) -> dict[str, Any]:
        """Gate the candidate object and return it.

        Args:
            candidate: The object to put through the gate.

        Returns:
            The gate-validated object.
        """
        return passed_through_check_gate(candidate, GATE_SCHEMA, node_id="sandbox")
