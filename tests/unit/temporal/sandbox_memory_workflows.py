"""A workflow that schedules the memory ingestion activity (spec 043, HST-03).

Kept in its own module because the sandbox re-imports the module a workflow is
defined in (see ``sandbox_workflows.py``). It imports the activity name and
payload models from :mod:`holodeck.temporal.models` with no pass-through, so
the sandbox re-imports and validates them. It never imports the activity
module or any memory code: a workflow schedules the activity by name.
"""

from __future__ import annotations

from datetime import timedelta

from temporalio import workflow

from holodeck.temporal.models import (
    MEMORY_INGEST_ACTIVITY,
    ActivityParameters,
    MemoryIngestInput,
    MemoryIngestResult,
)


@workflow.defn
class MemoryIngestWorkflow:
    """Runs the ingestion activity for one node and returns its counts."""

    @workflow.run
    async def run(self, node_id: str) -> MemoryIngestResult:
        """Schedule the ingestion activity.

        Args:
            node_id: The node whose memory to ingest.

        Returns:
            The activity's result.
        """
        params = ActivityParameters(start_to_close=timedelta(hours=2))
        result: MemoryIngestResult = await workflow.execute_activity(
            MEMORY_INGEST_ACTIVITY,
            MemoryIngestInput(node_id=node_id),
            result_type=MemoryIngestResult,
            heartbeat_timeout=timedelta(minutes=2),
            **params.to_activity_kwargs(),
        )
        return result
