"""The positive control for the sandbox-safety harness (spec 040, T5).

A workflow module that reads the wall clock while it is being imported. The
sandbox re-imports the defining module, so validation must reject this one —
without that, a harness that passes everything would prove nothing about the
D3 surface.
"""

from __future__ import annotations

import datetime

from temporalio import workflow

# Restricted: the sandbox refuses datetime.datetime.utcnow, and this runs while
# the sandbox is importing this very module.
IMPORTED_AT = datetime.datetime.utcnow()


@workflow.defn
class NondeterministicWorkflow:
    """Never validates — it is here to prove the harness can fail."""

    @workflow.run
    async def run(self) -> str:
        """Return the import timestamp.

        Returns:
            The wall-clock time this module was imported.
        """
        return IMPORTED_AT.isoformat()
