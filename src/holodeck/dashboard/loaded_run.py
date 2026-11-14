"""A saved run as the dashboard loaded it, with a stable id.

The dashboard addresses a run by :attr:`LoadedRun.run_id`, never by its
timestamp, because two runs can be saved with the same timestamp. The id is
the result file's name without ``.json``. The writer adds a 4-hex suffix when
that name is taken, so the ids in one results directory are distinct. Seed
runs have no file and take the same form from their timestamp.

``run_id`` is not saved: dumps exclude it, so a downloaded run reloads as a
plain :class:`EvalRun`.
"""

from __future__ import annotations

from pydantic import Field

from holodeck.models.eval_run import EvalRun


class LoadedRun(EvalRun):
    """An :class:`EvalRun` plus the id the dashboard addresses it by."""

    run_id: str = Field(
        ...,
        min_length=1,
        exclude=True,
        description=(
            "Result file name without '.json'; used in callbacks, compare "
            "columns, CSV rows and copied links."
        ),
    )

    @classmethod
    def from_run(cls, run: EvalRun, run_id: str) -> LoadedRun:
        """Attach ``run_id`` to an already validated run."""
        return cls(report=run.report, metadata=run.metadata, run_id=run_id)


def run_id_from_timestamp(timestamp: str) -> str:
    """The file-name form of ``timestamp`` (``:`` becomes ``-``), as the
    eval-run writer names result files."""
    return timestamp.replace(":", "-")
