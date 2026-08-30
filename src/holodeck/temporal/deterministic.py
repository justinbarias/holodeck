"""The deterministic surface a Temporal workflow may import (spec 040, D3).

Everything re-exported here is pure: no network, no clock, no filesystem at
call time, no backend SDK anywhere in its import graph. That is what lets a
workflow module import it under
``with workflow.unsafe.imports_passed_through():`` and still pass Temporal's
sandbox validation — proved by
``tests/unit/temporal/test_sandbox_safety.py``.

Nothing is reimplemented here. The surface is exactly the 036 primitives plus
the T2 scheduling helper, gathered into one import a workflow author can name:

* :func:`evaluate` — decision-table evaluation (hit policies, FEEL cells),
  returning a :class:`~holodeck.lib.workflow.table_eval.Verdict`.
* :func:`check_gate` — the schema gate on a plain object, the same check the
  agent activity applies to what the model produced.
* :class:`ActivityParameters` — builds the ``execute_activity`` timeout and
  retry keyword arguments, which in Temporal are the caller's to set
  (decision 10).
* :func:`load_decision_table` — **import-time only**, see below.

Decision tables load at workflow-module import time, never inside a workflow
run and never inside an activity: a table is policy-as-code, versioned and
deployed with the workflow that reads it, so it must be the same table on
every replay. Reading it during a run would put file I/O in the sandbox and
make replay depend on the worker's filesystem at that moment; passing it
through an activity would put the whole table in the workflow history.

Load it at module scope of a **sibling module** — not the workflow's own
module. The sandbox re-imports the workflow's defining module to validate it,
so a load there is re-executed inside the sandbox and refused; an import
served through ``imports_passed_through()`` is not re-executed::

    # policy.py — loaded once, in the worker process
    _TABLE = load_decision_table(Path(__file__).parent / "hardship.dmn.yaml")

    # workflow.py
    with workflow.unsafe.imports_passed_through():
        from holodeck.temporal.deterministic import evaluate
        from myproject.policy import _TABLE

``load_decision_table`` reads a file, so it is the one member of this surface
that is not callable from inside a workflow run.
"""

from __future__ import annotations

from pathlib import Path

from temporalio import workflow

from holodeck.lib.errors import ConfigError
from holodeck.lib.workflow.edge import check_gate
from holodeck.lib.workflow.table_eval import Verdict, evaluate
from holodeck.models.decision_table import DecisionTable
from holodeck.models.decision_table import load_decision_table as _load_decision_table
from holodeck.temporal.models import ActivityParameters


def load_decision_table(path: Path | str) -> DecisionTable:
    """Load a decision table — at workflow-module import time only.

    A table is policy-as-code, versioned with the workflow that reads it, so it
    must be the same table on every replay. This wrapper enforces what the
    module docstring documents: calling it from inside a workflow run (or from
    code re-imported inside the workflow sandbox) is refused, because file I/O
    there would make replay depend on the worker's filesystem at that moment.

    Args:
        path: The decision-table YAML file.

    Returns:
        The loaded :class:`DecisionTable`.

    Raises:
        ConfigError: If called during workflow execution or inside the
            workflow sandbox instead of at module import time.
        DecisionTableError: If the file is missing or not a valid table.
    """
    if workflow.unsafe.in_sandbox() or workflow.in_workflow():
        raise ConfigError(
            "load_decision_table",
            "decision tables load at import time, outside the sandbox; move "
            "this call to module scope of a sibling module (not the workflow's "
            "own module — the sandbox re-imports that) and import it under "
            "imports_passed_through (decision 7)",
        )
    return _load_decision_table(path)


__all__ = [
    "ActivityParameters",
    "DecisionTable",
    "Verdict",
    "check_gate",
    "evaluate",
    "load_decision_table",
]
