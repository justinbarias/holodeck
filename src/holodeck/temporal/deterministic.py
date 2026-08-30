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
through an activity would put the whole table in the workflow history. Load it
at module scope::

    with workflow.unsafe.imports_passed_through():
        from holodeck.temporal.deterministic import evaluate, load_decision_table

    _TABLE = load_decision_table(Path(__file__).parent / "hardship.dmn.yaml")

``load_decision_table`` reads a file, so it is the one member of this surface
that is not callable from inside a workflow run.
"""

from __future__ import annotations

from holodeck.lib.workflow.edge import check_gate
from holodeck.lib.workflow.table_eval import Verdict, evaluate
from holodeck.models.decision_table import DecisionTable, load_decision_table
from holodeck.temporal.models import ActivityParameters

__all__ = [
    "ActivityParameters",
    "DecisionTable",
    "Verdict",
    "check_gate",
    "evaluate",
    "load_decision_table",
]
