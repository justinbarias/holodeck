"""The hardship table, loaded once at import time (spec 040, T9, decision 7).

Kept in its own module so the workflow can import it under
``workflow.unsafe.imports_passed_through()``. That is what makes the pattern
work under Temporal's sandbox: the sandbox re-imports the *workflow's* module
to validate it, but a passed-through import is served from the already-loaded
real module instead of being executed again. The file read therefore happens
exactly once, in the worker process, at registration — never on a replay and
never inside the sandbox, which is what ``load_decision_table`` enforces.

The workflow module itself stays under full sandbox validation, which is the
point: only the one line that touches the disk is lifted out.
"""

from __future__ import annotations

from pathlib import Path

from holodeck.temporal.deterministic import DecisionTable, load_decision_table

TABLE_PATH = Path(__file__).parent / "tables" / "hardship.dmn.yaml"

TABLE: DecisionTable = load_decision_table(TABLE_PATH)

__all__ = ["TABLE", "TABLE_PATH"]
