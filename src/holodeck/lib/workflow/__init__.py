"""Deterministic primitives kept from the 036 spine (archived).

FEEL evaluation (``feel``), DMN decision-table evaluation (``table_eval``),
and the schema-gated edge executor (``edge``). The DAG runner, ``input_data``
validation, and the ``holodeck workflow`` CLI were removed in the pivot to
Temporal — see ``SPEC.md`` at the project root. These modules are the reuse
surface for the Temporal activity wrapper (gate inside the activity) and the
workflow-safe decision-table helper.
"""
