"""Decision-table evaluation and hit policies for the deterministic spine (036, T4).

Evaluates a loaded :class:`~holodeck.models.decision_table.DecisionTable`
against a context of named inputs (the gated objects / verdicts of the input
nodes) and produces exactly one :class:`Verdict`, or fails loudly.

Evaluation is pure: input expressions and unary-test cells go through the
embedded FEEL evaluator (``holodeck.lib.workflow.feel``), never an LLM.

The two FR-012 loud failures live here — no rule matched and no ``default`` is
declared, and a ``UNIQUE`` table that matched more than one rule — both raising
:class:`~holodeck.lib.errors.TableEvalError`. FEEL failures keep their own
locator-bearing :class:`~holodeck.lib.errors.FeelEvaluationError` and propagate
unchanged.

Hit-policy evaluation is in-house by decision (research.md, "Hit-policy
evaluation: in-house (T4)"): the embedded evaluator handles expressions only.
"""

from __future__ import annotations

import copy
import datetime
from collections.abc import Iterator, Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator

from holodeck.lib.errors import TableEvalError
from holodeck.lib.workflow import feel
from holodeck.models.decision_table import (
    DecisionTable,
    FeelType,
    HitPolicy,
    Rule,
    TableInput,
)


class Verdict(BaseModel):
    """The single decision a table produced, with its provenance.

    Carries enough identity for the run record (T10) and OTel spans (T13) to
    reconstruct *which* rule of *which* table version decided: a no-match that
    fell through to the table's ``default`` is marked explicitly rather than
    being indistinguishable from a rule hit.

    ``outputs`` is deep-copied on construction, the same as
    :class:`~holodeck.lib.workflow.edge.GatedOutput`. Pydantic's ``frozen``
    only rebind-guards the top-level attributes; without the copy a caller
    holding the returned verdict could mutate a nested value and corrupt the
    ``DecisionTable`` the outputs were built from (``rule.then``/``default``
    are shared, not copied, by :func:`evaluate`).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    table_id: str = Field(description="Id of the table that produced the verdict.")
    table_version: str = Field(
        description="Version label of the table, snapshotted for replay (FR-013).",
    )
    outputs: dict[str, Any] = Field(
        description="The emitted output values, one per declared output.",
    )
    matched_rule_index: int | None = Field(
        description="1-based index of the winning rule, or None when the "
        "table's default was used.",
    )
    matched_rule_annotation: str | None = Field(
        default=None,
        description="Annotation of the winning rule, when it carries one.",
    )
    is_default: bool = Field(
        default=False,
        description="True when no rule matched and the declared default was "
        "emitted.",
    )

    @field_validator("outputs", mode="before")
    @classmethod
    def _detach(cls, value: Any) -> Any:
        """Deep-copy the mapping so ``frozen`` holds all the way down.

        Args:
            value: The mapping as supplied by the caller.

        Returns:
            A copy that shares no nested container with the caller's object.
        """
        return copy.deepcopy(value)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def rule_identity(self) -> str:
        """Human-readable identity of the decision source, for logs and spans."""
        if self.is_default:
            return "default, no rule matched"
        return f"rule {self.matched_rule_index}"


def evaluate(table: DecisionTable, named_inputs: Mapping[str, Any]) -> Verdict:
    """Evaluate a decision table against named inputs and return one verdict.

    Each input column's ``expression`` is evaluated once against
    ``named_inputs``; every rule's ``when`` cells are then evaluated as DMN
    unary tests against those column values. A cell absent from ``when`` is the
    irrelevant cell and always matches; a rule matches when all of its cells do.
    The table's hit policy resolves the matches.

    Args:
        table: The loaded, statically validated decision table.
        named_inputs: Variable bindings for the input expressions — the gated
            objects / verdicts of the input nodes, keyed by node name.

    Returns:
        The :class:`Verdict` for the winning rule, or for the table's
        ``default`` when no rule matched.

    Raises:
        TableEvalError: If no rule matched and the table declares no
            ``default``, if a ``UNIQUE`` table matched more than one rule, if a
            ``days`` column produced a value that is not a duration, or if a
            ``number`` column produced a boolean.
        FeelEvaluationError: If an input expression or a rule cell fails to
            evaluate (propagated with its locator).
    """
    column_values = {
        inp.name: _column_value(table, inp, named_inputs) for inp in table.inputs
    }
    matches = _matching_rules(table, column_values)

    if table.hit_policy is HitPolicy.FIRST:
        # Document order decides, so stop at the first hit rather than
        # evaluating cells that cannot change the outcome.
        first = next(matches, None)
        if first is None:
            return _no_match(table)
        return _verdict(table, *first)

    matched = list(matches)
    if not matched:
        return _no_match(table)

    if table.hit_policy is HitPolicy.UNIQUE:
        if len(matched) > 1:
            indices = [index for index, _ in matched]
            raise TableEvalError(
                f"table '{table.id}': hit policy UNIQUE requires exactly one "
                f"matching rule, but {len(matched)} matched: rules {indices}"
            )
        return _verdict(table, *matched[0])

    # PRIORITY: DMN ranks matched rules by the position of their output entries
    # in the output's declared `values` list (index 0 = highest priority). With
    # several outputs the rank is the tuple of per-output positions in declared
    # output order, compared lexicographically — the first output dominates, and
    # later outputs only break ties. Document order breaks a full tie (`min`
    # keeps the earliest minimum).
    index, rule = min(matched, key=lambda match: _priority_rank(table, match[1]))
    return _verdict(table, index, rule)


def _matching_rules(
    table: DecisionTable, column_values: dict[str, Any]
) -> Iterator[tuple[int, Rule]]:
    """Yield ``(1-based index, rule)`` for every rule whose cells all match."""
    for index, rule in enumerate(table.rules, start=1):
        if all(
            feel.evaluate_unary_test(
                cell,
                column_values[cell_name],
                locator=f"table '{table.id}' rule {index} cell '{cell_name}'",
            )
            for cell_name, cell in rule.when.items()
        ):
            yield index, rule


def _column_value(
    table: DecisionTable, inp: TableInput, named_inputs: Mapping[str, Any]
) -> Any:
    """Evaluate one input column's expression, coercing ``days`` columns."""
    locator = f"table '{table.id}' input '{inp.name}'"
    value = feel.evaluate_expression(
        inp.expression, dict(named_inputs), locator=locator
    )
    if inp.type is FeelType.DAYS:
        return _to_days(value, locator)
    if inp.type is FeelType.NUMBER:
        return _to_number(value, locator)
    return value


def _to_days(value: Any, locator: str) -> float:
    """Coerce a ``days`` column value to a number of days.

    Bare date subtraction in FEEL yields a ``datetime.timedelta``, which is not
    comparable against the numeric literals rule cells use (research.md caveat
    1), so the conversion happens at the column boundary. An expression that
    already produced a number is widened to the same ``float``, so whole days,
    fractional days and negative durations all compare alike.

    Args:
        value: The evaluated column value.
        locator: Human-readable location for the error message.

    Returns:
        The value as a number of days.

    Raises:
        TableEvalError: If the value is neither a duration nor a number.
    """
    if isinstance(value, datetime.timedelta):
        return value.total_seconds() / 86400
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    raise TableEvalError(
        f"{locator}: a 'days' column must evaluate to a duration or a number, "
        f"got {type(value).__name__}"
    )


def _to_number(value: Any, locator: str) -> Any:
    """Reject a boolean in a ``number`` column; pass anything else through.

    ``bool`` is an ``int`` subclass, so the embedded evaluator would happily
    compare ``True`` against ``<= 90`` as ``1`` and match — a silent
    mistyping. Non-numeric values still fail loudly at the first rule cell with
    that cell's locator, which is more precise than failing here.

    Args:
        value: The evaluated column value.
        locator: Human-readable location for the error message.

    Returns:
        The value, unchanged.

    Raises:
        TableEvalError: If the value is a boolean.
    """
    if isinstance(value, bool):
        raise TableEvalError(
            f"{locator}: a 'number' column must not evaluate to a boolean"
        )
    return value


def _priority_rank(table: DecisionTable, rule: Rule) -> tuple[int, ...]:
    """Rank a matched rule under PRIORITY; lower tuples are higher priority."""
    ranks: list[int] = []
    for output in table.outputs:
        # T3 guarantees every output declares `values` under PRIORITY and that
        # every rule's entry is one of them, so the lookup always resolves.
        values = output.values or []
        ranks.append(values.index(rule.then[output.name]))
    return tuple(ranks)


def _verdict(table: DecisionTable, index: int, rule: Rule) -> Verdict:
    """Build the verdict for a winning rule."""
    return Verdict(
        table_id=table.id,
        table_version=table.version,
        outputs=dict(rule.then),
        matched_rule_index=index,
        matched_rule_annotation=rule.annotation,
    )


def _no_match(table: DecisionTable) -> Verdict:
    """Emit the declared default, or fail loudly when there is none (FR-012)."""
    if table.default is None:
        raise TableEvalError(
            f"table '{table.id}': no rule matched and the table declares no "
            f"default output"
        )
    return Verdict(
        table_id=table.id,
        table_version=table.version,
        outputs=dict(table.default),
        matched_rule_index=None,
        is_default=True,
    )
