"""Decision-table models for the deterministic spine (036, T3).

Defines the ``tables/*.dmn.yaml`` artifact: one DMN decision table — input
expressions (full FEEL), unary-test rule cells (keyed by input name), output
values, a hit policy, and a required version label (FR-013). Loading a table
statically validates every FEEL expression against the FR-010 subset with a
table/rule/cell locator (FR-010/FR-012), so off-subset or malformed policy
logic never reaches evaluation.

Two failure channels, matching ``holodeck.lib.errors``:

* **Structural / cross-field** problems (duplicate names, a rule cell naming an
  undeclared input, an output entry outside its declared values, an incomplete
  ``then``) raise :class:`DecisionTableError`.
* **FEEL** problems (malformed or out-of-subset expressions) raise
  :class:`FeelValidationError` with a locator, via the FEEL wrapper.

Plain field-shape errors (missing ``version``, wrong type) surface as pydantic
``ValidationError`` as usual.

Design input: ``specs/036-deterministic-spine/dmn-yaml-mapping.md``.
"""

from __future__ import annotations

import sys
from collections import Counter
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from holodeck.lib.errors import DecisionTableError
from holodeck.lib.workflow import feel

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self


# ``provenance`` records how a table was authored, never what it decides.
# Reserving the root name keeps it out of every input expression and rule cell,
# so it can never influence a match (FR-032).
_RESERVED_FEEL_ROOTS: frozenset[str] = frozenset({"provenance"})


class HitPolicy(str, Enum):
    """DMN hit policy — how a table resolves when several rules match.

    ``UNIQUE`` requires exactly one match (multi-match is a loud error per
    FR-012); ``FIRST`` takes the first matching rule in document order;
    ``PRIORITY`` resolves by the output's ``values`` ordering (highest first).
    """

    UNIQUE = "UNIQUE"
    FIRST = "FIRST"
    PRIORITY = "PRIORITY"


class FeelType(str, Enum):
    """The declared type of a table input or output (DMN ``typeRef``).

    ``days`` is a HoloDeck convenience for a date difference coerced to a
    number of days (research.md caveat 1); ``date`` columns are supplied as
    ``datetime.date`` objects at the workflow boundary (caveat 6).
    """

    NUMBER = "number"
    STRING = "string"
    BOOLEAN = "boolean"
    DAYS = "days"
    DATE = "date"


class TableInput(BaseModel):
    """A table input column: a full FEEL expression over the named inputs.

    The ``expression`` is the only place full FEEL (arithmetic, date
    differences, dot-paths) is allowed; rule cells are restricted to unary
    tests against the value this expression produces.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Column name; rule cells are keyed by it.")
    expression: str = Field(
        description="Full FEEL expression evaluated against the workflow inputs.",
    )
    type: FeelType = Field(description="Declared type of the column value.")


class TableOutput(BaseModel):
    """A table output column (DMN ``<output>``).

    When ``values`` is declared, every rule (and the default) must emit one of
    them; under ``PRIORITY`` the list also fixes the resolution order, highest
    first.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Output field name in the emitted verdict.")
    type: FeelType = Field(description="Declared type of the output value.")
    values: list[str] | None = Field(
        default=None,
        description="Allowed output values; doubles as PRIORITY ordering.",
    )


class Provenance(BaseModel):
    """How a decision table came to exist (FR-029).

    Non-executable metadata: it is snapshotted into the run record and the
    node's OTel span (FR-031) but is never visible to FEEL and never affects
    rule matching (FR-032). A hand-authored table omits the block entirely.
    """

    model_config = ConfigDict(extra="forbid")

    generated_by: str | None = Field(
        default=None,
        description="Model identifier that drafted the table; absent when "
        "hand-authored.",
    )
    source: str | None = Field(
        default=None,
        description="The policy authority, e.g. 'Social Security Guide 3.11.13.50'.",
    )
    source_doc: str | None = Field(
        default=None,
        description="Path or reference to the source document.",
    )
    source_sha256: str | None = Field(
        default=None,
        description="SHA-256 digest of the source document.",
    )
    reviewed_by: str | None = Field(
        default=None,
        description="Human reviewer who signed the table off.",
    )
    reviewed_at: datetime | None = Field(
        default=None,
        description="When the review was recorded (ISO 8601).",
    )

    @property
    def awaiting_review(self) -> bool:
        """Whether generated policy still lacks a recorded human reviewer.

        Returns:
            ``True`` when ``generated_by`` is set and ``reviewed_by`` is not —
            the state the FR-030 review gate refuses to run.
        """
        return self.generated_by is not None and self.reviewed_by is None


class Rule(BaseModel):
    """One decision rule: keyed unary-test cells mapping to an output entry.

    ``when`` is keyed by input name (diff-friendly, immune to column reorder);
    an input absent from ``when`` is the irrelevant cell (always matches), as
    ``-`` is in DMN. ``then`` must produce every declared output (a verdict is
    fully specified).
    """

    model_config = ConfigDict(extra="forbid")

    when: dict[str, str] = Field(
        default_factory=dict,
        description="Unary-test cell per input name; omitted inputs are "
        "irrelevant (always match).",
    )
    then: dict[str, Any] = Field(
        description="Output entry — one value per declared output.",
    )
    annotation: str | None = Field(
        default=None,
        description="Optional human-readable rationale for the rule.",
    )


class DecisionTable(BaseModel):
    """The ``tables/*.dmn.yaml`` artifact: one DMN decision table.

    Validation at load enforces unique input/output names, that every rule
    cell and output entry references a declared column, output-value
    membership, ``then`` completeness, and the FEEL subset on every expression
    and cell — all before any evaluation (FR-010/FR-012). It also rejects any
    FEEL reference to the non-executable ``provenance`` block (FR-032).
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="Table identifier; used in locators and verdicts.")
    name: str | None = Field(default=None, description="Human-readable table name.")
    version: str = Field(
        description="Required version label, snapshotted into the run record "
        "(FR-013).",
    )
    hit_policy: HitPolicy = Field(description="How multiple matches resolve.")
    inputs: list[TableInput] = Field(
        min_length=1,
        description="Input columns (full FEEL expressions).",
    )
    outputs: list[TableOutput] = Field(
        min_length=1,
        description="Output columns produced by every rule.",
    )
    rules: list[Rule] = Field(
        min_length=1,
        description="The decision rules, in document order.",
    )
    default: dict[str, Any] | None = Field(
        default=None,
        description="Optional default output entry; without it, a no-match "
        "fails loudly (FR-012).",
    )
    source: str | None = Field(
        default=None,
        description="Optional authority annotation (knowledgeSource-lite).",
    )
    provenance: Provenance | None = Field(
        default=None,
        description="Optional non-executable record of how the table was "
        "authored and reviewed (FR-029).",
    )

    @model_validator(mode="after")
    def _validate(self) -> Self:
        """Run cross-field structural checks then static FEEL validation."""
        input_names = [inp.name for inp in self.inputs]
        output_names = [out.name for out in self.outputs]
        self._reject_duplicate_names(input_names, "input")
        self._reject_duplicate_names(output_names, "output")

        input_set = set(input_names)
        output_by_name = {out.name: out for out in self.outputs}

        # PRIORITY ranks matched rules by each output's `values` ordering
        # (highest first). Without `values` there is nothing to rank by, so the
        # table would silently degrade to first-match rather than failing. Catch
        # it at load time, where the locator is precise (FR-011, FR-012).
        if self.hit_policy is HitPolicy.PRIORITY:
            unranked = [out.name for out in self.outputs if out.values is None]
            if unranked:
                raise DecisionTableError(
                    f"table '{self.id}': hit policy PRIORITY requires 'values' "
                    f"on every output to define the priority ordering; "
                    f"missing on: {unranked}"
                )

        for idx, rule in enumerate(self.rules, start=1):
            for cell_name in rule.when:
                if cell_name not in input_set:
                    raise DecisionTableError(
                        f"table '{self.id}' rule {idx}: 'when' references "
                        f"unknown input '{cell_name}'"
                    )
            self._check_output_entry(rule.then, output_by_name, f"rule {idx} 'then'")
        if self.default is not None:
            self._check_output_entry(self.default, output_by_name, "default")

        # Static FEEL subset enforcement (raises FeelValidationError w/ locator).
        for inp in self.inputs:
            feel.validate_expression(
                inp.expression,
                locator=f"table '{self.id}' input '{inp.name}'",
                reserved_roots=_RESERVED_FEEL_ROOTS,
            )
        for idx, rule in enumerate(self.rules, start=1):
            for cell_name, cell in rule.when.items():
                feel.validate_unary_test(
                    cell,
                    locator=f"table '{self.id}' rule {idx} cell '{cell_name}'",
                    reserved_roots=_RESERVED_FEEL_ROOTS,
                )
        return self

    def _reject_duplicate_names(self, names: list[str], kind: str) -> None:
        """Raise if any input/output ``kind`` name is declared twice."""
        dups = sorted({n for n, count in Counter(names).items() if count > 1})
        if dups:
            raise DecisionTableError(
                f"table '{self.id}': duplicate {kind} name(s): {dups}"
            )

    def _check_output_entry(
        self,
        entry: dict[str, Any],
        output_by_name: dict[str, TableOutput],
        where: str,
    ) -> None:
        """Validate one ``then``/``default`` entry against the declared outputs.

        Every declared output must be present (a verdict is complete), no
        unknown outputs may appear, and any value for an output with declared
        ``values`` must be one of them.
        """
        keys = set(entry)
        declared = set(output_by_name)
        unknown = sorted(keys - declared)
        if unknown:
            raise DecisionTableError(
                f"table '{self.id}' {where}: unknown output(s): {unknown}"
            )
        missing = sorted(declared - keys)
        if missing:
            raise DecisionTableError(
                f"table '{self.id}' {where}: missing output(s): {missing}"
            )
        for name, value in entry.items():
            allowed = output_by_name[name].values
            if allowed is not None and value not in allowed:
                raise DecisionTableError(
                    f"table '{self.id}' {where}: output '{name}' value {value!r} "
                    f"is not one of {allowed}"
                )


def load_decision_table(path: str | Path) -> DecisionTable:
    """Load and validate a decision table from a ``*.dmn.yaml`` file.

    Args:
        path: Path to the decision-table YAML file.

    Returns:
        The validated :class:`DecisionTable`.

    Raises:
        DecisionTableError: If the file cannot be read, is not a YAML mapping,
            or has a structural/cross-field problem.
        FeelValidationError: If any FEEL expression is malformed or out-of-subset.
        pydantic.ValidationError: On a field-shape problem (e.g. missing
            ``version``).
    """
    file_path = Path(path)
    try:
        raw = file_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise DecisionTableError(
            f"could not read decision table '{file_path}': {exc}"
        ) from exc
    data = yaml.safe_load(raw)
    if not isinstance(data, dict):
        raise DecisionTableError(
            f"decision table '{file_path}' must be a YAML mapping, got "
            f"{type(data).__name__}"
        )
    return DecisionTable.model_validate(data)
