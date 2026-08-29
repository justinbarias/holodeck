"""Workflow models for the deterministic spine (036).

Defines the ``workflow.yaml`` artifact: a DAG of determination nodes. Edge
nodes run a HoloDeck agent behind a schema gate and emit a gate-validated
input object (never a verdict). Policy and human nodes evaluate a DMN decision
table over named inputs and emit a verdict; a human node adds a CLI-prompted
decision on top of the table's recommendation.

The node union is discriminated by *shape* (presence of ``edge`` vs.
``decision``/``requires_human``) rather than an explicit ``type`` tag, mirroring
how the YAML reads. Combined with ``extra="forbid"`` on every model, this makes
"AI output as a verdict" unrepresentable by construction (FR-005, SC-008): an
edge node cannot declare ``decision``/``hit_policy``/``inputs``, and there is no
node kind through which agent output reaches a verdict.

Design input: ``specs/036-deterministic-spine/dmn-yaml-mapping.md``.
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from graphlib import CycleError, TopologicalSorter
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Discriminator,
    Field,
    Tag,
    model_validator,
)

from holodeck.lib.errors import FeelValidationError
from holodeck.lib.workflow.feel import _LITERAL_NAMES, referenced_roots
from holodeck.models.decision_table import _RESERVED_FEEL_ROOTS

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

# A node id or input_data key is used as a FEEL root name — a node's verdict
# is dot-pathed by its id (`prior_state.income`), and an input_data key binds
# the same way. Anything outside this shape either fails to parse as a single
# identifier (`prior-state.income` parses as subtraction) or silently shadows
# a FEEL literal — both break downstream, after every upstream edge agent has
# already run, with a message that does not point back at the offending name.
#
# Kept as a string as well as a compiled pattern because it is published: it is
# emitted verbatim into `schemas/workflow.schema.json` (see the node models and
# `Workflow.input_data`), so an author's editor rejects the same names this
# validator does rather than showing green on a file the runner refuses.
_FEEL_SAFE_IDENTIFIER_PATTERN = r"^[A-Za-z_][A-Za-z0-9_]*$"
_FEEL_SAFE_IDENTIFIER: re.Pattern[str] = re.compile(_FEEL_SAFE_IDENTIFIER_PATTERN)

# Published alongside each node id, and as `propertyNames` on `input_data`, so
# `schemas/workflow.schema.json` carries the same shape constraint the model
# enforces. It is `json_schema_extra` rather than `Field(pattern=...)` on
# purpose: pydantic's own pattern failure reads "String should match pattern
# '^[A-Za-z_]...'", whereas `_validate_feel_safe_name` tells the author *why*
# the name cannot work. Both read the one constant above, so they cannot drift.
#
# The keyword restriction (`_binds_as_a_feel_variable`) is deliberately not
# published: it is a property of the embedded FEEL grammar, not a regular
# language, so JSON Schema cannot express it. An editor therefore accepts
# `id: date` and the model rejects it — the shape is published, the grammar is
# not.
_FEEL_SAFE_NAME_JSON_SCHEMA: dict[str, Any] = {"pattern": _FEEL_SAFE_IDENTIFIER_PATTERN}


def _binds_as_a_feel_variable(name: str) -> bool:
    """Report whether FEEL parses ``name`` as a reference to ``name``.

    The identifier pattern above matches the grammar's ``CNAME`` terminal, but
    the grammar also has keyword rules that win over it — ``date``, ``not``,
    ``list``, ``today`` and some twenty others parse as keywords, so a node id
    of ``date`` yields ``malformed FEEL expression`` against the *table* that
    reads it. The check is a round trip through the real parser rather than a
    hand-copied keyword list: the list belongs to the embedded grammar, and a
    copy of it here would silently drift the next time that dependency moves.

    Args:
        name: A candidate node id or input_data key.

    Returns:
        ``True`` if parsing ``name`` yields exactly one referenced root and it
        is ``name`` itself.
    """
    try:
        return referenced_roots(name, locator="name") == {name}
    except FeelValidationError:
        return False


def _validate_feel_safe_name(name: str, kind: str) -> None:
    """Reject a node id / input_data key FEEL cannot bind unambiguously.

    Args:
        name: The node id or input_data key to check.
        kind: Human-readable label for the error message (e.g. "node id").

    Raises:
        ValueError: If the name is not a FEEL-safe identifier, collides with a
            FEEL literal (``true``/``false``/``null``) or a name FEEL reserves
            as non-executable metadata (e.g. ``provenance``), or is a FEEL
            keyword the grammar will not bind as a variable.
    """
    if not _FEEL_SAFE_IDENTIFIER.match(name):
        raise ValueError(
            f"{kind} {name!r} is not a FEEL-safe identifier: it must start "
            "with a letter or underscore and contain only letters, digits, "
            "and underscores, or it will not resolve when referenced from a "
            "decision table"
        )
    if name in _LITERAL_NAMES:
        raise ValueError(
            f"{kind} {name!r} collides with the FEEL literal {name!r}; a "
            "binding under this name would be invisible to every expression"
        )
    if name in _RESERVED_FEEL_ROOTS:
        raise ValueError(
            f"{kind} {name!r} collides with a name FEEL reserves as "
            "non-executable metadata and cannot be referenced from a "
            "decision table"
        )
    if not _binds_as_a_feel_variable(name):
        raise ValueError(
            f"{kind} {name!r} is a FEEL keyword, not a variable name: a "
            f"decision table reading {name}.field does not parse, and the "
            f"failure it reports names the table rather than this {kind}"
        )


class EdgeRef(BaseModel):
    """The agent reference on an edge node (``edge: {agent: ...}``)."""

    model_config = ConfigDict(extra="forbid")

    agent: str = Field(
        description="Path to the edge agent's agent.yaml, relative to workflow.yaml.",
    )


class GateRef(BaseModel):
    """The schema gate on an edge node (``gate: {schema: ...}``).

    The gate's JSON Schema is the typed boundary the agent's structured output
    must satisfy before it may cross into the spine (FR-007).
    """

    # `schema` shadows BaseModel.schema(), hence the schema_path attribute name.
    #
    # serialize_by_alias makes a plain model_dump() emit `schema` rather than
    # `schema_path`. Re-validation works either way (populate_by_name accepts
    # both), so this is not a round-trip bug — the risk is emitted artifacts:
    # a bare dump written to YAML/JSON produced `schema_path:`, which does not
    # match the published workflow.schema.json. Matters from T10 onward, where
    # the run record serialises gate references.
    #
    # serialization_alias alone does NOT fix this — it still requires
    # model_dump(by_alias=True). serialize_by_alias is the config that changes
    # the default (pydantic >= 2.11; this project is on 2.13).
    model_config = ConfigDict(
        extra="forbid", populate_by_name=True, serialize_by_alias=True
    )

    schema_path: str = Field(
        alias="schema",
        description=(
            "Path to the gate JSON Schema, relative to workflow.yaml. The edge "
            "agent's structured_output is validated against it at the gate."
        ),
    )


class InputDataRef(BaseModel):
    """A declared fact of record entering the run from outside the spine.

    Named for DMN's ``<inputData>``. These are the prior state and case facts a
    run computes its transition from — supplied by the caller and validated
    against their declared JSON Schema before any node executes (FR-025).

    The model is deliberately closed and carries *only* a schema path: there is
    no ``edge``/``agent`` field, so "an agent produced this fact of record" is
    unrepresentable (FR-027, SC-010).
    """

    # Same schema/schema_path aliasing as GateRef, for the same reason — see
    # the comment there. A bare model_dump() must emit `schema` so emitted
    # artifacts (the T10 run record persists the input_data payload, FR-028)
    # match the published workflow.schema.json.
    model_config = ConfigDict(
        extra="forbid", populate_by_name=True, serialize_by_alias=True
    )

    schema_path: str = Field(
        alias="schema",
        description=(
            "Path to this fact's JSON Schema, relative to workflow.yaml. The "
            "supplied value is validated against it before any node runs."
        ),
    )


class DraftRef(BaseModel):
    """The optional drafting agent on a human node (``draft: {agent: ...}``)."""

    model_config = ConfigDict(extra="forbid")

    agent: str = Field(
        description="Path to the drafting agent's agent.yaml, relative to "
        "workflow.yaml. Produces only the advisory fields in ai_may_draft.",
    )


class EdgeNode(BaseModel):
    """A leaf node that runs an agent and emits a gate-validated input object.

    Edge nodes never emit a verdict and never declare a decision table; the
    closed schema (no ``decision``/``hit_policy``/``inputs`` fields) is what
    keeps AI output off the spine (FR-005).
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(
        description="Unique node id within the workflow.",
        json_schema_extra=_FEEL_SAFE_NAME_JSON_SCHEMA,
    )
    edge: EdgeRef = Field(description="The agent that produces this node's object.")
    gate: GateRef = Field(description="The schema gate the agent output must cross.")
    source: str | None = Field(
        default=None,
        description="Optional authority annotation (knowledgeSource-lite), e.g. "
        '"Hardship Policy v4.2 §72"; recorded and emitted as a span attribute.',
    )


class PolicyNode(BaseModel):
    """A pure determination node: a DMN table evaluated over named inputs.

    The hit policy is a property of the referenced decision table (DMN-faithful,
    single source of truth), not of the node — see ``DecisionTable.hit_policy``.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(
        description="Unique node id within the workflow.",
        json_schema_extra=_FEEL_SAFE_NAME_JSON_SCHEMA,
    )
    decision: str = Field(
        description="Path to the decision table (tables/*.dmn.yaml), relative to "
        "workflow.yaml.",
    )
    inputs: list[str] = Field(
        min_length=1,
        description="Ids of the nodes whose verdicts/objects feed this table.",
    )
    source: str | None = Field(
        default=None,
        description="Optional authority annotation (knowledgeSource-lite).",
    )


class HumanNode(BaseModel):
    """The bright line: a DMN table recommends, a named human decides.

    The table is evaluated to produce a *recommendation* (refinements §2); the
    CLI prompts a human to accept or override it. AI output may only draft the
    advisory fields named in ``ai_may_draft`` — never the verdict (FR-015a).
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(
        description="Unique node id within the workflow.",
        json_schema_extra=_FEEL_SAFE_NAME_JSON_SCHEMA,
    )
    decision: str = Field(
        description="Path to the decision table (tables/*.dmn.yaml), relative to "
        "workflow.yaml.",
    )
    inputs: list[str] = Field(
        min_length=1,
        description="Ids of the nodes whose verdicts/objects feed this table.",
    )
    requires_human: Literal[True] = Field(
        description="Marks this node as the human determination (always true).",
    )
    decided_by: str | None = Field(
        default=None,
        description="Declared decision-maker role or name (refinements §6); "
        "confirmed or entered at the CLI prompt and recorded verbatim.",
    )
    draft: DraftRef | None = Field(
        default=None,
        description="Optional drafting agent; produces only ai_may_draft fields.",
    )
    ai_may_draft: list[str] | None = Field(
        default=None,
        description="Advisory fields the drafting agent may produce (POC: "
        "[reasons]). Never populates the verdict.",
    )
    source: str | None = Field(
        default=None,
        description="Optional authority annotation (knowledgeSource-lite).",
    )

    @model_validator(mode="after")
    def _validate_draft_pairing(self) -> Self:
        """``draft`` and ``ai_may_draft`` are declared together and non-empty."""
        if (self.draft is None) != (self.ai_may_draft is None):
            raise ValueError(
                "'draft' and 'ai_may_draft' must be declared together "
                "(a drafting agent needs fields to draft, and vice versa)"
            )
        if self.ai_may_draft is not None and len(self.ai_may_draft) == 0:
            raise ValueError("'ai_may_draft' must be non-empty when declared")
        return self


def _node_kind(v: Any) -> str:
    """Discriminate a node by shape for the ``NodeUnion``.

    Edge nodes carry an ``edge`` block; human nodes set ``requires_human``;
    everything else is a policy node. Discriminating on shape (not an explicit
    tag) keeps the YAML clean and lets ``extra="forbid"`` reject off-pattern
    mixes (e.g. an edge node that also declares ``decision``).

    Args:
        v: Node data as a dict (from YAML) or an already-built node model.

    Returns:
        The discriminator tag: ``"edge"``, ``"human"``, or ``"policy"``.
    """
    if isinstance(v, dict):
        if "edge" in v:
            return "edge"
        if v.get("requires_human"):
            return "human"
        return "policy"
    if isinstance(v, EdgeNode):
        return "edge"
    if isinstance(v, HumanNode):
        return "human"
    return "policy"


# Discriminated union of node kinds, tagged by shape (see ``_node_kind``).
NodeUnion = Annotated[
    Annotated[EdgeNode, Tag("edge")]
    | Annotated[PolicyNode, Tag("policy")]
    | Annotated[HumanNode, Tag("human")],
    Discriminator(_node_kind),
]


class Workflow(BaseModel):
    """The ``workflow.yaml`` artifact: a DAG of determination nodes.

    The graph is the DMN Decision Requirement Diagram in logical form: a node's
    ``inputs`` name the nodes below it. The graph must be acyclic with all
    references resolved — both are enforced at load, before any agent runs
    (FR-003).
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Workflow identifier.")
    version: str = Field(description="Workflow version label.")
    source: str | None = Field(
        default=None,
        description="Optional authority annotation (knowledgeSource-lite).",
    )
    input_data: dict[str, InputDataRef] | None = Field(
        default=None,
        description="Typed facts of record supplied to the run from outside the "
        "spine (prior state, case facts), keyed by name. Referenceable from a "
        "node's inputs: exactly like a node verdict. Never LLM-produced.",
        # propertyNames, not patternProperties: the latter constrains only the
        # keys that already match and says nothing about the rest, so it would
        # publish the shape without rejecting anything.
        json_schema_extra={"propertyNames": _FEEL_SAFE_NAME_JSON_SCHEMA},
    )
    nodes: list[NodeUnion] = Field(
        min_length=1,
        description="The workflow's determination and edge nodes.",
    )

    @model_validator(mode="after")
    def _validate_dag(self) -> Self:
        """Enforce unique ids, resolved input refs, and acyclicity (FR-003)."""
        ids = [node.id for node in self.nodes]
        duplicates = sorted({i for i, count in Counter(ids).items() if count > 1})
        if duplicates:
            raise ValueError(f"Duplicate node id(s): {duplicates}")

        id_set = set(ids)
        fact_names = set(self.input_data or {})

        for node_id in ids:
            _validate_feel_safe_name(node_id, "node id")
        for fact_name in fact_names:
            _validate_feel_safe_name(fact_name, "input_data name")

        # One name, one meaning: a fact of record and a node cannot share a name,
        # or `inputs: [x]` would be ambiguous between a supplied fact and a
        # computed verdict.
        collisions = sorted(fact_names & id_set)
        if collisions:
            raise ValueError(
                f"input_data name(s) collide with node id(s): {collisions}"
            )

        graph: dict[str, set[str]] = {}
        for node in self.nodes:
            inputs: list[str] = getattr(node, "inputs", None) or []
            for ref in inputs:
                if ref not in id_set and ref not in fact_names:
                    raise ValueError(
                        f"Node '{node.id}' references unknown input '{ref}'"
                    )
            # graphlib maps a node to the set of nodes that must precede it.
            #
            # input_data names are deliberately excluded: they are facts already
            # in hand when the run starts, not work to schedule. Feeding them to
            # graphlib would put them in the topological order as things to
            # execute, and every consumer of that order would then have to filter
            # them back out. They cannot participate in a cycle either — nothing
            # produces them — so dropping them costs no validation.
            graph[node.id] = {ref for ref in inputs if ref in id_set}

        try:
            TopologicalSorter(graph).prepare()
        except CycleError as exc:
            cycle = exc.args[1] if len(exc.args) > 1 else exc.args
            raise ValueError(f"Workflow inputs form a cycle: {cycle}") from exc
        return self
