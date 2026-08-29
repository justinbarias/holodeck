"""Edge-node models: the schema-gated agent boundary.

What survives of the 036 deterministic spine (see ``specs/036-deterministic-spine/``,
archived) after the pivot to Temporal (``SPEC.md``): an :class:`EdgeNode` names
an agent and the JSON Schema gate its structured output must cross. The DAG,
policy/human nodes, and the ``workflow.yaml`` artifact were removed with the
overlay engine; the gate executor in ``holodeck.lib.workflow.edge`` consumes
these models and is reused by the Temporal activity wrapper.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class EdgeRef(BaseModel):
    """The agent reference on an edge node (``edge: {agent: ...}``)."""

    model_config = ConfigDict(extra="forbid")

    # Confinement lives in holodeck.lib.workflow.edge.resolve_agent_path, the
    # same control load_gate_schema applies to gate.schema. Consumers resolve
    # this reference through that function, never by joining the path
    # themselves.
    agent: str = Field(
        description="Path to the edge agent's agent.yaml.",
    )


class GateRef(BaseModel):
    """The schema gate on an edge node (``gate: {schema: ...}``).

    The gate's JSON Schema is the typed boundary the agent's structured output
    must satisfy before it may cross.
    """

    # `schema` shadows BaseModel.schema(), hence the schema_path attribute name.
    #
    # serialize_by_alias makes a plain model_dump() emit `schema` rather than
    # `schema_path`. Re-validation works either way (populate_by_name accepts
    # both), so this is not a round-trip bug — the risk is emitted artifacts:
    # a bare dump written to YAML/JSON produced `schema_path:`, which does not
    # match the authored form.
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
            "Path to the gate JSON Schema. The edge agent's structured_output "
            "is validated against it at the gate."
        ),
    )


class EdgeNode(BaseModel):
    """A leaf node that runs an agent and emits a gate-validated input object.

    Edge nodes never emit a verdict; the closed schema (``extra="forbid"``) is
    what keeps unvalidated AI output from crossing the boundary.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(
        description="Unique node id.",
    )
    edge: EdgeRef = Field(description="The agent that produces this node's object.")
    gate: GateRef = Field(description="The schema gate the agent output must cross.")
    source: str | None = Field(
        default=None,
        description="Optional authority annotation (knowledgeSource-lite), e.g. "
        '"Hardship Policy v4.2 §72"; recorded and emitted as a span attribute.',
    )
