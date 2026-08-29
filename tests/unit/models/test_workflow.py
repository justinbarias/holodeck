"""Tests for the surviving edge-node models.

Ported from the archived 036 model suite: the ``GateRef`` alias behavior is a
serialization contract — ``serialize_by_alias`` makes a bare ``model_dump()``
emit ``schema``, matching the authored form — and pydantic silently ignores
that config key before 2.11, so these tests are what enforce it.
"""

import pytest

from holodeck.models.workflow import EdgeNode, EdgeRef, GateRef

pytestmark = pytest.mark.unit


class TestGateRefAlias:
    """The schema/schema_path alias must round-trip without by_alias=True."""

    def test_bare_model_dump_emits_schema_key(self) -> None:
        gate = GateRef.model_validate({"schema": "gates/out.schema.json"})

        dumped = gate.model_dump()

        assert dumped == {"schema": "gates/out.schema.json"}
        assert "schema_path" not in dumped

    def test_dump_round_trips_through_validation(self) -> None:
        gate = GateRef.model_validate({"schema": "gates/out.schema.json"})

        revalidated = GateRef.model_validate(gate.model_dump())

        assert revalidated.schema_path == gate.schema_path


class TestEdgeNode:
    def test_constructs_from_authored_shape(self) -> None:
        node = EdgeNode.model_validate(
            {
                "id": "classify",
                "edge": {"agent": "agents/classifier.yaml"},
                "gate": {"schema": "gates/classification.schema.json"},
            }
        )

        assert node.edge == EdgeRef(agent="agents/classifier.yaml")
        assert node.gate.schema_path == "gates/classification.schema.json"

    def test_extra_fields_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="decision"):
            EdgeNode.model_validate(
                {
                    "id": "classify",
                    "edge": {"agent": "agents/classifier.yaml"},
                    "gate": {"schema": "gates/classification.schema.json"},
                    "decision": "tables/points.dmn.yaml",
                }
            )
