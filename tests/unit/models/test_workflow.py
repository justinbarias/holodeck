"""Tests for the workflow DAG models (036, T2).

Covers node discrimination by shape, the closed-schema invariant that keeps AI
output off the spine (FR-005/SC-008), and DAG validation — unique ids, resolved
input references, and cycle rejection at load time (FR-003).
"""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from holodeck.models.workflow import (
    EdgeNode,
    GateRef,
    HumanNode,
    InputDataRef,
    PolicyNode,
    Workflow,
)


def _edge(node_id: str) -> dict:
    """A minimal valid edge-node dict."""
    return {
        "id": node_id,
        "edge": {"agent": f"agents/{node_id}/agent.yaml"},
        "gate": {"schema": f"schemas/{node_id}.json"},
    }


def _policy(node_id: str, inputs: list[str]) -> dict:
    """A minimal valid policy-node dict.

    The hit policy lives on the referenced table, not the node, so it is not
    set here.
    """
    return {
        "id": node_id,
        "decision": f"tables/{node_id}.dmn.yaml",
        "inputs": inputs,
    }


def _workflow(nodes: list[dict]) -> dict:
    """A workflow dict wrapping the given nodes."""
    return {"name": "wf", "version": "2026-06-01.1", "nodes": nodes}


@pytest.mark.unit
class TestNodeDiscrimination:
    """Nodes are discriminated by shape, not an explicit type tag."""

    def test_edge_node_parses(self) -> None:
        wf = Workflow.model_validate(_workflow([_edge("income")]))
        assert isinstance(wf.nodes[0], EdgeNode)
        assert wf.nodes[0].edge.agent == "agents/income/agent.yaml"
        assert wf.nodes[0].gate.schema_path == "schemas/income.json"

    def test_policy_node_parses(self) -> None:
        wf = Workflow.model_validate(
            _workflow([_edge("income"), _policy("afford", ["income"])])
        )
        assert isinstance(wf.nodes[1], PolicyNode)
        assert wf.nodes[1].decision == "tables/afford.dmn.yaml"

    def test_human_node_parses(self) -> None:
        human = {
            "id": "final",
            "decision": "tables/final.dmn.yaml",
            "inputs": ["afford"],
            "requires_human": True,
            "decided_by": "Hardship Officer",
            "draft": {"agent": "agents/reasons/agent.yaml"},
            "ai_may_draft": ["reasons"],
        }
        wf = Workflow.model_validate(
            _workflow([_edge("income"), _policy("afford", ["income"]), human])
        )
        node = wf.nodes[2]
        assert isinstance(node, HumanNode)
        assert node.requires_human is True
        assert node.decided_by == "Hardship Officer"
        assert node.ai_may_draft == ["reasons"]

    def test_requires_human_false_is_policy_node(self) -> None:
        # A node without `edge` and without a truthy `requires_human` is a
        # policy node, even if the key is present-but-false.
        node = _policy("afford", ["income"])
        node["requires_human"] = False
        # `requires_human` is not a PolicyNode field -> extra=forbid rejects it,
        # confirming the node discriminated as policy (not human).
        with pytest.raises(ValidationError) as exc:
            Workflow.model_validate(_workflow([_edge("income"), node]))
        assert "requires_human" in str(exc.value)


@pytest.mark.unit
class TestClosedSchemaInvariant:
    """FR-005/SC-008: AI output can never become a verdict, by construction."""

    def test_edge_node_cannot_declare_decision(self) -> None:
        node = _edge("income")
        node["decision"] = "tables/sneaky.dmn.yaml"
        with pytest.raises(ValidationError) as exc:
            Workflow.model_validate(_workflow([node]))
        assert "decision" in str(exc.value)

    def test_edge_node_cannot_declare_hit_policy(self) -> None:
        node = _edge("income")
        node["hit_policy"] = "UNIQUE"
        with pytest.raises(ValidationError) as exc:
            Workflow.model_validate(_workflow([node]))
        assert "hit_policy" in str(exc.value)

    def test_edge_node_cannot_declare_inputs(self) -> None:
        node = _edge("income")
        node["inputs"] = ["other"]
        with pytest.raises(ValidationError) as exc:
            Workflow.model_validate(_workflow([node]))
        assert "inputs" in str(exc.value)

    def test_policy_node_cannot_declare_an_agent_edge(self) -> None:
        # `edge` would discriminate the node AS an edge; the leftover decision
        # field then fails. Either way, no node carries both an agent and a
        # table.
        node = _policy("afford", ["income"])
        node["edge"] = {"agent": "agents/x/agent.yaml"}
        with pytest.raises(ValidationError):
            Workflow.model_validate(_workflow([_edge("income"), node]))


@pytest.mark.unit
class TestDagValidation:
    """FR-003: unique ids, resolved refs, acyclic — all at load time."""

    def test_valid_multilevel_dag(self) -> None:
        nodes = [
            _edge("income"),
            _edge("residency"),
            _policy("afford", ["income", "residency"]),
            _policy("risk", ["income"]),
            _policy("final", ["afford", "risk"]),
        ]
        wf = Workflow.model_validate(_workflow(nodes))
        assert len(wf.nodes) == 5

    def test_duplicate_ids_rejected(self) -> None:
        nodes = [_edge("income"), _edge("income")]
        with pytest.raises(ValidationError) as exc:
            Workflow.model_validate(_workflow(nodes))
        assert "Duplicate node id" in str(exc.value)
        assert "income" in str(exc.value)

    def test_unresolved_input_reference_rejected(self) -> None:
        nodes = [_edge("income"), _policy("afford", ["income", "ghost"])]
        with pytest.raises(ValidationError) as exc:
            Workflow.model_validate(_workflow(nodes))
        assert "unknown input 'ghost'" in str(exc.value)
        assert "afford" in str(exc.value)

    def test_cycle_rejected(self) -> None:
        # afford <-> risk reference each other.
        nodes = [
            _policy("afford", ["risk"]),
            _policy("risk", ["afford"]),
        ]
        with pytest.raises(ValidationError) as exc:
            Workflow.model_validate(_workflow(nodes))
        assert "cycle" in str(exc.value).lower()

    def test_self_reference_is_a_cycle(self) -> None:
        nodes = [_policy("afford", ["afford"])]
        with pytest.raises(ValidationError) as exc:
            Workflow.model_validate(_workflow(nodes))
        assert "cycle" in str(exc.value).lower()


@pytest.mark.unit
class TestHumanNodeDraftPairing:
    """FR-015a: a drafting agent and its advisory fields go together."""

    def _human(self, **overrides: object) -> dict:
        node = {
            "id": "final",
            "decision": "tables/final.dmn.yaml",
            "inputs": ["afford"],
            "requires_human": True,
        }
        node.update(overrides)
        return node

    def test_draft_without_ai_may_draft_rejected(self) -> None:
        node = self._human(draft={"agent": "agents/reasons/agent.yaml"})
        with pytest.raises(ValidationError) as exc:
            Workflow.model_validate(
                _workflow([_policy("afford", ["x"]), _edge("x"), node])
            )
        assert "declared together" in str(exc.value)

    def test_ai_may_draft_without_draft_rejected(self) -> None:
        node = self._human(ai_may_draft=["reasons"])
        with pytest.raises(ValidationError) as exc:
            Workflow.model_validate(
                _workflow([_edge("x"), _policy("afford", ["x"]), node])
            )
        assert "declared together" in str(exc.value)

    def test_empty_ai_may_draft_rejected(self) -> None:
        node = self._human(
            draft={"agent": "agents/reasons/agent.yaml"}, ai_may_draft=[]
        )
        with pytest.raises(ValidationError) as exc:
            Workflow.model_validate(
                _workflow([_edge("x"), _policy("afford", ["x"]), node])
            )
        assert "non-empty" in str(exc.value)

    def test_human_node_without_draft_is_valid(self) -> None:
        node = self._human()
        wf = Workflow.model_validate(
            _workflow([_edge("x"), _policy("afford", ["x"]), node])
        )
        assert isinstance(wf.nodes[2], HumanNode)
        assert wf.nodes[2].draft is None


@pytest.mark.unit
class TestSourceAnnotation:
    """The optional knowledgeSource-lite `source:` field (dmn-yaml-mapping)."""

    def test_source_on_workflow_and_nodes(self) -> None:
        nodes = [_edge("income"), _policy("afford", ["income"])]
        nodes[0]["source"] = "Income Policy §1"
        nodes[1]["source"] = "Hardship Policy v4.2 §72"
        wf_dict = _workflow(nodes)
        wf_dict["source"] = "Hardship Policy v4.2"
        wf = Workflow.model_validate(wf_dict)
        assert wf.source == "Hardship Policy v4.2"
        assert wf.nodes[0].source == "Income Policy §1"
        assert wf.nodes[1].source == "Hardship Policy v4.2 §72"


@pytest.mark.unit
class TestInputDataDeclaration:
    """FR-025/FR-026: workflow-level facts of record, referenceable by name."""

    def test_input_data_block_parses(self) -> None:
        wf_dict = _workflow([_policy("zone", ["prior_state"])])
        wf_dict["input_data"] = {"prior_state": {"schema": "schemas/prior.json"}}

        wf = Workflow.model_validate(wf_dict)

        assert wf.input_data is not None
        assert wf.input_data["prior_state"].schema_path == "schemas/prior.json"

    def test_workflow_without_input_data_is_unchanged(self) -> None:
        wf = Workflow.model_validate(_workflow([_edge("income")]))

        assert wf.input_data is None

    def test_input_data_name_resolves_in_node_inputs(self) -> None:
        # FR-026: a fact name is referenceable exactly like a node verdict.
        nodes = [_edge("excuse"), _policy("zone", ["prior_state", "excuse"])]
        wf_dict = _workflow(nodes)
        wf_dict["input_data"] = {"prior_state": {"schema": "schemas/prior.json"}}

        wf = Workflow.model_validate(wf_dict)

        assert wf.nodes[1].inputs == ["prior_state", "excuse"]

    def test_undeclared_fact_name_still_rejected(self) -> None:
        wf_dict = _workflow([_policy("zone", ["prior_state"])])
        wf_dict["input_data"] = {"other": {"schema": "schemas/other.json"}}

        with pytest.raises(ValidationError) as exc:
            Workflow.model_validate(wf_dict)

        assert "unknown input 'prior_state'" in str(exc.value)

    def test_collision_with_node_id_rejected(self) -> None:
        # One name, one meaning: `inputs: [zone]` must not be ambiguous.
        nodes = [_edge("zone"), _policy("final", ["zone"])]
        wf_dict = _workflow(nodes)
        wf_dict["input_data"] = {"zone": {"schema": "schemas/zone.json"}}

        with pytest.raises(ValidationError) as exc:
            Workflow.model_validate(wf_dict)

        assert "collide" in str(exc.value)
        assert "zone" in str(exc.value)

    def test_fact_names_are_not_scheduled_as_nodes(self) -> None:
        # input_data names are facts already in hand, not work to execute, so
        # they must not appear as graph nodes. A fact sharing a name with a
        # *node's* dependency set would otherwise widen the topological order.
        wf_dict = _workflow([_policy("zone", ["prior_state"])])
        wf_dict["input_data"] = {"prior_state": {"schema": "schemas/prior.json"}}

        wf = Workflow.model_validate(wf_dict)

        assert [node.id for node in wf.nodes] == ["zone"]

    def test_bare_model_dump_emits_schema_key(self) -> None:
        ref = InputDataRef.model_validate({"schema": "schemas/prior.json"})

        assert ref.model_dump() == {"schema": "schemas/prior.json"}


@pytest.mark.unit
class TestInputDataIsNeverAgentProduced:
    """FR-027/SC-010: an agent-produced fact of record is unrepresentable."""

    def test_input_data_entry_cannot_declare_an_agent(self) -> None:
        wf_dict = _workflow([_policy("zone", ["prior_state"])])
        wf_dict["input_data"] = {
            "prior_state": {
                "schema": "schemas/prior.json",
                "agent": "agents/sneaky/agent.yaml",
            }
        }

        with pytest.raises(ValidationError) as exc:
            Workflow.model_validate(wf_dict)

        assert "agent" in str(exc.value)

    def test_input_data_entry_cannot_declare_an_edge(self) -> None:
        wf_dict = _workflow([_policy("zone", ["prior_state"])])
        wf_dict["input_data"] = {
            "prior_state": {
                "schema": "schemas/prior.json",
                "edge": {"agent": "agents/sneaky/agent.yaml"},
            }
        }

        with pytest.raises(ValidationError) as exc:
            Workflow.model_validate(wf_dict)

        assert "edge" in str(exc.value)

    def test_published_schema_closes_the_input_data_entry(self) -> None:
        # SC-010 against the published artifact editors validate against, not
        # just the in-process model.
        schema_path = (
            Path(__file__).resolve().parents[3] / "schemas" / "workflow.schema.json"
        )
        schema = json.loads(schema_path.read_text())

        entry = schema["$defs"]["InputDataRef"]

        assert entry["additionalProperties"] is False
        assert set(entry["properties"]) == {"schema"}
        assert schema["properties"]["input_data"]["anyOf"][0][
            "additionalProperties"
        ] == {"$ref": "#/$defs/InputDataRef"}


class TestGateRefSerialization:
    """A bare model_dump() must emit the published key, not the attribute name.

    ``schema`` shadows ``BaseModel.schema()``, so the attribute is
    ``schema_path``. Without serialize_by_alias a plain dump emits
    ``schema_path``, producing YAML/JSON that does not match
    workflow.schema.json. Re-validation is unaffected (populate_by_name accepts
    both) — the risk is emitted artifacts, from T10's run record onward.
    """

    def test_bare_model_dump_emits_schema_key(self) -> None:
        """model_dump() without by_alias still emits 'schema'."""
        gate = GateRef.model_validate({"schema": "schemas/income.json"})

        assert gate.model_dump() == {"schema": "schemas/income.json"}

    def test_dump_round_trips_through_validation(self) -> None:
        """A bare dump re-validates and preserves the path."""
        gate = GateRef.model_validate({"schema": "schemas/income.json"})

        assert GateRef.model_validate(gate.model_dump()).schema_path == (
            "schemas/income.json"
        )
