"""``HoloDeckPlugin`` wiring (spec 040, T6).

The plugin is sugar over the T3 factory (decision 14): the tests check that it
sets the Pydantic data converter (decision 15), registers exactly one activity
per node, and produces definitions indistinguishable from calling
:func:`~holodeck.temporal.activity.agent_activity` by hand.

No server and no model: the plugin's ``configure_client``/``configure_worker``
hooks are called directly with plain config dicts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from temporalio import activity
from temporalio.contrib.pydantic import pydantic_data_converter

from holodeck.lib.backends.claude_backend import ClaudeBackend
from holodeck.lib.backends.openai_agents_backend import OpenAIAgentsBackend
from holodeck.lib.errors import ConfigError
from holodeck.models.workflow import EdgeNode
from holodeck.temporal.activity import agent_activity
from holodeck.temporal.plugin import DEFAULT_PLUGIN_NAME, HoloDeckPlugin

pytestmark = pytest.mark.unit

AGENT_YAML = """\
name: hardship-evidence
description: Edge agent under test
model:
  provider: anthropic
  name: claude-sonnet-4-20250514
instructions:
  inline: "Extract the applicant's income evidence."
response_format:
  type: object
  properties:
    net_income:
      type: number
"""

GATE_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "properties": {"net_income": {"type": "number"}},
    "required": ["net_income"],
}


@pytest.fixture(autouse=True)
def forbid_real_backends(monkeypatch: pytest.MonkeyPatch) -> None:
    """Blow up if any concrete backend is constructed (proves zero LLM calls)."""

    def _boom(*args: Any, **kwargs: Any) -> None:
        raise AssertionError(
            "a real backend was constructed — this test must never reach an LLM"
        )

    monkeypatch.setattr(ClaudeBackend, "__init__", _boom)
    monkeypatch.setattr(OpenAIAgentsBackend, "__init__", _boom)


@pytest.fixture
def base_dir(tmp_path: Path) -> Path:
    """A worker base directory holding two edge agents and their gates."""
    (tmp_path / "agents").mkdir()
    (tmp_path / "gates").mkdir()
    for stem in ("evidence", "letter"):
        (tmp_path / "agents" / f"{stem}.yaml").write_text(AGENT_YAML, encoding="utf-8")
        (tmp_path / "gates" / f"{stem}.schema.json").write_text(
            json.dumps(GATE_SCHEMA), encoding="utf-8"
        )
    return tmp_path


def _node(node_id: str, stem: str | None = None) -> EdgeNode:
    """Build an edge node pointing at one of the fixture agents."""
    stem = stem or node_id
    return EdgeNode(
        id=node_id,
        edge={"agent": f"agents/{stem}.yaml"},  # type: ignore[arg-type]
        gate={"schema": f"gates/{stem}.schema.json"},  # type: ignore[arg-type]
    )


@pytest.fixture
def nodes() -> list[EdgeNode]:
    """The two nodes the plugin registers."""
    return [_node("evidence"), _node("letter")]


class TestClientConfiguration:
    """What the plugin does to the client config."""

    def test_sets_the_pydantic_data_converter(
        self, base_dir: Path, nodes: list[EdgeNode]
    ):
        """Typed payload models require the Pydantic converter (decision 15)."""
        # Arrange
        plugin = HoloDeckPlugin(nodes, base_dir)

        # Act
        config = plugin.configure_client({})  # type: ignore[typeddict-item]

        # Assert
        assert config["data_converter"] is pydantic_data_converter

    def test_default_plugin_name(self, base_dir: Path, nodes: list[EdgeNode]):
        """The plugin reports a stable name to Temporal."""
        # Act
        plugin = HoloDeckPlugin(nodes, base_dir)

        # Assert
        assert plugin.name() == DEFAULT_PLUGIN_NAME


class TestWorkerConfiguration:
    """What the plugin does to the worker config."""

    def test_registers_one_activity_per_node(
        self, base_dir: Path, nodes: list[EdgeNode]
    ):
        """One node, one activity, named after the node id."""
        # Arrange
        plugin = HoloDeckPlugin(nodes, base_dir)

        # Act
        config = plugin.configure_worker({})  # type: ignore[typeddict-item]

        # Assert
        names = [
            activity._Definition.must_from_callable(fn).name
            for fn in config["activities"]
        ]
        assert names == ["evidence", "letter"]

    def test_appends_to_activities_the_caller_already_registered(
        self, base_dir: Path, nodes: list[EdgeNode]
    ):
        """The plugin adds to a worker config, it does not replace it."""

        # Arrange
        @activity.defn(name="pre-existing")
        async def other() -> None:
            return None

        plugin = HoloDeckPlugin(nodes, base_dir)

        # Act
        config = plugin.configure_worker({"activities": [other]})  # type: ignore[typeddict-item]

        # Assert
        names = [
            activity._Definition.must_from_callable(fn).name
            for fn in config["activities"]
        ]
        assert names == ["pre-existing", "evidence", "letter"]

    def test_exposed_activities_match_the_registered_ones(
        self, base_dir: Path, nodes: list[EdgeNode]
    ):
        """``agent_activities`` is what a hand-built Worker would register."""
        # Arrange
        plugin = HoloDeckPlugin(nodes, base_dir)

        # Act
        config = plugin.configure_worker({})  # type: ignore[typeddict-item]

        # Assert
        assert list(config["activities"]) == plugin.agent_activities


class TestParityWithTheFactory:
    """The plugin must not be a second implementation."""

    def test_definitions_match_manual_factory_wiring(
        self, base_dir: Path, nodes: list[EdgeNode]
    ):
        """Plugin-built and hand-built activity definitions are identical."""
        # Arrange
        manual = [agent_activity(node, base_dir) for node in nodes]
        plugin = HoloDeckPlugin(nodes, base_dir)

        # Act
        built = plugin.agent_activities

        # Assert
        for from_plugin, from_factory in zip(built, manual, strict=True):
            plugin_defn = activity._Definition.must_from_callable(from_plugin)
            factory_defn = activity._Definition.must_from_callable(from_factory)
            assert plugin_defn.name == factory_defn.name
            assert plugin_defn.is_async == factory_defn.is_async
            assert plugin_defn.arg_types == factory_defn.arg_types
            assert plugin_defn.ret_type is factory_defn.ret_type


class TestConstructionFaults:
    """Authoring faults surface at construction, before a worker starts."""

    def test_duplicate_node_ids_are_refused(self, base_dir: Path):
        """Two activities cannot share a name; the second would shadow the first."""
        # Arrange
        duplicated = [_node("evidence"), _node("evidence", stem="letter")]

        # Act / Assert
        with pytest.raises(ConfigError) as excinfo:
            HoloDeckPlugin(duplicated, base_dir)
        assert "duplicate node id" in str(excinfo.value)

    def test_factory_faults_surface_at_construction(self, base_dir: Path):
        """A node whose agent escapes the base dir is refused here, not later."""
        # Arrange
        escaping = EdgeNode(
            id="escape",
            edge={"agent": "../outside.yaml"},  # type: ignore[arg-type]
            gate={"schema": "gates/evidence.schema.json"},  # type: ignore[arg-type]
        )

        # Act / Assert
        with pytest.raises(ConfigError):
            HoloDeckPlugin([escaping], base_dir)

    def test_empty_node_list_registers_nothing(self, base_dir: Path):
        """A plugin with no nodes still sets the converter and adds no activity."""
        # Arrange
        plugin = HoloDeckPlugin([], base_dir)

        # Act
        worker_config = plugin.configure_worker({})  # type: ignore[typeddict-item]
        client_config = plugin.configure_client({})  # type: ignore[typeddict-item]

        # Assert
        assert plugin.agent_activities == []
        assert not worker_config.get("activities")
        assert client_config["data_converter"] is pydantic_data_converter
