"""``HoloDeckPlugin``: sugar over the activity factory (spec 040, T6).

Decision 14 makes the developer API two layers. The factory
(:func:`~holodeck.temporal.activity.agent_activity`) is the testable core; this
plugin is the one-liner that wires a list of edge nodes into a Temporal client
and worker:

* it sets ``temporalio.contrib.pydantic.pydantic_data_converter`` on the client
  so the typed payload models cross the wire (decision 15), and
* it registers one activity per node, built by the factory — there is no
  parallel implementation here.

A client plugin propagates to every worker created from that client, so passing
this to ``Client.connect(..., plugins=[plugin])`` is enough; passing it to the
``Worker`` as well is unnecessary::

    plugin = HoloDeckPlugin(nodes=config.nodes, base_dir=config_dir)
    client = await Client.connect(address, plugins=[plugin])
    worker = Worker(client, task_queue="agents")

Because the activities are built in the constructor, every authoring fault the
factory settles at bind time — an agent path escaping ``base_dir``, an
unusable gate, an agent that could never produce structured output — surfaces
here, before a worker starts.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path

from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.plugin import SimplePlugin

from holodeck.lib.errors import ConfigError
from holodeck.models.workflow import EdgeNode
from holodeck.temporal.activity import agent_activity
from holodeck.temporal.models import AgentActivityInput, AgentActivityResult

_ActivityCallable = Callable[[AgentActivityInput], Awaitable[AgentActivityResult]]

DEFAULT_PLUGIN_NAME = "holodeck"


class HoloDeckPlugin(SimplePlugin):
    """Registers HoloDeck agent activities and the Pydantic data converter.

    Attributes:
        agent_activities: The activities built from the nodes, in the order the
            nodes were given. Exposed so a caller that builds its own ``Worker``
            without the plugin can register exactly the same objects.
    """

    def __init__(
        self,
        nodes: Sequence[EdgeNode],
        base_dir: Path,
        *,
        name: str = DEFAULT_PLUGIN_NAME,
    ) -> None:
        """Build one activity per edge node and configure the converter.

        Args:
            nodes: The edge nodes to expose as activities. Each becomes an
                activity named after its node id (decision 11).
            base_dir: Directory the nodes' ``edge.agent`` and ``gate.schema``
                paths resolve against, and which they may not escape.
            name: Plugin name reported to Temporal. Rarely overridden; useful
                when two HoloDeck plugins are composed onto one client.

        Raises:
            ConfigError: If two nodes share an id — activity names must be
                unique, and the second registration would silently shadow the
                first — or if any node fails the factory's bind-time checks.
            GateSchemaError: If a node's gate schema is unusable.
            FileNotFoundError: If a node's ``agent.yaml`` does not exist.
        """
        seen: set[str] = set()
        for node in nodes:
            if node.id in seen:
                raise ConfigError(
                    "nodes",
                    f"duplicate node id '{node.id}': an activity name must be "
                    f"unique, and the second registration would shadow the first",
                )
            seen.add(node.id)

        self.agent_activities: list[_ActivityCallable] = [
            agent_activity(node, base_dir) for node in nodes
        ]
        super().__init__(
            name,
            data_converter=pydantic_data_converter,
            activities=list(self.agent_activities),
        )


__all__ = ["DEFAULT_PLUGIN_NAME", "HoloDeckPlugin"]
