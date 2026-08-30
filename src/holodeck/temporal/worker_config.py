"""``worker.yaml``: what a HoloDeck worker registers (spec 040, T7).

The D4 configuration surface (decision 8): a connection block and a list of
inline edge nodes. Nothing else belongs here — decision 9 makes ``nodes:``
*registration only*, with zero control flow, and decision 10 puts every
execution knob (``start_to_close_timeout``, ``RetryPolicy``, …) on the
caller's ``execute_activity`` command through
:class:`~holodeck.temporal.models.ActivityParameters`. A timeout in
``worker.yaml`` would be a lie: the server would never read it.

::

    temporal:
      address: localhost:7233
      namespace: default
      task_queue: hardship
      tls: false
    nodes:
      - id: evidence
        edge: {agent: agents/evidence.yaml}
        gate: {schema: gates/evidence.schema.json}

Both models close over their fields (``extra="forbid"``): a misspelled
``task_que`` must fail at load, not silently connect a worker to the wrong
queue.

:func:`load_worker_config` applies the three ``TEMPORAL_*`` overrides at the
same precedence the rest of HoloDeck uses (shell env wins over the file), then
resolves every node's ``edge.agent`` through
:func:`~holodeck.lib.workflow.edge.resolve_agent_path` so a path escaping the
config directory is refused at load rather than at worker start. Gate schemas
are deliberately *not* read here — that stays bind-time in the activity
factory, which is the only place that can report a gate as unusable.

This module is pure configuration and imports no ``temporalio`` client or
worker machinery, so a CLI can load and report a config without the extra's
runtime being live.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)
from pydantic import ValidationError as PydanticValidationError

from holodeck.config.validator import flatten_pydantic_errors
from holodeck.lib.errors import ConfigError
from holodeck.lib.errors import FileNotFoundError as HoloDeckFileNotFoundError
from holodeck.lib.workflow.edge import resolve_agent_path
from holodeck.models.workflow import EdgeNode

DEFAULT_ADDRESS = "localhost:7233"
DEFAULT_NAMESPACE = "default"

# Only these three. TLS is intentionally file-only in v1: an env var that can
# silently downgrade a connection to plaintext is a worse default than editing
# the file.
ENV_ADDRESS = "TEMPORAL_ADDRESS"
ENV_NAMESPACE = "TEMPORAL_NAMESPACE"
ENV_TASK_QUEUE = "TEMPORAL_TASK_QUEUE"


class TemporalConnection(BaseModel):
    """How the worker reaches the Temporal service.

    Attributes:
        address: ``host:port`` of the Temporal frontend.
        namespace: Temporal namespace the worker polls in.
        task_queue: Task queue the worker polls. Required — a default would
            silently point every unconfigured worker at the same queue.
        tls: Whether to connect with TLS. A simple on/off in v1; client
            certificate material is not modelled yet.
    """

    model_config = ConfigDict(extra="forbid")

    address: str = Field(default=DEFAULT_ADDRESS)
    namespace: str = Field(default=DEFAULT_NAMESPACE)
    task_queue: str
    tls: bool = Field(default=False)

    @field_validator("address", "namespace", "task_queue")
    @classmethod
    def _non_blank(cls, value: str, info: ValidationInfo) -> str:
        """Refuse a blank connection string.

        A blank value can only mean a broken secret, template, or environment
        override; every one of these fields silently misroutes the worker if
        it defaults or falls back, so all three fail closed.

        Args:
            value: The configured value.
            info: Field context (names the offending field).

        Returns:
            The value unchanged.

        Raises:
            ValueError: If the value is empty or only whitespace. Pydantic
                wraps this into the ``ValidationError`` the loader converts to
                a :class:`~holodeck.lib.errors.ConfigError`.
        """
        if not value.strip():
            raise ValueError(f"{info.field_name} must not be blank")
        return value


class WorkerConfig(BaseModel):
    """A parsed ``worker.yaml``.

    Attributes:
        temporal: The connection block.
        nodes: The edge nodes this worker registers as activities, in file
            order. At least one — a worker with no activities has nothing to
            do.
        base_dir: Directory of the ``worker.yaml`` the config came from. Node
            paths resolve against it and may not escape it.
    """

    model_config = ConfigDict(extra="forbid")

    temporal: TemporalConnection
    nodes: list[EdgeNode] = Field(min_length=1)
    # A real field rather than a post-parse attribute so mypy sees it and
    # ``model_validator`` can use it, but ``exclude=True`` keeps it out of
    # model_dump(): base_dir is where the file was found, not part of the
    # authored document, and round-tripping a dump into YAML must not emit it.
    # It has no default, so constructing a WorkerConfig without it is a
    # validation error rather than a silently wrong relative root.
    base_dir: Path = Field(exclude=True)

    @model_validator(mode="after")
    def _unique_node_ids(self) -> WorkerConfig:
        """Refuse duplicate node ids.

        Returns:
            The validated config.

        Raises:
            ConfigError: If two nodes share an id. Raised directly rather than
                as a ``ValueError`` so the message reaches the caller intact,
                matching the wording ``HoloDeckPlugin`` uses for the same
                fault.
        """
        seen: set[str] = set()
        for node in self.nodes:
            if node.id in seen:
                raise ConfigError(
                    "nodes",
                    f"duplicate node id '{node.id}': an activity name must be "
                    f"unique, and the second registration would shadow the first",
                )
            seen.add(node.id)
        return self


def _apply_env_overrides(temporal: dict[str, Any]) -> None:
    """Overlay the ``TEMPORAL_*`` environment variables onto the block.

    Applied before validation so an env-supplied ``task_queue`` satisfies the
    required field, and so the value is subject to the same checks a file
    value is. Shell environment wins over the file, matching ConfigLoader's
    precedence. A present-but-empty variable overrides too — and then fails
    the non-blank validators. A broken secret or template must fail loud at
    load, not silently fall back to whatever the file says (fail closed).

    Args:
        temporal: The mutable ``temporal:`` mapping from the parsed YAML.
    """
    for key, env_name in (
        ("address", ENV_ADDRESS),
        ("namespace", ENV_NAMESPACE),
        ("task_queue", ENV_TASK_QUEUE),
    ):
        value = os.environ.get(env_name)
        if value is not None:
            temporal[key] = value


def load_worker_config(path: str | Path) -> WorkerConfig:
    """Load, validate, and confine a ``worker.yaml``.

    Args:
        path: Path to the ``worker.yaml`` file.

    Returns:
        The validated configuration, with ``base_dir`` set to the file's
        directory.

    Raises:
        FileNotFoundError: (``holodeck.lib.errors``) If the file does not
            exist or cannot be read.
        ConfigError: If the YAML is malformed, its root is not a mapping, it
            fails schema validation, or a node's ``edge.agent`` path escapes
            the config directory.
    """
    config_path = Path(path).resolve()
    base_dir = config_path.parent

    try:
        raw = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise HoloDeckFileNotFoundError(
            str(path),
            f"Worker configuration file not found at {path}. "
            f"Please ensure the file exists at this path.",
        ) from exc

    try:
        document = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise ConfigError(
            "yaml_parse",
            f"Failed to parse YAML file {config_path}: {exc}",
        ) from exc

    if not isinstance(document, dict):
        raise ConfigError(
            "worker_config",
            f"Worker configuration in {config_path} must be a YAML mapping, "
            f"got {type(document).__name__}",
        )

    if "base_dir" in document:
        # base_dir is discovered, not authored. It is a model field, so
        # extra="forbid" would not catch it; refuse it here so a stray key
        # cannot redirect path confinement.
        raise ConfigError(
            "base_dir",
            f"Worker configuration in {config_path} may not set 'base_dir'; "
            f"it is the directory of the configuration file itself",
        )

    temporal_block = document.get("temporal")
    if temporal_block is None:
        temporal_block = {}
    if isinstance(temporal_block, dict):
        # Copy so the override never mutates the caller-visible parsed
        # document, and so a re-read of the file is unaffected.
        temporal_block = dict(temporal_block)
        _apply_env_overrides(temporal_block)
    document = {**document, "temporal": temporal_block}

    try:
        config = WorkerConfig(**document, base_dir=base_dir)
    except PydanticValidationError as exc:
        error_text = "\n".join(flatten_pydantic_errors(exc))
        raise ConfigError(
            "worker_config",
            f"Invalid worker configuration in {config_path}:\n{error_text}",
        ) from exc

    # Confinement at load (acceptance criterion 3): an agent path escaping the
    # config directory is an authoring fault the operator should see before a
    # worker starts, not at bind time inside the factory. Gate schemas are not
    # touched here — reading and judging them stays in the factory, whose
    # GateSchemaError channel exists for exactly that.
    for node in config.nodes:
        resolve_agent_path(node, base_dir)

    return config


__all__ = [
    "DEFAULT_ADDRESS",
    "DEFAULT_NAMESPACE",
    "ENV_ADDRESS",
    "ENV_NAMESPACE",
    "ENV_TASK_QUEUE",
    "TemporalConnection",
    "WorkerConfig",
    "load_worker_config",
]
