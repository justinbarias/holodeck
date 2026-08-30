"""``worker.yaml`` parsing, env overrides, and path confinement (spec 040, T7).

The config layer is registration only (decisions 8 and 9), so the tests check
three things: the document's shape is closed, the three ``TEMPORAL_*``
overrides win over file values at the documented precedence, and an
``edge.agent`` path that escapes the config directory is refused at load —
before a worker ever starts.

No server and no model: every case writes a real ``worker.yaml`` under
``tmp_path`` and reads it back.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from holodeck.lib.errors import ConfigError
from holodeck.lib.errors import FileNotFoundError as HoloDeckFileNotFoundError
from holodeck.temporal.worker_config import (
    DEFAULT_ADDRESS,
    DEFAULT_NAMESPACE,
    ENV_ADDRESS,
    ENV_NAMESPACE,
    ENV_TASK_QUEUE,
    WorkerConfig,
    load_worker_config,
)

pytestmark = pytest.mark.unit

FULL_YAML = """\
temporal:
  address: temporal.internal:7233
  namespace: hardship-ns
  task_queue: hardship
  tls: true
nodes:
  - id: evidence
    edge:
      agent: agents/evidence.yaml
    gate:
      schema: gates/evidence.schema.json
  - id: letter
    edge:
      agent: agents/letter.yaml
    gate:
      schema: gates/letter.schema.json
"""

MINIMAL_YAML = """\
temporal:
  task_queue: hardship
nodes:
  - id: evidence
    edge:
      agent: agents/evidence.yaml
    gate:
      schema: gates/evidence.schema.json
"""


def write_config(tmp_path: Path, body: str, name: str = "worker.yaml") -> Path:
    """Write a worker configuration file and return its path.

    Args:
        tmp_path: The test's temporary directory.
        body: YAML document body.
        name: File name to write.

    Returns:
        Path to the written file.
    """
    path = tmp_path / name
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def clear_temporal_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start every case from an environment with no ``TEMPORAL_*`` overrides."""
    for name in (ENV_ADDRESS, ENV_NAMESPACE, ENV_TASK_QUEUE):
        monkeypatch.delenv(name, raising=False)


class TestParsing:
    """A well-formed document parses into the expected model."""

    def test_full_document_parses(self, tmp_path: Path) -> None:
        # Arrange
        path = write_config(tmp_path, FULL_YAML)

        # Act
        config = load_worker_config(path)

        # Assert
        assert config.temporal.address == "temporal.internal:7233"
        assert config.temporal.namespace == "hardship-ns"
        assert config.temporal.task_queue == "hardship"
        assert config.temporal.tls is True
        assert [node.id for node in config.nodes] == ["evidence", "letter"]
        assert config.nodes[0].edge.agent == "agents/evidence.yaml"
        assert config.nodes[0].gate.schema_path == "gates/evidence.schema.json"

    def test_defaults_applied_when_optional_keys_omitted(self, tmp_path: Path) -> None:
        # Arrange
        path = write_config(tmp_path, MINIMAL_YAML)

        # Act
        config = load_worker_config(path)

        # Assert
        assert config.temporal.address == DEFAULT_ADDRESS
        assert config.temporal.namespace == DEFAULT_NAMESPACE
        assert config.temporal.tls is False

    def test_base_dir_is_the_configs_directory(self, tmp_path: Path) -> None:
        # Arrange
        config_dir = tmp_path / "deploy"
        config_dir.mkdir()
        path = write_config(config_dir, MINIMAL_YAML)

        # Act
        config = load_worker_config(path)

        # Assert
        assert config.base_dir == config_dir.resolve()

    def test_base_dir_is_excluded_from_serialization(self, tmp_path: Path) -> None:
        # Arrange
        config = load_worker_config(write_config(tmp_path, MINIMAL_YAML))

        # Act
        dumped = config.model_dump()

        # Assert
        assert "base_dir" not in dumped
        assert dumped["temporal"]["task_queue"] == "hardship"

    def test_accepts_a_string_path(self, tmp_path: Path) -> None:
        # Arrange
        path = write_config(tmp_path, MINIMAL_YAML)

        # Act
        config = load_worker_config(str(path))

        # Assert
        assert isinstance(config, WorkerConfig)


class TestClosedSchema:
    """Unknown keys are refused at every level (``extra="forbid"``)."""

    def test_unknown_root_key_refused(self, tmp_path: Path) -> None:
        # Arrange
        path = write_config(tmp_path, MINIMAL_YAML + "workflows: []\n")

        # Act / Assert
        with pytest.raises(ConfigError, match="workflows"):
            load_worker_config(path)

    def test_unknown_temporal_key_refused(self, tmp_path: Path) -> None:
        # Arrange — the timeout knobs decision 10 keeps caller-side.
        path = write_config(
            tmp_path,
            MINIMAL_YAML.replace(
                "  task_queue:", "  start_to_close: 60s\n  task_queue:"
            ),
        )

        # Act / Assert
        with pytest.raises(ConfigError, match="start_to_close"):
            load_worker_config(path)

    def test_unknown_node_key_refused(self, tmp_path: Path) -> None:
        # Arrange
        path = write_config(tmp_path, MINIMAL_YAML + "    retry_policy: aggressive\n")

        # Act / Assert
        with pytest.raises(ConfigError, match="retry_policy"):
            load_worker_config(path)

    def test_authored_base_dir_refused(self, tmp_path: Path) -> None:
        # Arrange
        path = write_config(tmp_path, MINIMAL_YAML + "base_dir: /etc\n")

        # Act / Assert
        with pytest.raises(ConfigError, match="base_dir"):
            load_worker_config(path)


class TestRequiredFields:
    """Fields without a sane default must be present and meaningful."""

    def test_missing_task_queue_refused(self, tmp_path: Path) -> None:
        # Arrange
        path = write_config(
            tmp_path, MINIMAL_YAML.replace("  task_queue: hardship\n", "")
        )

        # Act / Assert
        with pytest.raises(ConfigError, match="task_queue"):
            load_worker_config(path)

    @pytest.mark.parametrize("blank", ['""', '"   "'])
    def test_blank_task_queue_refused(self, tmp_path: Path, blank: str) -> None:
        # Arrange
        path = write_config(
            tmp_path,
            MINIMAL_YAML.replace("task_queue: hardship", f"task_queue: {blank}"),
        )

        # Act / Assert
        with pytest.raises(ConfigError, match="task_queue"):
            load_worker_config(path)

    def test_empty_nodes_list_refused(self, tmp_path: Path) -> None:
        # Arrange
        path = write_config(
            tmp_path,
            """\
            temporal:
              task_queue: hardship
            nodes: []
            """,
        )

        # Act / Assert
        with pytest.raises(ConfigError, match="nodes"):
            load_worker_config(path)

    def test_missing_nodes_refused(self, tmp_path: Path) -> None:
        # Arrange
        path = write_config(
            tmp_path,
            """\
            temporal:
              task_queue: hardship
            """,
        )

        # Act / Assert
        with pytest.raises(ConfigError, match="nodes"):
            load_worker_config(path)

    def test_duplicate_node_ids_refused(self, tmp_path: Path) -> None:
        # Arrange
        path = write_config(
            tmp_path,
            FULL_YAML.replace("  - id: letter", "  - id: evidence"),
        )

        # Act / Assert
        with pytest.raises(ConfigError, match="duplicate node id 'evidence'"):
            load_worker_config(path)


class TestEnvOverrides:
    """Shell environment wins over file values, matching ConfigLoader."""

    @pytest.mark.parametrize(
        ("env_name", "attribute", "value"),
        [
            (ENV_ADDRESS, "address", "env.temporal:7233"),
            (ENV_NAMESPACE, "namespace", "env-ns"),
            (ENV_TASK_QUEUE, "task_queue", "env-queue"),
        ],
    )
    def test_each_variable_overrides_the_file(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        env_name: str,
        attribute: str,
        value: str,
    ) -> None:
        # Arrange
        path = write_config(tmp_path, FULL_YAML)
        monkeypatch.setenv(env_name, value)

        # Act
        config = load_worker_config(path)

        # Assert
        assert getattr(config.temporal, attribute) == value

    def test_env_supplies_a_missing_required_task_queue(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        path = write_config(
            tmp_path, MINIMAL_YAML.replace("  task_queue: hardship\n", "")
        )
        monkeypatch.setenv(ENV_TASK_QUEUE, "env-queue")

        # Act
        config = load_worker_config(path)

        # Assert
        assert config.temporal.task_queue == "env-queue"

    @pytest.mark.parametrize("env_name", [ENV_ADDRESS, ENV_NAMESPACE, ENV_TASK_QUEUE])
    def test_empty_env_value_fails_closed(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        env_name: str,
    ) -> None:
        """A present-but-empty override is a broken secret or template.

        It must refuse to start, never silently fall back to the file value —
        for the task queue that fallback would route the worker to unintended
        workloads.
        """
        # Arrange
        path = write_config(tmp_path, FULL_YAML)
        monkeypatch.setenv(env_name, "")

        # Act / Assert
        with pytest.raises(ConfigError, match="must not be blank"):
            load_worker_config(path)

    @pytest.mark.parametrize("env_name", [ENV_ADDRESS, ENV_NAMESPACE, ENV_TASK_QUEUE])
    @pytest.mark.parametrize("padded", ["hardship ", " hardship", "\thardship\n"])
    def test_padded_env_value_fails_closed(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        env_name: str,
        padded: str,
    ) -> None:
        """Leading/trailing whitespace is refused, never silently kept.

        A padded task queue polls a *different* queue than the intended one —
        a worker that quietly polls the wrong queue looks identical to one
        that is down. Same fail-closed rule as the blank case.
        """
        # Arrange
        path = write_config(tmp_path, FULL_YAML)
        monkeypatch.setenv(env_name, padded)

        # Act / Assert
        with pytest.raises(ConfigError, match="leading or trailing whitespace"):
            load_worker_config(path)

    def test_tls_has_no_env_override(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        path = write_config(tmp_path, MINIMAL_YAML)
        monkeypatch.setenv("TEMPORAL_TLS", "true")

        # Act
        config = load_worker_config(path)

        # Assert
        assert config.temporal.tls is False


class TestFileFaults:
    """Missing, malformed, and mis-shaped files fail on their own channels."""

    def test_missing_file_raises_holodeck_file_not_found(self, tmp_path: Path) -> None:
        # Act / Assert
        with pytest.raises(HoloDeckFileNotFoundError):
            load_worker_config(tmp_path / "absent.yaml")

    def test_invalid_yaml_raises_config_error(self, tmp_path: Path) -> None:
        # Arrange
        path = write_config(tmp_path, "temporal: [unclosed\n")

        # Act / Assert
        with pytest.raises(ConfigError, match="Failed to parse YAML"):
            load_worker_config(path)

    @pytest.mark.parametrize("body", ["- one\n- two\n", "just a string\n", ""])
    def test_non_mapping_root_raises_config_error(
        self, tmp_path: Path, body: str
    ) -> None:
        # Arrange
        path = write_config(tmp_path, body)

        # Act / Assert
        with pytest.raises(ConfigError, match="mapping"):
            load_worker_config(path)


class TestPathConfinement:
    """``edge.agent`` may not escape the configuration directory."""

    def test_relative_escape_refused(self, tmp_path: Path) -> None:
        # Arrange
        config_dir = tmp_path / "deploy"
        config_dir.mkdir()
        path = write_config(
            config_dir,
            MINIMAL_YAML.replace("agents/evidence.yaml", "../outside/agent.yaml"),
        )

        # Act / Assert
        with pytest.raises(ConfigError, match="escapes the workflow directory"):
            load_worker_config(path)

    def test_absolute_escape_refused(self, tmp_path: Path) -> None:
        # Arrange
        config_dir = tmp_path / "deploy"
        config_dir.mkdir()
        path = write_config(
            config_dir,
            MINIMAL_YAML.replace("agents/evidence.yaml", "/etc/agent.yaml"),
        )

        # Act / Assert
        with pytest.raises(ConfigError, match="escapes the workflow directory"):
            load_worker_config(path)

    def test_escape_in_a_later_node_refused(self, tmp_path: Path) -> None:
        # Arrange — confinement must not stop at the first node.
        path = write_config(
            tmp_path, FULL_YAML.replace("agents/letter.yaml", "../letter.yaml")
        )

        # Act / Assert
        with pytest.raises(ConfigError, match="nodes.letter.edge.agent|escapes"):
            load_worker_config(path)

    def test_nested_path_inside_the_directory_allowed(self, tmp_path: Path) -> None:
        # Arrange — a subdirectory is not an escape, and the file need not
        # exist yet: existence is settled at bind time, not at load.
        path = write_config(
            tmp_path,
            MINIMAL_YAML.replace("agents/evidence.yaml", "agents/sub/evidence.yaml"),
        )

        # Act
        config = load_worker_config(path)

        # Assert
        assert config.nodes[0].edge.agent == "agents/sub/evidence.yaml"


_PROBE = """
import sys
import holodeck.temporal.worker_config
leaked = [name for name in {forbidden!r} if name in sys.modules]
print("LEAKED:", ", ".join(leaked) if leaked else "none")
sys.exit(1 if leaked else 0)
"""

_FORBIDDEN = ("temporalio.worker", "temporalio.client")


class TestImportPurity:
    """The config layer is pure — no Temporal client or worker machinery."""

    def test_import_pulls_no_temporal_runtime(self) -> None:
        # Act — a subprocess, so an SDK already imported by the test session
        # cannot mask a regression.
        result = subprocess.run(  # noqa: S603
            [sys.executable, "-c", _PROBE.format(forbidden=_FORBIDDEN)],
            capture_output=True,
            text=True,
            timeout=120,
        )

        # Assert
        assert result.returncode == 0, (
            "importing holodeck.temporal.worker_config failed or pulled in the "
            f"Temporal runtime (stdout: {result.stdout.strip()!r}, "
            f"stderr: {result.stderr.strip()!r})"
        )
