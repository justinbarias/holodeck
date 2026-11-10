"""`holodeck deploy build` pins HoloDeck and records a manifest (spec 042, T23)."""

from __future__ import annotations

import json
import zipfile
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from holodeck.cli.commands.deploy import _prepare_build_context, build
from holodeck.config.loader import ConfigLoader
from holodeck.deploy.builder import BuildResult, ImageRef
from holodeck.deploy.install import (
    HoloDeckInstall,
    bundled_constraints,
    bundled_overrides,
)

AGENT_YAML = """\
name: pinned-agent
model:
  provider: openai
  name: gpt-4o
instructions:
  inline: Be brief.
tools:
  - name: docs
    type: vectorstore
    description: Search docs
    source: ./data
    database:
      provider: qdrant
deployment:
  registry:
    url: ghcr.io
    repository: test-org/pinned-agent
    tag_strategy: latest
  target:
    provider: aws
    aws:
      region: us-east-1
  protocol: rest
  port: 8080
"""

FREEZE = "# holodeck-ai 1.2.3; extras: openai-agents,qdrant\npydantic==2.13.0\n"


@pytest.fixture
def agent_config(tmp_path: Path) -> Path:
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "a.txt").write_text("hello")
    path = tmp_path / "agent.yaml"
    path.write_text(AGENT_YAML)
    return path


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def holodeck_version(request: pytest.FixtureRequest) -> Iterator[None]:
    version = getattr(request, "param", "1.2.3")
    with patch(
        "holodeck.cli.commands.deploy._installed_holodeck_version",
        return_value=version,
    ):
        yield


def _wheel(directory: Path, version: str) -> Path:
    path = directory / f"holodeck_ai-{version}-py3-none-any.whl"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("holodeck/deploy/constraints.txt", "pydantic==2.13.0\n")
        archive.writestr("holodeck/deploy/overrides.txt", "pytz>=2024\n")
    return path


@pytest.mark.unit
@pytest.mark.usefixtures("holodeck_version")
def test_release_dry_run_shows_exact_pin_and_versioned_base(
    runner: CliRunner, agent_config: Path
) -> None:
    result = runner.invoke(build, [str(agent_config), "--dry-run"])

    assert result.exit_code == 0, result.output
    assert '"holodeck-ai[openai-agents,qdrant]==1.2.3"' in result.output
    assert "FROM ghcr.io/justinbarias/holodeck-base:1.2.3" in result.output
    assert "--constraint /tmp/holodeck-constraints.txt" in result.output
    assert "--prerelease" not in result.output


@pytest.mark.unit
@pytest.mark.usefixtures("holodeck_version")
def test_base_image_override(runner: CliRunner, agent_config: Path) -> None:
    result = runner.invoke(
        build, [str(agent_config), "--dry-run", "--base-image", "reg.io/b@sha256:ab"]
    )

    assert result.exit_code == 0, result.output
    assert "FROM reg.io/b@sha256:ab" in result.output


@pytest.mark.unit
@pytest.mark.parametrize("holodeck_version", ["0.7.2.dev61"], indirect=True)
@pytest.mark.usefixtures("holodeck_version")
def test_dev_version_without_wheel_fails_with_a_clear_message(
    runner: CliRunner, agent_config: Path, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(build, [str(agent_config), "--dry-run"])

    assert result.exit_code == 3
    assert "0.7.2.dev61 is a development version" in result.output
    assert "uv build --wheel" in result.output


@pytest.mark.unit
@pytest.mark.parametrize("holodeck_version", ["0.7.2.dev61"], indirect=True)
@pytest.mark.usefixtures("holodeck_version")
def test_dev_version_with_wheel_installs_the_wheel(
    runner: CliRunner, agent_config: Path, tmp_path: Path
) -> None:
    wheel = _wheel(tmp_path, "0.7.2.dev61")

    result = runner.invoke(
        build, [str(agent_config), "--dry-run", "--wheel", str(wheel)]
    )

    assert result.exit_code == 0, result.output
    assert f"COPY {wheel.name} /tmp/wheels/{wheel.name}" in result.output
    installed = "/tmp/wheels/" + wheel.name  # noqa: S108
    assert f"{installed}[openai-agents,qdrant]" in result.output
    assert "FROM ghcr.io/justinbarias/holodeck-base:latest" in result.output


@pytest.mark.unit
@pytest.mark.parametrize("holodeck_version", ["0.7.2.dev61"], indirect=True)
@pytest.mark.usefixtures("holodeck_version")
def test_dev_version_finds_the_wheel_in_dist(
    runner: CliRunner, agent_config: Path, tmp_path: Path, monkeypatch
) -> None:
    (tmp_path / "dist").mkdir()
    wheel = _wheel(tmp_path / "dist", "0.7.2.dev61")
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(build, [str(agent_config), "--dry-run"])

    assert result.exit_code == 0, result.output
    assert f"COPY {wheel.name}" in result.output


@pytest.mark.unit
@pytest.mark.usefixtures("holodeck_version")
def test_build_context_carries_constraints_and_wheel(
    agent_config: Path, tmp_path: Path
) -> None:
    wheel = _wheel(tmp_path, "0.7.2.dev61")
    install = HoloDeckInstall(
        version="0.7.2.dev61",
        constraints="pydantic==2.13.0\n",
        overrides="pytz>=2024\n",
        wheel=wheel,
    )
    agent = ConfigLoader().load_agent_yaml(str(agent_config), substitute_env=False)

    context = _prepare_build_context(
        agent, agent.deployment, agent_config.parent, "t", install=install
    )

    try:
        assert (context / "holodeck-constraints.txt").read_text() == (
            "pydantic==2.13.0\n"
        )
        assert (context / "holodeck-overrides.txt").read_text() == "pytz>=2024\n"
        assert (context / wheel.name).read_bytes() == wheel.read_bytes()
        assert wheel.name in (context / "Dockerfile").read_text()
    finally:
        import shutil

        shutil.rmtree(context, ignore_errors=True)


@pytest.mark.unit
@pytest.mark.usefixtures("holodeck_version")
def test_release_build_context_uses_the_bundled_constraints(
    agent_config: Path,
) -> None:
    agent = ConfigLoader().load_agent_yaml(str(agent_config), substitute_env=False)

    context = _prepare_build_context(agent, agent.deployment, agent_config.parent, "t")

    try:
        assert (context / "holodeck-constraints.txt").read_text() == (
            bundled_constraints()
        )
        assert (context / "holodeck-overrides.txt").read_text() == bundled_overrides()
        assert not list(context.glob("*.whl"))
    finally:
        import shutil

        shutil.rmtree(context, ignore_errors=True)


@pytest.mark.unit
@pytest.mark.usefixtures("holodeck_version")
def test_build_writes_the_image_manifest(runner: CliRunner, agent_config: Path) -> None:
    builder = MagicMock()
    builder.build.return_value = BuildResult(
        image_id="sha256:built",
        image_name="ghcr.io/test-org/pinned-agent",
        tag="latest",
        full_name="ghcr.io/test-org/pinned-agent:latest",
    )
    builder.read_image_file.return_value = FREEZE
    builder.inspect_image.side_effect = lambda ref: (
        ImageRef(ref, "sha256:built")
        if ref == "sha256:built"
        else ImageRef(ref, "sha256:baseid", "sha256:basedigest")
    )

    with patch("holodeck.deploy.builder.ContainerBuilder", return_value=builder):
        result = runner.invoke(build, [str(agent_config), "--no-pull"])

    assert result.exit_code == 0, result.output
    assert builder.build.call_args.kwargs["pull"] is False
    builder.read_image_file.assert_called_once_with(
        "sha256:built", "/opt/holodeck/manifest.txt"
    )
    manifest_file = agent_config.parent / ".holodeck" / "builds" / "latest.json"
    assert str(manifest_file) in result.output
    manifest = json.loads(manifest_file.read_text())
    assert manifest["holodeck"]["version"] == "1.2.3"
    assert manifest["holodeck"]["extras"] == ["openai-agents", "qdrant"]
    assert manifest["holodeck"]["constraints_sha256"].startswith("sha256:")
    assert manifest["base_image"] == {
        "digest": "sha256:basedigest",
        "image_id": "sha256:baseid",
        "reference": "ghcr.io/justinbarias/holodeck-base:1.2.3",
    }
    assert manifest["image"]["image_id"] == "sha256:built"
    assert manifest["packages"] == ["pydantic==2.13.0"]
    assert manifest["platform"] == "linux/amd64"


@pytest.mark.unit
@pytest.mark.usefixtures("holodeck_version")
def test_build_pulls_by_default(runner: CliRunner, agent_config: Path) -> None:
    builder = MagicMock()
    builder.build.return_value = BuildResult(
        image_id="sha256:built",
        image_name="ghcr.io/test-org/pinned-agent",
        tag="latest",
        full_name="ghcr.io/test-org/pinned-agent:latest",
    )
    builder.read_image_file.return_value = FREEZE
    builder.inspect_image.return_value = ImageRef("x", "sha256:i")

    with patch("holodeck.deploy.builder.ContainerBuilder", return_value=builder):
        result = runner.invoke(build, [str(agent_config)])

    assert result.exit_code == 0, result.output
    assert builder.build.call_args.kwargs["pull"] is True
