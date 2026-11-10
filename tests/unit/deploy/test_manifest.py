"""The image manifest records what a build produced (spec 042, AC-11)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from holodeck.deploy.builder import ImageRef
from holodeck.deploy.install import HoloDeckInstall
from holodeck.deploy.manifest import (
    build_manifest,
    manifest_path,
    parse_freeze,
    sha256_text,
    write_manifest,
)
from holodeck.lib.errors import DeploymentError

FREEZE = """\
# holodeck-ai 1.2.3; extras: qdrant
pydantic==2.13.0
Alpha==1.0
qdrant-client==1.18.0
pydantic==2.13.0
"""


def _manifest(install: HoloDeckInstall | None = None, freeze: str = FREEZE):
    return build_manifest(
        agent="my-agent",
        platform="linux/amd64",
        install=install
        or HoloDeckInstall(version="1.2.3", constraints="a==1\n", overrides="b>=2\n"),
        extras=["qdrant", "claude-otel", "qdrant"],
        base_image=ImageRef(
            reference="ghcr.io/o/base:1.2.3",
            image_id="sha256:base",
            digest="sha256:basedigest",
        ),
        image_reference="ghcr.io/o/agent:abc",
        image=ImageRef(reference="sha256:img", image_id="sha256:img"),
        freeze_text=freeze,
    )


@pytest.mark.unit
def test_parse_freeze_drops_comments_and_duplicates_and_sorts() -> None:
    assert parse_freeze(FREEZE) == [
        "Alpha==1.0",
        "pydantic==2.13.0",
        "qdrant-client==1.18.0",
    ]


@pytest.mark.unit
def test_manifest_records_version_extras_base_digest_and_packages() -> None:
    manifest = _manifest()

    assert manifest.holodeck.version == "1.2.3"
    assert manifest.holodeck.source == "pypi"
    assert manifest.holodeck.extras == ["claude-otel", "qdrant"]
    assert manifest.holodeck.constraints_sha256 == sha256_text("a==1\n")
    assert manifest.holodeck.overrides_sha256 == sha256_text("b>=2\n")
    assert manifest.base_image.reference == "ghcr.io/o/base:1.2.3"
    assert manifest.base_image.digest == "sha256:basedigest"
    assert manifest.image.image_id == "sha256:img"
    assert manifest.image.digest is None
    assert manifest.platform == "linux/amd64"
    assert "qdrant-client==1.18.0" in manifest.packages


@pytest.mark.unit
def test_locally_built_base_image_records_its_id_without_a_digest() -> None:
    manifest = build_manifest(
        agent="a",
        platform="linux/amd64",
        install=HoloDeckInstall(version="1.2.3", constraints=""),
        extras=[],
        base_image=ImageRef(reference="local-base:t", image_id="sha256:localid"),
        image_reference="x:y",
        image=ImageRef(reference="sha256:i", image_id="sha256:i"),
        freeze_text=FREEZE,
    )

    assert manifest.base_image.digest is None
    assert manifest.base_image.image_id == "sha256:localid"


@pytest.mark.unit
def test_wheel_build_records_the_wheel_hash(tmp_path: Path) -> None:
    wheel = tmp_path / "holodeck_ai-1.2.3.dev1-py3-none-any.whl"
    wheel.write_bytes(b"wheel-bytes")

    manifest = _manifest(
        HoloDeckInstall(version="1.2.3.dev1", constraints="", wheel=wheel)
    )

    assert manifest.holodeck.source == "wheel"
    assert manifest.holodeck.wheel_sha256 is not None
    assert manifest.holodeck.wheel_sha256.startswith("sha256:")


@pytest.mark.unit
def test_empty_package_list_is_an_error() -> None:
    with pytest.raises(DeploymentError, match="manifest is empty"):
        _manifest(freeze="# only a header\n")


@pytest.mark.unit
def test_manifest_file_is_deterministic_and_sorted(tmp_path: Path) -> None:
    first = tmp_path / "a.json"
    second = tmp_path / "b.json"

    write_manifest(first, _manifest())
    write_manifest(second, _manifest())

    assert first.read_text() == second.read_text()
    data = json.loads(first.read_text())
    assert list(data) == sorted(data)
    assert "created" not in first.read_text()


@pytest.mark.unit
def test_manifest_path_is_next_to_deployment_state(tmp_path: Path) -> None:
    assert manifest_path(tmp_path, "abc1234") == (
        tmp_path / ".holodeck" / "builds" / "abc1234.json"
    )
