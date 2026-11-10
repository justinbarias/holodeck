"""ContainerBuilder image lookups used by the image manifest (spec 042, AC-11)."""

from __future__ import annotations

import io
import tarfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from holodeck.deploy.builder import ContainerBuilder, ImageRef, _pick_repo_digest
from holodeck.lib.errors import DeploymentError


@pytest.fixture
def client() -> MagicMock:
    return MagicMock()


@pytest.fixture
def builder(client: MagicMock) -> ContainerBuilder:
    with patch("docker.from_env", return_value=client):
        return ContainerBuilder()


@pytest.mark.unit
@pytest.mark.parametrize(
    ("reference", "repo_digests", "expected"),
    [
        ("ghcr.io/o/base:1.0", ["ghcr.io/o/base@sha256:aa"], "sha256:aa"),
        ("ghcr.io/o/base:1.0", [], None),
        (
            "ghcr.io/o/base:1.0",
            ["other/repo@sha256:bb", "ghcr.io/o/base@sha256:aa"],
            "sha256:aa",
        ),
        ("localhost:5000/base:1.0", ["localhost:5000/base@sha256:cc"], "sha256:cc"),
        ("ghcr.io/o/base@sha256:aa", ["ghcr.io/o/base@sha256:aa"], "sha256:aa"),
        # No repository matches: fall back to the first recorded digest.
        ("renamed:1.0", ["ghcr.io/o/base@sha256:aa"], "sha256:aa"),
    ],
)
def test_pick_repo_digest(
    reference: str, repo_digests: list[str], expected: str | None
) -> None:
    assert _pick_repo_digest(reference, repo_digests) == expected


@pytest.mark.unit
def test_inspect_image_returns_id_and_registry_digest(
    builder: ContainerBuilder, client: MagicMock
) -> None:
    image = MagicMock()
    image.id = "sha256:imageid"
    image.attrs = {"RepoDigests": ["ghcr.io/o/base@sha256:abc"]}
    client.images.get.return_value = image

    assert builder.inspect_image("ghcr.io/o/base:1.0") == ImageRef(
        reference="ghcr.io/o/base:1.0", image_id="sha256:imageid", digest="sha256:abc"
    )


@pytest.mark.unit
def test_inspect_image_of_a_local_build_has_no_digest(
    builder: ContainerBuilder, client: MagicMock
) -> None:
    image = MagicMock()
    image.id = "sha256:local"
    image.attrs = {"RepoDigests": []}
    client.images.get.return_value = image

    ref = builder.inspect_image("holodeck-base-local:1")

    assert ref.digest is None
    assert ref.image_id == "sha256:local"


@pytest.mark.unit
def test_inspect_unknown_image_fails_clearly(
    builder: ContainerBuilder, client: MagicMock
) -> None:
    from docker.errors import ImageNotFound

    client.images.get.side_effect = ImageNotFound("nope")

    with pytest.raises(DeploymentError, match="Cannot inspect image"):
        builder.inspect_image("missing:1")


def _tar_with(name: str, content: str) -> list[bytes]:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        data = content.encode()
        info = tarfile.TarInfo(name)
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
    return [buffer.getvalue()]


@pytest.mark.unit
def test_read_image_file_reads_without_running_and_removes_the_container(
    builder: ContainerBuilder, client: MagicMock
) -> None:
    container = MagicMock()
    container.get_archive.return_value = (_tar_with("manifest.txt", "a==1\n"), {})
    client.containers.create.return_value = container

    text = builder.read_image_file("sha256:img", "/opt/holodeck/manifest.txt")

    assert text == "a==1\n"
    client.containers.create.assert_called_once_with("sha256:img")
    container.start.assert_not_called()
    container.get_archive.assert_called_once_with("/opt/holodeck/manifest.txt")
    container.remove.assert_called_once_with(force=True)


@pytest.mark.unit
def test_read_image_file_missing_path_fails_and_still_cleans_up(
    builder: ContainerBuilder, client: MagicMock
) -> None:
    from docker.errors import NotFound

    container = MagicMock()
    container.get_archive.side_effect = NotFound("no such path")
    client.containers.create.return_value = container

    with pytest.raises(DeploymentError, match="Cannot read /opt/holodeck/manifest.txt"):
        builder.read_image_file("sha256:img", "/opt/holodeck/manifest.txt")

    container.remove.assert_called_once_with(force=True)


@pytest.mark.unit
@pytest.mark.parametrize("pull", [True, False])
def test_build_passes_the_pull_choice_to_docker(
    builder: ContainerBuilder, client: MagicMock, tmp_path: Path, pull: bool
) -> None:
    (tmp_path / "Dockerfile").write_text("FROM scratch\n")
    image = MagicMock()
    image.id = "sha256:x"
    client.images.build.return_value = (image, [])

    builder.build(str(tmp_path), "o/a", "t", pull=pull)

    assert client.images.build.call_args.kwargs["pull"] is pull


@pytest.mark.unit
def test_build_pulls_the_base_image_by_default(
    builder: ContainerBuilder, client: MagicMock, tmp_path: Path
) -> None:
    (tmp_path / "Dockerfile").write_text("FROM scratch\n")
    image = MagicMock()
    image.id = "sha256:x"
    client.images.build.return_value = (image, [])

    builder.build(str(tmp_path), "o/a", "t")

    assert client.images.build.call_args.kwargs["pull"] is True
