"""Choosing which HoloDeck an image installs (spec 042, FR-029)."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from holodeck.deploy.install import (
    DEFAULT_BASE_REPOSITORY,
    HoloDeckInstall,
    bundled_constraints,
    bundled_overrides,
    default_base_image,
    is_published_version,
    resolve_install,
    wheel_constraints,
    wheel_overrides,
    wheel_version,
)
from holodeck.lib.errors import DeploymentError


def _wheel(
    directory: Path,
    version: str,
    constraints: str | None = "x==1\n",
    overrides: str | None = "y>=1\n",
) -> Path:
    path = directory / f"holodeck_ai-{version}-py3-none-any.whl"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("holodeck/__init__.py", "")
        if constraints is not None:
            archive.writestr("holodeck/deploy/constraints.txt", constraints)
        if overrides is not None:
            archive.writestr("holodeck/deploy/overrides.txt", overrides)
    return path


@pytest.mark.unit
@pytest.mark.parametrize(
    ("version", "published"),
    [
        ("0.7.1", True),
        ("1.0.0", True),
        ("1.0", True),
        ("1.0.0rc1", True),
        ("1.0.0a2", True),
        ("1.0.0.post1", True),
        ("0.7.2.dev61", False),
        ("0.7.2.dev61+g1e86d7f", False),
        ("0.0.0.dev0", False),
        ("0.7.1+local", False),
        ("latest", False),
        ("", False),
    ],
)
def test_is_published_version(version: str, published: bool) -> None:
    assert is_published_version(version) is published


@pytest.mark.unit
def test_release_installs_from_pypi_with_bundled_constraints() -> None:
    install = resolve_install("0.7.1")

    assert install.version == "0.7.1"
    assert install.wheel is None
    assert install.source == "pypi"
    assert install.constraints == bundled_constraints()
    assert install.overrides == bundled_overrides()


@pytest.mark.unit
def test_bundled_constraints_pin_every_line() -> None:
    lines = [
        line
        for line in bundled_constraints().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    assert lines
    assert all("==" in line for line in lines)
    assert not any("holodeck-ai" in line for line in lines)


@pytest.mark.unit
def test_dev_version_without_wheel_fails_with_instructions(tmp_path: Path) -> None:
    with pytest.raises(DeploymentError) as error:
        resolve_install("0.7.2.dev61", search_dirs=[tmp_path])

    message = error.value.message
    assert "0.7.2.dev61" in message
    assert "not on PyPI" in message
    assert "uv build --wheel" in message
    assert "--wheel" in message


@pytest.mark.unit
def test_dev_version_finds_matching_wheel_in_search_dir(tmp_path: Path) -> None:
    wheel = _wheel(tmp_path, "0.7.2.dev61", constraints="a==1\n")
    _wheel(tmp_path, "0.7.2.dev60")

    install = resolve_install("0.7.2.dev61", search_dirs=[tmp_path])

    assert install.wheel == wheel
    assert install.version == "0.7.2.dev61"
    assert install.constraints == "a==1\n"
    assert install.overrides == "y>=1\n"
    assert install.source == "wheel"


@pytest.mark.unit
def test_explicit_wheel_wins_over_installed_version(tmp_path: Path) -> None:
    wheel = _wheel(tmp_path, "0.8.0.dev3", constraints="b==2\n")

    install = resolve_install("0.7.1", wheel=wheel)

    assert install.version == "0.8.0.dev3"
    assert install.wheel == wheel
    assert install.constraints == "b==2\n"


@pytest.mark.unit
def test_missing_wheel_fails(tmp_path: Path) -> None:
    with pytest.raises(DeploymentError, match="Wheel not found"):
        resolve_install("0.7.1", wheel=tmp_path / "holodeck_ai-1-py3-none-any.whl")


@pytest.mark.unit
def test_wheel_without_constraints_fails(tmp_path: Path) -> None:
    wheel = _wheel(tmp_path, "0.7.2.dev1", constraints=None)

    with pytest.raises(DeploymentError, match="constraints.txt"):
        wheel_constraints(wheel)


@pytest.mark.unit
def test_wheel_without_overrides_fails(tmp_path: Path) -> None:
    wheel = _wheel(tmp_path, "0.7.2.dev1", overrides=None)

    with pytest.raises(DeploymentError, match="overrides.txt"):
        wheel_overrides(wheel)


@pytest.mark.unit
def test_wheel_with_wrong_name_fails(tmp_path: Path) -> None:
    other = tmp_path / "other-1.0-py3-none-any.whl"
    other.write_bytes(b"")

    with pytest.raises(DeploymentError, match="not a holodeck-ai wheel"):
        wheel_version(other)


@pytest.mark.unit
def test_default_base_image_tracks_the_installed_version() -> None:
    release = HoloDeckInstall(version="0.7.1", constraints="")
    dev = HoloDeckInstall(version="0.7.2.dev1", constraints="", wheel=Path("w.whl"))

    assert default_base_image(release) == f"{DEFAULT_BASE_REPOSITORY}:0.7.1"
    assert default_base_image(dev) == f"{DEFAULT_BASE_REPOSITORY}:latest"
