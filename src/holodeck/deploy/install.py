"""Decide which HoloDeck an image installs, and which constraints go with it.

An image installs exactly one HoloDeck version. A published release comes from
PyPI, pinned with ``==``. A development version (``0.7.2.dev61``, from a
checkout) is not on PyPI, so the build context carries a wheel instead. In both
cases the constraints file comes from the package being installed: the one
shipped in the CLI for a release, the one inside the wheel for a wheel. Next
to the constraints go the uv overrides the lock was resolved with.
"""

from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

from holodeck.lib.errors import DeploymentError

CONSTRAINTS_RESOURCE = "constraints.txt"
OVERRIDES_RESOURCE = "overrides.txt"
DEFAULT_BASE_REPOSITORY = "ghcr.io/justinbarias/holodeck-base"
_WHEEL_DEPLOY_DIR = "holodeck/deploy/"

# Versions that exist on PyPI as published releases: final, alpha, beta, rc,
# post. ``.devN`` and ``+local`` versions come from a checkout.
_PUBLISHED_VERSION = re.compile(
    r"^\d+(\.\d+)*((a|b|rc)\d+)?(\.post\d+)?$",
)
_WHEEL_NAME = re.compile(r"^holodeck_ai-(?P<version>[^-]+)-[^-]+-[^-]+-[^-]+\.whl$")


@dataclass(frozen=True)
class HoloDeckInstall:
    """What one image installs.

    Attributes:
        version: Exact HoloDeck version.
        constraints: Text of the constraints file for that version.
        overrides: Text of the uv overrides file for that version.
        wheel: Local wheel to put in the build context; None installs from PyPI.
    """

    version: str
    constraints: str
    overrides: str = ""
    wheel: Path | None = None

    @property
    def source(self) -> str:
        """``"wheel"`` for a local wheel, ``"pypi"`` otherwise."""
        return "wheel" if self.wheel is not None else "pypi"


def default_base_image(install: HoloDeckInstall) -> str:
    """Return the base image reference for ``install``.

    A release builds on the base image tagged with the same version, so the
    base image and the installed HoloDeck agree. A development build has no
    such tag and uses ``latest``.
    """
    if install.wheel is None:
        return f"{DEFAULT_BASE_REPOSITORY}:{install.version}"
    return f"{DEFAULT_BASE_REPOSITORY}:latest"


def is_published_version(version: str) -> bool:
    """True when ``version`` can be installed from PyPI."""
    return _PUBLISHED_VERSION.match(version) is not None


def _bundled(resource: str) -> str:
    try:
        return (
            resources.files("holodeck.deploy")
            .joinpath(resource)
            .read_text(encoding="utf-8")
        )
    except OSError as exc:
        raise DeploymentError(
            operation="build",
            message=(
                f"This HoloDeck installation has no {resource}. "
                "Reinstall holodeck-ai from a release, or from a wheel built "
                "with `uv build --wheel`."
            ),
        ) from exc


def bundled_constraints() -> str:
    """Return the constraints file shipped with the installed HoloDeck."""
    return _bundled(CONSTRAINTS_RESOURCE)


def bundled_overrides() -> str:
    """Return the uv overrides file shipped with the installed HoloDeck."""
    return _bundled(OVERRIDES_RESOURCE)


def wheel_version(wheel: Path) -> str:
    """Return the HoloDeck version a wheel file name declares."""
    match = _WHEEL_NAME.match(wheel.name)
    if match is None:
        raise DeploymentError(
            operation="build",
            message=(
                f"{wheel} is not a holodeck-ai wheel. "
                "Expected a file named holodeck_ai-<version>-<tags>.whl."
            ),
        )
    return match.group("version")


def _wheel_member(wheel: Path, resource: str) -> str:
    member = _WHEEL_DEPLOY_DIR + resource
    try:
        with zipfile.ZipFile(wheel) as archive:
            return archive.read(member).decode("utf-8")
    except (OSError, KeyError, zipfile.BadZipFile) as exc:
        raise DeploymentError(
            operation="build",
            message=(
                f"{wheel} has no {member}. Build the wheel from a checkout "
                "that includes it (`uv build --wheel`)."
            ),
        ) from exc


def wheel_constraints(wheel: Path) -> str:
    """Read the constraints file from inside a HoloDeck wheel."""
    return _wheel_member(wheel, CONSTRAINTS_RESOURCE)


def wheel_overrides(wheel: Path) -> str:
    """Read the uv overrides file from inside a HoloDeck wheel."""
    return _wheel_member(wheel, OVERRIDES_RESOURCE)


def _find_dev_wheel(version: str, search_dirs: list[Path]) -> Path | None:
    for directory in search_dirs:
        matches = sorted(directory.glob(f"holodeck_ai-{version}-*.whl"))
        if matches:
            return matches[-1]
    return None


def resolve_install(
    installed_version: str,
    wheel: Path | None = None,
    search_dirs: list[Path] | None = None,
) -> HoloDeckInstall:
    """Choose the HoloDeck an image installs.

    Args:
        installed_version: Version of the running HoloDeck.
        wheel: Wheel passed by the user. It wins over everything else and may
            be any version.
        search_dirs: Where to look for ``holodeck_ai-<version>-*.whl`` when
            the running version is a development version and no wheel is
            given (for example ``./dist``).

    Raises:
        DeploymentError: A development version has no wheel, or a given wheel
            is not a usable HoloDeck wheel.
    """
    if wheel is not None:
        if not wheel.is_file():
            raise DeploymentError(
                operation="build", message=f"Wheel not found: {wheel}"
            )
        return HoloDeckInstall(
            version=wheel_version(wheel),
            constraints=wheel_constraints(wheel),
            overrides=wheel_overrides(wheel),
            wheel=wheel,
        )

    if is_published_version(installed_version):
        return HoloDeckInstall(
            version=installed_version,
            constraints=bundled_constraints(),
            overrides=bundled_overrides(),
        )

    found = _find_dev_wheel(installed_version, search_dirs or [])
    if found is not None:
        return HoloDeckInstall(
            version=wheel_version(found),
            constraints=wheel_constraints(found),
            overrides=wheel_overrides(found),
            wheel=found,
        )
    raise DeploymentError(
        operation="build",
        message=(
            f"HoloDeck {installed_version} is a development version and is not "
            "on PyPI, so an image cannot install it by version. Build a wheel "
            "with `uv build --wheel` and pass it with `--wheel dist/<file>.whl`, "
            "or run `holodeck deploy build` from a released HoloDeck."
        ),
    )
