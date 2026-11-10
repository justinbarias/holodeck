"""Image manifest: what a built HoloDeck image contains (spec 042, AC-11).

``holodeck deploy build`` writes one JSON manifest per build, next to the
deployment state in ``<agent dir>/.holodeck/builds/<tag>.json``. It records the
HoloDeck version and extras, the base image with its digest (or image ID when
the base image has no registry digest), the resulting image ID, and every
installed package. The manifest has no timestamps, so two builds of the same
inputs produce the same file apart from the image ID.

The package list is read from ``/opt/holodeck/manifest.txt`` inside the image,
which the generated Dockerfile writes with ``uv pip freeze`` after the install.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from holodeck.deploy.builder import ImageRef
from holodeck.deploy.install import HoloDeckInstall
from holodeck.lib.errors import DeploymentError

MANIFEST_VERSION = "1"


class BaseImageRecord(BaseModel):
    """The base image a build started from."""

    model_config = ConfigDict(extra="forbid")

    reference: str = Field(..., description="Tag or digest reference in the FROM line")
    digest: str | None = Field(
        default=None, description="Registry digest; None for a locally built image"
    )
    image_id: str = Field(..., description="Image ID of the base image")


class ImageRecord(BaseModel):
    """The image the build produced."""

    model_config = ConfigDict(extra="forbid")

    reference: str = Field(..., description="Image name and tag")
    image_id: str = Field(..., description="Image ID (sha256 of the image config)")
    digest: str | None = Field(
        default=None, description="Registry digest; set once the image is pushed"
    )


class HoloDeckRecord(BaseModel):
    """The HoloDeck package an image installs."""

    model_config = ConfigDict(extra="forbid")

    version: str = Field(..., description="Exact HoloDeck version")
    source: str = Field(..., description="pypi, or wheel for a development build")
    wheel_sha256: str | None = Field(
        default=None, description="SHA-256 of the wheel when source is wheel"
    )
    extras: list[str] = Field(default_factory=list, description="Extras installed")
    constraints_sha256: str = Field(
        ..., description="SHA-256 of the constraints file the install used"
    )
    overrides_sha256: str = Field(
        ..., description="SHA-256 of the uv overrides file the install used"
    )


class ImageManifest(BaseModel):
    """Everything needed to say what is inside a built image."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(default=MANIFEST_VERSION)
    agent: str = Field(..., description="Agent name")
    platform: str = Field(..., description="Target platform, e.g. linux/amd64")
    holodeck: HoloDeckRecord
    base_image: BaseImageRecord
    image: ImageRecord
    packages: list[str] = Field(
        ..., description="Installed packages as name==version, sorted"
    )


def parse_freeze(text: str) -> list[str]:
    """Return the sorted ``name==version`` lines of ``uv pip freeze`` output."""
    return sorted(
        {
            line.strip()
            for line in text.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
    )


def sha256_text(text: str) -> str:
    """Return ``sha256:<hex>`` of ``text``."""
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def build_manifest(
    *,
    agent: str,
    platform: str,
    install: HoloDeckInstall,
    extras: list[str],
    base_image: ImageRef,
    image_reference: str,
    image: ImageRef,
    freeze_text: str,
) -> ImageManifest:
    """Assemble the manifest of one build.

    Raises:
        DeploymentError: If the image reported no packages. An install that
            produced an empty list did not run, so the manifest would lie.
    """
    packages = parse_freeze(freeze_text)
    if not packages:
        raise DeploymentError(
            operation="build",
            message="The image package manifest is empty; the install did not run.",
        )
    return ImageManifest(
        agent=agent,
        platform=platform,
        holodeck=HoloDeckRecord(
            version=install.version,
            source=install.source,
            wheel_sha256=(
                _sha256_file(install.wheel) if install.wheel is not None else None
            ),
            extras=sorted(set(extras)),
            constraints_sha256=sha256_text(install.constraints),
            overrides_sha256=sha256_text(install.overrides),
        ),
        base_image=BaseImageRecord(
            reference=base_image.reference,
            digest=base_image.digest,
            image_id=base_image.image_id,
        ),
        image=ImageRecord(
            reference=image_reference,
            image_id=image.image_id,
            digest=image.digest,
        ),
        packages=packages,
    )


def manifest_path(agent_dir: Path, tag: str) -> Path:
    """Return where the manifest of the build tagged ``tag`` is written."""
    return agent_dir / ".holodeck" / "builds" / f"{tag}.json"


def write_manifest(path: Path, manifest: ImageManifest) -> None:
    """Write ``manifest`` as stable, sorted JSON."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True)
        path.write_text(payload + "\n", encoding="utf-8")
    except OSError as exc:
        raise DeploymentError(
            operation="build",
            message=f"Failed to write the image manifest to {path}: {exc}",
        ) from exc
