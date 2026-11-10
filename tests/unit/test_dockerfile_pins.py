"""Every Dockerfile and template installs an exact HoloDeck (spec 042, FR-029).

No tracked Dockerfile or Dockerfile template may resolve prereleases, and none
may install ``holodeck-ai`` without an exact version or a local wheel.
"""

from __future__ import annotations

import re
import subprocess  # nosec B404
from pathlib import Path

import pytest

from holodeck.deploy.dockerfile import HOLODECK_DOCKERFILE_TEMPLATE, generate_dockerfile

ROOT = Path(__file__).resolve().parents[2]

_PRERELEASE_FLAG = re.compile(r"(?<![\w-])--pre(release(=\S+)?)?(?![\w-])")
_PINNED = re.compile(r"holodeck-ai(\[[^\]]*\])?==\S+|\.whl")
_INSTALL = re.compile(r"\b(uv\s+)?pip\s+install\b")


def _tracked_dockerfiles() -> list[Path]:
    listing = subprocess.run(  # noqa: S603  # nosec B603
        ["git", "ls-files"],  # noqa: S607
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    return sorted(
        ROOT / name
        for name in listing
        if Path(name).name.startswith("Dockerfile") and (ROOT / name).is_file()
    )


def _instructions(text: str) -> list[str]:
    """Return Dockerfile instructions with continuations joined, comments dropped."""
    lines = [line for line in text.splitlines() if not line.lstrip().startswith("#")]
    return re.sub(r"\\\n", " ", "\n".join(lines)).splitlines()


def _holodeck_install_lines(text: str) -> list[str]:
    return [
        line
        for line in _instructions(text)
        if line.lstrip().startswith("RUN")
        and _INSTALL.search(line)
        and ("holodeck-ai" in line or ".whl" in line or "HOLODECK" in line)
    ]


def _rendered_templates() -> dict[str, str]:
    kwargs: dict = {"agent_name": "t", "port": 8080, "protocol": "rest"}
    return {
        "template (release)": generate_dockerfile(
            **kwargs, holodeck_version="1.2.3", extras=["qdrant"]
        ),
        "template (wheel)": generate_dockerfile(
            **kwargs,
            holodeck_version="0.7.2.dev1",
            holodeck_wheel="holodeck_ai-0.7.2.dev1-py3-none-any.whl",
            extras=["qdrant"],
        ),
        "template source": HOLODECK_DOCKERFILE_TEMPLATE,
    }


@pytest.mark.unit
def test_the_repository_ships_dockerfiles() -> None:
    names = {path.name for path in _tracked_dockerfiles()}
    assert {"Dockerfile", "Dockerfile.local"} <= names


@pytest.mark.unit
def test_no_dockerfile_resolves_prereleases() -> None:
    texts = {str(p.relative_to(ROOT)): p.read_text() for p in _tracked_dockerfiles()}
    texts.update(_rendered_templates())

    offenders = [
        f"{name}: {line.strip()}"
        for name, text in texts.items()
        for line in _instructions(text)
        if _PRERELEASE_FLAG.search(line)
    ]
    assert not offenders, "\n".join(offenders)


@pytest.mark.unit
def test_every_holodeck_install_is_exact_or_a_local_wheel() -> None:
    texts = {str(p.relative_to(ROOT)): p.read_text() for p in _tracked_dockerfiles()}
    texts.update(_rendered_templates())

    checked = 0
    offenders: list[str] = []
    for name, text in texts.items():
        for line in _holodeck_install_lines(text):
            checked += 1
            if not (_PINNED.search(line) or "${HOLODECK_VERSION}" in line):
                offenders.append(f"{name}: {line.strip()}")
    assert checked >= 4
    assert not offenders, "\n".join(offenders)


@pytest.mark.unit
def test_every_holodeck_install_uses_constraints() -> None:
    texts = {str(p.relative_to(ROOT)): p.read_text() for p in _tracked_dockerfiles()}
    texts.update(_rendered_templates())

    offenders = [
        f"{name}: {line.strip()}"
        for name, text in texts.items()
        for line in _holodeck_install_lines(text)
        # The first install of the base images is --no-deps: it only puts the
        # package on disk so its constraints file can be read.
        if "--no-deps" not in line and "--constraint" not in line
    ]
    assert not offenders, "\n".join(offenders)


@pytest.mark.unit
def test_base_images_pin_python_and_uv_by_digest() -> None:
    for path in _tracked_dockerfiles():
        text = path.read_text()
        assert re.search(r"python:3\.\d+(-slim)?@sha256:[0-9a-f]{64}", text), path
        assert re.search(r"astral-sh/uv:\d+\.\d+\.\d+@sha256:[0-9a-f]{64}", text), path
        assert "astral.sh/uv/install.sh" not in text, path
