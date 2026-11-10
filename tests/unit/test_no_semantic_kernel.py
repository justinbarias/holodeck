"""Guard: Semantic Kernel stays removed (042 T16, FR-003, AC-02).

Remediation: when this fails, do not bring Semantic Kernel back. Vector
storage goes through the native adapters behind ``holodeck.lib.retrieval``;
embeddings and contextual-retrieval chat go through LiteLLM
(``holodeck.lib.litellm_support``). If a dependency started to pull in
``semantic-kernel``, drop or replace that dependency.
"""

from __future__ import annotations

import importlib.util
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
NAMES = ("semantic_kernel", "semantic-kernel")

# Lines that may name Semantic Kernel: the description of a deprecated
# configuration key, kept for one release. Each entry is (path relative to the
# repository root, exact stripped line). None exist today.
ALLOWED: set[tuple[str, str]] = set()


def test_no_source_file_mentions_semantic_kernel() -> None:
    found = []
    for path in sorted(SRC.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        relative = path.relative_to(ROOT).as_posix()
        for number, line in enumerate(text.splitlines(), start=1):
            mentions = any(name in line for name in NAMES)
            if mentions and (relative, line.strip()) not in ALLOWED:
                found.append(f"{relative}:{number}: {line.strip()}")
    assert not found, "Semantic Kernel is referenced under src/:\n" + "\n".join(found)


def test_semantic_kernel_is_not_installed() -> None:
    assert importlib.util.find_spec("semantic_kernel") is None, (
        "semantic_kernel is importable in the dev environment; a dependency "
        "pulls in semantic-kernel. Run `uv tree --invert --package "
        "semantic-kernel` to find it."
    )


def test_the_lock_has_no_semantic_kernel_package() -> None:
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    names = {package["name"] for package in lock["package"]}
    assert "semantic-kernel" not in names
