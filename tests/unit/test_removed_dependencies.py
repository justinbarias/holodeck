"""Guard: dependencies removed by 042 stay out of the lock and the source.

Remediation: when this fails, do not bring the package back. DeepEval metrics
replace Azure AI evaluation (042 T18). If another dependency started to pull
a removed package in, run ``uv tree --invert --package <name>`` to find it and
drop or replace that dependency.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"

REMOVED_LOCK_PACKAGES = ("azure-ai-evaluation",)
REMOVED_IMPORTS = ("azure.ai.evaluation",)


@pytest.mark.parametrize("name", REMOVED_LOCK_PACKAGES)
def test_the_lock_has_no_removed_package(name: str) -> None:
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    names = {package["name"] for package in lock["package"]}
    assert name not in names


@pytest.mark.parametrize("module", REMOVED_IMPORTS)
def test_source_does_not_import_removed_module(module: str) -> None:
    found = [
        path.relative_to(ROOT).as_posix()
        for path in sorted(SRC.rglob("*.py"))
        if module in path.read_text(encoding="utf-8")
    ]
    assert not found, f"{module} is referenced under src/: {found}"
