"""MkDocs build hooks.

Currently:
- ``on_pre_build`` copies ``schemas/agent.schema.json`` into ``docs/schemas/``
  so the JSON Schema is published as a raw static asset at
  ``https://docs.useholodeck.ai/schemas/schema.json`` (canonical) and
  ``/schemas/agent.schema.json`` (legacy alias). This lets any editor that
  speaks ``yaml-language-server`` resolve the schema directly from the docs
  site for auto-complete and validation.

The copy targets are git-ignored — the upstream source of truth remains
``schemas/agent.schema.json`` at the repo root.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

SCHEMA_SOURCE = Path("schemas/agent.schema.json")
PUBLISHED_NAMES = ("schema.json", "agent.schema.json")


def on_pre_build(config: Any) -> None:
    """Copy the agent JSON Schema into the docs tree before the build."""
    repo_root = Path(config["config_file_path"]).parent
    src = repo_root / SCHEMA_SOURCE
    if not src.exists():
        return
    target_dir = Path(config["docs_dir"]) / "schemas"
    target_dir.mkdir(parents=True, exist_ok=True)
    for name in PUBLISHED_NAMES:
        shutil.copy2(src, target_dir / name)
