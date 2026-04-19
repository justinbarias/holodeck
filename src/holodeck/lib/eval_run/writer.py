"""Atomic writer for :class:`EvalRun` JSON artifacts.

Implements the ``mkstemp`` + ``fsync`` + ``os.replace`` pattern from
``specs/031-eval-runs-dashboard/research.md`` R3 so readers see either the
old file or the new file, never a partial one. A 4-hex collision suffix
is appended when the target path already exists (FR-008).
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import secrets
import tempfile
from pathlib import Path
from typing import Any

from pydantic import SecretStr

from holodeck.lib.eval_run.slugify import slugify
from holodeck.models.eval_run import EvalRun

logger = logging.getLogger(__name__)

_TIMESTAMP_COLON = re.compile(r":")


def _normalise_timestamp_for_filename(ts: str) -> str:
    """Replace colons with hyphens so the timestamp is filesystem-safe."""
    return _TIMESTAMP_COLON.sub("-", ts)


def _to_jsonable(value: Any) -> Any:
    """Walk a Pydantic ``mode='python'`` dump and unmask ``SecretStr`` leaves.

    The redactor has already replaced every secret value with the literal
    ``"***"``; this walker calls ``.get_secret_value()`` so the persisted JSON
    contains the redactor's placeholder verbatim instead of Pydantic's default
    ``"**********"``.
    """
    if isinstance(value, SecretStr):
        return value.get_secret_value()
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_to_jsonable(v) for v in value]
    return value


def _serialise(run: EvalRun) -> str:
    data = run.model_dump(mode="python")
    return json.dumps(_to_jsonable(data), indent=2, default=str)


def _resolve_target_path(run: EvalRun, agent_base_dir: Path) -> Path:
    slug = slugify(run.report.agent_name)
    parent = Path(agent_base_dir) / "results" / slug
    filename = f"{_normalise_timestamp_for_filename(run.report.timestamp)}.json"
    return parent / filename


def _avoid_collision(path: Path) -> Path:
    if not path.exists():
        return path
    suffix = secrets.token_hex(2)  # 4 hex chars
    return path.with_name(f"{path.stem}-{suffix}{path.suffix}")


def write_eval_run(run: EvalRun, agent_base_dir: Path) -> Path:
    """Atomically write an :class:`EvalRun` to disk and return the final path.

    Args:
        run: The run to persist.
        agent_base_dir: Directory containing ``agent.yaml``; ``results/`` is
            created underneath it (matching the convention used for
            ``instructions.file`` and vector-store paths).

    Returns:
        The absolute path of the written JSON file.

    Raises:
        OSError: If writing or replacing the target file fails. The temp file
            is removed before re-raising; no partial target survives.
    """
    target = _resolve_target_path(run, agent_base_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    target = _avoid_collision(target)

    payload = _serialise(run)

    tmp_fd, tmp_name = tempfile.mkstemp(
        prefix=target.name + ".", suffix=".tmp", dir=str(target.parent)
    )
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, target)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp_name)
        raise

    return target
