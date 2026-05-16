"""Extract a leaf value from a JSON-encoded response by path.

Used by standard evaluators (``equality``, ``numeric``, ``bleu``, ``rouge``,
``meteor``) so they can grade a single field of a structured envelope
instead of the whole string. Wired into the executor at the point where
the per-metric ``EvalKwargsBuilder`` is constructed (``executor.py:1613``);
no evaluator needs to know about this.

Path grammar — intentionally small, no JSONPath operators:

    answer
    result.value
    items[0].score
    data.items[2].score

Keys match ``[A-Za-z_][\\w-]*``; indices are non-negative integers in
square brackets. The compiled validator (``PATH_RE``) is exported so
``EvaluationMetric`` can reject typos at config-load time.
"""

from __future__ import annotations

import json
import re
from typing import Any

# Single segment: identifier optionally followed by one or more bracketed
# integer indices. A full path is one or more segments joined by '.'.
_SEGMENT = r"[A-Za-z_][\w-]*(?:\[\d+\])*"
PATH_RE = re.compile(rf"^{_SEGMENT}(?:\.{_SEGMENT})*$")

_TOKEN_RE = re.compile(r"([A-Za-z_][\w-]*)|\[(\d+)\]")


def is_valid_path(path: str) -> bool:
    """Return ``True`` iff ``path`` matches the supported grammar."""
    return bool(PATH_RE.match(path))


def _tokens(path: str) -> list[str | int]:
    """Split ``path`` into an ordered list of key (str) and index (int) tokens."""
    out: list[str | int] = []
    for match in _TOKEN_RE.finditer(path):
        key, idx = match.groups()
        out.append(key if key is not None else int(idx))
    return out


def extract(response: str, path: str) -> tuple[str | None, str | None]:
    """Extract ``response[path]`` from a JSON-encoded response.

    Args:
        response: Raw response string; must be a JSON document.
        path: Dotted/bracketed path validated against ``PATH_RE``.

    Returns:
        ``(value_str, None)`` on success — the leaf coerced with ``str()``.
        ``(None, error)`` when the response is not JSON, the path is
        missing, or the leaf is a container (``dict``/``list``).
    """
    try:
        data: Any = json.loads(response)
    except (TypeError, ValueError) as exc:
        snippet = response[:80].replace("\n", " ")
        return None, (
            f"response_path={path!r} but response is not JSON "
            f"({type(exc).__name__}: {exc}); got: {snippet!r}"
        )

    cursor: Any = data
    for token in _tokens(path):
        if isinstance(token, int):
            if not isinstance(cursor, list):
                return None, (
                    f"response_path={path!r}: expected list at index [{token}], "
                    f"got {type(cursor).__name__}"
                )
            if token >= len(cursor):
                return None, (
                    f"response_path={path!r}: index [{token}] out of range "
                    f"(len={len(cursor)})"
                )
            cursor = cursor[token]
        else:
            if not isinstance(cursor, dict):
                return None, (
                    f"response_path={path!r}: expected dict at key {token!r}, "
                    f"got {type(cursor).__name__}"
                )
            if token not in cursor:
                return None, (
                    f"response_path={path!r}: key {token!r} not found "
                    f"(available: {sorted(cursor.keys())})"
                )
            cursor = cursor[token]

    if isinstance(cursor, (dict, list)):
        return None, (
            f"response_path={path!r}: leaf is {type(cursor).__name__}, "
            "expected a scalar (str/number/bool/null)"
        )
    return str(cursor), None


__all__ = ["PATH_RE", "extract", "is_valid_path"]
