"""Slugifier for the eval-run results directory.

Per `specs/031-eval-runs-dashboard/research.md` R4: lowercase + alphanumerics +
`-`; consecutive `-` collapsed; leading/trailing `-` stripped. Empty result
raises `ValueError` (in practice this is unreachable because `Agent.name`
already rejects empty strings — but the writer treats it as a hard invariant).
"""

from __future__ import annotations

import re

_NON_SLUG = re.compile(r"[^a-z0-9-]+")
_REPEATED_HYPHEN = re.compile(r"-+")


def slugify(name: str) -> str:
    """Convert ``name`` to a filesystem-safe slug for ``results/<slug>/``.

    Args:
        name: The raw agent name.

    Returns:
        Lowercased slug containing only ``[a-z0-9-]`` with no leading or
        trailing hyphens and no internal runs of consecutive hyphens.

    Raises:
        ValueError: When the input slugifies to an empty string.
    """
    slug = _NON_SLUG.sub("-", name.lower())
    slug = _REPEATED_HYPHEN.sub("-", slug).strip("-")
    if not slug:
        raise ValueError(f"agent.name slugified to empty string: {name!r}")
    return slug
