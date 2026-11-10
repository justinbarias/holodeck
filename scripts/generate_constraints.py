"""Generate the install pins that ship inside the wheel, from ``uv.lock``.

Two files in ``src/holodeck/deploy/``:

* ``constraints.txt``: every package any runtime extra can pull in, pinned to
  the version in ``uv.lock``. A constraint only limits versions; it never
  installs a package.
* ``overrides.txt``: the ``[tool.uv] override-dependencies`` of this project.
  ``uv.lock`` was resolved with them, and ``uv pip install`` does not read
  ``pyproject.toml`` of an installed package, so an image has to pass them
  again or the pinned versions cannot be installed together.

Every HoloDeck image installs ``holodeck-ai[extras]==X.Y.Z`` with
``--constraint`` and ``--override`` pointing at these files, so two builds of
the same HoloDeck version resolve the same dependency versions (spec 042,
FR-029).

    python scripts/generate_constraints.py            # write the files
    python scripts/generate_constraints.py --check     # CI: fail if stale

``dev`` is excluded: images never run the test suite.
"""

from __future__ import annotations

import argparse
import subprocess  # nosec B404
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEPLOY_DIR = ROOT / "src" / "holodeck" / "deploy"
CONSTRAINTS_PATH = DEPLOY_DIR / "constraints.txt"
OVERRIDES_PATH = DEPLOY_DIR / "overrides.txt"
PYPROJECT_PATH = ROOT / "pyproject.toml"

_UV_EXPORT = (
    "uv",
    "export",
    "--frozen",
    "--all-extras",
    "--no-extra",
    "dev",
    "--no-dev",
    "--no-hashes",
    "--no-emit-project",
    "--no-annotate",
    "--no-header",
)

CONSTRAINTS_HEADER = """\
# Generated from uv.lock by scripts/generate_constraints.py. Do not edit.
# Regenerate with: make constraints
# Installed by every HoloDeck image with --constraint so that one HoloDeck
# version always resolves the same dependency versions (spec 042, FR-029).
"""

OVERRIDES_HEADER = """\
# Generated from [tool.uv] override-dependencies in pyproject.toml by
# scripts/generate_constraints.py. Do not edit.
# Regenerate with: make constraints
# Installed by every HoloDeck image with --override, next to constraints.txt.
"""

# Kept for callers that only care about the constraints text.
HEADER = CONSTRAINTS_HEADER


def normalize(exported: str) -> str:
    """Return the canonical constraints text for ``uv export`` output.

    Lines are de-duplicated and sorted, so the text does not depend on the
    order or formatting of a particular uv release.
    """
    lines = {
        line.strip()
        for line in exported.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    return CONSTRAINTS_HEADER + "\n".join(sorted(lines)) + "\n"


def render_constraints() -> str:
    """Run ``uv export`` against ``uv.lock`` and return the file text."""
    result = subprocess.run(  # noqa: S603  # nosec B603
        _UV_EXPORT,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"uv export failed:\n{result.stderr.strip()}")
    return normalize(result.stdout)


def render_overrides() -> str:
    """Return the overrides text from ``[tool.uv] override-dependencies``."""
    data = tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))
    overrides = data.get("tool", {}).get("uv", {}).get("override-dependencies", [])
    return OVERRIDES_HEADER + "".join(f"{line}\n" for line in sorted(overrides))


def is_stale(current: str, rendered: str) -> bool:
    """True when the committed text differs from what ``uv.lock`` produces."""
    return current != rendered


def _outputs() -> list[tuple[Path, str]]:
    return [
        (CONSTRAINTS_PATH, render_constraints()),
        (OVERRIDES_PATH, render_overrides()),
    ]


def main(argv: list[str] | None = None) -> int:
    """Write the files, or in ``--check`` mode verify them."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if a committed file differs from uv.lock.",
    )
    args = parser.parse_args(argv)

    outputs = _outputs()
    if args.check:
        stale = [
            path
            for path, rendered in outputs
            if is_stale(path.read_text() if path.exists() else "", rendered)
        ]
        for path in stale:
            print(
                f"{path} is out of date with uv.lock. Run: make constraints",
                file=sys.stderr,
            )
        if stale:
            return 1
        for path, _ in outputs:
            print(f"{path} is up to date.")
        return 0

    for path, rendered in outputs:
        path.write_text(rendered)
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
