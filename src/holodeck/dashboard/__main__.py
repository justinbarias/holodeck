"""CLI entry point: ``python -m holodeck.dashboard``.

Invoked by ``holodeck test view`` via subprocess. Kept thin — all imports
that require the ``dashboard`` extra live inside ``main()`` so that a plain
``python -m holodeck.dashboard`` without the extra surfaces a clear
``ModuleNotFoundError`` pointing at the install hint.
"""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m holodeck.dashboard")
    parser.add_argument(
        "--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)"
    )
    parser.add_argument("--port", type=int, default=8501, help="Port (default: 8501)")
    parser.add_argument("--debug", action="store_true", help="Dash debug mode")
    args = parser.parse_args()

    try:
        from holodeck.dashboard.app import main as run_app
    except ModuleNotFoundError as exc:
        sys.stderr.write(
            f"Dashboard extra not installed ({exc.name!r} missing).\n"
            "Install it with:\n"
            "  uv add 'holodeck-ai[dashboard]'   # or: pip install 'holodeck-ai[dashboard]'\n"
        )
        sys.exit(2)

    run_app(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
