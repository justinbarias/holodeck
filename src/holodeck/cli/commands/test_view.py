"""`holodeck test view` — launch the Dash dashboard.

Spawns `python -m holodeck.dashboard` as a subprocess and forwards Ctrl+C.
The dashboard package is an optional extra; if it's not installed we print
an install hint and exit 2 (FR-022, SC-007) — no Python traceback.

See specs/031-eval-runs-dashboard/contracts/cli.md §`holodeck test view`.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path

import click

from holodeck.config.loader import load_agent_with_config
from holodeck.lib.errors import ConfigError
from holodeck.lib.eval_run.slugify import slugify as _slugify

_INSTALL_HINT = (
    "Dashboard not installed. Install the optional extra:\n"
    "  uv add 'holodeck-ai[dashboard]'   # or: pip install 'holodeck-ai[dashboard]'\n"
)

_NETWORK_WARNING = (
    "Warning: Dash binds to 127.0.0.1 by default; if you override --host, "
    "firewall the port on shared infra.\n"
)


@click.command("view")
# NOTE: no exists=True — in --seed mode we don't need an agent.yaml on disk.
# Existence is validated in-body only when --seed is NOT passed.
@click.argument("agent_config", type=click.Path(), default="agent.yaml")
@click.option("--port", type=int, default=8501, help="Port for the Dash server.")
@click.option("--host", default="127.0.0.1", help="Bind host.")
@click.option("--no-browser", is_flag=True, help="Do not auto-open the browser.")
@click.option(
    "--seed", is_flag=True, hidden=True, help="Use the built-in seed dataset."
)
def view(
    agent_config: str,
    port: int,
    host: str,
    no_browser: bool,
    seed: bool,
) -> None:
    """Launch the Dash evaluation dashboard for this agent's run history.

    Reads `results/<slugify(agent.name)>/` under the agent config's directory.
    Pass `--seed` to run against the built-in golden fixture (24 runs, 6
    prompt versions) without needing any real results on disk.
    """
    # 1. Pre-flight: dashboard extra must be installed.
    if find_spec("dash") is None:
        sys.stderr.write(_INSTALL_HINT)
        raise click.exceptions.Exit(code=2)

    # 2. Load agent config to resolve slug + results dir (skipped in seed mode
    #    since the app ignores the results dir when HOLODECK_DASHBOARD_USE_SEED=1).
    agent_display_name = "customer-support"  # seed-mode default
    agent_slug = "customer-support"
    results_dir = Path.cwd()

    if not seed:
        agent_cfg_path = Path(agent_config).resolve()
        if not agent_cfg_path.is_file():
            sys.stderr.write(
                f"Agent config not found: {agent_config}\n"
                "Pass --seed to render the built-in golden dataset "
                "without an agent.yaml.\n"
            )
            raise click.exceptions.Exit(code=2)
        try:
            agent, _resolved, _loader = load_agent_with_config(str(agent_cfg_path))
        except (ConfigError, Exception) as exc:
            sys.stderr.write(f"Failed to load {agent_config}: {exc}\n")
            raise click.exceptions.Exit(code=2) from exc
        agent_display_name = agent.name
        agent_slug = _slugify(agent.name)
        agent_base_dir = agent_cfg_path.parent
        results_dir = agent_base_dir / "results" / agent_slug

    # 3. Network-safety warning (FR-020).
    sys.stderr.write(_NETWORK_WARNING)

    # 4. Build env + argv for the subprocess.
    env = os.environ.copy()
    env["HOLODECK_DASHBOARD_RESULTS_DIR"] = str(results_dir)
    env["HOLODECK_DASHBOARD_AGENT_NAME"] = agent_slug
    env["HOLODECK_DASHBOARD_AGENT_DISPLAY_NAME"] = agent_display_name
    if seed:
        env["HOLODECK_DASHBOARD_USE_SEED"] = "1"

    argv = [
        sys.executable,
        "-m",
        "holodeck.dashboard",
        "--host",
        host,
        "--port",
        str(port),
    ]

    if not seed:
        click.echo(f"Loading runs from {results_dir}")
    else:
        click.echo(
            "Seed mode: rendering the design-handoff dataset (24 runs) — "
            "no real results consulted."
        )
    click.echo(f"Dash serving on http://{host}:{port}/ (Ctrl+C to stop)")

    # 5. Launch subprocess with SIGINT forwarding (research R2).
    proc = subprocess.Popen(argv, env=env)  # noqa: S603

    def _forward_sigint(signum: int, frame: object) -> None:  # noqa: ARG001
        proc.send_signal(signal.SIGINT)

    prev_handler = signal.signal(signal.SIGINT, _forward_sigint)
    try:
        proc.wait()
    finally:
        signal.signal(signal.SIGINT, prev_handler)

    if proc.returncode not in (0, -signal.SIGINT, 130):
        raise click.exceptions.Exit(code=proc.returncode or 3)
