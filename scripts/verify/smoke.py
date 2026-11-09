"""Offline end-to-end smoke tier: real ``holodeck`` processes, fake provider.

For each backend (OpenAI Agents, Claude Agent SDK) this script:

1. runs ``holodeck test run`` on an offline fixture (deterministic tool
   assertion, no LLM judge) and requires exit 0;
2. starts ``holodeck serve`` and drives 2 threads x 2 turns over AG-UI
   ``POST /awp``; turn 2 of thread A asks for the smoke tool;
3. checks each run's events and, from the fake provider log, that thread B's
   provider requests never contain thread A's messages.

No credentials or network are needed. Artifacts land in ``.verify/smoke/<ts>/``.

Usage::

    uv run python scripts/verify/smoke.py [--backend openai|claude]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

if not __package__:
    # Run as a file path (PEP 366): import the package so relative imports work.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "scripts.verify"  # noqa: A001

from .agui import (  # noqa: E402
    TurnSpec,
    check_turn,
    drive,
    save_transcript,
)
from .fake_provider import TOOL_VERIFIED, read_log  # noqa: E402
from .procs import (  # noqa: E402
    ROOT,
    Check,
    background,
    free_port,
    print_table,
    run_cli,
    scrubbed_env,
    serve,
    tail,
    wait_http,
)

FIXTURES = ROOT / "tests" / "fixtures" / "verify"
OK_TEXT = "SMOKE-OK"
TRIGGER = "SMOKE_TOOL"


@dataclass(frozen=True)
class Backend:
    """One backend under smoke test."""

    name: str
    fixture: str
    module: str  # import that proves the backend's SDK is installed

    def env(self, provider_url: str, home: Path) -> dict[str, str]:
        """Return the scrubbed environment pointing this backend at the fake.

        ``HOME`` points at an empty per-run directory: the holodeck CLI loads
        ``~/.holodeck/.env`` with ``override=False``, which would otherwise put
        the developer's scrubbed credentials straight back.
        """
        home.mkdir(parents=True, exist_ok=True)
        common = {
            "HOME": str(home),
            "OPENAI_AGENTS_DISABLE_TRACING": "1",
        }
        if self.name == "openai":
            return scrubbed_env(
                {
                    **common,
                    "OPENAI_API_KEY": "sk-verify-smoke-fake",
                    "OPENAI_BASE_URL": f"{provider_url}/v1",
                }
            )
        return scrubbed_env(
            {
                **common,
                "ANTHROPIC_AUTH_TOKEN": "verify-smoke-fake",
                "HOLODECK_VERIFY_ANTHROPIC_URL": provider_url,
                # Keep the Claude Code CLI hermetic: no user settings, no
                # telemetry or update checks leaving the machine.
                "CLAUDE_CONFIG_DIR": str(home / ".claude"),
                "DISABLE_TELEMETRY": "1",
                "DISABLE_AUTOUPDATER": "1",
                "DISABLE_ERROR_REPORTING": "1",
                "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            }
        )


BACKENDS = (
    Backend(name="openai", fixture="smoke-openai.yaml", module="agents"),
    Backend(name="claude", fixture="smoke-claude.yaml", module="claude_agent_sdk"),
)


def conversation(backend: str) -> tuple[list[TurnSpec], list[str], str, str]:
    """Return the 2x2 turn plan, one nonce per turn, and the thread markers.

    Each prompt starts with its own nonce, and the fake provider echoes the
    start of the prompt it received, so a turn only passes when its own
    prompt reached the provider (a dropped or swapped prompt fails).
    """
    marker_a = f"ALPHA-{backend}-{uuid.uuid4().hex[:8]}"
    marker_b = f"BRAVO-{backend}-{uuid.uuid4().hex[:8]}"
    nonces = [f"T{n}-{uuid.uuid4().hex[:6]}" for n in range(1, 5)]
    turns = [
        TurnSpec("A", f"{nonces[0]} Hello, my code word is {marker_a}."),
        TurnSpec("B", f"{nonces[1]} Hello, my code word is {marker_b}."),
        TurnSpec("A", f"{nonces[2]} Use the smoke tool. {TRIGGER} {marker_a}", True),
        TurnSpec("B", f"{nonces[3]} Second turn, code word {marker_b} again."),
    ]
    return turns, nonces, marker_a, marker_b


def isolation_problems(log: Path, marker_a: str, marker_b: str) -> list[str]:
    """Check that requests carrying thread B never carry thread A's text."""
    b_requests = 0
    problems: list[str] = []
    for record in read_log(log):
        body = json.dumps(record.get("body"))
        if marker_b not in body:
            continue
        b_requests += 1
        if marker_a in body:
            problems.append(f"thread B request to {record.get('path')} leaks thread A")
    if b_requests == 0:
        problems.append("no provider request carried thread B's message")
    return problems


def missing_prerequisite(
    backend: Backend, platform: str, which: Callable[[str], str | None]
) -> str | None:
    """Return why *backend* cannot start on this host, or None.

    On Linux the Claude Code CLI refuses to start without bubblewrap while
    HoloDeck's default subprocess env scrubbing is on, and every turn would
    fail with an opaque "Backend session terminated unexpectedly".
    """
    if backend.name == "claude" and platform == "linux" and which("bwrap") is None:
        return (
            "bubblewrap (bwrap) not found: the Claude Code CLI needs it on Linux "
            "for HoloDeck's default subprocess env scrubbing. "
            "Install it: apt-get install -y bubblewrap"
        )
    return None


def smoke_backend(
    backend: Backend, workdir: Path, provider_url: str, provider_log: Path
) -> list[Check]:
    """Run the test-run and serve checks for one backend."""
    prefix = backend.name
    if importlib.util.find_spec(backend.module) is None:
        # Smoke promises both backends; a missing SDK is a failure, not a skip.
        reason = f"{backend.module} not installed (run `uv sync --all-extras`)"
        return [Check(f"{prefix}: setup", "FAIL", reason)]
    missing = missing_prerequisite(backend, sys.platform, shutil.which)
    if missing is not None:
        return [Check(f"{prefix}: setup", "FAIL", missing)]

    agents_dir = workdir / prefix / "agents"
    shutil.copytree(FIXTURES, agents_dir)
    config = agents_dir / backend.fixture
    env = backend.env(provider_url, workdir / prefix / "home")
    checks: list[Check] = []

    test_log = workdir / prefix / "test-run.log"
    code = run_cli(
        ["test", "run", str(config)],
        cwd=agents_dir,
        env=env,
        log=test_log,
        timeout=300,
    )
    checks.append(
        Check(
            f"{prefix}: holodeck test run",
            "PASS" if code == 0 else "FAIL",
            "" if code == 0 else f"exit {code}; see {test_log}\n{tail(test_log)}",
        )
    )

    turns, nonces, marker_a, marker_b = conversation(prefix)
    serve_log = workdir / prefix / "serve.log"
    with serve(config, env=env, log=serve_log) as (base_url, error):
        if error is not None:
            checks.append(
                Check(f"{prefix}: serve ready", "FAIL", f"{error}; see {serve_log}")
            )
            return checks
        results = drive(base_url, turns)

    save_transcript(results, workdir / prefix / "agui.jsonl")
    for result, nonce in zip(results, nonces, strict=True):
        name = f"{prefix}: thread {result.spec.thread} turn {result.index}"
        if result.spec.expect_tool:
            name += " (tool)"
        if result.transport_error:
            problems = [result.transport_error]
        else:
            problems = check_turn(
                result.events,
                expect_text=nonce,
                expect_tool=result.spec.expect_tool,
            )
            if OK_TEXT not in result.text:
                problems.append(f"text does not contain {OK_TEXT!r}")
            if result.spec.expect_tool and TOOL_VERIFIED not in result.text:
                problems.append(
                    f"tool result not verified (reply: {result.text[:120]!r})"
                )
        checks.append(Check(name, "FAIL" if problems else "PASS", "; ".join(problems)))
    problems = isolation_problems(provider_log, marker_a, marker_b)
    checks.append(
        Check(
            f"{prefix}: thread isolation",
            "FAIL" if problems else "PASS",
            "; ".join(problems),
        )
    )
    return checks


def unexpected_endpoints(log: Path) -> list[str]:
    """Return the distinct unhandled ``METHOD path`` pairs from the log."""
    seen: list[str] = []
    for record in read_log(log):
        if record.get("status") == 404:
            entry = f"{record.get('method')} {record.get('path')}"
            if entry not in seen:
                seen.append(entry)
    return seen


def main(argv: list[str] | None = None) -> int:
    """Run the smoke tier and return the process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend", choices=[b.name for b in BACKENDS], action="append"
    )
    args = parser.parse_args(argv)
    selected = [b for b in BACKENDS if not args.backend or b.name in args.backend]

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    workdir = ROOT / ".verify" / "smoke" / stamp
    workdir.mkdir(parents=True, exist_ok=True)
    provider_log = workdir / "provider.jsonl"
    port = free_port()
    provider_url = f"http://127.0.0.1:{port}"
    checks: list[Check] = []

    provider_args = [
        sys.executable,
        str(ROOT / "scripts" / "verify" / "fake_provider.py"),
        "--port",
        str(port),
        "--log",
        str(provider_log),
    ]
    with background(
        provider_args,
        cwd=ROOT,
        env=scrubbed_env({}),
        log=workdir / "provider.log",
    ) as provider:
        error = wait_http(f"{provider_url}/health", provider, timeout=30)
        if error is not None:
            checks.append(Check("fake provider", "FAIL", error))
        else:
            for backend in selected:
                checks.extend(
                    smoke_backend(backend, workdir, provider_url, provider_log)
                )

    print_table(f"verify-smoke ({workdir.relative_to(ROOT)})", checks)
    probes = unexpected_endpoints(provider_log)
    if probes:
        print(f"  unhandled provider endpoints (404): {', '.join(probes)}")
    failed = [c for c in checks if c.status != "PASS"]
    if not checks:
        failed = [Check("smoke", "FAIL", "no checks ran")]
    for check in failed:
        if "\n" in check.detail:
            print(f"\n--- {check.name} ---\n{check.detail}")
    print(
        f"\nverify-smoke: {'FAILED' if failed else 'PASSED'} "
        f"({sum(c.status == 'PASS' for c in checks)} passed, {len(failed)} failed)"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
