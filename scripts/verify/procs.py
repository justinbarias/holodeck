"""Subprocess helpers for the verify tiers: ports, serve, CLI runs, reporting."""

from __future__ import annotations

import os
import shutil
import signal
import socket
import subprocess  # noqa: S404 - starts local holodeck and fake-provider processes
import sys
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

import httpx

ROOT = Path(__file__).resolve().parents[2]

# Credentials and endpoints that must never leak into an offline run.
SECRET_ENV = (
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_ORG_ID",
    "OPENAI_PROJECT_ID",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "PINECONE_API_KEY",
    "AZURE_AI_SEARCH_ENDPOINT",
    "AZURE_AI_SEARCH_API_KEY",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    "CLAUDE_CODE_OAUTH_TOKEN",
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX",
    "CLAUDE_CODE_USE_FOUNDRY",
    "CLAUDECODE",
    "CLAUDE_CODE_ENTRYPOINT",
)


@dataclass
class Check:
    """One row of the pass/fail table."""

    name: str
    status: str  # PASS, FAIL or SKIP
    detail: str = ""


def scrubbed_env(
    extra: Mapping[str, str], *, keep: tuple[str, ...] = ()
) -> dict[str, str]:
    """Return ``os.environ`` without credentials (except *keep*), plus *extra*."""
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in SECRET_ENV or key in keep
    }
    env.update(extra)
    return env


def free_port() -> int:
    """Return a TCP port that is free on 127.0.0.1 right now."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def holodeck_bin() -> str:
    """Return the ``holodeck`` console script of the running environment."""
    candidate = Path(sys.executable).with_name("holodeck")
    if candidate.exists():
        return str(candidate)
    found = shutil.which("holodeck")
    if found is None:
        raise FileNotFoundError("holodeck CLI not found; run `uv sync --all-extras`")
    return found


def _group_alive(pgid: int) -> bool:
    """Return whether any process is left in process group *pgid*."""
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _stop(process: subprocess.Popen[bytes]) -> None:
    """Terminate a process and every descendant in its process group.

    The process leads its own session (``start_new_session=True``), so its
    group id is its pid. The group is signalled even when the leader has
    already exited, because children (the Claude CLI, MCP servers) can
    outlive it.
    """
    pgid = process.pid
    for sig, grace in ((signal.SIGTERM, 10.0), (signal.SIGKILL, 5.0)):
        if not _group_alive(pgid):
            break
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            break
        deadline = time.monotonic() + grace
        while _group_alive(pgid) and time.monotonic() < deadline:
            process.poll()
            time.sleep(0.1)
    with suppress(subprocess.TimeoutExpired):
        process.wait(timeout=5)


@contextmanager
def background(
    args: list[str], *, cwd: Path, env: Mapping[str, str], log: Path
) -> Iterator[subprocess.Popen[bytes]]:
    """Run *args* in the background with output in *log*; always stop it."""
    log.parent.mkdir(parents=True, exist_ok=True)
    handle: TextIO
    with log.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(  # noqa: S603
            args,
            cwd=cwd,
            env=dict(env),
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            yield process
        finally:
            _stop(process)


def wait_http(
    url: str,
    process: subprocess.Popen[bytes],
    *,
    timeout: float,
    ready: str | None = None,
) -> str | None:
    """Poll *url* until it answers (and contains *ready*, if given).

    Returns:
        ``None`` when ready, else the reason it never became ready.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return f"process exited with code {process.returncode}"
        try:
            response = httpx.get(url, timeout=2.0)
            if ready is None or ready in response.text:
                return None
        except httpx.HTTPError:
            pass
        time.sleep(0.5)
    return f"not ready after {timeout:.0f}s"


@contextmanager
def serve(
    config: Path, *, env: Mapping[str, str], log: Path, timeout: float = 180.0
) -> Iterator[tuple[str, str | None]]:
    """Start ``holodeck serve`` for *config*; yield ``(base_url, error)``."""
    port = free_port()
    args = [
        holodeck_bin(),
        "serve",
        str(config),
        "--port",
        str(port),
        "--host",
        "127.0.0.1",
    ]
    base_url = f"http://127.0.0.1:{port}"
    with background(args, cwd=config.parent, env=env, log=log) as process:
        error = wait_http(
            f"{base_url}/ready", process, timeout=timeout, ready='"ready":true'
        )
        yield base_url, error


def run_cli(
    args: list[str], *, cwd: Path, env: Mapping[str, str], log: Path, timeout: float
) -> int:
    """Run a holodeck CLI command to completion with output in *log*.

    Returns the exit code, or 124 on timeout. The command's whole process
    group is stopped afterwards either way, so no descendant outlives it.
    """
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(  # noqa: S603
            [holodeck_bin(), *args],
            cwd=cwd,
            env=dict(env),
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            code = 124
        finally:
            _stop(process)
    return code


def tail(path: Path, lines: int = 15) -> str:
    """Return the last *lines* of a log file, for failure details."""
    if not path.exists():
        return ""
    content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(content[-lines:])


def print_table(title: str, checks: list[Check]) -> None:
    """Print a fixed-width pass/fail table."""
    width = max([len(c.name) for c in checks] + [5])
    print(f"\n{title}")
    print(f"  {'check'.ljust(width)}  result  detail")
    for check in checks:
        detail = check.detail.splitlines()[0] if check.detail else ""
        print(f"  {check.name.ljust(width)}  {check.status:<6}  {detail[:140]}")
