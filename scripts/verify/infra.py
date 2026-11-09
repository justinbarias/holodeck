"""Docker infrastructure for the live verify tier: Qdrant and the Aspire dashboard.

Each service runs in a container with a fixed name. ``ensure`` reuses a
running container, restarts a stopped one, or starts a new one, then waits
until the service answers. When the host ports are already served by another
process of the same kind (for example a developer's own Aspire dashboard),
that process is reused and left alone. Containers stay up after the run so
traces can be browsed; ``stop_hint`` prints how to remove them.

Docker problems never raise: they come back as a not-ready ``InfraStatus``
with a reason, which the live tier reports as SKIP.
"""

from __future__ import annotations

import shutil
import socket
import subprocess  # noqa: S404 - runs the local docker CLI
import time
from collections.abc import Sequence
from dataclasses import dataclass

import httpx

DOCKER_TIMEOUT = 30.0
RUN_TIMEOUT = 600.0  # docker run may pull the pinned image first


@dataclass(frozen=True)
class Service:
    """One containerised service and how to tell that it is ready."""

    label: str
    container: str
    image: str
    ports: tuple[tuple[int, int], ...]  # (host port, container port)
    env: tuple[tuple[str, str], ...]
    ready_url: str
    ready_text: str | None = None
    tcp_ports: tuple[int, ...] = ()  # host ports that must accept connections


QDRANT = Service(
    label="qdrant",
    container="holodeck-verify-qdrant",
    image="qdrant/qdrant:v1.17.1",
    ports=((6333, 6333),),
    env=(),
    ready_url="http://127.0.0.1:6333/readyz",
    ready_text="all shards are ready",
)

# Image env (docker inspect): UI on 18888, OTLP/gRPC on 18889, OTLP/HTTP on
# 18890. Host 4317/4318 are the standard OTLP ports the fixtures export to.
ASPIRE = Service(
    label="aspire",
    container="holodeck-verify-aspire",
    image="mcr.microsoft.com/dotnet/aspire-dashboard:13.5.2",
    ports=((18888, 18888), (4317, 18889), (4318, 18890)),
    env=(("ASPIRE_DASHBOARD_UNSECURED_ALLOW_ANONYMOUS", "true"),),
    ready_url="http://127.0.0.1:18888/traces",
    ready_text="Aspire.Dashboard",
    tcp_ports=(4317, 4318),
)

ASPIRE_TRACES_URL = "http://127.0.0.1:18888/traces"


@dataclass(frozen=True)
class InfraStatus:
    """Outcome of ``ensure`` for one service."""

    service: Service
    ready: bool
    detail: str
    managed: bool = False  # True when the fixed-name container serves it


def _docker(args: Sequence[str], timeout: float = DOCKER_TIMEOUT) -> tuple[int, str]:
    """Run ``docker <args>``; return the exit code and combined output."""
    docker = shutil.which("docker")
    if docker is None:
        return 127, "docker CLI not found"
    try:
        done = subprocess.run(  # noqa: S603
            [docker, *args],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return 124, f"docker {args[0]} timed out after {timeout:.0f}s"
    return done.returncode, (done.stdout + done.stderr).strip()


def docker_problem() -> str | None:
    """Return why Docker cannot be used, or ``None`` when the daemon answers."""
    code, output = _docker(["info", "--format", "{{.ServerVersion}}"])
    if code == 0:
        return None
    lines = output.splitlines()
    return f"Docker unavailable: {lines[-1] if lines else f'exit {code}'}"


def container_state(name: str) -> str | None:
    """Return the container's state (``running``, ``exited``...) or ``None``."""
    code, output = _docker(["inspect", "--format", "{{.State.Status}}", name])
    return output if code == 0 else None


def port_open(port: int, host: str = "127.0.0.1") -> bool:
    """Return whether something accepts TCP connections on *host*:*port*."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1.0)
        return sock.connect_ex((host, port)) == 0


def run_args(service: Service) -> list[str]:
    """Return the ``docker run`` arguments that start *service*."""
    args = ["run", "-d", "--name", service.container, "--label", "holodeck.verify=1"]
    for host, container in service.ports:
        args += ["-p", f"127.0.0.1:{host}:{container}"]
    for key, value in service.env:
        args += ["-e", f"{key}={value}"]
    return [*args, service.image]


def probe(service: Service) -> str | None:
    """Return why *service* is not ready right now, or ``None`` when it is."""
    try:
        response = httpx.get(service.ready_url, timeout=2.0, follow_redirects=True)
    except httpx.HTTPError as exc:
        return f"{service.ready_url}: {type(exc).__name__}"
    if response.status_code != 200:
        return f"{service.ready_url}: HTTP {response.status_code}"
    if service.ready_text is not None and service.ready_text not in response.text:
        return f"{service.ready_url}: missing {service.ready_text!r}"
    closed = [port for port in service.tcp_ports if not port_open(port)]
    if closed:
        return f"port(s) {closed} not accepting connections"
    return None


def wait_ready(service: Service, timeout: float) -> str | None:
    """Poll ``probe`` until ready; return the last reason on timeout."""
    deadline = time.monotonic() + timeout
    reason = probe(service)
    while reason is not None and time.monotonic() < deadline:
        time.sleep(0.5)
        reason = probe(service)
    return reason


def ensure(service: Service, timeout: float = 60.0) -> InfraStatus:
    """Start or reuse *service* and wait until it is ready."""
    problem = docker_problem()
    if problem is not None:
        return InfraStatus(service, ready=False, detail=problem)

    state = container_state(service.container)
    if state == "running":
        action = f"reused container {service.container}"
    elif state is not None:
        code, output = _docker(["start", service.container])
        if code != 0:
            return InfraStatus(
                service, False, f"docker start {service.container}: {output[-200:]}"
            )
        action = f"restarted container {service.container}"
    else:
        busy = [host for host, _ in service.ports if port_open(host)]
        if busy:
            # Another process owns the ports. Reuse it when it answers like
            # this service (e.g. a developer's own dashboard); never stop it.
            reason = wait_ready(service, timeout=5.0)
            if reason is None:
                return InfraStatus(
                    service,
                    ready=True,
                    detail=f"reusing existing service on port(s) {busy} "
                    "(not managed by verify-live)",
                )
            return InfraStatus(
                service, False, f"host port(s) {busy} in use by another process"
            )
        code, output = _docker(run_args(service), timeout=RUN_TIMEOUT)
        if code != 0:
            return InfraStatus(
                service, False, f"docker run {service.image}: {output[-200:]}"
            )
        action = f"started container {service.container} ({service.image})"

    reason = wait_ready(service, timeout)
    if reason is not None:
        return InfraStatus(service, False, f"{action} but not ready: {reason}", True)
    return InfraStatus(service, True, action, managed=True)


def stop_hint(statuses: Sequence[InfraStatus]) -> str | None:
    """Return the command that removes the containers verify-live manages."""
    names = [s.service.container for s in statuses if s.managed]
    return f"docker rm -f {' '.join(names)}" if names else None
