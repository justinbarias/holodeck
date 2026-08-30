"""CLI command for running a Temporal worker (spec 040, T8).

Implements ``holodeck worker``: loads a ``worker.yaml`` (decision 8), builds
one activity per configured edge node through the T3 factory, and polls a task
queue with an **activities-only** worker. HoloDeck registers no workflows —
the workflow is the user's code, running in their own worker.

Every ``temporalio`` and ``holodeck.temporal`` import lives inside the command
body. The ``holodeck.temporal`` package runs ``require_temporalio()`` at import
time, so a module-scope import here would make ``holodeck --help`` fail on any
installation without the ``temporal`` extra. Deferred, the guard fires only
when someone actually runs the command, and it surfaces as the T1 message
rather than a traceback.

Timeouts and retry policies are deliberately absent from both the config and
this command (decision 10): in Temporal they ride the caller's
``execute_activity`` command, which is what
``holodeck.temporal.models.ActivityParameters`` builds. The only shutdown knob
here is the worker's own grace period for activities already in flight.
"""

from __future__ import annotations

import asyncio
import signal
import sys
from datetime import timedelta
from typing import TYPE_CHECKING

import click

from holodeck.lib.errors import HoloDeckError
from holodeck.lib.logging_config import get_logger

if TYPE_CHECKING:
    from holodeck.temporal.worker_config import WorkerConfig

logger = get_logger(__name__)

# How long the worker lets activities already in flight finish after a
# shutdown signal. An agent turn is a model call: killing it instantly wastes
# the tokens already spent and leaves the workflow waiting for a retry.
GRACEFUL_SHUTDOWN_SECONDS = 30


@click.command()
@click.option(
    "--config",
    "-c",
    "config_path",
    type=str,
    default="worker.yaml",
    help="Path to the worker configuration file (default: worker.yaml)",
)
@click.option(
    "--task-queue",
    type=str,
    default=None,
    help="Task queue to poll, overriding the value in the configuration file",
)
def worker(config_path: str, task_queue: str | None) -> None:
    """Run a Temporal worker that serves this project's agents as activities.

    The worker registers one activity per node in the configuration file,
    named after the node id, and polls the configured task queue until
    interrupted. It registers no workflows: workflow code is yours and runs in
    your own worker.

    Example:

        holodeck worker

        holodeck worker --config deploy/worker.yaml --task-queue hardship

    Options:

        --config / -c       Path to worker.yaml (default: worker.yaml)
        --task-queue        Task queue override for the configured value

    Timeouts and retry policies are not configured here — they belong to the
    workflow that schedules the activity.
    """
    try:
        # Deferred so `holodeck --help` and `holodeck worker --help` work
        # without the `temporal` extra; importing the package runs the guard.
        from holodeck.temporal.worker_config import load_worker_config

        config = load_worker_config(config_path)
        if task_queue is not None:
            config = _with_task_queue(config, task_queue)

        asyncio.run(_run_worker(config))
    except HoloDeckError as exc:
        logger.error(f"Worker failed to start: {exc}", exc_info=True)
        click.secho("Error: Failed to start the Temporal worker", fg="red", err=True)
        click.echo(f"  {exc}", err=True)
        sys.exit(1)
    except KeyboardInterrupt:
        # Reachable when the interrupt lands outside the asyncio loop's own
        # signal handling (e.g. during config load).
        logger.info("Worker interrupted by user (Ctrl+C)")
        click.echo()
        click.secho("Worker stopped.", fg="yellow")
        sys.exit(130)


def _with_task_queue(config: WorkerConfig, task_queue: str) -> WorkerConfig:
    """Return a copy of ``config`` whose task queue is ``task_queue``.

    Rebuilt through validation rather than ``model_copy``, so a blank
    ``--task-queue`` is refused by the same check that guards the file value.

    Args:
        config: The loaded configuration.
        task_queue: The overriding task queue name.

    Returns:
        A configuration with the override applied.

    Raises:
        ConfigError: If the override is blank.
    """
    from pydantic import ValidationError as PydanticValidationError

    from holodeck.lib.errors import ConfigError
    from holodeck.temporal.worker_config import TemporalConnection

    try:
        connection = TemporalConnection.model_validate(
            {**config.temporal.model_dump(), "task_queue": task_queue}
        )
    except PydanticValidationError as exc:
        raise ConfigError("task_queue", f"Invalid --task-queue value: {exc}") from exc
    return config.model_copy(update={"temporal": connection})


async def _run_worker(config: WorkerConfig) -> None:
    """Connect to Temporal and poll until interrupted.

    Args:
        config: The validated worker configuration.
    """
    from temporalio.client import Client
    from temporalio.contrib.opentelemetry import TracingInterceptor
    from temporalio.contrib.pydantic import pydantic_data_converter
    from temporalio.worker import Worker

    from holodeck.temporal.activity import agent_activity

    # Factory direct (decision 14's core layer): an activities-only worker
    # needs no plugin, and every authoring fault the factory settles at bind
    # time surfaces here, before a single poll.
    activities = [agent_activity(node, config.base_dir) for node in config.nodes]

    connection = config.temporal
    client = await Client.connect(
        connection.address,
        namespace=connection.namespace,
        # Decision 15: the typed payload models cross the wire as JSON.
        data_converter=pydantic_data_converter,
        # TracingInterceptor implements both the client and the worker
        # interceptor protocols, and a client interceptor that also satisfies
        # the worker one is picked up by every worker built from this client.
        # Registering it here therefore covers both sides exactly once.
        interceptors=[TracingInterceptor()],
        tls=connection.tls,
    )

    shutdown = asyncio.Event()
    _install_signal_handlers(shutdown)

    click.echo(f"Connected to Temporal at {connection.address}")
    click.echo(f"  namespace:  {connection.namespace}")
    click.echo(f"  task queue: {connection.task_queue}")
    click.echo(f"  TLS:        {'on' if connection.tls else 'off'}")
    for node in config.nodes:
        click.echo(f"  activity:   {node.id}")

    running = Worker(
        client,
        task_queue=connection.task_queue,
        # Activities only — HoloDeck ships no workflows.
        activities=activities,
        graceful_shutdown_timeout=timedelta(seconds=GRACEFUL_SHUTDOWN_SECONDS),
    )
    async with running:
        click.echo("Worker running. Press Ctrl+C to stop.")
        await _wait_for_shutdown(shutdown)

    click.secho("Worker stopped.", fg="yellow")


async def _wait_for_shutdown(shutdown: asyncio.Event) -> None:
    """Block until a termination signal has been observed.

    A one-line seam, deliberately module level: it is the only unbounded wait
    in the command, so it is also the only thing a test of the startup wiring
    has to stand down. The signal handling it waits on is tested directly, via
    :func:`_install_signal_handlers`.

    Args:
        shutdown: The event the signal handlers set.
    """
    await shutdown.wait()


def _install_signal_handlers(shutdown: asyncio.Event) -> None:
    """Arrange for SIGINT and SIGTERM to set ``shutdown``.

    Setting an event rather than raising lets the worker's own graceful
    shutdown run: in-flight activities get their grace period and the process
    exits 0 instead of on an exception.

    Args:
        shutdown: The event to set when a termination signal arrives.
    """
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, shutdown.set)
        except (NotImplementedError, RuntimeError, ValueError):
            # Not every platform or thread can install handlers (Windows, a
            # loop running off the main thread). The synchronous
            # KeyboardInterrupt path in `worker` remains the fallback.
            logger.debug(f"Could not install a handler for {sig!r}")


__all__ = ["GRACEFUL_SHUTDOWN_SECONDS", "worker"]
