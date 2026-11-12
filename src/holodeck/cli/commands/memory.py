"""``holodeck memory``: build and inspect an agent's temporal memory (spec 043).

``ingest`` writes episodes (CLI-01, ING-08), ``status`` reports the backend
(CLI-02) and ``purge`` clears a namespace (CLI-03). The memory stack is
imported inside each command, so ``holodeck --help`` does not load it.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

import click

if TYPE_CHECKING:
    from holodeck.lib.memory.service import IngestRunReport
    from holodeck.models.agent import Agent
    from holodeck.models.config import ExecutionConfig

T = TypeVar("T")

_AGENT_ARGUMENT = click.argument(
    "agent_config", type=click.Path(exists=True, dir_okay=False), required=True
)


def _load(agent_config: str) -> tuple[Agent, ExecutionConfig, str]:
    from holodeck.config.loader import load_agent_with_config
    from holodeck.lib.errors import HoloDeckError

    try:
        agent, execution_config, _loader = load_agent_with_config(agent_config)
    except HoloDeckError as error:
        # A YAML that does not load prints one line, like every other error here.
        raise click.ClickException(str(error)) from error
    return agent, execution_config, str(Path(agent_config).resolve().parent)


def _run(make: Callable[[], Coroutine[object, object, T]]) -> T:
    """Run a coroutine; print a memory or HoloDeck error as one line."""
    from holodeck.lib.errors import HoloDeckError, UserFixableError
    from holodeck.lib.memory.errors import TemporalMemoryError

    try:
        return asyncio.run(make())
    except TemporalMemoryError as error:
        raise click.ClickException(str(error)) from error
    except UserFixableError:
        raise  # the CLI group prints it: one line, exit 2
    except (HoloDeckError, FileNotFoundError) as error:
        raise click.ClickException(str(error)) from error


def _format_estimate(seconds: float) -> str:
    minutes, rest = divmod(round(seconds), 60)
    return f"{minutes} min {rest} s"


def _print_report(report: IngestRunReport) -> None:
    def tokens(value: int | None) -> str:
        return "not reported" if value is None else str(value)

    click.echo(f"Documents:          {report.documents}")
    click.echo(f"Episodes new:       {report.episodes_new}")
    click.echo(f"Episodes skipped:   {report.episodes_skipped}")
    click.echo(f"Episodes written:   {len(report.written)}")
    click.echo(f"LLM calls:          {report.llm_calls}")
    click.echo(f"Tokens in:          {tokens(report.input_tokens)}")
    click.echo(f"Tokens out:         {tokens(report.output_tokens)}")
    click.echo(f"Duration:           {report.duration_seconds:.1f} s")
    if report.dry_run:
        estimate = (
            "no estimate for this model"
            if report.estimated_seconds is None
            else _format_estimate(report.estimated_seconds)
        )
        click.echo(f"Estimated duration: {estimate}")


@click.group(name="memory")
def memory() -> None:
    """Build and inspect the temporal memory of an agent.

    The memory reads the chunks of the agent's hierarchical_document source
    tools and extracts dated facts into a graph store.
    """


@memory.command()
@_AGENT_ARGUMENT
@click.option(
    "--tool",
    "tools",
    multiple=True,
    help="Only this source tool. Repeat for several.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help=(
        "Count the new episodes from the source files on disk. Make no model "
        "call and no vector store write, and write no episode to the graph."
    ),
)
def ingest(agent_config: str, tools: tuple[str, ...], dry_run: bool) -> None:
    """Extract facts from the source tools into the graph.

    Source tools ingest their own documents first, except in a dry run.
    Episodes already in the graph are skipped, so a repeated run is cheap.
    """
    from holodeck.tools.temporal_memory import open_memory

    agent, execution_config, base_dir = _load(agent_config)

    async def run() -> IngestRunReport:
        async with open_memory(
            agent,
            base_dir=base_dir,
            execution_config=execution_config,
            preview_sources=dry_run,
        ) as session:
            return await session.memory.ingest(
                dry_run=dry_run, source_tools=list(tools) or None
            )

    report = _run(run)
    _print_report(report)
    if report.failed is not None:
        click.echo(f"Failed episode: {report.failed}", err=True)
        click.echo(f"Error: {report.error}", err=True)
        raise SystemExit(1)


@memory.command()
@_AGENT_ARGUMENT
def status(agent_config: str) -> None:
    """Show the backend, namespace, episode count and capabilities."""
    from holodeck.tools.temporal_memory import open_memory

    agent, execution_config, base_dir = _load(agent_config)

    async def run() -> tuple[str, str, int, list[str]]:
        async with open_memory(
            agent,
            base_dir=base_dir,
            execution_config=execution_config,
            initialize_sources=False,
        ) as session:
            namespace = session.config.resolved_namespace
            count = await session.backend.episode_count(namespace)
            capabilities = sorted(
                operation.value
                for operation in session.backend.capabilities().supported
            )
            return session.backend.name, namespace, count, capabilities

    backend, namespace, count, capabilities = _run(run)
    click.echo(f"Backend:   {backend}")
    click.echo(f"Namespace: {namespace}")
    click.echo(f"Episodes:  {count}")
    click.echo("Capabilities:")
    for capability in capabilities:
        click.echo(f"  {capability}")


@memory.command()
@_AGENT_ARGUMENT
@click.option("--yes", is_flag=True, help="Do not ask for confirmation.")
def purge(agent_config: str, yes: bool) -> None:
    """Remove every episode, entity and fact in the agent's namespace."""
    from holodeck.lib.memory.errors import MemoryConfigError
    from holodeck.tools.temporal_memory import find_memory_config, open_memory

    agent, execution_config, base_dir = _load(agent_config)
    try:
        config = find_memory_config(agent)
    except MemoryConfigError as error:
        raise click.ClickException(str(error)) from error
    namespace = config.resolved_namespace
    provider = config.backend.provider
    if not yes and not click.confirm(
        f"Remove every episode, entity and fact in namespace '{namespace}' "
        f"on backend '{provider}'? This cannot be undone",
        default=False,
    ):
        click.echo("Aborted. Nothing was purged.")
        return

    async def run() -> tuple[str, int]:
        async with open_memory(
            agent,
            base_dir=base_dir,
            execution_config=execution_config,
            initialize_sources=False,
        ) as session:
            count = await session.backend.episode_count(namespace)
            await session.backend.purge(namespace)
            return session.backend.name, count

    backend, count = _run(run)
    click.echo(
        f"Cleared namespace '{namespace}' on backend '{backend}' ({count} episodes)."
    )
