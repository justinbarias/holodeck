"""``holodeck vectorstore``: inspect and migrate retrieval collections.

Spec 042 FR-023 and AC-08. ``inspect`` is a read-only dry run. ``migrate``
rebuilds a tool's collection with the current ingestion pipeline into a new
collection and switches to it; ``rollback`` switches back; ``prune`` deletes
the collection kept for rollback. Only ``prune`` deletes a collection.
"""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

import click

if TYPE_CHECKING:
    from holodeck.lib.vectorstore_migration import InspectReport
    from holodeck.models.agent import Agent
    from holodeck.models.config import ExecutionConfig

T = TypeVar("T")

_AGENT_ARGUMENT = click.argument(
    "agent_config", type=click.Path(exists=True, dir_okay=False), required=True
)


def _load(agent_config: str) -> tuple[Agent, ExecutionConfig, str]:
    from holodeck.config.loader import load_agent_with_config

    agent, execution_config, _loader = load_agent_with_config(agent_config)
    return agent, execution_config, str(Path(agent_config).resolve().parent)


def _run(make: Callable[[], Coroutine[object, object, T]]) -> T:
    """Run a coroutine and turn HoloDeck errors into a clean CLI error."""
    from holodeck.lib.errors import HoloDeckError

    try:
        return asyncio.run(make())
    except (HoloDeckError, FileNotFoundError) as error:
        raise click.ClickException(str(error)) from error


def _confirm(message: str, yes: bool) -> None:
    if yes:
        return
    if not click.confirm(message, default=False):
        click.secho("Aborted.", fg="yellow")
        sys.exit(1)


@click.group(name="vectorstore")
def vectorstore() -> None:
    """Inspect and migrate vector store collections of retrieval tools.

    A collection records the pipeline (embedding model, parser, chunker and
    their settings) that wrote it. When the pipeline changes, HoloDeck
    refuses writes to the old collection; these commands rebuild it.
    """


@vectorstore.command()
@_AGENT_ARGUMENT
@click.option("--tool", "tool_name", default=None, help="Only this tool.")
@click.option("--json", "as_json", is_flag=True, help="Print JSON.")
def inspect(agent_config: str, tool_name: str | None, as_json: bool) -> None:
    """Report collection state and migration work. Read-only dry run.

    Downloads nothing, writes no file, creates no collection or manifest
    entry and calls no model. Chunks are counted by parsing and splitting
    local sources; remote sources are not estimated.
    """
    from holodeck.lib.vectorstore_migration import inspect_tool, retrieval_tools

    agent, execution_config, base_dir = _load(agent_config)

    async def run() -> list[InspectReport]:
        return [
            await inspect_tool(
                agent, tool, base_dir=base_dir, execution_config=execution_config
            )
            for tool in retrieval_tools(agent, tool_name)
        ]

    reports = _run(run)
    if as_json:
        click.echo(json.dumps({"tools": [r.to_dict() for r in reports]}, indent=2))
        return
    if not reports:
        click.echo("Agent has no vectorstore or hierarchical_document tools.")
    for report in reports:
        _print_report(report)


def _print_report(report: InspectReport) -> None:
    def show(value: object) -> str:
        return "unknown" if value is None else str(value)

    click.secho(f"Tool {report.tool} ({report.tool_type})", bold=True)
    click.echo(f"  Provider:            {report.provider}")
    click.echo(f"  Logical collection:  {report.logical_collection}")
    click.echo(f"  Physical collection: {report.physical_collection}")
    click.echo(f"  Exists:              {'yes' if report.exists else 'no'}")
    click.echo(f"  Records:             {show(report.record_count)}")
    if report.manifest_supported:
        click.echo(f"  Manifest state:      {report.manifest_state}")
    else:
        click.echo("  Manifest state:      not supported by this provider")
    click.echo(f"  Stored fingerprint:  {show(report.stored_fingerprint)}")
    click.echo(f"  Current fingerprint: {report.current_fingerprint}")
    click.echo(f"  Compatible:          {report.compatible}")
    for name, old, new in report.differences:
        click.echo(f"    {name}: {json.dumps(old)} -> {json.dumps(new)}")
    if report.previous_collection:
        click.echo(f"  Kept for rollback:   {report.previous_collection}")
    if report.migration_target:
        click.echo(f"  Migration target:    {report.migration_target}")
    click.echo(
        "  Needs re-embedding:  " + ("yes" if report.needs_reembedding else "no")
    )
    click.echo(
        f"  Migration work:      {report.source_files} source files, "
        f"{report.chunks} chunks, ~{report.estimated_tokens} tokens"
    )
    for uri in report.remote_sources:
        click.echo(f"  Remote source: not estimated ({uri})")
    click.echo(f"  {report.estimate_note}")
    click.echo()


@vectorstore.command()
@_AGENT_ARGUMENT
@click.option("--tool", "tool_name", required=True, help="Tool to migrate.")
@click.option("--yes", is_flag=True, help="Do not ask for confirmation.")
@click.option(
    "--restart",
    is_flag=True,
    help="Abandon an unfinished migration that targets a different pipeline.",
)
@click.option(
    "--drop-unlisted-sources",
    is_flag=True,
    help=(
        "Switch even if the old collection holds sources that no tool of this "
        "agent ingests; those records are not carried over."
    ),
)
def migrate(
    agent_config: str,
    tool_name: str,
    yes: bool,
    restart: bool,
    drop_unlisted_sources: bool,
) -> None:
    """Rebuild a tool's collection with the current pipeline and switch to it.

    --tool selects the collection. Every retrieval tool of the agent that uses
    the same collection is rebuilt with it. Re-embeds every source into a new
    collection, verifies it and then points the tools at it. The old
    collection is kept for rollback. An interrupted migration resumes where it
    stopped.
    """
    from holodeck.lib.vectorstore_migration import (
        migrate_tool,
        plan_migration,
        retrieval_tools,
    )

    agent, execution_config, base_dir = _load(agent_config)
    (tool,) = _run_selection(lambda: retrieval_tools(agent, tool_name))
    plan = _run(
        lambda: plan_migration(
            agent, tool, base_dir=base_dir, execution_config=execution_config
        )
    )
    click.echo(
        f"Migrate collection '{plan.logical_collection}' ({plan.provider}): "
        f"re-embed the sources of every tool that uses it into a new collection "
        f"with the current pipeline, verify it, then switch the tools to it. "
        f"The current collection is kept for rollback. This calls the "
        f"embedding model."
    )
    click.echo(f"  Tools rebuilt together: {', '.join(plan.tools)}")
    _confirm("Continue?", yes)
    result = _run(
        lambda: migrate_tool(
            agent,
            tool,
            base_dir=base_dir,
            execution_config=execution_config,
            restart=restart,
            drop_unlisted_sources=drop_unlisted_sources,
        )
    )
    if result.skipped:
        click.echo(result.message)
        return
    click.secho("Migration complete", fg="green", bold=True)
    click.echo(f"  Active collection: {result.active_collection}")
    click.echo(f"  Kept for rollback: {result.previous_collection}")
    click.echo(f"  Tools:             {', '.join(result.tools)}")
    click.echo(
        f"  Records: {result.record_count} from {result.sources} sources "
        f"({result.embedded_sources} embedded in this run)"
    )
    if result.dropped_sources:
        click.secho(
            f"  Not carried over (--drop-unlisted-sources): "
            f"{len(result.dropped_sources)} sources",
            fg="yellow",
        )


def _run_selection(select: Callable[[], list[T]]) -> list[T]:
    from holodeck.lib.errors import HoloDeckError

    try:
        return select()
    except HoloDeckError as error:
        raise click.ClickException(str(error)) from error


@vectorstore.command()
@_AGENT_ARGUMENT
@click.option("--tool", "tool_name", required=True, help="Tool to roll back.")
@click.option("--yes", is_flag=True, help="Do not ask for confirmation.")
def rollback(agent_config: str, tool_name: str, yes: bool) -> None:
    """Switch a tool back to the collection kept by the last migration."""
    from holodeck.lib.vectorstore_migration import retrieval_tools, rollback_tool

    agent, execution_config, base_dir = _load(agent_config)
    (tool,) = _run_selection(lambda: retrieval_tools(agent, tool_name))
    click.echo(
        f"Roll back tool '{tool.name}' to the collection kept by its last "
        f"migration. The newer collection is kept."
    )
    _confirm("Continue?", yes)
    result = _run(
        lambda: rollback_tool(
            agent, tool, base_dir=base_dir, execution_config=execution_config
        )
    )
    click.secho("Rollback complete", fg="green", bold=True)
    click.echo(f"  Active collection: {result.active_collection}")
    click.echo(f"  Kept collection:   {result.previous_collection}")


@vectorstore.command()
@_AGENT_ARGUMENT
@click.option("--tool", "tool_name", required=True, help="Tool to prune.")
@click.option("--yes", is_flag=True, help="Do not ask for confirmation.")
def prune(agent_config: str, tool_name: str, yes: bool) -> None:
    """Delete the collection kept for rollback. This cannot be undone."""
    from holodeck.lib.vectorstore_migration import prune_tool, retrieval_tools

    agent, execution_config, base_dir = _load(agent_config)
    (tool,) = _run_selection(lambda: retrieval_tools(agent, tool_name))
    click.echo(
        f"Permanently delete the collection kept for rollback of tool "
        f"'{tool.name}'. The active collection is not touched."
    )
    _confirm("Delete it?", yes)
    result = _run(
        lambda: prune_tool(
            agent, tool, base_dir=base_dir, execution_config=execution_config
        )
    )
    click.secho("Prune complete", fg="green", bold=True)
    click.echo(f"  Deleted collection: {result.deleted_collection}")
    click.echo(f"  Active collection:  {result.active_collection}")
