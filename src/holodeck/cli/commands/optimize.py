"""CLI command for `holodeck test optimize`.

Runs the compounding coordinate-descent optimizer over an agent's declared
numeric/textual axes, streaming per-trial scores and writing the run artifacts
(``best.yaml``, ``trials.jsonl``, ``report.md``). The original ``agent.yaml`` is
never mutated.
"""

import asyncio
import json
import sys
import uuid
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import click

from holodeck.lib.errors import OptimizerError
from holodeck.lib.logging_config import get_logger, setup_logging
from holodeck.lib.observability import (
    ObservabilityContext,
    get_tracer,
    initialize_observability,
    shutdown_observability,
)
from holodeck.models.agent import Agent
from holodeck.optimizer.config import OptimizerConfig
from holodeck.optimizer.loop import OptimizerLoop
from holodeck.optimizer.models import OptimizationResult, TrialRecord
from holodeck.optimizer.mutator import overlay_axes
from holodeck.optimizer.output import write_outputs
from holodeck.optimizer.progress import (
    ErrorEvent,
    JsonlEmitter,
    NullEmitter,
    ProgressEmitter,
    RunArtifacts,
    RunCompleted,
)
from holodeck.optimizer.proposers.numeric import NumericProposer
from holodeck.optimizer.proposers.textual import TextualProposer, load_critic_applier
from holodeck.optimizer.scorer import score

logger = get_logger(__name__)


def _resolve_config(
    base: OptimizerConfig,
    *,
    max_cycles: int | None,
    numeric_max_trials: int | None,
    numeric_patience: int | None,
    textual_max_trials: int | None,
    textual_patience: int | None,
    seed: int | None,
) -> OptimizerConfig:
    """Apply CLI overrides onto the YAML optimizer config."""
    updates: dict[str, object] = {}
    if max_cycles is not None:
        updates["max_cycles"] = max_cycles
    if seed is not None:
        updates["seed"] = seed

    numeric_updates = {
        k: v
        for k, v in (
            ("max_trials", numeric_max_trials),
            ("patience", numeric_patience),
        )
        if v is not None
    }
    if numeric_updates:
        updates["numeric_phase"] = base.numeric_phase.model_copy(update=numeric_updates)

    textual_updates = {
        k: v
        for k, v in (
            ("max_trials", textual_max_trials),
            ("patience", textual_patience),
        )
        if v is not None
    }
    if textual_updates:
        updates["textual_phase"] = base.textual_phase.model_copy(update=textual_updates)

    return base.model_copy(update=updates)


def _build_proposers(
    agent: Agent, config: OptimizerConfig
) -> tuple[NumericProposer | None, TextualProposer | None]:
    """Construct the numeric and textual proposers declared by the config."""
    numeric = (
        NumericProposer(axes=config.axes.numeric, seed=config.seed)
        if config.axes.numeric
        else None
    )
    textual = None
    if config.axes.textual:
        critic, applier = load_critic_applier(agent.model)
        textual = TextualProposer(
            axes=config.axes.textual, critic_agent=critic, applier_agent=applier
        )
    return numeric, textual


def _root_span_attributes(
    agent: Agent, config: OptimizerConfig, run_id: str
) -> dict[str, Any]:
    """Build OTel attributes for the ``holodeck.optimize`` root span.

    Only primitive values (OTel attributes cannot be ``None`` or nested), and
    no prompt text or resolved secrets — the loss weights are emitted as a
    compact JSON string, axes as counts.
    """
    attributes: dict[str, Any] = {
        "holodeck.optimize.run_id": run_id,
        "holodeck.optimize.agent_name": agent.name,
        "holodeck.optimize.max_cycles": config.max_cycles,
        "holodeck.optimize.axes.numeric": len(config.axes.numeric),
        "holodeck.optimize.axes.textual": len(config.axes.textual),
        "holodeck.optimize.loss": json.dumps(config.loss, sort_keys=True),
    }
    if config.seed is not None:
        attributes["holodeck.optimize.seed"] = config.seed
    return attributes


@click.command("optimize")
@click.argument("agent_config", type=click.Path(exists=True), default="agent.yaml")
@click.option(
    "--max-cycles",
    type=click.IntRange(min=1),
    default=None,
    help="Maximum numeric→textual cycles.",
)
@click.option(
    "--numeric-max-trials",
    type=click.IntRange(min=1),
    default=None,
    help="Hard cap on numeric-phase trials.",
)
@click.option(
    "--numeric-patience",
    type=click.IntRange(min=1),
    default=None,
    help="Numeric-phase consecutive non-accepts before stopping.",
)
@click.option(
    "--textual-max-trials",
    type=click.IntRange(min=1),
    default=None,
    help="Hard cap on textual-phase trials.",
)
@click.option(
    "--textual-patience",
    type=click.IntRange(min=1),
    default=None,
    help="Textual-phase consecutive non-accepts before stopping.",
)
@click.option("--seed", type=int, default=None, help="Seed for the numeric study.")
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(),
    default="results/optimizer",
    help="Base directory for optimizer run artifacts.",
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose debug output.")
@click.option("--quiet", "-q", is_flag=True, help="Suppress per-trial streaming.")
@click.option(
    "--progress",
    type=click.Choice(["plain", "json"]),
    default="plain",
    show_default=True,
    help=(
        "Progress output mode. 'plain' streams human-readable per-trial lines to "
        "stdout (unchanged). 'json' emits a versioned NDJSON event stream to stdout "
        "(one object per line, flushed per event) and routes all human and library "
        "logs to stderr, so a subprocess can read stdout as pure events. See the "
        "schema at schemas/optimize-progress.schema.json."
    ),
)
def optimize(
    agent_config: str,
    max_cycles: int | None,
    numeric_max_trials: int | None,
    numeric_patience: int | None,
    textual_max_trials: int | None,
    textual_patience: int | None,
    seed: int | None,
    output_dir: str,
    verbose: bool,
    quiet: bool,
    progress: str,
) -> None:
    """Optimize an agent's instructions and hyperparameters against its tests.

    AGENT_CONFIG is the path to the agent.yaml configuration file. The original
    file is never modified; the best candidate is written to
    ``<output-dir>/<run-id>/best.yaml``.
    """
    # Set when observability is enabled; gates the root span + shutdown.
    obs_context: ObservabilityContext | None = None
    effective_quiet = quiet and not verbose

    # Under `--progress json`, stdout is the machine channel (pure NDJSON) and all
    # human/library output is routed to stderr; `plain` keeps today's behavior with a
    # no-op emitter. Created before the try so the except blocks can emit a fatal error.
    json_progress = progress == "json"
    emitter: ProgressEmitter = (
        JsonlEmitter(sys.stdout) if json_progress else NullEmitter()
    )

    try:
        from holodeck.config.loader import load_agent_with_config

        agent, _resolved, loader = load_agent_with_config(agent_config)

        # Observability parity with `holodeck test`: when the agent enables
        # observability, OTel owns logging and each trial's eval emits GenAI
        # spans/metrics; otherwise fall back to traditional logging. Done after
        # the load so the branch can read agent.observability.
        if agent.observability and agent.observability.enabled:
            obs_context = initialize_observability(
                agent.observability, agent.name, verbose=verbose, quiet=effective_quiet
            )
        else:
            setup_logging(verbose=verbose, quiet=effective_quiet)

        if agent.evaluations is None or agent.evaluations.optimizer is None:
            raise OptimizerError(
                "No optimizer configuration found. Add an "
                "`evaluations.optimizer` block to the agent.yaml."
            )
        if not agent.test_cases:
            raise OptimizerError(
                "Optimization requires at least one test case in the agent "
                "configuration."
            )

        config = _resolve_config(
            agent.evaluations.optimizer,
            max_cycles=max_cycles,
            numeric_max_trials=numeric_max_trials,
            numeric_patience=numeric_patience,
            textual_max_trials=textual_max_trials,
            textual_patience=textual_patience,
            seed=seed,
        )

        numeric, textual = _build_proposers(agent, config)
        if numeric is None and textual is None:
            raise OptimizerError(
                "No numeric or textual axes declared under "
                "`evaluations.optimizer.axes`."
            )

        weights = config.loss

        async def scorer(candidate: Agent) -> tuple[float, object]:
            return await score(candidate, agent_config, weights)

        def on_trial(trial: TrialRecord) -> None:
            if effective_quiet:
                return
            if trial.error:
                status = f"skipped ({trial.error})"
            elif trial.accepted:
                status = "accepted ✓"
            else:
                status = "rejected"
            click.echo(
                f"  trial {trial.trial_id} [{trial.phase}] "
                f"loss={trial.loss:.4f} (best {trial.baseline_loss:.4f}) "
                f"— {status}"
            )

        started_at = datetime.now(timezone.utc)
        run_id = started_at.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]

        click.echo(f"Optimizing '{agent.name}' (run {run_id})…", err=json_progress)
        loop = OptimizerLoop(
            original_agent=agent,
            scorer=scorer,  # type: ignore[arg-type]
            config=config,
            numeric_proposer=numeric,
            textual_proposer=textual,
            run_id=run_id,
            # In json mode the trial events replace human stdout streaming.
            progress_callback=None if json_progress else on_trial,
            emitter=emitter,
            started_at=started_at,
        )

        async def _run() -> OptimizationResult:
            # Root span for the whole run; per-trial GenAI spans from each
            # candidate's eval nest underneath it. No-op when disabled.
            if obs_context is not None:
                span_ctx: Any = get_tracer(__name__).start_as_current_span(
                    "holodeck.optimize",
                    attributes=_root_span_attributes(agent, config, run_id),
                )
            else:
                span_ctx = nullcontext()
            with span_ctx:
                return await loop.run()

        result: OptimizationResult = asyncio.run(_run())

        # Rebuild best.yaml on the unsubstituted source so ${VAR} secret
        # placeholders survive instead of leaking env-resolved credentials.
        # Only the tuned axes are carried over from the resolved best agent.
        try:
            template = loader.load_agent_yaml(agent_config, substitute_env=False)
            axis_paths = [a.path for a in config.axes.numeric] + [
                a.path for a in config.axes.textual
            ]
            result.best_agent = overlay_axes(template, result.best_agent, axis_paths)
        except Exception:  # noqa: BLE001 — never regress; fall back with a warning.
            logger.warning(
                "Could not rebuild best.yaml from the unsubstituted config; it may "
                "contain resolved secrets — review before sharing.",
                exc_info=True,
            )

        run_dir = write_outputs(result, Path(output_dir))

        # Closing event: artifact paths are only known now that they are written.
        emitter.emit(
            RunCompleted(
                run_id=result.run_id,
                baseline_loss=result.baseline_loss,
                best_loss=result.best_loss,
                accepted=result.accepted_count,
                cycles=result.cycles_run,
                artifacts=RunArtifacts(
                    best_yaml=str(run_dir / "best.yaml"),
                    trials_jsonl=str(run_dir / "trials.jsonl"),
                    report_md=str(run_dir / "report.md"),
                ),
            )
        )

        click.echo(
            f"\nBaseline loss {result.baseline_loss:.4f} → best "
            f"{result.best_loss:.4f} "
            f"({result.accepted_count} accepted over {result.cycles_run} cycles).",
            err=json_progress,
        )
        click.echo(f"Artifacts written to {run_dir}", err=json_progress)

    except OptimizerError as exc:
        emitter.emit(ErrorEvent(message=str(exc), fatal=True))
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)
    except Exception as exc:  # noqa: BLE001
        emitter.emit(ErrorEvent(message=str(exc), fatal=True))
        logger.exception("Optimization failed")
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)
    finally:
        if obs_context is not None:
            shutdown_observability(obs_context)
