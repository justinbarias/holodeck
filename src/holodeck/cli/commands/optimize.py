"""CLI command for `holodeck test optimize`.

Runs the compounding coordinate-descent optimizer over an agent's declared
numeric/textual axes, streaming per-trial scores and writing the run artifacts
(``best.yaml``, ``trials.jsonl``, ``report.md``). The original ``agent.yaml`` is
never mutated.
"""

import asyncio
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import click

from holodeck.lib.errors import OptimizerError
from holodeck.lib.logging_config import get_logger, setup_logging
from holodeck.models.agent import Agent
from holodeck.optimizer.config import OptimizerConfig
from holodeck.optimizer.loop import OptimizerLoop
from holodeck.optimizer.models import OptimizationResult, TrialRecord
from holodeck.optimizer.output import write_outputs
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
) -> None:
    """Optimize an agent's instructions and hyperparameters against its tests.

    AGENT_CONFIG is the path to the agent.yaml configuration file. The original
    file is never modified; the best candidate is written to
    ``<output-dir>/<run-id>/best.yaml``.
    """
    effective_quiet = quiet and not verbose
    setup_logging(verbose=verbose, quiet=effective_quiet)

    try:
        from holodeck.config.loader import load_agent_with_config

        agent, _resolved, _loader = load_agent_with_config(agent_config)

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
                f"score={trial.score:.4f} (best {trial.baseline_score:.4f}) "
                f"— {status}"
            )

        run_id = (
            datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            + "-"
            + uuid.uuid4().hex[:6]
        )

        click.echo(f"Optimizing '{agent.name}' (run {run_id})…")
        loop = OptimizerLoop(
            original_agent=agent,
            scorer=scorer,  # type: ignore[arg-type]
            config=config,
            numeric_proposer=numeric,
            textual_proposer=textual,
            run_id=run_id,
            progress_callback=on_trial,
        )
        result: OptimizationResult = asyncio.run(loop.run())

        run_dir = write_outputs(result, Path(output_dir))

        click.echo(
            f"\nBaseline {result.baseline_score:.4f} → best "
            f"{result.best_score:.4f} "
            f"({result.accepted_count} accepted over {result.cycles_run} cycles)."
        )
        click.echo(f"Artifacts written to {run_dir}")

    except OptimizerError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Optimization failed")
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)
