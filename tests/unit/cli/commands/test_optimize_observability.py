"""Unit tests for `holodeck test optimize` observability wiring (T1).

Parity with `holodeck test`: when the agent enables observability the optimize
command must initialize the OTel context (so per-trial GenAI spans flow and the
"context not initialized" warning disappears), open a single ``holodeck.optimize``
root span around the run, and shut the context down afterwards — even on error.
When observability is disabled it must fall back to ``setup_logging`` and never
touch the observability lifecycle.

TDD: written before the implementation; these fail until ``optimize.py`` is wired.
"""

from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from holodeck.models.observability import ObservabilityConfig


def _make_agent(observability: ObservabilityConfig | None) -> MagicMock:
    agent = MagicMock()
    agent.name = "opt-agent"
    agent.observability = observability
    agent.evaluations.optimizer = MagicMock()
    agent.test_cases = [MagicMock()]
    agent.model = MagicMock()
    return agent


def _make_config() -> MagicMock:
    # Real, JSON-serializable values so the root-span attribute builder
    # (json.dumps(loss), len(axes), seed) runs without exploding.
    config = MagicMock()
    config.axes.numeric = []
    config.axes.textual = []
    config.loss = {"numeric": 1.0}
    config.max_cycles = 2
    config.seed = 42
    return config


def _make_result() -> MagicMock:
    result = MagicMock()
    result.baseline_loss = 0.5
    result.best_loss = 0.4
    result.accepted_count = 1
    result.cycles_run = 1
    result.best_agent = MagicMock()
    return result


def _invoke_optimize(
    observability: ObservabilityConfig | None,
    *,
    loop_side_effect: Exception | None = None,
) -> SimpleNamespace:
    """Run the optimize command with all heavy machinery patched out.

    Returns a namespace of the patched mocks plus the CLI result so tests can
    assert on the observability lifecycle without running a real optimization.
    """
    from holodeck.cli.commands.optimize import optimize

    agent = _make_agent(observability)
    config = _make_config()
    result = _make_result()

    m = SimpleNamespace(agent=agent, config=config, result=result)
    runner = CliRunner()
    with ExitStack() as stack:

        def p(target: str) -> MagicMock:
            return stack.enter_context(patch(target))

        m.load = p("holodeck.config.loader.load_agent_with_config")
        m.resolve = p("holodeck.cli.commands.optimize._resolve_config")
        m.build = p("holodeck.cli.commands.optimize._build_proposers")
        m.loop_cls = p("holodeck.cli.commands.optimize.OptimizerLoop")
        m.overlay = p("holodeck.cli.commands.optimize.overlay_axes")
        m.write = p("holodeck.cli.commands.optimize.write_outputs")
        m.tracer = p("holodeck.cli.commands.optimize.get_tracer")
        m.setup_logging = p("holodeck.cli.commands.optimize.setup_logging")
        m.init = p("holodeck.cli.commands.optimize.initialize_observability")
        m.shutdown = p("holodeck.cli.commands.optimize.shutdown_observability")

        m.load.return_value = (agent, MagicMock(), MagicMock())
        m.resolve.return_value = config
        m.build.return_value = (MagicMock(), None)
        m.overlay.return_value = result.best_agent
        m.init.return_value = MagicMock(name="obs_context")

        loop_instance = MagicMock()
        loop_instance.run = AsyncMock(return_value=result, side_effect=loop_side_effect)
        m.loop_cls.return_value = loop_instance

        with runner.isolated_filesystem():
            with open("agent.yaml", "w") as f:
                f.write("name: opt-agent\n")
            m.invoke_result = runner.invoke(optimize, ["agent.yaml"])

    return m


@pytest.mark.unit
class TestOptimizeObservabilityInit:
    """Initialization branch parity with the test command."""

    def test_initializes_observability_when_enabled(self) -> None:
        m = _invoke_optimize(ObservabilityConfig(enabled=True))
        m.init.assert_called_once_with(
            m.agent.observability, "opt-agent", verbose=False, quiet=False
        )

    def test_setup_logging_not_called_when_enabled(self) -> None:
        m = _invoke_optimize(ObservabilityConfig(enabled=True))
        m.setup_logging.assert_not_called()

    def test_setup_logging_called_when_observability_absent(self) -> None:
        m = _invoke_optimize(None)
        m.setup_logging.assert_called()
        m.init.assert_not_called()

    def test_setup_logging_called_when_observability_disabled(self) -> None:
        m = _invoke_optimize(ObservabilityConfig(enabled=False))
        m.setup_logging.assert_called()
        m.init.assert_not_called()


@pytest.mark.unit
class TestOptimizeRootSpan:
    """A single ``holodeck.optimize`` root span wraps the run when enabled."""

    def test_root_span_opened_when_enabled(self) -> None:
        m = _invoke_optimize(ObservabilityConfig(enabled=True))
        m.tracer.assert_called_once()
        start = m.tracer.return_value.start_as_current_span
        start.assert_called_once()
        assert start.call_args.args[0] == "holodeck.optimize"
        attributes = start.call_args.kwargs["attributes"]
        assert attributes["holodeck.optimize.run_id"]
        assert attributes["holodeck.optimize.agent_name"] == "opt-agent"

    def test_no_tracer_when_disabled(self) -> None:
        m = _invoke_optimize(None)
        m.tracer.assert_not_called()


@pytest.mark.unit
class TestOptimizeObservabilityShutdown:
    """The context is always shut down when it was initialized."""

    def test_shutdown_called_after_run(self) -> None:
        m = _invoke_optimize(ObservabilityConfig(enabled=True))
        m.shutdown.assert_called_once_with(m.init.return_value)

    def test_shutdown_called_on_exception(self) -> None:
        m = _invoke_optimize(
            ObservabilityConfig(enabled=True),
            loop_side_effect=RuntimeError("boom"),
        )
        m.shutdown.assert_called_once_with(m.init.return_value)
        assert m.invoke_result.exit_code == 1

    def test_shutdown_not_called_when_disabled(self) -> None:
        m = _invoke_optimize(None)
        m.shutdown.assert_not_called()
