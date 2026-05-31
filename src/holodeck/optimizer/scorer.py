"""Score an agent candidate by running the test suite and scalarizing it.

Thin wrapper over the unchanged ``TestExecutor``: it injects the candidate
``Agent`` via ``agent_config=`` and reuses the already-ingested vector stores
(``force_ingest=False``), then collapses the resulting ``TestReport`` into the
scalarized objective.
"""

from holodeck.lib.backends.base import AgentBackend
from holodeck.lib.test_runner.executor import TestExecutor
from holodeck.models.agent import Agent
from holodeck.models.test_result import TestReport
from holodeck.optimizer.loss import scalarize


async def score(
    agent: Agent,
    agent_config_path: str,
    loss_weights: dict[str, float],
    backend: AgentBackend | None = None,
) -> tuple[float, TestReport]:
    """Run the test suite for ``agent`` and return its scalarized score.

    Args:
        agent: The candidate agent to evaluate (injected, not loaded from disk).
        agent_config_path: Path to the original config (used for relative
            resolution of sources/files by the executor).
        loss_weights: Per-metric weights for the scalarized objective.
        backend: Optional pre-built backend to reuse across trials. When ``None``
            the executor auto-selects one via ``BackendSelector``.

    Returns:
        A tuple of ``(scalarized_score, report)``.
    """
    executor = TestExecutor(
        agent_config_path=agent_config_path,
        agent_config=agent,
        force_ingest=False,
        backend=backend,
    )
    report = await executor.execute_tests()
    return scalarize(report, loss_weights), report
