"""Tool initialization job manager.

Manages background asyncio tasks for initializing tools that require
data ingestion (vectorstore, hierarchical_document). Provides job lifecycle
management with conflict detection, capacity limits, and graceful shutdown.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

from opentelemetry.trace import StatusCode

from holodeck.lib.observability import get_tracer
from holodeck.lib.source_resolver import SourceResolver, sanitize_error_detail
from holodeck.lib.tool_initializer import ToolInitializerError, initialize_single_tool
from holodeck.serve.models import InitJobProgress, InitJobState, ToolInfoResponse

if TYPE_CHECKING:
    from holodeck.models.agent import Agent

logger = logging.getLogger(__name__)

# Tool types that support initialization (data ingestion).
INITIALIZABLE_TYPES: frozenset[str] = frozenset(
    {"vectorstore", "hierarchical_document"}
)

_DEFAULT_MAX_CONCURRENT = 4


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class ToolNotFoundError(ToolInitializerError):
    """Raised when a requested tool name does not exist in the agent config."""


class ToolNotInitializableError(ToolInitializerError):
    """Raised when a tool exists but does not support initialization."""


class InitJobConflictError(ToolInitializerError):
    """Raised when an init job is already active for the requested tool."""


class InitJobCapacityError(ToolInitializerError):
    """Raised when the maximum number of concurrent init jobs is reached."""


class InitManagerShuttingDownError(ToolInitializerError):
    """Raised when a job is requested after shutdown has been initiated."""


# ---------------------------------------------------------------------------
# InitJob dataclass
# ---------------------------------------------------------------------------


@dataclass
class InitJob:
    """Mutable internal state for a single tool initialization job.

    This is intentionally a dataclass (not Pydantic) because it represents
    mutable in-process state that is updated by the background task.

    Attributes:
        tool_name: Name of the tool being initialized.
        created_at: Timestamp when the job was created.
        state: Current lifecycle state.
        started_at: Timestamp when processing began.
        completed_at: Timestamp when processing finished.
        message: Human-readable status message.
        error_detail: Sanitized error detail on failure.
        progress: Progress tracking during ingestion.
        force: Whether force re-initialization was requested.
    """

    tool_name: str
    created_at: datetime
    state: InitJobState = field(default=InitJobState.PENDING)
    started_at: datetime | None = field(default=None)
    completed_at: datetime | None = field(default=None)
    message: str | None = field(default=None)
    error_detail: str | None = field(default=None)
    progress: InitJobProgress | None = field(default=None)
    force: bool = field(default=False)


# ---------------------------------------------------------------------------
# ToolInitManager
# ---------------------------------------------------------------------------


class ToolInitManager:
    """Manages background tool initialization jobs.

    Provides start, query, and shutdown semantics for tool init tasks.
    All public mutation methods are synchronous (no ``await``) because the
    asyncio single-threaded model guarantees atomicity between ``await``
    suspension points.

    Args:
        agent: The loaded agent configuration.
        max_concurrent: Maximum number of concurrent init jobs.
    """

    def __init__(
        self,
        agent: Agent,
        max_concurrent: int = _DEFAULT_MAX_CONCURRENT,
    ) -> None:
        self._agent = agent
        self._max_concurrent = max_concurrent
        self._jobs: dict[str, InitJob] = {}
        self._tasks: dict[str, asyncio.Task[Any]] = {}
        self._active_count: int = 0
        self._shutting_down: bool = False

    # -- public interface ---------------------------------------------------

    def start_init_job(self, tool_name: str, *, force: bool = False) -> InitJob:
        """Create and schedule a new tool initialization job.

        Args:
            tool_name: Name of the tool to initialize.
            force: If ``True``, force re-ingestion even if data exists.

        Returns:
            The newly created :class:`InitJob`.

        Raises:
            InitManagerShuttingDownError: Manager is shutting down.
            ToolNotFoundError: Tool name not found in agent config.
            ToolNotInitializableError: Tool type does not support init.
            InitJobConflictError: An active job already exists for this tool.
            InitJobCapacityError: Concurrent job limit reached.
        """
        if self._shutting_down:
            raise InitManagerShuttingDownError(
                "Cannot start init jobs: manager is shutting down"
            )

        # Locate tool config by name.
        tool_config = self._find_tool(tool_name)

        # Validate tool type supports initialization.
        if tool_config.type not in INITIALIZABLE_TYPES:
            raise ToolNotInitializableError(
                f"Tool '{tool_name}' (type={tool_config.type}) does not "
                "support initialization. Only vectorstore and "
                "hierarchical_document tools can be initialized."
            )

        # Check for existing active job.
        existing = self._jobs.get(tool_name)
        if existing is not None and existing.state in (
            InitJobState.PENDING,
            InitJobState.IN_PROGRESS,
        ):
            raise InitJobConflictError(
                f"Tool '{tool_name}' already has an active init job "
                f"(state={existing.state.value})"
            )

        # Capacity check.
        if self._active_count >= self._max_concurrent:
            raise InitJobCapacityError(
                f"Maximum concurrent init jobs ({self._max_concurrent}) reached"
            )

        # Create the job and schedule the background task.
        job = InitJob(
            tool_name=tool_name,
            created_at=datetime.now(),
            force=force,
        )
        self._jobs[tool_name] = job
        self._active_count += 1
        task = asyncio.create_task(self._run_init_job(job))
        self._tasks[tool_name] = task

        tracer = get_tracer(__name__)
        with tracer.start_as_current_span("holodeck.serve.tool_init.start") as span:
            span.set_attribute("tool_init.job.tool_name", tool_name)
            span.set_attribute("tool_init.job.state", job.state.value)
            span.set_attribute("tool_init.job.force", force)

        logger.info("Scheduled init job for tool '%s' (force=%s)", tool_name, force)
        return job

    def get_job(self, tool_name: str) -> InitJob | None:
        """Return the current init job for *tool_name*, or ``None``.

        Args:
            tool_name: Name of the tool to look up.

        Returns:
            The :class:`InitJob` if one exists, otherwise ``None``.
        """
        return self._jobs.get(tool_name)

    def get_all_tool_statuses(self) -> list[ToolInfoResponse]:
        """Return init status for every configured tool.

        Iterates all tools in the agent configuration, derives whether each
        supports initialization, and cross-references any existing init jobs
        for the current state.

        Returns:
            A list of :class:`ToolInfoResponse` for each configured tool.
        """
        statuses: list[ToolInfoResponse] = []
        for tool in self._agent.tools or []:
            supports_init = tool.type in INITIALIZABLE_TYPES
            job = self._jobs.get(tool.name)
            statuses.append(
                ToolInfoResponse(
                    name=tool.name,
                    type=tool.type,
                    supports_init=supports_init,
                    init_status=job.state if job is not None else None,
                )
            )
        return statuses

    async def shutdown(self) -> None:
        """Gracefully shut down the manager.

        Cancels all running tasks, marks interrupted jobs as failed,
        and prevents new jobs from being started. Safe to call multiple
        times (idempotent).
        """
        self._shutting_down = True

        # Collect tasks that are still running.
        active_tasks: list[asyncio.Task[Any]] = [
            t for t in self._tasks.values() if not t.done()
        ]

        for task in active_tasks:
            task.cancel()

        if active_tasks:
            await asyncio.gather(*active_tasks, return_exceptions=True)

        # Mark any non-terminal jobs as failed.
        for job in self._jobs.values():
            if job.state in (InitJobState.PENDING, InitJobState.IN_PROGRESS):
                job.state = InitJobState.FAILED
                job.completed_at = datetime.now()
                job.error_detail = "Manager shutdown"

        logger.info("ToolInitManager shut down (%d tasks cancelled)", len(active_tasks))

    # -- internals ----------------------------------------------------------

    def _find_tool(self, tool_name: str) -> Any:
        """Look up a tool config by name.

        Args:
            tool_name: The tool name to search for.

        Returns:
            The matching tool configuration object.

        Raises:
            ToolNotFoundError: If the tool is not found.
        """
        tools = self._agent.tools
        if not tools:
            raise ToolNotFoundError(
                f"Tool '{tool_name}' not found (agent has no tools)"
            )
        for tool in tools:
            if tool.name == tool_name:
                return tool
        raise ToolNotFoundError(f"Tool '{tool_name}' not found in agent configuration")

    async def _run_init_job(self, job: InitJob) -> None:
        """Execute the tool initialization job.

        Transitions: pending -> in_progress -> completed/failed.
        Uses SourceResolver for URI resolution and initialize_single_tool
        for the actual ingestion work.

        Args:
            job: The init job to run.
        """
        tracer = get_tracer(__name__)
        with tracer.start_as_current_span("holodeck.serve.tool_init.progress") as span:
            span.set_attribute("tool_init.job.tool_name", job.tool_name)
            span.set_attribute("tool_init.job.force", job.force)

            job.state = InitJobState.IN_PROGRESS
            job.started_at = datetime.now()

            def _progress_callback(processed: int, total: int | None) -> None:
                job.progress = InitJobProgress(
                    documents_processed=processed,
                    total_documents=total,
                )

            try:
                # Find tool config to get source
                tool_config = None
                for t in self._agent.tools or []:
                    if t.name == job.tool_name:
                        tool_config = t
                        break

                source = getattr(tool_config, "source", "") if tool_config else ""

                async with SourceResolver.resolve_context(source) as resolved:
                    await initialize_single_tool(
                        agent=self._agent,
                        tool_name=job.tool_name,
                        force_ingest=job.force,
                        progress_callback=_progress_callback,
                        source_override=(
                            resolved.local_path if resolved.is_remote else None
                        ),
                    )

                job.state = InitJobState.COMPLETED
                job.completed_at = datetime.now()
                job.message = f"Tool '{job.tool_name}' initialized successfully"

                # Record success span attributes
                duration_ms = int(
                    (job.completed_at - job.started_at).total_seconds() * 1000
                )
                span.set_attribute("tool_init.job.state", job.state.value)
                span.set_attribute("tool_init.job.duration_ms", duration_ms)
                if job.progress:
                    span.set_attribute(
                        "tool_init.job.documents_processed",
                        job.progress.documents_processed,
                    )
                span.add_event("holodeck.serve.tool_init.complete")

            except asyncio.CancelledError:
                job.state = InitJobState.FAILED
                job.completed_at = datetime.now()
                job.message = "Cancelled due to server shutdown"
                span.set_status(StatusCode.ERROR, job.message)
                span.add_event("holodeck.serve.tool_init.failed")
            except Exception as exc:
                job.state = InitJobState.FAILED
                job.completed_at = datetime.now()
                job.error_detail = sanitize_error_detail(str(exc))
                job.message = f"Initialization failed for tool '{job.tool_name}'"
                logger.error("Init job failed for '%s': %s", job.tool_name, exc)
                span.set_status(StatusCode.ERROR, str(exc))
                span.add_event("holodeck.serve.tool_init.failed")
            finally:
                self._active_count -= 1
