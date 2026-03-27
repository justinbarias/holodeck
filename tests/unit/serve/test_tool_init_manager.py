"""Tests for ToolInitManager."""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from holodeck.serve.models import InitJobProgress, InitJobState


class TestInitJob:
    """Tests for InitJob dataclass (T009)."""

    def test_default_values(self) -> None:
        from holodeck.serve.tool_init_manager import InitJob

        job = InitJob(tool_name="test_tool", created_at=datetime.now())
        assert job.state == InitJobState.PENDING
        assert job.started_at is None
        assert job.completed_at is None
        assert job.message is None
        assert job.error_detail is None
        assert job.progress is None
        assert job.force is False

    def test_mutable_state(self) -> None:
        from holodeck.serve.tool_init_manager import InitJob

        job = InitJob(tool_name="test_tool", created_at=datetime.now())
        job.state = InitJobState.IN_PROGRESS
        job.started_at = datetime.now()
        job.progress = InitJobProgress(documents_processed=5, total_documents=10)
        assert job.state == InitJobState.IN_PROGRESS
        assert job.progress.documents_processed == 5


def _make_mock_agent(tools=None):
    """Create a mock agent with tools."""
    agent = MagicMock()
    agent.tools = tools
    return agent


def _make_vectorstore_tool(name="knowledge_base"):
    """Create a mock vectorstore tool config."""
    tool = MagicMock()
    tool.name = name
    tool.type = "vectorstore"
    return tool


def _make_hierarchical_doc_tool(name="docs"):
    """Create a mock hierarchical_document tool config."""
    tool = MagicMock()
    tool.name = name
    tool.type = "hierarchical_document"
    return tool


def _make_mcp_tool(name="api"):
    """Create a mock MCP tool config."""
    tool = MagicMock()
    tool.name = name
    tool.type = "mcp"
    return tool


def _make_function_tool(name="calc"):
    """Create a mock function tool config."""
    tool = MagicMock()
    tool.name = name
    tool.type = "function"
    return tool


class TestToolInitManagerStartJob:
    """Tests for ToolInitManager.start_init_job() (T010)."""

    @pytest.mark.asyncio
    async def test_start_job_vectorstore_creates_pending(self) -> None:
        from holodeck.serve.tool_init_manager import ToolInitManager

        vs_tool = _make_vectorstore_tool()
        agent = _make_mock_agent(tools=[vs_tool])
        manager = ToolInitManager(agent=agent)

        # Patch _run_init_job to prevent actual execution
        with patch.object(manager, "_run_init_job", new_callable=AsyncMock):
            job = manager.start_init_job("knowledge_base")
            assert job.state == InitJobState.PENDING
            assert job.tool_name == "knowledge_base"

    @pytest.mark.asyncio
    async def test_start_job_hierarchical_doc_succeeds(self) -> None:
        from holodeck.serve.tool_init_manager import ToolInitManager

        hd_tool = _make_hierarchical_doc_tool()
        agent = _make_mock_agent(tools=[hd_tool])
        manager = ToolInitManager(agent=agent)

        with patch.object(manager, "_run_init_job", new_callable=AsyncMock):
            job = manager.start_init_job("docs")
            assert job.state == InitJobState.PENDING

    @pytest.mark.asyncio
    async def test_start_job_nonexistent_tool_raises(self) -> None:
        from holodeck.serve.tool_init_manager import ToolInitManager, ToolNotFoundError

        agent = _make_mock_agent(tools=[_make_vectorstore_tool()])
        manager = ToolInitManager(agent=agent)

        with pytest.raises(ToolNotFoundError):
            manager.start_init_job("nonexistent")

    @pytest.mark.asyncio
    async def test_start_job_non_init_tool_raises(self) -> None:
        from holodeck.serve.tool_init_manager import (
            ToolInitManager,
            ToolNotInitializableError,
        )

        agent = _make_mock_agent(tools=[_make_mcp_tool()])
        manager = ToolInitManager(agent=agent)

        with pytest.raises(ToolNotInitializableError):
            manager.start_init_job("api")

    @pytest.mark.asyncio
    async def test_start_job_function_tool_raises(self) -> None:
        from holodeck.serve.tool_init_manager import (
            ToolInitManager,
            ToolNotInitializableError,
        )

        agent = _make_mock_agent(tools=[_make_function_tool()])
        manager = ToolInitManager(agent=agent)

        with pytest.raises(ToolNotInitializableError):
            manager.start_init_job("calc")

    @pytest.mark.asyncio
    async def test_start_job_active_pending_raises_conflict(self) -> None:
        from holodeck.serve.tool_init_manager import (
            InitJobConflictError,
            ToolInitManager,
        )

        agent = _make_mock_agent(tools=[_make_vectorstore_tool()])
        manager = ToolInitManager(agent=agent)

        with patch.object(manager, "_run_init_job", new_callable=AsyncMock):
            manager.start_init_job("knowledge_base")
            with pytest.raises(InitJobConflictError):
                manager.start_init_job("knowledge_base")

    @pytest.mark.asyncio
    async def test_start_job_completed_replaces(self) -> None:
        from holodeck.serve.tool_init_manager import ToolInitManager

        agent = _make_mock_agent(tools=[_make_vectorstore_tool()])
        manager = ToolInitManager(agent=agent)

        with patch.object(manager, "_run_init_job", new_callable=AsyncMock):
            job1 = manager.start_init_job("knowledge_base")
            # Simulate completion
            job1.state = InitJobState.COMPLETED
            job2 = manager.start_init_job("knowledge_base")
            assert job2.state == InitJobState.PENDING
            assert job2 is not job1

    @pytest.mark.asyncio
    async def test_start_job_failed_replaces(self) -> None:
        from holodeck.serve.tool_init_manager import ToolInitManager

        agent = _make_mock_agent(tools=[_make_vectorstore_tool()])
        manager = ToolInitManager(agent=agent)

        with patch.object(manager, "_run_init_job", new_callable=AsyncMock):
            job1 = manager.start_init_job("knowledge_base")
            job1.state = InitJobState.FAILED
            job2 = manager.start_init_job("knowledge_base")
            assert job2.state == InitJobState.PENDING

    @pytest.mark.asyncio
    async def test_start_job_at_capacity_raises(self) -> None:
        from holodeck.serve.tool_init_manager import (
            InitJobCapacityError,
            ToolInitManager,
        )

        tools = [_make_vectorstore_tool(f"tool_{i}") for i in range(4)]
        agent = _make_mock_agent(tools=tools)
        manager = ToolInitManager(agent=agent, max_concurrent=3)

        with patch.object(manager, "_run_init_job", new_callable=AsyncMock):
            # Fill up capacity
            manager.start_init_job("tool_0")
            manager._active_count = 3  # Simulate 3 active jobs

            with pytest.raises(InitJobCapacityError):
                manager.start_init_job("tool_3")

    @pytest.mark.asyncio
    async def test_start_job_shutting_down_raises(self) -> None:
        from holodeck.serve.tool_init_manager import (
            InitManagerShuttingDownError,
            ToolInitManager,
        )

        agent = _make_mock_agent(tools=[_make_vectorstore_tool()])
        manager = ToolInitManager(agent=agent)
        manager._shutting_down = True

        with pytest.raises(InitManagerShuttingDownError):
            manager.start_init_job("knowledge_base")

    @pytest.mark.asyncio
    async def test_start_job_force_propagated(self) -> None:
        from holodeck.serve.tool_init_manager import ToolInitManager

        agent = _make_mock_agent(tools=[_make_vectorstore_tool()])
        manager = ToolInitManager(agent=agent)

        with patch.object(manager, "_run_init_job", new_callable=AsyncMock):
            job = manager.start_init_job("knowledge_base", force=True)
            assert job.force is True

    @pytest.mark.asyncio
    async def test_start_job_no_tools_raises(self) -> None:
        from holodeck.serve.tool_init_manager import ToolInitManager, ToolNotFoundError

        agent = _make_mock_agent(tools=None)
        manager = ToolInitManager(agent=agent)

        with pytest.raises(ToolNotFoundError):
            manager.start_init_job("anything")


class TestToolInitManagerGetJob:
    """Tests for ToolInitManager.get_job() (T012)."""

    @pytest.mark.asyncio
    async def test_get_job_returns_existing(self) -> None:
        from holodeck.serve.tool_init_manager import ToolInitManager

        agent = _make_mock_agent(tools=[_make_vectorstore_tool()])
        manager = ToolInitManager(agent=agent)

        with patch.object(manager, "_run_init_job", new_callable=AsyncMock):
            created = manager.start_init_job("knowledge_base")
            found = manager.get_job("knowledge_base")
            assert found is created

    def test_get_job_returns_none_for_unknown(self) -> None:
        from holodeck.serve.tool_init_manager import ToolInitManager

        agent = _make_mock_agent(tools=[])
        manager = ToolInitManager(agent=agent)
        assert manager.get_job("nonexistent") is None


class TestToolInitManagerShutdown:
    """Tests for ToolInitManager.shutdown() (T012)."""

    @pytest.mark.asyncio
    async def test_shutdown_sets_flag(self) -> None:
        from holodeck.serve.tool_init_manager import ToolInitManager

        agent = _make_mock_agent(tools=[])
        manager = ToolInitManager(agent=agent)
        await manager.shutdown()
        assert manager._shutting_down is True

    @pytest.mark.asyncio
    async def test_shutdown_cancels_active_tasks(self) -> None:
        from holodeck.serve.tool_init_manager import ToolInitManager

        agent = _make_mock_agent(tools=[_make_vectorstore_tool()])
        manager = ToolInitManager(agent=agent)

        # Create a long-running mock task
        async def long_task():
            await asyncio.sleep(100)

        task = asyncio.create_task(long_task())
        manager._tasks["knowledge_base"] = task

        from holodeck.serve.tool_init_manager import InitJob

        job = InitJob(tool_name="knowledge_base", created_at=datetime.now())
        job.state = InitJobState.IN_PROGRESS
        manager._jobs["knowledge_base"] = job

        await manager.shutdown()

        assert task.cancelled() or task.done()
        assert job.state == InitJobState.FAILED

    @pytest.mark.asyncio
    async def test_shutdown_idempotent(self) -> None:
        from holodeck.serve.tool_init_manager import ToolInitManager

        agent = _make_mock_agent(tools=[])
        manager = ToolInitManager(agent=agent)
        await manager.shutdown()
        await manager.shutdown()  # Should not raise
        assert manager._shutting_down is True

    @pytest.mark.asyncio
    async def test_start_after_shutdown_raises(self) -> None:
        from holodeck.serve.tool_init_manager import (
            InitManagerShuttingDownError,
            ToolInitManager,
        )

        agent = _make_mock_agent(tools=[_make_vectorstore_tool()])
        manager = ToolInitManager(agent=agent)
        await manager.shutdown()

        with pytest.raises(InitManagerShuttingDownError):
            manager.start_init_job("knowledge_base")


class TestRunInitJob:
    """Tests for ToolInitManager._run_init_job() (T011)."""

    @pytest.mark.asyncio
    async def test_transitions_to_in_progress(self) -> None:
        """Job transitions from pending to in_progress with started_at set."""
        from holodeck.serve.tool_init_manager import InitJob, ToolInitManager

        agent = _make_mock_agent(tools=[_make_vectorstore_tool()])
        manager = ToolInitManager(agent=agent)
        job = InitJob(tool_name="knowledge_base", created_at=datetime.now())

        with (
            patch("holodeck.serve.tool_init_manager.SourceResolver") as mock_sr,
            patch(
                "holodeck.serve.tool_init_manager.initialize_single_tool",
                new_callable=AsyncMock,
            ),
        ):
            mock_resolved = MagicMock()
            mock_resolved.local_path = Path("/tmp/data")  # noqa: S108
            mock_resolved.is_remote = False
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__ = AsyncMock(return_value=mock_resolved)
            mock_ctx.__aexit__ = AsyncMock(return_value=False)
            mock_sr.resolve_context.return_value = mock_ctx

            await manager._run_init_job(job)

        assert job.started_at is not None
        assert job.state == InitJobState.COMPLETED

    @pytest.mark.asyncio
    async def test_success_transitions_to_completed(self) -> None:
        """Successful job sets state=completed and completed_at."""
        from holodeck.serve.tool_init_manager import InitJob, ToolInitManager

        agent = _make_mock_agent(tools=[_make_vectorstore_tool()])
        manager = ToolInitManager(agent=agent)
        job = InitJob(tool_name="knowledge_base", created_at=datetime.now())

        with (
            patch("holodeck.serve.tool_init_manager.SourceResolver") as mock_sr,
            patch(
                "holodeck.serve.tool_init_manager.initialize_single_tool",
                new_callable=AsyncMock,
            ),
        ):
            mock_resolved = MagicMock()
            mock_resolved.local_path = Path("/tmp/data")  # noqa: S108
            mock_resolved.is_remote = False
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__ = AsyncMock(return_value=mock_resolved)
            mock_ctx.__aexit__ = AsyncMock(return_value=False)
            mock_sr.resolve_context.return_value = mock_ctx

            await manager._run_init_job(job)

        assert job.state == InitJobState.COMPLETED
        assert job.completed_at is not None
        assert job.message is not None

    @pytest.mark.asyncio
    async def test_failure_transitions_to_failed(self) -> None:
        """Exception sets state=failed with sanitized error_detail."""
        from holodeck.serve.tool_init_manager import InitJob, ToolInitManager

        agent = _make_mock_agent(tools=[_make_vectorstore_tool()])
        manager = ToolInitManager(agent=agent)
        job = InitJob(tool_name="knowledge_base", created_at=datetime.now())

        with (
            patch("holodeck.serve.tool_init_manager.SourceResolver") as mock_sr,
            patch(
                "holodeck.serve.tool_init_manager.initialize_single_tool",
                new_callable=AsyncMock,
            ) as mock_init,
        ):
            mock_resolved = MagicMock()
            mock_resolved.local_path = Path("/tmp/data")  # noqa: S108
            mock_resolved.is_remote = False
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__ = AsyncMock(return_value=mock_resolved)
            mock_ctx.__aexit__ = AsyncMock(return_value=False)
            mock_sr.resolve_context.return_value = mock_ctx
            mock_init.side_effect = RuntimeError("AKIAIOSFODNN7EXAMPLE leaked")

            await manager._run_init_job(job)

        assert job.state == InitJobState.FAILED
        assert job.error_detail is not None
        assert "AKIAIOSFODNN7EXAMPLE" not in job.error_detail

    @pytest.mark.asyncio
    async def test_cancelled_marks_failed(self) -> None:
        """CancelledError sets state=failed with shutdown message."""
        from holodeck.serve.tool_init_manager import InitJob, ToolInitManager

        agent = _make_mock_agent(tools=[_make_vectorstore_tool()])
        manager = ToolInitManager(agent=agent)
        job = InitJob(tool_name="knowledge_base", created_at=datetime.now())

        with (
            patch("holodeck.serve.tool_init_manager.SourceResolver") as mock_sr,
            patch(
                "holodeck.serve.tool_init_manager.initialize_single_tool",
                new_callable=AsyncMock,
            ) as mock_init,
        ):
            mock_resolved = MagicMock()
            mock_resolved.local_path = Path("/tmp/data")  # noqa: S108
            mock_resolved.is_remote = False
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__ = AsyncMock(return_value=mock_resolved)
            mock_ctx.__aexit__ = AsyncMock(return_value=False)
            mock_sr.resolve_context.return_value = mock_ctx
            mock_init.side_effect = asyncio.CancelledError()

            await manager._run_init_job(job)

        assert job.state == InitJobState.FAILED
        assert "shutdown" in job.message.lower()

    @pytest.mark.asyncio
    async def test_decrements_active_count_on_success(self) -> None:
        """_active_count decremented after successful job."""
        from holodeck.serve.tool_init_manager import InitJob, ToolInitManager

        agent = _make_mock_agent(tools=[_make_vectorstore_tool()])
        manager = ToolInitManager(agent=agent)
        manager._active_count = 1
        job = InitJob(tool_name="knowledge_base", created_at=datetime.now())

        with (
            patch("holodeck.serve.tool_init_manager.SourceResolver") as mock_sr,
            patch(
                "holodeck.serve.tool_init_manager.initialize_single_tool",
                new_callable=AsyncMock,
            ),
        ):
            mock_resolved = MagicMock()
            mock_resolved.local_path = Path("/tmp/data")  # noqa: S108
            mock_resolved.is_remote = False
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__ = AsyncMock(return_value=mock_resolved)
            mock_ctx.__aexit__ = AsyncMock(return_value=False)
            mock_sr.resolve_context.return_value = mock_ctx

            await manager._run_init_job(job)

        assert manager._active_count == 0

    @pytest.mark.asyncio
    async def test_decrements_active_count_on_failure(self) -> None:
        """_active_count decremented even on failure."""
        from holodeck.serve.tool_init_manager import InitJob, ToolInitManager

        agent = _make_mock_agent(tools=[_make_vectorstore_tool()])
        manager = ToolInitManager(agent=agent)
        manager._active_count = 1
        job = InitJob(tool_name="knowledge_base", created_at=datetime.now())

        with (
            patch("holodeck.serve.tool_init_manager.SourceResolver") as mock_sr,
            patch(
                "holodeck.serve.tool_init_manager.initialize_single_tool",
                new_callable=AsyncMock,
            ) as mock_init,
        ):
            mock_resolved = MagicMock()
            mock_resolved.local_path = Path("/tmp/data")  # noqa: S108
            mock_resolved.is_remote = False
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__ = AsyncMock(return_value=mock_resolved)
            mock_ctx.__aexit__ = AsyncMock(return_value=False)
            mock_sr.resolve_context.return_value = mock_ctx
            mock_init.side_effect = RuntimeError("boom")

            await manager._run_init_job(job)

        assert manager._active_count == 0

    @pytest.mark.asyncio
    async def test_progress_callback_updates_job(self) -> None:
        """Progress callback from initialize_single_tool updates job.progress."""
        from holodeck.serve.tool_init_manager import InitJob, ToolInitManager

        agent = _make_mock_agent(tools=[_make_vectorstore_tool()])
        manager = ToolInitManager(agent=agent)
        job = InitJob(tool_name="knowledge_base", created_at=datetime.now())

        with (
            patch("holodeck.serve.tool_init_manager.SourceResolver") as mock_sr,
            patch(
                "holodeck.serve.tool_init_manager.initialize_single_tool",
                new_callable=AsyncMock,
            ) as mock_init,
        ):
            mock_resolved = MagicMock()
            mock_resolved.local_path = Path("/tmp/data")  # noqa: S108
            mock_resolved.is_remote = False
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__ = AsyncMock(return_value=mock_resolved)
            mock_ctx.__aexit__ = AsyncMock(return_value=False)
            mock_sr.resolve_context.return_value = mock_ctx

            async def fake_init(*args, **kwargs):
                cb = kwargs.get("progress_callback")
                if cb:
                    cb(3, 10)

            mock_init.side_effect = fake_init

            await manager._run_init_job(job)

        assert job.progress is not None
        assert job.progress.documents_processed == 3
        assert job.progress.total_documents == 10


class TestToolInitManagerReInit:
    """Tests for re-initialization scenarios (US4)."""

    @pytest.mark.asyncio
    async def test_reinit_completed_creates_fresh_job_with_new_timestamp(self) -> None:
        """Replacing a completed job produces a new InitJob with fresh created_at."""
        from holodeck.serve.tool_init_manager import ToolInitManager

        agent = _make_mock_agent(tools=[_make_vectorstore_tool()])
        manager = ToolInitManager(agent=agent)

        with patch.object(manager, "_run_init_job", new_callable=AsyncMock):
            job1 = manager.start_init_job("knowledge_base")
            ts1 = job1.created_at
            job1.state = InitJobState.COMPLETED

            job2 = manager.start_init_job("knowledge_base")

            assert job2 is not job1
            assert job2.state == InitJobState.PENDING
            assert job2.created_at >= ts1

    @pytest.mark.asyncio
    async def test_reinit_failed_replaces_stale_task_in_tasks_dict(self) -> None:
        """Replacing a failed job updates _tasks to point to the new asyncio.Task."""
        from holodeck.serve.tool_init_manager import ToolInitManager

        agent = _make_mock_agent(tools=[_make_vectorstore_tool()])
        manager = ToolInitManager(agent=agent)

        with patch.object(manager, "_run_init_job", new_callable=AsyncMock):
            job1 = manager.start_init_job("knowledge_base")
            old_task = manager._tasks["knowledge_base"]
            job1.state = InitJobState.FAILED

            manager.start_init_job("knowledge_base")
            new_task = manager._tasks["knowledge_base"]

            assert new_task is not old_task

    @pytest.mark.asyncio
    async def test_reinit_active_count_correct_after_replacement(self) -> None:
        """_active_count is 1 after re-init of a completed job (not double-counted)."""
        from holodeck.serve.tool_init_manager import ToolInitManager

        agent = _make_mock_agent(tools=[_make_vectorstore_tool()])
        manager = ToolInitManager(agent=agent)

        with patch.object(manager, "_run_init_job", new_callable=AsyncMock):
            job1 = manager.start_init_job("knowledge_base")
            # Simulate completion (decrement as _run_init_job would)
            job1.state = InitJobState.COMPLETED
            manager._active_count -= 1  # Simulates finally block in _run_init_job

            manager.start_init_job("knowledge_base")

            assert manager._active_count == 1

    @pytest.mark.asyncio
    async def test_reinit_with_force_true_propagated(self) -> None:
        """Re-init with force=True stores force on the replacement job."""
        from holodeck.serve.tool_init_manager import ToolInitManager

        agent = _make_mock_agent(tools=[_make_vectorstore_tool()])
        manager = ToolInitManager(agent=agent)

        with patch.object(manager, "_run_init_job", new_callable=AsyncMock):
            job1 = manager.start_init_job("knowledge_base")
            job1.state = InitJobState.COMPLETED
            manager._active_count -= 1

            job2 = manager.start_init_job("knowledge_base", force=True)

            assert job2.force is True

    @pytest.mark.asyncio
    async def test_reinit_pending_raises_conflict_even_with_force(self) -> None:
        """force=True does NOT bypass 409 for PENDING/IN_PROGRESS jobs."""
        from holodeck.serve.tool_init_manager import (
            InitJobConflictError,
            ToolInitManager,
        )

        agent = _make_mock_agent(tools=[_make_vectorstore_tool()])
        manager = ToolInitManager(agent=agent)

        with patch.object(manager, "_run_init_job", new_callable=AsyncMock):
            manager.start_init_job("knowledge_base")

            with pytest.raises(InitJobConflictError):
                manager.start_init_job("knowledge_base", force=True)


class TestOTelInstrumentation:
    """Tests for OTel span instrumentation in ToolInitManager (T031)."""

    @pytest.mark.asyncio
    async def test_start_job_creates_otel_span(self) -> None:
        """start_init_job creates a holodeck.serve.tool_init.start span."""
        from holodeck.serve.tool_init_manager import ToolInitManager

        agent = _make_mock_agent(tools=[_make_vectorstore_tool()])
        manager = ToolInitManager(agent=agent)

        with (
            patch.object(manager, "_run_init_job", new_callable=AsyncMock),
            patch("holodeck.serve.tool_init_manager.get_tracer") as mock_get_tracer,
        ):
            mock_tracer = MagicMock()
            mock_span = MagicMock()
            mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(
                return_value=mock_span
            )
            mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(
                return_value=False
            )
            mock_get_tracer.return_value = mock_tracer

            manager.start_init_job("knowledge_base")

            mock_tracer.start_as_current_span.assert_called_with(
                "holodeck.serve.tool_init.start"
            )

    @pytest.mark.asyncio
    async def test_start_job_span_attributes(self) -> None:
        """start_init_job span sets tool_name, state, and force attributes."""
        from holodeck.serve.tool_init_manager import ToolInitManager

        agent = _make_mock_agent(tools=[_make_vectorstore_tool()])
        manager = ToolInitManager(agent=agent)

        with (
            patch.object(manager, "_run_init_job", new_callable=AsyncMock),
            patch("holodeck.serve.tool_init_manager.get_tracer") as mock_get_tracer,
        ):
            mock_tracer = MagicMock()
            mock_span = MagicMock()
            mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(
                return_value=mock_span
            )
            mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(
                return_value=False
            )
            mock_get_tracer.return_value = mock_tracer

            manager.start_init_job("knowledge_base", force=True)

            calls = mock_span.set_attribute.call_args_list
            attr_dict = {c[0][0]: c[0][1] for c in calls}
            assert attr_dict["tool_init.job.tool_name"] == "knowledge_base"
            assert attr_dict["tool_init.job.force"] is True

    @pytest.mark.asyncio
    async def test_run_job_success_creates_span(self) -> None:
        """_run_init_job wraps execution in an OTel span on success."""
        from holodeck.serve.tool_init_manager import InitJob, ToolInitManager

        agent = _make_mock_agent(tools=[_make_vectorstore_tool()])
        manager = ToolInitManager(agent=agent)
        job = InitJob(tool_name="knowledge_base", created_at=datetime.now())

        with (
            patch("holodeck.serve.tool_init_manager.SourceResolver") as mock_sr,
            patch(
                "holodeck.serve.tool_init_manager.initialize_single_tool",
                new_callable=AsyncMock,
            ),
            patch("holodeck.serve.tool_init_manager.get_tracer") as mock_get_tracer,
        ):
            mock_tracer = MagicMock()
            mock_span = MagicMock()
            mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(
                return_value=mock_span
            )
            mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(
                return_value=False
            )
            mock_get_tracer.return_value = mock_tracer

            mock_resolved = MagicMock()
            mock_resolved.local_path = Path("/tmp/data")  # noqa: S108
            mock_resolved.is_remote = False
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__ = AsyncMock(return_value=mock_resolved)
            mock_ctx.__aexit__ = AsyncMock(return_value=False)
            mock_sr.resolve_context.return_value = mock_ctx

            await manager._run_init_job(job)

        mock_tracer.start_as_current_span.assert_called_with(
            "holodeck.serve.tool_init.progress"
        )
        # Verify complete event was added
        mock_span.add_event.assert_any_call("holodeck.serve.tool_init.complete")

    @pytest.mark.asyncio
    async def test_run_job_failure_sets_error_status(self) -> None:
        """_run_init_job sets StatusCode.ERROR on span when job fails."""
        from holodeck.serve.tool_init_manager import InitJob, ToolInitManager

        agent = _make_mock_agent(tools=[_make_vectorstore_tool()])
        manager = ToolInitManager(agent=agent)
        job = InitJob(tool_name="knowledge_base", created_at=datetime.now())

        with (
            patch("holodeck.serve.tool_init_manager.SourceResolver") as mock_sr,
            patch(
                "holodeck.serve.tool_init_manager.initialize_single_tool",
                new_callable=AsyncMock,
            ) as mock_init,
            patch("holodeck.serve.tool_init_manager.get_tracer") as mock_get_tracer,
        ):
            mock_tracer = MagicMock()
            mock_span = MagicMock()
            mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(
                return_value=mock_span
            )
            mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(
                return_value=False
            )
            mock_get_tracer.return_value = mock_tracer

            mock_resolved = MagicMock()
            mock_resolved.local_path = Path("/tmp/data")  # noqa: S108
            mock_resolved.is_remote = False
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__ = AsyncMock(return_value=mock_resolved)
            mock_ctx.__aexit__ = AsyncMock(return_value=False)
            mock_sr.resolve_context.return_value = mock_ctx
            mock_init.side_effect = RuntimeError("boom")

            await manager._run_init_job(job)

        from opentelemetry.trace import StatusCode

        mock_span.set_status.assert_called_once()
        status_call = mock_span.set_status.call_args[0]
        assert status_call[0] == StatusCode.ERROR
        mock_span.add_event.assert_any_call("holodeck.serve.tool_init.failed")


class TestGetAllToolStatuses:
    """Tests for ToolInitManager.get_all_tool_statuses() (US3)."""

    def test_no_tools_returns_empty(self) -> None:
        """Agent with no tools returns an empty list."""
        from holodeck.serve.tool_init_manager import ToolInitManager

        agent = _make_mock_agent(tools=None)
        manager = ToolInitManager(agent=agent)

        result = manager.get_all_tool_statuses()

        assert result == []

    def test_empty_tools_list_returns_empty(self) -> None:
        """Agent with empty tools list returns an empty list."""
        from holodeck.serve.tool_init_manager import ToolInitManager

        agent = _make_mock_agent(tools=[])
        manager = ToolInitManager(agent=agent)

        result = manager.get_all_tool_statuses()

        assert result == []

    def test_mixed_tools_correct_supports_init(self) -> None:
        """Vectorstore/hierarchical_document have supports_init=True, others False."""
        from holodeck.serve.tool_init_manager import ToolInitManager

        tools = [
            _make_vectorstore_tool("kb"),
            _make_hierarchical_doc_tool("docs"),
            _make_mcp_tool("api"),
            _make_function_tool("calc"),
        ]
        agent = _make_mock_agent(tools=tools)
        manager = ToolInitManager(agent=agent)

        result = manager.get_all_tool_statuses()

        assert len(result) == 4
        by_name = {r.name: r for r in result}
        assert by_name["kb"].supports_init is True
        assert by_name["docs"].supports_init is True
        assert by_name["api"].supports_init is False
        assert by_name["calc"].supports_init is False

    def test_tool_types_in_response(self) -> None:
        """Each response includes the correct tool type."""
        from holodeck.serve.tool_init_manager import ToolInitManager

        tools = [_make_vectorstore_tool("kb"), _make_mcp_tool("api")]
        agent = _make_mock_agent(tools=tools)
        manager = ToolInitManager(agent=agent)

        result = manager.get_all_tool_statuses()

        by_name = {r.name: r for r in result}
        assert by_name["kb"].type == "vectorstore"
        assert by_name["api"].type == "mcp"

    def test_no_job_means_init_status_none(self) -> None:
        """Tools without prior init jobs have init_status=None."""
        from holodeck.serve.tool_init_manager import ToolInitManager

        agent = _make_mock_agent(tools=[_make_vectorstore_tool("kb")])
        manager = ToolInitManager(agent=agent)

        result = manager.get_all_tool_statuses()

        assert result[0].init_status is None

    @pytest.mark.asyncio
    async def test_existing_job_reflects_state(self) -> None:
        """Tools with an init job show the current job state."""
        from holodeck.serve.tool_init_manager import ToolInitManager

        agent = _make_mock_agent(tools=[_make_vectorstore_tool("kb")])
        manager = ToolInitManager(agent=agent)

        with patch.object(manager, "_run_init_job", new_callable=AsyncMock):
            job = manager.start_init_job("kb")
            job.state = InitJobState.COMPLETED

        result = manager.get_all_tool_statuses()

        assert result[0].init_status == InitJobState.COMPLETED

    @pytest.mark.asyncio
    async def test_mixed_tools_with_and_without_jobs(self) -> None:
        """Some tools have jobs, others don't — both reported correctly."""
        from holodeck.serve.tool_init_manager import ToolInitManager

        tools = [
            _make_vectorstore_tool("kb"),
            _make_mcp_tool("api"),
            _make_hierarchical_doc_tool("docs"),
        ]
        agent = _make_mock_agent(tools=tools)
        manager = ToolInitManager(agent=agent)

        with patch.object(manager, "_run_init_job", new_callable=AsyncMock):
            job = manager.start_init_job("kb")
            job.state = InitJobState.IN_PROGRESS

        result = manager.get_all_tool_statuses()
        by_name = {r.name: r for r in result}

        assert by_name["kb"].init_status == InitJobState.IN_PROGRESS
        assert by_name["kb"].supports_init is True
        assert by_name["api"].init_status is None
        assert by_name["api"].supports_init is False
        assert by_name["docs"].init_status is None
        assert by_name["docs"].supports_init is True
