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
