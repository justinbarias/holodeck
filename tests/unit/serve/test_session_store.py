"""Unit tests for SessionStore and ServerSession.

Tests cover session lifecycle, ULID generation, TTL cleanup,
and session management operations.
"""

import re
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from holodeck.serve.session_store import ServerSession, SessionStore


class TestServerSession:
    """Tests for ServerSession dataclass."""

    def test_server_session_creation(self) -> None:
        """Test ServerSession creation with required fields."""
        mock_executor = MagicMock()
        session = ServerSession(agent_executor=mock_executor)

        assert session.agent_executor is mock_executor
        assert session.message_count == 0
        assert session.session_id is not None
        assert session.created_at is not None
        assert session.last_activity is not None

    def test_server_session_ulid_format(self) -> None:
        """Test session_id is valid ULID format."""
        mock_executor = MagicMock()
        session = ServerSession(agent_executor=mock_executor)

        # ULID is 26 characters, uppercase alphanumeric (Crockford Base32)
        ulid_pattern = r"^[0-9A-HJKMNP-TV-Z]{26}$"
        assert re.match(ulid_pattern, session.session_id) is not None

    def test_server_session_timestamps(self) -> None:
        """Test ServerSession timestamps are UTC."""
        mock_executor = MagicMock()
        session = ServerSession(agent_executor=mock_executor)

        assert session.created_at.tzinfo is not None
        assert session.last_activity.tzinfo is not None
        # Both should be very close to now
        now = datetime.now(timezone.utc)
        assert abs((session.created_at - now).total_seconds()) < 1
        assert abs((session.last_activity - now).total_seconds()) < 1

    def test_server_session_custom_session_id(self) -> None:
        """Test ServerSession accepts custom session_id."""
        mock_executor = MagicMock()
        custom_id = "01ARZ3NDEKTSV4RRFFQ69G5FAV"
        session = ServerSession(session_id=custom_id, agent_executor=mock_executor)

        assert session.session_id == custom_id

    def test_server_session_message_count_increment(self) -> None:
        """Test message_count can be incremented."""
        mock_executor = MagicMock()
        session = ServerSession(agent_executor=mock_executor)

        assert session.message_count == 0
        session.message_count += 1
        assert session.message_count == 1


class TestSessionStore:
    """Tests for SessionStore class."""

    def test_session_store_initialization(self) -> None:
        """Test SessionStore initializes with default TTL."""
        store = SessionStore()
        assert store.ttl_seconds == 1800  # 30 minutes default
        assert len(store.sessions) == 0

    def test_session_store_custom_ttl(self) -> None:
        """Test SessionStore with custom TTL."""
        store = SessionStore(ttl_seconds=3600)
        assert store.ttl_seconds == 3600

    def test_session_store_create(self) -> None:
        """Test SessionStore.create() creates a new session."""
        store = SessionStore()
        mock_executor = MagicMock()

        session = store.create(mock_executor)

        assert session is not None
        assert session.agent_executor is mock_executor
        assert session.session_id in store.sessions
        assert len(store.sessions) == 1

    def test_session_store_create_unique_ids(self) -> None:
        """Test SessionStore.create() generates unique session IDs."""
        store = SessionStore()
        mock_executor = MagicMock()

        session1 = store.create(mock_executor)
        session2 = store.create(mock_executor)

        assert session1.session_id != session2.session_id
        assert len(store.sessions) == 2

    def test_session_store_create_with_custom_session_id(self) -> None:
        """Test SessionStore.create() accepts custom session_id."""
        store = SessionStore()
        mock_executor = MagicMock()
        custom_id = "my-custom-thread-id"

        session = store.create(mock_executor, session_id=custom_id)

        assert session.session_id == custom_id
        assert store.get(custom_id) is session
        assert len(store.sessions) == 1

    def test_session_store_create_duplicate_session_id_raises(self) -> None:
        """Test SessionStore.create() raises on duplicate session_id."""
        store = SessionStore()
        mock_executor = MagicMock()
        custom_id = "unique-session-id"

        store.create(mock_executor, session_id=custom_id)

        with pytest.raises(ValueError, match="already exists"):
            store.create(mock_executor, session_id=custom_id)

    def test_session_store_get_existing(self) -> None:
        """Test SessionStore.get() returns existing session."""
        store = SessionStore()
        mock_executor = MagicMock()
        session = store.create(mock_executor)

        retrieved = store.get(session.session_id)

        assert retrieved is session
        assert retrieved.agent_executor is mock_executor

    def test_session_store_get_nonexistent(self) -> None:
        """Test SessionStore.get() returns None for nonexistent session."""
        store = SessionStore()

        result = store.get("nonexistent-session-id")

        assert result is None

    def test_session_store_delete_existing(self) -> None:
        """Test SessionStore.delete() removes existing session."""
        store = SessionStore()
        mock_executor = MagicMock()
        session = store.create(mock_executor)
        session_id = session.session_id

        result = store.delete(session_id)

        assert result is True
        assert store.get(session_id) is None
        assert len(store.sessions) == 0

    def test_session_store_delete_nonexistent(self) -> None:
        """Test SessionStore.delete() returns False for nonexistent session."""
        store = SessionStore()

        result = store.delete("nonexistent-session-id")

        assert result is False

    def test_session_store_touch_updates_activity(self) -> None:
        """Test SessionStore.touch() updates last_activity timestamp."""
        store = SessionStore()
        mock_executor = MagicMock()
        session = store.create(mock_executor)
        original_activity = session.last_activity

        # Simulate some time passing
        import time

        time.sleep(0.01)

        store.touch(session.session_id)

        assert session.last_activity > original_activity

    def test_session_store_touch_nonexistent_does_nothing(self) -> None:
        """Test SessionStore.touch() silently handles nonexistent session."""
        store = SessionStore()

        # Should not raise
        store.touch("nonexistent-session-id")

    def test_session_store_cleanup_expired(self) -> None:
        """Test SessionStore.cleanup_expired() removes expired sessions."""
        store = SessionStore(ttl_seconds=1)  # 1 second TTL for testing
        mock_executor = MagicMock()
        session = store.create(mock_executor)

        # Manually set last_activity to past
        session.last_activity = datetime.now(timezone.utc) - timedelta(seconds=10)

        count = store.cleanup_expired()

        assert count == 1
        assert len(store.sessions) == 0

    def test_session_store_cleanup_preserves_active(self) -> None:
        """Test SessionStore.cleanup_expired() preserves active sessions."""
        store = SessionStore(ttl_seconds=3600)  # 1 hour TTL
        mock_executor = MagicMock()
        session = store.create(mock_executor)

        count = store.cleanup_expired()

        assert count == 0
        assert len(store.sessions) == 1
        assert store.get(session.session_id) is session

    def test_session_store_cleanup_mixed(self) -> None:
        """Test SessionStore.cleanup_expired() with mixed active/expired sessions."""
        store = SessionStore(ttl_seconds=60)
        mock_executor = MagicMock()

        # Create active session
        active_session = store.create(mock_executor)

        # Create expired session
        expired_session = store.create(mock_executor)
        expired_time = datetime.now(timezone.utc) - timedelta(seconds=120)
        expired_session.last_activity = expired_time

        count = store.cleanup_expired()

        assert count == 1
        assert len(store.sessions) == 1
        assert store.get(active_session.session_id) is active_session
        assert store.get(expired_session.session_id) is None

    def test_session_store_active_count(self) -> None:
        """Test SessionStore tracks active session count."""
        store = SessionStore()
        mock_executor = MagicMock()

        assert store.active_count == 0

        store.create(mock_executor)
        assert store.active_count == 1

        store.create(mock_executor)
        assert store.active_count == 2

    def test_session_store_get_all(self) -> None:
        """Test SessionStore can retrieve all sessions."""
        store = SessionStore()
        mock_executor = MagicMock()

        session1 = store.create(mock_executor)
        session2 = store.create(mock_executor)

        all_sessions = store.get_all()

        assert len(all_sessions) == 2
        assert session1 in all_sessions
        assert session2 in all_sessions

    def test_session_store_max_sessions_default(self) -> None:
        """Test SessionStore has default max_sessions of 1000."""
        store = SessionStore()
        assert store.max_sessions == 1000

    def test_session_store_custom_max_sessions(self) -> None:
        """Test SessionStore with custom max_sessions."""
        store = SessionStore(max_sessions=5)
        assert store.max_sessions == 5

    def test_session_store_create_rejects_when_at_max(self) -> None:
        """Test SessionStore.create() raises when max_sessions reached."""
        store = SessionStore(max_sessions=2)
        mock_executor = MagicMock()

        store.create(mock_executor)
        store.create(mock_executor)

        with pytest.raises(RuntimeError, match="Maximum session limit"):
            store.create(mock_executor)

    def test_session_store_cleanup_interval_default(self) -> None:
        """Test SessionStore has default cleanup_interval of 300 seconds."""
        store = SessionStore()
        assert store.cleanup_interval_seconds == 300

    def test_session_store_cleanup_task_initially_none(self) -> None:
        """Test SessionStore._cleanup_task is None initially."""
        store = SessionStore()
        assert store._cleanup_task is None


class TestSessionStoreCleanupTask:
    """Tests for SessionStore cleanup task."""

    @pytest.mark.asyncio
    async def test_start_cleanup_task(self) -> None:
        """Test start_cleanup_task creates a background task."""
        store = SessionStore(cleanup_interval_seconds=1)

        await store.start_cleanup_task()

        assert store._cleanup_task is not None
        assert not store._cleanup_task.done()

        # Cleanup
        await store.stop_cleanup_task()

    @pytest.mark.asyncio
    async def test_stop_cleanup_task(self) -> None:
        """Test stop_cleanup_task cancels the background task."""
        store = SessionStore(cleanup_interval_seconds=1)

        await store.start_cleanup_task()
        await store.stop_cleanup_task()

        assert store._cleanup_task is None

    @pytest.mark.asyncio
    async def test_stop_cleanup_task_when_not_started(self) -> None:
        """Test stop_cleanup_task handles case when no task exists."""
        store = SessionStore()

        # Should not raise
        await store.stop_cleanup_task()

        assert store._cleanup_task is None

    @pytest.mark.asyncio
    async def test_cleanup_loop_removes_expired_sessions(self) -> None:
        """Test cleanup loop removes expired sessions."""

        store = SessionStore(ttl_seconds=0, cleanup_interval_seconds=0)
        mock_executor = MagicMock()

        # Create a session that will be immediately expired
        store.create(mock_executor)
        assert store.active_count == 1

        # Run cleanup once manually
        store.cleanup_expired()

        assert store.active_count == 0

    @pytest.mark.asyncio
    async def test_start_cleanup_task_idempotent(self) -> None:
        """Test calling start_cleanup_task twice uses same task."""
        store = SessionStore(cleanup_interval_seconds=1)

        await store.start_cleanup_task()
        first_task = store._cleanup_task

        await store.start_cleanup_task()
        second_task = store._cleanup_task

        # Should be same task since it's still running
        assert first_task is second_task

        # Cleanup
        await store.stop_cleanup_task()
