"""Session management for the Agent Local Server.

Provides in-memory session storage with TTL-based expiration.
Sessions maintain conversation context across multiple requests.
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

from ulid import ULID

from holodeck.lib.logging_config import get_logger

if TYPE_CHECKING:
    from holodeck.chat.executor import AgentExecutor

logger = get_logger(__name__)


@dataclass
class ServerSession:
    """Individual conversation session with an agent.

    Maintains state for a single conversation, including the agent executor
    instance that preserves conversation history.

    Attributes:
        session_id: Unique identifier in ULID format.
        agent_executor: Agent execution context with conversation history.
        created_at: UTC timestamp when session was created.
        last_activity: UTC timestamp of last request in session.
        message_count: Number of messages exchanged in session.
    """

    agent_executor: AgentExecutor
    session_id: str = field(default_factory=lambda: str(ULID()))
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_activity: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    message_count: int = 0


class SessionStore:
    """In-memory session storage with TTL-based cleanup.

    Manages conversation sessions for the Agent Local Server.
    Sessions expire after a configurable TTL period of inactivity.
    Includes optional automatic background cleanup and max session limits.

    Attributes:
        sessions: Dictionary mapping session IDs to ServerSession objects.
        ttl_seconds: Time-to-live for sessions in seconds (default: 30 minutes).
        max_sessions: Maximum number of sessions allowed (default: 1000).
        cleanup_interval_seconds: Interval for automatic cleanup (default: 300).
    """

    def __init__(
        self,
        ttl_seconds: int = 1800,
        max_sessions: int = 1000,
        cleanup_interval_seconds: int = 300,
    ) -> None:
        """Initialize session store.

        Args:
            ttl_seconds: Session timeout in seconds. Default is 1800 (30 minutes).
            max_sessions: Maximum sessions before rejecting new ones. Default is 1000.
            cleanup_interval_seconds: Interval for auto-cleanup. Default is 300 (5 min).
        """
        self.sessions: dict[str, ServerSession] = {}
        self.ttl_seconds = ttl_seconds
        self.max_sessions = max_sessions
        self.cleanup_interval_seconds = cleanup_interval_seconds
        self._cleanup_task: asyncio.Task[None] | None = None

    @property
    def active_count(self) -> int:
        """Return count of active sessions."""
        return len(self.sessions)

    def get(self, session_id: str) -> ServerSession | None:
        """Retrieve a session by ID.

        Args:
            session_id: The session identifier to look up.

        Returns:
            The ServerSession if found, None otherwise.
        """
        return self.sessions.get(session_id)

    def get_all(self) -> list[ServerSession]:
        """Retrieve all active sessions.

        Returns:
            List of all active ServerSession objects.
        """
        return list(self.sessions.values())

    def create(
        self,
        agent_executor: AgentExecutor,
        session_id: str | None = None,
    ) -> ServerSession:
        """Create a new session with the given agent executor.

        Args:
            agent_executor: The AgentExecutor instance for this session.
            session_id: Optional custom session ID. If not provided, a new
                ULID will be generated. Useful for mapping external IDs
                (like AG-UI thread_id) to sessions.

        Returns:
            The newly created ServerSession.

        Raises:
            RuntimeError: If max_sessions limit is reached.
            ValueError: If session_id already exists.
        """
        if len(self.sessions) >= self.max_sessions:
            raise RuntimeError(
                f"Maximum session limit ({self.max_sessions}) reached. "
                "Try again later or increase max_sessions."
            )

        if session_id is not None and session_id in self.sessions:
            raise ValueError(f"Session with ID '{session_id}' already exists")

        # Pass session_id directly to constructor to avoid mutation
        if session_id is not None:
            session = ServerSession(
                agent_executor=agent_executor, session_id=session_id
            )
        else:
            session = ServerSession(agent_executor=agent_executor)

        self.sessions[session.session_id] = session
        return session

    def delete(self, session_id: str) -> bool:
        """Delete a session by ID.

        Args:
            session_id: The session identifier to delete.

        Returns:
            True if session was deleted, False if not found.
        """
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False

    def touch(self, session_id: str) -> None:
        """Update the last_activity timestamp for a session.

        This should be called on each request to prevent session expiration.

        Args:
            session_id: The session identifier to update.
        """
        session = self.sessions.get(session_id)
        if session:
            session.last_activity = datetime.now(timezone.utc)

    def cleanup_expired(self) -> int:
        """Remove all expired sessions.

        Sessions are considered expired if their last_activity timestamp
        is older than the configured TTL.

        Returns:
            Number of sessions removed.
        """
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(seconds=self.ttl_seconds)

        expired_ids = [
            session_id
            for session_id, session in self.sessions.items()
            if session.last_activity < cutoff
        ]

        for session_id in expired_ids:
            del self.sessions[session_id]

        return len(expired_ids)

    async def start_cleanup_task(self) -> None:
        """Start the background cleanup task.

        This should be called when the server starts to enable
        automatic cleanup of expired sessions.
        """
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            logger.info(
                f"Started session cleanup task (interval: "
                f"{self.cleanup_interval_seconds}s, TTL: {self.ttl_seconds}s)"
            )

    async def stop_cleanup_task(self) -> None:
        """Stop the background cleanup task.

        This should be called when the server stops to cleanly
        terminate the background task.
        """
        if self._cleanup_task is not None and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._cleanup_task
            logger.info("Stopped session cleanup task")
        self._cleanup_task = None

    async def _cleanup_loop(self) -> None:
        """Background loop that periodically cleans up expired sessions."""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval_seconds)
                count = self.cleanup_expired()
                if count > 0:
                    logger.info(f"Cleaned up {count} expired session(s)")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in session cleanup: {e}")
