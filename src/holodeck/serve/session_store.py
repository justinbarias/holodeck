"""Session management for the Agent Local Server.

Provides in-memory session storage with TTL-based expiration.
Sessions maintain conversation context across multiple requests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

from ulid import ULID

if TYPE_CHECKING:
    from holodeck.chat.executor import AgentExecutor


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

    Attributes:
        sessions: Dictionary mapping session IDs to ServerSession objects.
        ttl_seconds: Time-to-live for sessions in seconds (default: 30 minutes).
    """

    def __init__(self, ttl_seconds: int = 1800) -> None:
        """Initialize session store.

        Args:
            ttl_seconds: Session timeout in seconds. Default is 1800 (30 minutes).
        """
        self.sessions: dict[str, ServerSession] = {}
        self.ttl_seconds = ttl_seconds

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

    def create(self, agent_executor: AgentExecutor) -> ServerSession:
        """Create a new session with the given agent executor.

        Args:
            agent_executor: The AgentExecutor instance for this session.

        Returns:
            The newly created ServerSession.
        """
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
