"""Chat session state management (placeholder)."""


class ChatSessionManager:
    """Maintains chat session lifecycle and history."""

    def __init__(self, max_messages: int = 50) -> None:
        """Initialize session manager with a message limit."""
        self.max_messages = max_messages
        self.history: list[str] = []

    def start(self) -> None:
        """Start a new chat session."""
        self.history.clear()

    def add_message(self, message: str) -> None:
        """Add a message to session history."""
        self.history.append(message)

    def warn_threshold(self) -> int | None:
        """Return warning threshold index when approaching limits."""
        if not self.max_messages:
            return None
        return int(self.max_messages * 0.8)

    def is_at_capacity(self) -> bool:
        """Return True when the session has reached the message limit."""
        return len(self.history) >= self.max_messages
