"""Message validation and orchestration (placeholder)."""


class MessageValidator:
    """Validates user messages before sending to the agent."""

    def __init__(self, max_length: int = 10_000) -> None:
        """Initialize validator with length constraints."""
        self.max_length = max_length

    def validate(self, message: str) -> tuple[bool, str | None]:
        """Return (is_valid, error_message) for a given message."""
        if not message or message.isspace():
            return False, "Message cannot be empty."
        if len(message) > self.max_length:
            return False, "Message exceeds maximum length."
        return True, None
