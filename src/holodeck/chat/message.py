"""Message validation and orchestration."""

from __future__ import annotations

from holodeck.lib.validation import ValidationPipeline


class MessageValidator:
    """Validates user messages before sending to the agent.

    Uses ValidationPipeline to enforce content standards including
    empty message detection, size limits, control character filtering,
    and UTF-8 validation.
    """

    def __init__(self, max_length: int = 10_000) -> None:
        """Initialize validator with length constraints.

        Args:
            max_length: Maximum message length in characters. Defaults to 10,000.
        """
        self._pipeline = ValidationPipeline(max_length=max_length)

    def validate(self, message: str | None) -> tuple[bool, str | None]:
        """Validate a message and return validation status.

        Args:
            message: User message to validate (None, empty, or any content).

        Returns:
            Tuple of (is_valid: bool, error_message: str | None).
            If valid, error_message is None.
            If invalid, error_message describes the validation failure.

        Validation checks:
        - Message is not None or empty
        - Message does not exceed max_length
        - Message contains no control characters
        - Message is valid UTF-8
        """
        return self._pipeline.validate(message)
