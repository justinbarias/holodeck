"""Token usage tracking model."""

from types import NotImplementedType

from pydantic import BaseModel, Field, ValidationInfo, field_validator


class TokenUsage(BaseModel):
    """Token usage metadata.

    Supports arithmetic operations for accumulating token counts:
        total = TokenUsage.zero()
        total = total + new_usage  # or total += new_usage
    """

    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)

    @field_validator("total_tokens")
    @classmethod
    def validate_total(cls, value: int, info: ValidationInfo) -> int:
        """Ensure totals equal prompt + completion."""
        prompt = info.data.get("prompt_tokens", 0)
        completion = info.data.get("completion_tokens", 0)
        if value != prompt + completion:
            raise ValueError(
                f"total_tokens ({value}) must equal "
                f"prompt_tokens ({prompt}) + completion_tokens ({completion})"
            )
        return value

    def __add__(self, other: object) -> "TokenUsage | NotImplementedType":
        """Add two TokenUsage instances together.

        Args:
            other: Another TokenUsage instance to add.

        Returns:
            New TokenUsage with summed token counts, or NotImplemented if
            other is not a TokenUsage instance.
        """
        if not isinstance(other, TokenUsage):
            return NotImplemented
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )

    @classmethod
    def zero(cls) -> "TokenUsage":
        """Create a TokenUsage instance with all counts set to zero.

        Returns:
            TokenUsage with prompt_tokens=0, completion_tokens=0, total_tokens=0.
        """
        return cls(prompt_tokens=0, completion_tokens=0, total_tokens=0)
