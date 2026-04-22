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
    cache_creation_tokens: int = Field(default=0, ge=0)
    cache_read_tokens: int = Field(default=0, ge=0)

    @field_validator("total_tokens")
    @classmethod
    def validate_total(cls, value: int, info: ValidationInfo) -> int:
        """Ensure total_tokens >= prompt + completion.

        Relaxed from strict equality: cache reads may push total above
        prompt+completion at some providers (data-model.md §10a).
        """
        prompt = info.data.get("prompt_tokens", 0)
        completion = info.data.get("completion_tokens", 0)
        if value < prompt + completion:
            raise ValueError(
                f"total_tokens ({value}) must be >= "
                f"prompt_tokens ({prompt}) + completion_tokens ({completion})"
            )
        return value

    def __add__(self, other: object) -> "TokenUsage | NotImplementedType":
        """Add two TokenUsage instances element-wise over all four countable fields."""
        if not isinstance(other, TokenUsage):
            return NotImplemented
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            cache_creation_tokens=(
                self.cache_creation_tokens + other.cache_creation_tokens
            ),
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
        )

    @classmethod
    def zero(cls) -> "TokenUsage":
        """Create a TokenUsage instance with all counts at zero."""
        return cls(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            cache_creation_tokens=0,
            cache_read_tokens=0,
        )
