"""Token usage tracking model."""

from pydantic import BaseModel, Field, ValidationInfo, field_validator


class TokenUsage(BaseModel):
    """Token usage metadata."""

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
