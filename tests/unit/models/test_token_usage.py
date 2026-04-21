"""Unit tests for holodeck.models.token_usage module."""

import pytest
from pydantic import ValidationError

from holodeck.models.token_usage import TokenUsage


@pytest.mark.unit
class TestTokenUsage:
    """Tests for TokenUsage model."""

    def test_valid_token_usage(self) -> None:
        """Test creating a valid TokenUsage instance."""
        usage = TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        assert usage.prompt_tokens == 10
        assert usage.completion_tokens == 20
        assert usage.total_tokens == 30

    def test_invalid_total_raises_validation_error(self) -> None:
        """Test that total_tokens < prompt+completion raises ValidationError.

        Cache tokens may push the total above strict equality, so the
        validator enforces `>=` rather than `==` (data-model.md §10a).
        """
        with pytest.raises(ValidationError):
            TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=5)

    def test_total_above_prompt_plus_completion_allowed(self) -> None:
        """Relaxed validator: total above prompt+completion is valid (cache reads)."""
        usage = TokenUsage(
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=50,
            cache_read_tokens=20,
        )
        assert usage.total_tokens == 50

    def test_negative_tokens_raises_validation_error(self) -> None:
        """Test that negative token values raise ValidationError."""
        with pytest.raises(ValidationError):
            TokenUsage(prompt_tokens=-1, completion_tokens=0, total_tokens=-1)


@pytest.mark.unit
class TestTokenUsageAddition:
    """Tests for TokenUsage.__add__ method."""

    def test_add_two_token_usages(self) -> None:
        """Test adding two TokenUsage instances."""
        usage1 = TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        usage2 = TokenUsage(prompt_tokens=5, completion_tokens=15, total_tokens=20)
        result = usage1 + usage2
        assert result.prompt_tokens == 15
        assert result.completion_tokens == 35
        assert result.total_tokens == 50

    def test_add_with_zero_values(self) -> None:
        """Test adding TokenUsage with zero values (identity property)."""
        usage1 = TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        zero = TokenUsage.zero()
        result = usage1 + zero
        assert result.prompt_tokens == 10
        assert result.completion_tokens == 20
        assert result.total_tokens == 30

    def test_add_with_non_token_usage_returns_not_implemented(self) -> None:
        """Test that adding non-TokenUsage returns NotImplemented."""
        usage = TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        result = usage.__add__(42)  # type: ignore
        assert result is NotImplemented

    def test_iadd_works(self) -> None:
        """Test that += operator works with TokenUsage."""
        total = TokenUsage.zero()
        usage = TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        total += usage
        assert total.prompt_tokens == 10
        assert total.completion_tokens == 20
        assert total.total_tokens == 30


@pytest.mark.unit
class TestTokenUsageZero:
    """Tests for TokenUsage.zero class method."""

    def test_zero_returns_all_zeros(self) -> None:
        """Test that zero() returns TokenUsage with all counts at 0."""
        zero = TokenUsage.zero()
        assert zero.prompt_tokens == 0
        assert zero.completion_tokens == 0
        assert zero.total_tokens == 0
