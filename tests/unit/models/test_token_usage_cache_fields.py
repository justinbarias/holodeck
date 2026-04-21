"""Tests for TokenUsage cache fields (T006).

data-model.md §10a adds `cache_creation_tokens` / `cache_read_tokens`
with ge=0 defaults, extends __add__ element-wise over all four fields,
and relaxes total_tokens to `>= prompt + completion`.
"""

import pytest
from pydantic import ValidationError

from holodeck.models.token_usage import TokenUsage


@pytest.mark.unit
class TestTokenUsageCacheFields:
    def test_cache_fields_default_zero(self) -> None:
        usage = TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        assert usage.cache_creation_tokens == 0
        assert usage.cache_read_tokens == 0

    def test_cache_fields_reject_negative(self) -> None:
        with pytest.raises(ValidationError):
            TokenUsage(
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                cache_creation_tokens=-1,
            )
        with pytest.raises(ValidationError):
            TokenUsage(
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                cache_read_tokens=-5,
            )

    def test_total_tokens_relaxed_ge_prompt_plus_completion(self) -> None:
        # Cache reads can push total above strict equality — must be accepted.
        usage = TokenUsage(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=42,
            cache_creation_tokens=3,
            cache_read_tokens=24,
        )
        assert usage.total_tokens == 42

    def test_total_tokens_below_prompt_plus_completion_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=14)

    def test_add_sums_all_four_fields(self) -> None:
        a = TokenUsage(
            prompt_tokens=1,
            completion_tokens=2,
            total_tokens=3,
            cache_creation_tokens=4,
            cache_read_tokens=5,
        )
        b = TokenUsage(
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            cache_creation_tokens=40,
            cache_read_tokens=50,
        )
        result = a + b
        assert result.prompt_tokens == 11
        assert result.completion_tokens == 22
        assert result.total_tokens == 33
        assert result.cache_creation_tokens == 44
        assert result.cache_read_tokens == 55

    def test_zero_returns_all_zero_cache_fields(self) -> None:
        z = TokenUsage.zero()
        assert z.cache_creation_tokens == 0
        assert z.cache_read_tokens == 0
