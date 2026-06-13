"""Unit tests for holodeck.lib.backends.openai_agents_cost.

The cost accountant is exercised with mocked SDK ``Usage`` / ``ModelResponse``
shapes — no network calls and no real credentials. The `openai-agents` package
is installed (dev extra) so the lazy SDK imports inside ``build_cost_hooks`` and
``_response_text`` resolve.
"""

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from holodeck.lib.backends.base import BackendBudgetExceededError
from holodeck.lib.backends.openai_agents_cost import (
    PRICE_TABLE,
    CostAccountant,
    ModelPrice,
    build_cost_hooks,
    lookup_price,
    usage_cost_usd,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _usage(
    input_tokens: int = 0,
    output_tokens: int = 0,
    cached_tokens: int = 0,
) -> SimpleNamespace:
    """Build a Usage-shaped object for the accountant under test."""
    return SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        input_tokens_details=SimpleNamespace(cached_tokens=cached_tokens),
    )


def _model_response(usage: object, text: str = "") -> object:
    """Build a ModelResponse-shaped object with optional assistant text.

    Text, when present, is wrapped in a real ``ResponseOutputMessage`` /
    ``ResponseOutputText`` pair so ``_response_text`` isinstance checks pass.
    """
    from openai.types.responses import ResponseOutputMessage, ResponseOutputText

    output: list[object] = []
    if text:
        part = ResponseOutputText(type="output_text", text=text, annotations=[])
        message = ResponseOutputMessage(
            id="msg-1",
            type="message",
            role="assistant",
            status="completed",
            content=[part],
        )
        output.append(message)
    return SimpleNamespace(output=output, usage=usage)


def _agent(model: object) -> object:
    """Build an SDK-agent-shaped object carrying a ``model`` attribute."""
    return SimpleNamespace(model=model)


# ---------------------------------------------------------------------------
# Price table + lookup
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLookupPrice:
    """Model-name resolution against the bundled price table."""

    @pytest.mark.parametrize(
        "name",
        [
            "gpt-5",
            "gpt-5-mini",
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4.1",
            "o1",
            "o3",
            "o4-mini",
        ],
    )
    def test_exact_models_priced(self, name: str) -> None:
        assert lookup_price(name) is PRICE_TABLE[name]

    def test_case_insensitive(self) -> None:
        assert lookup_price("GPT-4o") is PRICE_TABLE["gpt-4o"]

    def test_dated_variant_resolves_to_base(self) -> None:
        # A dated deployment/model id falls back to the base family entry.
        assert lookup_price("gpt-4o-2024-08-06") is PRICE_TABLE["gpt-4o"]

    def test_unknown_model_returns_none(self) -> None:
        assert lookup_price("some-opaque-deployment") is None


# ---------------------------------------------------------------------------
# Cost math
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestUsageCost:
    """Per-response cost computation under a known price."""

    def test_input_and_output_priced_per_million(self) -> None:
        price = ModelPrice(input_usd_per_1m=2.0, output_usd_per_1m=10.0)
        cost = usage_cost_usd(
            _usage(input_tokens=1_000_000, output_tokens=500_000), price
        )
        # 1M input @ $2 + 0.5M output @ $10 = $2 + $5 = $7.
        assert cost == pytest.approx(7.0)

    def test_cached_tokens_billed_at_cached_rate(self) -> None:
        price = ModelPrice(
            input_usd_per_1m=2.0,
            output_usd_per_1m=10.0,
            cached_input_usd_per_1m=0.5,
        )
        # 1M input of which 400k cached: 600k @ $2 + 400k @ $0.5 = $1.2 + $0.2.
        cost = usage_cost_usd(
            _usage(input_tokens=1_000_000, output_tokens=0, cached_tokens=400_000),
            price,
        )
        assert cost == pytest.approx(1.4)

    def test_cached_falls_back_to_input_rate_when_unset(self) -> None:
        price = ModelPrice(input_usd_per_1m=2.0, output_usd_per_1m=10.0)
        cost = usage_cost_usd(
            _usage(input_tokens=1_000_000, cached_tokens=400_000), price
        )
        # No cached rate → all input billed at $2 → $2.
        assert cost == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Accountant
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCostAccountant:
    """Accumulation + unknown-model degradation."""

    def test_record_accumulates_known_model(self) -> None:
        acc = CostAccountant(budget_usd=100.0)
        acc.record("gpt-4o", _usage(input_tokens=1_000_000, output_tokens=0))
        # gpt-4o input is $2.5 / 1M.
        assert acc.accumulated_cost_usd == pytest.approx(2.5)
        assert acc.over_budget() is False

    def test_over_budget_when_cost_reaches_cap(self) -> None:
        acc = CostAccountant(budget_usd=1.0)
        acc.record("gpt-4o", _usage(input_tokens=1_000_000))  # $2.5
        assert acc.over_budget() is True

    def test_unknown_model_warns_once_and_does_not_enforce(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        acc = CostAccountant(budget_usd=0.0001)
        with caplog.at_level(logging.WARNING):
            acc.record("opaque-deployment", _usage(input_tokens=1_000_000))
            acc.record("opaque-deployment", _usage(input_tokens=1_000_000))
        # No cost accrued; budget never enforced for the unknown model.
        assert acc.accumulated_cost_usd == 0.0
        assert acc.over_budget() is False
        warnings = [r for r in caplog.records if "no price for model" in r.message]
        # Warned exactly once despite two records.
        assert len(warnings) == 1


# ---------------------------------------------------------------------------
# RunHooks factory
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBuildCostHooks:
    """The RunHooks subclass prices responses and enforces the budget."""

    @pytest.mark.asyncio
    async def test_under_budget_does_not_raise(self) -> None:
        acc = CostAccountant(budget_usd=100.0)
        hooks = build_cost_hooks(acc)
        usage = _usage(input_tokens=1_000_000)  # gpt-4o → $2.5
        response = _model_response(usage, text="partial answer")
        await hooks.on_llm_end(MagicMock(), _agent("gpt-4o"), response)
        assert acc.accumulated_cost_usd == pytest.approx(2.5)
        assert acc.partial_response == "partial answer"

    @pytest.mark.asyncio
    async def test_over_budget_raises_with_partial_and_cost(self) -> None:
        acc = CostAccountant(budget_usd=1.0)
        hooks = build_cost_hooks(acc)
        usage = _usage(input_tokens=1_000_000)  # gpt-4o → $2.5 > $1.0
        response = _model_response(usage, text="the partial response")
        with pytest.raises(BackendBudgetExceededError) as excinfo:
            await hooks.on_llm_end(MagicMock(), _agent("gpt-4o"), response)
        err = excinfo.value
        assert err.partial_response == "the partial response"
        assert err.accumulated_cost_usd == pytest.approx(2.5)
        assert err.budget_usd == 1.0

    @pytest.mark.asyncio
    async def test_accumulates_across_calls_then_trips(self) -> None:
        acc = CostAccountant(budget_usd=4.0)
        hooks = build_cost_hooks(acc)
        # gpt-4o input $2.5/1M; two calls of 1M input = $5 total > $4.
        await hooks.on_llm_end(
            MagicMock(), _agent("gpt-4o"), _model_response(_usage(1_000_000), "a")
        )
        with pytest.raises(BackendBudgetExceededError):
            await hooks.on_llm_end(
                MagicMock(), _agent("gpt-4o"), _model_response(_usage(1_000_000), "b")
            )
        # Latest partial snapshot is preserved.
        assert acc.partial_response == "b"

    @pytest.mark.asyncio
    async def test_unknown_model_never_trips(self) -> None:
        acc = CostAccountant(budget_usd=0.0001)
        hooks = build_cost_hooks(acc)
        response = _model_response(_usage(1_000_000), text="x")
        # Unknown model contributes no cost → no raise despite a tiny budget.
        await hooks.on_llm_end(MagicMock(), _agent("mystery-model"), response)
        assert acc.accumulated_cost_usd == 0.0

    @pytest.mark.asyncio
    async def test_azure_model_object_resolves_deployment_name(self) -> None:
        acc = CostAccountant(budget_usd=100.0)
        hooks = build_cost_hooks(acc)
        # Azure path: agent.model is an OpenAIResponsesModel-like object whose
        # ``model`` attribute is the deployment name.
        azure_model = SimpleNamespace(model="gpt-4o")
        response = _model_response(_usage(1_000_000), text="ok")
        await hooks.on_llm_end(MagicMock(), _agent(azure_model), response)
        assert acc.accumulated_cost_usd == pytest.approx(2.5)
