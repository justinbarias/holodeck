"""Cost accounting for the OpenAI Agents backend (``max_budget_usd``, FR-032).

Provides a ``RunHooks`` cost accountant that accumulates spend on every LLM-end
event from the run's per-response :class:`~agents.usage.Usage` multiplied by a
bundled, versioned per-model price table. When the accumulated cost exceeds the
configured ``max_budget_usd`` the hook raises
:class:`~holodeck.lib.backends.base.BackendBudgetExceededError` carrying the
partial response and the accumulated cost; the SDK ``Runner`` propagates the
exception, which the backend maps onto the standard ``ExecutionResult`` error
path (``is_error`` / ``error_reason``) with the partial response preserved.

The price table is a bundled constant (USD per 1M tokens). It is intentionally
allowed to go stale: an unknown model logs a single warning and the hook
degrades to a no-op for that model rather than crashing the run. Update the
table (``PRICE_TABLE_VERSION``) when OpenAI publishes new pricing.

Every ``import agents`` is performed lazily inside the factory so importing this
module never pulls the SDK (SC-005). The ``RunHooks`` subclass therefore lives
inside :func:`build_cost_hooks`, which imports the SDK base class and returns a
ready instance bound to a shared :class:`CostAccountant`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from holodeck.lib.backends.base import BackendBudgetExceededError

if TYPE_CHECKING:  # pragma: no cover - typing only, no runtime SDK import
    from agents import Agent, RunHooks
    from agents.items import ModelResponse
    from agents.run_context import RunContextWrapper
    from agents.usage import Usage

logger = logging.getLogger(__name__)

# Bump when the price table below is refreshed against published OpenAI pricing.
PRICE_TABLE_VERSION = "2026-06-13"


@dataclass(frozen=True)
class ModelPrice:
    """USD price per 1 million tokens for a single model.

    Attributes:
        input_usd_per_1m: Price per 1M (uncached) input tokens.
        output_usd_per_1m: Price per 1M output tokens.
        cached_input_usd_per_1m: Price per 1M cached input tokens. When unset,
            cached input tokens are billed at ``input_usd_per_1m``.
    """

    input_usd_per_1m: float
    output_usd_per_1m: float
    cached_input_usd_per_1m: float | None = None


# Bundled, versioned per-model price table (USD per 1M tokens). Prices reflect
# OpenAI's published list pricing for the current model families. Azure
# deployment names that embed a base model (e.g. ``gpt-4o-mini``) resolve via
# the same table; opaque deployment names fall through to the unknown-model
# warning path and disable enforcement for that model.
PRICE_TABLE: dict[str, ModelPrice] = {
    # gpt-5 family
    "gpt-5": ModelPrice(1.25, 10.0, cached_input_usd_per_1m=0.125),
    "gpt-5-mini": ModelPrice(0.25, 2.0, cached_input_usd_per_1m=0.025),
    "gpt-5-nano": ModelPrice(0.05, 0.4, cached_input_usd_per_1m=0.005),
    # gpt-4.1 family
    "gpt-4.1": ModelPrice(2.0, 8.0, cached_input_usd_per_1m=0.5),
    "gpt-4.1-mini": ModelPrice(0.4, 1.6, cached_input_usd_per_1m=0.1),
    "gpt-4.1-nano": ModelPrice(0.1, 0.4, cached_input_usd_per_1m=0.025),
    # gpt-4o family
    "gpt-4o": ModelPrice(2.5, 10.0, cached_input_usd_per_1m=1.25),
    "gpt-4o-mini": ModelPrice(0.15, 0.6, cached_input_usd_per_1m=0.075),
    # o-series reasoning models
    "o1": ModelPrice(15.0, 60.0, cached_input_usd_per_1m=7.5),
    "o1-mini": ModelPrice(1.1, 4.4, cached_input_usd_per_1m=0.55),
    "o3": ModelPrice(2.0, 8.0, cached_input_usd_per_1m=0.5),
    "o3-mini": ModelPrice(1.1, 4.4, cached_input_usd_per_1m=0.55),
    "o4-mini": ModelPrice(1.1, 4.4, cached_input_usd_per_1m=0.275),
}


def lookup_price(model_name: str) -> ModelPrice | None:
    """Resolve the price for *model_name*, tolerating dated/versioned suffixes.

    Matches the exact model id first, then progressively strips trailing
    ``-<suffix>`` segments so dated variants (``gpt-4o-2024-08-06``) resolve to
    their base entry (``gpt-4o``). Returns ``None`` when no prefix matches.

    Args:
        model_name: The configured model / Azure deployment name.

    Returns:
        The matched :class:`ModelPrice`, or ``None`` when unknown.
    """
    name = model_name.strip().lower()
    if name in PRICE_TABLE:
        return PRICE_TABLE[name]
    parts = name.split("-")
    while len(parts) > 1:
        parts.pop()
        candidate = "-".join(parts)
        if candidate in PRICE_TABLE:
            return PRICE_TABLE[candidate]
    return None


def usage_cost_usd(usage: Usage, price: ModelPrice) -> float:
    """Compute the USD cost of a single response's *usage* under *price*.

    Cached input tokens are billed at the model's cached rate (falling back to
    the standard input rate when the model has no cached price); the remaining
    input tokens and all output tokens are billed at their standard rates.

    Args:
        usage: The per-response :class:`~agents.usage.Usage`.
        price: The resolved price for the model that produced the response.

    Returns:
        The cost of this response in USD.
    """
    input_tokens = int(usage.input_tokens or 0)
    output_tokens = int(usage.output_tokens or 0)

    cached = 0
    details = usage.input_tokens_details
    if details is not None:
        cached = int(getattr(details, "cached_tokens", 0) or 0)
    # Cached tokens are a subset of input tokens; never bill them twice.
    cached = min(cached, input_tokens)
    uncached_input = input_tokens - cached

    cached_rate = (
        price.cached_input_usd_per_1m
        if price.cached_input_usd_per_1m is not None
        else price.input_usd_per_1m
    )
    return (
        uncached_input * price.input_usd_per_1m
        + cached * cached_rate
        + output_tokens * price.output_usd_per_1m
    ) / 1_000_000.0


@dataclass
class CostAccountant:
    """Mutable accumulator shared across the turns of one run or session.

    A single accountant is shared across every turn of a session so the budget
    covers the whole session, not just one turn. ``partial_response`` is updated
    on each LLM-end event so the budget error can surface whatever text the model
    produced before the cap tripped.

    Attributes:
        budget_usd: The configured ``max_budget_usd`` cap.
        accumulated_cost_usd: Running total spend in USD.
        partial_response: Latest assistant text snapshot seen on an LLM-end
            event (the most recent message content).
        _warned_models: Model names already warned about as unknown, so the
            warning is emitted once per model.
    """

    budget_usd: float
    accumulated_cost_usd: float = 0.0
    partial_response: str = ""
    _warned_models: set[str] = field(default_factory=set)

    def record(self, model_name: str, usage: Usage) -> None:
        """Accumulate the cost of one response and refresh the partial text.

        An unknown model logs a warning once and contributes no cost (the hook
        degrades to a no-op for that model rather than crashing the run).

        Args:
            model_name: The model / deployment name that produced the response.
            usage: The per-response usage to price and accumulate.
        """
        price = lookup_price(model_name)
        if price is None:
            if model_name not in self._warned_models:
                self._warned_models.add(model_name)
                logger.warning(
                    "max_budget_usd: no price for model '%s' (price table %s); "
                    "budget enforcement disabled for this model.",
                    model_name,
                    PRICE_TABLE_VERSION,
                )
            return
        self.accumulated_cost_usd += usage_cost_usd(usage, price)

    def over_budget(self) -> bool:
        """Return whether the accumulated cost has reached the budget cap."""
        return self.accumulated_cost_usd >= self.budget_usd


def _response_text(response: ModelResponse) -> str:
    """Extract concatenated assistant text from a ``ModelResponse``.

    Walks the response output items and joins the text of every
    ``ResponseOutputMessage`` content part. Returns an empty string when the
    response carries no assistant text (e.g. a tool-call-only turn).

    Args:
        response: The SDK ``ModelResponse`` from an LLM-end event.

    Returns:
        The concatenated assistant text, or ``""`` when there is none.
    """
    from openai.types.responses import ResponseOutputMessage, ResponseOutputText

    text = ""
    for item in response.output:
        if isinstance(item, ResponseOutputMessage):
            for part in item.content:
                if isinstance(part, ResponseOutputText):
                    text += part.text or ""
    return text


def build_cost_hooks(accountant: CostAccountant) -> RunHooks:
    """Build a ``RunHooks`` cost accountant bound to a shared *accountant*.

    The SDK base class is imported here (not at module import) so this module
    stays SDK-free (SC-005); the ``RunHooks`` subclass is defined inside this
    factory for the same reason. The returned hook accumulates spend on every
    LLM-end event and raises :class:`BackendBudgetExceededError` once the
    accumulated cost reaches the configured budget.

    Args:
        accountant: The shared accumulator. Reuse one accountant across a
            session's turns so the budget covers the whole session.

    Returns:
        A ``RunHooks`` instance ready to pass as ``Runner.run(..., hooks=...)``.
    """
    from agents import RunHooks as SDKRunHooks

    class _CostHooks(SDKRunHooks):
        """RunHooks that prices each LLM response and enforces the budget."""

        def __init__(self, accountant: CostAccountant) -> None:
            self._accountant = accountant

        async def on_llm_end(
            self,
            context: RunContextWrapper[object],
            agent: Agent[object],
            response: ModelResponse,
        ) -> None:
            """Price this response, refresh partial text, enforce the budget.

            Args:
                context: The run context wrapper (unused).
                agent: The agent that issued the call (its model names the price).
                response: The model response, carrying per-response usage.

            Raises:
                BackendBudgetExceededError: When the accumulated cost reaches the
                    configured budget.
            """
            del context
            text = _response_text(response)
            if text:
                self._accountant.partial_response = text
            model_name = _agent_model_name(agent)
            self._accountant.record(model_name, response.usage)
            if self._accountant.over_budget():
                raise BackendBudgetExceededError(
                    partial_response=self._accountant.partial_response,
                    accumulated_cost_usd=self._accountant.accumulated_cost_usd,
                    budget_usd=self._accountant.budget_usd,
                )

    return _CostHooks(accountant)


def _agent_model_name(agent: Agent[object]) -> str:
    """Return the model name for *agent* (deployment name for Azure models).

    The SDK ``Agent.model`` is either a model-name string (OpenAI) or an
    ``OpenAIResponsesModel`` carrying a ``model`` attribute (Azure). Both resolve
    to the string used to look up the price.

    Args:
        agent: The SDK agent that issued the LLM call.

    Returns:
        The model / deployment name, or ``""`` when it cannot be resolved.
    """
    model = getattr(agent, "model", None)
    if isinstance(model, str):
        return model
    name = getattr(model, "model", None)
    return name if isinstance(name, str) else ""
