# Reranker Interface Contract

**Branch**: `015-vectorstore-reranking`
**Date**: 2025-12-23

## Overview

This document defines the internal interface contract for reranker implementations.

**Key Design Decision**: vLLM's rerank endpoints are fully Cohere API compatible, so a single `CohereReranker` implementation handles both providers. The difference is only in the `base_url` parameter.

## BaseReranker Abstract Interface

**Location**: `src/holodeck/lib/rerankers/base.py`

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class RerankResult:
    """Result from reranking a single document.

    Attributes:
        index: Original index in the input documents list
        score: Relevance score normalized to [0.0, 1.0]
        document: Original document text (optional, for reference)
    """
    index: int
    score: float
    document: str | None = None


class BaseReranker(ABC):
    """Abstract base class for reranker implementations.

    All reranker providers must implement this interface to ensure
    consistent behavior across providers.
    """

    @abstractmethod
    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int | None = None,
    ) -> list[RerankResult]:
        """Rerank documents by relevance to query.

        Args:
            query: The search query to rank documents against.
            documents: List of document texts to rerank.
            top_n: Maximum number of results to return. If None, returns
                all documents reranked.

        Returns:
            List of RerankResult sorted by descending relevance score.
            The list length is min(len(documents), top_n) if top_n is set.

        Raises:
            RerankerError: If reranking fails (connection error, auth error, etc.)
        """
        pass

    @abstractmethod
    async def validate(self) -> None:
        """Validate reranker configuration and connectivity.

        Called during agent initialization to fail fast on configuration errors.

        Raises:
            RerankerError: If configuration is invalid or service unreachable.
        """
        pass

    @property
    @abstractmethod
    def provider(self) -> str:
        """Return the provider name for logging and metrics."""
        pass
```

## Error Handling Contract

**Location**: `src/holodeck/lib/rerankers/base.py`

### Error Handling Behavior

The `recoverable` attribute determines how `VectorStoreTool` handles reranker errors:

| Error Class | `recoverable` | Behavior |
|-------------|---------------|----------|
| `RerankerAuthError` | `False` | **Fail fast** - raise error to caller |
| `RerankerConnectionError` | `True` | **Fallback** - return original vector search results |
| `RerankerRateLimitError` | `True` | **Fallback** - return original vector search results |
| `RerankerError` (base) | `False` (default) | **Fail fast** - unless explicitly set recoverable |

**Rationale**: Authentication and configuration errors (non-recoverable) should fail fast to surface problems early. Transient errors (network, rate limits) should gracefully degrade to vector search results to maintain service availability.

```python
class RerankerError(Exception):
    """Base exception for reranker errors.

    Attributes:
        message: Human-readable error description
        provider: Reranker provider that raised the error
        recoverable: Whether the error is transient and may succeed on retry.
                     When True, VectorStoreTool falls back to vector search results.
                     When False, VectorStoreTool raises the error to fail fast.
    """

    def __init__(
        self,
        message: str,
        provider: str,
        recoverable: bool = False,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.provider = provider
        self.recoverable = recoverable


class RerankerAuthError(RerankerError):
    """Authentication or authorization error.

    Raised when API key is invalid or lacks permissions.
    Not recoverable without configuration change.

    Behavior: Fail fast (raise to caller)
    """

    def __init__(self, message: str, provider: str) -> None:
        super().__init__(message, provider, recoverable=False)


class RerankerConnectionError(RerankerError):
    """Network or connection error.

    Raised when reranker service is unreachable or request times out.
    May be recoverable on retry.

    Behavior: Fallback to vector search results
    """

    def __init__(self, message: str, provider: str) -> None:
        super().__init__(message, provider, recoverable=True)


class RerankerRateLimitError(RerankerError):
    """Rate limit exceeded.

    Raised when API rate limit is exceeded.
    Recoverable after backoff period.

    Behavior: Fallback to vector search results
    """

    def __init__(self, message: str, provider: str, retry_after: float | None = None) -> None:
        super().__init__(message, provider, recoverable=True)
        self.retry_after = retry_after
```

## CohereReranker Implementation Contract

**Location**: `src/holodeck/lib/rerankers/cohere.py`

This single implementation handles both Cohere cloud and vLLM servers.

```python
class CohereReranker(BaseReranker):
    """Cohere reranker implementation (works with both Cohere cloud and vLLM).

    vLLM's /v1/rerank endpoint is fully Cohere API compatible, so this single
    implementation handles both providers by varying the base_url.

    Args:
        api_key: Cohere API key (ignored for vLLM)
        model: Rerank model name (default: "rerank-v3.5")
        base_url: Optional custom base URL for vLLM servers.
                  If None, uses Cohere cloud API.
        timeout: Request timeout in seconds (default: 30.0)

    Raises:
        ImportError: If cohere package is not installed

    Examples:
        # Cohere cloud
        reranker = CohereReranker(api_key="co-xxx", model="rerank-v3.5")

        # vLLM server
        reranker = CohereReranker(
            api_key="ignored",
            model="BAAI/bge-reranker-base",
            base_url="http://localhost:8000"
        )
    """

    def __init__(
        self,
        api_key: str,
        model: str = "rerank-v3.5",
        base_url: str | None = None,
        timeout: float = 30.0,
    ) -> None: ...

    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int | None = None,
    ) -> list[RerankResult]:
        """Rerank using Cohere SDK (works with both Cohere and vLLM).

        Maps Cohere API response to RerankResult format.
        Scores are already normalized to [0, 1].
        """
        ...

    async def validate(self) -> None:
        """Validate connectivity by making a minimal rerank call."""
        ...

    @property
    def provider(self) -> str:
        """Return provider name based on whether base_url is set."""
        return "vllm" if self._base_url else "cohere"
```

## Factory Function Contract

**Location**: `src/holodeck/lib/rerankers/__init__.py`

```python
from holodeck.models.tool import (
    CohereRerankerConfig,
    RerankerConfig,
    VLLMRerankerConfig,
)


def create_reranker(config: RerankerConfig) -> BaseReranker:
    """Create a reranker instance from configuration.

    Both Cohere and vLLM configs create a CohereReranker instance,
    since vLLM is Cohere API compatible.

    Args:
        config: RerankerConfig (CohereRerankerConfig or VLLMRerankerConfig)

    Returns:
        Configured CohereReranker instance

    Raises:
        ImportError: If cohere package is not installed

    Examples:
        # Cohere cloud config
        config = CohereRerankerConfig(
            provider="cohere",
            api_key="co-xxx",
            model="rerank-v3.5"
        )
        reranker = create_reranker(config)  # base_url=None

        # vLLM config
        config = VLLMRerankerConfig(
            provider="vllm",
            url="http://localhost:8000",
            model="BAAI/bge-reranker-base"
        )
        reranker = create_reranker(config)  # base_url=config.url
    """
    from holodeck.lib.rerankers.cohere import CohereReranker

    if isinstance(config, CohereRerankerConfig):
        return CohereReranker(
            api_key=config.api_key,
            model=config.model,
            base_url=None,  # Use Cohere cloud
        )
    elif isinstance(config, VLLMRerankerConfig):
        return CohereReranker(
            api_key="vllm-placeholder",  # Ignored by vLLM
            model=config.model,
            base_url=config.url,
            timeout=config.timeout,
        )
    else:
        raise ValueError(f"Unsupported reranker config type: {type(config)}")
```

## Integration with VectorStoreTool

**Location**: `src/holodeck/tools/vectorstore_tool.py`

The VectorStoreTool will integrate reranking into the search flow:

```python
class VectorStoreTool:
    async def search(self, query: str) -> str:
        """Search with optional reranking.

        Flow:
        1. Calculate effective_rerank_top_n (rerank_top_n or top_k * 3)
        2. Perform vector search with effective_rerank_top_n limit
        3. If rerank enabled:
           a. Call reranker.rerank(query, documents, top_n=top_k)
           b. On failure, log warning and use original results
        4. Apply min_similarity_score filter
        5. Return top_k results
        """
        ...
```

## Logging Contract

All reranker implementations must log at DEBUG level:

```python
# On successful rerank
logger.debug(
    f"Reranker completed: provider={self.provider}, "
    f"input_docs={len(documents)}, output_docs={len(results)}, "
    f"latency_ms={latency_ms:.2f}"
)

# On failure (before raising)
logger.warning(
    f"Reranker failed: provider={self.provider}, "
    f"latency_ms={latency_ms:.2f}, error={error_message}"
)
```
