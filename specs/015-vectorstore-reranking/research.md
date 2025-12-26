# Research: Vectorstore Reranking Extension

**Branch**: `015-vectorstore-reranking`
**Date**: 2025-12-23

## Overview

This document captures research findings for implementing reranking support in vectorstore tools. The research covers Cohere and vLLM reranking APIs, best practices, and integration patterns.

## Cohere Rerank API

### Decision
Use the official `cohere` Python SDK with async support via `AsyncClient`.

### Rationale
- First-party SDK with full async support matches HoloDeck's async patterns
- Handles authentication, retries, and error handling
- Supports latest models (rerank-v4.0-pro, rerank-v4.0-fast, rerank-v3.5)
- Relevance scores normalized to [0, 1] range

### API Details

**Request Parameters**:
- `model`: Model identifier (e.g., "rerank-v3.5", "rerank-v4.0-fast")
- `query`: The search query to rank documents against
- `documents`: List of text strings to rerank (max 1000 recommended)
- `top_n`: Optional limit on returned results (defaults to all)

**Response Structure**:
```python
response = await co.rerank(
    model="rerank-v3.5",
    query="search query",
    documents=["doc1", "doc2", "doc3"],
    top_n=10
)

for result in response.results:
    result.index        # Original document index
    result.relevance_score  # Score in [0, 1], higher = more relevant
    result.document     # Optional: original document text
```

**Available Models**:
- `rerank-v4.0-pro`: Highest quality, multilingual
- `rerank-v4.0-fast`: Faster, multilingual
- `rerank-v3.5`: Good balance of speed/quality, multilingual
- `rerank-multilingual-v3.0`: Legacy multilingual model

### Alternatives Considered
1. **Direct REST API**: Rejected - SDK provides better error handling and async support
2. **LangChain CohereRerank**: Rejected - adds unnecessary dependency, HoloDeck doesn't use LangChain

### Sources
- [Cohere Rerank Overview](https://docs.cohere.com/docs/rerank-overview)
- [Cohere Python SDK - Reranking API](https://deepwiki.com/cohere-ai/cohere-python/3.3-reranking-api)
- [Cohere Best Practices](https://docs.cohere.com/docs/reranking-best-practices)

---

## vLLM Rerank API

### Decision
Use the Cohere Python SDK with custom `base_url` pointing to vLLM server.

### Rationale
- vLLM's `/v1/rerank` and `/v2/rerank` endpoints are **fully Cohere API compatible**
- Single SDK for both Cohere cloud and vLLM self-hosted
- Reduces code complexity - one reranker implementation instead of two
- Cohere SDK handles async, retries, and error handling for both providers
- vLLM ignores the `api_key` parameter, so any placeholder value works

### API Compatibility

vLLM explicitly supports the Cohere SDK as documented in their examples:

```python
import cohere

# Cohere cloud (default base_url)
co_cloud = cohere.AsyncClient(api_key="real-cohere-key")

# vLLM server (custom base_url, fake api_key)
co_vllm = cohere.AsyncClient(
    base_url="http://localhost:8000",
    api_key="sk-fake-key"  # vLLM ignores this
)

# Same API call works for both!
result = await co_vllm.rerank(
    model="BAAI/bge-reranker-base",
    query="What is the capital of France?",
    documents=["Paris is the capital", "Reranking is useful"]
)
```

**Key Points**:
- vLLM supports `/rerank`, `/v1/rerank`, and `/v2/rerank` endpoints
- Response format identical to Cohere's format
- Supports BAAI/bge-reranker models and other cross-encoder models
- Self-hosted solution for privacy-conscious users

### Alternatives Considered
1. **Separate httpx client for vLLM**: Rejected - Cohere SDK works with vLLM, no need for separate implementation
2. **OpenAI SDK**: Rejected - `/v1/rerank` is not a standard OpenAI endpoint

### Sources
- [vLLM Cohere Rerank Client Example](https://docs.vllm.ai/en/stable/examples/online_serving/cohere_rerank_client.html)
- [vLLM GitHub - cohere_rerank_client.py](https://github.com/vllm-project/vllm/blob/main/examples/online_serving/cohere_rerank_client.py)
- [vLLM OpenAI-Compatible Server](https://docs.vllm.ai/en/latest/serving/openai_compatible_server/)

---

## Error Handling Strategy

### Decision
Implement graceful fallback to original vector search results on any reranker failure.

### Rationale
- Reranking is an enhancement, not a critical path requirement
- Users should still get results even if reranker is unavailable
- Aligns with FR-006 in spec

### Implementation Pattern
```python
async def search_with_rerank(self, query: str) -> list[QueryResult]:
    # Get initial vector search results (fetch more for reranking)
    candidates = await self._search_documents(query, top_n=self.rerank_top_n)

    if not self.config.rerank or not candidates:
        return candidates[:self.config.top_k]

    try:
        reranked = await self._reranker.rerank(query, candidates)
        return reranked[:self.config.top_k]
    except Exception as e:
        logger.warning(f"Reranker failed, using original results: {e}")
        return candidates[:self.config.top_k]
```

---

## Logging Strategy

### Decision
Log reranker calls at DEBUG level with latency and result counts.

### Rationale
- Aligns with FR-011 in spec
- Debug level doesn't clutter production logs
- Provides sufficient information for troubleshooting

### Implementation Pattern
```python
import time
import logging

logger = logging.getLogger(__name__)

async def rerank(self, query: str, documents: list[str]) -> list[RerankResult]:
    start_time = time.perf_counter()

    try:
        results = await self._call_reranker(query, documents)

        latency_ms = (time.perf_counter() - start_time) * 1000
        logger.debug(
            f"Reranker completed: provider={self.provider}, "
            f"input_docs={len(documents)}, output_docs={len(results)}, "
            f"latency_ms={latency_ms:.2f}"
        )
        return results
    except Exception as e:
        latency_ms = (time.perf_counter() - start_time) * 1000
        logger.warning(
            f"Reranker failed: provider={self.provider}, "
            f"latency_ms={latency_ms:.2f}, error={e}"
        )
        raise
```

---

## Default Configuration

### Decision
Default `rerank_top_n` to `top_k * 3`.

### Rationale
- Clarified during `/speckit.clarify` session
- Provides good balance of candidate diversity vs. latency
- 3x multiplier is common in reranking literature

### Implementation
```python
@property
def effective_rerank_top_n(self) -> int:
    """Get effective rerank_top_n with default calculation."""
    if self.config.rerank_top_n is not None:
        return self.config.rerank_top_n
    return self.config.top_k * 3
```

---

## Dependency Management

### Decision
Add `cohere` as an optional dependency under `[rerank]` extra (used for both Cohere and vLLM).

### Rationale
- Single SDK for both providers simplifies dependency management
- Not all users need reranking, so keep it optional
- Cohere SDK is lightweight and well-maintained

### pyproject.toml Update
```toml
[project.optional-dependencies]
rerank = ["cohere>=5.0.0"]
```

### Import Pattern
```python
def _get_cohere_client(self, base_url: str | None = None) -> Any:
    """Get Cohere client for reranking.

    Args:
        base_url: Optional custom base URL for vLLM servers.
                  If None, uses Cohere cloud API.
    """
    try:
        import cohere
    except ImportError as e:
        raise ImportError(
            "Reranking requires the cohere package. "
            "Install with: uv add holodeck-ai[rerank]"
        ) from e

    if base_url:
        # vLLM mode - use custom base_url, api_key is ignored
        return cohere.AsyncClient(
            base_url=base_url,
            api_key="vllm-placeholder"
        )
    else:
        # Cohere cloud mode
        return cohere.AsyncClient(api_key=self.api_key)
```

---

## Summary

| Topic | Decision | Key Rationale |
|-------|----------|---------------|
| Cohere SDK | Use `cohere.AsyncClient` | First-party, async support, error handling |
| vLLM Client | Use `cohere.AsyncClient` with custom `base_url` | vLLM is Cohere API compatible, single implementation |
| Error Handling | Graceful fallback | Reranking is enhancement, not critical path |
| Logging | DEBUG level with latency | Per FR-011, sufficient for troubleshooting |
| Default rerank_top_n | `top_k * 3` | Clarified during spec, common best practice |
| Dependencies | Optional `[rerank]` extra | Single SDK for both providers |

## Architecture Simplification

The discovery that vLLM is fully Cohere API compatible allows us to simplify from two implementations to one:

```
Before (planned):
├── lib/rerankers/
│   ├── base.py          # Abstract base class
│   ├── cohere.py        # Cohere SDK implementation
│   └── vllm.py          # httpx implementation

After (simplified):
├── lib/rerankers/
│   ├── base.py          # Abstract base class
│   └── cohere.py        # Single implementation for both providers
```

The `CohereReranker` class accepts an optional `base_url` parameter:
- `base_url=None`: Uses Cohere cloud API (default)
- `base_url="http://..."`: Uses vLLM server with Cohere-compatible API
