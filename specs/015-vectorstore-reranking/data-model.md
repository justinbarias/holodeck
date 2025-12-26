# Data Model: Vectorstore Reranking Extension

**Branch**: `015-vectorstore-reranking`
**Date**: 2025-12-23

## Overview

This document defines the data models for the vectorstore reranking feature. The design extends the existing `VectorstoreTool` configuration model and introduces new reranker configuration models.

## Entity Relationship Diagram

```
┌─────────────────────────┐
│    VectorstoreTool      │
│  (existing, extended)   │
├─────────────────────────┤
│ + rerank: bool          │
│ + rerank_top_n: int?    │
│ + reranker: RerankerCfg?│
└───────────┬─────────────┘
            │ 0..1
            ▼
┌─────────────────────────┐
│     RerankerConfig      │
│   (discriminated union) │
├─────────────────────────┤
│ provider: Literal[...]  │◄─── discriminator field
└───────────┬─────────────┘
            │
    ┌───────┴───────┐
    ▼               ▼
┌───────────┐ ┌───────────┐
│  Cohere   │ │   VLLM    │
│RerankerCfg│ │RerankerCfg│
├───────────┤ ├───────────┤
│ api_key   │ │ url       │
│ model     │ │ model     │
└───────────┘ └───────────┘
```

## Models

### VectorstoreTool (Extended)

**File**: `src/holodeck/models/tool.py`

Extends the existing `VectorstoreTool` model with reranking configuration.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| rerank | `bool` | `False` | Enable reranking of search results |
| rerank_top_n | `int \| None` | `None` | Number of candidates to rerank (default: `top_k * 3`) |
| reranker | `RerankerConfig \| None` | `None` | Reranker provider configuration |

**Validation Rules**:

*Pydantic Model Validation* (at configuration load time in `tool.py`):
- If `rerank=True` and `reranker=None`, raise `ValidationError` with message: "reranker configuration required when rerank is enabled"
- If `rerank_top_n` is provided, must be >= 1 and <= 1000

*Tool Initialization Validation* (at runtime in `vectorstore_tool.py`):
- If `rerank_top_n < top_k`, emit warning log: "rerank_top_n is less than top_k, reranking may not improve results"

**Implementation Note**: The `rerank_top_n < top_k` check is performed during `VectorStoreTool` initialization, not in the Pydantic model, because it requires comparing two fields that may have runtime-computed defaults (`rerank_top_n` defaults to `top_k * 3` when not specified).

---

### RerankerProvider (Enum)

**File**: `src/holodeck/models/tool.py`

```python
class RerankerProvider(str, Enum):
    """Supported reranker providers."""
    COHERE = "cohere"
    VLLM = "vllm"
```

---

### CohereRerankerConfig

**File**: `src/holodeck/models/tool.py`

Configuration for Cohere reranking provider.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| provider | `Literal["cohere"]` | `"cohere"` | Provider discriminator |
| api_key | `str` | (required) | Cohere API key (supports env var substitution) |
| model | `str` | `"rerank-v3.5"` | Cohere rerank model name |

**Validation Rules**:
- `api_key` must be non-empty after env var substitution
- `model` must be one of: `rerank-v4.0-pro`, `rerank-v4.0-fast`, `rerank-v3.5`, `rerank-multilingual-v3.0` (or any string for future models)

**YAML Example**:
```yaml
reranker:
  provider: cohere
  api_key: ${COHERE_API_KEY}
  model: rerank-v3.5
```

---

### VLLMRerankerConfig

**File**: `src/holodeck/models/tool.py`

Configuration for vLLM-hosted reranking provider.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| provider | `Literal["vllm"]` | `"vllm"` | Provider discriminator |
| url | `str` | (required) | vLLM server base URL |
| model | `str` | (required) | Model name served by vLLM |
| timeout | `float` | `30.0` | Request timeout in seconds |

**Validation Rules**:
- `url` must be a valid HTTP(S) URL
- `url` must not be empty
- `model` must not be empty
- `timeout` must be > 0

**YAML Example**:
```yaml
reranker:
  provider: vllm
  url: http://localhost:8000
  model: BAAI/bge-reranker-base
  timeout: 30.0
```

**Implementation Note**: Although vLLM has a separate config model for clarity, the underlying implementation uses the Cohere SDK with a custom `base_url`. vLLM's rerank endpoints are fully Cohere API compatible.

---

### RerankerConfig (Discriminated Union)

**File**: `src/holodeck/models/tool.py`

Union type for all reranker configurations, discriminated by `provider` field.

```python
RerankerConfig = Annotated[
    Annotated[CohereRerankerConfig, Tag("cohere")]
    | Annotated[VLLMRerankerConfig, Tag("vllm")],
    Discriminator("provider"),
]
```

---

### RerankResult (Internal)

**File**: `src/holodeck/lib/rerankers/base.py`

Internal result type returned by reranker implementations.

| Field | Type | Description |
|-------|------|-------------|
| index | `int` | Original index in input documents |
| score | `float` | Relevance score (0.0 to 1.0) |
| document | `str \| None` | Original document text (optional) |

```python
@dataclass
class RerankResult:
    """Result from reranking a document."""
    index: int
    score: float
    document: str | None = None
```

---

## State Transitions

The reranking feature does not introduce new state machines. It enhances the existing search flow:

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│    Query     │ ──▶ │Vector Search │ ──▶ │   Rerank     │ ──▶ │   Return     │
│              │     │(rerank_top_n)│     │ (if enabled) │     │   top_k      │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
                            │                    │
                            │                    ▼
                            │              ┌──────────────┐
                            │              │   Fallback   │
                            │              │ (on failure) │
                            │              └──────┬───────┘
                            │                     │
                            └─────────────────────┘
```

## YAML Configuration Examples

### Minimal Configuration (Cohere)

```yaml
tools:
  - name: knowledge_search
    type: vectorstore
    description: Search product documentation
    source: data/docs/
    top_k: 5
    rerank: true
    reranker:
      provider: cohere
      api_key: ${COHERE_API_KEY}
```

### Full Configuration (vLLM)

```yaml
tools:
  - name: knowledge_search
    type: vectorstore
    description: Search product documentation
    source: data/docs/
    top_k: 5
    min_similarity_score: 0.5
    rerank: true
    rerank_top_n: 20
    reranker:
      provider: vllm
      url: http://reranker.internal:8000
      model: BAAI/bge-reranker-large
      timeout: 60.0
```

### Reranking Disabled (Default)

```yaml
tools:
  - name: knowledge_search
    type: vectorstore
    description: Search product documentation
    source: data/docs/
    top_k: 5
    # rerank defaults to false, no reranker config needed
```

## Backward Compatibility

All new fields have defaults that preserve existing behavior:
- `rerank` defaults to `False`
- `rerank_top_n` defaults to `None` (calculated as `top_k * 3` at runtime)
- `reranker` defaults to `None`

Existing YAML configurations without reranking fields will continue to work unchanged.
