# Quickstart: Vectorstore Reranking

**Branch**: `015-vectorstore-reranking`
**Date**: 2025-12-23

## Overview

This guide shows how to enable reranking for vectorstore tools to improve search result relevance. Reranking uses a specialized model to re-score and reorder initial vector search results.

## Prerequisites

- HoloDeck installed and configured
- An existing vectorstore tool configuration
- For Cohere: A Cohere API key
- For vLLM: A running vLLM server with a reranker model

## Quick Start with Cohere

### 1. Install reranking dependency

```bash
uv add holodeck-ai[rerank]
```

> **Note**: The same `[rerank]` extra works for both Cohere and vLLM since vLLM is Cohere API compatible.

### 2. Set your API key

```bash
export COHERE_API_KEY=your-api-key-here
```

### 3. Update your agent configuration

```yaml
# agent.yaml
name: my-agent
model:
  provider: openai
  name: gpt-4o

tools:
  - name: knowledge_search
    type: vectorstore
    description: Search product documentation
    source: data/docs/
    top_k: 5
    rerank: true  # Enable reranking
    reranker:
      provider: cohere
      api_key: ${COHERE_API_KEY}
      model: rerank-v3.5  # Optional, this is the default
```

### 4. Test it out

```bash
holodeck chat
```

## Quick Start with vLLM

### 1. Install reranking dependency (same as Cohere)

```bash
uv add holodeck-ai[rerank]
```

### 2. Start vLLM with a reranker model

```bash
vllm serve BAAI/bge-reranker-base --port 8000
```

### 3. Update your agent configuration

```yaml
# agent.yaml
name: my-agent
model:
  provider: openai
  name: gpt-4o

tools:
  - name: knowledge_search
    type: vectorstore
    description: Search product documentation
    source: data/docs/
    top_k: 5
    rerank: true
    reranker:
      provider: vllm
      url: http://localhost:8000
      model: BAAI/bge-reranker-base
```

## Configuration Options

### Basic Reranking

```yaml
rerank: true
reranker:
  provider: cohere
  api_key: ${COHERE_API_KEY}
```

### Custom Number of Candidates

By default, reranking considers `top_k * 3` candidates. Override this:

```yaml
rerank: true
rerank_top_n: 30  # Rerank top 30 results, return top_k
reranker:
  provider: cohere
  api_key: ${COHERE_API_KEY}
```

### Combine with Similarity Threshold

```yaml
top_k: 5
min_similarity_score: 0.6  # Applied before reranking
rerank: true
reranker:
  provider: cohere
  api_key: ${COHERE_API_KEY}
```

## Available Models

### Cohere Models

| Model | Description |
|-------|-------------|
| `rerank-v4.0-pro` | Highest quality, multilingual |
| `rerank-v4.0-fast` | Faster, multilingual |
| `rerank-v3.5` | Good balance (default) |
| `rerank-multilingual-v3.0` | Legacy multilingual |

### vLLM Models

Common open-source reranker models:
- `BAAI/bge-reranker-base`
- `BAAI/bge-reranker-large`
- `BAAI/bge-reranker-v2-m3`

## Troubleshooting

### Cohere API Key Error

```
RerankerAuthError: Invalid API key
```

**Solution**: Verify your `COHERE_API_KEY` environment variable is set correctly.

### vLLM Connection Error

```
RerankerConnectionError: Could not connect to vLLM server
```

**Solution**: Ensure your vLLM server is running and the URL is correct.

### Slow Performance

If reranking adds too much latency:

1. Reduce `rerank_top_n` to a lower value
2. Use a faster model (e.g., `rerank-v4.0-fast` for Cohere)
3. Use a smaller reranker model for vLLM

### Fallback Behavior

If the reranker fails during a search, HoloDeck automatically falls back to the original vector search results. You'll see a warning in the logs:

```
WARNING: Reranker failed, using original results: <error message>
```

## Complete Example

```yaml
# agent.yaml
name: customer-support-agent
description: AI assistant for customer support

model:
  provider: azure_openai
  name: gpt-4o
  deployment_name: gpt-4o-deployment

instructions:
  file: instructions.md

tools:
  - name: product_docs
    type: vectorstore
    description: Search product documentation and FAQs
    source: data/knowledge-base/
    chunk_size: 500
    chunk_overlap: 50
    top_k: 5
    min_similarity_score: 0.5
    rerank: true
    rerank_top_n: 20
    reranker:
      provider: cohere
      api_key: ${COHERE_API_KEY}
      model: rerank-v3.5
    database:
      provider: qdrant
      connection_string: http://localhost:6333
```

## Next Steps

- Read the [Cohere Rerank Best Practices](https://docs.cohere.com/docs/reranking-best-practices)
- Learn about [vLLM Reranker Setup](https://docs.vllm.ai/en/latest/serving/openai_compatible_server/)
- Explore [HoloDeck Evaluation Metrics](/docs/evaluations.md) to measure reranking quality
