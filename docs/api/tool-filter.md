# Tool Filter API Reference

The tool filter subsystem implements
[Anthropic's Tool Search pattern](https://docs.anthropic.com/en/docs/build-with-claude/tool-use#tool-search)
for reducing token usage by dynamically filtering tools per request based on
semantic similarity to the user's query. Instead of sending every registered tool
to the LLM on each call, only the most relevant tools are included.

The subsystem lives in `holodeck.lib.tool_filter` and exposes four public symbols:
`ToolMetadata`, `ToolFilterConfig`, `ToolIndex`, and `ToolFilterManager`.

---

## Configuration

`ToolFilterConfig` is the Pydantic model that controls filtering behavior.
It is typically embedded in an agent's YAML configuration.

```yaml
tool_filter:
  enabled: true
  top_k: 5
  similarity_threshold: 0.3
  search_method: hybrid        # semantic | bm25 | hybrid
  always_include:
    - get_user_context
  always_include_top_n_used: 3
```

::: holodeck.lib.tool_filter.models.ToolFilterConfig
    options:
      docstring_style: google
      show_source: true

---

## Tool Metadata

`ToolMetadata` represents a single tool inside the index. It is created
automatically by `ToolIndex.build_from_kernel` and carries the embedding
vector (when available), parameter descriptions, and runtime usage counts.

::: holodeck.lib.tool_filter.models.ToolMetadata
    options:
      docstring_style: google
      show_source: true

---

## Tool Index

`ToolIndex` is the in-memory search index that holds all `ToolMetadata`
entries and supports three search strategies:

| Method     | Description                                             |
| ---------- | ------------------------------------------------------- |
| `semantic` | Cosine similarity over embedding vectors                |
| `bm25`     | Classic BM25 keyword scoring (no embeddings required)   |
| `hybrid`   | Reciprocal Rank Fusion of semantic and BM25 results     |

When the embedding service is unavailable, semantic search automatically
falls back to BM25.

::: holodeck.lib.tool_filter.index.ToolIndex
    options:
      docstring_style: google
      show_source: true

---

## Tool Filter Manager

`ToolFilterManager` is the main orchestrator. It wires together the
`ToolIndex`, the embedding service, and Semantic Kernel's
`FunctionChoiceBehavior` to transparently reduce the tool set on every
agent invocation.

### Typical lifecycle

```python
from holodeck.lib.tool_filter import ToolFilterConfig, ToolFilterManager

config = ToolFilterConfig(
    enabled=True,
    top_k=5,
    similarity_threshold=0.3,
    search_method="hybrid",
)

manager = ToolFilterManager(config, kernel, embedding_service)
await manager.initialize()

# Per-request filtering
filtered_tool_names = await manager.filter_tools("What's the weather?")

# Or apply directly to execution settings
settings = await manager.prepare_execution_settings(query, base_settings)

# After execution, record which tools the model actually called
manager.record_tool_usage(result.tool_calls)
```

::: holodeck.lib.tool_filter.manager.ToolFilterManager
    options:
      docstring_style: google
      show_source: true

---

## Module-Level Helpers

The `index` module also exposes two private helper functions used
internally by `ToolIndex`. They are not part of the public API but are
documented here for completeness.

::: holodeck.lib.tool_filter.index._cosine_similarity
    options:
      docstring_style: google
      show_source: true

::: holodeck.lib.tool_filter.index._tokenize
    options:
      docstring_style: google
      show_source: true
