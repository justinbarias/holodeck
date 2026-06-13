# Tools

Tools are agent capabilities defined in `agent.yaml`. They work on both backends — the [OpenAI](openai-backend.md) backend (`openai`/`azure_openai`) and the [Claude](claude-backend.md) backend (`anthropic`/`ollama`) dispatch every tool type through the same loader.

## Quick start

A minimal agent with one function tool — deterministic enough to verify with `holodeck test`.

```yaml
# agent.yaml
name: math-agent

model:
  provider: openai
  name: gpt-4o-mini
  api_key: ${OPENAI_API_KEY}
  temperature: 0.0

instructions:
  inline: "You are a calculator. Use the tools to answer arithmetic questions."

tools:
  - name: subtract
    type: function
    description: "Compute a - b. Pass a and b as numeric strings; returns the difference as a string."
    file: tools/math.py
    function: subtract
```

```python
# tools/math.py
def subtract(a: str, b: str) -> str:
    """Compute ``a - b`` for two numeric string operands.

    Args:
        a: Minuend as a numeric string (e.g. ``"206588"``).
        b: Subtrahend as a numeric string.

    Returns:
        The difference ``a - b`` as a string.
    """
    return str(float(a.replace(",", "")) - float(b.replace(",", "")))
```

```bash
holodeck chat agent.yaml   # then ask: "What is 206588 minus 1500?"
# Agent calls subtract(a="206588", b="1500") → 205088.0
```

## How it works

HoloDeck supports five tool types: **function** (Python callables), **vectorstore** (semantic search), **hierarchical_document** (structure-aware hybrid search), **mcp** (Model Context Protocol servers), and **prompt** (LLM-powered semantic functions — schema-defined, execution still planned). Both backends load tools at agent-startup, derive each tool's schema from its config (function type hints, MCP server discovery, etc.), and surface them to the model. RAG tools (`vectorstore`, `hierarchical_document`) embed your data — see [Vector Stores](vector-stores.md) for provider setup. MCP servers run as subprocesses managed by the agent lifecycle — see the [MCP CLI](mcp-cli.md) guide for discovery and config. The sections below are condensed reference for each type.

### Common fields

All tools share `name` (1–100 chars, alphanumeric + underscores, unique on the agent — used to reference the tool in test cases and logs), `description` (≤500 chars, helps the model decide when to call it), and `type` (the discriminator that determines which extra fields apply).

```yaml
tools:
  - name: search_kb
    description: Search company knowledge base for answers
    type: vectorstore
```

| Tool Type               | Use case                                | Status         |
| ----------------------- | --------------------------------------- | -------------- |
| `function`              | Custom Python logic                     | ✅ Implemented |
| `vectorstore`           | Semantic search over data               | ✅ Implemented |
| `hierarchical_document` | Structure-aware hybrid document search  | ✅ Implemented |
| `mcp`                   | Model Context Protocol servers (stdio)  | ✅ Implemented |
| `prompt`                | LLM-powered semantic functions          | 🚧 Planned     |

---

## Function tools ✅

Execute custom Python functions defined in your project. Function tools are imported at agent-config load time via `importlib.util.spec_from_file_location`, validated as callable, and surfaced to the model with the function's signature and docstring as the tool's parameter schema and description. `src/holodeck/lib/function_tool_loader.py:load_function_tool` is shared by both backends.

**Use when:** custom business logic the agent calls deterministically; pure functions (math, parsing, formatting); lightweight integrations with libraries already in your venv; anything where an MCP subprocess would be overkill.

### Fields

| Field      | Type   | Required | Notes                                                                 |
|------------|--------|----------|-----------------------------------------------------------------------|
| `name`     | string | Yes      | Tool identifier (unique on agent)                                     |
| `description` | string | Yes   | Shown to the model — the function's docstring complements it          |
| `type`     | `"function"` | Yes | Discriminator                                                         |
| `file`     | string (path) | Yes | Python file containing the function. Resolved relative to the directory holding `agent.yaml` (absolute paths also work) |
| `function` | string | Yes      | Name of the callable inside `file`                                    |
| `parameters` | dict | No       | Optional schema for introspection. Backends derive JSON schema from type annotations when omitted |
| `defer_loading` | bool | No (`true`) | When `true`, excluded from the initial tool context and surfaced via tool filtering on demand. Set `false` for tools the agent should always have available |

See `src/holodeck/models/tool.py:FunctionTool` for the full Pydantic model. A complete working sample lives at `sample/financial-assistant/claude/` (wires `subtract` and `divide` into an agent).

### How loading works

At agent-startup (not first call), the shared loader:

1. Resolves `tool.file` relative to the agent project root (the directory holding `agent.yaml`); absolute paths pass through unchanged.
2. Builds an import spec and executes the module in isolation under a synthesized name (`holodeck_function_tool_<tool.name>`), so two tools with the same source file don't collide.
3. Looks up `tool.function` and verifies the attribute is callable.
4. Returns the callable to the backend, which wraps it as a native tool (an OpenAI Agents SDK function tool or a Claude SDK in-process MCP tool).

Any failure — missing file, import error, missing attribute, attribute that isn't callable — is re-raised as a `ConfigError` keyed on `tools.<tool.name>`, so the agent fails to start with a clear message instead of crashing mid-test (FR-025).

### Best practices

- Keep each function focused on a single task — small, well-named tools are easier for the model to pick.
- Add type hints on every parameter and the return, plus a short docstring; both feed the tool schema and description.
- Make the function self-contained — the loader executes the module fresh, so don't rely on side-effects from other modules.
- Return JSON-serializable values (strings, numbers, dicts, lists). Arbitrary objects fail when the backend serializes the result.
- Validate inputs and raise a meaningful `ValueError` — the runtime captures it as the tool error and surfaces it to the agent.
- Avoid long-running calls; prefer an MCP server when you need a separate process or async I/O at scale.
- Prefer pure functions — trivially testable in isolation (e.g., `tests/unit/sample/test_financial_tools.py`).

---

## Vectorstore tools ✅

Semantic search over unstructured or structured data — knowledge bases, FAQs, RAG context retrieval.

!!! warning "Anthropic users: embedding provider required"
    Anthropic does not provide embedding models. When using `model.provider: anthropic`
    with vectorstore tools, you **must** define `embedding_provider` at the agent level.
    See [Embedding Provider](agent-configuration.md#embedding-provider).

### Basic example

```yaml
- name: search-kb
  description: Search knowledge base for answers
  type: vectorstore
  source: knowledge_base/
```

### Fields

`source` (required) is the data file or directory to index. Supported: single files (`.txt`, `.md`, `.pdf`, `.json`, `.csv`), directories (recursively indexed), or remote URLs (auto-cached locally).

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `source` | path | — | **Required.** File, directory, or remote URL to index |
| `embedding_model` | string | provider default | e.g. `text-embedding-3-small`, `nomic-embed-text:latest` |
| `vector_field` | string \| list | auto-detect | Field(s) to vectorize (JSON/CSV). XOR with `vector_fields` |
| `meta_fields` | list | all fields | Metadata fields to include in results |
| `chunk_size` | int | `512` | Characters per chunk (> 0) |
| `chunk_overlap` | int | `0` | Characters to overlap between chunks (≥ 0) |
| `record_path` | string | — | Dot-path to array in nested JSON, e.g. `data.records` |
| `record_prefix` | string | — | Prefix added to record fields |
| `meta_prefix` | string | — | Prefix added to metadata fields |
| `database` | object \| string | in-memory | Provider config or named reference to `config.yaml` |

```yaml
- name: search-docs
  description: Search technical documentation
  type: vectorstore
  source: docs/
  embedding_model: text-embedding-3-small
  vector_field: [title, content]
  meta_fields: [source, date, url]
  chunk_size: 1024
  chunk_overlap: 128
```

!!! note "Embedding provider by backend"
    - **OpenAI / Azure OpenAI / Ollama** (OpenAI Agents or Claude backend): use the agent's own model provider for embeddings; the tool's `embedding_model` selects which embedding model from that provider.
    - **Anthropic** (Claude backend): Anthropic offers no embedding models — the agent-level `embedding_provider` is used instead, and `embedding_model` selects the model within it.

!!! danger "Mismatched embedding models across tools"
    If an agent defines both **vectorstore** and **hierarchical_document** tools with
    **different** `embedding_model` values, HoloDeck raises a validation error at startup.
    All embedding-based tools in an agent must share one embedding model (they share a
    single embedding provider instance).

### Database providers

HoloDeck supports multiple vector database backends through vector store connector libraries; switch providers via configuration without changing agent code. Provider options: `in-memory` (built-in, dev/test), `postgres` (pgvector), `qdrant`, `chromadb`, `pinecone`, `azure-ai-search`. Install all at once with `uv add holodeck-ai[vectorstores]`.

```yaml
# Inline
- name: search-kb
  type: vectorstore
  source: knowledge_base/
  database:
    provider: postgres
    connection_string: postgresql://user:password@localhost:5432/mydb
```

```yaml
# Named reference (database: my-postgres-store) defined in config.yaml
vectorstores:
  my-postgres-store:
    provider: postgres
    connection_string: ${DATABASE_URL}
```

For full provider setup (Docker, connection strings, Azure AI Search, env vars), see [Vector Stores](vector-stores.md).

### Data formats

- **Text** (`.txt`, `.md`): the whole file is chunked and vectorized.
- **JSON** (array of objects): each object is a record; `vector_field` picks the text field(s).
- **JSON** (nested): use `record_path: data.records` to reach the array.
- **CSV**: header row defines fields; `vector_field` selects which to embed.

---

## Hierarchical document tools ✅

Structure-aware hybrid document search combining semantic, keyword, and exact-match modalities with contextual embeddings for superior retrieval quality (Anthropic contextual retrieval approach — ~49% better retrieval).

!!! warning "Anthropic users: embedding provider required"
    Anthropic does not provide embedding models. When using `model.provider: anthropic`
    with hierarchical document tools, you **must** define `embedding_provider` at the agent level.
    See [Embedding Provider](agent-configuration.md#embedding-provider).

**Use when:** searching structured/legal/regulatory documents with section hierarchy; hybrid search; RAG needing contextual embeddings; documents with heading-based structure (markdown, PDF, Word).

### Basic example

```yaml
tools:
  - name: docs_search
    type: hierarchical_document
    description: "Search company documentation"
    source: "./docs/"
```

The tool automatically converts documents to markdown (markitdown), parses hierarchy from headings, chunks by structure (max 800 tokens), generates LLM context per chunk, embeds the contextualized text, and builds BM25 + exact-match indices.

### Fields

`source` (required) is a document file or directory (`.md`, `.pdf`, `.docx`, `.txt`; directories recursively indexed).

**Chunking**

| Field | Default | Range | Notes |
| --- | --- | --- | --- |
| `chunking_strategy` | `structure` | `structure`, `token` | `structure` parses markdown headings; `token` uses fixed splitting |
| `max_chunk_tokens` | `800` | 100–2000 | Max tokens per chunk |
| `chunk_overlap` | `50` | 0–200 | Token overlap between chunks |

**Document domain** (`document_domain`, default `none`) selects subsection-detection patterns: `us_legislative` ((a)(1)(A)(i)), `au_legislative` ((1)(a)(i)(A)), `academic` (1./1.1/1.1.1), `technical` (Step 1, Note:, Warning:), `legal_contract` (Article I, Section 1, (a) clauses). `max_subsection_depth` (int, default unlimited) caps nesting.

**Search**

| Field | Default | Range | Notes |
| --- | --- | --- | --- |
| `search_mode` | `hybrid` | `semantic`, `keyword`, `exact`, `hybrid` | Search modality |
| `top_k` | `10` | 1–100 | Results to return |
| `min_score` | None | 0.0–1.0 | Minimum similarity threshold |
| `semantic_weight` | `0.5` | 0.0–1.0 | Hybrid weight |
| `keyword_weight` | `0.3` | 0.0–1.0 | Hybrid weight |
| `exact_weight` | `0.2` | 0.0–1.0 | Hybrid weight |
| `rrf_k` | `60` | ≥ 1 | Reciprocal Rank Fusion constant |

In `hybrid` mode, the three weights must sum to 1.0. Pick modes by query type: `semantic` for concepts, `keyword` for technical terms, `exact` for section references (e.g. "Section 203(a)(1)"), `hybrid` for general purpose.

**Contextual embeddings** (`contextual_embeddings`, default `true`) prepend an LLM-generated summary to each chunk before embedding for ~49% better retrieval (Anthropic research); uses Claude Haiku by default. Tune with `context_max_tokens` (default 100, range 50–200) and `context_concurrency` (default 10, range 1–50). Same embedding-provider-by-backend rules as vectorstore tools apply.

**Feature extraction**: `extract_definitions` (default `true`) indexes term definitions; `extract_cross_references` (default `true`) resolves cross-references between sections.

**Reranking**: set `enable_reranking: true` and supply `reranker_model` (an LLMProvider config) for LLM-based reranking of results.

**Storage**: `database` (DatabaseConfig or named reference, default in-memory) — see [Vector Stores](vector-stores.md) for provider options. `keyword_index` configures the BM25 backend (default in-memory `rank_bm25`).

### Keyword index: OpenSearch (production)

For large corpora or multi-instance deployments, offload BM25 to an external OpenSearch cluster:

```yaml
keyword_index:
  provider: opensearch
  endpoint: "https://search.example.com:9200"
  index_name: "my-keyword-index"
  username: "${OPENSEARCH_USERNAME}"
  password: "${OPENSEARCH_PASSWORD}"
  verify_certs: true
  timeout_seconds: 10
```

Fields: `provider` (`in-memory` | `opensearch`), `endpoint`, `index_name`, `username`/`password` (basic auth) or `api_key`, `verify_certs` (default `true`), `timeout_seconds` (default 10, range 1–120). For local dev, run OpenSearch via Docker with `discovery.type=single-node` and `DISABLE_SECURITY_PLUGIN=true`, then use `endpoint: "http://localhost:9200"` with no auth. On Linux you may need `sudo sysctl -w vm.max_map_count=262144`.

### Example configurations

```yaml
# Legal/regulatory — section references + definitions
- name: legal_search
  type: hierarchical_document
  description: "Search legal documents"
  source: "./legal/"
  search_mode: hybrid
  document_domain: us_legislative
  extract_definitions: true
  extract_cross_references: true
  semantic_weight: 0.4
  keyword_weight: 0.3
  exact_weight: 0.3
```

```yaml
# Large corpus with persistence
- name: knowledge_base
  type: hierarchical_document
  description: "Persistent knowledge base"
  source: "./knowledge/"
  database:
    provider: postgres
    connection_string: "${POSTGRES_URL}"
  top_k: 20
```

### Result format

Each result includes a score, source path, location breadcrumb (`Chapter 5 > Section 5.3 > Reporting Requirements`), section id, the chunk text, and any relevant definitions:

```
[1] Score: 0.847 | Source: docs/compliance.md
Location: Chapter 5 > Section 5.3 > Reporting Requirements
Section: sec_5_3

Annual reporting must be submitted within 60 days of the fiscal year end.

Relevant definitions:
  • Covered entity: Any organization with annual revenue exceeding...
```

### Performance & cost

- Default 800-token chunks work well; increase for dense technical content, decrease for granular retrieval.
- Tune weights: more `semantic_weight` for concepts, `keyword_weight` for terminology, `exact_weight` for regulatory text.
- For >1000 documents, use a persistent database to avoid re-indexing on restart.
- Keep `contextual_embeddings` on (default) for the ~49% accuracy gain; cost is ~$0.03 per 100-page document via Claude Haiku. Raise `context_concurrency` to speed ingestion.
- Very large documents are auto-truncated for context while preserving chunk content.

---

## MCP tools ✅

Model Context Protocol (MCP) server integrations let agents interact with external systems through a standardized protocol (stdio transport). Browse available servers at the [official registry](https://github.com/modelcontextprotocol/servers) — filesystem, GitHub, Slack, PostgreSQL, and more.

For discovery, adding, listing, and removing servers via the CLI (`holodeck mcp search/add/list/remove`) and global-vs-agent config precedence, see the [MCP CLI](mcp-cli.md) guide. This section covers the `agent.yaml` shape.

### Basic example

```yaml
- name: filesystem
  description: Read and write files in the workspace
  type: mcp
  command: npx
  args: ["-y", "@modelcontextprotocol/server-filesystem", "./data"]
```

### Fields

`command` (required, stdio) is how the server launches — one of `npx` (npm packages, auto-installs), `node` (local `.js` files), `uvx` (Python packages via uv), `docker` (containers). `args` (required) are the command-line arguments, usually the package name plus config.

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `command` | enum | — | **Required.** `npx`, `node`, `uvx`, `docker` |
| `args` | list[str] | — | **Required.** Server package name + arguments |
| `transport` | enum | `stdio` | Only `stdio` is implemented |
| `config` | object | — | Server-specific config (passed via `MCP_CONFIG` env var) |
| `env` | object | — | Env vars for the server process; supports `${VAR}` substitution |
| `env_file` | path | — | Load env vars from a `.env` file (`env` overrides) |
| `request_timeout` | int | `30` | Per-request timeout (seconds) |
| `encoding` | string | `utf-8` | stdio character encoding |

```yaml
tools:
  - type: mcp
    name: filesystem
    description: Read and write files in the workspace data directory
    command: npx
    args: ["-y", "@modelcontextprotocol/server-filesystem", "./sample/data"]
    config:
      allowed_directories: ["./sample/data"]
    request_timeout: 30
```

### Sample servers

```yaml
# GitHub
- name: github
  type: mcp
  description: GitHub repository operations
  command: npx
  args: ["-y", "@modelcontextprotocol/server-github"]
  env:
    GITHUB_PERSONAL_ACCESS_TOKEN: "${GITHUB_TOKEN}"
```

```yaml
# Python server via uvx (fetch web content)
- name: mcp-server-fetch
  type: mcp
  description: Fetch web content
  command: uvx
  args: ["mcp-server-fetch"]
```

```yaml
# Local Node.js file
- name: my-custom-server
  type: mcp
  description: Custom local MCP server
  command: node
  args: ["./tools/my-mcp-server.js", "--config", "./config.json"]
```

```yaml
# Containerized
- name: custom-server
  type: mcp
  description: Custom containerized server
  command: docker
  args: ["run", "-i", "--rm", "my-mcp-server:latest"]
```

Other common servers: `@modelcontextprotocol/server-sqlite` (database queries), `@modelcontextprotocol/server-brave-search` (web search, needs `BRAVE_API_KEY`), `@modelcontextprotocol/server-puppeteer` (browser automation), `basic-memory` via uvx (short-term scratchpad across turns).

### Prerequisites

MCP tools require the runtime for your `command`: `npx`/`node` → Node.js 18+ ([nodejs.org](https://nodejs.org/)), `uvx` → [uv](https://astral.sh/uv), `docker` → [Docker](https://docker.com/). Verify with `node --version`, `uvx --version`, or `docker --version`.

### Lifecycle

MCP plugins are managed automatically: connected on agent startup, tools discovered and registered, and properly closed when the session ends. Always terminate chat sessions cleanly (`exit`/`quit`) so servers shut down.

---

## Prompt tools 🚧

> **Status:** Planned — configuration schema defined, execution not yet implemented.

LLM-powered semantic functions with Mustache-style template substitution: text generation with templates, specialized task prompts, reusable prompt chains, A/B testing.

### Fields

Provide either `template` (inline, ≤5000 chars, `{{variable}}` syntax) **or** `file` (path to an external template, relative to `agent.yaml`) — not both. `parameters` (required, ≥1) maps template variables to schemas the agent fills. Optional `model` overrides the agent's model for this tool.

```yaml
- name: code-reviewer
  description: Review code for best practices
  type: prompt
  file: prompts/code_review.txt
  model:
    provider: openai
    name: gpt-4
    temperature: 0.3
  parameters:
    code:
      type: string
      description: Code to review
    language:
      type: string
      description: Programming language
      enum: [python, javascript, go, java]
```

Template syntax supports simple variables (`{{name}}`), conditionals (`{{#if description}}...{{/if}}`), and loops (`{{#each items}}- {{this}}{{/each}}`).

---

## Tool comparison

| Feature        | Function        | Vectorstore             | Hierarchical Document       | MCP                     | Prompt          |
| -------------- | --------------- | ----------------------- | --------------------------- | ----------------------- | --------------- |
| **Status**     | ✅ Implemented  | ✅ Implemented          | ✅ Implemented               | ✅ Implemented          | 🚧 Planned      |
| **Use case**   | Custom logic    | Search data             | Structured document search  | External integrations   | Template-based  |
| **Execution**  | Python function | Vector similarity       | Hybrid RRF fusion           | MCP protocol (stdio)    | LLM generation  |
| **Setup**      | Python files    | Data files              | Document files              | Server config + runtime | Template text   |
| **Parameters** | Defined in code | Implicit (search query) | Implicit (search query)     | Server-specific tools   | Defined in YAML |
| **Latency**    | Low (<10ms)     | Medium (~100ms)         | Medium (~100-300ms)         | Medium (~50-500ms)      | High (LLM call) |
| **Cost**       | Internal        | Embedding API           | Embedding + context LLM     | Server resource         | LLM tokens      |

---

## Troubleshooting

**Function tools** — all resolution failures raise `ConfigError` at agent startup (keyed on `tools.<name>`), so a misconfiguration fails fast instead of mid-test:

- **File not found**: `tool.file` doesn't resolve (relative paths resolve against the agent project root).
- **Import failure**: the module raised during `exec_module` — the original traceback is chained.
- **Function not found / not callable**: no matching attribute, or the attribute isn't callable (e.g. a constant).
- **Runtime error inside the function**: caught by the backend and surfaced to the model as a tool error so the agent can recover.

**Vectorstore tools** — no data found returns empty results; invalid path or unsupported format errors at startup (config validation).

**Hierarchical document tools** — no results: check `source` path and supported format; slow ingestion: split very large PDFs or use persistent storage; missing context: ensure `contextual_embeddings: true` (default) and that documents have heading structure.

**MCP tools** — server unavailable, invalid config, or runtime (npx/uvx/docker) not found error at startup; connection timeout is configurable via `request_timeout`; runtime errors are returned as tool error responses to the LLM.

**Prompt tools** — invalid template errors at startup; LLM failure is a soft failure (logged, error message returned); template-rendering errors surface during execution.

### General best practices

Use descriptive names and clear descriptions (the model decides when to call a tool from them). Define parameters clearly, handle errors gracefully, test with realistic data, version tool files in source control, and include test cases that exercise each tool.

## Next steps

- [Agent Configuration](agent-configuration.md) — tool usage in context
- [Vector Stores](vector-stores.md) — RAG provider setup
- [MCP CLI](mcp-cli.md) — discovering and managing MCP servers
- [File References](file-references.md) — path resolution
- [Examples](../examples/) — complete tool usage
