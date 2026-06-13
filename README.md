# 🧪 HoloDeck

**Build, Test, and Deploy AI Agents — No Code Required**

HoloDeck is an open-source experimentation platform that enables teams to create, evaluate, and deploy AI agents through simple YAML configuration. Go from hypothesis to production API in minutes, not weeks.

[![PyPI version](https://badge.fury.io/py/holodeck-ai.svg)](https://badge.fury.io/py/holodeck-ai)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

---

## ✨ Features

- **🎯 No-Code Agent Definition** - Define agents using simple YAML configuration
- **🧠 Claude Native** - First-class Anthropic integration (extended thinking, subagents, native tool bridging); also serves local Ollama models
- **⚡ OpenAI Agents Native** - First-class OpenAI / Azure OpenAI backend on the OpenAI Agents SDK — function + RAG tools, MCP, structured output, reasoning effort, budgets, and model fallback
- **🔀 Multi-Backend Architecture** - Automatic routing by `model.provider`: `openai` / `azure_openai` → OpenAI Agents backend, `anthropic` / `ollama` → Claude backend
- **🧪 Hypothesis-Driven Testing** - Test agent behaviors against structured test cases
- **📊 Integrated Evaluations** - DeepEval LLM-as-judge metrics (GEval, RAG) plus NLP metrics (F1, BLEU, ROUGE)
- **📈 Evaluation Dashboard** - `holodeck test view` launches an interactive Dash UI for run history, regression detection, prompt-version drift, and side-by-side run comparison
- **🔌 Tool Ecosystem** - Extend agents with MCP servers, vector store search, and hierarchical document tools
- **💾 RAG Support** - Native vector database integration (ChromaDB, Qdrant, PostgreSQL, Pinecone)
- **🤖 Open-Source First** - Designed to work with Ollama for local, free inference

---

## 🚀 Quick Start

### Installation

```bash
pip install holodeck-ai
```

The Claude backend (`anthropic` / `ollama`) is included. For the OpenAI / Azure backend, add the `openai-agents` extra (and a vector-store extra such as `qdrant` for RAG):

```bash
pip install 'holodeck-ai[openai-agents]'
# with RAG: pip install 'holodeck-ai[openai-agents,qdrant]'
```

### Create Your First Agent

Use the interactive wizard to create a new agent:

```bash
# Start the interactive wizard
holodeck init
```

The wizard guides you through configuration:

```
? Enter agent name: research-agent
? Select LLM provider: Ollama (Local, llama3.2:latest)
? Select vector store: ChromaDB (Local, http://localhost:8000)
? Select evaluation metrics: rag-faithfulness, rag-answer_relevancy
? Select MCP servers: Brave Search, Memory, Sequential Thinking
```

You can also use command-line options:

```bash
# Pre-select template
holodeck init --template research

# Non-interactive mode with defaults
holodeck init --name my-agent --llm ollama --non-interactive
```

This creates:

```
research-agent/
├── agent.yaml              # Agent configuration
├── instructions/
│   └── system-prompt.md   # Agent instructions
├── data/                  # Grounding data (optional)
└── tools/                 # Tool configuration
```

### Define Your Agent

Edit `agent.yaml`:

```yaml
name: "research-agent"
description: "Research assistant that finds and synthesizes information"

model:
  provider: ollama
  name: llama3.2:latest
  temperature: 0.3

instructions:
  file: instructions/system-prompt.md

tools:
  # Vector store for semantic search
  - name: search_papers
    type: vectorstore
    source: data/papers_index.json
    description: "Search research papers and documents"
    database:
      provider: chromadb
      connection_string: http://localhost:8000

  # MCP server for web search
  - name: brave_search
    type: mcp
    description: "Search the web using Brave Search"
    command: npx
    args: ["-y", "@brave/brave-search-mcp-server"]

evaluations:
  model:
    provider: ollama
    name: llama3.2:latest
  metrics:
    - type: rag
      metric_type: faithfulness
      threshold: 0.8
    - type: rag
      metric_type: answer_relevancy
      threshold: 0.7
    - type: geval
      name: "Coherence"
      criteria: "Evaluate whether the response is clear and well-structured."
      threshold: 0.7

test_cases:
  - input: "Find recent papers on machine learning"
    expected_tools: ["search_papers"]

  - input: "What are the latest trends in AI research?"
    expected_tools: ["brave_search"]
    ground_truth: "Should summarize current AI research trends"
```

### Define a Claude-Native Agent

HoloDeck auto-selects the Claude Agent SDK backend when `model.provider` is `anthropic`. No code changes needed — just configure via YAML:

> **Prerequisites:** Node.js 18+, `ANTHROPIC_API_KEY` environment variable, and a separate `embedding_provider` for vectorstore tools (Anthropic does not provide embeddings).

```yaml
name: "claude-research-agent"
description: "Research assistant powered by Claude"

model:
  provider: anthropic
  name: claude-sonnet-4-20250514
  temperature: 0.3

# Anthropic can't generate embeddings — specify an external embedding provider
embedding_provider:
  provider: ollama
  name: nomic-embed-text:latest

instructions:
  file: instructions/system-prompt.md

claude:
  permission_mode: manual # manual | acceptEdits | acceptAll
  extended_thinking:
    enabled: true
    budget_tokens: 10000
  web_search: true # Built-in web search capability
  bash:
    enabled: false
  file_system:
    read: true
    write: false

tools:
  - name: search_papers
    type: vectorstore
    source: data/papers_index.json
    description: "Search research papers"
    database:
      provider: chromadb
      connection_string: http://localhost:8000

  - name: brave_search
    type: mcp
    description: "Search the web using Brave Search"
    command: npx
    args: ["-y", "@brave/brave-search-mcp-server"]
```

### Test Your Agent

```bash
# Run test cases with evaluations (uses agent.yaml by default)
cd research-agent
holodeck test

# Interactive chat session
holodeck chat

# With verbose output
holodeck chat --verbose
```

### Visualise Run History — `holodeck test view`

![HoloDeck evaluation dashboard — Summary view](docs/assets/dashboard/summary.png)

HoloDeck persists every `holodeck test` invocation as a timestamped JSON under `results/<agent-slug>/`. Launch the interactive dashboard to explore them:

```bash
# Install the optional extra once
pip install 'holodeck-ai[dashboard]'

# Then from any agent directory
holodeck test view
# → http://127.0.0.1:8501/

# Demo the UI with the built-in seed dataset (no real runs required)
holodeck test view --seed
```

The dashboard reloads dynamically — leave `holodeck test view` running in one terminal, iterate with `holodeck test` in another, and new runs appear within ~5 s without a restart. See the [Dashboard guide](docs/guides/dashboard.md) for the full feature list (Summary / Explorer / Compare views, filters, prompt-version boundaries).

### Version Your System Prompt

When `instructions.file` is used, optional YAML frontmatter at the top of the prompt file versions and labels the run:

```markdown
---
version: v1.2.0
author: your-name
description: Shortened citation format; stricter refusal on off-topic.
tags:
  - rag
  - customer-support
---

# System Prompt

You are a customer support specialist.
...
```

The dashboard groups runs by `version`, draws a boundary marker on the pass-rate chart whenever the version changes, and surfaces `tags` as filter chips. If you omit `version:`, HoloDeck derives `auto-<sha256[:8]>` from the body so every run still has a stable id.

**Output:**

```
🧪 Running HoloDeck Tests...

✅ Test 1/2: Find recent papers on machine learning
   Faithfulness: 0.92 (threshold: 0.8) ✓
   Answer Relevancy: 0.88 (threshold: 0.7) ✓
   Coherence: 0.85 (threshold: 0.7) ✓
   Tools Used: [search_papers] ✓

✅ Test 2/2: What are the latest trends in AI research?
   Faithfulness: 0.89 (threshold: 0.8) ✓
   Answer Relevancy: 0.91 (threshold: 0.7) ✓
   Coherence: 0.87 (threshold: 0.7) ✓
   Tools Used: [brave_search] ✓

📊 Overall Results: 2/2 passed (100%)
```

---

## 🛠️ Development

### Prerequisites

- Python 3.10 or higher
- Git
- UV (package manager)

### Getting Started

```bash
# Clone the repository
git clone https://github.com/justinbarias/holodeck.git
cd holodeck

# Initialize development environment
make init

# Activate virtual environment
source .venv/bin/activate
```

### Running Tests

```bash
# Run all tests
make test

# Run unit tests only
make test-unit

# Run integration tests only
make test-integration

# Run tests with coverage report
make test-coverage

# Run failed tests only
make test-failed

# Run tests in parallel
make test-parallel
```

### Code Quality

```bash
# Format code with Black + Ruff
make format

# Check formatting (CI-safe)
make format-check

# Run linting
make lint

# Auto-fix linting issues
make lint-fix

# Type checking with MyPy
make type-check

# Security scanning
make security

# Run complete CI pipeline locally
make ci
```

### Pre-commit Hooks

```bash
# Install pre-commit hooks
make install-hooks

# Run hooks on all files
make pre-commit
```

### Code Style

HoloDeck follows the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html) with:

- **Formatting:** Black (88 character line length)
- **Linting:** Ruff (comprehensive rule set)
- **Type Checking:** MyPy (strict mode)
- **Security:** Bandit, Safety, detect-secrets
- **Target:** Python 3.10+

### Full Contributing Guide

For detailed development instructions, commit message format, PR workflow, and troubleshooting, see [**docs/contributing.md**](docs/contributing.md).

---

## 📖 Core Concepts

### Agent Definition

Agents are defined using declarative YAML configuration. HoloDeck automatically selects the correct backend based on `model.provider` — the OpenAI Agents SDK for `openai` / `azure_openai`, the Claude Agent SDK for `anthropic` / `ollama`:

```yaml
name: "research-agent"
model:
  provider: ollama
  name: llama3.2:latest
  temperature: 0.3
instructions: |
  You are a research assistant that helps users find
  accurate information from trusted sources.
tools:
  - type: vectorstore
    name: search_papers
  - type: mcp
    name: brave_search
```

### Tools

Extend agent capabilities with vector search and MCP tools:

#### Vector Store Tools

Enable semantic search over your documents and data:

```yaml
tools:
  - name: search_docs
    type: vectorstore
    description: "Search knowledge base for relevant information"
    source: data/documents/
    embedding_model: nomic-embed-text:latest
    database:
      provider: chromadb
      connection_string: http://localhost:8000
```

**Supported Vector Stores:**

- **ChromaDB** - Lightweight, Python-native (recommended for development)
- **PostgreSQL pgvector** - Production-grade with SQL capabilities
- **Qdrant** - High-performance vector database
- **Pinecone** - Serverless managed cloud

#### MCP (Model Context Protocol) Tools

HoloDeck supports the Model Context Protocol for standardized tool integration:

```yaml
tools:
  - name: brave_search
    type: mcp
    description: "Search the web using Brave Search"
    command: npx
    args: ["-y", "@brave/brave-search-mcp", "${BRAVE_API_KEY}"]

  - name: filesystem
    type: mcp
    description: "Read and write files"
    command: npx
    args: ["-y", "@modelcontextprotocol/server-filesystem"]

  - name: memory
    type: mcp
    description: "Persistent memory for conversations"
    command: npx
    args: ["-y", "@modelcontextprotocol/server-memory"]
```

**MCP Server Management:**

```bash
# Search for MCP servers
holodeck mcp search filesystem

# Add an MCP server to your agent
holodeck mcp add io.github.modelcontextprotocol/server-filesystem

# List installed servers
holodeck mcp list
```

### Evaluations

Built-in evaluation metrics powered by DeepEval with support for local models:

**DeepEval Metrics (Recommended):**

- **GEval** - Custom criteria evaluation using chain-of-thought prompting
- **RAG Faithfulness** - Detect hallucinations by comparing response to retrieved context
- **RAG Answer Relevancy** - Measure how well responses address the user's question
- **RAG Context Precision** - Evaluate retrieval ranking quality

**NLP Metrics (Standard):**

- **F1 Score** - Precision and recall balance
- **BLEU** - Translation/generation quality
- **ROUGE** - Summarization quality
- **METEOR** - Semantic similarity

**Configuration:**

```yaml
evaluations:
  model:
    provider: ollama
    name: llama3.2:latest
    temperature: 0.0

  metrics:
    # DeepEval GEval - custom criteria
    - type: geval
      name: "Coherence"
      criteria: "Evaluate whether the response is clear and well-structured."
      threshold: 0.7

    # DeepEval RAG metrics
    - type: rag
      metric_type: faithfulness
      threshold: 0.8

    - type: rag
      metric_type: answer_relevancy
      threshold: 0.7

    # NLP metrics (no LLM required)
    - type: standard
      metric: f1_score
      threshold: 0.7
```

### Test Cases

Define structured test scenarios with support for multimodal inputs:

#### Basic Text Test Cases

```yaml
test_cases:
  - name: "Research query"
    input: "Find papers on machine learning optimization"
    expected_tools: ["search_papers"]

  - name: "Web search"
    input: "What are the latest AI trends?"
    ground_truth: "Summary of current AI research trends"
    expected_tools: ["brave_search"]
    evaluations:
      - type: rag
        metric_type: faithfulness
      - type: standard
        metric: f1_score
```

#### Multimodal Test Cases with Files

**Image Input:**

```yaml
test_cases:
  - name: "Image analysis"
    input: "Describe what is shown in this image"
    files:
      - path: tests/fixtures/diagram.jpg
        type: image
        description: "Architecture diagram"
    ground_truth: "The image shows a system architecture diagram"
```

**PDF Document Input:**

```yaml
test_cases:
  - name: "Document analysis"
    input: "Summarize the key points in this document"
    files:
      - path: tests/fixtures/report.pdf
        type: pdf
        description: "Research report"
    expected_tools: ["search_papers"]
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    HOLODECK PLATFORM                     │
└─────────────────────────────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  Backend     │  │  Evaluation  │  │  Deployment  │
│  Abstraction │  │  Framework   │  │  & Serving   │
└──────────────┘  └──────────────┘  └──────────────┘
   ┌────┴────┐         │                  │
   ▼         ▼         │                  │
┌──────┐ ┌──────┐      │                  │
│OpenAI│ │Claude│      ├─ DeepEval       ├─ FastAPI
│Agents│ │Backend│      ├─ NLP Metrics    ├─ Docker
└──────┘ └──────┘      ├─ Custom GEval   ├─ Cloud Deploy
   │         │         └─ Reporting      └─ Monitoring
   ├─OpenAI  ├─ Anthropic (Claude)
   ├─Azure   ├─ Ollama (local)
   ├─MCP     ├─ Tool Adapters / MCP Bridge
   └─RAG     └─ OTel Bridge
```

---

## 🎯 Use Cases

### Research Assistant

```bash
holodeck init research --template research
# Pre-configured with: Paper search, MCP web search, RAG evaluations
```

### Customer Support Agent

```bash
holodeck init support --template customer-support
# Pre-configured with: FAQ vectorstore, structured issue data
```

### Conversational Agent

```bash
holodeck init chatbot --template conversational
# Pre-configured with: Simple Q&A, FAQ vectorstore
```

---

## 📊 Monitoring & Observability

HoloDeck ships **native OpenTelemetry** instrumentation that follows the [Semantic Conventions for Generative AI](https://opentelemetry.io/docs/specs/semconv/gen-ai/). Traces, metrics, and logs are emitted on both backends — turn it on with an `observability` block in your `agent.yaml`.

### Basic configuration

```yaml
# agent.yaml
observability:
  enabled: true
  service_name: customer-support-agent

  traces:
    enabled: true
    sample_rate: 1.0          # Sample 100% of traces
    capture_content: true     # Capture prompt/response bodies (omit for PII-sensitive workloads)

  metrics:
    enabled: true
    export_interval_ms: 5000  # Export metrics every 5s

  logs:
    enabled: true
    level: INFO
```

### Exporters

Configure one or more exporters under `observability.exporters`. The built-in
exporters are `console`, `otlp`, `prometheus`, and `azure_monitor`.

```yaml
observability:
  enabled: true
  service_name: customer-support-agent
  exporters:
    # Local debugging — pretty-print spans to stdout
    console:
      enabled: true

    # OTLP — Jaeger, Grafana Tempo, Aspire Dashboard, Datadog, etc.
    otlp:
      enabled: true
      endpoint: ${OTEL_EXPORTER_OTLP_ENDPOINT}   # e.g. http://localhost:4317
      protocol: grpc                              # or http/protobuf
      insecure: true                              # plaintext for local collectors
      headers:                                    # e.g. vendor auth headers
        x-api-key: ${OTEL_API_KEY}

    # Prometheus — scrape metrics from a pull endpoint
    prometheus:
      enabled: true
      port: 8889

    # Azure Monitor / Application Insights
    azure_monitor:
      enabled: true
      connection_string: ${APPLICATIONINSIGHTS_CONNECTION_STRING}
```

### What gets instrumented

Following the GenAI semantic conventions, HoloDeck records spans and metrics for
model calls (token usage, latency, model name), tool invocations, and
evaluation runs — so you can trace a full agent turn end-to-end in your existing
observability stack.

---

## 🗺️ Roadmap

- [x] **v0.1** - Core agent engine + CLI
- [x] **v0.2** - Evaluation framework (DeepEval, NLP), Tools (MCP, Vectorstore)
- [x] **v0.3** - Claude Agent SDK native backend, multi-backend abstraction, API deployment, OpenTelemetry observability
- [x] **v0.6** - OpenAI Agents SDK native backend (OpenAI + Azure OpenAI), litellm-based embeddings
- [ ] **v0.7** - `holodeck serve` / `holodeck deploy` on the OpenAI Agents backend
- [ ] **v0.8** - Web UI (no-code editor)
- [ ] **v0.9** - Multi-agent orchestration
- [ ] **v1.0** - Production-ready release (SSO, audit logs, RBAC)

---

## 📚 Documentation

- **[Quickstart Guide](docs/getting-started/quickstart.md)** - Get your first agent running
- **[Installation](docs/getting-started/installation.md)** - Installation and setup
- **[Agent Configuration](docs/guides/agent-configuration.md)** - Configure your agents
- **[Tools Guide](docs/guides/tools.md)** - Vectorstore and MCP tools
- **[Evaluations Guide](docs/guides/evaluations.md)** - DeepEval and NLP metrics
- **[Dashboard Guide](docs/guides/dashboard.md)** - `holodeck test view` run-history dashboard
- **[Global Configuration](docs/guides/global-config.md)** - Shared settings
- **[Vector Stores](docs/guides/vector-stores.md)** - Set up vector databases
- **[MCP CLI](docs/guides/mcp-cli.md)** - Manage MCP servers
- **[LLM Providers](docs/guides/llm-providers.md)** - Configure LLM providers
- **[Contributing](docs/contributing.md)** - Development guide

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details

---

## 🙏 Acknowledgments

Built with:

- [Claude Agent SDK](https://docs.anthropic.com/en/docs/agents-and-tools/claude-code/claude-code-sdk-docs) - Native Anthropic agent framework
- [OpenAI Agents SDK](https://github.com/openai/openai-agents-python) - Native OpenAI / Azure OpenAI agent framework
- [litellm](https://github.com/BerriAI/litellm) - Unified embeddings + completion gateway
- [Semantic Kernel](https://github.com/microsoft/semantic-kernel) - Vector store connector layer
- [DeepEval](https://github.com/confident-ai/deepeval) - LLM evaluation framework
- [markitdown](https://github.com/microsoft/markitdown) - Document cracking into markdown for LLMs
- [FastAPI](https://fastapi.tiangolo.com/) - API deployment (planned)
- [ChromaDB](https://www.trychroma.com/) - Vector database
- [Qdrant](https://qdrant.tech/) - Vector database
- [PostgreSQL pgvector](https://github.com/pgvector/pgvector) - Vector database
- [Pinecone](https://www.pinecone.io/) - Vector database

Development tools:

- [spec-kit](https://github.com/spec-kit/spec-kit) - Spec-driven development
- [Claude Code](https://claude.ai/code) - AI-assisted development

Inspired by:

- Pytorch, Keras - Deep learning frameworks
- Promptflow - by its simplicity in defining semantic functions

---

## 💬 Community

- **GitHub Discussions**: [Ask questions](https://github.com/justinbarias/holodeck/discussions)

---

<p align="center">
  Made with ❤️ by the HoloDeck team
</p>

<p align="center">
  <a href="https://useholodeck.ai/">Website</a> •
  <a href="https://docs.useholodeck.ai/">Docs</a> •
  <a href="https://github.com/justinbarias/holodeck-samples">Examples</a> •
</p>
