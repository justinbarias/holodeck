# 🧪 HoloDeck

**Build, Test, and Deploy AI Agents — No Code Required**

HoloDeck is an open-source experimentation platform that enables teams to create, evaluate, and deploy AI agents through simple YAML configuration. Go from hypothesis to production API in minutes, not weeks.

[![PyPI version](https://badge.fury.io/py/holodeck-ai.svg)](https://badge.fury.io/py/holodeck-ai)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

---

## ✨ Features

- **🎯 No-Code Agent Definition** - Define agents using simple YAML configuration
- **🧪 Hypothesis-Driven Testing** - Test agent behaviors against structured test cases
- **📊 Integrated Evaluations** - DeepEval LLM-as-judge metrics (GEval, RAG) plus NLP metrics (F1, BLEU, ROUGE)
- **🔌 Tool Ecosystem** - Extend agents with MCP servers and vector store search
- **💾 RAG Support** - Native vector database integration (ChromaDB, Qdrant, PostgreSQL, Pinecone)
- **🤖 Open-Source First** - Designed to work with Ollama for local, free inference

---

## 🚀 Quick Start

### Installation

```bash
pip install holodeck-ai
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

Agents are defined using declarative YAML configuration:

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
│   Agent      │  │  Evaluation  │  │  Deployment  │
│   Engine     │  │  Framework   │  │  (Planned)   │
└──────────────┘  └──────────────┘  └──────────────┘
        │                  │                  │
        ├─ LLM Providers   ├─ DeepEval       ├─ FastAPI
        ├─ MCP Tools       ├─ NLP Metrics    ├─ Docker
        ├─ Vectorstore     ├─ Custom GEval   ├─ Cloud Deploy
        └─ Memory          └─ Reporting      └─ Monitoring
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

## 📊 Monitoring & Observability (Planned)

HoloDeck provides comprehensive observability with native **OpenTelemetry** support and **Semantic Conventions for Generative AI**.

### OpenTelemetry Integration

HoloDeck automatically instruments your agents with OpenTelemetry traces, metrics, and logs following the [OpenTelemetry Semantic Conventions for Generative AI](https://opentelemetry.io/docs/specs/semconv/gen-ai/).

**Basic Configuration:**

```yaml
# agent.yaml
observability:
  enabled: true
  service_name: "customer-support-agent"

  opentelemetry:
    enabled: true
    endpoint: "http://localhost:4318" # OTLP endpoint
    protocol: grpc # or http/protobuf

    traces:
      enabled: true
      sample_rate: 1.0 # Sample 100% of traces

    metrics:
      enabled: true
      interval: 60 # Export metrics every 60s

    logs:
      enabled: true
      level: info
```

**Export to Observability Platforms:**

```yaml
observability:
  opentelemetry:
    enabled: true
    exporters:
      # Jaeger
      - type: otlp
        endpoint: "http://jaeger:4318"

      # Prometheus (metrics)
      - type: prometheus
        port: 8889

      # Datadog
      - type: otlp
        endpoint: "https://api.datadoghq.com"
        headers:
          DD-API-KEY: "${DATADOG_API_KEY}"
```

### Built-in Metrics

**Request Metrics:**

- `gen_ai.client.operation.duration` - Operation duration histogram
- `gen_ai.client.token.usage` - Token usage counter
- `gen_ai.client.request.count` - Request counter
- `gen_ai.client.error.count` - Error counter

**Agent-Specific Metrics:**

- `holodeck.agent.requests.total` - Total agent requests
- `holodeck.agent.requests.duration` - Request duration histogram
- `holodeck.agent.tokens.total` - Total tokens used
- `holodeck.agent.cost.total` - Total cost (USD)
- `holodeck.tools.invocations.total` - Tool invocation count
- `holodeck.evaluations.score` - Evaluation scores gauge

### Cost Tracking

HoloDeck automatically tracks costs based on token usage and model pricing:

```yaml
observability:
  cost_tracking:
    enabled: true

    # Custom pricing (overrides defaults)
    pricing:
      openai:
        gpt-4o:
          input: 0.0025 # per 1K tokens
          output: 0.0100
        gpt-4o-mini:
          input: 0.00015
          output: 0.00060

    # Cost alerts
    alerts:
      - threshold: 100.00 # USD
        period: daily
        notify: "${ALERT_EMAIL}"
```

---

## 🗺️ Roadmap

- [x] **v0.1** - Core agent engine + CLI
- [x] **v0.2** - Evaluation framework (DeepEval, NLP), Tools (MCP, Vectorstore)
- [x] **v0.3** - API deployment, OpenTelemetry observability
- [ ] **v0.4** - Web UI (no-code editor)
- [ ] **v0.5** - Multi-agent orchestration
- [ ] **v0.6** - Enterprise features (SSO, audit logs, RBAC)
- [ ] **v1.0** - Production-ready release

---

## 📚 Documentation

- **[Quickstart Guide](docs/getting-started/quickstart.md)** - Get your first agent running
- **[Installation](docs/getting-started/installation.md)** - Installation and setup
- **[Agent Configuration](docs/guides/agent-configuration.md)** - Configure your agents
- **[Tools Guide](docs/guides/tools.md)** - Vectorstore and MCP tools
- **[Evaluations Guide](docs/guides/evaluations.md)** - DeepEval and NLP metrics
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

- [Semantic Kernel](https://github.com/microsoft/semantic-kernel) - Agent framework
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
