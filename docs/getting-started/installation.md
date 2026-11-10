# Installation Guide

Get HoloDeck installed and ready to build AI agents.

## Prerequisites

- **Python 3.11+** (check with `python --version`)
- **uv** - The fast Python package installer
- **Node.js 18+** (required for Claude Agent SDK backend) — check with `node --version`

> **Note**: Node.js is required for the Claude backend, which serves `provider: anthropic` and local `provider: ollama` models. The OpenAI Agents backend (`provider: openai`, `azure_openai`) does not require it.

### Installing uv

If you don't have uv installed:

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# macOS (Homebrew)
brew install uv

# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Verify uv is installed:

```bash
uv --version
```

## Install HoloDeck CLI

Install HoloDeck as a global tool using uv:

```bash
uv tool install holodeck-ai@latest --prerelease allow --python 3.11
```

This installs the `holodeck` command-line tool globally, available from any directory.

### Backends: what's included vs. an extra

- **Claude backend** — `provider: anthropic` and local `provider: ollama`. **Included in the base install** (no extra required); needs Node.js 18+ (see [Prerequisites](#prerequisites)).
- **OpenAI Agents backend** — `provider: openai` and `azure_openai`. Shipped as an **optional extra** so non-OpenAI installs stay lean (the SDK is imported only when this backend runs). Install it with the `openai-agents` extra:

```bash
uv tool install "holodeck-ai[openai-agents]@latest" --prerelease allow --python 3.11
```

Extras compose — e.g. the OpenAI backend plus a Qdrant vector store for RAG:

```bash
uv tool install "holodeck-ai[openai-agents,qdrant]@latest" --prerelease allow --python 3.11
```

> **Without the extra**, an `openai` / `azure_openai` agent fails at startup with a missing-dependency error pointing you here.

### Install profiles

The base install has what every user needs to build, chat with, serve and test
an agent on the Claude backend. Features with heavy or rarely used dependencies
are extras. Add only the extras you use; they compose
(`holodeck-ai[openai-agents,eval,qdrant]`).

| Install | Enables | Key packages |
| --- | --- | --- |
| `holodeck-ai` (base) | CLI, config, Claude backend, MCP, `serve`, `chat`, `test run` with code, equality and numeric metrics, document ingestion, in-memory BM25 keyword index, OpenTelemetry | `claude-agent-sdk`, `pydantic`, `fastapi`, `markitdown`, `litellm`, `tiktoken` |
| `[eval]` | DeepEval metrics (`type: geval`, `type: rag`) | `deepeval`, `anthropic`, `ollama` |
| `[nlp]` | NLP overlap metrics (`bleu`, `rouge`, `meteor`) | `sacrebleu`, `rouge-score`, `nltk` |
| `[optimizer]` | `holodeck test optimize` | `optuna` |
| `[opensearch]` | OpenSearch keyword index (`keyword_index.provider: opensearch`) | `opensearch-py` |
| `[openai-agents]` | OpenAI Agents backend (`openai`, `azure_openai`) | `openai-agents`, `openai` |
| `[dashboard]` | `holodeck test view` | `dash`, `plotly` |
| `[temporal]` | `holodeck worker` | `temporalio` |
| `[qdrant]`, `[postgres]`, `[chromadb]`, … `[vectorstores]` | Vector store providers | one client each; `vectorstores` has all |
| `[otel-prometheus]`, `[otel-azmon]`, `[otel-all]`, `[claude-otel]` | Extra telemetry exporters and instrumentation | |
| `[deploy]`, `[deploy-azure]`, … `[deploy-all]`, `[s3]`, `[azure-blob]`, `[all-sources]` | `holodeck deploy` and cloud sources | `docker`, cloud SDKs |
| `[all]` | Every extra above | |

When a command needs an extra that is not installed, it stops before any agent
or model call and prints the install command, for example:

```text
Error: DeepEval metrics needs the 'eval' extra, which is not installed. Install it with: pip install 'holodeck-ai[eval]'
```

Commands that do not use the feature still run.

#### Upgrading from an earlier release

Before this release, the base install also contained DeepEval, the NLP metric
libraries, Optuna, opensearch-py, MkDocs Material and Twine.

| You ran | Now run |
| --- | --- |
| `pip install holodeck-ai` and use DeepEval or NLP metrics | `pip install "holodeck-ai[eval,nlp]"` |
| `pip install holodeck-ai` and use `holodeck test optimize` | `pip install "holodeck-ai[optimizer]"` (add `eval,nlp` for its metrics) |
| `pip install holodeck-ai` and use an OpenSearch keyword index | `pip install "holodeck-ai[opensearch]"` |
| `pip install holodeck-ai` and want everything as before | `pip install "holodeck-ai[all]"` |

With `uv tool`, use the same extras:
`uv tool install "holodeck-ai[eval,nlp]@latest" --prerelease allow --python 3.11`.

### Install with Vector Store Providers (Optional)

If you plan to use semantic search with vector databases, install with extras:

```bash
# Individual providers
uv tool install "holodeck-ai[postgres]@latest" --prerelease allow --python 3.11
uv tool install "holodeck-ai[qdrant]@latest" --prerelease allow --python 3.11
uv tool install "holodeck-ai[pinecone]@latest" --prerelease allow --python 3.11
uv tool install "holodeck-ai[chromadb]@latest" --prerelease allow --python 3.11

# Or install all vector store providers at once
uv tool install "holodeck-ai[vectorstores]@latest" --prerelease allow --python 3.11
```

## Verify Installation

Check that HoloDeck is installed correctly:

```bash
holodeck --version
# Output: holodeck 0.2.0
```

View available commands:

```bash
holodeck --help
```

## Set Up LLM Provider

HoloDeck supports multiple LLM providers. Ollama is recommended for local development as it requires no API keys and runs entirely on your machine.

### Ollama (Recommended)

Ollama runs LLMs locally on your machine - no API keys required.

**Install Ollama:**

```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows: Download from https://ollama.com/download
```

**Start Ollama and pull a model:**

```bash
# Start the Ollama service
ollama serve

# Pull a model (in another terminal)
ollama pull llama3.2
# Or for a smaller model:
ollama pull phi3
```

**Verify Ollama is running:**

```bash
curl http://localhost:11434/api/tags
```

No environment variables needed - HoloDeck connects to Ollama at `http://localhost:11434` by default.

### Cloud Providers (Optional)

For cloud-based LLMs, set up credentials using environment variables or a `.env` file.

#### Environment Variables

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."

# Azure OpenAI
export AZURE_OPENAI_API_KEY="your-key-here"
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"

# Anthropic (API Key)
export ANTHROPIC_API_KEY="sk-ant-..."

# Anthropic (OAuth Token — recommended for Claude Code users)
export CLAUDE_CODE_OAUTH_TOKEN="your-oauth-token"
```

#### `.env` File

Create a `.env` file in your project directory:

```bash
# .env (never commit this file!)
OPENAI_API_KEY=sk-...
```

Add `.env` to `.gitignore`:

```bash
echo ".env" >> .gitignore
echo ".env.local" >> .gitignore
```

## Supported LLM Providers

HoloDeck supports multiple LLM providers:

### Ollama (Recommended)

Run LLMs locally with no API keys. Supports Llama, Mistral, Phi, and many more models.

```bash
# No environment variables needed
# Default endpoint: http://localhost:11434
```

### OpenAI

Runs on the OpenAI Agents backend — install the [`openai-agents` extra](#backends-whats-included-vs-an-extra) first.

```bash
OPENAI_API_KEY=sk-...
OPENAI_ORG_ID=optional-org-id
```

### Azure OpenAI

Also runs on the OpenAI Agents backend (requires the [`openai-agents` extra](#backends-whats-included-vs-an-extra)). `name` is your **deployment** name.

```bash
AZURE_OPENAI_API_KEY=your-key-here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
```

### Anthropic

Claude models via the native Claude Agent SDK backend. Requires Node.js 18+.

```bash
# Option 1: API Key
ANTHROPIC_API_KEY=sk-ant-...

# Option 2: OAuth Token (recommended for Claude Code users)
CLAUDE_CODE_OAUTH_TOKEN=your-oauth-token
```

## Upgrading HoloDeck

To upgrade to the latest version:

```bash
uv tool upgrade holodeck-ai
```

To reinstall with a specific version:

```bash
uv tool install holodeck-ai@0.3.0 --prerelease allow --python 3.11 --force
```

## Uninstalling

To remove HoloDeck:

```bash
uv tool uninstall holodeck-ai
```

## Troubleshooting

### "Python 3.11+ required"

Check your Python version and upgrade if needed:

```bash
python --version

# macOS (Homebrew)
brew install python@3.11

# Ubuntu/Debian
sudo apt-get install python3.11

# Windows: Download from python.org
```

### "holodeck: command not found"

The CLI isn't in your PATH. Try:

```bash
# Reinstall HoloDeck
uv tool install holodeck-ai@latest --prerelease allow --python 3.11 --force

# Ensure uv tools are in PATH
# Add to your shell profile (~/.bashrc, ~/.zshrc, etc.):
export PATH="$HOME/.local/bin:$PATH"

# Then reload your shell
source ~/.zshrc  # or ~/.bashrc
```

### "uv: command not found"

Install uv first. See [Installing uv](#installing-uv) above.

### "Error: API key not found" or "Invalid credentials"

Verify your environment variables are set:

```bash
# Check if variables are set
echo $AZURE_OPENAI_API_KEY  # macOS/Linux
echo %AZURE_OPENAI_API_KEY%  # Windows

# Or check .env file exists
cat .env
```

If using a `.env` file, ensure it's in your project directory.

## Next Steps

- [Quickstart Guide](quickstart.md) - Build your first agent in 5 minutes
- [Agent Configuration Guide](../guides/agent-configuration.md) - Full configuration reference
- [Example Agents](../examples/README.md) - Browse example agents
- [Global Configuration](../guides/global-config.md) - Configure defaults
