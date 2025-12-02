# Installation Guide

Get HoloDeck installed and ready to build AI agents.

## Prerequisites

- **Python 3.10+** (check with `python --version`)
- **uv** - The fast Python package installer

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
uv tool install holodeck-ai@latest --prerelease allow --python 3.10
```

This installs the `holodeck` command-line tool globally, available from any directory.

### Install with Vector Store Providers (Optional)

If you plan to use semantic search with vector databases, install with extras:

```bash
# Individual providers
uv tool install "holodeck-ai[postgres]@latest" --prerelease allow --python 3.10
uv tool install "holodeck-ai[qdrant]@latest" --prerelease allow --python 3.10
uv tool install "holodeck-ai[pinecone]@latest" --prerelease allow --python 3.10
uv tool install "holodeck-ai[chromadb]@latest" --prerelease allow --python 3.10

# Or install all vector store providers at once
uv tool install "holodeck-ai[vectorstores]@latest" --prerelease allow --python 3.10
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

# Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."
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

```bash
OPENAI_API_KEY=sk-...
OPENAI_ORG_ID=optional-org-id
```

### Azure OpenAI

```bash
AZURE_OPENAI_API_KEY=your-key-here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
```

### Anthropic

```bash
ANTHROPIC_API_KEY=sk-ant-...
```

## Upgrading HoloDeck

To upgrade to the latest version:

```bash
uv tool upgrade holodeck-ai
```

To reinstall with a specific version:

```bash
uv tool install holodeck-ai@0.3.0 --prerelease allow --python 3.10 --force
```

## Uninstalling

To remove HoloDeck:

```bash
uv tool uninstall holodeck-ai
```

## Troubleshooting

### "Python 3.10+ required"

Check your Python version and upgrade if needed:

```bash
python --version

# macOS (Homebrew)
brew install python@3.10

# Ubuntu/Debian
sudo apt-get install python3.10

# Windows: Download from python.org
```

### "holodeck: command not found"

The CLI isn't in your PATH. Try:

```bash
# Reinstall HoloDeck
uv tool install holodeck-ai@latest --prerelease allow --python 3.10 --force

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
