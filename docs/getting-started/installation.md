# Installation Guide

Get HoloDeck up and running in minutes to start building and testing AI agents.

## Prerequisites

- **Python 3.13+** (check with `python --version`)
- **pip** (usually included with Python)

## Quick Start

### 1. Install HoloDeck CLI

```bash
pip install holodeck-ai
```

This installs the `holodeck` command-line tool and all required dependencies.

### 2. Verify Installation

```bash
holodeck --version
# Output: holodeck 0.2.0
```

Check that the CLI is accessible:

```bash
holodeck --help
```

### 3. Set Up API Credentials

HoloDeck supports OpenAI, Azure OpenAI, and Anthropic. Create a `.env` file in your project:

```bash
# .env (never commit this file!)
AZURE_OPENAI_API_KEY=your-key-here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
```

Or set environment variables:

```bash
export AZURE_OPENAI_API_KEY="your-key-here"
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
```

## Bootstrap Your First Agent

After installation, create your first agent project:

### 1. Create a Project Directory

```bash
mkdir my-first-agent
cd my-first-agent
```

### 2. Create Global Configuration

Create `config.yaml` for your project's LLM provider settings:

```bash
cat > config.yaml << 'EOF'
# HoloDeck Configuration
# Models and API credentials for your agents

providers:
  azure_openai:
    provider: azure_openai
    name: gpt-4o
    temperature: 0.3
    max_tokens: 2048
    endpoint: ${AZURE_OPENAI_ENDPOINT}
    api_key: ${AZURE_OPENAI_API_KEY}

execution:
  llm_timeout: 60
  file_timeout: 30
  cache_enabled: true
  verbose: false
EOF
```

### 3. Create Your First Agent

Create `agent.yaml` with a simple agent:

```bash
cat > agent.yaml << 'EOF'
name: my-first-agent
description: A helpful assistant to get you started

model:
  provider: azure_openai
  # Model and API key come from config.yaml

instructions:
  inline: |
    You are a helpful AI assistant.
    Be concise, accurate, and friendly.

test_cases:
  - name: "Simple greeting"
    input: "Hello! What can you do?"
    ground_truth: "I can help you with information, answer questions, and provide guidance."
    evaluations:
      - f1_score

evaluations:
  model:
    provider: azure_openai

  metrics:
    - metric: f1_score
      threshold: 0.7
EOF
```

### 4. Create `.env` File with Credentials

```bash
cat > .env << 'EOF'
# Add to .gitignore - never commit secrets!
AZURE_OPENAI_API_KEY=your-api-key-here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
EOF
```

**⚠️ Important**: Add `.env` to `.gitignore`:

```bash
echo ".env" >> .gitignore
echo ".env.local" >> .gitignore
```

## Running Your Agent

After bootstrapping your project, test your agent:

### 1. Run Agent Tests

```bash
# Test the agent and run evaluations
holodeck test agent.yaml
```

This command will:
- Load your agent configuration
- Execute test cases against the agent
- Run evaluation metrics
- Display results with pass/fail status

### 2. Interactive Chat Mode

```bash
# Chat interactively with your agent
holodeck chat agent.yaml
```

This starts an interactive session where you can:
- Send messages to your agent
- See responses in real-time
- Test agent behavior manually

### 3. Initialize from Templates

For more structured projects, use templates:

```bash
# Create a new project with a template
holodeck init

# This prompts you to:
# 1. Enter project name
# 2. Select a template (conversational, customer-support, research)
# 3. Configure project metadata
# 4. Choose LLM providers
```

## Verification Checklist

Verify your setup is working:

```bash
# ✓ Check HoloDeck CLI is installed
holodeck --version
# Expected: holodeck version 0.2.0

# ✓ Check help to see available commands
holodeck --help
# Should show: test, chat, init, deploy commands

# ✓ Test your agent (from project directory with config.yaml and agent.yaml)
holodeck test agent.yaml
# Should load agent and run test cases

# ✓ Try interactive chat
holodeck chat agent.yaml --input "Hello!"
# Should get a response from your agent
```

## Supported LLM Providers

HoloDeck supports multiple LLM providers. Choose one and set up credentials:

### Azure OpenAI (Recommended)

```bash
# .env or environment variables
AZURE_OPENAI_API_KEY=your-key-here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
```

### OpenAI

```bash
OPENAI_API_KEY=sk-...
OPENAI_ORG_ID=optional-org-id
```

### Anthropic

```bash
ANTHROPIC_API_KEY=sk-ant-...
```

## Troubleshooting

### "Python 3.13+ required"

Check your Python version and upgrade if needed:

```bash
python --version

# macOS (Homebrew)
brew install python@3.13

# Ubuntu/Debian
sudo apt-get install python3.13

# Windows: Download from python.org
```

### "holodeck: command not found"

The CLI isn't in your PATH. Try:

```bash
# Reinstall HoloDeck
pip install --upgrade holodeck-ai

# Try using Python module directly
python -m holodeck --version
```

### "Error: config.yaml not found"

Create a `config.yaml` file in your project directory. See [Bootstrap Your First Agent](#bootstrap-your-first-agent) section above.

### "Error: API key not found" or "Invalid credentials"

Verify your environment variables are set:

```bash
# Check if variables are set
echo $AZURE_OPENAI_API_KEY  # macOS/Linux
echo %AZURE_OPENAI_API_KEY%  # Windows

# Or check .env file exists
cat .env
```

If using `.env` file, ensure it's in the same directory as `agent.yaml`.

### "Error: Failed to load agent.yaml"

Common issues:
- YAML syntax error (check indentation)
- File path incorrect
- Required fields missing (`name`, `model.provider`, `instructions`)

Verify your YAML:
```bash
# Check YAML syntax
python -c "import yaml; yaml.safe_load(open('agent.yaml'))"
```

## Next Steps

- ✅ Installation complete!
- 📖 [Read the Quickstart Guide →](quickstart.md)
- 📚 [View Agent Configuration Guide →](../guides/agent-configuration.md)
- 📁 [Explore Example Agents →](../examples/README.md)
- 🛠️ [Learn Global Configuration →](../guides/global-config.md)
