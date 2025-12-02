# Quickstart Guide

Get your first AI agent running in 5 minutes using the HoloDeck CLI.

## Prerequisites

Complete the [Installation Guide](installation.md) before continuing.

---

## Quick Start with CLI

### Step 1: Initialize a New Agent Project

Use the `holodeck init` command to create a new project with templates:

```bash
# Create a basic conversational agent
holodeck init my-chatbot

# Or choose a different template
holodeck init research-agent --template research
holodeck init support-bot --template customer-support

# With metadata
holodeck init my-agent --description "My AI agent" --author "Your Name"
```

This creates a complete project structure:

```
my-chatbot/
├── agent.yaml              # Main configuration
├── instructions/
│   └── system-prompt.md   # Agent behavior
├── tools/                 # Custom functions
├── data/                  # Grounding data
└── tests/                 # Test cases
```

### Step 2: Edit Your Agent Configuration

```bash
cd my-chatbot
```

Open `agent.yaml` and customize:

- Agent name and description
- Model provider (OpenAI, Azure, Anthropic)
- Instructions/system prompt
- Tools and data sources
- Test cases

### Step 3: Run Your Agent

#### Interactive Chat

Start an interactive chat session with your agent:

```bash
# Basic chat (from parent directory)
holodeck chat my-chatbot/agent.yaml

# Or cd into the project directory first
cd my-chatbot
holodeck chat agent.yaml

# Verbose mode with detailed status panel
holodeck chat my-chatbot/agent.yaml --verbose

# Quiet mode (no logging, but spinner still shows)
holodeck chat my-chatbot/agent.yaml --quiet
```

**Chat Features:**

- **Animated Spinner**: Shows braille animation during agent execution (even in quiet mode)
- **Token Tracking**: Displays cumulative token usage across the conversation
- **Adaptive Status Display**:
  - **Default mode**: Inline status `[messages | execution time]`
  - **Verbose mode**: Rich status panel with token breakdown
  - **Quiet mode**: No status display (spinner only)

Example verbose output:

```
╭─── Chat Status ─────────────────────────╮
│ Session Time: 00:05:23                  │
│ Messages: 3 / 50 (6%)                   │
│ Total Tokens: 1,234                     │
│   ├─ Prompt: 890                        │
│   └─ Completion: 344                    │
│ Last Response: 1.2s                     │
╰─────────────────────────────────────────╯
Agent: Your response here
```

#### Run Tests

```bash
# Run all tests (from parent directory)
holodeck test my-chatbot/agent.yaml

# Deploy locally
holodeck deploy my-chatbot/agent.yaml --port 8000
```

## Manual Setup (Alternative)

If you prefer to create files manually instead of using `holodeck init`:

### Step 1: Create Your Agent Configuration

Create `agent.yaml`:

```yaml
name: "my-assistant"
description: "A helpful AI assistant"

model:
  provider: azure_openai
  # Provider settings come from config.yaml

instructions:
  inline: |
    You are a helpful AI assistant.
    Answer questions accurately and concisely.

test_cases:
  - name: "greeting"
    input: "Hello! What can you do?"
    ground_truth: "I can help you with information and answer questions."
    evaluations:
      - f1_score

evaluations:
  model:
    provider: azure_openai

  metrics:
    - metric: f1_score
      threshold: 0.7
```

### Step 2: Create Project Configuration

Initialize `config.yaml` using the CLI:

```bash
holodeck config init -p
```

Then edit the generated file to configure your LLM provider:

```yaml
# config.yaml
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
```

### Step 3: Test Your Agent

```bash
# Run agent tests
holodeck test agent.yaml

# Chat interactively
holodeck chat agent.yaml
```

## Common Commands

```bash
# Show all available commands
holodeck --help

# Create new project from template
holodeck init my-project

# Test your agent and run evaluations
holodeck test my-project/agent.yaml

# Interactive chat with your agent
holodeck chat my-project/agent.yaml

# Chat with verbose output (detailed status panel)
holodeck chat my-project/agent.yaml --verbose

# Chat with quiet mode (no logging, spinner only)
holodeck chat my-project/agent.yaml --quiet

# Chat with custom max messages limit
holodeck chat my-project/agent.yaml --max-messages 100

# Deploy as API
holodeck deploy my-project/agent.yaml --port 8000
```

## Tips & Tricks

### Interactive Chat Features

The `holodeck chat` command provides real-time feedback during conversation:

**Default Mode (Recommended for Most Users)**

```bash
holodeck chat agent.yaml
```

Shows inline status after each response:

```
Agent: Your response here [3/50 | 1.2s]
```

**Verbose Mode (For Detailed Monitoring)**

```bash
holodeck chat agent.yaml --verbose
```

Displays a rich status panel with token breakdown:

```
╭─── Chat Status ─────────────────────────╮
│ Session Time: 00:05:23                  │
│ Messages: 3 / 50 (6%)                   │
│ Total Tokens: 1,234                     │
│   ├─ Prompt: 890                        │
│   └─ Completion: 344                    │
│ Last Response: 1.2s                     │
╰─────────────────────────────────────────╯
```

**Quiet Mode (For Clean Output)**

```bash
holodeck chat agent.yaml --quiet
```

Shows only responses and spinner during execution, no status display.

**Monitor Token Usage**
All chat modes track cumulative token usage across the conversation. This helps you:

- Monitor API costs in real-time
- Understand token consumption patterns
- Plan for context window limits
- Optimize prompt sizes

### Use Environment Variables for Secrets

Never hardcode API keys in config files. Always use environment variables:

```yaml
# config.yaml
providers:
  azure_openai:
    provider: azure_openai
    api_key: ${AZURE_OPENAI_API_KEY} # From environment
    endpoint: ${AZURE_OPENAI_ENDPOINT} # From environment
```

### Test Locally Before Deploying

```bash
# Run tests first
holodeck test my-project/agent.yaml

# Then try interactive chat
holodeck chat my-project/agent.yaml

# Finally deploy if tests pass
holodeck deploy my-project/agent.yaml --port 8000
```

### Organize Multiple Agents

If you have multiple agents, create separate directories:

```
my-project/
├── config.yaml          # Shared configuration
├── agent1/
│   └── agent.yaml
├── agent2/
│   └── agent.yaml
└── .env                 # Shared credentials
```

Test each agent:

```bash
holodeck test agent1/agent.yaml
holodeck test agent2/agent.yaml
```

## Troubleshooting

For installation issues, see the [Installation Guide](installation.md#troubleshooting).

### "Error: config.yaml not found"

Create a `config.yaml` file in your project directory. See [Manual Setup](#manual-setup-alternative) above.

### "Error: API key not found" or "Invalid credentials"

Verify your environment variables:

```bash
# Check if set
echo $AZURE_OPENAI_API_KEY

# Or verify .env file
cat .env
```

Make sure `.env` is in the same directory as `agent.yaml`.

### "Error: Failed to load agent.yaml"

Check your YAML syntax:

```bash
# Validate YAML
python -c "import yaml; yaml.safe_load(open('agent.yaml'))"
```

Common issues:

- Incorrect indentation (must use spaces, not tabs)
- Missing required fields: `name`, `model.provider`, `instructions`
- Invalid YAML syntax

## Next Steps

- [Agent Configuration Reference](../guides/agent-configuration.md)
- [Explore Tool Types](../guides/tools.md)
- [Learn About Evaluations](../guides/evaluations.md)
- [Browse Examples](../examples/README.md)
- [Global Configuration Guide](../guides/global-config.md)

## Getting Help

- **Report bugs**: [GitHub Issues](https://github.com/justinbarias/holodeck/issues)
- **Ask questions**: [GitHub Discussions](https://github.com/justinbarias/holodeck/discussions)
- **Full docs**: [https://docs.useholodeck.ai](https://docs.useholodeck.ai)
