# Quickstart Guide

Get up and running with HoloDeck in 5 minutes. This guide covers loading a configuration, handling errors, and understanding the basics.

## Before You Start

Ensure you've completed the [Installation Guide](installation.md):

```bash
pip install holodeck-ai
python -m holodeck --version  # Should output: holodeck 0.1.0
```

Set up your API key (example for OpenAI):

```bash
export OPENAI_API_KEY="sk-your-key-here"
```

## Step 1: Create Your First Agent (agent.yaml)

Create a file called `my-agent.yaml`:

```yaml
name: "Quick Start Agent"
description: "A simple agent to get started with HoloDeck"
model:
  provider: "openai"
  name: "gpt-4o-mini"
  temperature: 0.7
  max_tokens: 500
instructions:
  inline: |
    You are a helpful AI assistant.
    Answer questions accurately and concisely.
```

This minimal agent has:

- **name**: Human-readable agent name
- **model**: LLM provider and configuration
- **instructions**: How the agent should behave (inline text or file reference)

## Step 2: Load and Use the Agent

Create a Python script `load_agent.py`:

```python
from holodeck.config.loader import ConfigLoader

# Create a loader
loader = ConfigLoader()

# Load the agent configuration
agent = loader.load_agent_yaml("my-agent.yaml")

# Access agent properties
print(f"Agent Name: {agent.name}")
print(f"Description: {agent.description}")
print(f"Model: {agent.model.name}")
print(f"Provider: {agent.model.provider}")
```

Run it:

```bash
python load_agent.py
```

Expected output:

```
Agent Name: Quick Start Agent
Description: A simple agent to get started with HoloDeck
Model: gpt-4o-mini
Provider: openai
```

## Step 3: Handle Errors Gracefully

Real-world scenarios require error handling. Update `load_agent.py`:

```python
from holodeck.config.loader import ConfigLoader
from holodeck.lib.errors import ConfigError

loader = ConfigLoader()

try:
    agent = loader.load_agent_yaml("my-agent.yaml")
    print(f"✓ Successfully loaded agent: {agent.name}")

except FileNotFoundError as e:
    print(f"❌ File not found: {e}")
    print("Tip: Make sure my-agent.yaml exists in the current directory")

except ConfigError as e:
    print(f"❌ Configuration error: {e}")
    print("Tip: Check your YAML syntax and required fields")
```

Try it by running:

```bash
# Load successfully
python load_agent.py

# Simulate missing file
python load_agent.py  # (rename/delete my-agent.yaml first)
```

## Step 4: Common Error Scenarios

### Missing Required Fields

Create `invalid-agent.yaml`:

```yaml
name: "Incomplete Agent"
# Missing: model and instructions!
```

Load it and see what happens:

```python
from holodeck.config.loader import ConfigLoader
from holodeck.lib.errors import ConfigError

try:
    agent = ConfigLoader().load_agent_yaml("invalid-agent.yaml")
except ConfigError as e:
    print(f"Configuration Error:\n{e}")
```

Output:

```
Configuration Error:
Field 'model' is required but missing
Field 'instructions' is required but missing
```

### Invalid YAML Syntax

Create `bad-syntax.yaml`:

```yaml
name: Invalid YAML
model:
  provider: openai
  - this line is invalid
instructions: |
  Broken indentation
 bad spacing
```

HoloDeck will catch parsing errors:

```python
try:
    agent = ConfigLoader().load_agent_yaml("bad-syntax.yaml")
except ConfigError as e:
    print(f"Error: {e}")
```

### Invalid Model Configuration

```yaml
name: "Bad Model Config"
model:
  provider: "invalid_provider" # ❌ Not a valid provider
  name: "gpt-4o"
instructions:
  inline: "Help"
```

Error output:

```
Configuration Error:
Field 'provider' must be one of: openai, azure_openai, anthropic
```

## Step 5: Add Tools and Test Cases

Expand your agent with tools and test cases:

```yaml
name: "Research Assistant"
description: "An agent that searches and analyzes information"

model:
  provider: "openai"
  name: "gpt-4o"
  temperature: 0.5
  max_tokens: 2000

instructions:
  file: "instructions.md" # Load from file

tools:
  - type: "vectorstore"
    source: "knowledge-base.json"
    vector_field: "embeddings"
    chunk_size: 500

  - type: "mcp"
    server: "web-search"
    description: "Search the web for current information"

test_cases:
  - input: "What are the latest developments in AI?"
    expected_tools: ["web-search"]
    ground_truth: "AI is rapidly evolving..."

evaluations:
  metrics:
    - name: "groundedness"
      threshold: 0.8
    - name: "relevance"
      threshold: 0.75
```

## Step 6: Using Instructions from Files

For longer instructions, use a separate file:

**instructions.md**:

```markdown
You are a research assistant focused on providing accurate, cited information.

## Guidelines

1. Always cite your sources
2. Use web search for current information
3. Provide comprehensive summaries
4. Flag uncertain information

## Constraints

- Keep responses under 2000 tokens
- Prefer primary sources over secondary
```

**agent.yaml**:

```yaml
name: Research Assistant
instructions:
  file: "instructions.md" # Relative to agent.yaml location
```

Load it:

```python
loader = ConfigLoader()
agent = loader.load_agent_yaml("agent.yaml")
print(agent.instructions)  # Will contain full instructions from file
```

## Step 7: Environment Variables

Use environment variables for sensitive data:

**agent.yaml**:

```yaml
name: "Configured Agent"
model:
  provider: "openai"
  name: "gpt-4o"
instructions:
  inline: "Help users with their questions"
```

Your `.env` file:

```bash
OPENAI_API_KEY=sk-...
OPENAI_ORG_ID=org-...
```

Load and use:

```python
from dotenv import load_dotenv
load_dotenv()  # Loads .env file

from holodeck.config.loader import ConfigLoader
agent = ConfigLoader().load_agent_yaml("agent.yaml")
# API keys are now available from environment
```

## Complete Example

Here's a complete working example with best practices:

**agent.yaml**:

```yaml
name: "Smart Assistant"
description: "An intelligent assistant with search capabilities"

model:
  provider: "openai"
  name: "gpt-4o-mini"
  temperature: 0.7
  max_tokens: 1500

instructions:
  inline: |
    You are a helpful and knowledgeable AI assistant.
    Provide accurate, concise answers to user questions.

tools:
  - type: "mcp"
    server: "filesystem"
    description: "Access local files and documents"

test_cases:
  - input: "What can you do?"
    ground_truth: "Describe my capabilities including file access and question answering"
```

**main.py**:

```python
#!/usr/bin/env python3
"""Example: Load and validate an HoloDeck agent."""

from holodeck.config.loader import ConfigLoader
from holodeck.lib.errors import ConfigError

def main():
    try:
        # Load agent configuration
        loader = ConfigLoader()
        agent = loader.load_agent_yaml("agent.yaml")

        # Display agent information
        print(f"✓ Agent loaded successfully")
        print(f"  Name: {agent.name}")
        print(f"  Description: {agent.description}")
        print(f"  Model: {agent.model.name} ({agent.model.provider})")
        print(f"  Tools: {len(agent.tools or [])} configured")
        print(f"  Test Cases: {len(agent.test_cases or [])} defined")

        return agent

    except FileNotFoundError as e:
        print(f"❌ Configuration file not found: {e}")
        print(f"   Please create agent.yaml in the current directory")
        return None

    except ConfigError as e:
        print(f"❌ Configuration error: {e}")
        print(f"   Please review your agent.yaml file")
        return None

if __name__ == "__main__":
    agent = main()
    if agent:
        print("\n✓ Ready to use agent!")
```

Run it:

```bash
python main.py
```

## Common Patterns

### Pattern 1: Load and Validate Only

```python
from holodeck.config.loader import ConfigLoader
from holodeck.lib.errors import ConfigError

try:
    agent = ConfigLoader().load_agent_yaml("agent.yaml")
    print(f"✓ Valid agent: {agent.name}")
except ConfigError as e:
    print(f"✗ Invalid configuration: {e}")
```

### Pattern 2: Graceful Degradation

```python
from holodeck.config.loader import ConfigLoader
from holodeck.lib.errors import ConfigError

loader = ConfigLoader()
agent = None

try:
    agent = loader.load_agent_yaml("agent.yaml")
except ConfigError as e:
    print(f"Warning: Could not load agent.yaml: {e}")
    # Fall back to default or create new agent
```

### Pattern 3: Batch Processing

```python
from pathlib import Path
from holodeck.config.loader import ConfigLoader
from holodeck.lib.errors import ConfigError

agents = []
errors = []

loader = ConfigLoader()
for yaml_file in Path(".").glob("agents/*.yaml"):
    try:
        agent = loader.load_agent_yaml(str(yaml_file))
        agents.append(agent)
    except ConfigError as e:
        errors.append((yaml_file, e))

print(f"Loaded {len(agents)} agents, {len(errors)} errors")
```

## Next Steps

- 📖 [Read Agent Configuration Reference →](../guides/agent-configuration.md)
- 🔧 [Explore Tool Types →](../guides/tools.md)
- 📊 [Learn About Evaluations →](../guides/evaluations.md)
- 💡 [Browse Examples →](../examples/README.md)
- ✍️ [API Reference →](../api/models.md)

## Troubleshooting

### "ConfigError: Field 'X' is required"

Your YAML is missing a required field. Check:

- `name` - Agent name
- `model` - LLM provider configuration
- `instructions` - Agent behavior instructions

### "FileNotFoundError: agent.yaml not found"

The loader couldn't find `agent.yaml`. Ensure:

- File exists: `ls -la agent.yaml`
- Correct path: Use absolute path if needed `loader.load_agent_yaml("/full/path/agent.yaml")`
- Working directory: `pwd` shows correct location

### "ConfigError: Field 'provider' must be one of..."

Your model provider is invalid. Use one of:

- `openai` (default)
- `azure_openai` (requires AZURE_OPENAI_ENDPOINT)
- `anthropic` (default: claude-3-sonnet)

### "Module not found: holodeck"

HoloDeck isn't installed. Run:

```bash
pip install holodeck-ai
```

## Getting Help

- 🐛 **Report bugs**: [GitHub Issues](https://github.com/anthropics/holodeck/issues)
- 💬 **Ask questions**: [GitHub Discussions](https://github.com/anthropics/holodeck/discussions)
- 📚 **Full docs**: [https://docs.holodeck.ai](https://docs.holodeck.ai)
