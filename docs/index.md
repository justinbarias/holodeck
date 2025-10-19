# HoloDeck - AI Agent Experimentation Platform

![HoloDeck Logo](assets/holodeck.png)

**HoloDeck** is an open-source experimentation platform for building, testing, and deploying AI agents through **YAML configuration**. Define intelligent agents entirely through configuration—no code required.

## Key Features

- **No-Code Agent Definition**: Define agents, tools, and evaluations in simple YAML files
- **Multi-Provider Support**: OpenAI, Azure OpenAI, Anthropic (add more via MCP)
- **Flexible Tool Integration**: Vector stores, custom functions, MCP servers, and AI-powered tools
- **Built-in Testing & Evaluation**: Run evaluations with multiple metrics, customize models per metric
- **Production-Ready**: Deploy agents as FastAPI endpoints with Docker support
- **Multimodal Test Support**: Images, PDFs, Word docs, Excel sheets, and mixed media in test cases

## Quick Start

### 1. Install HoloDeck

```bash
pip install holodeck-ai
```

### 2. Create a Simple Agent

Create `my-agent.yaml`:

```yaml
name: "My First Agent"
description: "A helpful AI assistant"
model:
  provider: "openai"
  name: "gpt-4o-mini"
  temperature: 0.7
  max_tokens: 1000
instructions:
  inline: |
    You are a helpful AI assistant.
    Answer questions accurately and concisely.
```

### 3. Load and Use the Agent

```python
from holodeck.config.loader import ConfigLoader

# Load agent configuration
loader = ConfigLoader()
agent = loader.load_agent_yaml("my-agent.yaml")

print(f"Loaded agent: {agent.name}")
print(f"Model: {agent.model.name}")
```

## Documentation

- **[Getting Started](getting-started/installation.md)** - Installation and setup
- **[Quickstart Guide](getting-started/quickstart.md)** - Minimal working example with error handling
- **[Agent Configuration](guides/agent-configuration.md)** - Complete schema reference
- **[Tools Guide](guides/tools.md)** - All tool types explained with examples
- **[Evaluations](guides/evaluations.md)** - Testing and evaluation framework
- **[Global Configuration](guides/global-config.md)** - System-wide settings and precedence rules
- **[API Reference](api/models.md)** - Python API documentation
- **[Architecture](architecture/overview.md)** - System design and components

## Examples

Browse **[complete examples](examples/README.md)**:

- `basic_agent.yaml` - Minimal valid agent
- `with_tools.yaml` - All tool types
- `with_evaluations.yaml` - Testing and metrics
- `with_global_config.yaml` - Configuration precedence

## Features Overview

### Define Agents in YAML

```yaml
name: Research Assistant
description: Answers research questions with citations
model:
  provider: openai
  name: gpt-4o
tools:
  - type: vectorstore
    source: "research-papers.json"
    vector_field: "embeddings"
    chunk_size: 500
  - type: mcp
    server: "web-search"
    description: Search the web for current information
evaluations:
  metrics:
    - name: groundedness
      threshold: 0.8
    - name: relevance
      threshold: 0.75
```

### Support Multiple Tool Types

1. **Vector Search** - Semantic search over documents and embeddings
2. **Functions** - Execute custom Python code from files
3. **MCP Servers** - Standardized integrations (GitHub, filesystem, databases, custom)
4. **Prompt Tools** - AI-powered semantic functions with templates

### Flexible Model Configuration

Configure LLM models at three levels:

- **Global**: Default model for all agents
- **Agent**: Override for specific agent
- **Metric**: Fine-grained per-evaluation-metric (GPT-4 for critical metrics, GPT-4o-mini for others)

### Multimodal Test Cases

Test agents with rich media:

```yaml
test_cases:
  - input: "Analyze this image and PDF"
    files:
      - image.png
      - document.pdf
    expected_tools: ["vision_analyzer"]
    ground_truth: "Expected analysis result"
```

## Project Status

**Version**: 0.1.0 (Pre-release)

- ✅ Core configuration schema (Pydantic models)
- ✅ YAML parsing and validation
- ✅ Environment variable support
- ✅ File reference resolution
- ⏳ CLI interface (planned for v0.2)
- ⏳ Agent execution engine (planned for v0.2)
- ⏳ Evaluation framework (planned for v0.2)
- ⏳ Deployment tools (planned for v0.3)

## Community & Support

- **GitHub Issues**: Report bugs or suggest features
- **Discussions**: Ask questions and share ideas
- **Contributing**: Read [CONTRIBUTING.md](CONTRIBUTING.md) to get involved

## License

MIT License - See LICENSE file for details

---

**Next Steps**:

- [Get started with installation →](getting-started/installation.md)
- [Try the quickstart →](getting-started/quickstart.md)
- [Explore examples →](examples/README.md)
