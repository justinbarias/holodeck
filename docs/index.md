# HoloDeck - AI Agent Experimentation Platform

![HoloDeck Logo](assets/holodeck.png)

**HoloDeck** is an open-source experimentation platform for building, testing, and deploying AI agents through **YAML configuration**. Define intelligent agents entirely through configuration—no code required.

## Key Features

- **No-Code Agent Definition**: Define agents, tools, and evaluations in simple YAML files
- **Multi-Provider Support**: Ollama (local inference), OpenAI, Azure OpenAI, Anthropic
- **Flexible Tool Integration**: Vector stores, custom functions, MCP servers, and AI-powered tools
- **Built-in Testing & Evaluation**: Run evaluations with multiple metrics, customize models per metric
- **Multimodal Test Support**: Images, PDFs, Word docs, Excel sheets, and mixed media in test cases

## Get Started

```bash
uv tool install holodeck-ai@latest --prerelease allow --python 3.10
```

[Installation Guide](getting-started/installation.md) | [Quickstart Tutorial](getting-started/quickstart.md)

## Documentation

- **[Agent Configuration](guides/agent-configuration.md)** - Complete schema reference
- **[Tools Guide](guides/tools.md)** - All tool types explained with examples
- **[Evaluations](guides/evaluations.md)** - Testing and evaluation framework
- **[Global Configuration](guides/global-config.md)** - System-wide settings and precedence rules
- **[API Reference](api/models.md)** - Python API documentation

## Examples

Browse **[complete examples](examples/README.md)** including basic agents, tool integrations, evaluations, and configuration patterns.

## Project Status

**Version**: 0.2.0 (Development)

- Core configuration schema (Pydantic models)
- YAML parsing and validation
- Environment variable support
- CLI interface (holodeck command with init, test, chat, deploy)
- Agent execution engine (LLM provider integration, tool execution, memory)
- Interactive chat with spinner, token tracking, and adaptive status display
- Evaluation framework (AI-powered and NLP metrics with threshold validation)
- Deployment tools (planned for v0.3)
- Multi-agent orchestration (planned for v0.3)
- OpenTelemetry instrumentation (planned for v0.3)

## Community & Support

- **GitHub Issues**: Report bugs or suggest features
- **Discussions**: Ask questions and share ideas
- **Contributing**: Read [contributing guide](contributing.md) to get involved

## License

MIT License - See LICENSE file for details
