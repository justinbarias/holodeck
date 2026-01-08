# Changelog

All notable changes to HoloDeck will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned Features

- **Deployment Engine**: Convert agents to production FastAPI endpoints
- **CLI Commands**: `holodeck deploy`
- **Plugin System**: Pre-built plugin packages for common integrations
- **Agent Framework Migration**: Plans to migrate away from Semantic Kernel to another agent framework (either Agents Framework, or Google ADK, alternatively support for Claude Agent SDK)

---

## [0.3.0] - 2026-01-08

### Added

- **OpenTelemetry Observability**: Full observability instrumentation with GenAI semantic conventions
  - OpenTelemetry configuration models (traces, metrics, logs)
  - OTLP export support for traces and metrics
- **Agent Local Server (`holodeck serve`)**: REST API server for agents
  - FastAPI-based REST endpoints for agent invocation
  - AG-UI compliant endpoint for agent interaction

---

## [0.1.7] - 2025-12-27

### Added

- **MCP CLI Commands**: Complete CLI for managing MCP servers
  - `holodeck mcp search`: Search MCP registry for servers
  - `holodeck mcp add`: Add MCP servers to configuration
  - `holodeck mcp list`: List configured servers (agent and global)
  - `holodeck mcp remove`: Remove MCP servers from configuration
  - Global MCP server merge into agent configurations
- **Structured Data Ingestion**: Loader and vectorstore integration for structured data sources
- **Vectorstore Reranking**: Reranking support for vectorstore search results
- **Interactive Config Wizard Enhancements**:
  - Template selection step
  - LLM provider selection step
- **DeepEval Metrics**: DeepEval integration as alternative/complement to Azure AI Evals
- **CLI Defaults**: `agent.yaml` as default config for `chat` and `test` commands
- **New Package Entrypoint**: Added `holodeck-ai` script entrypoint

### Changed

- **Vector Store Providers**: Removed Redis support, added PostgreSQL (pgvector), Pinecone, and Qdrant
- **Documentation**: Updated for `uv tool install`, Ollama as preferred provider
- **Test Progress/Reporting**: Improved display and refactored agent_factory
- **Schema Validation**: Relaxed validation for better flexibility

### Fixed

- Telemetry warning in CLI
- CNAME configuration bug

---

## [0.1.6] - 2025-11-28

### Added

- **MCP Tool Integration**: Full Model Context Protocol (MCP) tool support with stdio transport
- MCP server configuration and connection management
- Tool discovery and invocation via MCP protocol

### Fixed

- Instruction loading issues in agent configuration

---

## [0.1.5] - 2025-11-27

### Added

- **Project and User Config Support**: Execution config resolution now supports project-level and user-level configuration files

### Fixed

- ChromaDB connection issues

---

## [0.1.4] - 2025-11-27

### Fixed

- PyPI release by removing local version identifiers

---

## [0.1.3] - 2025-11-27

### Added

- **ChromaDB Support**: Explicit ChromaDB vector store integration

### Changed

- **Package Manager**: Switched from Poetry to uv for faster dependency management

### Fixed

- Test logging improvements
- RedisVL compatibility issues
- CLI quiet mode behavior

---

## [0.1.2] - 2025-11-26

### Added

- **Ollama Endpoint Support**: Local LLM execution via Ollama
- **Vector Stores Setup Guide**: Comprehensive Redis vector store documentation
- Claude Code integration for development assistance

---

## [0.1.1] - 2025-11-25

### Added

- **Semantic Kernel Vector Store Abstractions**: Support for all vector store providers (Redis, ChromaDB, etc.)
- Agent config execution settings applied to Semantic Kernel

---

## [0.1.0] - 2025-11-23

### Added

- **Chat Models and Validation Pipeline**: Scaffold for interactive chat functionality
- **Markdown Report Generation**: Comprehensive test result reporting (T123-T127)
- **Progress Display Enhancements**: Spinner, ANSI colors, elapsed time display
- **Per-Test Metric Resolution**: EvaluationMetric objects for fine-grained metric configuration (T095-T096)
- **File Processing Improvements**: Enhanced file input handling

### Changed

- Consolidated and refactored tests to parameterized tests for better maintainability
- Config init command improvements

---

## [0.0.14] - 2025-11-15

### Fixed

- Poetry development dependencies
- MkDocs build step
- Poetry version configuration
- Various Poetry configuration issues

---

## [0.0.7] - 2025-11-08

### Added

- **Agent Execution Implementation**: Core agent execution engine
- **Evaluators**: User Story 1 evaluator implementation
- **Response Format Definition**: Phase 4 implementation (T014-T019)
- **Global Settings Configuration**: Phase 2 & 3 with TDD approach

---

## [0.0.6] - 2025-10-25

### Added

- **`holodeck init` Command**: Complete project initialization with templates
  - Phase 8: Polish & QA for init command
  - Phase 7: Project metadata specification (US5)
  - Phase 5: Sample files and examples generation (US3)
  - User Story 2: Project template selection (Phase 4)
- Core init engine implementation
- Basic agent creation from templates
- ConfigLoader returns GlobalConfig rather than dict

---

## [0.0.5] - 2025-10-20

### Fixed

- Version tag configuration

---

## [0.0.4] - 2025-10-20

### Added

- GitHub release workflow
- Automated PyPI publishing

---

## [0.0.1] - 2025-10-19

### Added - User Story 1: Define Agent Configuration

#### Core Features

- **Agent Configuration Schema**: Complete YAML-based agent configuration with Pydantic validation

  - Agent metadata (name, description)
  - LLM provider configuration (OpenAI, Azure OpenAI, Anthropic)
  - Model parameters (temperature, max_tokens)
  - Instructions (inline or file-based)
  - Tools array with type discrimination
  - Test cases with expected behavior validation
  - Evaluation metrics with flexible model configuration

- **Configuration Loading & Validation** (`ConfigLoader`):

  - Load and parse agent.yaml files
  - Validate against Pydantic schema with user-friendly error messages
  - File path resolution (relative to agent.yaml directory)
  - Environment variable substitution (${VAR_NAME} pattern)
  - Precedence hierarchy: agent.yaml > environment variables > global config

- **Global Configuration Support**:
  - Load ~/.holodeck/config.yaml for system-wide settings
  - Provider configurations at global level
  - Tool configurations at global level
  - Configuration merging with proper precedence

#### Data Models

- **LLMProvider Model**:

  - Multi-provider support (openai, azure_openai, anthropic)
  - Model selection and parameter configuration
  - Temperature range validation (0-2)
  - Max tokens validation (>0)
  - Azure-specific endpoint configuration

- **Tool Models** (Discriminated Union):

  - **VectorstoreTool**: Vector search with source, embedding model, chunk size/overlap
  - **FunctionTool**: Python function tools with parameters schema
  - **MCPTool**: Model Context Protocol server integration
  - **PromptTool**: AI-powered semantic functions with template support
  - Tool type validation and discrimination

- **Evaluation Models**:

  - Metric configuration with name, threshold, enabled flag
  - Per-metric model override for flexible configuration
  - AI-powered and NLP metrics support

- **TestCase Model**:

  - Test inputs with expected behaviors
  - Ground truth for validation
  - Expected tool usage tracking
  - Evaluation metrics per test

- **Agent Model**:

  - Complete agent definition
  - All field validations and constraints
  - Tool and evaluation composition

- **GlobalConfig Model**:
  - Provider registry
  - Vectorstore configurations
  - Deployment settings

#### Error Handling

- **Custom Exception Hierarchy**:

  - `HoloDeckError`: Base exception
  - `ConfigError`: Configuration-specific errors
  - `ValidationError`: Schema validation errors with field details
  - `FileNotFoundError`: File resolution errors with path suggestions

- **Human-Readable Error Messages**:
  - Field names and types in validation errors
  - Actual vs. expected values
  - File paths with suggestions
  - Nested error flattening for complex schemas

#### Infrastructure & Tooling

- **Development Setup**:

  - Makefile with 30+ development commands
  - Poetry dependency management
  - Pre-commit hooks (black, ruff, mypy, detect-secrets)
  - Python 3.10+ support

- **Testing**:

  - Unit test suite with 11 test files covering all models
  - Integration test suite for end-to-end workflows
  - 80%+ code coverage requirement
  - Test execution: `make test`, `make test-coverage`, `make test-parallel`

- **Code Quality**:

  - Black code formatting (88 char line length)
  - Ruff linting (pycodestyle, pyflakes, isort, flake8-bugbear, pyupgrade, pep8-naming, flake8-simplify, bandit)
  - MyPy type checking with strict settings
  - Security scanning (safety, bandit, detect-secrets)
  - Automated pre-commit validation

- **Documentation**:
  - MkDocs site configuration with Material theme
  - Getting Started guide (installation, quickstart)
  - Configuration guides (agent config, tools, evaluations, global config, file references)
  - Example agent configurations (basic, with tools, with evaluations, with global config)
  - API reference documentation (ConfigLoader, Pydantic models)
  - Architecture documentation (configuration loading flow)

### Features Summary by Component

#### ConfigLoader API

```python
loader = ConfigLoader()
agent = loader.load_agent_yaml("agent.yaml")  # Returns Agent instance
```

- Parse YAML to Agent instances
- Automatic environment variable substitution
- File reference resolution with validation
- Configuration precedence handling
- Comprehensive error reporting

#### Schema Support

- **File References**: Instructions and tool definitions can be loaded from files
- **Environment Variables**: ${ENV_VAR} patterns supported throughout configs
- **Type Discrimination**: Tool types automatically validated and parsed
- **Nested Validation**: Complex nested structures validated properly

#### Testing Coverage

**Unit Tests** (11 files):

- `test_errors.py` - Exception handling and messaging
- `test_env_loader.py` - Environment variable substitution
- `test_defaults.py` - Default configuration handling
- `test_validator.py` - Validation utilities
- `test_tool_models.py` - Tool type validation and discrimination
- `test_llm_models.py` - LLM provider configuration
- `test_evaluation_models.py` - Evaluation metric configuration
- `test_testcase_models.py` - Test case validation
- `test_agent_models.py` - Agent schema validation
- `test_globalconfig_models.py` - Global configuration handling
- `test_config_loader.py` - ConfigLoader functionality

**Integration Tests** (1 file):

- `test_config_end_to_end.py` - Full workflow testing

### Known Limitations

#### Version 0.0.1 Scope

- **CLI Not Implemented**: No command-line interface (planned for User Story 2)
- **No Agent Execution**: Agent models are validated but not executed (Phase 2 feature)
- **No Tool Execution**: Tools are defined but not executed (Phase 2 feature)
- **No Evaluation Engine**: Metrics are configured but not executed (Phase 2 feature)
- **No Deployment**: No FastAPI endpoint generation or Docker deployment (Phase 2-3 features)
- **No Observability**: OpenTelemetry integration planned for Phase 2
- **No Plugin System**: Plugin packages not yet available (Phase 3 feature)

#### Validation Limitations

- **File Validation**: Only checks file existence, not content validity
- **LLM Provider APIs**: No actual API testing (would require credentials)
- **Tool Validation**: Type validation only, no runtime validation

#### Known Issues

None reported in 0.0.1.

---

## How to Use This Changelog

- **[Unreleased]**: Features coming in future releases
- **Semantic Versioning**: MAJOR.MINOR.PATCH
  - **MAJOR**: Breaking changes or new architecture
  - **MINOR**: New features and functionality
  - **PATCH**: Bug fixes and improvements
- **Categories**: Added (new features), Changed (modifications), Fixed (bug fixes), Deprecated (to be removed), Removed (deprecated features deleted), Security (security fixes)

---

## Roadmap

- [x] **v0.1** - Core agent engine + CLI
- [x] **v0.2** - Evaluation framework
- [ ] **v0.3** - API deployment
- [ ] **v0.4** - Multi-agent orchestration & workflows
- [ ] **v0.5** - Web UI (no-code editor)
- [ ] **v0.6** - Enterprise features (SSO, audit logs, RBAC)
- [ ] **v1.0** - Production-ready release

---

## Previous Versions

### Development Versions

- **Pre-0.0.1**: Architecture planning and vision definition
  - Project vision (VISION.md)
  - Architecture documentation
  - Specification and planning

---

## Contributing

See [CONTRIBUTING.md](contributing.md) for guidelines on:

- Development setup
- Running tests
- Code style requirements
- Submitting pull requests

## License

HoloDeck is released under the MIT License. See LICENSE file for details.

---

## Changelog Format

We follow [Keep a Changelog](https://keepachangelog.com/) format:

- **Added**: New features
- **Changed**: Changes to existing functionality
- **Deprecated**: Features to be removed in future versions
- **Removed**: Features that have been removed
- **Fixed**: Bug fixes
- **Security**: Security-related changes

---

## Quick Links

- [Getting Started](getting-started/quickstart.md)
- [Configuration Guide](guides/agent-configuration.md)
- [API Reference](api/models.md)
- [Contributing Guide](contributing.md)

---

[unreleased]: https://github.com/justinbarias/holodeck/compare/0.3.0...HEAD
[0.3.0]: https://github.com/justinbarias/holodeck/compare/0.1.7...0.3.0
[0.1.7]: https://github.com/justinbarias/holodeck/compare/0.1.6...0.1.7
[0.1.6]: https://github.com/justinbarias/holodeck/compare/0.1.5...0.1.6
[0.1.5]: https://github.com/justinbarias/holodeck/compare/0.1.4...0.1.5
[0.1.4]: https://github.com/justinbarias/holodeck/compare/0.1.3...0.1.4
[0.1.3]: https://github.com/justinbarias/holodeck/compare/0.1.2...0.1.3
[0.1.2]: https://github.com/justinbarias/holodeck/compare/0.1.1...0.1.2
[0.1.1]: https://github.com/justinbarias/holodeck/compare/0.1.0...0.1.1
[0.1.0]: https://github.com/justinbarias/holodeck/compare/0.0.14...0.1.0
[0.0.14]: https://github.com/justinbarias/holodeck/compare/0.0.7...0.0.14
[0.0.7]: https://github.com/justinbarias/holodeck/compare/0.0.6...0.0.7
[0.0.6]: https://github.com/justinbarias/holodeck/compare/0.0.5...0.0.6
[0.0.5]: https://github.com/justinbarias/holodeck/compare/0.0.4...0.0.5
[0.0.4]: https://github.com/justinbarias/holodeck/compare/0.0.1...0.0.4
[0.0.1]: https://github.com/justinbarias/holodeck/releases/tag/0.0.1
