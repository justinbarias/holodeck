# Changelog

All notable changes to HoloDeck will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned Features

- **Agent Engine Phase 2**: Agent execution and LLM interaction
- **Tool Execution**: Dynamic tool loading and execution
- **Evaluation Engine**: AI-powered metrics (groundedness, relevance) and NLP metrics (F1, BLEU, ROUGE)
- **Deployment Engine**: Convert agents to production FastAPI endpoints
- **CLI Commands**: `holodeck run`, `holodeck evaluate`, `holodeck deploy`
- **Observability**: OpenTelemetry integration with GenAI semantic conventions
- **Plugin System**: Pre-built plugin packages for common integrations
- **Vector Stores**: Redis/Postgres-backed semantic search support

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
  - Python 3.14+ support

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

### User Story 1: Define Agent Configuration ✅ **0.0.1**

Implemented in 0.0.1:
- YAML-based agent configuration
- Schema validation
- Configuration loading and merging
- File references and environment variables

### User Story 2: Initialize New Agent Project (planned 0.1.0)

- `holodeck init` command
- Project scaffolding
- Template agent creation
- Directory structure setup

### User Story 3: Execute Agent (planned 0.1.0)

- `holodeck run` command
- Agent execution engine
- LLM provider integration
- Tool execution runtime

### User Story 4: Evaluate Results (planned 0.2.0)

- `holodeck evaluate` command
- Evaluation metric execution
- AI-powered and NLP metrics
- Result reporting

### User Story 5: Deploy Agent (planned 0.2.0)

- `holodeck deploy` command
- FastAPI endpoint generation
- Docker containerization
- Cloud deployment support

### User Story 6: Observability & Monitoring (planned 0.2.0)

- OpenTelemetry integration
- Trace and log instrumentation
- Cost tracking
- Performance monitoring

### User Story 7: Plugin System (planned 0.3.0)

- Plugin registry
- Plugin installation
- Pre-built plugin packages
- Custom plugin development

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
