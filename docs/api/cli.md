# CLI API Reference

HoloDeck provides a command-line interface for project initialization, agent testing,
interactive chat, HTTP serving, deployment, MCP server management, and configuration.
This section documents the programmatic CLI API -- every public class, function, and
exception exposed by the `holodeck.cli` package.

---

## Main CLI

Entry point for the HoloDeck CLI application using Click.
Registers all seven subcommands and loads `.env` files on startup.

::: holodeck.cli.main.main
    options:
      docstring_style: google
      show_source: true

::: holodeck.cli.main._load_dotenv_files
    options:
      docstring_style: google
      show_source: true

---

## CLI Commands

### Init Command

Initialize a new HoloDeck project with bundled templates and an interactive wizard.

::: holodeck.cli.commands.init.init
    options:
      docstring_style: google
      show_source: true

::: holodeck.cli.commands.init.validate_template
    options:
      docstring_style: google
      show_source: true

::: holodeck.cli.commands.init._parse_comma_arg
    options:
      docstring_style: google
      show_source: true

---

### Test Command

Run tests for a HoloDeck agent with evaluation metrics and report generation.

::: holodeck.cli.commands.test.test
    options:
      docstring_style: google
      show_source: true

::: holodeck.cli.commands.test.SpinnerThread
    options:
      docstring_style: google
      show_source: true
      members:
        - __init__
        - run
        - stop

::: holodeck.cli.commands.test._save_report
    options:
      docstring_style: google
      show_source: true

---

### Chat Command

Start an interactive multi-turn chat session with an agent.

::: holodeck.cli.commands.chat.chat
    options:
      docstring_style: google
      show_source: true

::: holodeck.cli.commands.chat.ChatSpinnerThread
    options:
      docstring_style: google
      show_source: true
      members:
        - __init__
        - run
        - stop

::: holodeck.cli.commands.chat._run_chat_session
    options:
      docstring_style: google
      show_source: true

---

### Config Command

Manage HoloDeck global and project configuration files.

::: holodeck.cli.commands.config.config
    options:
      docstring_style: google
      show_source: true

::: holodeck.cli.commands.config.init
    options:
      docstring_style: google
      show_source: true

---

### Deploy Command

Build container images and deploy agents to cloud providers.

::: holodeck.cli.commands.deploy.deploy
    options:
      docstring_style: google
      show_source: true

::: holodeck.cli.commands.deploy.build
    options:
      docstring_style: google
      show_source: true

::: holodeck.cli.commands.deploy.run
    options:
      docstring_style: google
      show_source: true

::: holodeck.cli.commands.deploy.status
    options:
      docstring_style: google
      show_source: true

::: holodeck.cli.commands.deploy.destroy
    options:
      docstring_style: google
      show_source: true

::: holodeck.cli.commands.deploy.handle_deployment_errors
    options:
      docstring_style: google
      show_source: true

::: holodeck.cli.commands.deploy._generate_dockerfile_content
    options:
      docstring_style: google
      show_source: true

::: holodeck.cli.commands.deploy._prepare_build_context
    options:
      docstring_style: google
      show_source: true

::: holodeck.cli.commands.deploy._display_build_success
    options:
      docstring_style: google
      show_source: true

::: holodeck.cli.commands.deploy._load_agent_and_deployment
    options:
      docstring_style: google
      show_source: true

::: holodeck.cli.commands.deploy._ensure_azure_provider
    options:
      docstring_style: google
      show_source: true

::: holodeck.cli.commands.deploy._resolve_image_uri
    options:
      docstring_style: google
      show_source: true

---

### MCP Command

Manage MCP (Model Context Protocol) servers -- search, add, list, and remove.

::: holodeck.cli.commands.mcp.mcp
    options:
      docstring_style: google
      show_source: true

::: holodeck.cli.commands.mcp.search
    options:
      docstring_style: google
      show_source: true

::: holodeck.cli.commands.mcp.list_cmd
    options:
      docstring_style: google
      show_source: true

::: holodeck.cli.commands.mcp.add
    options:
      docstring_style: google
      show_source: true

::: holodeck.cli.commands.mcp.remove
    options:
      docstring_style: google
      show_source: true

#### MCP Helper Functions

::: holodeck.cli.commands.mcp._truncate
    options:
      docstring_style: google
      show_source: true

::: holodeck.cli.commands.mcp._get_transports
    options:
      docstring_style: google
      show_source: true

::: holodeck.cli.commands.mcp._get_transport_list
    options:
      docstring_style: google
      show_source: true

::: holodeck.cli.commands.mcp._get_version_display
    options:
      docstring_style: google
      show_source: true

::: holodeck.cli.commands.mcp._output_table
    options:
      docstring_style: google
      show_source: true

::: holodeck.cli.commands.mcp._output_json
    options:
      docstring_style: google
      show_source: true

::: holodeck.cli.commands.mcp._format_version_for_json
    options:
      docstring_style: google
      show_source: true

::: holodeck.cli.commands.mcp._extract_version_from_args
    options:
      docstring_style: google
      show_source: true

::: holodeck.cli.commands.mcp._list_output_table
    options:
      docstring_style: google
      show_source: true

::: holodeck.cli.commands.mcp._list_output_json
    options:
      docstring_style: google
      show_source: true

---

### Serve Command

Start an HTTP server exposing an agent via AG-UI or REST protocol.

::: holodeck.cli.commands.serve.serve
    options:
      docstring_style: google
      show_source: true

::: holodeck.cli.commands.serve._run_server
    options:
      docstring_style: google
      show_source: true

::: holodeck.cli.commands.serve._display_startup_info
    options:
      docstring_style: google
      show_source: true

---

## CLI Utilities

### Project Initializer

Project initialization and scaffolding logic.

::: holodeck.cli.utils.project_init.ProjectInitializer
    options:
      docstring_style: google
      show_source: true
      members:
        - __init__
        - validate_inputs
        - load_template
        - initialize

::: holodeck.cli.utils.project_init.get_model_for_provider
    options:
      docstring_style: google
      show_source: true

::: holodeck.cli.utils.project_init.get_mcp_server_config
    options:
      docstring_style: google
      show_source: true

::: holodeck.cli.utils.project_init.get_vectorstore_endpoint
    options:
      docstring_style: google
      show_source: true

::: holodeck.cli.utils.project_init.get_provider_api_key_env_var
    options:
      docstring_style: google
      show_source: true

::: holodeck.cli.utils.project_init.get_provider_endpoint_env_var
    options:
      docstring_style: google
      show_source: true

---

### Interactive Wizard

Interactive configuration wizard for `holodeck init`.

::: holodeck.cli.utils.wizard.run_wizard
    options:
      docstring_style: google
      show_source: true

::: holodeck.cli.utils.wizard.is_interactive
    options:
      docstring_style: google
      show_source: true

::: holodeck.cli.utils.wizard.WizardCancelledError
    options:
      docstring_style: google
      show_source: true

---

## CLI Exceptions

CLI-specific exception hierarchy. All exceptions inherit from `CLIError`.

::: holodeck.cli.exceptions.CLIError
    options:
      docstring_style: google
      show_source: true

::: holodeck.cli.exceptions.ValidationError
    options:
      docstring_style: google
      show_source: true

::: holodeck.cli.exceptions.InitError
    options:
      docstring_style: google
      show_source: true

::: holodeck.cli.exceptions.TemplateError
    options:
      docstring_style: google
      show_source: true

::: holodeck.cli.exceptions.ChatConfigError
    options:
      docstring_style: google
      show_source: true

::: holodeck.cli.exceptions.ChatAgentInitError
    options:
      docstring_style: google
      show_source: true

::: holodeck.cli.exceptions.ChatRuntimeError
    options:
      docstring_style: google
      show_source: true

::: holodeck.cli.exceptions.ChatValidationError
    options:
      docstring_style: google
      show_source: true

---

## Usage from Python

You can invoke CLI commands programmatically:

```python
from holodeck.cli.main import main
from click.testing import CliRunner

runner = CliRunner()

# Initialize a new project
result = runner.invoke(main, ['init', '--template', 'conversational', '--name', 'my-agent'])
print(result.output)

# Run tests
result = runner.invoke(main, ['test', 'path/to/agent.yaml'])
print(result.output)

# Start an interactive chat session
result = runner.invoke(main, ['chat', 'agent.yaml'])
print(result.output)

# Start an HTTP server
result = runner.invoke(main, ['serve', 'agent.yaml', '--port', '9000'])
print(result.output)

# Build a container image
result = runner.invoke(main, ['deploy', 'build', 'agent.yaml', '--dry-run'])
print(result.output)

# Deploy to cloud
result = runner.invoke(main, ['deploy', 'run', 'agent.yaml'])
print(result.output)

# Check deployment status
result = runner.invoke(main, ['deploy', 'status', 'agent.yaml'])
print(result.output)

# Destroy a deployment
result = runner.invoke(main, ['deploy', 'destroy', 'agent.yaml', '--force'])
print(result.output)

# Search MCP registry
result = runner.invoke(main, ['mcp', 'search', 'filesystem'])
print(result.output)

# Add an MCP server
result = runner.invoke(main, ['mcp', 'add', 'io.github.modelcontextprotocol/server-filesystem'])
print(result.output)

# List installed MCP servers
result = runner.invoke(main, ['mcp', 'list'])
print(result.output)

# Remove an MCP server
result = runner.invoke(main, ['mcp', 'remove', 'filesystem'])
print(result.output)

# Initialize configuration
result = runner.invoke(main, ['config', 'init', '-g'])
print(result.output)
```

## CLI Entry Point

The CLI is registered as the `holodeck` command via `pyproject.toml`:

```toml
[project.scripts]
holodeck = "holodeck.cli.main:main"
```

After installation, use from terminal:

```bash
# Initialize a new project with interactive wizard
holodeck init

# Quick non-interactive setup
holodeck init --name my-agent --non-interactive

# Run tests (defaults to agent.yaml in current directory)
holodeck test

# Or specify explicit path
holodeck test agent.yaml --verbose --output report.md

# Interactive chat session
holodeck chat agent.yaml

# Start HTTP server with AG-UI protocol
holodeck serve agent.yaml --port 8000 --protocol ag-ui

# Build and deploy containers
holodeck deploy build agent.yaml --tag v1.0.0
holodeck deploy run agent.yaml
holodeck deploy status agent.yaml
holodeck deploy destroy agent.yaml

# MCP server management
holodeck mcp search filesystem
holodeck mcp add io.github.modelcontextprotocol/server-filesystem
holodeck mcp list
holodeck mcp list --all
holodeck mcp remove filesystem

# Configuration management
holodeck config init -g
holodeck config init -p
```

## Related Documentation

- [Template Models](models.md#template-models): Template manifest and metadata models
- [Configuration Loading](config-loader.md): Configuration system
- [Test Runner](test-runner.md): Test execution
