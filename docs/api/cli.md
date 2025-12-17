# CLI API Reference

HoloDeck provides a command-line interface for project initialization, agent testing,
and configuration management. This section documents the programmatic CLI API.

## Main CLI

Entry point for the HoloDeck CLI application using Click.

::: holodeck.cli.main.main
    options:
      docstring_style: google

## CLI Commands

### Init Command

Initialize a new HoloDeck project with bundled templates.

::: holodeck.cli.commands.init.init
    options:
      docstring_style: google
      show_source: true

### Test Command

Run tests for a HoloDeck agent with evaluation and reporting.

::: holodeck.cli.commands.test.test
    options:
      docstring_style: google
      show_source: true

## CLI Utilities

Project initialization and scaffolding utilities.

::: holodeck.cli.utils.project_init.ProjectInitializer
    options:
      docstring_style: google
      show_source: true

## CLI Exceptions

CLI-specific exception handling.

::: holodeck.cli.exceptions.CLIError
    options:
      docstring_style: google

::: holodeck.cli.exceptions.ValidationError
    options:
      docstring_style: google

::: holodeck.cli.exceptions.InitError
    options:
      docstring_style: google

::: holodeck.cli.exceptions.TemplateError
    options:
      docstring_style: google

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
```

## CLI Entry Point

The CLI is registered as the `holodeck` command via `pyproject.toml`:

```toml
[project.scripts]
holodeck = "holodeck.cli.main:main"
```

After installation, use from terminal:

```bash
holodeck init --template conversational --name my-agent

# Run tests (defaults to agent.yaml in current directory)
holodeck test

# Or specify explicit path
holodeck test agent.yaml
```

## Related Documentation

- [Project Templates](../api/models.md#templates): Available templates
- [Configuration Loading](config-loader.md): Configuration system
- [Test Runner](test-runner.md): Test execution
