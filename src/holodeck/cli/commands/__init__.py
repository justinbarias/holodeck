"""CLI commands for HoloDeck - Agent initialization and testing.

This package contains Click command implementations for the HoloDeck CLI,
including:

- `init`: Initialize a new HoloDeck project with templates
- `test`: Run tests for agents and evaluations
- `config init`: Initialize global or project configuration files
- (Future) `chat`: Interactive chat interface with agents
- (Future) `deploy`: Deploy agents to production

Example:
    from holodeck.cli.commands.init import init_command
    from holodeck.cli.commands.test import test_command

Attributes:
    init_command: Click command for project initialization
    test_command: Click command for running tests
"""
