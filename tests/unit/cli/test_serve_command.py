"""Unit tests for the holodeck serve CLI command.

Tests cover:
- Serve command initialization and options
- Error handling for ConfigError, KeyboardInterrupt, and general exceptions
- _run_server async function
- _display_startup_info for AG-UI and REST protocols
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from holodeck.cli.commands.serve import _display_startup_info, _run_server, serve
from holodeck.lib.errors import ConfigError
from holodeck.models.config import ExecutionConfig
from holodeck.serve.models import ProtocolType

if TYPE_CHECKING:
    pass


@pytest.fixture
def runner() -> CliRunner:
    """Create a CLI runner."""
    return CliRunner()


@pytest.fixture
def temp_agent_config(tmp_path: Path) -> Path:
    """Create a temporary agent.yaml config file."""
    agent_file = tmp_path / "agent.yaml"
    agent_file.write_text(
        """
name: test-agent
description: A test agent

model:
  provider: openai
  name: gpt-4o

instructions:
  inline: You are a helpful assistant.
"""
    )
    return agent_file


@pytest.fixture
def mock_agent() -> MagicMock:
    """Create a mock agent configuration."""
    agent = MagicMock()
    agent.name = "test-agent"
    agent.description = "A test agent"
    # Set execution to None to avoid MagicMock issues with Pydantic validation
    agent.execution = None
    # Set observability to None to skip OTel initialization in tests
    agent.observability = None
    return agent


@pytest.fixture
def mock_execution_config() -> ExecutionConfig:
    """Create a mock execution configuration."""
    return ExecutionConfig(
        llm_timeout=60,
        file_timeout=30,
        download_timeout=30,
        cache_enabled=True,
        cache_dir=".holodeck/cache",
        verbose=False,
        quiet=False,
    )


class TestServeCommandOptions:
    """Tests for serve command options and parameters."""

    def test_serve_command_default_options(self, runner: CliRunner) -> None:
        """Test serve command with default options."""
        # Run with a non-existent file to check options are parsed
        result = runner.invoke(serve, ["--help"])
        assert result.exit_code == 0
        assert "--port" in result.output
        assert "--host" in result.output
        assert "--protocol" in result.output
        assert "--verbose" in result.output
        assert "--cors-origins" in result.output

    def test_serve_command_protocol_choices(self, runner: CliRunner) -> None:
        """Test serve command accepts valid protocol choices."""
        result = runner.invoke(serve, ["--help"])
        assert "ag-ui" in result.output
        assert "rest" in result.output


class TestServeCommandExecution:
    """Tests for serve command execution with mocked dependencies."""

    @patch("holodeck.config.loader.ConfigLoader.resolve_execution_config")
    @patch("holodeck.config.loader.ConfigLoader.load_global_config")
    @patch("holodeck.config.loader.ConfigLoader.load_project_config")
    @patch("holodeck.config.loader.ConfigLoader.load_agent_yaml")
    @patch("asyncio.run")
    @patch("holodeck.cli.commands.serve.setup_logging")
    def test_serve_command_loads_agent_config(
        self,
        mock_setup_logging: MagicMock,
        mock_asyncio_run: MagicMock,
        mock_load_agent: MagicMock,
        mock_load_project: MagicMock,
        mock_load_global: MagicMock,
        mock_resolve_exec: MagicMock,
        runner: CliRunner,
        temp_agent_config: Path,
        mock_agent: MagicMock,
        mock_execution_config: ExecutionConfig,
    ) -> None:
        """Test serve command loads agent configuration correctly."""
        mock_load_agent.return_value = mock_agent
        mock_load_project.return_value = None
        mock_load_global.return_value = None
        mock_resolve_exec.return_value = mock_execution_config

        result = runner.invoke(serve, [str(temp_agent_config)])

        assert result.exit_code == 0
        mock_load_agent.assert_called_once_with(str(temp_agent_config))

    @patch("holodeck.config.loader.ConfigLoader.resolve_execution_config")
    @patch("holodeck.config.loader.ConfigLoader.load_global_config")
    @patch("holodeck.config.loader.ConfigLoader.load_project_config")
    @patch("holodeck.config.loader.ConfigLoader.load_agent_yaml")
    @patch("asyncio.run")
    @patch("holodeck.cli.commands.serve.setup_logging")
    @patch("holodeck.config.context.agent_base_dir")
    def test_serve_command_sets_base_dir_context(
        self,
        mock_ctx: MagicMock,
        mock_setup_logging: MagicMock,
        mock_asyncio_run: MagicMock,
        mock_load_agent: MagicMock,
        mock_load_project: MagicMock,
        mock_load_global: MagicMock,
        mock_resolve_exec: MagicMock,
        runner: CliRunner,
        temp_agent_config: Path,
        mock_agent: MagicMock,
        mock_execution_config: ExecutionConfig,
    ) -> None:
        """Test serve command sets agent_base_dir context."""
        mock_load_agent.return_value = mock_agent
        mock_load_project.return_value = None
        mock_load_global.return_value = None
        mock_resolve_exec.return_value = mock_execution_config

        result = runner.invoke(serve, [str(temp_agent_config)])

        assert result.exit_code == 0
        mock_ctx.set.assert_called_once()

    @patch("holodeck.config.loader.ConfigLoader.resolve_execution_config")
    @patch("holodeck.config.loader.ConfigLoader.load_global_config")
    @patch("holodeck.config.loader.ConfigLoader.load_project_config")
    @patch("holodeck.config.loader.ConfigLoader.load_agent_yaml")
    @patch("asyncio.run")
    @patch("holodeck.cli.commands.serve.setup_logging")
    def test_serve_command_parses_cors_origins(
        self,
        mock_setup_logging: MagicMock,
        mock_asyncio_run: MagicMock,
        mock_load_agent: MagicMock,
        mock_load_project: MagicMock,
        mock_load_global: MagicMock,
        mock_resolve_exec: MagicMock,
        runner: CliRunner,
        temp_agent_config: Path,
        mock_agent: MagicMock,
        mock_execution_config: ExecutionConfig,
    ) -> None:
        """Test serve command parses comma-separated CORS origins."""
        mock_load_agent.return_value = mock_agent
        mock_load_project.return_value = None
        mock_load_global.return_value = None
        mock_resolve_exec.return_value = mock_execution_config

        result = runner.invoke(
            serve,
            [
                str(temp_agent_config),
                "--cors-origins",
                "http://localhost:3000, https://example.com",
            ],
        )

        assert result.exit_code == 0
        # The _run_server should have been called with parsed origins
        call_kwargs = mock_asyncio_run.call_args
        assert call_kwargs is not None

    @patch("holodeck.config.loader.ConfigLoader.resolve_execution_config")
    @patch("holodeck.config.loader.ConfigLoader.load_global_config")
    @patch("holodeck.config.loader.ConfigLoader.load_project_config")
    @patch("holodeck.config.loader.ConfigLoader.load_agent_yaml")
    @patch("asyncio.run")
    @patch("holodeck.cli.commands.serve.setup_logging")
    def test_serve_command_protocol_ag_ui(
        self,
        mock_setup_logging: MagicMock,
        mock_asyncio_run: MagicMock,
        mock_load_agent: MagicMock,
        mock_load_project: MagicMock,
        mock_load_global: MagicMock,
        mock_resolve_exec: MagicMock,
        runner: CliRunner,
        temp_agent_config: Path,
        mock_agent: MagicMock,
        mock_execution_config: ExecutionConfig,
    ) -> None:
        """Test serve command with AG-UI protocol."""
        mock_load_agent.return_value = mock_agent
        mock_load_project.return_value = None
        mock_load_global.return_value = None
        mock_resolve_exec.return_value = mock_execution_config

        result = runner.invoke(serve, [str(temp_agent_config), "--protocol", "ag-ui"])

        assert result.exit_code == 0

    @patch("holodeck.config.loader.ConfigLoader.resolve_execution_config")
    @patch("holodeck.config.loader.ConfigLoader.load_global_config")
    @patch("holodeck.config.loader.ConfigLoader.load_project_config")
    @patch("holodeck.config.loader.ConfigLoader.load_agent_yaml")
    @patch("asyncio.run")
    @patch("holodeck.cli.commands.serve.setup_logging")
    def test_serve_command_protocol_rest(
        self,
        mock_setup_logging: MagicMock,
        mock_asyncio_run: MagicMock,
        mock_load_agent: MagicMock,
        mock_load_project: MagicMock,
        mock_load_global: MagicMock,
        mock_resolve_exec: MagicMock,
        runner: CliRunner,
        temp_agent_config: Path,
        mock_agent: MagicMock,
        mock_execution_config: ExecutionConfig,
    ) -> None:
        """Test serve command with REST protocol."""
        mock_load_agent.return_value = mock_agent
        mock_load_project.return_value = None
        mock_load_global.return_value = None
        mock_resolve_exec.return_value = mock_execution_config

        result = runner.invoke(serve, [str(temp_agent_config), "--protocol", "rest"])

        assert result.exit_code == 0

    @patch("holodeck.config.loader.ConfigLoader.resolve_execution_config")
    @patch("holodeck.config.loader.ConfigLoader.load_global_config")
    @patch("holodeck.config.loader.ConfigLoader.load_project_config")
    @patch("holodeck.config.loader.ConfigLoader.load_agent_yaml")
    @patch("asyncio.run")
    @patch("holodeck.cli.commands.serve.setup_logging")
    def test_serve_command_verbose_mode(
        self,
        mock_setup_logging: MagicMock,
        mock_asyncio_run: MagicMock,
        mock_load_agent: MagicMock,
        mock_load_project: MagicMock,
        mock_load_global: MagicMock,
        mock_resolve_exec: MagicMock,
        runner: CliRunner,
        temp_agent_config: Path,
        mock_agent: MagicMock,
        mock_execution_config: ExecutionConfig,
    ) -> None:
        """Test serve command with verbose mode enabled."""
        mock_load_agent.return_value = mock_agent
        mock_load_project.return_value = None
        mock_load_global.return_value = None
        mock_resolve_exec.return_value = mock_execution_config

        result = runner.invoke(serve, [str(temp_agent_config), "--verbose"])

        assert result.exit_code == 0
        mock_setup_logging.assert_called_with(verbose=True, quiet=False)

    @patch("holodeck.config.loader.ConfigLoader.resolve_execution_config")
    @patch("holodeck.config.loader.ConfigLoader.load_global_config")
    @patch("holodeck.config.loader.ConfigLoader.load_project_config")
    @patch("holodeck.config.loader.ConfigLoader.load_agent_yaml")
    @patch("asyncio.run")
    @patch("holodeck.cli.commands.serve.setup_logging")
    def test_serve_command_custom_port(
        self,
        mock_setup_logging: MagicMock,
        mock_asyncio_run: MagicMock,
        mock_load_agent: MagicMock,
        mock_load_project: MagicMock,
        mock_load_global: MagicMock,
        mock_resolve_exec: MagicMock,
        runner: CliRunner,
        temp_agent_config: Path,
        mock_agent: MagicMock,
        mock_execution_config: ExecutionConfig,
    ) -> None:
        """Test serve command with custom port."""
        mock_load_agent.return_value = mock_agent
        mock_load_project.return_value = None
        mock_load_global.return_value = None
        mock_resolve_exec.return_value = mock_execution_config

        result = runner.invoke(serve, [str(temp_agent_config), "--port", "9000"])

        assert result.exit_code == 0


class TestServeCommandErrorHandling:
    """Tests for serve command error handling."""

    @patch("holodeck.config.loader.ConfigLoader.load_agent_yaml")
    @patch("holodeck.cli.commands.serve.setup_logging")
    def test_serve_command_config_error(
        self,
        mock_setup_logging: MagicMock,
        mock_load_agent: MagicMock,
        runner: CliRunner,
        temp_agent_config: Path,
    ) -> None:
        """Test serve command handles ConfigError gracefully."""
        mock_load_agent.side_effect = ConfigError("model", "Invalid configuration")

        result = runner.invoke(serve, [str(temp_agent_config)])

        assert result.exit_code == 1
        assert "Error: Failed to load agent configuration" in result.output
        assert "Invalid configuration" in result.output

    @patch("holodeck.config.loader.ConfigLoader.resolve_execution_config")
    @patch("holodeck.config.loader.ConfigLoader.load_global_config")
    @patch("holodeck.config.loader.ConfigLoader.load_project_config")
    @patch("holodeck.config.loader.ConfigLoader.load_agent_yaml")
    @patch("asyncio.run")
    @patch("holodeck.cli.commands.serve.setup_logging")
    def test_serve_command_keyboard_interrupt(
        self,
        mock_setup_logging: MagicMock,
        mock_asyncio_run: MagicMock,
        mock_load_agent: MagicMock,
        mock_load_project: MagicMock,
        mock_load_global: MagicMock,
        mock_resolve_exec: MagicMock,
        runner: CliRunner,
        temp_agent_config: Path,
        mock_agent: MagicMock,
        mock_execution_config: ExecutionConfig,
    ) -> None:
        """Test serve command handles KeyboardInterrupt gracefully."""
        mock_load_agent.return_value = mock_agent
        mock_load_project.return_value = None
        mock_load_global.return_value = None
        mock_resolve_exec.return_value = mock_execution_config
        mock_asyncio_run.side_effect = KeyboardInterrupt()

        result = runner.invoke(serve, [str(temp_agent_config)])

        assert result.exit_code == 130
        assert "Server stopped." in result.output

    @patch("holodeck.config.loader.ConfigLoader.resolve_execution_config")
    @patch("holodeck.config.loader.ConfigLoader.load_global_config")
    @patch("holodeck.config.loader.ConfigLoader.load_project_config")
    @patch("holodeck.config.loader.ConfigLoader.load_agent_yaml")
    @patch("asyncio.run")
    @patch("holodeck.cli.commands.serve.setup_logging")
    def test_serve_command_unexpected_error(
        self,
        mock_setup_logging: MagicMock,
        mock_asyncio_run: MagicMock,
        mock_load_agent: MagicMock,
        mock_load_project: MagicMock,
        mock_load_global: MagicMock,
        mock_resolve_exec: MagicMock,
        runner: CliRunner,
        temp_agent_config: Path,
        mock_agent: MagicMock,
        mock_execution_config: ExecutionConfig,
    ) -> None:
        """Test serve command handles unexpected errors gracefully."""
        mock_load_agent.return_value = mock_agent
        mock_load_project.return_value = None
        mock_load_global.return_value = None
        mock_resolve_exec.return_value = mock_execution_config
        mock_asyncio_run.side_effect = RuntimeError("Something went wrong")

        result = runner.invoke(serve, [str(temp_agent_config)])

        assert result.exit_code == 1
        assert "Error:" in result.output
        assert "Something went wrong" in result.output


class TestRunServer:
    """Tests for _run_server async function."""

    @pytest.mark.asyncio
    async def test_run_server_creates_agent_server(
        self, mock_agent: MagicMock, mock_execution_config: ExecutionConfig
    ) -> None:
        """Test _run_server creates AgentServer with correct params."""
        with (
            patch("holodeck.serve.server.AgentServer") as mock_server_class,
            patch("uvicorn.Config"),
            patch("uvicorn.Server") as mock_server_cls,
            patch("holodeck.cli.commands.serve._display_startup_info"),
        ):
            mock_server = MagicMock()
            mock_server.create_app.return_value = MagicMock()
            mock_server.start = AsyncMock()
            mock_server.stop = AsyncMock()
            mock_server_class.return_value = mock_server

            mock_uvicorn_server = MagicMock()
            mock_uvicorn_server.serve = AsyncMock()
            mock_server_cls.return_value = mock_uvicorn_server

            await _run_server(
                agent=mock_agent,
                host="127.0.0.1",
                port=8000,
                protocol=ProtocolType.AG_UI,
                cors_origins=["*"],
                verbose=False,
                execution_config=mock_execution_config,
            )

            mock_server_class.assert_called_once_with(
                agent_config=mock_agent,
                protocol=ProtocolType.AG_UI,
                host="127.0.0.1",
                port=8000,
                cors_origins=["*"],
                debug=False,
                execution_config=mock_execution_config,
                observability_enabled=False,
            )
            mock_server.create_app.assert_called_once()
            mock_server.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_server_calls_stop_on_cleanup(
        self, mock_agent: MagicMock, mock_execution_config: ExecutionConfig
    ) -> None:
        """Test _run_server calls server.stop() in finally block."""
        with (
            patch("holodeck.serve.server.AgentServer") as mock_server_class,
            patch("uvicorn.Config"),
            patch("uvicorn.Server") as mock_server_cls,
            patch("holodeck.cli.commands.serve._display_startup_info"),
        ):
            mock_server = MagicMock()
            mock_server.create_app.return_value = MagicMock()
            mock_server.start = AsyncMock()
            mock_server.stop = AsyncMock()
            mock_server_class.return_value = mock_server

            mock_uvicorn_server = MagicMock()
            mock_uvicorn_server.serve = AsyncMock()
            mock_server_cls.return_value = mock_uvicorn_server

            await _run_server(
                agent=mock_agent,
                host="127.0.0.1",
                port=8000,
                protocol=ProtocolType.AG_UI,
                cors_origins=["*"],
                verbose=False,
                execution_config=mock_execution_config,
            )

            mock_server.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_server_uvicorn_config_verbose_mode(
        self, mock_agent: MagicMock, mock_execution_config: ExecutionConfig
    ) -> None:
        """Test _run_server configures uvicorn with debug log level when verbose."""
        with (
            patch("holodeck.serve.server.AgentServer") as mock_server_class,
            patch("uvicorn.Config") as mock_config,
            patch("uvicorn.Server") as mock_server_cls,
            patch("holodeck.cli.commands.serve._display_startup_info"),
        ):
            mock_server = MagicMock()
            mock_server.create_app.return_value = MagicMock()
            mock_server.start = AsyncMock()
            mock_server.stop = AsyncMock()
            mock_server_class.return_value = mock_server

            mock_uvicorn_server = MagicMock()
            mock_uvicorn_server.serve = AsyncMock()
            mock_server_cls.return_value = mock_uvicorn_server

            await _run_server(
                agent=mock_agent,
                host="127.0.0.1",
                port=8000,
                protocol=ProtocolType.AG_UI,
                cors_origins=["*"],
                verbose=True,
                execution_config=mock_execution_config,
            )

            # Check uvicorn.Config was called with debug log level
            config_call = mock_config.call_args
            assert config_call is not None
            assert config_call.kwargs["log_level"] == "debug"


class TestDisplayStartupInfo:
    """Tests for _display_startup_info function."""

    def test_display_startup_info_ag_ui_protocol(
        self, mock_agent: MagicMock, capsys
    ) -> None:
        """Test startup info display for AG-UI protocol."""
        _display_startup_info(
            agent=mock_agent,
            protocol=ProtocolType.AG_UI,
            host="127.0.0.1",
            port=8000,
        )

        captured = capsys.readouterr()
        output = captured.out

        assert "HoloDeck Agent Server" in output
        assert "test-agent" in output
        assert "ag-ui" in output
        assert "http://127.0.0.1:8000" in output
        assert "/awp" in output
        assert "/health" in output
        assert "/ready" in output
        assert "Ctrl+C" in output

    def test_display_startup_info_rest_protocol(
        self, mock_agent: MagicMock, capsys
    ) -> None:
        """Test startup info display for REST protocol."""
        _display_startup_info(
            agent=mock_agent,
            protocol=ProtocolType.REST,
            host="0.0.0.0",  # noqa: S104 - test data only
            port=9000,
        )

        captured = capsys.readouterr()
        output = captured.out

        assert "HoloDeck Agent Server" in output
        assert "test-agent" in output
        assert "rest" in output
        assert "http://0.0.0.0:9000" in output
        assert "/chat" in output
        assert "/stream" in output
        assert "/health" in output
        assert "/ready" in output

    def test_display_startup_info_shows_endpoints(
        self, mock_agent: MagicMock, capsys
    ) -> None:
        """Test startup info shows endpoints section."""
        _display_startup_info(
            agent=mock_agent,
            protocol=ProtocolType.AG_UI,
            host="127.0.0.1",
            port=8000,
        )

        captured = capsys.readouterr()
        output = captured.out

        assert "Endpoints:" in output
        assert "GET" in output
        assert "POST" in output
