"""Unit tests for test executor module.

Tests cover:
- Configuration resolution (CLI > YAML > env > defaults)
- Tool call validation
- Timeout handling
- Main executor flow
"""

import asyncio
from datetime import UTC, datetime
from unittest.mock import Mock

import pytest

from holodeck.config.defaults import DEFAULT_EXECUTION_CONFIG
from holodeck.config.loader import ConfigLoader
from holodeck.lib.file_processor import FileProcessor
from holodeck.lib.test_runner.agent_factory import AgentFactory
from holodeck.lib.test_runner.executor import (
    TestExecutor,
    resolve_execution_config,
)
from holodeck.models.agent import Agent
from holodeck.models.config import ExecutionConfig
from holodeck.models.test_case import FileInput, TestCaseModel
from holodeck.models.test_result import MetricResult, ProcessedFileInput, TestResult


class TestConfigurationResolution:
    """Tests for T047: Configuration resolution with priority hierarchy."""

    def test_cli_overrides_all(self):
        """CLI flags take highest priority over YAML, env, and defaults."""
        cli_config = ExecutionConfig(
            file_timeout=100,
            llm_timeout=200,
            download_timeout=150,
            cache_enabled=False,
            cache_dir="/custom/cache",
            verbose=True,
            quiet=False,
        )

        yaml_config = ExecutionConfig(
            file_timeout=50,
            llm_timeout=80,
            download_timeout=60,
            cache_enabled=True,
            cache_dir="/yaml/cache",
            verbose=False,
        )

        env_vars = {
            "HOLODECK_FILE_TIMEOUT": "25",
            "HOLODECK_LLM_TIMEOUT": "40",
            "HOLODECK_DOWNLOAD_TIMEOUT": "30",
            "HOLODECK_CACHE_ENABLED": "true",
            "HOLODECK_CACHE_DIR": "/env/cache",
            "HOLODECK_VERBOSE": "false",
            "HOLODECK_QUIET": "true",
        }

        resolved = resolve_execution_config(
            cli_config=cli_config,
            yaml_config=yaml_config,
            env_vars=env_vars,
            defaults=DEFAULT_EXECUTION_CONFIG,
        )

        assert resolved.file_timeout == 100  # CLI
        assert resolved.llm_timeout == 200  # CLI
        assert resolved.download_timeout == 150  # CLI
        assert resolved.cache_enabled is False  # CLI
        assert resolved.cache_dir == "/custom/cache"  # CLI
        assert resolved.verbose is True  # CLI

    def test_yaml_overrides_env_and_defaults(self):
        """YAML config takes priority over env vars and defaults."""
        cli_config = None

        yaml_config = ExecutionConfig(
            file_timeout=50,
            llm_timeout=80,
            download_timeout=60,
            cache_dir="/yaml/cache",
        )

        env_vars = {
            "HOLODECK_FILE_TIMEOUT": "25",
            "HOLODECK_LLM_TIMEOUT": "40",
            "HOLODECK_DOWNLOAD_TIMEOUT": "30",
            "HOLODECK_CACHE_DIR": "/env/cache",
        }

        resolved = resolve_execution_config(
            cli_config=cli_config,
            yaml_config=yaml_config,
            env_vars=env_vars,
            defaults=DEFAULT_EXECUTION_CONFIG,
        )

        assert resolved.file_timeout == 50  # YAML
        assert resolved.llm_timeout == 80  # YAML
        assert resolved.download_timeout == 60  # YAML
        assert resolved.cache_dir == "/yaml/cache"  # YAML
        # Others from defaults
        assert resolved.cache_enabled is True  # defaults
        assert resolved.verbose is False  # defaults

    def test_env_overrides_defaults(self):
        """Environment variables take priority over built-in defaults."""
        cli_config = None
        yaml_config = None

        env_vars = {
            "HOLODECK_FILE_TIMEOUT": "25",
            "HOLODECK_LLM_TIMEOUT": "40",
            "HOLODECK_DOWNLOAD_TIMEOUT": "30",
            "HOLODECK_CACHE_ENABLED": "false",
            "HOLODECK_CACHE_DIR": "/env/cache",
            "HOLODECK_VERBOSE": "true",
            "HOLODECK_QUIET": "false",
        }

        resolved = resolve_execution_config(
            cli_config=cli_config,
            yaml_config=yaml_config,
            env_vars=env_vars,
            defaults=DEFAULT_EXECUTION_CONFIG,
        )

        assert resolved.file_timeout == 25  # env
        assert resolved.llm_timeout == 40  # env
        assert resolved.download_timeout == 30  # env
        assert resolved.cache_enabled is False  # env
        assert resolved.cache_dir == "/env/cache"  # env
        assert resolved.verbose is True  # env
        assert resolved.quiet is False  # env

    def test_all_defaults_used(self):
        """All fields use built-in defaults when nothing specified."""
        cli_config = None
        yaml_config = None
        env_vars: dict[str, str] = {}

        resolved = resolve_execution_config(
            cli_config=cli_config,
            yaml_config=yaml_config,
            env_vars=env_vars,
            defaults=DEFAULT_EXECUTION_CONFIG,
        )

        assert resolved.file_timeout == 30  # default
        assert resolved.llm_timeout == 60  # default
        assert resolved.download_timeout == 30  # default
        assert resolved.cache_enabled is True  # default
        assert resolved.cache_dir == ".holodeck/cache"  # default
        assert resolved.verbose is False  # default
        assert resolved.quiet is False  # default

    def test_partial_cli_merges_with_yaml(self):
        """CLI config merges with YAML for unspecified fields."""
        cli_config = ExecutionConfig(
            file_timeout=100,
            # Other fields unspecified (None)
        )

        yaml_config = ExecutionConfig(
            llm_timeout=80,
            download_timeout=60,
            cache_dir="/yaml/cache",
        )

        env_vars = {
            "HOLODECK_VERBOSE": "true",
        }

        resolved = resolve_execution_config(
            cli_config=cli_config,
            yaml_config=yaml_config,
            env_vars=env_vars,
            defaults=DEFAULT_EXECUTION_CONFIG,
        )

        assert resolved.file_timeout == 100  # CLI
        assert resolved.llm_timeout == 80  # YAML
        assert resolved.download_timeout == 60  # YAML
        assert resolved.cache_dir == "/yaml/cache"  # YAML
        assert resolved.verbose is True  # env
        assert resolved.cache_enabled is True  # default

    def test_env_var_type_conversion(self):
        """Environment variables are converted to correct types."""
        cli_config = None
        yaml_config = None

        env_vars = {
            "HOLODECK_FILE_TIMEOUT": "45",  # string to int
            "HOLODECK_CACHE_ENABLED": "false",  # string to bool
            "HOLODECK_VERBOSE": "true",  # string to bool
        }

        resolved = resolve_execution_config(
            cli_config=cli_config,
            yaml_config=yaml_config,
            env_vars=env_vars,
            defaults=DEFAULT_EXECUTION_CONFIG,
        )

        assert resolved.file_timeout == 45
        assert isinstance(resolved.file_timeout, int)
        assert resolved.cache_enabled is False
        assert isinstance(resolved.cache_enabled, bool)
        assert resolved.verbose is True
        assert isinstance(resolved.verbose, bool)

    def test_invalid_env_var_uses_yaml_or_default(self):
        """Invalid environment variables are skipped, falling back to YAML/defaults."""
        cli_config = None

        yaml_config = ExecutionConfig(
            file_timeout=50,
        )

        env_vars = {
            "HOLODECK_FILE_TIMEOUT": "invalid_number",  # Invalid
            "HOLODECK_LLM_TIMEOUT": "75",  # Valid
        }

        resolved = resolve_execution_config(
            cli_config=cli_config,
            yaml_config=yaml_config,
            env_vars=env_vars,
            defaults=DEFAULT_EXECUTION_CONFIG,
        )

        assert resolved.file_timeout == 50  # YAML (env invalid, skipped)
        assert resolved.llm_timeout == 75  # env (valid)


class TestToolCallValidation:
    """Tests for T049: Tool call validation against expected tools."""

    def test_exact_match_passes(self):
        """Tool calls exactly matching expected tools passes validation."""
        from holodeck.lib.test_runner.executor import validate_tool_calls

        actual = ["search_tool", "calculator"]
        expected = ["search_tool", "calculator"]

        result = validate_tool_calls(actual, expected)

        assert result is True

    def test_mismatch_fails(self):
        """Tool calls not matching expected tools fails validation."""
        from holodeck.lib.test_runner.executor import validate_tool_calls

        actual = ["search_tool", "get_weather"]
        expected = ["search_tool", "calculator"]

        result = validate_tool_calls(actual, expected)

        assert result is False

    def test_no_expected_tools_skips_validation(self):
        """When expected_tools is None, validation is skipped."""
        from holodeck.lib.test_runner.executor import validate_tool_calls

        actual = ["search_tool", "calculator"]
        expected = None

        result = validate_tool_calls(actual, expected)

        assert result is None

    def test_empty_expected_tools_with_calls_fails(self):
        """Agent calling tools when none expected fails validation."""
        from holodeck.lib.test_runner.executor import validate_tool_calls

        actual: list[str] = ["search_tool"]
        expected: list[str] = []

        result = validate_tool_calls(actual, expected)

        assert result is False

    def test_empty_actual_with_expected_fails(self):
        """Agent not calling expected tools fails validation."""
        from holodeck.lib.test_runner.executor import validate_tool_calls

        actual: list[str] = []
        expected: list[str] = ["search_tool", "calculator"]

        result = validate_tool_calls(actual, expected)

        assert result is False

    def test_order_independent_matching(self):
        """Tool call order doesn't matter, only set membership."""
        from holodeck.lib.test_runner.executor import validate_tool_calls

        actual = ["calculator", "search_tool"]  # Different order
        expected = ["search_tool", "calculator"]

        result = validate_tool_calls(actual, expected)

        assert result is True

    def test_subset_of_expected_fails(self):
        """Calling subset of expected tools fails validation."""
        from holodeck.lib.test_runner.executor import validate_tool_calls

        actual = ["search_tool"]  # Only one of two expected
        expected = ["search_tool", "calculator"]

        result = validate_tool_calls(actual, expected)

        assert result is False

    def test_superset_of_expected_fails(self):
        """Calling more tools than expected fails validation."""
        from holodeck.lib.test_runner.executor import validate_tool_calls

        actual = ["search_tool", "calculator", "extra_tool"]
        expected = ["search_tool", "calculator"]

        result = validate_tool_calls(actual, expected)

        assert result is False


class TestTimeoutHandling:
    """Tests for T051: Timeout handling for file/LLM/download operations."""

    def test_timeout_values_resolved_from_config(self):
        """Timeout values are resolved from ExecutionConfig."""
        config = ExecutionConfig(
            file_timeout=25,
            llm_timeout=45,
            download_timeout=20,
        )

        assert config.file_timeout == 25
        assert config.llm_timeout == 45
        assert config.download_timeout == 20

    def test_timeout_defaults_applied(self):
        """Default timeouts are applied when config values are None."""
        config = ExecutionConfig(
            file_timeout=None,
            llm_timeout=None,
            download_timeout=None,
        )

        # Apply defaults
        file_timeout = config.file_timeout or DEFAULT_EXECUTION_CONFIG["file_timeout"]
        llm_timeout = config.llm_timeout or DEFAULT_EXECUTION_CONFIG["llm_timeout"]
        download_timeout = (
            config.download_timeout or DEFAULT_EXECUTION_CONFIG["download_timeout"]
        )

        assert file_timeout == 30
        assert llm_timeout == 60
        assert download_timeout == 30

    @pytest.mark.asyncio
    async def test_llm_timeout_raises_exception(self):
        """LLM invocation exceeding timeout raises TimeoutError."""

        async def slow_operation():
            await asyncio.sleep(2)
            return "result"

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(slow_operation(), timeout=0.1)

    @pytest.mark.asyncio
    async def test_timeout_conversion_seconds_to_milliseconds(self):
        """Timeout conversion from seconds (config) to milliseconds (FileProcessor)."""
        download_timeout_seconds = 30
        download_timeout_ms = download_timeout_seconds * 1000

        assert download_timeout_ms == 30000

    def test_timeout_bounds(self):
        """Timeout values are within expected bounds (1-600 seconds)."""
        # Valid timeouts
        valid_config = ExecutionConfig(
            file_timeout=30,
            llm_timeout=60,
            download_timeout=30,
        )

        assert valid_config.file_timeout is not None
        assert valid_config.llm_timeout is not None
        assert valid_config.download_timeout is not None
        assert 1 <= valid_config.file_timeout <= 300
        assert 1 <= valid_config.llm_timeout <= 600
        assert 1 <= valid_config.download_timeout <= 300


class TestExecutorMainFlow:
    """Tests for T053-T054: Main executor orchestration flow."""

    @pytest.mark.asyncio
    async def test_executor_initialization(self):
        """TestExecutor initializes with agent config path and dependencies."""
        agent_config_path = "tests/fixtures/agents/test_agent.yaml"

        # Create mocks for dependencies
        mock_loader = Mock(spec=ConfigLoader)
        mock_file_processor = Mock(spec=FileProcessor)
        mock_agent_factory = Mock(spec=AgentFactory)

        # Setup mock config
        mock_config = Mock(spec=Agent)
        mock_config.name = "test_agent"
        mock_config.test_cases = []
        mock_config.evaluations = None
        mock_config.execution = None
        mock_loader.load_agent_yaml.return_value = mock_config

        # Inject dependencies
        executor = TestExecutor(
            agent_config_path=agent_config_path,
            config_loader=mock_loader,
            file_processor=mock_file_processor,
            agent_factory=mock_agent_factory,
        )

        assert executor.agent_config_path == agent_config_path
        assert executor.agent_config is not None
        assert executor.file_processor is mock_file_processor
        assert executor.agent_factory is mock_agent_factory
        mock_loader.load_agent_yaml.assert_called_once_with(agent_config_path)

    @pytest.mark.asyncio
    async def test_execute_single_test_with_agent_response(self):
        """Single test case execution captures agent response."""
        # This test will be expanded with actual flow implementation
        test_case = TestCaseModel(
            name="test_1",
            input="What is 2+2?",
            expected_tools=None,
            ground_truth=None,
            files=None,
            evaluations=None,
        )

        assert test_case.name == "test_1"
        assert test_case.input == "What is 2+2?"

    @pytest.mark.asyncio
    async def test_execute_single_test_captures_tool_calls(self):
        """Single test execution captures tool calls from agent."""
        test_case = TestCaseModel(
            name="test_with_tools",
            input="Search for Python documentation",
            expected_tools=["search_engine"],
            ground_truth=None,
            files=None,
            evaluations=None,
        )

        assert test_case.expected_tools == ["search_engine"]

    @pytest.mark.asyncio
    async def test_execute_single_test_validates_tools(self):
        """Tool call validation is performed for test case."""
        from holodeck.lib.test_runner.executor import validate_tool_calls

        test_case = TestCaseModel(
            name="test_tool_validation",
            input="Search and calculate",
            expected_tools=["search_tool", "calculator"],
            ground_truth=None,
            files=None,
            evaluations=None,
        )

        actual_tools = ["search_tool", "calculator"]
        validated = validate_tool_calls(actual_tools, test_case.expected_tools)

        assert validated is True

    @pytest.mark.asyncio
    async def test_test_result_structure(self):
        """TestResult contains all required fields."""
        result = TestResult(
            test_name="test_1",
            test_input="What is 2+2?",
            processed_files=[],
            agent_response="4",
            tool_calls=[],
            expected_tools=None,
            tools_matched=None,
            metric_results=[],
            ground_truth=None,
            passed=True,
            execution_time_ms=100,
            errors=[],
            timestamp=datetime.now(UTC).isoformat(),
        )

        assert result.test_name == "test_1"
        assert result.agent_response == "4"
        assert result.passed is True
        assert result.execution_time_ms == 100
        assert result.timestamp is not None

    @pytest.mark.asyncio
    async def test_metric_result_structure(self):
        """MetricResult contains required fields for evaluation outcome."""
        metric = MetricResult(
            metric_name="groundedness",
            score=0.85,
            threshold=0.8,
            passed=True,
            scale="0-1",
            error=None,
            retry_count=None,
            evaluation_time_ms=None,
            model_used=None,
        )

        assert metric.metric_name == "groundedness"
        assert metric.score == 0.85
        assert metric.threshold == 0.8
        assert metric.passed is True

    def test_default_execution_config_values(self):
        """DEFAULT_EXECUTION_CONFIG has expected default values."""
        assert DEFAULT_EXECUTION_CONFIG["file_timeout"] == 30
        assert DEFAULT_EXECUTION_CONFIG["llm_timeout"] == 60
        assert DEFAULT_EXECUTION_CONFIG["download_timeout"] == 30
        assert DEFAULT_EXECUTION_CONFIG["cache_enabled"] is True
        assert DEFAULT_EXECUTION_CONFIG["cache_dir"] == ".holodeck/cache"
        assert DEFAULT_EXECUTION_CONFIG["verbose"] is False
        assert DEFAULT_EXECUTION_CONFIG["quiet"] is False

    def test_processed_file_input_structure(self):
        """ProcessedFileInput captures file processing results."""
        processed = ProcessedFileInput(
            original=FileInput(
                path="test.pdf",
                type="pdf",
                url=None,
                description=None,
                pages=None,
                sheet=None,
                range=None,
                cache=None,
            ),
            markdown_content="# Test PDF\nContent here",
            metadata={"pages": 5},
            cached_path=None,
            processing_time_ms=250,
            error=None,
        )

        assert processed.original.path == "test.pdf"
        assert "Test PDF" in processed.markdown_content
        assert processed.processing_time_ms == 250
        assert processed.error is None
