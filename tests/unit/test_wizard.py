"""Unit tests for the interactive wizard module.

Tests the wizard functions with mocked InquirerPy prompts.
"""

from unittest.mock import MagicMock, patch

import pytest

from holodeck.cli.utils.wizard import (
    WizardCancelledError,
    _prompt_agent_name,
    _prompt_evals,
    _prompt_llm_provider,
    _prompt_mcp_servers,
    _prompt_template,
    _prompt_vectorstore,
    is_interactive,
    run_wizard,
)
from holodeck.models.wizard_config import (
    EVAL_CHOICES,
    LLM_PROVIDER_CHOICES,
    MCP_SERVER_CHOICES,
    VECTOR_STORE_CHOICES,
    WizardResult,
    get_template_choices,
)


class TestIsInteractive:
    """Tests for the is_interactive() function."""

    def test_is_interactive_when_both_tty(self) -> None:
        """Test returns True when stdin and stdout are TTY."""
        with (
            patch("sys.stdin") as mock_stdin,
            patch("sys.stdout") as mock_stdout,
        ):
            mock_stdin.isatty.return_value = True
            mock_stdout.isatty.return_value = True

            result = is_interactive()

            assert result is True

    def test_is_interactive_when_stdin_not_tty(self) -> None:
        """Test returns False when stdin is not TTY."""
        with (
            patch("sys.stdin") as mock_stdin,
            patch("sys.stdout") as mock_stdout,
        ):
            mock_stdin.isatty.return_value = False
            mock_stdout.isatty.return_value = True

            result = is_interactive()

            assert result is False

    def test_is_interactive_when_stdout_not_tty(self) -> None:
        """Test returns False when stdout is not TTY."""
        with (
            patch("sys.stdin") as mock_stdin,
            patch("sys.stdout") as mock_stdout,
        ):
            mock_stdin.isatty.return_value = True
            mock_stdout.isatty.return_value = False

            result = is_interactive()

            assert result is False

    def test_is_interactive_when_neither_tty(self) -> None:
        """Test returns False when neither stdin nor stdout are TTY."""
        with (
            patch("sys.stdin") as mock_stdin,
            patch("sys.stdout") as mock_stdout,
        ):
            mock_stdin.isatty.return_value = False
            mock_stdout.isatty.return_value = False

            result = is_interactive()

            assert result is False


class TestPromptAgentName:
    """Tests for the _prompt_agent_name() function."""

    @patch("holodeck.cli.utils.wizard.inquirer")
    def test_prompt_agent_name_returns_value(self, mock_inquirer: MagicMock) -> None:
        """Test returns the entered agent name."""
        mock_text = MagicMock()
        mock_text.execute.return_value = "my-agent"
        mock_inquirer.text.return_value = mock_text

        result = _prompt_agent_name()

        assert result == "my-agent"
        mock_inquirer.text.assert_called_once()

    @patch("holodeck.cli.utils.wizard.inquirer")
    def test_prompt_agent_name_with_default(self, mock_inquirer: MagicMock) -> None:
        """Test uses default value when provided."""
        mock_text = MagicMock()
        mock_text.execute.return_value = "test-agent"
        mock_inquirer.text.return_value = mock_text

        result = _prompt_agent_name(default="test-agent")

        assert result == "test-agent"
        call_kwargs = mock_inquirer.text.call_args[1]
        assert call_kwargs["default"] == "test-agent"

    @patch("holodeck.cli.utils.wizard.inquirer")
    def test_prompt_agent_name_keyboard_interrupt(
        self, mock_inquirer: MagicMock
    ) -> None:
        """Test raises KeyboardInterrupt when user cancels."""
        mock_text = MagicMock()
        mock_text.execute.side_effect = KeyboardInterrupt()
        mock_inquirer.text.return_value = mock_text

        with pytest.raises(KeyboardInterrupt):
            _prompt_agent_name()


class TestPromptTemplate:
    """Tests for the _prompt_template() function."""

    @patch("holodeck.cli.utils.wizard.inquirer")
    def test_prompt_template_returns_value(self, mock_inquirer: MagicMock) -> None:
        """Test returns the selected template."""
        mock_select = MagicMock()
        mock_select.execute.return_value = "research"
        mock_inquirer.select.return_value = mock_select

        result = _prompt_template()

        assert result == "research"
        mock_inquirer.select.assert_called_once()

    @patch("holodeck.cli.utils.wizard.inquirer")
    def test_prompt_template_with_default(self, mock_inquirer: MagicMock) -> None:
        """Test uses default value when provided."""
        mock_select = MagicMock()
        mock_select.execute.return_value = "customer-support"
        mock_inquirer.select.return_value = mock_select

        result = _prompt_template(default="customer-support")

        assert result == "customer-support"
        call_kwargs = mock_inquirer.select.call_args[1]
        assert call_kwargs["default"] == "customer-support"

    @patch("holodeck.cli.utils.wizard.inquirer")
    def test_prompt_template_has_all_choices(self, mock_inquirer: MagicMock) -> None:
        """Test includes all template choices."""
        mock_select = MagicMock()
        mock_select.execute.return_value = "conversational"
        mock_inquirer.select.return_value = mock_select

        _prompt_template()

        call_kwargs = mock_inquirer.select.call_args[1]
        choices = call_kwargs["choices"]
        template_choices = get_template_choices()
        assert len(choices) == len(template_choices)

    @patch("holodeck.cli.utils.wizard.inquirer")
    def test_prompt_template_keyboard_interrupt(self, mock_inquirer: MagicMock) -> None:
        """Test raises KeyboardInterrupt when user cancels."""
        mock_select = MagicMock()
        mock_select.execute.side_effect = KeyboardInterrupt()
        mock_inquirer.select.return_value = mock_select

        with pytest.raises(KeyboardInterrupt):
            _prompt_template()


class TestPromptLLMProvider:
    """Tests for the _prompt_llm_provider() function."""

    @patch("holodeck.cli.utils.wizard.inquirer")
    def test_prompt_llm_provider_returns_value(self, mock_inquirer: MagicMock) -> None:
        """Test returns the selected provider."""
        mock_select = MagicMock()
        mock_select.execute.return_value = "openai"
        mock_inquirer.select.return_value = mock_select

        result = _prompt_llm_provider()

        assert result == "openai"
        mock_inquirer.select.assert_called_once()

    @patch("holodeck.cli.utils.wizard.inquirer")
    def test_prompt_llm_provider_with_default(self, mock_inquirer: MagicMock) -> None:
        """Test uses default value when provided."""
        mock_select = MagicMock()
        mock_select.execute.return_value = "anthropic"
        mock_inquirer.select.return_value = mock_select

        result = _prompt_llm_provider(default="anthropic")

        assert result == "anthropic"
        call_kwargs = mock_inquirer.select.call_args[1]
        assert call_kwargs["default"] == "anthropic"

    @patch("holodeck.cli.utils.wizard.inquirer")
    def test_prompt_llm_provider_has_all_choices(
        self, mock_inquirer: MagicMock
    ) -> None:
        """Test includes all LLM provider choices."""
        mock_select = MagicMock()
        mock_select.execute.return_value = "ollama"
        mock_inquirer.select.return_value = mock_select

        _prompt_llm_provider()

        call_kwargs = mock_inquirer.select.call_args[1]
        choices = call_kwargs["choices"]
        assert len(choices) == len(LLM_PROVIDER_CHOICES)


class TestPromptVectorstore:
    """Tests for the _prompt_vectorstore() function."""

    @patch("holodeck.cli.utils.wizard.inquirer")
    def test_prompt_vectorstore_returns_value(self, mock_inquirer: MagicMock) -> None:
        """Test returns the selected vector store."""
        mock_select = MagicMock()
        mock_select.execute.return_value = "qdrant"
        mock_inquirer.select.return_value = mock_select

        result = _prompt_vectorstore()

        assert result == "qdrant"
        mock_inquirer.select.assert_called_once()

    @patch("click.secho")
    @patch("holodeck.cli.utils.wizard.inquirer")
    def test_prompt_vectorstore_in_memory_warning(
        self, mock_inquirer: MagicMock, mock_secho: MagicMock
    ) -> None:
        """Test displays warning for in-memory selection."""
        mock_select = MagicMock()
        mock_select.execute.return_value = "in-memory"
        mock_inquirer.select.return_value = mock_select

        result = _prompt_vectorstore()

        assert result == "in-memory"
        mock_secho.assert_called_once()
        warning_call = mock_secho.call_args
        assert "ephemeral" in warning_call[0][0].lower()
        assert warning_call[1]["fg"] == "yellow"

    @patch("holodeck.cli.utils.wizard.inquirer")
    def test_prompt_vectorstore_has_all_choices(self, mock_inquirer: MagicMock) -> None:
        """Test includes all vector store choices."""
        mock_select = MagicMock()
        mock_select.execute.return_value = "chromadb"
        mock_inquirer.select.return_value = mock_select

        _prompt_vectorstore()

        call_kwargs = mock_inquirer.select.call_args[1]
        choices = call_kwargs["choices"]
        assert len(choices) == len(VECTOR_STORE_CHOICES)


class TestPromptEvals:
    """Tests for the _prompt_evals() function."""

    @patch("holodeck.cli.utils.wizard.inquirer")
    def test_prompt_evals_returns_list(self, mock_inquirer: MagicMock) -> None:
        """Test returns list of selected evals."""
        mock_checkbox = MagicMock()
        mock_checkbox.execute.return_value = [
            "rag-faithfulness",
            "rag-answer_relevancy",
        ]
        mock_inquirer.checkbox.return_value = mock_checkbox

        result = _prompt_evals()

        assert result == ["rag-faithfulness", "rag-answer_relevancy"]
        mock_inquirer.checkbox.assert_called_once()

    @patch("holodeck.cli.utils.wizard.inquirer")
    def test_prompt_evals_empty_selection(self, mock_inquirer: MagicMock) -> None:
        """Test allows empty selection."""
        mock_checkbox = MagicMock()
        mock_checkbox.execute.return_value = []
        mock_inquirer.checkbox.return_value = mock_checkbox

        result = _prompt_evals()

        assert result == []

    @patch("holodeck.cli.utils.wizard.inquirer")
    def test_prompt_evals_has_all_choices(self, mock_inquirer: MagicMock) -> None:
        """Test includes all eval choices."""
        mock_checkbox = MagicMock()
        mock_checkbox.execute.return_value = []
        mock_inquirer.checkbox.return_value = mock_checkbox

        _prompt_evals()

        call_kwargs = mock_inquirer.checkbox.call_args[1]
        choices = call_kwargs["choices"]
        assert len(choices) == len(EVAL_CHOICES)


class TestPromptMCPServers:
    """Tests for the _prompt_mcp_servers() function."""

    @patch("holodeck.cli.utils.wizard.inquirer")
    def test_prompt_mcp_servers_returns_list(self, mock_inquirer: MagicMock) -> None:
        """Test returns list of selected MCP servers."""
        mock_checkbox = MagicMock()
        mock_checkbox.execute.return_value = ["brave-search", "memory"]
        mock_inquirer.checkbox.return_value = mock_checkbox

        result = _prompt_mcp_servers()

        assert result == ["brave-search", "memory"]
        mock_inquirer.checkbox.assert_called_once()

    @patch("holodeck.cli.utils.wizard.inquirer")
    def test_prompt_mcp_servers_empty_selection(self, mock_inquirer: MagicMock) -> None:
        """Test allows empty selection."""
        mock_checkbox = MagicMock()
        mock_checkbox.execute.return_value = []
        mock_inquirer.checkbox.return_value = mock_checkbox

        result = _prompt_mcp_servers()

        assert result == []

    @patch("holodeck.cli.utils.wizard.inquirer")
    def test_prompt_mcp_servers_has_all_choices(self, mock_inquirer: MagicMock) -> None:
        """Test includes all MCP server choices."""
        mock_checkbox = MagicMock()
        mock_checkbox.execute.return_value = []
        mock_inquirer.checkbox.return_value = mock_checkbox

        _prompt_mcp_servers()

        call_kwargs = mock_inquirer.checkbox.call_args[1]
        choices = call_kwargs["choices"]
        assert len(choices) == len(MCP_SERVER_CHOICES)


class TestRunWizard:
    """Tests for the run_wizard() function."""

    @patch("holodeck.cli.utils.wizard._prompt_mcp_servers")
    @patch("holodeck.cli.utils.wizard._prompt_evals")
    @patch("holodeck.cli.utils.wizard._prompt_vectorstore")
    @patch("holodeck.cli.utils.wizard._prompt_llm_provider")
    @patch("holodeck.cli.utils.wizard._prompt_template")
    @patch("holodeck.cli.utils.wizard._prompt_agent_name")
    def test_run_wizard_returns_result(
        self,
        mock_agent_name: MagicMock,
        mock_template: MagicMock,
        mock_llm: MagicMock,
        mock_vectorstore: MagicMock,
        mock_evals: MagicMock,
        mock_mcp: MagicMock,
    ) -> None:
        """Test run_wizard returns WizardResult with all values."""
        mock_agent_name.return_value = "test-agent"
        mock_template.return_value = "research"
        mock_llm.return_value = "openai"
        mock_vectorstore.return_value = "qdrant"
        mock_evals.return_value = ["rag-faithfulness"]
        mock_mcp.return_value = ["brave-search", "memory"]

        result = run_wizard()

        assert isinstance(result, WizardResult)
        assert result.agent_name == "test-agent"
        assert result.template == "research"
        assert result.llm_provider == "openai"
        assert result.vector_store == "qdrant"
        assert result.evals == ["rag-faithfulness"]
        assert result.mcp_servers == ["brave-search", "memory"]

    @patch("holodeck.cli.utils.wizard._prompt_mcp_servers")
    @patch("holodeck.cli.utils.wizard._prompt_evals")
    @patch("holodeck.cli.utils.wizard._prompt_vectorstore")
    @patch("holodeck.cli.utils.wizard._prompt_llm_provider")
    @patch("holodeck.cli.utils.wizard._prompt_template")
    @patch("holodeck.cli.utils.wizard._prompt_agent_name")
    def test_run_wizard_skips_agent_name(
        self,
        mock_agent_name: MagicMock,
        mock_template: MagicMock,
        mock_llm: MagicMock,
        mock_vectorstore: MagicMock,
        mock_evals: MagicMock,
        mock_mcp: MagicMock,
    ) -> None:
        """Test run_wizard skips agent name when skip_agent_name=True."""
        mock_template.return_value = "conversational"
        mock_llm.return_value = "ollama"
        mock_vectorstore.return_value = "chromadb"
        mock_evals.return_value = []
        mock_mcp.return_value = []

        result = run_wizard(skip_agent_name=True, agent_name_default="preset-agent")

        mock_agent_name.assert_not_called()
        assert result.agent_name == "preset-agent"

    @patch("holodeck.cli.utils.wizard._prompt_mcp_servers")
    @patch("holodeck.cli.utils.wizard._prompt_evals")
    @patch("holodeck.cli.utils.wizard._prompt_vectorstore")
    @patch("holodeck.cli.utils.wizard._prompt_llm_provider")
    @patch("holodeck.cli.utils.wizard._prompt_template")
    @patch("holodeck.cli.utils.wizard._prompt_agent_name")
    def test_run_wizard_skips_template(
        self,
        mock_agent_name: MagicMock,
        mock_template: MagicMock,
        mock_llm: MagicMock,
        mock_vectorstore: MagicMock,
        mock_evals: MagicMock,
        mock_mcp: MagicMock,
    ) -> None:
        """Test run_wizard skips template when skip_template=True."""
        mock_agent_name.return_value = "test-agent"
        mock_llm.return_value = "ollama"
        mock_vectorstore.return_value = "chromadb"
        mock_evals.return_value = []
        mock_mcp.return_value = []

        result = run_wizard(skip_template=True, template_default="research")

        mock_template.assert_not_called()
        assert result.template == "research"

    @patch("holodeck.cli.utils.wizard._prompt_mcp_servers")
    @patch("holodeck.cli.utils.wizard._prompt_evals")
    @patch("holodeck.cli.utils.wizard._prompt_vectorstore")
    @patch("holodeck.cli.utils.wizard._prompt_llm_provider")
    @patch("holodeck.cli.utils.wizard._prompt_template")
    @patch("holodeck.cli.utils.wizard._prompt_agent_name")
    def test_run_wizard_skips_llm(
        self,
        mock_agent_name: MagicMock,
        mock_template: MagicMock,
        mock_llm: MagicMock,
        mock_vectorstore: MagicMock,
        mock_evals: MagicMock,
        mock_mcp: MagicMock,
    ) -> None:
        """Test run_wizard skips LLM when skip_llm=True."""
        mock_agent_name.return_value = "test-agent"
        mock_template.return_value = "conversational"
        mock_vectorstore.return_value = "chromadb"
        mock_evals.return_value = []
        mock_mcp.return_value = []

        result = run_wizard(skip_llm=True, llm_default="anthropic")

        mock_llm.assert_not_called()
        assert result.llm_provider == "anthropic"

    @patch("holodeck.cli.utils.wizard._prompt_mcp_servers")
    @patch("holodeck.cli.utils.wizard._prompt_evals")
    @patch("holodeck.cli.utils.wizard._prompt_vectorstore")
    @patch("holodeck.cli.utils.wizard._prompt_llm_provider")
    @patch("holodeck.cli.utils.wizard._prompt_template")
    @patch("holodeck.cli.utils.wizard._prompt_agent_name")
    def test_run_wizard_skips_all(
        self,
        mock_agent_name: MagicMock,
        mock_template: MagicMock,
        mock_llm: MagicMock,
        mock_vectorstore: MagicMock,
        mock_evals: MagicMock,
        mock_mcp: MagicMock,
    ) -> None:
        """Test run_wizard skips all prompts when all skip flags are True."""
        result = run_wizard(
            skip_agent_name=True,
            skip_template=True,
            skip_llm=True,
            skip_vectorstore=True,
            skip_evals=True,
            skip_mcp=True,
            agent_name_default="skip-agent",
            template_default="customer-support",
            llm_default="openai",
            vectorstore_default="qdrant",
            evals_defaults=["rag-faithfulness"],
            mcp_defaults=["brave-search"],
        )

        mock_agent_name.assert_not_called()
        mock_template.assert_not_called()
        mock_llm.assert_not_called()
        mock_vectorstore.assert_not_called()
        mock_evals.assert_not_called()
        mock_mcp.assert_not_called()

        assert result.agent_name == "skip-agent"
        assert result.template == "customer-support"
        assert result.llm_provider == "openai"
        assert result.vector_store == "qdrant"
        assert result.evals == ["rag-faithfulness"]
        assert result.mcp_servers == ["brave-search"]

    @patch("holodeck.cli.utils.wizard._prompt_agent_name")
    def test_run_wizard_keyboard_interrupt(self, mock_agent_name: MagicMock) -> None:
        """Test run_wizard raises WizardCancelledError on KeyboardInterrupt."""
        mock_agent_name.side_effect = KeyboardInterrupt()

        with pytest.raises(WizardCancelledError) as exc_info:
            run_wizard()

        assert "cancelled" in str(exc_info.value).lower()


class TestWizardCancelledError:
    """Tests for the WizardCancelledError exception."""

    def test_is_exception(self) -> None:
        """Test WizardCancelledError is an Exception."""
        error = WizardCancelledError("Test message")

        assert isinstance(error, Exception)

    def test_message(self) -> None:
        """Test WizardCancelledError has correct message."""
        error = WizardCancelledError("Wizard cancelled by user")

        assert str(error) == "Wizard cancelled by user"


class TestWizardResultValidation:
    """Tests for WizardResult model validation."""

    def test_valid_result(self) -> None:
        """Test creating valid WizardResult."""
        result = WizardResult(
            agent_name="my-agent",
            llm_provider="ollama",
            vector_store="chromadb",
            evals=["rag-faithfulness"],
            mcp_servers=["brave-search"],
        )

        assert result.agent_name == "my-agent"
        assert result.llm_provider == "ollama"
        assert result.vector_store == "chromadb"
        assert result.evals == ["rag-faithfulness"]
        assert result.mcp_servers == ["brave-search"]

    def test_invalid_agent_name(self) -> None:
        """Test WizardResult rejects invalid agent name."""
        with pytest.raises(ValueError, match="Agent name"):
            WizardResult(
                agent_name="123invalid",  # Starts with digit
                llm_provider="ollama",
                vector_store="chromadb",
                evals=[],
                mcp_servers=[],
            )

    def test_invalid_llm_provider(self) -> None:
        """Test WizardResult rejects invalid LLM provider."""
        with pytest.raises(ValueError, match="Invalid LLM provider"):
            WizardResult(
                agent_name="my-agent",
                llm_provider="invalid-provider",
                vector_store="chromadb",
                evals=[],
                mcp_servers=[],
            )

    def test_invalid_vector_store(self) -> None:
        """Test WizardResult rejects invalid vector store."""
        with pytest.raises(ValueError, match="Invalid vector store"):
            WizardResult(
                agent_name="my-agent",
                llm_provider="ollama",
                vector_store="invalid-store",
                evals=[],
                mcp_servers=[],
            )

    def test_invalid_eval(self) -> None:
        """Test WizardResult rejects invalid eval."""
        with pytest.raises(ValueError, match="Invalid evaluation metric"):
            WizardResult(
                agent_name="my-agent",
                llm_provider="ollama",
                vector_store="chromadb",
                evals=["invalid-eval"],
                mcp_servers=[],
            )

    def test_invalid_mcp_server(self) -> None:
        """Test WizardResult rejects invalid MCP server."""
        with pytest.raises(ValueError, match="Invalid MCP server"):
            WizardResult(
                agent_name="my-agent",
                llm_provider="ollama",
                vector_store="chromadb",
                evals=[],
                mcp_servers=["invalid-server"],
            )

    def test_empty_evals_allowed(self) -> None:
        """Test WizardResult allows empty evals list."""
        result = WizardResult(
            agent_name="my-agent",
            llm_provider="ollama",
            vector_store="chromadb",
            evals=[],
            mcp_servers=["brave-search"],
        )

        assert result.evals == []

    def test_empty_mcp_servers_allowed(self) -> None:
        """Test WizardResult allows empty MCP servers list."""
        result = WizardResult(
            agent_name="my-agent",
            llm_provider="ollama",
            vector_store="chromadb",
            evals=["rag-faithfulness"],
            mcp_servers=[],
        )

        assert result.mcp_servers == []
