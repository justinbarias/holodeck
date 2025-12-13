"""Unit tests for wizard configuration models.

This module contains comprehensive tests for all wizard-related
Pydantic models and predefined choice lists.
"""

import pytest
from pydantic import ValidationError

from holodeck.models.wizard_config import (
    EVAL_CHOICES,
    LLM_PROVIDER_CHOICES,
    MCP_SERVER_CHOICES,
    VALID_EVALS,
    VALID_LLM_PROVIDERS,
    VALID_MCP_SERVERS,
    VALID_VECTOR_STORES,
    VECTOR_STORE_CHOICES,
    EvalChoice,
    LLMProviderChoice,
    MCPServerChoice,
    TemplateChoice,
    VectorStoreChoice,
    WizardResult,
    WizardState,
    WizardStep,
    get_default_evals,
    get_default_llm_provider,
    get_default_mcp_servers,
    get_default_vector_store,
    get_template_choices,
)


class TestWizardStep:
    """Tests for WizardStep enum."""

    def test_enum_values(self) -> None:
        """Test that enum has all expected values."""
        assert WizardStep.AGENT_NAME.value == "agent_name"
        assert WizardStep.TEMPLATE.value == "template"
        assert WizardStep.LLM_PROVIDER.value == "llm_provider"
        assert WizardStep.VECTOR_STORE.value == "vector_store"
        assert WizardStep.EVALS.value == "evals"
        assert WizardStep.MCP_SERVERS.value == "mcp_servers"
        assert WizardStep.COMPLETE.value == "complete"

    def test_enum_count(self) -> None:
        """Test that enum has exactly 7 steps."""
        assert len(WizardStep) == 7

    def test_string_representation(self) -> None:
        """Test that enum values can be used as strings."""
        assert str(WizardStep.AGENT_NAME) == "WizardStep.AGENT_NAME"
        assert WizardStep.AGENT_NAME == "agent_name"


class TestWizardState:
    """Tests for WizardState model."""

    def test_default_values(self) -> None:
        """Test default values when creating a new WizardState."""
        state = WizardState()
        assert state.current_step == WizardStep.AGENT_NAME
        assert state.agent_name is None
        assert state.llm_provider is None
        assert state.vector_store is None
        assert state.evals == []
        assert state.mcp_servers == []
        assert state.is_cancelled is False

    def test_field_assignments(self) -> None:
        """Test that fields can be assigned values."""
        state = WizardState(
            current_step=WizardStep.LLM_PROVIDER,
            agent_name="test-agent",
            llm_provider="openai",
            vector_store="redis",
            evals=["rag-faithfulness"],
            mcp_servers=["brave-search"],
            is_cancelled=False,
        )
        assert state.current_step == WizardStep.LLM_PROVIDER
        assert state.agent_name == "test-agent"
        assert state.llm_provider == "openai"
        assert state.vector_store == "redis"
        assert state.evals == ["rag-faithfulness"]
        assert state.mcp_servers == ["brave-search"]

    def test_state_transitions(self) -> None:
        """Test that state can be modified to simulate wizard flow."""
        state = WizardState()

        # Simulate wizard progression
        state.agent_name = "my-agent"
        state.current_step = WizardStep.LLM_PROVIDER
        assert state.agent_name == "my-agent"

        state.llm_provider = "ollama"
        state.current_step = WizardStep.VECTOR_STORE
        assert state.llm_provider == "ollama"

    def test_no_extra_fields(self) -> None:
        """Test that model rejects extra fields."""
        with pytest.raises(ValidationError):
            WizardState(extra_field="invalid")


class TestWizardResult:
    """Tests for WizardResult model."""

    def test_valid_creation(self) -> None:
        """Test creating a valid WizardResult."""
        result = WizardResult(
            agent_name="test-agent",
            llm_provider="ollama",
            vector_store="chromadb",
            evals=["rag-faithfulness"],
            mcp_servers=["brave-search"],
        )
        assert result.agent_name == "test-agent"
        assert result.llm_provider == "ollama"
        assert result.vector_store == "chromadb"
        assert result.evals == ["rag-faithfulness"]
        assert result.mcp_servers == ["brave-search"]

    def test_empty_evals_and_mcp_servers_allowed(self) -> None:
        """Test that empty evals and mcp_servers lists are valid."""
        result = WizardResult(
            agent_name="test-agent",
            llm_provider="ollama",
            vector_store="chromadb",
            evals=[],
            mcp_servers=[],
        )
        assert result.evals == []
        assert result.mcp_servers == []

    def test_required_fields(self) -> None:
        """Test that required fields raise ValidationError when missing."""
        with pytest.raises(ValidationError) as exc_info:
            WizardResult(
                llm_provider="ollama",
                vector_store="chromadb",
            )
        assert "agent_name" in str(exc_info.value).lower()

    @pytest.mark.parametrize(
        "agent_name,should_pass",
        [
            ("my-agent", True),
            ("test_agent", True),
            ("Agent123", True),
            ("a", True),
            ("agent-with-dashes", True),
            ("agent_with_underscores", True),
            ("", False),  # Empty
            ("123agent", False),  # Starts with digit
            ("agent@name", False),  # Invalid character
            ("agent name", False),  # Space not allowed
            ("agent.name", False),  # Dot not allowed
        ],
        ids=[
            "simple-hyphen",
            "simple-underscore",
            "with-numbers",
            "single-char",
            "multiple-hyphens",
            "multiple-underscores",
            "empty",
            "starts-digit",
            "at-sign",
            "space",
            "dot",
        ],
    )
    def test_agent_name_validation(self, agent_name: str, should_pass: bool) -> None:
        """Test agent name validation with various inputs."""
        if should_pass:
            result = WizardResult(
                agent_name=agent_name,
                llm_provider="ollama",
                vector_store="chromadb",
            )
            assert result.agent_name == agent_name
        else:
            with pytest.raises(ValidationError):
                WizardResult(
                    agent_name=agent_name,
                    llm_provider="ollama",
                    vector_store="chromadb",
                )

    def test_agent_name_too_long(self) -> None:
        """Test that agent name longer than 64 characters fails."""
        with pytest.raises(ValidationError) as exc_info:
            WizardResult(
                agent_name="a" * 65,
                llm_provider="ollama",
                vector_store="chromadb",
            )
        assert "64 characters" in str(exc_info.value)

    @pytest.mark.parametrize(
        "llm_provider",
        ["ollama", "openai", "azure_openai", "anthropic"],
    )
    def test_valid_llm_providers(self, llm_provider: str) -> None:
        """Test that all valid LLM providers are accepted."""
        result = WizardResult(
            agent_name="test-agent",
            llm_provider=llm_provider,
            vector_store="chromadb",
        )
        assert result.llm_provider == llm_provider

    def test_invalid_llm_provider(self) -> None:
        """Test that invalid LLM provider raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            WizardResult(
                agent_name="test-agent",
                llm_provider="invalid-provider",
                vector_store="chromadb",
            )
        assert "invalid llm provider" in str(exc_info.value).lower()

    @pytest.mark.parametrize(
        "vector_store",
        ["chromadb", "qdrant", "in-memory"],
    )
    def test_valid_vector_stores(self, vector_store: str) -> None:
        """Test that all valid vector stores are accepted."""
        result = WizardResult(
            agent_name="test-agent",
            llm_provider="ollama",
            vector_store=vector_store,
        )
        assert result.vector_store == vector_store

    def test_invalid_vector_store(self) -> None:
        """Test that invalid vector store raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            WizardResult(
                agent_name="test-agent",
                llm_provider="ollama",
                vector_store="invalid-store",
            )
        assert "invalid vector store" in str(exc_info.value).lower()

    def test_valid_evals(self) -> None:
        """Test that all valid evals are accepted."""
        all_evals = [
            "rag-faithfulness",
            "rag-answer_relevancy",
            "rag-context_precision",
            "rag-context_recall",
        ]
        result = WizardResult(
            agent_name="test-agent",
            llm_provider="ollama",
            vector_store="chromadb",
            evals=all_evals,
        )
        assert result.evals == all_evals

    def test_invalid_eval(self) -> None:
        """Test that invalid eval raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            WizardResult(
                agent_name="test-agent",
                llm_provider="ollama",
                vector_store="chromadb",
                evals=["invalid-eval"],
            )
        assert "invalid evaluation metric" in str(exc_info.value).lower()

    def test_valid_mcp_servers(self) -> None:
        """Test that all valid MCP servers are accepted."""
        all_servers = [
            "brave-search",
            "memory",
            "sequentialthinking",
            "filesystem",
            "github",
            "postgres",
        ]
        result = WizardResult(
            agent_name="test-agent",
            llm_provider="ollama",
            vector_store="chromadb",
            mcp_servers=all_servers,
        )
        assert result.mcp_servers == all_servers

    def test_invalid_mcp_server(self) -> None:
        """Test that invalid MCP server raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            WizardResult(
                agent_name="test-agent",
                llm_provider="ollama",
                vector_store="chromadb",
                mcp_servers=["invalid-server"],
            )
        assert "invalid mcp server" in str(exc_info.value).lower()

    def test_model_is_frozen(self) -> None:
        """Test that WizardResult is immutable."""
        result = WizardResult(
            agent_name="test-agent",
            llm_provider="ollama",
            vector_store="chromadb",
        )
        with pytest.raises(ValidationError):
            result.agent_name = "new-name"


class TestLLMProviderChoice:
    """Tests for LLMProviderChoice model and LLM_PROVIDER_CHOICES list."""

    def test_model_creation(self) -> None:
        """Test creating a valid LLMProviderChoice."""
        choice = LLMProviderChoice(
            value="test",
            display_name="Test Provider",
            description="A test provider",
            is_default=False,
            default_model="test-model",
            requires_api_key=True,
            api_key_env_var="TEST_API_KEY",
            requires_endpoint=False,
        )
        assert choice.value == "test"
        assert choice.display_name == "Test Provider"
        assert choice.requires_api_key is True

    def test_choices_list_has_four_items(self) -> None:
        """Test that LLM_PROVIDER_CHOICES has exactly 4 items."""
        assert len(LLM_PROVIDER_CHOICES) == 4

    def test_only_one_default(self) -> None:
        """Test that only one choice has is_default=True."""
        defaults = [c for c in LLM_PROVIDER_CHOICES if c.is_default]
        assert len(defaults) == 1
        assert defaults[0].value == "ollama"

    def test_all_choices_are_valid(self) -> None:
        """Test that all choices have valid values."""
        for choice in LLM_PROVIDER_CHOICES:
            assert choice.value in VALID_LLM_PROVIDERS
            assert choice.display_name
            assert choice.description
            assert choice.default_model

    def test_api_key_requirements(self) -> None:
        """Test API key requirements for each provider."""
        for choice in LLM_PROVIDER_CHOICES:
            if choice.value == "ollama":
                assert choice.requires_api_key is False
            else:
                assert choice.requires_api_key is True
                assert choice.api_key_env_var is not None


class TestVectorStoreChoice:
    """Tests for VectorStoreChoice model and VECTOR_STORE_CHOICES list."""

    def test_model_creation(self) -> None:
        """Test creating a valid VectorStoreChoice."""
        choice = VectorStoreChoice(
            value="test",
            display_name="Test Store",
            description="A test store",
            is_default=False,
            default_endpoint="http://localhost:1234",
            persistence="local",
            connection_required=True,
        )
        assert choice.value == "test"
        assert choice.persistence == "local"

    def test_choices_list_has_three_items(self) -> None:
        """Test that VECTOR_STORE_CHOICES has exactly 3 items."""
        assert len(VECTOR_STORE_CHOICES) == 3

    def test_only_one_default(self) -> None:
        """Test that only one choice has is_default=True."""
        defaults = [c for c in VECTOR_STORE_CHOICES if c.is_default]
        assert len(defaults) == 1
        assert defaults[0].value == "chromadb"

    def test_all_choices_are_valid(self) -> None:
        """Test that all choices have valid values."""
        for choice in VECTOR_STORE_CHOICES:
            assert choice.value in VALID_VECTOR_STORES
            assert choice.display_name
            assert choice.description
            assert choice.persistence in ["local", "remote", "none"]


class TestEvalChoice:
    """Tests for EvalChoice model and EVAL_CHOICES list."""

    def test_model_creation(self) -> None:
        """Test creating a valid EvalChoice."""
        choice = EvalChoice(
            value="test-eval",
            display_name="Test Eval",
            description="A test evaluation metric",
            is_default=False,
            metric_type="ai",
        )
        assert choice.value == "test-eval"
        assert choice.metric_type == "ai"

    def test_choices_list_has_four_items(self) -> None:
        """Test that EVAL_CHOICES has exactly 4 items."""
        assert len(EVAL_CHOICES) == 4

    def test_two_defaults(self) -> None:
        """Test that exactly two choices have is_default=True."""
        defaults = [c for c in EVAL_CHOICES if c.is_default]
        assert len(defaults) == 2
        default_values = {c.value for c in defaults}
        assert default_values == {"rag-faithfulness", "rag-answer_relevancy"}

    def test_all_choices_are_valid(self) -> None:
        """Test that all choices have valid values."""
        for choice in EVAL_CHOICES:
            assert choice.value in VALID_EVALS
            assert choice.display_name
            assert choice.description
            assert choice.metric_type in ["ai", "nlp"]


class TestMCPServerChoice:
    """Tests for MCPServerChoice model and MCP_SERVER_CHOICES list."""

    def test_model_creation(self) -> None:
        """Test creating a valid MCPServerChoice."""
        choice = MCPServerChoice(
            value="test-server",
            display_name="Test Server",
            description="A test MCP server",
            is_default=False,
            package_identifier="@test/server",
            command="npx",
        )
        assert choice.value == "test-server"
        assert choice.package_identifier == "@test/server"

    def test_choices_list_has_six_items(self) -> None:
        """Test that MCP_SERVER_CHOICES has exactly 6 items."""
        assert len(MCP_SERVER_CHOICES) == 6

    def test_three_defaults(self) -> None:
        """Test that exactly three choices have is_default=True."""
        defaults = [c for c in MCP_SERVER_CHOICES if c.is_default]
        assert len(defaults) == 3
        default_values = {c.value for c in defaults}
        assert default_values == {"brave-search", "memory", "sequentialthinking"}

    def test_all_choices_are_valid(self) -> None:
        """Test that all choices have valid values."""
        for choice in MCP_SERVER_CHOICES:
            assert choice.value in VALID_MCP_SERVERS
            assert choice.display_name
            assert choice.description
            assert choice.package_identifier.startswith("@")
            assert choice.command == "npx"


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_get_default_llm_provider(self) -> None:
        """Test get_default_llm_provider returns correct value."""
        assert get_default_llm_provider() == "ollama"

    def test_get_default_vector_store(self) -> None:
        """Test get_default_vector_store returns correct value."""
        assert get_default_vector_store() == "chromadb"

    def test_get_default_evals(self) -> None:
        """Test get_default_evals returns correct list."""
        defaults = get_default_evals()
        assert len(defaults) == 2
        assert "rag-faithfulness" in defaults
        assert "rag-answer_relevancy" in defaults

    def test_get_default_mcp_servers(self) -> None:
        """Test get_default_mcp_servers returns correct list."""
        defaults = get_default_mcp_servers()
        assert len(defaults) == 3
        assert "brave-search" in defaults
        assert "memory" in defaults
        assert "sequentialthinking" in defaults


class TestValidSets:
    """Tests for validation constants."""

    def test_valid_llm_providers(self) -> None:
        """Test VALID_LLM_PROVIDERS contains expected values."""
        assert (
            frozenset(["ollama", "openai", "azure_openai", "anthropic"])
            == VALID_LLM_PROVIDERS
        )

    def test_valid_vector_stores(self) -> None:
        """Test VALID_VECTOR_STORES contains expected values."""
        assert frozenset(["chromadb", "qdrant", "in-memory"]) == VALID_VECTOR_STORES

    def test_valid_evals(self) -> None:
        """Test VALID_EVALS contains expected values."""
        assert (
            frozenset(
                [
                    "rag-faithfulness",
                    "rag-answer_relevancy",
                    "rag-context_precision",
                    "rag-context_recall",
                ]
            )
            == VALID_EVALS
        )

    def test_valid_mcp_servers(self) -> None:
        """Test VALID_MCP_SERVERS contains expected values."""
        assert (
            frozenset(
                [
                    "brave-search",
                    "memory",
                    "sequentialthinking",
                    "filesystem",
                    "github",
                    "postgres",
                ]
            )
            == VALID_MCP_SERVERS
        )


class TestTemplateChoice:
    """Tests for TemplateChoice model."""

    def test_model_creation(self) -> None:
        """Test creating a valid TemplateChoice."""
        choice = TemplateChoice(
            value="test-template",
            display_name="Test Template",
            description="A test template for testing",
        )
        assert choice.value == "test-template"
        assert choice.display_name == "Test Template"
        assert choice.description == "A test template for testing"

    def test_model_creation_with_empty_description(self) -> None:
        """Test creating TemplateChoice with empty description (default)."""
        choice = TemplateChoice(
            value="minimal",
            display_name="Minimal Template",
        )
        assert choice.value == "minimal"
        assert choice.display_name == "Minimal Template"
        assert choice.description == ""

    def test_model_is_frozen(self) -> None:
        """Test that TemplateChoice is immutable."""
        choice = TemplateChoice(
            value="test",
            display_name="Test",
        )
        with pytest.raises(ValidationError):
            choice.value = "new-value"


class TestGetTemplateChoices:
    """Tests for get_template_choices() helper function."""

    def test_returns_list_of_template_choices(self) -> None:
        """Test get_template_choices returns list of TemplateChoice objects."""
        choices = get_template_choices()
        assert isinstance(choices, list)
        assert len(choices) >= 3  # At least 3 built-in templates
        for choice in choices:
            assert isinstance(choice, TemplateChoice)

    def test_includes_known_templates(self) -> None:
        """Test that known templates are included."""
        choices = get_template_choices()
        values = {c.value for c in choices}
        assert "conversational" in values
        assert "research" in values
        assert "customer-support" in values


class TestWizardResultTemplateField:
    """Tests for template field in WizardResult."""

    def test_template_default_value(self) -> None:
        """Test WizardResult uses 'conversational' as default template."""
        result = WizardResult(
            agent_name="test-agent",
            llm_provider="ollama",
            vector_store="chromadb",
        )
        assert result.template == "conversational"

    def test_template_custom_value(self) -> None:
        """Test WizardResult accepts custom template value."""
        result = WizardResult(
            agent_name="test-agent",
            template="research",
            llm_provider="ollama",
            vector_store="chromadb",
        )
        assert result.template == "research"

    @pytest.mark.parametrize(
        "template",
        ["conversational", "research", "customer-support"],
    )
    def test_valid_templates(self, template: str) -> None:
        """Test that all valid templates are accepted."""
        result = WizardResult(
            agent_name="test-agent",
            template=template,
            llm_provider="ollama",
            vector_store="chromadb",
        )
        assert result.template == template

    def test_invalid_template(self) -> None:
        """Test that invalid template raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            WizardResult(
                agent_name="test-agent",
                template="invalid-template",
                llm_provider="ollama",
                vector_store="chromadb",
            )
        assert "invalid template" in str(exc_info.value).lower()
