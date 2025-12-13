"""Test fixtures and defaults for wizard tests.

This module provides default values and test fixtures for testing
the interactive initialization wizard functionality.
"""

# Default selections for testing
DEFAULT_LLM_PROVIDER = "ollama"
DEFAULT_VECTOR_STORE = "chromadb"
DEFAULT_EVALS = ["rag-faithfulness", "rag-answer_relevancy"]
DEFAULT_MCP_SERVERS = ["brave-search", "memory", "sequentialthinking"]
DEFAULT_AGENT_NAME = "my-agent"

# Valid LLM providers
VALID_LLM_PROVIDERS = ["ollama", "openai", "azure_openai", "anthropic"]

# Valid vector stores (aligned with DatabaseConfig.provider in holodeck.models.tool)
VALID_VECTOR_STORES = ["chromadb", "qdrant", "in-memory"]

# Valid evaluation metrics
VALID_EVALS = [
    "rag-faithfulness",
    "rag-answer_relevancy",
    "rag-context_precision",
    "rag-context_recall",
]

# Valid MCP servers
VALID_MCP_SERVERS = [
    "brave-search",
    "memory",
    "sequentialthinking",
    "filesystem",
    "github",
    "postgres",
]

# Sample valid agent names for testing
VALID_AGENT_NAMES = [
    "my-agent",
    "test_agent",
    "Agent123",
    "a",
    "agent-with-dashes",
    "agent_with_underscores",
]

# Sample invalid agent names for testing
INVALID_AGENT_NAMES = [
    "",  # Empty
    "123agent",  # Starts with digit
    "agent@name",  # Invalid character
    "agent name",  # Space not allowed
    "agent.name",  # Dot not allowed
]
