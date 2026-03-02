"""Default configuration templates for HoloDeck."""

import logging

logger = logging.getLogger(__name__)


# Ollama provider defaults
OLLAMA_DEFAULTS: dict[str, int | float | str | None] = {
    "endpoint": "http://localhost:11434",
    "temperature": 0.3,
    "max_tokens": 1000,
    "top_p": None,
    "api_key": None,
}

# Ollama provider embedding model defaults
OLLAMA_EMBEDDING_DEFAULTS: dict[str, str | None] = {
    "embedding_model": "nomic-embed-text:latest",
}

# Execution configuration defaults
DEFAULT_EXECUTION_CONFIG: dict[str, int | bool | str] = {
    "file_timeout": 30,  # seconds
    "llm_timeout": 60,  # seconds
    "download_timeout": 30,  # seconds
    "cache_enabled": True,
    "cache_dir": ".holodeck/cache",
    "verbose": False,
    "quiet": False,
}

# Embedding model dimension defaults
EMBEDDING_MODEL_DIMENSIONS: dict[str, int] = {
    # OpenAI models
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
    # Ollama models
    "nomic-embed-text:latest": 768,
    "mxbai-embed-large": 1024,
    "snowflake-arctic-embed": 1024,
}


def get_embedding_dimensions(
    model_name: str | None,
    provider: str = "openai",
) -> int:
    """Get embedding dimensions for a model.

    Resolution order:
    1. Known model in EMBEDDING_MODEL_DIMENSIONS
    2. Provider default (openai: 1536, ollama: 768)
    3. Fallback to 1536 with warning

    Args:
        model_name: Embedding model name (e.g., "text-embedding-3-small")
        provider: LLM provider ("openai", "azure_openai", "ollama")

    Returns:
        Embedding dimensions for the model
    """
    if model_name and model_name in EMBEDDING_MODEL_DIMENSIONS:
        return EMBEDDING_MODEL_DIMENSIONS[model_name]

    if provider == "ollama":
        if model_name:
            logger.warning(
                f"Unknown Ollama model '{model_name}', assuming 768 dimensions. "
                "Set 'embedding_dimensions' explicitly if different."
            )
        return 768

    if model_name:
        logger.warning(
            f"Unknown embedding model '{model_name}', assuming 1536 dimensions. "
            f"Supported: {', '.join(EMBEDDING_MODEL_DIMENSIONS.keys())}. "
            "Set 'embedding_dimensions' explicitly if different."
        )
    return 1536
