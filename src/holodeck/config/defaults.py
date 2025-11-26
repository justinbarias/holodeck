"""Default configuration templates for HoloDeck."""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def get_default_model_config(provider: str = "openai") -> dict[str, Any]:
    """Get default model configuration for a provider.

    Args:
        provider: LLM provider name (openai, azure_openai, anthropic, ollama)

    Returns:
        Dictionary with default model configuration
    """
    defaults = {
        "openai": {
            "provider": "openai",
            "name": "gpt-4o-mini",
            "temperature": 0.7,
            "max_tokens": 2048,
        },
        "azure_openai": {
            "provider": "azure_openai",
            "name": "gpt-4o",
            "temperature": 0.7,
            "max_tokens": 2048,
        },
        "anthropic": {
            "provider": "anthropic",
            "name": "claude-3-haiku-20240307",
            "temperature": 0.7,
            "max_tokens": 2048,
        },
        "ollama": {
            "provider": "ollama",
            "endpoint": "http://localhost:11434",
            "temperature": 0.3,
            "max_tokens": 1000,
            "top_p": None,
            "api_key": None,
        },
    }
    return defaults.get(provider, defaults["openai"])


def get_default_tool_config(tool_type: str | None = None) -> dict[str, Any]:
    """Get default configuration template for a tool type.

    Args:
        tool_type: Tool type (vectorstore, function, mcp, prompt).
            If None, returns generic.

    Returns:
        Dictionary with default tool configuration
    """
    if tool_type is None:
        return {"type": "function"}

    defaults: dict[str, dict[str, Any]] = {
        "vectorstore": {
            "type": "vectorstore",
            "source": "",
            "embedding_model": "text-embedding-3-small",
        },
        "function": {
            "type": "function",
            "file": "",
            "function": "",
        },
        "mcp": {
            "type": "mcp",
            "server": "",
        },
        "prompt": {
            "type": "prompt",
            "template": "",
            "parameters": {},
        },
    }
    return defaults.get(tool_type, {})


def get_default_evaluation_config(metric_name: str | None = None) -> dict[str, Any]:
    """Get default evaluation configuration.

    Args:
        metric_name: Specific metric name. If None, returns generic structure.

    Returns:
        Dictionary with default evaluation configuration
    """
    # Default per-metric configs
    metric_defaults = {
        "groundedness": {
            "metric": "groundedness",
            "threshold": 4.0,
            "enabled": True,
            "scale": 5,
        },
        "relevance": {
            "metric": "relevance",
            "threshold": 4.0,
            "enabled": True,
            "scale": 5,
        },
        "coherence": {
            "metric": "coherence",
            "threshold": 3.5,
            "enabled": True,
            "scale": 5,
        },
        "safety": {
            "metric": "safety",
            "threshold": 4.0,
            "enabled": True,
            "scale": 5,
        },
        "f1_score": {
            "metric": "f1_score",
            "threshold": 0.85,
            "enabled": True,
        },
        "bleu": {
            "metric": "bleu",
            "threshold": 0.7,
            "enabled": True,
        },
        "rouge": {
            "metric": "rouge",
            "threshold": 0.7,
            "enabled": True,
        },
    }
    if metric_name is None:
        return {
            "metrics": [
                {"metric": "groundedness", "threshold": 4.0},
                {"metric": "relevance", "threshold": 4.0},
            ]
        }
    return metric_defaults.get(metric_name, {})


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
    # Check if model is in the known mappings
    if model_name and model_name in EMBEDDING_MODEL_DIMENSIONS:
        return EMBEDDING_MODEL_DIMENSIONS[model_name]

    # Provider-specific defaults
    if provider == "ollama":
        if model_name:
            logger.warning(
                f"Unknown Ollama model '{model_name}', assuming 768 dimensions. "
                "Set 'embedding_dimensions' explicitly if different."
            )
        return 768

    # OpenAI/Azure default
    if model_name:
        logger.warning(
            f"Unknown embedding model '{model_name}', assuming 1536 dimensions. "
            f"Supported: {', '.join(EMBEDDING_MODEL_DIMENSIONS.keys())}. "
            "Set 'embedding_dimensions' explicitly if different."
        )
    return 1536
