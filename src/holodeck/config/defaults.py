"""Default configuration templates for HoloDeck."""

from typing import Any


def get_default_model_config(provider: str = "openai") -> dict[str, Any]:
    """Get default model configuration for a provider.

    Args:
        provider: LLM provider name (openai, azure_openai, anthropic)

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
