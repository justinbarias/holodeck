"""HoloDeck - Build, test, and deploy AI agents through YAML configuration.

HoloDeck is an open-source experimentation platform for building, testing, and
deploying AI agents through YAML configuration files. No code required.

Main features:
- Define agents entirely in YAML
- Support for multiple LLM providers (OpenAI, Azure, Anthropic)
- Flexible tool system (vectorstore, function, MCP, prompt)
- Built-in evaluation and testing framework
- OpenTelemetry observability
"""

from importlib.metadata import PackageNotFoundError, version

from holodeck.config.loader import ConfigLoader
from holodeck.lib.errors import ConfigError, HoloDeckError, ValidationError

try:
    __version__ = version("holodeck")
except PackageNotFoundError:
    # Package not installed, development mode
    __version__ = "0.0.0.dev0"

__all__ = [
    "__version__",
    "ConfigLoader",
    "ConfigError",
    "HoloDeckError",
    "ValidationError",
]
