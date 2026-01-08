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

import os

# CRITICAL: Disable deepeval telemetry BEFORE any imports to prevent it from
# setting a TracerProvider. This allows HoloDeck to control OTel configuration.
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")

from importlib.metadata import PackageNotFoundError, version

from holodeck.config.loader import ConfigLoader
from holodeck.lib.errors import ConfigError, HoloDeckError, ValidationError
from holodeck.lib.logging_config import setup_logging

try:
    __version__ = version("holodeck-ai")
except PackageNotFoundError:
    # Package not installed, development mode
    __version__ = "0.0.0.dev0"

# Initialize logging on package import
# This ensures consistent logging configuration across all modules
setup_logging()

__all__ = [
    "__version__",
    "ConfigLoader",
    "ConfigError",
    "HoloDeckError",
    "ValidationError",
    "setup_logging",
]
