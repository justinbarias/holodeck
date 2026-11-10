"""Release that removed the retired vector store providers and Azure AI metrics.

This module has no imports so config models can share one version string
without an import cycle or heavy imports at model load. The next breaking
release after the latest tag (v0.7.1) is 0.8.0; docs name it as literal text.
"""

REMOVED_IN_VERSION = "0.8.0"
