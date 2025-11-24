"""Tools module for HoloDeck agent capabilities.

This module provides tool implementations that extend agent capabilities:
- VectorStoreTool: Semantic search over unstructured documents
"""

from holodeck.tools.vectorstore_tool import VectorStoreTool

__all__ = ["VectorStoreTool"]
