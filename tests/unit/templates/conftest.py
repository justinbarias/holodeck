"""Shared fixtures for template tests."""

import tempfile
from pathlib import Path

import pytest

from holodeck.lib.template_engine import TemplateRenderer


@pytest.fixture
def renderer() -> TemplateRenderer:
    """Create a TemplateRenderer instance."""
    return TemplateRenderer()


@pytest.fixture
def template_temp_dir() -> Path:
    """Create a temporary directory for template files."""
    return Path(tempfile.mkdtemp())
