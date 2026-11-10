"""Fixtures for the deploy build integration tests."""

from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def released_holodeck_version() -> Iterator[None]:
    """Pretend the running HoloDeck is a published release.

    A checkout reports a development version, which `deploy build` refuses to
    install without a wheel.
    """
    with patch(
        "holodeck.cli.commands.deploy._installed_holodeck_version",
        return_value="1.2.3",
    ):
        yield
