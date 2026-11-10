"""Fixtures shared by the deploy unit tests."""

from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import patch

import pytest

from tests.unit.deploy.helpers import TEST_HOLODECK_VERSION


@pytest.fixture(autouse=True)
def released_holodeck_version() -> Iterator[None]:
    """Pretend the running HoloDeck is a published release.

    The real version in a checkout is a development version, which `deploy
    build` refuses to install without a wheel.
    """
    with patch(
        "holodeck.cli.commands.deploy._installed_holodeck_version",
        return_value=TEST_HOLODECK_VERSION,
    ):
        yield
