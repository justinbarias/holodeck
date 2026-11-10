"""Shared helpers for the deploy unit tests."""

from __future__ import annotations

from functools import partial

from holodeck.deploy.dockerfile import generate_dockerfile as _generate_dockerfile

# Tests that do not care which HoloDeck the image installs pin a fixed version.
TEST_HOLODECK_VERSION = "1.2.3"

generate_dockerfile = partial(
    _generate_dockerfile, holodeck_version=TEST_HOLODECK_VERSION
)
