"""The ``temporal`` extra is guarded at package import (spec 040, T1).

``temporalio`` ships only in ``holodeck-ai[temporal]``. Importing
``holodeck.temporal`` without it must name the missing extra through the
project error hierarchy, not surface a bare ``ModuleNotFoundError`` raised
somewhere deeper in the import graph.
"""

import importlib
import subprocess
import sys

import pytest

from holodeck.lib.errors import ConfigError

pytestmark = pytest.mark.unit

# Blocks `temporalio` at import time, then imports the package and reports what
# came out. Runs in a subprocess so the guard executes on a cold interpreter,
# the way an install without the extra would hit it.
_ABSENT_EXTRA_SCRIPT = """
import sys

class _Blocker:
    def find_spec(self, name, path=None, target=None):
        if name == "temporalio" or name.startswith("temporalio."):
            raise ModuleNotFoundError(name)
        return None

sys.meta_path.insert(0, _Blocker())
for name in [m for m in sys.modules if m.startswith("temporalio")]:
    del sys.modules[name]

try:
    import holodeck.temporal  # noqa: F401
except Exception as exc:
    print(type(exc).__name__)
    print(str(exc))
else:
    print("NoError")
"""


@pytest.fixture
def temporal_module():
    """Import ``holodeck.temporal`` fresh and leave sys.modules as found."""
    # Arrange
    saved = sys.modules.pop("holodeck.temporal", None)
    module = importlib.import_module("holodeck.temporal")
    yield module
    if saved is not None:
        sys.modules["holodeck.temporal"] = saved


def test_import_succeeds_when_extra_is_installed(temporal_module):
    """The dev environment installs all extras, so the guard must pass."""
    # Assert
    assert callable(temporal_module.require_temporalio)
    assert temporal_module.require_temporalio() is None


def test_guard_raises_config_error_when_spec_is_missing(temporal_module, monkeypatch):
    """A missing ``temporalio`` spec is reported as a ConfigError."""
    # Arrange
    monkeypatch.setattr(temporal_module, "find_spec", lambda name: None)

    # Act
    with pytest.raises(ConfigError) as excinfo:
        temporal_module.require_temporalio()

    # Assert
    assert excinfo.value.field == "dependencies"
    assert "holodeck-ai[temporal]" in excinfo.value.message


def test_cold_import_without_the_extra_raises_config_error():
    """Importing the package on an interpreter without the SDK is guarded."""
    # Act
    result = subprocess.run(
        [sys.executable, "-c", _ABSENT_EXTRA_SCRIPT],
        capture_output=True,
        text=True,
        check=True,
    )

    # Assert
    exc_name, _, message = result.stdout.partition("\n")
    assert exc_name.strip() == "ConfigError"
    assert "holodeck-ai[temporal]" in message
