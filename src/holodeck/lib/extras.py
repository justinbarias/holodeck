"""Optional install extras: one error and one import helper for all of them.

``pip install holodeck-ai`` installs only what every user needs. Features with
heavy or rarely used dependencies (DeepEval metrics, NLP metrics, the
optimizer, the OpenSearch keyword index, vector store clients, the dashboard)
live in extras. A feature whose extra is missing raises ``MissingExtraError``
with the exact install command, and the CLI prints it without a traceback.
"""

from __future__ import annotations

import importlib
from types import ModuleType

from holodeck.lib.errors import UserFixableError


def install_command(extra: str) -> str:
    """Return the pip command that installs *extra*."""
    return f"pip install 'holodeck-ai[{extra}]'"


class MissingExtraError(UserFixableError):
    """A feature needs an install extra that is not installed.

    Attributes:
        feature: What the user selected, e.g. ``"DeepEval metrics"``.
        extra: The extra that provides it, e.g. ``"eval"``.
        install_command: The command that installs the extra.
    """

    def __init__(self, feature: str, extra: str, detail: str = "") -> None:
        self.feature = feature
        self.extra = extra
        self.install_command = install_command(extra)
        message = (
            f"{feature} needs the '{extra}' extra, which is not installed. "
            f"Install it with: {self.install_command}"
        )
        if detail:
            message = f"{message} ({detail})"
        super().__init__(message)


def _is_missing(error: ModuleNotFoundError, module_name: str) -> bool:
    """True when *error* says *module_name* (or a parent package) is absent."""
    missing = error.name
    if not missing:
        return False
    return module_name == missing or module_name.startswith(f"{missing}.")


def require_extra(module_name: str, extra: str, feature: str) -> ModuleType:
    """Import *module_name*, or raise ``MissingExtraError`` naming *extra*.

    Only a missing *module_name* (or one of its parent packages) is reported as
    a missing extra. Any other import failure inside an installed package
    propagates unchanged, so a broken install is not mistaken for an absent one.
    """
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        if _is_missing(error, module_name):
            raise MissingExtraError(feature, extra) from error
        raise
