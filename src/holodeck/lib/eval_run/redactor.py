"""Redactor for the eval-run agent-config snapshot.

Implements the two-rule policy from
``specs/031-eval-runs-dashboard/research.md`` R5:

1. **Name allowlist**: any leaf field whose name is in
   :data:`REDACTED_FIELD_NAMES` is replaced with :data:`REDACTED_PLACEHOLDER`.
2. **Type-driven**: any field annotated as :class:`pydantic.SecretStr` (or
   ``SecretStr | None``) is replaced with :data:`REDACTED_PLACEHOLDER`.

Operates on a Pydantic model instance and returns a deep copy with the same
type so the result still passes :class:`Agent` validation.
"""

from __future__ import annotations

import types
import typing
from typing import Any, TypeVar, Union, get_args, get_origin

from pydantic import BaseModel, SecretStr

REDACTED_FIELD_NAMES: frozenset[str] = frozenset({"api_key", "password", "secret"})
"""Leaf field names that are always redacted regardless of their declared type."""

REDACTED_PLACEHOLDER: str = "***"
"""The literal string written in place of a redacted value."""


_M = TypeVar("_M", bound=BaseModel)


def _annotation_is_secret(annotation: Any) -> bool:
    """Return True if ``annotation`` is ``SecretStr`` or ``SecretStr | None``."""
    if annotation is SecretStr:
        return True
    origin = get_origin(annotation)
    if origin is Union or origin is types.UnionType:
        return any(arg is SecretStr for arg in get_args(annotation))
    return False


def _redact_value(value: Any) -> Any:
    """Walk a serialisable value tree and redact in-place by reconstruction."""
    if isinstance(value, BaseModel):
        return _redact_model(value)
    if isinstance(value, list):
        return [_redact_value(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_redact_value(v) for v in value)
    if isinstance(value, dict):
        return {k: _redact_value(v) for k, v in value.items()}
    return value


def _redact_model(instance: BaseModel) -> BaseModel:
    """Return a deep copy of ``instance`` with secret leaves replaced.

    Secret-typed leaves (``SecretStr``) become ``SecretStr(REDACTED_PLACEHOLDER)``
    so that downstream callers can still ``.get_secret_value()`` on them. Plain
    ``str`` leaves whose name is in :data:`REDACTED_FIELD_NAMES` become the
    bare placeholder string.
    """
    cls = type(instance)
    overrides: dict[str, Any] = {}
    for field_name, field_info in cls.model_fields.items():
        current = getattr(instance, field_name)
        if current is None:
            continue
        is_secret_type = _annotation_is_secret(field_info.annotation)
        is_allowlisted_name = field_name in REDACTED_FIELD_NAMES
        if is_secret_type:
            overrides[field_name] = SecretStr(REDACTED_PLACEHOLDER)
        elif is_allowlisted_name:
            overrides[field_name] = REDACTED_PLACEHOLDER
        else:
            redacted_child = _redact_value(current)
            if redacted_child is not current:
                overrides[field_name] = redacted_child
    if not overrides:
        return instance.model_copy(deep=True)
    return instance.model_copy(update=overrides, deep=True)


def redact(instance: _M) -> _M:
    """Return a redacted deep copy of a Pydantic ``BaseModel``.

    The returned value has the same Pydantic type as the input, so callers can
    embed it inside :class:`EvalRunMetadata` (which requires ``agent_config:
    Agent``).

    Args:
        instance: The model to redact (typically an :class:`Agent`).

    Returns:
        A deep-copied instance with secret-bearing leaves replaced by
        :data:`REDACTED_PLACEHOLDER`.
    """
    redacted = _redact_model(instance)
    return typing.cast(_M, redacted)
