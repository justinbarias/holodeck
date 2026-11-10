"""Keep configured secrets out of the errors that adapters raise.

Client libraries echo what they were given: an HTTP stack names a rejected
header value, a driver prints the connection string. Every adapter turns
client errors into ``StoreConnectionError`` or ``RetrievalError``. It does so
through ``translated`` here, so a secret that the client repeated is replaced
by ``***`` and, when one was found anywhere in the original error chain, the
original is not chained to the new error.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from typing import NoReturn, TypeVar

from holodeck.lib.retrieval.errors import RetrievalError

MASK = "***"

_E = TypeVar("_E", bound=RetrievalError)


def _variants(secret: str) -> set[str]:
    """Return the forms in which a client may repeat ``secret``.

    A value with a newline is shown escaped (``key\\n``) by most HTTP stacks,
    and a value with stray whitespace is often shown stripped.
    """
    forms = {secret, secret.strip()}
    forms.add(secret.encode("unicode_escape").decode("ascii", "backslashreplace"))
    forms.add(repr(secret)[1:-1])
    return {form for form in forms if form.strip()}


def secret_values(*values: object) -> tuple[str, ...]:
    """Return the distinct non-empty text secrets among ``values``, longest first.

    Non-text values and empty strings are ignored. The longest comes first, so
    a connection string is masked as a whole before the key inside it.
    """
    found: set[str] = set()
    for value in values:
        if isinstance(value, str) and value.strip():
            found.update(_variants(value))
    return tuple(sorted(found, key=len, reverse=True))


def redact(text: str, secrets: Iterable[str]) -> str:
    """Return ``text`` with every secret replaced by ``***``."""
    for secret in secrets:
        if secret:
            text = text.replace(secret, MASK)
    return text


def _chain(error: BaseException) -> Iterator[BaseException]:
    seen: set[int] = set()
    pending: list[BaseException | None] = [error]
    while pending:
        current = pending.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        pending.extend((current.__cause__, current.__context__))


def leaks(error: BaseException, secrets: Iterable[str]) -> bool:
    """Return True when the message of ``error`` or of its chain holds a secret."""
    needles = [secret for secret in secrets if secret]
    return any(
        needle in f"{item} {item!r}" for item in _chain(error) for needle in needles
    )


def translated(
    error: BaseException,
    make: Callable[[str], _E],
    secrets: Iterable[str],
    *,
    text: str | None = None,
) -> NoReturn:
    """Raise ``make(message)`` for ``error`` with the secrets masked.

    ``text`` replaces ``str(error)`` as the message when the error has a
    better one (a gRPC error keeps it in ``details()``).

    The new error is chained to ``error`` (``raise ... from error``) unless a
    secret appears in ``error`` or anything it was raised from: then it is
    raised ``from None``, so no traceback or log record shows the original.
    """
    known = tuple(secrets)
    message = redact(str(error) if text is None else text, known)
    new = make(message)
    if leaks(error, known):
        raise new from None
    raise new from error


def check_header_value(setting: str, value: str) -> None:
    """Raise ``ValueError`` when ``value`` cannot be an HTTP header value.

    A control character (a newline from a file or an environment variable,
    most often) makes the HTTP stack reject the request and repeat the value
    in its error. The message names the setting and never the value.
    """
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ValueError(
            f"'{setting}' contains a control character, such as a newline or a tab. "
            "Remove it from the value."
        )
