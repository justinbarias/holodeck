"""Turn dates written by people and models into UTC instants, and back.

One parser serves two callers: front-matter effective dates and the date
arguments of the memory tools. A provision dated ``2022-07-01`` and a query
for ``2022-07-01`` must become the same instant, so both go through here.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

from holodeck.lib.memory.errors import MemoryValueError


def parse_instant(value: str | date | datetime, field: str = "value") -> datetime:
    """Return ``value`` as a timezone-aware UTC datetime.

    * A date with no time is the start of that day in UTC.
    * A datetime with an offset is converted to UTC.
    * A datetime with no offset is taken as UTC.

    Args:
        value: An ISO 8601 date or datetime, as text or as an object. YAML
            front matter gives ``date`` and ``datetime`` objects.
        field: Name used in the error message.

    Raises:
        MemoryValueError: If the text is not an ISO 8601 date or datetime.
    """
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, date):
        parsed = datetime(value.year, value.month, value.day)
    else:
        text = value.strip()
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            raise MemoryValueError(
                f"{field} is not a date: '{value}'. Use ISO 8601, for example "
                "2022-07-01 or 2022-07-01T09:30:00Z."
            ) from None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    try:
        return parsed.astimezone(UTC)
    except OverflowError:
        raise MemoryValueError(
            f"{field} is not a date: '{value}' is outside the supported range."
        ) from None


def format_instant(value: datetime) -> str:
    """Return a UTC instant as ISO 8601 text with a ``Z`` suffix."""
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
