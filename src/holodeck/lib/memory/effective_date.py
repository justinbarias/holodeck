"""The effective date of a source document, and front matter in chunk text.

A document's effective date is the reference time of its episodes (spec D4).
The order is: the front-matter key, then the file's mtime when it is reliable,
then the ingestion time.

The functions here read files synchronously. An async caller runs
``effective_date`` with ``asyncio.to_thread``.
"""

from __future__ import annotations

import re
from datetime import UTC, date, datetime
from pathlib import Path

import yaml
from frontmatter.default_handlers import YAMLHandler

from holodeck.lib.memory.chunk_source import SourceDocument
from holodeck.lib.memory.errors import MemoryValueError
from holodeck.lib.memory.instants import parse_instant
from holodeck.lib.memory.types import ReferenceTimeSource, require_utc

_YAML = YAMLHandler()
_BOM = b"\xef\xbb\xbf"
_DELIMITER = b"---"
_TEXT_BOM = "\ufeff"
_OPENING = re.compile(r"---[ \t]*\r?\n")


def _split_front_matter(text: str) -> tuple[dict[object, object], str] | None:
    """Return ``(front matter, body)``, or None when ``text`` has none.

    Front matter is a YAML mapping between two ``---`` lines at the start of
    ``text``. A block that is not a mapping is not front matter.

    Raises:
        yaml.YAMLError: If the block is not valid YAML, or holds a value
            YAML cannot build, such as an impossible date.
    """
    text = text.removeprefix(_TEXT_BOM)
    # The library accepts three or more dashes. Front matter opens with
    # exactly three, so a Markdown rule of four is not taken for it.
    if not _OPENING.match(text) or not _YAML.detect(text):
        return None
    try:
        block, body = _YAML.split(text)
    except ValueError:
        return None
    try:
        loaded: object = _YAML.load(block)
    except ValueError as exc:
        # PyYAML builds dates itself and raises ValueError for 2022-13-01.
        raise yaml.YAMLError(str(exc)) from None
    if not isinstance(loaded, dict):
        return None
    return loaded, body


def strip_front_matter(text: str) -> str:
    """Return chunk ``text`` without a leading front-matter block.

    Text with no front matter, or with a block that is not valid YAML, is
    returned unchanged. Leading whitespace after the block is removed.
    """
    try:
        split = _split_front_matter(text)
    except yaml.YAMLError:
        return text
    if split is None:
        return text
    return split[1].lstrip()


def _read_front_matter(path: Path) -> dict[object, object] | None:
    """Return the front matter of the file at ``path``, or None.

    Only a file that starts with ``---`` is read in full, so binary sources
    such as PDF cost one short read.
    """
    try:
        with path.open("rb") as handle:
            head = handle.read(len(_BOM) + len(_DELIMITER))
        if not head.removeprefix(_BOM).startswith(_DELIMITER):
            return None
        text = path.read_bytes().decode("utf-8-sig")
    except (OSError, UnicodeDecodeError):
        # A file that is gone or unreadable has no front matter to offer.
        return None
    try:
        split = _split_front_matter(text)
    except yaml.YAMLError as exc:
        raise MemoryValueError(
            f"front matter in {path} is not valid YAML: {exc}"
        ) from None
    return None if split is None else split[0]


def document_series(document: SourceDocument, key: str) -> str:
    """Return the front-matter value of ``key`` as text, or "" when absent.

    The memory reads ``series_id`` here: documents with the same series are
    versions of one document. A string or an integer is accepted. This
    function reads the file and blocks.

    Raises:
        MemoryValueError: If the value is another type, or the front matter
            is not valid YAML.
    """
    front_matter = _read_front_matter(document.path)
    if front_matter is None or key not in front_matter:
        return ""
    value = front_matter[key]
    if isinstance(value, bool) or not isinstance(value, str | int):
        raise MemoryValueError(f"{key} is not a string: {value!r}")
    return str(value).strip()


def effective_date(
    document: SourceDocument, key: str, now: datetime
) -> tuple[datetime, ReferenceTimeSource]:
    """Return the effective date of ``document`` and where it came from.

    1. The front-matter value of ``key``.
    2. Else the file's mtime, when ``document.mtime_reliable``.
    3. Else ``now`` (ingestion time).

    The result is timezone-aware UTC. This function reads the file and
    blocks; call it with ``asyncio.to_thread`` from async code.

    Raises:
        MemoryValueError: If the front-matter value is not a date (it does
            not fall through), the front matter is not valid YAML, or ``now``
            is not UTC.
    """
    require_utc(now, "now")
    front_matter = _read_front_matter(document.path)
    if front_matter is not None and key in front_matter:
        value = front_matter[key]
        if isinstance(value, str | date):
            return parse_instant(value, key), "front_matter"
        raise MemoryValueError(
            f"{key} is not a date: {value!r}. Use ISO 8601, for example "
            "2022-07-01 or 2022-07-01T09:30:00Z."
        )
    if document.mtime_reliable:
        try:
            mtime = document.path.stat().st_mtime
        except OSError:
            return now, "ingestion"
        return datetime.fromtimestamp(mtime, UTC), "mtime"
    return now, "ingestion"
