"""Tests for the shared date parser (spec 043, HST-02; design seam 6)."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta, timezone

import pytest

from holodeck.lib.memory.errors import MemoryValueError
from holodeck.lib.memory.instants import format_instant, parse_instant

pytestmark = pytest.mark.unit

MIDNIGHT = datetime(2022, 7, 1, tzinfo=UTC)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2022-07-01", MIDNIGHT),
        (" 2022-07-01 ", MIDNIGHT),
        (date(2022, 7, 1), MIDNIGHT),
        ("2022-07-01T09:30:00", datetime(2022, 7, 1, 9, 30, tzinfo=UTC)),
        (datetime(2022, 7, 1, 9, 30), datetime(2022, 7, 1, 9, 30, tzinfo=UTC)),
        ("2022-07-01T09:30:00Z", datetime(2022, 7, 1, 9, 30, tzinfo=UTC)),
        ("2022-07-01T09:30:00+10:00", datetime(2022, 6, 30, 23, 30, tzinfo=UTC)),
        (
            datetime(2022, 7, 1, 9, 30, tzinfo=timezone(timedelta(hours=10))),
            datetime(2022, 6, 30, 23, 30, tzinfo=UTC),
        ),
    ],
)
def test_hst_02_dates_become_utc_instants(
    value: str | date | datetime, expected: datetime
) -> None:
    parsed = parse_instant(value)

    assert parsed == expected
    assert parsed.utcoffset() == timedelta(0)


def test_a_date_argument_and_a_front_matter_date_meet_at_one_instant() -> None:
    assert parse_instant("2022-07-01") == parse_instant(date(2022, 7, 1))


@pytest.mark.parametrize("value", ["last July", "", "2022-13-01", "01/07/2022"])
def test_text_that_is_not_a_date_names_the_field(value: str) -> None:
    with pytest.raises(MemoryValueError, match="valid_at is not a date"):
        parse_instant(value, "valid_at")


def test_hst_02_instants_are_written_with_a_z_suffix() -> None:
    assert format_instant(MIDNIGHT) == "2022-07-01T00:00:00Z"
    offset = datetime(2022, 7, 1, 9, 30, tzinfo=timezone(timedelta(hours=10)))
    assert format_instant(offset) == "2022-06-30T23:30:00Z"


@pytest.mark.parametrize(
    "value", ["0001-01-01T00:00:00+01:00", "9999-12-31T23:00:00-02:00"]
)
def test_an_instant_outside_the_supported_range_names_the_field(value: str) -> None:
    with pytest.raises(MemoryValueError, match="valid_at is not a date"):
        parse_instant(value, "valid_at")
