"""Small shared building blocks: icons, state words, revision chips, cues.

Icons are inline SVG drawn through CSS masks (``.ico-*`` in
``02-holodeck.css``) so they follow ``currentColor``; Dash's html package has
no ``<svg>`` element. Every state pairs an icon with a word.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from dash import html

from holodeck.dashboard.revisions import RevisionColour, parse_run_time

StateKind = Literal["pass", "fail", "regression", "warn", "miss", "done"]

_STATE_ICON: dict[str, str] = {
    "pass": "check",  # nosec B105
    "done": "check",
    "fail": "x",
    "regression": "down",
    "warn": "warn",
    "miss": "minus",
}

_STATE_WORD: dict[str, str] = {
    "pass": "Pass",  # nosec B105
    "done": "Done",
    "fail": "Fail",
    "regression": "Regression",
    "warn": "Warning",
    "miss": "Not run",
}


def icon(name: str, extra_class: str = "") -> html.Span:
    """Decorative icon; the adjacent text carries the meaning."""
    cls = f"ico ico-{name}"
    if extra_class:
        cls += f" {extra_class}"
    return html.Span(className=cls, **{"aria-hidden": "true"})


def sr_only(text: str) -> html.Span:
    return html.Span(text, className="sr-only")


def state_word(
    kind: StateKind,
    text: str | None = None,
    *,
    badge: bool = False,
) -> html.Span:
    """Icon plus word in the state colour; ``badge`` adds a tinted fill."""
    cls = f"state state-{kind}"
    if badge:
        cls += " state-badge"
    return html.Span(
        [icon(_STATE_ICON[kind]), text if text is not None else _STATE_WORD[kind]],
        className=cls,
    )


def revision_chip(
    version: str,
    colour: RevisionColour,
    *,
    text: str | None = None,
    extra_class: str = "",
) -> html.Span:
    """Revision-coloured chip for a prompt version (and only for one)."""
    cls = "rv"
    if extra_class:
        cls += f" {extra_class}"
    return html.Span(
        text if text is not None else version,
        className=cls,
        style={"background": colour.tint, "color": colour.ink},
        title=f"Prompt version {version}",
    )


def cue(text: str, *, strong: bool = True, extra_class: str = "") -> html.Span:
    """Courier Prime run ids, case names, metric names and data numbers."""
    cls = "cue"
    if strong:
        cls += " cue-strong"
    if extra_class:
        cls += f" {extra_class}"
    return html.Span(text, className=cls)


def signed(value: float, fmt: str = "{:.1f}", *, zero: str = "± 0.0") -> str:
    """``+1.2`` / ``−1.2`` (true minus sign) / ``± 0.0``."""
    if round(value, 6) == 0:
        return zero
    body = fmt.format(abs(value))
    if body.strip("0.") == "":
        return zero
    return ("+" if value > 0 else "−") + body


def delta_class(value: float, *, lower_is_better: bool = False) -> str:
    """``delta-up`` (good), ``delta-down`` (bad) or ``delta-flat``."""
    if round(value, 6) == 0:
        return "delta-flat"
    good = value < 0 if lower_is_better else value > 0
    return "delta-up" if good else "delta-down"


def fmt_day(ts: datetime) -> str:
    """``16 Apr`` without a platform-specific strftime flag."""
    return f"{ts.day} {ts.strftime('%b')}"


def fmt_when(ts: datetime) -> str:
    """``16 Apr 22:59`` (the saved timestamp's own zone)."""
    return f"{ts.day} {ts.strftime('%b %H:%M')}"


def fmt_date(ts: datetime) -> str:
    """``16 Apr 2026``."""
    return f"{ts.day} {ts.strftime('%b %Y')}"


def parse_ts(run_id: str) -> datetime:
    """Parse a run id as an aware datetime (naive read as UTC)."""
    return parse_run_time(run_id)
