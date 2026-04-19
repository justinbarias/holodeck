"""Tests for the assistant-bubble renderer in the Explorer view."""

from __future__ import annotations

import json

import pytest
from dash import dcc, html

from holodeck.dashboard.views.explorer import _render_assistant_body


@pytest.mark.unit
def test_plain_prose_renders_as_markdown():
    comp = _render_assistant_body("Sorry, I couldn't find that.")
    assert isinstance(comp, dcc.Markdown)
    assert comp.children == "Sorry, I couldn't find that."
    assert comp.className == "md-assistant"


@pytest.mark.unit
def test_markdown_prose_renders_as_markdown():
    text = "**Title I > §103** is the definition clause.\n\n> quoted"
    comp = _render_assistant_body(text)
    assert isinstance(comp, dcc.Markdown)
    assert comp.children == text


@pytest.mark.unit
def test_json_object_renders_as_prettified_pre():
    raw = '{"answer":"yes","refs":[1,2,3]}'
    comp = _render_assistant_body(raw)
    assert isinstance(comp, html.Pre)
    assert "lang-json" in comp.className
    # Prettified (indent=2) output should round-trip to the same parsed value.
    assert json.loads(comp.children) == {"answer": "yes", "refs": [1, 2, 3]}
    assert "\n" in comp.children  # indent=2 forces newlines


@pytest.mark.unit
def test_json_array_renders_as_prettified_pre():
    raw = "[1, 2, 3]"
    comp = _render_assistant_body(raw)
    assert isinstance(comp, html.Pre)
    assert json.loads(comp.children) == [1, 2, 3]


@pytest.mark.unit
def test_leading_whitespace_json_still_detected():
    raw = '   \n  {"k": 1}'
    comp = _render_assistant_body(raw)
    assert isinstance(comp, html.Pre)


@pytest.mark.unit
def test_invalid_json_starting_with_brace_falls_back_to_markdown():
    # Looks JSON-ish but isn't parseable.
    raw = "{not actually json}"
    comp = _render_assistant_body(raw)
    assert isinstance(comp, dcc.Markdown)


@pytest.mark.unit
def test_bare_number_not_treated_as_json():
    # A bare number parses as JSON via json.loads, but we require a { or [
    # prefix — otherwise chatty numeric replies ("42.") would flip to JSON.
    comp = _render_assistant_body("42")
    assert isinstance(comp, dcc.Markdown)


@pytest.mark.unit
def test_json_embedded_midsentence_stays_markdown():
    raw = 'The answer is {"value": 5} in the record.'
    comp = _render_assistant_body(raw)
    assert isinstance(comp, dcc.Markdown)
