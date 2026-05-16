"""Unit tests for ``holodeck.lib.evaluators.response_path``.

Covers path grammar validation and JSON-extraction behaviour for the
``EvaluationMetric.response_path`` feature.
"""

from __future__ import annotations

import json

import pytest

from holodeck.lib.evaluators.response_path import extract, is_valid_path


@pytest.mark.unit
@pytest.mark.parametrize(
    "path",
    [
        "answer",
        "result",
        "result.value",
        "data.items[0]",
        "data.items[0].score",
        "choices[12].text",
        "snake_case_key",
        "kebab-case-key",
        "a.b.c.d.e",
    ],
)
def test_is_valid_path_accepts_supported_grammar(path: str) -> None:
    assert is_valid_path(path) is True


@pytest.mark.unit
@pytest.mark.parametrize(
    "path",
    [
        "",
        " ",
        ".",
        ".answer",
        "answer.",
        "1answer",
        "answer..value",
        "answer[",
        "answer[]",
        "answer[a]",
        "answer[-1]",
        "items[0",
        "items 0",
        "$.answer",
        "answer/value",
        "answer.value!",
    ],
)
def test_is_valid_path_rejects_bad_grammar(path: str) -> None:
    assert is_valid_path(path) is False


@pytest.mark.unit
def test_extract_flat_key() -> None:
    value, err = extract(json.dumps({"answer": 0.5}), "answer")
    assert err is None
    assert value == "0.5"


@pytest.mark.unit
def test_extract_nested_key() -> None:
    body = json.dumps({"result": {"value": "31903.6"}})
    value, err = extract(body, "result.value")
    assert err is None
    assert value == "31903.6"


@pytest.mark.unit
def test_extract_index() -> None:
    body = json.dumps({"items": [{"score": 0.9}, {"score": 0.1}]})
    value, err = extract(body, "items[1].score")
    assert err is None
    assert value == "0.1"


@pytest.mark.unit
def test_extract_coerces_scalars_to_str() -> None:
    for raw, expected in (
        ({"x": 42}, "42"),
        ({"x": 3.14}, "3.14"),
        ({"x": "hi"}, "hi"),
        ({"x": True}, "True"),
        ({"x": False}, "False"),
        ({"x": None}, "None"),
    ):
        value, err = extract(json.dumps(raw), "x")
        assert err is None, raw
        assert value == expected, raw


@pytest.mark.unit
def test_extract_rejects_non_json_response() -> None:
    value, err = extract("approximately 31903.6 million", "answer")
    assert value is None
    assert err is not None
    assert "not JSON" in err


@pytest.mark.unit
def test_extract_rejects_missing_key() -> None:
    value, err = extract(json.dumps({"foo": 1}), "bar")
    assert value is None
    assert err is not None
    assert "'bar'" in err
    assert "not found" in err


@pytest.mark.unit
def test_extract_rejects_index_out_of_range() -> None:
    value, err = extract(json.dumps({"xs": [1, 2]}), "xs[5]")
    assert value is None
    assert err is not None
    assert "out of range" in err


@pytest.mark.unit
def test_extract_rejects_type_mismatch_dict_expected() -> None:
    value, err = extract(json.dumps({"a": 1}), "a.b")
    assert value is None
    assert err is not None
    assert "expected dict" in err


@pytest.mark.unit
def test_extract_rejects_type_mismatch_list_expected() -> None:
    value, err = extract(json.dumps({"a": 1}), "a[0]")
    assert value is None
    assert err is not None
    assert "expected list" in err


@pytest.mark.unit
def test_extract_rejects_container_leaf() -> None:
    value, err = extract(json.dumps({"a": {"b": 1}}), "a")
    assert value is None
    assert err is not None
    assert "leaf is dict" in err
