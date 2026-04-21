"""Tests for external `test_cases_file` loader support."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from holodeck.config.loader import ConfigLoader
from holodeck.lib.errors import ConfigError


def _write_agent_yaml(path: Path, body: str) -> None:
    path.write_text(dedent("""\
            name: t
            description: test
            model:
              provider: anthropic
              name: claude-sonnet-4-6
            instructions:
              inline: hi
            """) + body)


@pytest.mark.unit
def test_test_cases_file_top_level_list(tmp_path: Path) -> None:
    cases_path = tmp_path / "cases.yaml"
    cases_path.write_text(dedent("""\
            - name: ext-case
              input: hello
              expected_tools: [lookup]
            """))
    agent_path = tmp_path / "agent.yaml"
    _write_agent_yaml(agent_path, f"test_cases_file: {cases_path.name}\n")

    agent = ConfigLoader().load_agent_yaml(str(agent_path))

    assert agent.test_cases is not None
    assert len(agent.test_cases) == 1
    assert agent.test_cases[0].name == "ext-case"


@pytest.mark.unit
def test_test_cases_file_nested_under_key(tmp_path: Path) -> None:
    cases_path = tmp_path / "cases.yaml"
    cases_path.write_text(dedent("""\
            test_cases:
              - name: multi
                turns:
                  - input: first
                  - input: second
            """))
    agent_path = tmp_path / "agent.yaml"
    _write_agent_yaml(agent_path, f"test_cases_file: {cases_path.name}\n")

    agent = ConfigLoader().load_agent_yaml(str(agent_path))

    assert agent.test_cases is not None
    assert len(agent.test_cases) == 1
    assert agent.test_cases[0].turns is not None
    assert len(agent.test_cases[0].turns) == 2


@pytest.mark.unit
def test_test_cases_file_conflict_with_inline(tmp_path: Path) -> None:
    cases_path = tmp_path / "cases.yaml"
    cases_path.write_text("- {name: a, input: x}\n")
    agent_path = tmp_path / "agent.yaml"
    body = (
        f"test_cases_file: {cases_path.name}\n"
        "test_cases:\n"
        "  - name: inline\n"
        "    input: y\n"
    )
    _write_agent_yaml(agent_path, body)

    with pytest.raises(ConfigError) as exc:
        ConfigLoader().load_agent_yaml(str(agent_path))
    assert "test_cases_file" in str(exc.value)


@pytest.mark.unit
def test_test_cases_file_missing_path(tmp_path: Path) -> None:
    agent_path = tmp_path / "agent.yaml"
    _write_agent_yaml(agent_path, "test_cases_file: no-such-file.yaml\n")
    with pytest.raises(ConfigError) as exc:
        ConfigLoader().load_agent_yaml(str(agent_path))
    assert "test_cases_file" in str(exc.value)
    assert "not found" in str(exc.value).lower()


@pytest.mark.unit
def test_test_cases_file_not_a_list(tmp_path: Path) -> None:
    cases_path = tmp_path / "cases.yaml"
    cases_path.write_text("foo: bar\n")
    agent_path = tmp_path / "agent.yaml"
    _write_agent_yaml(agent_path, f"test_cases_file: {cases_path.name}\n")
    with pytest.raises(ConfigError) as exc:
        ConfigLoader().load_agent_yaml(str(agent_path))
    assert "list of test cases" in str(exc.value)
