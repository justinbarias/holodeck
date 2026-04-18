"""Unit tests for build_eval_run_metadata (T017).

Asserts:
- git_commit captured when in a git repo (mocked subprocess)
- git_commit None when git rev-parse fails or times out
- holodeck_version read via importlib.metadata.version
- cli_args echoes sys.argv[1:]
"""

from __future__ import annotations

import subprocess
from unittest.mock import patch

import pytest

from holodeck.lib.eval_run.metadata import build_eval_run_metadata
from holodeck.models.agent import Agent
from holodeck.models.eval_run import PromptVersion


def _make_agent() -> Agent:
    return Agent(
        name="meta-agent",
        model={"provider": "openai", "name": "gpt-4o"},
        instructions={"inline": "hi"},
    )


def _make_pv() -> PromptVersion:
    return PromptVersion(version="auto-deadbeef", source="inline", body_hash="0" * 64)


@pytest.mark.unit
class TestGitCommitCapture:
    def test_captured_when_git_succeeds(self):
        fake_completed = subprocess.CompletedProcess(
            args=["git", "rev-parse", "HEAD"],
            returncode=0,
            stdout="abc1234deadbeef\n",
            stderr="",
        )
        with patch(
            "holodeck.lib.eval_run.metadata.subprocess.run",
            return_value=fake_completed,
        ):
            meta = build_eval_run_metadata(_make_agent(), _make_pv(), argv=[])
        assert meta.git_commit == "abc1234deadbeef"

    def test_none_when_git_returns_nonzero(self):
        fake_completed = subprocess.CompletedProcess(
            args=["git", "rev-parse", "HEAD"],
            returncode=128,
            stdout="",
            stderr="not a git repo\n",
        )
        with patch(
            "holodeck.lib.eval_run.metadata.subprocess.run",
            return_value=fake_completed,
        ):
            meta = build_eval_run_metadata(_make_agent(), _make_pv(), argv=[])
        assert meta.git_commit is None

    def test_none_when_git_raises_timeout(self):
        with patch(
            "holodeck.lib.eval_run.metadata.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="git", timeout=2),
        ):
            meta = build_eval_run_metadata(_make_agent(), _make_pv(), argv=[])
        assert meta.git_commit is None

    def test_none_when_git_binary_missing(self):
        with patch(
            "holodeck.lib.eval_run.metadata.subprocess.run",
            side_effect=FileNotFoundError("git not installed"),
        ):
            meta = build_eval_run_metadata(_make_agent(), _make_pv(), argv=[])
        assert meta.git_commit is None


@pytest.mark.unit
class TestProvenanceFields:
    def test_holodeck_version_read_from_importlib(self):
        with patch(
            "holodeck.lib.eval_run.metadata.importlib_version",
            return_value="9.9.9",
        ):
            meta = build_eval_run_metadata(_make_agent(), _make_pv(), argv=[])
        assert meta.holodeck_version == "9.9.9"

    def test_holodeck_version_falls_back_when_package_missing(self):
        from importlib.metadata import PackageNotFoundError

        with patch(
            "holodeck.lib.eval_run.metadata.importlib_version",
            side_effect=PackageNotFoundError("holodeck-ai"),
        ):
            meta = build_eval_run_metadata(_make_agent(), _make_pv(), argv=[])
        # Use a sentinel value documented in data-model.md
        assert meta.holodeck_version == "0.0.0.dev0"

    def test_cli_args_echo_argv_when_passed(self):
        meta = build_eval_run_metadata(
            _make_agent(), _make_pv(), argv=["test", "agent.yaml", "--verbose"]
        )
        assert meta.cli_args == ["test", "agent.yaml", "--verbose"]

    def test_cli_args_default_to_sys_argv(self):
        with patch(
            "holodeck.lib.eval_run.metadata.sys.argv",
            ["holodeck", "test", "agent.yaml"],
        ):
            meta = build_eval_run_metadata(_make_agent(), _make_pv())
        assert meta.cli_args == ["test", "agent.yaml"]
