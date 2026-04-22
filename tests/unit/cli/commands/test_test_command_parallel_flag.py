"""Tests for `holodeck test --parallel-test-cases N` (T038)."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from click.testing import CliRunner

from holodeck.cli.commands.test import test
from tests.unit.cli.commands.test_test import (
    _create_agent_with_tests,
    _create_mock_report,
)


class TestParallelTestCasesFlag:
    def test_flag_accepted(self) -> None:
        runner = CliRunner()
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            with (
                patch("holodeck.config.loader.ConfigLoader") as mock_loader_class,
                patch("holodeck.cli.commands.test.TestExecutor") as mock_executor,
                patch("holodeck.cli.commands.test.ProgressIndicator"),
            ):
                mock_loader = MagicMock()
                mock_loader.load_agent_yaml.return_value = _create_agent_with_tests(0)
                mock_loader_class.return_value = mock_loader

                mock_instance = MagicMock()
                mock_instance.execute_tests = AsyncMock(
                    return_value=_create_mock_report(tmp_path)
                )
                mock_executor.return_value = mock_instance

                result = runner.invoke(test, [tmp_path, "--parallel-test-cases", "4"])
                assert "no such option" not in result.output.lower()
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_rejects_zero(self) -> None:
        runner = CliRunner()
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            result = runner.invoke(test, [tmp_path, "--parallel-test-cases", "0"])
            assert result.exit_code != 0
        finally:
            Path(tmp_path).unlink(missing_ok=True)
