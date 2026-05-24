"""Deploy-time warning when COPY surface contains credential-shaped files."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from holodeck.cli.commands.deploy import _warn_if_credential_files_in_copy_surface


@pytest.mark.unit
def test_warn_on_env_file(tmp_path: Path, caplog):
    (tmp_path / ".env").write_text("SECRET=1")
    with caplog.at_level(logging.WARNING):
        _warn_if_credential_files_in_copy_surface(
            instruction_files=[],
            data_directories=[str(tmp_path)],
            base_dir=tmp_path.parent,
        )
    assert any(".env" in r.message for r in caplog.records)


@pytest.mark.unit
def test_no_warning_on_clean_directory(tmp_path: Path, caplog):
    (tmp_path / "data.csv").write_text("a,b")
    with caplog.at_level(logging.WARNING):
        _warn_if_credential_files_in_copy_surface(
            instruction_files=[],
            data_directories=[str(tmp_path)],
            base_dir=tmp_path.parent,
        )
    assert not any("credential" in r.message.lower() for r in caplog.records)


@pytest.mark.unit
@pytest.mark.parametrize(
    "name", ["secrets.pem", "id_rsa", "azure-credentials.json", "service-account.json"]
)
def test_warn_on_credential_shaped_filenames(tmp_path: Path, caplog, name):
    (tmp_path / name).write_text("x")
    with caplog.at_level(logging.WARNING):
        _warn_if_credential_files_in_copy_surface(
            instruction_files=[],
            data_directories=[str(tmp_path)],
            base_dir=tmp_path.parent,
        )
    assert any(name in r.message for r in caplog.records)
