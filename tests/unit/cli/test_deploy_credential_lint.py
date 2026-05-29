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


@pytest.mark.unit
def test_no_crash_on_nonexistent_data_directory(tmp_path: Path, caplog):
    """Helper must continue (not raise) when a data directory does not exist."""
    missing = tmp_path / "__nonexistent_holodeck_test_dir__"
    with caplog.at_level(logging.WARNING):
        _warn_if_credential_files_in_copy_surface(
            instruction_files=[],
            data_directories=[str(missing)],
            base_dir=tmp_path,
        )
    # No exception; no warning emitted for missing directory.
    assert not any("credential" in r.message.lower() for r in caplog.records)


@pytest.mark.unit
def test_warns_on_env_local_matching_glob_pattern(tmp_path: Path, caplog):
    """`.env.*` glob pattern matches `.env.local`, `.env.production`, etc."""
    (tmp_path / ".env.local").write_text("SECRET=x")
    with caplog.at_level(logging.WARNING):
        _warn_if_credential_files_in_copy_surface(
            instruction_files=[],
            data_directories=[str(tmp_path)],
            base_dir=tmp_path.parent,
        )
    assert any(".env.local" in r.message for r in caplog.records)


@pytest.mark.unit
def test_symlink_entries_do_not_crash(tmp_path: Path, caplog):
    """Symlinks in the directory are traversed without raising."""
    target = tmp_path / "real.txt"
    target.write_text("hello")
    link = tmp_path / "link.pem"
    link.symlink_to(target)
    with caplog.at_level(logging.WARNING):
        _warn_if_credential_files_in_copy_surface(
            instruction_files=[],
            data_directories=[str(tmp_path)],
            base_dir=tmp_path.parent,
        )
    # The symlink matches *.pem so a warning should fire.
    assert any("link.pem" in r.message for r in caplog.records)


@pytest.mark.unit
def test_warns_on_credential_file_in_instruction_files(tmp_path: Path, caplog):
    """Individual instruction_files are also checked against patterns."""
    with caplog.at_level(logging.WARNING):
        _warn_if_credential_files_in_copy_surface(
            instruction_files=["secrets/my-service.key"],
            data_directories=[],
            base_dir=tmp_path,
        )
    assert any("my-service.key" in r.message for r in caplog.records)
