"""Tests for the results-dir fingerprint + memo in the dashboard app."""

from __future__ import annotations

import time

import pytest


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    # Make sure each test starts from a clean env.
    monkeypatch.delenv("HOLODECK_DASHBOARD_USE_SEED", raising=False)
    monkeypatch.delenv("HOLODECK_DASHBOARD_RESULTS_DIR", raising=False)


@pytest.mark.unit
def test_fingerprint_zero_when_dir_unset(monkeypatch):
    from holodeck.dashboard import app as app_mod

    assert app_mod._results_dir_fingerprint() == 0.0


@pytest.mark.unit
def test_fingerprint_zero_when_dir_missing(monkeypatch, tmp_path):
    from holodeck.dashboard import app as app_mod

    monkeypatch.setenv(
        "HOLODECK_DASHBOARD_RESULTS_DIR", str(tmp_path / "does-not-exist")
    )
    assert app_mod._results_dir_fingerprint() == 0.0


@pytest.mark.unit
def test_fingerprint_zero_in_seed_mode(monkeypatch, tmp_path):
    from holodeck.dashboard import app as app_mod

    monkeypatch.setenv("HOLODECK_DASHBOARD_USE_SEED", "1")
    monkeypatch.setenv("HOLODECK_DASHBOARD_RESULTS_DIR", str(tmp_path))
    (tmp_path / "run-1.json").write_text("{}")
    assert app_mod._results_dir_fingerprint() == 0.0


@pytest.mark.unit
def test_fingerprint_nonzero_when_dir_has_files(monkeypatch, tmp_path):
    from holodeck.dashboard import app as app_mod

    monkeypatch.setenv("HOLODECK_DASHBOARD_RESULTS_DIR", str(tmp_path))
    (tmp_path / "run-1.json").write_text("{}")
    assert app_mod._results_dir_fingerprint() > 0.0


@pytest.mark.unit
def test_fingerprint_changes_when_new_file_appears(monkeypatch, tmp_path):
    from holodeck.dashboard import app as app_mod

    monkeypatch.setenv("HOLODECK_DASHBOARD_RESULTS_DIR", str(tmp_path))
    (tmp_path / "run-1.json").write_text("{}")
    fp1 = app_mod._results_dir_fingerprint()
    # mtime resolution on some filesystems is 1s; wait long enough to change.
    time.sleep(1.1)
    (tmp_path / "run-2.json").write_text("{}")
    fp2 = app_mod._results_dir_fingerprint()
    assert fp2 > fp1


@pytest.mark.unit
def test_get_runs_memo_invalidates_on_new_file(monkeypatch, tmp_path):
    from holodeck.dashboard import app as app_mod

    monkeypatch.setenv("HOLODECK_DASHBOARD_RESULTS_DIR", str(tmp_path))
    app_mod._runs_memo.clear()

    call_count = {"n": 0}
    real_loader = app_mod.load_runs_for_app

    def counting_loader():
        call_count["n"] += 1
        return real_loader()

    monkeypatch.setattr(app_mod, "load_runs_for_app", counting_loader)

    # Empty dir: one load, then cached.
    app_mod.get_runs()
    app_mod.get_runs()
    assert call_count["n"] == 1

    # New file → fingerprint changes → memo busts → loader runs again.
    time.sleep(1.1)
    (tmp_path / "run-1.json").write_text("{}")
    app_mod.get_runs()
    assert call_count["n"] == 2
