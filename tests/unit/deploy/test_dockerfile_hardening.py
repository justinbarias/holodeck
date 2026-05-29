"""Generated Dockerfile applies P2a hardening (spec 034)."""

from __future__ import annotations

import pytest

from holodeck.deploy.dockerfile import generate_dockerfile


def _gen(**overrides) -> str:
    defaults: dict = {
        "agent_name": "t",
        "port": 8080,
        "protocol": "rest",
        "instruction_files": ["instructions.md"],
        "data_directories": ["data/"],
    }
    defaults.update(overrides)
    return generate_dockerfile(**defaults)


@pytest.mark.unit
def test_data_directory_copied_as_root_and_chmod_a_minus_w():
    df = _gen()
    assert "COPY --chown=root:root data/" in df
    assert "chmod -R a-w /app/data" in df


@pytest.mark.unit
def test_instruction_files_copied_as_root_and_chmod_a_minus_w():
    df = _gen()
    assert "COPY --chown=root:root instructions.md" in df
    assert "chmod a-w /app/instructions.md" in df or "chmod -R a-w /app" in df


@pytest.mark.unit
def test_scratch_dir_created_writable_for_holodeck_user():
    df = _gen()
    assert "mkdir -p /var/holodeck/work" in df
    assert "chown holodeck:holodeck /var/holodeck/work" in df


@pytest.mark.unit
def test_nodejs_omitted_by_default():
    df = _gen(needs_nodejs=False)
    assert "nodesource" not in df
    assert "apt-get install -y --no-install-recommends nodejs" not in df


@pytest.mark.unit
def test_nodejs_included_when_flag_true():
    df = _gen(needs_nodejs=True)
    assert "nodesource" in df
