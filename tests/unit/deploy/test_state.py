"""Unit tests for deployment state helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from holodeck.deploy.state import (
    compute_config_hash,
    get_deployment_record,
    load_state,
    save_state,
    update_deployment_record,
)
from holodeck.lib.errors import DeploymentError
from holodeck.models.deployment import CloudProvider, DeploymentConfig
from holodeck.models.deployment_state import DeploymentRecord, DeploymentState


def _make_record() -> DeploymentRecord:
    return DeploymentRecord(
        provider=CloudProvider.AZURE,
        service_id="service-123",
        service_name="agent",
        url="https://example.com",
        status="RUNNING",
        image_uri="ghcr.io/holodeck/agent:abc1234",
        config_hash="sha256:deadbeef",
    )


class TestDeploymentStateIO:
    """Tests for DeploymentState read/write helpers."""

    def test_load_state_missing_returns_default(self, tmp_path: Path) -> None:
        """Missing state file returns default state."""
        state_path = tmp_path / "deployments.json"

        state = load_state(state_path)

        assert state.version == "1.0"
        assert state.deployments == {}

    def test_save_and_load_state_round_trip(self, tmp_path: Path) -> None:
        """Saving and loading state preserves records."""
        state_path = tmp_path / "deployments.json"
        record = _make_record()
        state = DeploymentState(deployments={"agent": record})

        save_state(state_path, state)
        loaded = load_state(state_path)

        assert loaded.version == "1.0"
        assert "agent" in loaded.deployments
        loaded_record = loaded.deployments["agent"]
        assert loaded_record.service_id == record.service_id
        assert loaded_record.url == record.url
        assert loaded_record.provider == record.provider

    def test_load_state_invalid_json_raises(self, tmp_path: Path) -> None:
        """Invalid JSON should raise DeploymentError."""
        state_path = tmp_path / "deployments.json"
        state_path.write_text("{invalid}", encoding="utf-8")

        with pytest.raises(DeploymentError, match="Invalid deployment state format"):
            load_state(state_path)

    def test_load_state_invalid_schema_raises(self, tmp_path: Path) -> None:
        """Invalid schema should raise DeploymentError."""
        state_path = tmp_path / "deployments.json"
        payload = {"version": "1.0", "deployments": []}
        state_path.write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(DeploymentError, match="Invalid deployment state format"):
            load_state(state_path)


class TestDeploymentRecordHelpers:
    """Tests for deployment record helper functions."""

    def test_update_deployment_record_sets_timestamps(self, tmp_path: Path) -> None:
        """Update helper sets created_at and updated_at timestamps."""
        state_path = tmp_path / "deployments.json"
        record = _make_record()

        updated = update_deployment_record(state_path, "agent", record)

        assert updated.created_at is not None
        assert updated.updated_at is not None

        updated_again = update_deployment_record(
            state_path,
            "agent",
            updated.model_copy(update={"status": "UPDATING"}),
        )

        assert updated_again.created_at == updated.created_at
        assert updated_again.updated_at >= updated.created_at

    def test_get_deployment_record_returns_none(self, tmp_path: Path) -> None:
        """Missing deployment record returns None."""
        state_path = tmp_path / "deployments.json"

        record = get_deployment_record(state_path, "missing")

        assert record is None

    def test_compute_config_hash_stable(self) -> None:
        """Config hash is deterministic for same config."""
        config = DeploymentConfig(
            registry={"url": "ghcr.io", "repository": "org/agent"},
            target={
                "provider": "azure",
                "azure": {
                    "subscription_id": "00000000-0000-0000-0000-000000000000",
                    "resource_group": "rg",
                },
            },
        )

        first = compute_config_hash(config)
        second = compute_config_hash(config)

        assert first == second
