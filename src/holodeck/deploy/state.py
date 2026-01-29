"""Deployment state tracking helpers."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError

from holodeck.lib.errors import DeploymentError
from holodeck.models.deployment import DeploymentConfig
from holodeck.models.deployment_state import DeploymentRecord, DeploymentState

STATE_VERSION = "1.0"


def get_state_path(agent_path: Path) -> Path:
    """Return the deployment state file path for an agent config."""
    return agent_path.parent / ".holodeck" / "deployments.json"


def compute_config_hash(config: DeploymentConfig) -> str:
    """Compute a deterministic hash for the deployment configuration."""
    payload = json.dumps(
        config.model_dump(mode="json", exclude_unset=True),
        sort_keys=True,
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def load_state(state_path: Path) -> DeploymentState:
    """Load deployment state data from disk."""
    if not state_path.exists():
        return DeploymentState(version=STATE_VERSION)

    try:
        content = state_path.read_text(encoding="utf-8")
        if not content.strip():
            return DeploymentState(version=STATE_VERSION)
    except OSError as exc:
        raise DeploymentError(
            operation="state",
            message=f"Failed to read deployment state at {state_path}: {exc}",
        ) from exc

    try:
        state = DeploymentState.model_validate_json(content)
    except ValidationError as exc:
        raise DeploymentError(
            operation="state",
            message=f"Invalid deployment state format in {state_path}: {exc}",
        ) from exc

    if not state.version:
        state = state.model_copy(update={"version": STATE_VERSION})
    return state


def save_state(state_path: Path, state: DeploymentState) -> None:
    """Persist deployment state data to disk."""
    try:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(state.model_dump(mode="json"), indent=2, sort_keys=True)
        state_path.write_text(payload, encoding="utf-8")
    except OSError as exc:
        raise DeploymentError(
            operation="state",
            message=f"Failed to write deployment state to {state_path}: {exc}",
        ) from exc


def get_deployment_record(state_path: Path, agent_name: str) -> DeploymentRecord | None:
    """Return a deployment record for a specific agent."""
    state = load_state(state_path)
    return state.deployments.get(agent_name)


def update_deployment_record(
    state_path: Path, agent_name: str, record: DeploymentRecord
) -> DeploymentRecord:
    """Update deployment record for an agent and persist it."""
    state = load_state(state_path)
    existing = state.deployments.get(agent_name)
    now = datetime.now(timezone.utc)

    created_at = record.created_at or (existing.created_at if existing else None) or now
    updated_record = record.model_copy(
        update={"created_at": created_at, "updated_at": now}
    )

    state.deployments[agent_name] = updated_record
    if not state.version:
        state = state.model_copy(update={"version": STATE_VERSION})
    save_state(state_path, state)
    return updated_record
