"""Pydantic models for persisted EvalRun artifacts.

An `EvalRun` wraps the existing `TestReport` and adds an `EvalRunMetadata`
block that captures a redacted snapshot of the agent configuration plus run
provenance (prompt version, holodeck version, sanitized CLI args, git
commit). One `EvalRun` is persisted per `holodeck test` invocation.

Models in this module are populated incrementally — see
`specs/031-eval-runs-dashboard/data-model.md`.
"""

from __future__ import annotations
