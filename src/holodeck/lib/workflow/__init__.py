"""Deterministic-spine workflow engine (036).

A DAG of determination nodes: edge nodes (agents behind schema gates) feed
policy nodes (DMN decision tables + FEEL) feeding a human node. This package
holds the engine — FEEL evaluation, table evaluation, edge gating, and the
runner — while the artifacts are modeled in ``holodeck.models.workflow`` and
``holodeck.models.decision_table``.
"""
