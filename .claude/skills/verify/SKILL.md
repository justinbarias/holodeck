---
name: verify
description: Run HoloDeck's tiered verify suite (make verify-fast / verify-deps / verify-smoke / verify-profiles / verify-live) and report evidence per tier. Use when the user says "verify", "run the verify suite", "is this ready", "ready to merge/ship", or asks to check a branch before a PR.
---

# Verify

Pick tiers from what changed, run them, and report evidence. Never claim an unrun tier passed.

## 1. Find the changes

```bash
git diff --name-only main...HEAD
git status --porcelain
```

Use the union of both lists.

## 2. Choose tiers

| Tier | Run when | Command |
| --- | --- | --- |
| fast | always | `make verify-fast` |
| deps | `pyproject.toml`, `uv.lock`, `.github/`, `docker/`, `security/`, `scripts/verify/audit_deps.py` or `sample/copilotkit/package*.json` changed | `make verify-deps` |
| smoke | `src/holodeck/serve`, `src/holodeck/lib/backends`, `src/holodeck/lib/test_runner`, `src/holodeck/cli`, `src/holodeck/chat`, `src/holodeck/agent`, `scripts/verify/`, `tests/fixtures/verify`, `pyproject.toml` or `uv.lock` changed (SDK bumps can break a backend that unit tests mock away) | `make verify-smoke` |
| profiles | `pyproject.toml`, `uv.lock`, an extra, a lazy optional import (`lib/extras.py`, `lib/retrieval/adapters`, `lib/evaluators`) or `scripts/verify/profiles.py` changed. Builds a wheel and installs about 17 profiles; downloads packages and takes several minutes. Not part of `make verify` | `make verify-profiles` (`PROFILE_ARGS="--profile base"` for one) |
| live | backends, serve, tools or `tests/fixtures/verify` changed **and** `CLAUDE_CODE_OAUTH_TOKEN` or `OPENAI_API_KEY` is set (shell or `.verify/live.env`). Costs real model calls: run only when the user asks or approves | `make verify-live` |

- Check credentials with `[ -n "$CLAUDE_CODE_OAUTH_TOKEN" ]`, `[ -n "$OPENAI_API_KEY" ]` and `[ -f .verify/live.env ]`. Never print their values.
- Live starts Docker containers for Qdrant, Chroma, Postgres (pgvector), Weaviate, the Cosmos DB emulator and the Aspire dashboard. Without Docker, or with `LIVE_ARGS=--no-infra`, local vector store and trace checks SKIP. Pinecone (`PINECONE_API_KEY`) and Azure AI Search (`AZURE_AI_SEARCH_ENDPOINT`, `AZURE_AI_SEARCH_API_KEY`) have no emulator: they run only with those credentials and report SKIP with the reason otherwise.
- When only vector store tools or adapters changed, `make verify-providers` runs just the provider matrix (one case per provider, no AG-UI turns). Each provider row also proves the provider held the run's collection, so an in-memory fallback fails.
- Live is the AGENTS.md live validation loop without the CopilotKit browser step. Exit 2 means every backend was skipped: report it as SKIPPED, not passed.
- Run cheaper tiers first. If a tier fails, stop, report the failure, and do not run later tiers unless the user asks.

## 3. Report

For each tier give: ran / skipped (and why), PASS or FAIL, and evidence:

- fast: the pytest summary line and any failing gate.
- deps: the `pip-audit:` summary, ignored advisories, npm audit result.
- smoke: the pass/fail table and the `.verify/smoke/<ts>/` directory.
- profiles: the per-profile summary table (packages, site-packages size, PASS or FAIL) and any failing check detail.
- live: the per-case table (SKIP rows with reasons, including the Pinecone and Azure AI Search rows when their credentials are not set), the `.verify/live/<ts>/` directory, and the Aspire URL for the user's trace review (the script does not check traces).

Say which tiers were not run. Detail lives in `docs/RELIABILITY.md` ("Verify suite").
