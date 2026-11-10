# Reliability

Owner: runtime and deployment maintainers. Review this document when lifecycle or deployment behavior changes.

## Runtime contracts

Use the backend lifecycle from [base.py](../src/holodeck/lib/backends/base.py).
Preserve initialization, session preparation, cancellation, streaming, and teardown behavior.
Use async clients or thread boundaries for blocking work.
For timeout, retry, or session changes, exercise the affected lifecycle tests.

The [server](../src/holodeck/serve/server.py) provides `/health`, `/health/agent`, AG-UI `/awp`, and REST endpoints.
Health probes establish readiness. A representative request establishes useful agent behavior.
Temporal retries must preserve the distinction between retryable model failures and configuration errors.
See [retry classification tests](../tests/unit/temporal/test_retry_classification.py).

## Local reproduction and evidence

1. Select a committed fixture or a minimal agent configuration that reproduces the problem.
2. Use separate result directories, ports, and container names for concurrent worktrees.
3. Capture the command, configuration identity, response, and relevant redacted logs or traces.
4. Exercise the failed behavior again after the change.
5. Record the observed result and any unavailable external dependency in the execution plan.

Use the [observability guide](guides/observability.md) for configured OpenTelemetry exporters and GenAI signals.
Do not assume a collector exists in every checkout.
The repository does not provision the article's per-worktree observability stack automatically.
That gap is tracked in [technical debt](exec-plans/tech-debt-tracker.md).

## Verify suite

`make verify` runs the three offline tiers. Run `make verify-profiles` when packaging changes and `make verify-live` when you have credentials; neither is part of `make verify`.
The `/verify` skill ([SKILL.md](../.claude/skills/verify/SKILL.md)) selects tiers from the changed files.

| Tier | Command | Proves | Credentials |
| --- | --- | --- | --- |
| all offline | `make verify` | The three offline tiers below (fast, deps, smoke) in one run. It does not run profiles, live or providers | None |
| fast | `make verify-fast` | Harness, lock, Ruff, Black, MyPy, schema and image-constraints checks pass, and the full pytest suite passes | None |
| deps | `make verify-deps` | The lock is current. No unreviewed pip-audit advisory exists. npm audit of `sample/copilotkit` passes (skipped when that sample is absent) | None (network to the advisory databases) |
| smoke | `make verify-smoke` | Both backends complete real `holodeck test run` and `holodeck serve` AG-UI turns, including a tool call and thread isolation | None |
| profiles | `make verify-profiles` | The built wheel installs and works in a clean venv for each extras profile, and unselected packages are absent (details below) | None (network to the package index) |
| live | `make verify-live` | Every tool type each backend supports, a vector store case on every supported provider, plus serve AG-UI turns, against real providers; traces in a local Aspire dashboard | `CLAUDE_CODE_OAUTH_TOKEN` (Claude), `OPENAI_API_KEY` (OpenAI, and embeddings for both); Docker for the vector stores and Aspire; optional `PINECONE_API_KEY` and `AZURE_AI_SEARCH_ENDPOINT` + `AZURE_AI_SEARCH_API_KEY` |
| providers | `make verify-providers` | Only the vector store matrix of the live tier: the provider table below, with no AG-UI turns | Same as live |

**deps.** [audit_deps.py](../scripts/verify/audit_deps.py) audits every locked package (all extras and groups).
Accepted advisories live in [audit-ignores.toml](../security/audit-ignores.toml).
Each entry has `id`, `package`, `reason`, `owner` and `expires`. An `id` can be the advisory id or any alias.
The audit fails on an unignored advisory, an expired entry, or an entry that matches no current advisory.

**smoke.** [smoke.py](../scripts/verify/smoke.py) starts [fake_provider.py](../scripts/verify/fake_provider.py).
That process implements the OpenAI Responses and Anthropic Messages APIs on a local port.
The script points the [verify fixtures](../tests/fixtures/verify/smoke-openai.yaml) at it with a scrubbed environment and fake keys.
For each backend, it runs `holodeck test run`, then 2 threads x 2 turns over `POST /awp`.
It asserts `RUN_STARTED` first, `RUN_FINISHED` last, no `RUN_ERROR`, scripted text, and `TOOL_CALL_*` events on the tool turn.
It also asserts that no request for thread B carries thread A's text.
Artifacts, including the provider request log, go to `.verify/smoke/<UTC timestamp>/`.
On Linux the Claude backend needs `bubblewrap` (`bwrap`): the Claude Code CLI will not start without it while HoloDeck's default subprocess env scrubbing is on. Smoke fails fast with an install hint when it is missing.
CI runs this tier on Python 3.11. A weekly [scheduled workflow](../.github/workflows/scheduled-verify.yml) runs `verify-deps` and `verify-smoke`.

**profiles.** [profiles.py](../scripts/verify/profiles.py) builds the wheel once, then for each profile creates a throwaway venv on Python 3.11 (the lowest supported; `--python` overrides it) and installs the wheel with that profile's extras.
The versions come from `uv.lock` (`uv export` as constraints, plus the `[tool.uv]` overrides), so a run is reproducible and a new upstream release cannot fail it.
Checks run inside the venv from an empty directory ([profile_probe.py](../scripts/verify/profile_probe.py) and `holodeck` processes), so the source tree cannot be imported by accident.
Each profile asserts that the modules its extras provide import, and that every other feature package is absent by import and by installed distribution name. `semantic-kernel`, `azure-ai-evaluation` and `evaluate` are absent in every profile, `all` included, and `uv pip check` reports no broken requirements (the `pytz` override aside).

| Profile | Extras | Feature checks |
| --- | --- | --- |
| `base` | none | `--version`, `--help` and every command's `--help`; `holodeck init` from the packaged templates; the config loader; an in-memory `vectorstore` tool round trip with placeholder embeddings; each missing store client names its extra; a DeepEval metric under `holodeck test run`, `test optimize` and `test view` each exit non-zero with the install command and no traceback |
| `eval-nlp` | `eval`, `nlp` | DeepEval evaluators and judge models construct (no model call); BLEU and ROUGE score offline; METEOR scores or names the NLTK download command; the executor builds every metric |
| `optimizer`, `opensearch` | the extra | An Optuna study suggests a value; the OpenSearch index builds against an unreachable host |
| `openai-agents`, `dashboard`, `temporal`, `deploy` | `openai-agents`, `dashboard`, `temporal`, `deploy-all` | Both backends import; the dashboard app and assets, and `test view --help`; `holodeck worker --help` and the Temporal modules; the deploy modules and `deploy --help` |
| one per store | `qdrant`, `chromadb`, `postgres`, `pinecone`, `weaviate`, `azure-ai-search`, `azure-cosmos` | `create_adapter` builds the store's adapter and never raises `MissingClientError`; Qdrant (`:memory:`) and Chroma (a temp directory) also run a real upsert and vector search; every other store client stays absent |
| `all` | `all` | Every module in the wheel imports, with every extra present |

`--profile NAME` (repeatable) and `--group base|features|stores|all` choose profiles; `--keep` leaves the venvs for debugging.
Run it after any change to `pyproject.toml`, `uv.lock`, the extras or a lazy import. It downloads and unpacks hundreds of packages per profile, so it is not in `make verify`.
CI runs it as the `profiles` job of [profiles.yml](../.github/workflows/profiles.yml): four matrix groups on GitHub-hosted runners (so it never queues behind the self-hosted `test` lane) with Python 3.11 and the uv cache. It runs on pull requests that change `pyproject.toml`, `uv.lock`, `.gitignore`, `src/` or the verifier, and on every push to `main`.
A new extra needs a row in `EXTRA_MODULES`; `tests/unit/test_verify_profiles.py` fails without one.

**live.** [live.py](../scripts/verify/live.py) runs the [live validation loop](../AGENTS.md) without the CopilotKit browser step.
It runs each backend whose credential is set. Put credentials in `.verify/live.env` (gitignored; copy [live.env.example](../tests/fixtures/verify/live.env.example)). Shell variables win.
Claude uses [live-full-claude.yaml](../tests/fixtures/verify/live-full-claude.yaml) and OpenAI uses [live-full-openai.yaml](../tests/fixtures/verify/live-full-openai.yaml).

| Tool type | Claude | OpenAI |
| --- | --- | --- |
| `function` (`smoke_echo`, `leak_probe`) | yes | yes |
| `mcp` stdio (local FastMCP server, started with `uvx`) | yes | yes |
| `vectorstore` on every provider in the table below, `hierarchical_document` on Qdrant, PostgreSQL and Weaviate | needs `OPENAI_API_KEY` | yes |
| subagents (`claude.agents` / `openai.agents` handoff) | yes | yes |
| `skill` | no: the Claude backend does not adapt skills yet | yes |
| `prompt` | no: no runtime adapter on any backend | no |

Each tool type has one multi-turn test case with `expected_tools` and a `type: code` grader that checks the tool result or reply for a fixed marker. No case uses an LLM judge.
The script reports PASS, FAIL or SKIP for each test case from the results JSON under the run copy's `results/`.
Without `OPENAI_API_KEY` (embeddings), it removes every vector tool from the run copy. Without a provider's container or credentials, it removes that provider's tools. Each removed tool's cases are reported as SKIP with the reason, never PASS. The other cases still run.
Then it starts `holodeck serve` and sends 2 threads x 2 turns. Thread A turn 2 must emit `TOOL_CALL_START` for `smoke_echo`. Thread B turn 2 must contain B's code word and not A's.
It also fails when a credential value occurs in any file of the run directory.
Each run uses an empty `HOME` and `CLAUDE_CONFIG_DIR`, so `~/.holodeck/.env` and a stored Claude login cannot replace the credential under test.

Vector store providers. Each has one `vectorstore` tool and case per backend, with its own small corpus under [rag/](../tests/fixtures/verify/rag) and a marker string the answer must contain. `hierarchical_document` runs on three providers over the same handbook corpus, each with its own tool name and collection.

| Provider | Infrastructure | What the case proves |
| --- | --- | --- |
| `qdrant` | container `holodeck-verify-qdrant` | Document chunks through the native adapter; `hierarchical_document` with Qdrant vector search plus the in-process BM25 leg |
| `chromadb` | container `holodeck-verify-chroma` (HTTP client) | Document chunks through the server client |
| `postgres` | container `holodeck-verify-postgres` (pgvector) | Document chunks in a pgvector table; `hierarchical_document` through the BM25 fallback, since the adapter has no hybrid search |
| `weaviate` | container `holodeck-verify-weaviate` (HTTP 8089, gRPC 50059) | Document chunks through the native adapter; `hierarchical_document` through Weaviate's native hybrid search |
| `azure-cosmos-nosql` | container `holodeck-verify-cosmos`, the Linux emulator (HTTP, published key). It accepts only the `diskANN` vector index and has no full-text search, so the tool sets `vector_index: diskANN` | Document chunks in a Cosmos container with a vector policy |
| `pinecone` | Credential-gated: `PINECONE_API_KEY` (serverless index, `aws` / `us-east-1`). No local emulator: Pinecone Local is end-of-life and rejects the v10 control plane | Document chunks in a hosted serverless index |
| `azure-ai-search` | Credential-gated: `AZURE_AI_SEARCH_ENDPOINT` and `AZURE_AI_SEARCH_API_KEY`. Azure has no emulator | Document chunks in a hosted search index |

The agent YAML is static, so the hosted providers' tools stay in the fixtures. The script drops a provider's tools and cases when its prerequisite is missing, and the SKIP row names the missing variable or the unavailable container.
A tool falls back to in-memory storage when its provider cannot be reached (a missing client library or a retired provider name is an error, not a fallback), and a case could then still pass. So each tool gets a collection name unique to the run (`verify_<backend><id>_<tool>`).
After the cases and the AG-UI turns, the script creates the provider's adapter, asks for that collection, and reports `provider <name> (<tool>) holds records`. The row is FAIL when the collection is missing or empty.
Then it deletes every run collection, so a hosted service keeps no data. A failed delete is a FAIL row named `cleanup`.
Pinecone reads can lag a write, so the count is polled for up to 30 seconds.

Docker containers (left running after the run; `--no-infra` skips them, and local vector store and trace checks then SKIP; hosted providers still run when their credentials are set):

| Container | Image | Host ports |
| --- | --- | --- |
| `holodeck-verify-qdrant` | `qdrant/qdrant:v1.17.1` | 6333 |
| `holodeck-verify-chroma` | `chromadb/chroma:1.5.9` (matches the locked client) | 8765 |
| `holodeck-verify-postgres` | `pgvector/pgvector:pg17` (throwaway credentials, loopback only) | 54329 |
| `holodeck-verify-weaviate` | `cr.weaviate.io/semitechnologies/weaviate:1.40.4` (anonymous, no vectorizer) | 8089 HTTP, 50059 gRPC |
| `holodeck-verify-cosmos` | `mcr.microsoft.com/cosmosdb/linux/azure-cosmos-emulator:vnext-preview` (`--protocol http`; first start takes about a minute) | 8081 gateway, 1234 data explorer |
| `holodeck-verify-aspire` | `mcr.microsoft.com/dotnet/aspire-dashboard:13.5.2` (anonymous) | 18888 UI, 4317 OTLP/gRPC, 4318 OTLP/HTTP |

If another service of the same kind already serves those ports, the script reuses it and does not stop it.
Stop the managed containers with `docker rm -f holodeck-verify-qdrant holodeck-verify-chroma holodeck-verify-postgres holodeck-verify-weaviate holodeck-verify-cosmos holodeck-verify-aspire`.
Traces are for human review at <http://127.0.0.1:18888/traces>. Look for the services `holodeck-verify-live-claude` and `holodeck-verify-live-openai`.
The `leak_probe` tool returns a fake `ghp_` token. Its span must show `[REDACTED:github-token]`, never the token.

Pass flags through `LIVE_ARGS`, for example `make verify-live LIVE_ARGS="--backend claude --no-infra"`. `--limit N` runs only the first N test cases.
`--only providers` (or `make verify-providers`) runs only the vector store cases: it keeps the test cases for vector tools and skips the AG-UI turns.
Cost: each backend makes about 20 small model calls (`claude-haiku-5-5`, `gpt-6-luna`) plus a few embedding calls; hosted providers add a serverless index or search index that the run creates and deletes.
Artifacts (SSE transcripts, logs, the run copy of the agent) go to `.verify/live/<UTC timestamp>/`.
Exit code 2 means every backend was skipped. Report that as skipped, not passed.
A backend with its credential set and no passing case or turn fails.

## End-to-end deployment validation

Run this loop only when the user explicitly requests deployment validation or authorizes the target deployment.
This loop publishes images and changes a live service.
Reuse authorization already supplied in the conversation.
The [deployment guide](guides/deployment.md) describes supported configuration and CLI options.

The former local sample, `sample/financial-assistant/claude`, is git-ignored and may not exist in another checkout.
Use the user-selected agent, registry, and target. Do not assume a personal cloud endpoint.

1. Build the working-tree wheel with `uv build --wheel`. Keep exactly the intended wheel in the Docker build context.
2. Build [docker/Dockerfile.local](../docker/Dockerfile.local) with that wheel, using `--no-cache` and the target platform. Give the image a name that is not a published tag, for example `holodeck-base-local:test`.
3. Inspect the installed wheel version inside the image.
4. Build the agent image with `holodeck deploy build --wheel dist/<wheel> --base-image <local base> --no-pull`. Without `--no-pull`, [builder.py](../src/holodeck/deploy/builder.py) passes `pull=True` to Docker.
5. Publish the verified tag to the authorized registry.
6. Deploy with `holodeck deploy run` for the selected agent configuration.
7. Wait for health with a bounded timeout, then send a deterministic request with a known expected answer.
8. Record the image digest, target, response, and relevant telemetry. Keep the build manifest (`.holodeck/builds/<tag>.json`).

By default the build pulls the base image, which can replace a locally tagged base image with its published registry version. Use `--no-pull` only for a base image you built locally.
Azure Container Apps requires a compatible target image, commonly `linux/amd64` for this repository's deployment loop.
Readiness and cold-start timings vary. Measure them instead of relying on the old sample's timings.
Local image builds (steps 1 to 4) need no registry access and are safe to run. Pushing images and `holodeck deploy run` need the authorization described above.

### Image pins and manifest

Every image installs `holodeck-ai[extras]==X.Y.Z` with `--constraint` and `--override` files shipped in the wheel (`src/holodeck/deploy/constraints.txt` and `overrides.txt`, generated from `uv.lock`). Regenerate them with `make constraints`; `make constraints-check` fails when they are stale and runs in `make verify-fast`. A build that uses `--wheel` takes the files from that wheel. The Python base image is pinned by digest; update the digest on purpose.
`holodeck deploy build` writes `.holodeck/builds/<tag>.json` (HoloDeck version, extras, base image reference and digest or image ID, built image ID, package list), and every image keeps its package list in `/opt/holodeck/manifest.txt`. Serving images carry only the extras the agent needs; they never carry `eval`, `nlp` or `optimizer`. See the [deployment guide](guides/deployment.md#reproducible-images).
