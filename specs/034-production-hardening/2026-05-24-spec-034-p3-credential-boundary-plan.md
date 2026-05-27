# Spec 034 Phase 3 — Credential Boundary Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land spec 034 Phase 3 — move LLM-provider credentials, embedding-provider credentials, and tool-bound credentials out of the agent container's process environment and behind an Envoy sidecar that holds them, enforces a domain allowlist derived from `agent.yaml`, and terminates the credential injection step on the sidecar side. Gated behind opt-in `deployment.security_profile: hardened`.

**Architecture summary:** One new top-level field (`deployment.security_profile`). One new generator module (`envoy.py`) that emits an Envoy `bootstrap.yaml`. One deployer extension (Azure Container Apps two-container revision with ACA Secrets mounted only to the sidecar). One backend guard (`ANTHROPIC_BASE_URL` required when running under hardened profile; agent fails-closed if any credential-bearing env var is observed inside its own process). No changes to the YAML hook surface, no changes to the SDK, no changes to vector-store or embedding tool implementations beyond their env-var read points.

**Tech stack:** Python 3.10+, Pydantic v2, `azure.mgmt.appcontainers` models (Secret, ContainerAppSecret, Volume(SECRET), VolumeMount, ManagedServiceIdentity), `envoyproxy/envoy:v1.31-latest` (sidecar image), pytest `-n auto`.

**Phase split:** P3 ships as **one sequential PR train**, not parallel sub-phases. The Envoy generator must be in place before the ACA deployer can reference it; the deployer must be in place before the backend can require `ANTHROPIC_BASE_URL`; the backend guard must be in place before the credential-leak test can pass. P3 is smaller in surface area than P2 but deeper in invariants — every task is load-bearing for the next.

---

## Coverage check — what P3 actually closes

Cross-referenced against [Anthropic secure-deployment doc](https://code.claude.com/docs/en/agent-sdk/secure-deployment) §"Proxy pattern" and §"Credential injection":

| Anthropic recommendation | P3 task |
|---|---|
| Credentials not in agent container env | Task 6, Task 9 (deployer moves creds to sidecar-only secret) + Task 10 (backend boot guard) |
| `ANTHROPIC_BASE_URL` → localhost proxy | Task 7 (deployer sets agent env) + Task 8 (backend requires it under hardened profile) |
| Domain allowlist on egress | Task 3, Task 4 (Envoy `route_config` derived from agent.yaml) |
| `credential_injector` HTTP filter for adding `Authorization` headers | Task 4 (Envoy Lua filter injects per-route headers) |
| Sidecar holds credentials, agent has none | Task 6 (ACA Secret mounted only to sidecar volume, not agent) |
| TLS-terminating proxy for arbitrary HTTPS | **Out of v1** — documented in Task 15; current scope is the Anthropic API endpoint + the embedding-provider endpoint + declared HTTP-MCP servers (the three endpoints HoloDeck routes today) |
| Custom-tool pattern for non-Anthropic creds | Already supported via existing MCP surface; documented in Task 15 |
| Block non-allowlisted domains | Task 4 (route catch-all → 403 with structured `application/problem+json`) |

**Gaps the plan does not close (documented in Task 15):**

1. **AWS Bedrock / GCP Vertex / Anthropic Foundry credential routing.** SigV4 (AWS) and GCP-auth (Vertex) sign requests with credentials baked into the AWS/GCP SDK client *before* the HTTP layer. A localhost proxy that just forwards the request would either need to (a) re-sign with its own credentials, or (b) be paired with cloud-native workload identity (IRSA/Workload Identity/Azure Managed Identity) so the agent's SDK pulls credentials from the metadata service inside the sidecar's lane. ACA does not run on EKS/GKE, so options (a) and (b) both require new work. **P3 v1 rejects `security_profile: hardened` for `auth_provider in {bedrock, vertex, foundry}` with a clear error pointing at the follow-up spec.** Direct Anthropic API (`api_key`, `oauth_token`, `custom`) is fully supported.
2. **TLS interception for arbitrary HTTPS.** Envoy can terminate TLS only with a CA cert injected into the agent's trust store; we don't ship that in v1. Means: HTTPS-MCP servers with bearer-token auth in headers go through opaque tunnels — the sidecar enforces the destination domain but cannot inject the bearer token. Operators who need credential injection for non-Anthropic HTTPS endpoints use the custom-tool pattern.
3. **Per-tool credential mapping in the default profile.** Default profile keeps env-var model. Operators who need credential isolation opt into `hardened`.

These omissions are stated by spec 034 §"What we are deliberately *not* doing in v1" — P3 enforces the scope; it does not narrow or widen it.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│ Azure Container App revision (security_profile: hardened)            │
│                                                                       │
│  ┌─────────────────────┐                ┌──────────────────────────┐ │
│  │ agent container      │                │ envoy sidecar            │ │
│  │  • holodeck serve    │  HTTP/local    │  • bootstrap.yaml        │ │
│  │  • no creds in env   │ ──host loop──► │    (config map mount)    │ │
│  │  • ANTHROPIC_BASE_URL│ http://        │  • Secrets mounted ONLY  │ │
│  │    = http://         │ localhost:7000 │    here                  │ │
│  │    localhost:7000    │                │  • Lua filter injects    │ │
│  │  • HOLODECK_         │                │    `Authorization`       │ │
│  │    HARDENED=1        │                │  • Domain allowlist      │ │
│  │  • probes /health    │                │    → upstream            │ │
│  │    /ready as before  │                │  • Everything else:      │ │
│  └─────────────────────┘                │    403 + problem+json    │ │
│                                          └────────────┬─────────────┘ │
│                                                       │               │
│  ┌────────────────────────────────────────────────────┘               │
│  │ ACA volumes:                                                       │
│  │   • envoy-config (config-map style, plaintext bootstrap.yaml)      │
│  │   • envoy-secrets (Secret-mounted: creds.env, file mode 0400)      │
│  │   mounted ONLY on the envoy container, NOT the agent container     │
│  └────────────────────────────────────────────────────────────────────┘
└──────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼ (egress)
                        api.anthropic.com  ← Authorization header injected
                        <embed>.openai.azure.com  ← api-key header injected
                        <mcp-http-server>.example  ← TLS pass-through, no inject
                        (everything else: rejected at Envoy)
```

**Credential flow (api_key path, end-to-end):**

1. Operator sets `ANTHROPIC_API_KEY` and `AZURE_OPENAI_API_KEY` on the host running `holodeck deploy run`.
2. Deployer reads them once, creates ACA `Secrets` (`anthropic-api-key`, `azure-openai-api-key`), discards in-memory.
3. Deployer creates ACA `Volume(storage_type=SECRET)` named `envoy-secrets`, projecting the secret values as files under `/var/run/secrets/envoy/`.
4. Deployer mounts `envoy-secrets` **only** on the envoy container.
5. Deployer mounts the Envoy bootstrap (with the allowlist baked in) as a `Volume(storage_type=SECRET)` projecting `bootstrap.yaml` (kept in a secret so the operator can rotate without rebuilding the image).
6. Deployer sets `ANTHROPIC_BASE_URL=http://localhost:7000` and `HOLODECK_HARDENED=1` on the agent container. **No credential env vars on the agent container.**
7. On startup, the agent's `claude_backend.py` reads `HOLODECK_HARDENED=1` and asserts: (a) `ANTHROPIC_BASE_URL` is set, (b) no `ANTHROPIC_API_KEY` / `CLAUDE_CODE_OAUTH_TOKEN` / `AWS_*` / `AZURE_OPENAI_API_KEY` / `QDRANT_API_KEY` is present in `os.environ`. Failure on either → fail closed.
8. Agent issues a request to `http://localhost:7000/v1/messages`.
9. Envoy matches the path against the route table, matches the upstream `api.anthropic.com`, runs the credential-injector Lua filter which reads `/var/run/secrets/envoy/anthropic_api_key` and adds `x-api-key: <value>`.
10. Envoy forwards the request to `api.anthropic.com` over plain TLS.

**Why the bootstrap is in a secret, not a config-map style mount.** ACA does not expose Kubernetes `ConfigMap`. The only file-projection primitive ACA offers is a `Volume(storage_type=SECRET)` projecting `ContainerAppSecret` values. Treating the bootstrap as a secret is a minor abuse — the bootstrap isn't sensitive — but it gives us atomic file projection without rebuilding the sidecar image on every allowlist edit. Acceptable; the alternative is baking the bootstrap into a per-agent envoy image, which is much heavier.

---

## Credential surface — concrete inventory

Before the task list, the full set of credentials P3 must displace.

### LLM-provider credentials (read in `src/holodeck/lib/backends/validators.py:146-234`)

| auth_provider | Env vars on the SDK subprocess today | Endpoint hit | P3 disposition |
|---|---|---|---|
| `api_key` | `ANTHROPIC_API_KEY` | `api.anthropic.com` | Moved to sidecar. Envoy injects `x-api-key`. |
| `oauth_token` | `CLAUDE_CODE_OAUTH_TOKEN` | claude.ai backend (`api.anthropic.com`) | Moved to sidecar. Envoy injects `authorization: Bearer …` (header name confirmed by SDK source — Task 2 verification step). |
| `custom` | `ANTHROPIC_AUTH_TOKEN` + `ANTHROPIC_BASE_URL` | Operator-supplied | Token moved to sidecar; `ANTHROPIC_BASE_URL` re-pointed at localhost. Sidecar's upstream is the operator's original endpoint, derived from `agent.model.endpoint`. |
| `bedrock` | `CLAUDE_CODE_USE_BEDROCK=1`, `AWS_REGION` (+ AWS access key/secret from outside HoloDeck) | `bedrock-runtime.<region>.amazonaws.com` | **Rejected by hardened profile in v1.** SigV4 signing happens inside the AWS SDK client; not transparently proxiable. Task 8 emits a load-time error. |
| `vertex` | `CLAUDE_CODE_USE_VERTEX=1`, `CLOUD_ML_REGION`, `ANTHROPIC_VERTEX_PROJECT_ID` (+ ADC credentials) | `<region>.aiplatform.googleapis.com` | **Rejected by hardened profile in v1.** Same reason. Task 8 emits a load-time error. |
| `foundry` | `CLAUDE_CODE_USE_FOUNDRY=1`, `ANTHROPIC_FOUNDRY_RESOURCE` or `ANTHROPIC_FOUNDRY_BASE_URL` | Operator-supplied Foundry endpoint | **Rejected by hardened profile in v1.** Foundry credential context is opaque. Task 8 emits a load-time error. |

### Embedding-provider credentials (read by Semantic Kernel `AgentFactory`, injected via `set_embedding_service` on vectorstore tools)

| Provider | Env vars on the serve process today | Endpoint hit | P3 disposition |
|---|---|---|---|
| `openai` (embeddings) | `OPENAI_API_KEY` | `api.openai.com/v1/embeddings` | Moved to sidecar. Envoy injects `authorization: Bearer …`. Upstream domain added to allowlist. |
| `azure_openai` (embeddings) | `AZURE_OPENAI_API_KEY` + endpoint from agent.yaml | `<resource>.openai.azure.com/openai/...` | Moved to sidecar. Envoy injects `api-key: …`. Upstream domain (`<resource>.openai.azure.com`) derived from `agent.embedding_provider.endpoint` and added to allowlist. |

Embedding tool implementations read these via the SK service, which makes the same HTTP call we need to intercept. The intercept happens at the SK service's base URL — we set the SK service's base URL to `http://localhost:7000` and let Envoy route by `:authority` / path. **Concrete code path to investigate in Task 5:** how the SK service base URL is configured at construction time. If the SK service does not respect a base-URL override, fall back to setting `OPENAI_BASE_URL` / `AZURE_OPENAI_ENDPOINT` env vars (Task 5 spike).

### Vector-store credentials

| Backend | Where credentials live today | P3 disposition |
|---|---|---|
| Qdrant Cloud | `connection_string` field on `agent.tools[*].database` (SecretStr in YAML) — embedded URL with API key | Sidecar terminates; Envoy injects `api-key` header on `https://<cluster>.qdrant.tech`. Connection string is *parsed* in `_resolve_database_config()` (base_tool.py:83) but the parsed URL+key are then handed to the Qdrant client which makes the actual HTTPS call. **Task 5 spike**: confirm Qdrant Python client supports a base-URL override that points at localhost; if not, document Qdrant as out-of-scope for hardened-profile v1. |
| Pinecone | `api_key` in connection params | Same risk as Qdrant — client may not respect a generic proxy. **Task 5 spike.** |
| PostgreSQL (pgvector) | `postgresql://user:pass@host/db` | Out of scope. P3 is HTTP-only; pgvector uses the postgres protocol. Document in Task 15. |
| Azure AI Search | `connection_string` with admin key | Sidecar terminates; Envoy injects `api-key` header. Same SDK-override question as Qdrant — Task 5 spike. |
| ChromaDB | `connection_string` (HTTP URL) or `persist_directory` (local) | Local: no creds, no proxy needed. Remote HTTP: Task 5 spike. |
| In-memory | None | N/A |
| OpenSearch | `connection_string` w/ user:pass | Same shape as pgvector if not HTTP; if HTTP, same shape as Qdrant. Task 5 spike. |

**v1 ground rule:** P3 supports vector-store backends only where (a) the wire protocol is HTTPS and (b) the Python client lets us override the base URL. Backends that fail either test are explicitly rejected in `hardened` mode with a load-time error pointing the operator at the default profile or at the custom-tool pattern. Spec 034 §"What … does *not* fix" already permits this scope.

### MCP-server credentials

| Transport | How creds reach today | P3 disposition |
|---|---|---|
| `stdio` (the only one HoloDeck routes today — see `mcp_bridge.py:51-95`) | `tool.env` + `tool.env_file`, passed to subprocess via `McpStdioServerConfig.env` | **Unchanged.** Stdio MCPs run inside the agent container; their HTTPS egress (e.g. a Brave Search MCP calling `api.search.brave.com`) goes through the agent's HTTP stack. To route them through the sidecar: Task 7 sets `HTTPS_PROXY=http://localhost:7000` on the agent container, so any well-behaved Python/Node MCP that respects proxy env vars uses Envoy. The MCP's own API keys still live in its declared env (no way to move them to the sidecar without per-MCP custom filters). **This is documented as a known boundary in Task 15:** the SDK subprocess (Claude Code CLI) and its stdio MCP children inherit the agent's env, so their creds are inside the agent's blast radius. The sidecar protects the *destination* of their egress but not their per-MCP API keys. |
| `http` / `sse` MCP | `headers` dict on the tool config (already in agent.yaml). Today `build_claude_mcp_configs` logs a warning and skips them. | Out of scope in P3. Document. |

### OTEL credentials

`OTEL_EXPORTER_OTLP_ENDPOINT` is typically internal (Aspire dashboard, OTLP collector inside the same cluster) and not a credential. Optional `OTEL_EXPORTER_OTLP_HEADERS` could carry an auth token; P3 leaves it on the agent container — it's a telemetry path, not an LLM credential path, and routing it through Envoy adds a moving part with no clear win. Documented in Task 15.

---

## File map

**New files:**

- `src/holodeck/deploy/envoy.py` — `build_envoy_bootstrap(agent: Agent) -> str` that returns a YAML string. Pure function; no I/O. Tested without a real Envoy.
- `src/holodeck/deploy/sidecar.py` — `build_sidecar_container(agent: Agent, envoy_secret_name: str, creds_secret_name: str)` returning an ACA `Container` model. Pulled out of the deployer so the deployer stays focused on orchestration.
- `tests/unit/deploy/test_envoy_generator.py`
- `tests/unit/deploy/test_aca_hardened_profile.py`
- `tests/unit/lib/backends/test_hardened_boot_guard.py`
- `tests/integration/security/test_hardened_profile_e2e.py` (live-ACA, mark `@pytest.mark.slow`)
- `docs/security/hardened-profile.md` (operator-facing)
- `sample/financial-assistant/claude/agent.hardened.yaml` (a copy of the existing sample with `security_profile: hardened`, used by the integration test)

**Modified files:**

- `src/holodeck/models/deployment.py` — add `SecurityProfile` enum (`DEFAULT`, `HARDENED`); add `security_profile: SecurityProfile = DEFAULT` on `DeploymentConfig`.
- `src/holodeck/lib/backends/validators.py` — add `validate_hardened_profile_compatibility(agent)`: raise `ConfigError` if `security_profile: hardened` and `auth_provider in {bedrock, vertex, foundry}`; raise if vector-store backend not in `_HARDENED_SUPPORTED_BACKENDS`; raise if any required allowlist endpoint missing.
- `src/holodeck/lib/backends/claude_backend.py` — `_apply_hardened_boot_guard()` called at the top of `build_options()` when `HOLODECK_HARDENED=1`: assert no credential env vars in `os.environ`; assert `ANTHROPIC_BASE_URL` is set; assert `ANTHROPIC_BASE_URL` points at localhost.
- `src/holodeck/deploy/deployers/azure_containerapps.py` — accept a new `sidecars: list[Container] | None` and `secrets: list[Secret] | None` kwarg; when set, emit two containers + the secret-mounted volumes.
- `src/holodeck/deploy/builder.py` (or the equivalent in `cli/commands/deploy.py`) — when `security_profile: hardened`, build the sidecar container + secrets list and pass them through.
- `src/holodeck/cli/commands/deploy.py` and `serve.py` — echo the hardened-profile resolution block per spec 034 §"CLI surface".
- `schemas/agent.schema.json` — regenerate via Pydantic dump to surface the new `security_profile` field.

---

## Tasks

### Task 1: Add `SecurityProfile` enum and `security_profile` field

**Files:**
- Modify: `src/holodeck/models/deployment.py`
- Test: `tests/unit/models/test_deployment.py` (append; add the file if it doesn't exist)
- Modify: `schemas/agent.schema.json` (regenerate after the model change)

**Acceptance criteria:**
- [ ] `SecurityProfile` enum with values `DEFAULT` and `HARDENED` is exported from `holodeck.models.deployment`.
- [ ] `DeploymentConfig.security_profile: SecurityProfile = SecurityProfile.DEFAULT` round-trips through YAML.
- [ ] Schema regenerated; `security_profile` appears under `deployment` with description and enum constraint.

**Verification:**
- [ ] `pytest tests/unit/models/test_deployment.py -n auto -v` passes (round-trip + default + invalid-value).
- [ ] `python -c "from holodeck.models.deployment import SecurityProfile; print(SecurityProfile.HARDENED)"` prints `SecurityProfile.HARDENED`.
- [ ] `git diff schemas/agent.schema.json` shows only the additive `security_profile` block.

**Dependencies:** None.
**Estimated scope:** S — one model file + one test file.

---

### Task 2: Spike — verify base-URL override paths

This task does **not** write production code. It produces a 1–2 page memo (`specs/034-production-hardening/2026-05-24-spec-034-p3-baseurl-spike.md`) that confirms or rejects three assumptions the rest of the plan depends on.

**Questions to answer (with code references, not opinions):**

- [ ] **Anthropic API via `ANTHROPIC_BASE_URL`.** Does the `claude-agent-sdk` (and the underlying Claude Code CLI) actually route through `ANTHROPIC_BASE_URL` for both `api_key` and `oauth_token` auth, including for the OAuth refresh flow? Cite the SDK source or the upstream CLI source. If OAuth refresh hits a hard-coded URL, that's a P3 blocker for `oauth_token` — document.
- [ ] **Anthropic auth header.** For `oauth_token`, confirm the header name the SDK actually sends (`authorization: Bearer …` vs. something OAuth-specific). The Envoy Lua filter has to know what header to inject; getting this wrong silently fails closed.
- [ ] **OpenAI / Azure OpenAI base-URL overrides.** Does the Semantic Kernel `OpenAITextEmbedding` accept a constructor arg (or env var) that we can use to redirect to `http://localhost:7000`? Same for `AzureTextEmbedding`. Cite the SK constructor signatures and whether HoloDeck currently exposes them.
- [ ] **Qdrant / Azure AI Search clients.** Do their Python clients accept a base-URL override that survives the `connection_string` parser? If not, can we override `connection_string` at deploy time to point at `https://localhost:7000` with SNI-rewriting? (Probably no — SNI to localhost won't match the upstream cert.) The honest answer here is likely: vector stores stay outside the sidecar in v1, and we document that.

**Acceptance criteria:**
- [ ] Memo committed under `specs/034-production-hardening/`.
- [ ] Each question has a yes/no answer with a file:line citation.
- [ ] For every "no" answer, the memo lists which provider/backend it removes from hardened-profile v1.

**Verification:** Human review of the memo — this is a research task, not an implementable contract.

**Dependencies:** None (can run in parallel with Task 1).
**Estimated scope:** M — investigation across two SDKs and 2–3 vector-store clients. Budget 1–2 days.

---

### Task 3: Implement `build_envoy_bootstrap`

**Files:**
- Create: `src/holodeck/deploy/envoy.py`
- Create: `tests/unit/deploy/test_envoy_generator.py`

**Design:**
- [ ] `build_envoy_bootstrap(agent: Agent) -> str` returns the full `bootstrap.yaml` content as a string. Pure function. No I/O.
- [ ] Listener on `0.0.0.0:7000`, HTTP/1.1 + HTTP/2.
- [ ] One `route_config` per allowlisted upstream domain. Each route matches by `:authority` header (so requests with `Host: api.anthropic.com` route to the anthropic upstream; everything else falls through to the catch-all `direct_response` with status 403 and an `application/problem+json` body matching spec 034 §"Backpressure 429" shape).
- [ ] Per-route `http_filters` chain: one Lua filter that reads `/var/run/secrets/envoy/<credname>` synchronously at filter init (Envoy hot-restarts pick up new secret values; this is fine) and adds the appropriate header. Header name per route (anthropic → `x-api-key`, openai → `authorization: Bearer`, azure openai → `api-key`).
- [ ] Upstream clusters declared with `transport_socket` of type `tls` so Envoy terminates the TLS upstream-side with the system trust store.
- [ ] Access log to stdout with the route name, response code, and response_time so an operator can `kubectl logs` (or ACA `az containerapp logs show`) and see what got allowed/denied.

**Allowlist derivation (this is the load-bearing logic):**
- [ ] `api.anthropic.com` is always present when `agent.model.provider == anthropic` and `auth_provider in {api_key, oauth_token}`.
- [ ] When `auth_provider == custom`, the upstream is the host of `agent.model.endpoint`.
- [ ] When the agent has any tool with `embedding_provider` set, the embedding provider's endpoint host is added.
- [ ] When the agent has any vectorstore/hierarchical_document tool with a `connection_string` whose scheme is `https`, the connection_string's host is added (Task 5 may downgrade this).
- [ ] No other domains are added.

**Acceptance criteria:**
- [ ] Given the financial-assistant sample's agent.yaml, `build_envoy_bootstrap` returns a YAML that includes upstream clusters for `api.anthropic.com` and the configured `<resource>.openai.azure.com` and the Qdrant cluster, in that order, and nothing else.
- [ ] Catch-all returns `403` with a problem+json body whose `type` matches `https://holodeck.dev/errors/egress-denied`.
- [ ] Generated YAML parses cleanly with PyYAML's safe loader.
- [ ] Generated YAML is byte-stable across calls (deterministic ordering of clusters → reproducible builds).

**Verification:**
- [ ] `pytest tests/unit/deploy/test_envoy_generator.py -n auto -v` covers: api_key path, oauth_token path, custom path, embedding-only agent, no-tool agent, agent with multiple vectorstores.
- [ ] `echo "$(holodeck-internal: dump-envoy-config sample/financial-assistant/claude)" | docker run --rm -i envoyproxy/envoy:v1.31-latest envoy --mode validate -c -` returns exit 0 (config validates against real Envoy). **Bash helper script lives in `scripts/validate_envoy.sh`**; not a CLI command.

**Dependencies:** Task 1 (`security_profile` field), Task 2 (auth-header confirmation).
**Estimated scope:** M — single module, one test file. The bulk of the cost is in getting the Envoy YAML right; allow 1–2 days plus the spike feedback.

---

### Task 4: Lua credential-injector filter

Subtask of Task 3, called out separately because it's the source of subtle bugs.

**Acceptance criteria:**
- [ ] Lua filter reads `/var/run/secrets/envoy/<credname>` *once* on filter init (not per request) and caches in `envoy.streamInfo`-adjacent state.
- [ ] If the file is missing or empty, the filter responds with `500` + `application/problem+json` body. The agent sees the failure as an upstream error; it does **not** fall back to direct upstream.
- [ ] The injected header replaces (not appends to) any header of the same name from the agent, so a compromised agent can't override the auth header with its own.
- [ ] The filter strips `Authorization` / `x-api-key` / `api-key` from the request before injection, regardless of upstream — defense against a leak from the agent.

**Verification:**
- [ ] Unit test the Lua via Envoy's test scaffold (`docker run --rm envoyproxy/envoy:v1.31-latest envoy --mode validate`) for the bootstrap, then a runtime test in Task 13.
- [ ] Test that a request with a forged `x-api-key` header from the agent is stripped before reaching upstream (visible in access log: stripped count = 1).

**Dependencies:** Task 3 (same file).
**Estimated scope:** S — small Lua snippet, three test cases.

---

### Task 5: Vector-store / embedding-provider compatibility audit

Concrete output of the Task 2 spike for the vector-store and embedding-provider axes specifically. May land as a single PR with Task 2 — split here for tracking.

**Acceptance criteria:**
- [ ] Maintain `_HARDENED_SUPPORTED_BACKENDS: frozenset[str]` constant on `holodeck.lib.backends.validators` listing the vector-store backend keys that work under hardened profile in v1.
- [ ] Maintain `_HARDENED_SUPPORTED_EMBED_PROVIDERS: frozenset[ProviderEnum]` listing the embedding providers that work.
- [ ] For each *unsupported* combination, `validate_hardened_profile_compatibility` emits a `ConfigError` whose `detail` field names the backend/provider and points at `docs/security/hardened-profile.md#supported-backends`.

**Verification:**
- [ ] `pytest tests/unit/lib/backends/test_hardened_validator.py -n auto -v` covers each supported and unsupported combo.

**Dependencies:** Task 2.
**Estimated scope:** S — one constant, one validator.

---

### Task 6: Deployer — ACA Secrets + `Volume(SECRET)` mounts

**Files:**
- Modify: `src/holodeck/deploy/deployers/azure_containerapps.py`
- Test: `tests/unit/deploy/test_aca_hardened_profile.py`

**Acceptance criteria:**
- [ ] `deploy()` accepts new kwargs: `secrets: list[ContainerAppSecret] | None`, `sidecar_containers: list[Container] | None`, `agent_volume_mounts: list[VolumeMount] | None`, `sidecar_volume_mounts: list[VolumeMount] | None`, `extra_volumes: list[Volume] | None`. All default to `None`. When `None`, behavior is byte-identical to today.
- [ ] When `secrets` is non-None, the `Configuration` includes a `secrets=[…]` block.
- [ ] When `sidecar_containers` is non-None, the `Template.containers` includes them after the agent container.
- [ ] When `extra_volumes` is non-None, they're appended to the existing `tmp` / `sdk-scratch` volumes.
- [ ] Critical invariant: the agent container's `volume_mounts` do **not** include any volume whose source is a secret. Asserted by the unit test.
- [ ] Echoes the new resolution block (per spec 034 CLI surface) when `sidecar_containers` is non-None — secrets named, allowlist hosts listed.

**Verification:**
- [ ] `pytest tests/unit/deploy/test_aca_hardened_profile.py -n auto -v` covers: secret-on-sidecar-only, two-container template, allowlist echo.
- [ ] The unit test inspects the in-memory `ContainerApp` model object — no Azure SDK calls, no `vcrpy`.

**Dependencies:** Task 1.
**Estimated scope:** M — deployer is a thick file; the change is additive but the test surface is wide.

---

### Task 7: Build the sidecar `Container` and wire it through

**Files:**
- Create: `src/holodeck/deploy/sidecar.py`
- Modify: `src/holodeck/deploy/builder.py` (or `cli/commands/deploy.py` — find where it constructs the deployer call)
- Test: `tests/unit/deploy/test_sidecar.py`

**Acceptance criteria:**
- [ ] `build_sidecar(agent, env_credentials)` returns `(Container, list[ContainerAppSecret], list[Volume], list[VolumeMount], list[VolumeMount])` where the tuple is `(sidecar_container, secrets, volumes, sidecar_mounts, agent_mounts_to_add)`.
- [ ] The sidecar container image is `envoyproxy/envoy:v1.31-latest`, command `["envoy", "-c", "/etc/envoy/bootstrap.yaml"]`, no env vars besides `ENVOY_UID=1000`.
- [ ] The Envoy bootstrap is projected as a secret (per the architecture note above) at `/etc/envoy/bootstrap.yaml`.
- [ ] Each credential is projected at `/var/run/secrets/envoy/<credname>` with file mode 0400.
- [ ] On the agent container side, set: `ANTHROPIC_BASE_URL=http://localhost:7000`, `HTTPS_PROXY=http://localhost:7000`, `HOLODECK_HARDENED=1`. **Remove** any of `{ANTHROPIC_API_KEY, CLAUDE_CODE_OAUTH_TOKEN, AWS_*, AZURE_OPENAI_API_KEY, OPENAI_API_KEY, QDRANT_API_KEY, ANTHROPIC_AUTH_TOKEN}` from the agent's env-var list before passing to the deployer.
- [ ] The builder reads credential env vars from the operator's host environment (the same surface `holodeck deploy run` reads today) and constructs the secret list. **Once the secret list is built, the local copies are zeroed in `env_credentials` and not retained in process memory longer than necessary** — best-effort, not a guarantee against a debugger.

**Verification:**
- [ ] `pytest tests/unit/deploy/test_sidecar.py -n auto -v` covers: financial-assistant agent yields exactly three secrets (anthropic, azure-openai-embed, qdrant-key), agent container env contains `HOLODECK_HARDENED=1` and no credential env vars, sidecar container contains exactly two `VolumeMount`s (`envoy-config`, `envoy-secrets`).

**Dependencies:** Task 3, Task 6.
**Estimated scope:** M — moderate composition logic, lots of small assertions.

---

### Task 8: Backend boot guard

**Files:**
- Modify: `src/holodeck/lib/backends/claude_backend.py`
- Modify: `src/holodeck/lib/backends/validators.py` (`validate_hardened_profile_compatibility`)
- Test: `tests/unit/lib/backends/test_hardened_boot_guard.py`

**Acceptance criteria:**
- [ ] New `_apply_hardened_boot_guard(agent)` called at the top of `build_options()`. When `os.environ.get("HOLODECK_HARDENED") == "1"`:
  - [ ] Assert `ANTHROPIC_BASE_URL` is set and starts with `http://localhost:` or `http://127.0.0.1:`. Raise `ConfigError("hardened_profile", ...)` if not.
  - [ ] Assert no key in `os.environ` matches the credential-bearing patterns `{ANTHROPIC_API_KEY, CLAUDE_CODE_OAUTH_TOKEN, ANTHROPIC_AUTH_TOKEN, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AZURE_OPENAI_API_KEY, OPENAI_API_KEY, QDRANT_API_KEY, GOOGLE_API_KEY, GEMINI_API_KEY}`. Raise `ConfigError("hardened_profile", ...)` listing the leaked var names. **Be careful to name them by class, not echo the value.**
  - [ ] Set `_HOLODECK_HARDENED_OK=1` on `options.env` so downstream logging knows the agent is running under hardened profile.
- [ ] `validate_hardened_profile_compatibility(agent)` called from agent load. Raises `ConfigError` when `auth_provider in {bedrock, vertex, foundry}` and `security_profile == hardened`. Same shape as the existing migration error for `acceptAll`.
- [ ] `ANTHROPIC_BASE_URL` injection in `build_options()` is reconciled: when `auth_provider == custom` AND `security_profile == hardened`, the env var is set by Task 7's deployer, not by `build_options()`. `build_options()` no longer rewrites it under hardened profile.

**Verification:**
- [ ] `pytest tests/unit/lib/backends/test_hardened_boot_guard.py -n auto -v` covers: pass-through when `HOLODECK_HARDENED=0`, fail-closed on missing `ANTHROPIC_BASE_URL`, fail-closed on each credential pattern, success on a clean hardened env.
- [ ] `pytest tests/unit/lib/backends/test_hardened_validator.py -n auto -v` covers: each cloud auth provider raises under hardened, `api_key`/`oauth_token`/`custom` pass.

**Dependencies:** Task 1, Task 5.
**Estimated scope:** S — small but invariant-heavy.

---

### Task 9: Wire the deploy command end-to-end under `security_profile: hardened`

**Files:**
- Modify: `src/holodeck/cli/commands/deploy.py`

**Acceptance criteria:**
- [ ] `holodeck deploy run path/to/agent` reads `deployment.security_profile`. When `hardened`:
  - [ ] Calls `validate_hardened_profile_compatibility(agent)` and surfaces errors.
  - [ ] Calls `build_envoy_bootstrap(agent)` and `build_sidecar(agent, env_credentials)`.
  - [ ] Calls deployer.deploy(..., secrets=…, sidecar_containers=…, extra_volumes=…, sidecar_volume_mounts=…) with the assembled fixtures.
  - [ ] Echoes the hardened-profile resolution block per spec 034 §"CLI surface".
- [ ] When `default` (or unset), the path is byte-identical to today.

**Verification:**
- [ ] Unit-level: `pytest tests/unit/cli/test_deploy_hardened.py -n auto -v` mocks the deployer and asserts that the right call shape is produced.
- [ ] Manual: `holodeck deploy run sample/financial-assistant/claude --dry-run` (if `--dry-run` exists; add it if it doesn't) prints the resolution block correctly. Otherwise skip; integration test (Task 13) is the runtime check.

**Dependencies:** Task 6, Task 7, Task 8.
**Estimated scope:** S — glue.

---

### Task 10: `serve` command surfaces hardened-profile state

**Files:**
- Modify: `src/holodeck/cli/commands/serve.py`

**Acceptance criteria:**
- [ ] At startup, when `os.environ.get("HOLODECK_HARDENED") == "1"`, the serve banner adds: `Security profile: hardened`, `Envoy base URL: <ANTHROPIC_BASE_URL>`.
- [ ] When not set, no change.

**Verification:**
- [ ] `pytest tests/unit/serve/test_banner.py -n auto -v` covers both branches (use existing banner test if present; otherwise add).

**Dependencies:** Task 8.
**Estimated scope:** XS — one if-block.

---

### Task 11: Sample agent for the integration test

**Files:**
- Create: `sample/financial-assistant/claude/agent.hardened.yaml` (copy of `agent.yaml` with `deployment.security_profile: hardened`)

**Acceptance criteria:**
- [ ] The sample loads cleanly under `holodeck` config validation when the credential env vars are present.
- [ ] The sample's allowlist (derived by the Envoy generator) is exactly: `api.anthropic.com`, `<resource>.openai.azure.com`, `<qdrant-cluster>.qdrant.tech`. Not a fourth host.

**Verification:**
- [ ] `holodeck test load sample/financial-assistant/claude/agent.hardened.yaml` (if a load-only CLI exists; otherwise add via `--validate-only` flag on `deploy run`).
- [ ] `pytest tests/unit/deploy/test_envoy_generator.py::test_financial_assistant_hardened_allowlist -n auto -v`.

**Dependencies:** Task 3.
**Estimated scope:** XS.

---

### Task 12: Docs — `docs/security/hardened-profile.md`

**Files:**
- Create: `docs/security/hardened-profile.md`

**Acceptance criteria:**
- [ ] Operator-facing. Covers: when to use it, what it does, what it doesn't (links to spec 034), supported auth providers, supported vector-store backends, the `bedrock`/`vertex`/`foundry` exclusion and the follow-up spec link (Task 16), the egress-denied error and how to fix it (add the host to a tool config or open a feature request).
- [ ] One worked example (the financial-assistant sample with `security_profile: hardened`).
- [ ] Cross-links to `docs/security/permissions.md` (P1b) and `docs/security/container-hardening.md` (P2a).

**Verification:**
- [ ] Human review.

**Dependencies:** Tasks 3, 7, 8, 11.
**Estimated scope:** S.

---

### Task 13: Integration test — live ACA deploy under hardened profile

**Files:**
- Create: `tests/integration/security/test_hardened_profile_e2e.py`

This is the smoke test that exercises the full machinery against a real ACA. Marked `@pytest.mark.slow` and `@pytest.mark.integration`; not part of `make test` by default. Run it via the manual deploy validation loop in `CLAUDE.md`.

**Acceptance criteria:**
- [ ] Test deploys `sample/financial-assistant/claude/agent.hardened.yaml` to a known ACA environment.
- [ ] After `/ready` returns 200, the test:
  1. Hits the AG-UI endpoint with a known-good ConvFinQA query and confirms a 200 with sensible content.
  2. Uses `az containerapp exec` (or equivalent) to `printenv` inside the agent container and asserts none of the credential env vars are present.
  3. Sends a request through the AG-UI surface that would trigger a (hypothetical) tool call to a non-allowlisted domain (e.g. by injecting via a malicious prompt). Confirms the request is rejected by Envoy with a 403 + problem+json body.
  4. Confirms the agent's response stream surfaces the upstream 403 cleanly (not a crash).

**Verification:**
- [ ] `pytest tests/integration/security/test_hardened_profile_e2e.py -m slow -n auto` against a deployed ACA passes.

**Dependencies:** All P3 tasks.
**Estimated scope:** L — full E2E. The setup cost is real; budget half a sprint for the first green run.

---

### Task 14: CHANGELOG + spec 034 status table update

**Files:**
- Modify: `specs/034-production-hardening/2026-05-18-production-hardening-for-claude-agents.md` (status table)
- Modify: top-level CHANGELOG (if one exists; otherwise skip)

**Acceptance criteria:**
- [ ] Status tracker for P3 changes from `⏳ not started` to `✅ shipped` with the branch/PR reference.
- [ ] One sentence summarizing what landed and what was deferred (bedrock/vertex/foundry — link to Task 16).

**Verification:** Human review.
**Dependencies:** Task 13 green.
**Estimated scope:** XS.

---

### Task 15: Documented gaps — `docs/security/hardened-profile.md#known-limitations`

**Acceptance criteria:**
- [ ] Bedrock / Vertex / Foundry exclusion + the SigV4/GCP-auth reason in one paragraph each. Link to Task 16 follow-up.
- [ ] TLS interception out of scope (no CA injection in v1).
- [ ] Stdio MCP credentials remain in the agent container.
- [ ] HTTP/SSE MCPs not routed (existing limitation, not new).
- [ ] OTLP path stays direct.
- [ ] pgvector and any non-HTTP vector store: out of scope.

**Verification:** Human review.
**Dependencies:** Task 12.
**Estimated scope:** XS (lives inside Task 12's doc; called out for tracking).

---

### Task 16: Follow-up spec stub for cloud-SDK auth providers

**Files:**
- Create: `specs/034-production-hardening/2026-05-24-spec-034-p3-followup-cloud-auth.md` (one-page stub)

**Acceptance criteria:**
- [ ] Names the problem: SigV4 and GCP-auth don't proxy transparently.
- [ ] Names three candidate paths: (a) sidecar re-signs requests using its own AWS/GCP credentials (requires the agent's SDK to talk plain HTTP+JSON to the sidecar, which means a new transport mode in `claude-agent-sdk` — coordinate with upstream); (b) cloud-native workload identity (IRSA on EKS, Workload Identity on GKE, Azure Managed Identity on AKS — none apply to ACA); (c) require operators to use the custom-tool pattern for non-Anthropic Anthropic-routed access.
- [ ] Calls out that ACA-specific deployments cannot use option (b), so options (a) and (c) are the realistic ones for HoloDeck's primary deployer.

**Verification:** Human review. The stub is not implementable; it is the index entry so the work doesn't get lost.
**Dependencies:** None.
**Estimated scope:** XS.

---

## Checkpoint structure

```
Tasks 1, 2          (parallel)        → SPIKE GATE
Tasks 3, 5           (parallel after 1, 2) → ENVOY + VALIDATOR GATE
Tasks 4, 6           (parallel after 3)    → SIDECAR/DEPLOYER GATE
Tasks 7, 8           (after 4 + 6)         → INTEGRATION GATE
Tasks 9, 10, 11      (parallel after 7+8)  → CLI/SAMPLE GATE
Task 12, 15          (after 11)            → DOCS GATE
Task 13              (after 12)            → E2E GATE (slow)
Tasks 14, 16         (after 13)            → CLOSE
```

Checkpoints (run between phases):

- **After Task 2 (SPIKE GATE):** Open questions resolved or scope narrowed. If the spike reveals `oauth_token` can't be proxied, drop it from v1 and update Task 5's supported-providers set before any production code lands.
- **After Tasks 3+5 (ENVOY GATE):** Pure-function generator unit-tests green; validator rejects all unsupported combos. No deployer changes yet — the whole P3 design is provable against unit tests at this point.
- **After Tasks 6+7 (DEPLOYER GATE):** Deployer unit tests green; the financial-assistant sample produces a believable two-container `ContainerApp` model object. Manual eyeball of the YAML the deployer would emit.
- **After Task 8 (INTEGRATION GATE):** Backend fails closed locally with `HOLODECK_HARDENED=1` and one credential env var present. This is the proof that the boot guard works before the deployer ever runs.
- **After Tasks 9–11 (CLI GATE):** `holodeck deploy run` produces correct CLI output. Resolution block matches the spec.
- **After Task 13 (E2E GATE):** Live ACA deploy passes the three smoke checks. **This is the only point at which P3 is mergeable.**

---

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Task 2 spike discovers `oauth_token` can't route through `ANTHROPIC_BASE_URL` for refresh | Drop `oauth_token` from hardened-profile v1; document in Task 15; revisit when SDK adds proxy support |
| Task 2 spike discovers no vector-store client supports a base-URL override | Vector-store creds remain on the agent container; Envoy enforces destination domain via `HTTPS_PROXY`; document |
| Envoy bootstrap drift across `v1.31` minor versions | Pin `envoyproxy/envoy:v1.31.x` to a specific patch; revalidate when bumping |
| ACA Secrets propagation lag (creates → not yet projected when sidecar starts) | Sidecar's Lua filter reads on first request, not init; if file missing, returns 500 (visible failure, not silent) — and we already have ACA's normal revision-readiness gating |
| The agent loops on the 403 from Envoy, racking up tokens | Already mitigated by `max_turns` default (P1a) at 20; document the failure mode in Task 12 |
| An operator's host env happens to have a stray `OPENAI_API_KEY` that the deployer reads and stuffs into ACA Secrets unintentionally | Task 7 only reads env vars that match credential patterns *and* that the agent.yaml's tool/embedding/auth config actually declares a need for. Unread vars are never copied. |
| Lua filter caches a credential file at filter init; subsequent secret rotation requires Envoy hot-restart | Document. ACA revision update triggers a fresh container, which re-inits the filter, which picks up the new secret. This is the supported rotation path. |
| Agent code path bypasses `ANTHROPIC_BASE_URL` (e.g. a tool that directly imports `anthropic` client without honoring the env var) | Task 8's boot guard makes this loud: any direct call without going through the SDK fails because the agent has no credentials. The 403 from Envoy is a secondary safety net. |
| `HTTPS_PROXY=http://localhost:7000` breaks stdio MCP servers that don't honor proxy env vars (e.g. a Node MCP using a non-proxy-aware fetch lib) | Document. Operator opts that MCP out of the sidecar by setting a `NO_PROXY` entry on the agent container — but then the MCP's egress is direct from the container. Acceptable trade-off; the sidecar protects the *LLM credential boundary*, not every outbound packet. |

---

## Open questions resolved before implementation

1. **Should the bootstrap be in a secret or a config-map style mount?** Decided: secret. ACA does not expose ConfigMap; this is the only file-projection primitive available.
2. **Should P3 ship on GCP Cloud Run and AWS App Runner deployers too?** No — v1 is ACA only. The other deployers have their own credential-mount surfaces (Cloud Run Secret Manager, App Runner secrets) and would each be their own task train. Documented as follow-ups in Task 16.
3. **Should the agent container's `HOLODECK_HARDENED` env var be a name an attacker could spoof to bypass checks?** No — the env var only gates *additional* checks; the absence of credentials in the agent's env is the real invariant. An attacker who unsets `HOLODECK_HARDENED` doesn't gain credentials they don't already have.
4. **Should we ship a deny-by-default DNS resolver in the sidecar?** No — Envoy's allowlist by `:authority` is sufficient. Adding a CoreDNS sidecar would couple two infrastructure concerns; out of scope.
5. **Where do credentials in P3 actually live?** Azure Container App Secrets, mounted via `Volume(storage_type=SECRET)` only to the sidecar. Not in the image, not in the agent container env, not in any other Azure secret store.

---

## v1 contract

- One new schema field: `deployment.security_profile: default | hardened`.
- Default profile: zero behavior change.
- Hardened profile: ACA only, auth providers `api_key` and `oauth_token` (and `custom` when the operator's endpoint is plain Anthropic-shaped), vector stores limited to HTTPS-overridable backends.
- Credentials in the agent container under hardened profile: zero. Asserted at boot.
- Allowlist derived from `agent.yaml`. No silent fallback. Non-allowlisted egress: 403.
- Sidecar holds credentials; agent fails closed if it sees any.
- Bedrock / Vertex / Foundry deferred to follow-up spec; rejected with a clear error.
- TLS interception, pgvector, HTTP/SSE MCP routing, GCP/AWS deployers: documented as out of v1.
