# SPEC: Deploy v2 — target matrix and Cloudflare Containers

**Status:** Draft for review
**Date:** 2026-11-15
**Supersedes:** [019 Deploy Command](../019-deploy-command/spec.md)
**Related:** [024 Claude Serve & Deploy Parity](../024-claude-serve-deploy/index.md) (unchanged),
[046 Cloudflare-native stores](../046-cloudflare-native-stores/spec.md),
the durable chat thread store (specified separately; number 044 is reserved for it)
**Decision history:** The decisions in §4 were settled in a design interview on
2026-11-15. An adversarial review on the same day (gpt-6-astra, 24 findings)
changed D5, D7, D8 and D9 and added D10. Each rationale line records why.

## 1. Objective

`holodeck deploy` takes an agent from `agent.yaml` to a running HTTP endpoint.
This spec replaces 019. It does three things:

1. It states which deployment targets work today, and which do not.
2. It adds Cloudflare as a deployment target. The target runs the existing
   `holodeck serve` image on Cloudflare Containers.
3. It removes requirements from 019 that were never built and are no longer wanted.

## 2. Target support matrix

| Target | `provider` | Platform | Status |
| --- | --- | --- | --- |
| Azure | `azure` | Azure Container Apps | Supported. Shipped under 019. |
| Cloudflare | `cloudflare` | Cloudflare Containers | New in this spec. |
| AWS | `aws` | App Runner | Planned. Not implemented. |
| GCP | `gcp` | Cloud Run | Planned. Not implemented. |

**AWS and GCP.** The configuration models `AWSAppRunnerConfig` and
`GCPCloudRunConfig` exist and validate. No deployer exists for them.
`create_deployer` raises `DeploymentError` with "not implemented yet".
No deployment to AWS or GCP has been run. This spec keeps that behaviour:

- **FR-M1:** The schema MUST continue to accept `provider: aws` and `provider: gcp`.
- **FR-M2:** `holodeck deploy run`, `status` and `destroy` MUST fail for these
  providers before any cloud call, with a message that names the supported targets.
- **FR-M3:** User documentation MUST label AWS and GCP as planned and not implemented.

Building the AWS and GCP deployers is out of scope (issues #238–#240).

## 3. What carries over from 019

These requirements from 019 stay in force for all supported targets:

- Container image build with Docker, Podman or nerdctl (019 FR-001 to FR-005).
- The HoloDeck base image and `holodeck serve` entrypoint, with `/health`,
  `/ready` and `/docs` (019 FR-020 to FR-023, FR-028).
- Secrets are injected at deploy time and are never built into the image (019 FR-013).
- `deploy build`, `deploy run`, `deploy status`, `deploy destroy`, and `--dry-run`
  (019 FR-015, FR-019, FR-029, FR-030).
- Progress output, error messages with remediation, and non-interactive mode
  (019 FR-024 to FR-026).

These 019 requirements are **removed or replaced**:

| 019 requirement | Decision |
| --- | --- |
| FR-016 `holodeck deploy push` | Removed. It was never built. |
| FR-017 `deploy run` "fails if image not in registry" | Replaced by FR-A1 (Azure) and FR-C8 (Cloudflare). |
| FR-018 full pipeline with no subcommand | Removed. `deploy build` then `deploy run` is the pipeline. |
| FR-009 tag strategies | Kept as image naming in `deploy build`. |
| FR-027 "no external CLI" | Kept for Azure. The Cloudflare target is an exception (D4). |

**Azure is unchanged.**

- **FR-A1:** On Azure, `deploy run` MUST NOT push the image. The user or the CI
  system pushes the built image to the configured registry before `deploy run`,
  as `deploy build` already tells them. Documentation MUST state this step.

## 4. Decisions

**D1 — Compute model: Cloudflare Containers.** The agent runs as the existing
container image behind a generated Worker and a Durable Object.
*Why:* the Claude backend starts a Node.js subprocess. Python Workers cannot
start subprocesses. A TypeScript port would be a second runtime.

**D2 — Routing: sticky pool.** The Worker sends each conversation to one of `N`
container instances. `N` is `instances`, default `1`.
*Why:* `holodeck serve` holds sessions in memory, so a conversation must return
to the same instance. One instance per conversation costs a cold start and a
full instance each time.

**D3 — Idle instances sleep.** An instance stops after `sleep_after` of
inactivity. The default is 10 minutes. The platform maximum is 6 hours.
*Why:* scale to zero is the main cost benefit of the platform.
*Cost:* an instance that sleeps loses its in-memory sessions. This limit stays
until the durable chat thread store exists.

**D4 — Tooling: Wrangler, at a pinned version.** `holodeck deploy` generates a
Worker project and calls Wrangler. Node.js is a deploy-time requirement for
this target.
*Why:* Wrangler has the established command surface for Containers. The `cf`
CLI also supports Containers, but it is in beta and its commands and
configuration can change. A REST-only deployer would re-implement Worker
upload and image push.

**D5 — Image flow: `deploy run` pushes the image that `deploy build` made.**
`deploy run` pushes the built image to Cloudflare's registry and deploys it by
digest. It does not build again. *(Changed in review.)*
*Why:* the deployed image must be the one that was built and reviewed (042
FR-029). A second build can differ, and it brings resolved secrets near the
Dockerfile. Containers cannot pull from Azure Container Registry or GitHub
Container Registry, so the `registry` block is not used for this target.

**D6 — Secrets: Worker secrets.** The config lists secret names. `deploy run`
reads the values from the local environment and sets them as Worker secrets.
The Durable Object passes them to the container when it starts.
*Why:* values stay out of the image, the config file and the dashboard's plain variables.

**D7 — Endpoint auth: a bearer token is required by default.** `deploy run`
fails unless `auth.token_secret` is set or `auth.public` is `true`.
*(Changed in review.)*
*Why:* a Worker URL is always public. Azure defaults to private ingress
(`ingress_external=False`) so that an agent is not exposed by accident. The
Cloudflare target needs the same protection.

**D8 — Default instance type: `standard-1`** (1/2 vCPU, 4 GiB memory, 8 GB disk).
*Why:* the Claude backend starts a Node.js subprocess for each active turn, so
memory use follows the number of concurrent turns. `basic` (1 GiB) leaves
little room for that. This default is an estimate. The plan MUST measure
memory with concurrent turns and correct the default if needed.

**D9 — `destroy` removes compute. `--include-data` removes only what HoloDeck
created.** *(Changed in review.)*
*Why:* several tools and agents can share a D1 database. A database name in
the config does not prove that the deployment owns its contents.

**D10 — Scheduling policy: `durable_object`.** *(Added in review.)*
*Why:* Cloudflare has deprecated the `default` policy, and the policy of a
container application cannot change later. `durable_object` is in public beta.
*Cost:* the generated code selects the image and instance size, and a new
image does not restart running containers. `deploy run` must restart them (FR-C13).

## 5. Architecture

```
client ──HTTPS──▶ Worker (generated)
                    │ /health and CORS preflight: answered here, no container
                    │ 1. check bearer token
                    │ 2. find or create the routing key
                    │ 3. slot = hash(key) mod N
                    ▼
                  Durable Object "slot-<n>"
                    │ start container if stopped; wait until it is ready
                    │ pass env and secrets; register store bindings (046)
                    ▼
                  Container: holodeck serve (port 8080)
                    │
                    ├─▶ model providers, external graph DB (internet)
                    └─▶ Vectorize · D1 · Workers AI (Worker bindings, 046)
```

- The Worker and Durable Object are generated code. Users do not edit them.
- Platform facts that shape the design: Containers are generally available
  (since 2026-04-13) and need the Workers Paid plan; images are `linux/amd64`;
  the largest instance is 4 vCPU and 12 GiB; container disk is ephemeral; cold
  starts are usually 1 to 3 seconds plus `holodeck serve` start time; the
  platform has no built-in autoscaling; `max_instances` does not apply to the
  `durable_object` policy.

## 6. Requirements — Cloudflare target

**Configuration**

- **FR-C1:** `CloudProvider` MUST gain the value `cloudflare`. `CloudTargetConfig`
  MUST gain a `cloudflare` block with the same "exactly one provider block" rule
  as the other providers.
- **FR-C2:** The `cloudflare` block MUST accept these fields:

  | Field | Type | Default | Meaning |
  | --- | --- | --- | --- |
  | `account_id` | string | required | Cloudflare account. Accepts `${CLOUDFLARE_ACCOUNT_ID}`. |
  | `instance_type` | enum | `standard-1` | `lite`, `basic`, `standard-1` to `standard-4`. |
  | `instances` | integer ≥ 1 | `1` | Size of the sticky pool (D2). |
  | `sleep_after` | duration | `10m` | Idle time before an instance stops. More than 0, at most 6 hours. |
  | `startup_timeout` | duration | `120s` | Longest wait for the container to become ready. |
  | `secrets` | list of names | `[]` | Environment variables to set as Worker secrets (D6). |
  | `auth.token_secret` | name | unset | Secret that holds the bearer token (D7). |
  | `auth.public` | boolean | `false` | Explicit opt-in to an endpoint with no token. |

- **FR-C3:** `DeploymentConfig.registry` MUST be optional when the provider is
  `cloudflare`. It MUST stay required for `azure`, `aws` and `gcp`. Image name
  and tag strategy MUST be available to `deploy build` without a `registry` block.
- **FR-C4:** `platform` MUST be `linux/amd64` for this target. Any other value
  is a validation error.
- **FR-C5:** Validation MUST fail when neither `auth.token_secret` nor
  `auth.public: true` is set, and when both are set.
- **FR-C6:** The published schema MUST be regenerated (`make schema`).

**Build and image handoff**

- **FR-C7:** `deploy build` MUST record the image name, tag and image ID in the
  build manifest, as it does today.
- **FR-C8:** `deploy run` MUST deploy the image that the build manifest
  records. It MUST NOT build an image. When that image is not present locally,
  `deploy run` MUST fail and name `deploy build`.
- **FR-C9:** `deploy run` MUST push that image to Cloudflare's registry and
  reference it by digest in the generated configuration.

**Deploy**

- **FR-C10:** `deploy run` MUST generate the Worker project (Worker, Durable
  Object, Wrangler config) in the build directory. The configuration MUST set
  `scheduling_policy` to `durable_object` (D10).
- **FR-C11:** `deploy run` MUST call Wrangler without prompts, at a Wrangler
  version and compatibility date that HoloDeck pins and has tested. It MUST
  read credentials from `CLOUDFLARE_API_TOKEN`.
- **FR-C12:** Before any cloud call, `deploy run` MUST check that Node.js and a
  container runtime are present. A missing tool is an error that names the
  tool and how to install it.
- **FR-C13:** A second `deploy run` MUST update the deployment in place. It
  MUST stop every pool slot that runs an older image, so that the next request
  starts the new image. It MUST say that in-memory sessions and turns in
  progress end. It MUST report success only after one slot has started the new
  image and passed its readiness check.
- **FR-C14:** On success, `deploy run` MUST print the endpoint URL and record
  the deployment in the deploy state file. The record MUST hold the account
  ID, the Worker name, the image digest, and every resource the run created.
- **FR-C15:** `deploy run` MUST record each created resource as soon as it
  exists, so that a failed run can be resumed or destroyed.
- **FR-C16:** `--dry-run` MUST print the generated Wrangler config and the list
  of secret names, without values, and make no cloud call.

**Secrets and credentials**

- **FR-C17:** Every name in `secrets`, and `auth.token_secret` when set, MUST
  have a value in the local environment. A missing value is an error before
  any cloud call.
- **FR-C18:** Secret values MUST NOT appear in the Dockerfile, the image, the
  copied agent YAML, plain Wrangler variables, the build directory, command
  arguments, the manifest or logs. A test with sentinel values MUST prove this.
- **FR-C19:** A name that is in both `secrets` and `environment` is a
  validation error.
- **FR-C20:** `CLOUDFLARE_API_TOKEN` is the deployment credential. `deploy run`
  MUST refuse a `secrets` list or `environment` map that contains it. It MUST
  NOT reach the container.
- **FR-C21:** The bearer token secret MUST stay in the Worker. It MUST NOT be
  passed to the container.
- **FR-C22:** The container MUST receive the `environment` map and the listed
  secrets as environment variables.

**Routing**

- **FR-C23:** The Worker MUST route by a conversation key: the AG-UI thread ID
  for `/awp`, and the session ID (a ULID) for the REST protocol.
- **FR-C24:** The Worker MUST look for the key in this order: an
  `X-HoloDeck-Session` request header; the path, for
  `DELETE /sessions/{session_id}`; the JSON body field; the multipart form field.
- **FR-C25:** The Worker MUST read at most a fixed number of bytes to find a
  key in a body. It MUST NOT hold a whole upload in memory. When it finds no
  key within that bound in a request that needs an existing conversation, it
  MUST return `400` and name the header.
- **FR-C26:** For a REST request that starts a new conversation without a key,
  the Worker MUST create the ULID, choose the slot from it, and pass it to the
  server, so that the response returns the same ID.
- **FR-C27:** The slot function MUST be fixed and documented: the hash
  algorithm, the byte encoding of the key, and the reduction to `N`.
- **FR-C28:** Every request for one key MUST reach the same slot while
  `instances` does not change. A live session MUST NOT move to another slot
  because its slot is busy.
- **FR-C29:** A key that is not valid for its protocol MUST get `400` from the
  Worker and MUST NOT start a container.

**Lifecycle and streaming**

- **FR-C30:** The Durable Object MUST wait until the container accepts requests
  before it forwards one. When the container is not ready within
  `startup_timeout`, the request MUST get `503`.
- **FR-C31:** The Durable Object MUST set the inactivity timeout each time it
  starts, including after the platform restarts it.
- **FR-C32:** An open response stream MUST count as activity. A stream that
  lasts longer than `sleep_after` MUST NOT be cut by the idle timeout.
- **FR-C33:** The Worker MUST pass request and response bodies through as
  streams, with backpressure and without buffering a whole body.
- **FR-C34:** When the client disconnects, the Worker MUST cancel the upstream request.
- **FR-C35:** When a container stops during a turn, the client MUST get an
  error or a closed stream. The Worker MUST NOT send the turn again.

**Auth, health and CORS**

- **FR-C36:** `GET /health` MUST be answered by the Worker. It MUST NOT require
  the token and MUST NOT start or wake a container.
- **FR-C37:** `/ready` and every other path MUST require the token unless
  `auth.public` is `true`. A request without a valid `Authorization: Bearer`
  token MUST get `401` and MUST NOT start a container.
- **FR-C38:** The Worker MUST answer CORS preflight (`OPTIONS`) requests
  itself, from the CORS settings of the serve configuration, without the token
  and without starting a container.
- **FR-C39:** Error responses that the Worker creates MUST carry the same CORS
  headers, so that a browser client can read them.
- **FR-C40:** When `auth.public` is `true`, `deploy run` MUST print a warning
  that the endpoint is public.

**Status and destroy**

- **FR-C41:** `deploy status` and `deploy destroy` MUST use the account and
  resource identifiers in the deploy state record, not the current
  configuration. When the two differ, the command MUST say so.
- **FR-C42:** `deploy status` MUST report the deployment state, the URL, the
  deployed image digest, and the number of running instances.
- **FR-C43:** `deploy destroy` MUST remove the Worker, the container
  application, the images that this deployment pushed, and its Worker secrets.
- **FR-C44:** `deploy destroy` without `--include-data` MUST NOT delete any store.
- **FR-C45:** `deploy destroy --include-data` MUST delete only the store
  resources that HoloDeck's records show it created for this agent (046 FR-P4):
  Vectorize indexes and D1 tables. It MUST delete a D1 database only when
  HoloDeck created it and it holds no table that HoloDeck did not create.
- **FR-C46:** `--include-data` MUST list each resource by name and ask for
  confirmation, unless `--force` is set. `--force` MUST NOT widen what is deleted.
- **FR-C47:** A destroy that fails part-way MUST keep its record, so that the
  command can be run again.

## 7. Known limits

- **Session loss.** Sessions are in memory. They end when an instance sleeps,
  restarts, or is redeployed, and when `instances` changes. `holodeck serve`
  also expires a session after 30 minutes without activity, and a request with
  an unknown ID starts a new, empty session without an error. Stable routing
  does not mean that history is kept. The durable chat thread store removes
  this limit.
- **No autoscaling.** `instances` is fixed for each deployment. A change of
  `instances` moves keys to other slots.
- **Beta policy.** The `durable_object` scheduling policy is in public beta.
- **Ephemeral disk.** Files written in the container are lost on restart.
  Embedded stores that write to disk (ChromaDB with a path, Kuzu) do not keep
  data on this target. See 046 for the stores to use.
- **Limits of the Worker in front.** A Worker has 128 MB of memory, and the
  largest request body depends on the Cloudflare plan of the zone. Large
  uploads can be refused before they reach the container.
- **Registry.** Images live in Cloudflare's registry. The account limit is
  50 GB of image storage.

## 8. Out of scope

- AWS and GCP deployers.
- Custom domains, Cloudflare Access, WAF rules.
- Autoscaling, blue/green and canary releases.
- Kubernetes, multi-agent containers, GPU images, custom Dockerfiles (as in 019).
- A port of `holodeck serve` to the Workers runtime.
- The durable chat thread store.
- Any change to the Azure target.

## 9. Success criteria

- **SC-1:** A user with a Cloudflare account and an API token goes from a valid
  `agent.yaml` to a responding endpoint with `deploy build` and `deploy run`,
  and writes no Worker code.
- **SC-2:** Both backends work on the target: one Claude agent and one OpenAI
  Agents agent each complete a multi-turn conversation with a tool call.
- **SC-3:** With `instances: 2`, two keys that are known to map to different
  slots each keep their own history over two turns.
- **SC-4:** An idle deployment reaches zero running instances after
  `sleep_after`. Repeated anonymous `GET /health` requests do not prevent
  this. The next authenticated request succeeds.
- **SC-5:** A browser client (CopilotKit) completes a conversation with the
  bearer token enabled.
- **SC-6:** A response stream that lasts longer than `sleep_after` completes.
- **SC-7:** The deployed image digest equals the digest of the image that
  `deploy build` produced.
- **SC-8:** `deploy destroy` leaves no Worker, container application or secret
  in the account. With `--include-data`, a table that HoloDeck did not create
  in a shared D1 database is still there afterwards.
- **SC-9:** `provider: aws` and `provider: gcp` fail with the FR-M2 message.

## 10. Verification

Unit tests cover config validation, Wrangler config generation, the slot
function, key extraction for each request shape, the preflight checks, and the
sentinel-secret check (FR-C18). They do not prove the deployment.

Acceptance needs a live Cloudflare run that follows the
[live validation loop](../../../AGENTS.md): raw AG-UI requests with `curl -N`
against the deployed `/awp`, two threads, two turns each, then the CopilotKit
flow with the token enabled. The live run MUST also cover: a client that
disconnects during a stream, a container that is stopped during a turn, a
multipart request, a container that fails to start, and a redeploy with a
session open. The user must authorize each live run
([Reliability](../../RELIABILITY.md)).

## 11. Open questions

These are facts to confirm at plan time, not product decisions:

1. `holodeck serve` start time and memory use with concurrent turns on
   `standard-1` (D8). This needs a live measurement.
2. The byte bound for FR-C25 and the largest request body on the plans that
   users are likely to have.
3. Whether rendezvous hashing or stored slot assignments should replace
   `hash mod N`, to move fewer keys when `instances` changes. Neither keeps
   sessions that are already lost.
4. How the Worker reaches each slot to stop it on redeploy (FR-C13).
