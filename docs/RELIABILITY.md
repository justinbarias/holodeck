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

## End-to-end deployment validation

Run this loop only when the user explicitly requests deployment validation or authorizes the target deployment.
This loop publishes images and changes a live service.
Reuse authorization already supplied in the conversation.
The [deployment guide](guides/deployment.md) describes supported configuration and CLI options.

The former local sample, `sample/financial-assistant/claude`, is git-ignored and may not exist in another checkout.
Use the user-selected agent, registry, and target. Do not assume a personal cloud endpoint.

1. Build the working-tree wheel with `uv build --wheel`. Keep exactly the intended wheel in the Docker build context.
2. Build [docker/Dockerfile.local](../docker/Dockerfile.local) with that wheel, using `--no-cache` and the target platform.
3. Inspect the installed wheel version inside the image.
4. Account for [builder.py](../src/holodeck/deploy/builder.py), which currently passes `pull=True` to Docker.
5. Build the agent image with `holodeck deploy build`, then publish the verified tag to the authorized registry.
6. Deploy with `holodeck deploy run` for the selected agent configuration.
7. Wait for health with a bounded timeout, then send a deterministic request with a known expected answer.
8. Record the image digest, target, response, and relevant telemetry.

`pull=True` can replace a locally tagged base image with its published registry version.
If validation requires a temporary local override, keep that edit isolated and restore only that edit afterward.
Do not commit an override that disables normal base-image pulling.
Azure Container Apps requires a compatible target image, commonly `linux/amd64` for this repository's deployment loop.
Readiness and cold-start timings vary. Measure them instead of relying on the old sample's timings.
