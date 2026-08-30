# Temporal Worker

This page is the host-operator reference for HoloDeck activities. Read the
[Temporal Integration guide](temporal.md) first for workflow code and patterns.

## `worker.yaml` host

`worker.yaml` registers agents as activities. It contains no workflow control
flow, activity timeout, or retry policy.

```yaml
temporal:
  address: localhost:7233
  namespace: default
  task_queue: hardship
  tls: false
nodes:
  - id: evidence
    edge:
      agent: agents/evidence.yaml
    gate:
      schema: gates/evidence.schema.json
    source: Hardship Policy v4.2 §72
  - id: letter
    edge:
      agent: agents/letter.yaml
    gate:
      schema: gates/letter.schema.json
```

### Field reference

| Field | Required | Default | Meaning |
| --- | --- | --- | --- |
| `temporal` | no | empty block | Temporal connection block. Unknown fields fail validation. |
| `temporal.address` | no | `localhost:7233` | Temporal frontend as `host:port`. A blank value is invalid. |
| `temporal.namespace` | no | `default` | Namespace that the worker polls. A blank value is invalid. |
| `temporal.task_queue` | effective config | none | Queue that the worker polls. Supply it in the file or through `TEMPORAL_TASK_QUEUE`. |
| `temporal.tls` | no | `false` | Enables TLS. Client certificate fields are not available in v1. |
| `nodes` | yes | none | Non-empty list of activity registrations. Unknown fields fail validation. |
| `nodes[].id` | yes | none | Unique, non-blank activity name. Changing it affects replay compatibility. |
| `nodes[].edge.agent` | yes | none | Path to `agent.yaml`, relative to the directory of `worker.yaml`. |
| `nodes[].gate.schema` | yes | none | Path to the JSON Schema gate, relative to the same directory. |
| `nodes[].source` | no | `null` | Authority annotation, such as a policy citation. It does not control execution. |

`base_dir` is discovered from the location of `worker.yaml`. You cannot set it
in the file. Agent and gate paths must remain inside this directory after path
resolution. The loader checks agent confinement. Activity binding checks gate
confinement and gate validity.

### Environment overrides

The shell environment overrides file values:

| Variable | Field |
| --- | --- |
| `TEMPORAL_ADDRESS` | `temporal.address` |
| `TEMPORAL_NAMESPACE` | `temporal.namespace` |
| `TEMPORAL_TASK_QUEUE` | `temporal.task_queue` |

A present but blank variable overrides the file and then fails validation.
HoloDeck does not fall back to the file value. This fail-closed rule prevents a
broken secret or template from routing work to the wrong service or queue.

There is no `TEMPORAL_TLS` override. TLS is file-only because an environment
value could silently downgrade a secure connection to plaintext.

## Run `holodeck worker`

```bash
holodeck worker --config worker.yaml
holodeck worker --config deploy/worker.yaml --task-queue urgent
```

`--config` (or `-c`) defaults to `worker.yaml`. `--task-queue` overrides the
environment and file values. The command registers activities only and never
registers workflows.

The file or `TEMPORAL_TASK_QUEUE` must supply an initial queue. The CLI applies
`--task-queue` after the file passes validation.

The worker configures `pydantic_data_converter` and `TracingInterceptor`. On
`SIGINT` or `SIGTERM`, it gives in-flight activities up to 30 seconds to
finish. Normal signal shutdown exits `0`. A startup `HoloDeckError` exits `1`.
A `KeyboardInterrupt` outside the asynchronous signal path exits `130`.

## Two-terminal quickstart

From the repository root, activate the environment and start a local Temporal
server. Keep it in a separate terminal, or run it in the background:

```bash
source .venv/bin/activate
export CLAUDE_CODE_OAUTH_TOKEN=your-token
temporal server start-dev
```

Terminal 1 hosts the two HoloDeck activities:

```bash
cd sample/temporal-hardship
env -u CLAUDECODE holodeck worker --config worker.yaml
```

Terminal 2 hosts the user-authored workflow and starts one execution:

```bash
cd sample/temporal-hardship
python run_workflow.py
```

If the application terminals do not use the sample defaults, set
`TEMPORAL_ADDRESS`, `TEMPORAL_NAMESPACE`, and `TEMPORAL_TASK_QUEUE` in both.

## Observability

An agent activity emits the same OpenTelemetry GenAI spans as a HoloDeck test
run when the agent enables observability. `holodeck worker` installs Temporal's
`TracingInterceptor`, so these GenAI spans nest below
`RunActivity:<activity-name>` spans in the same trace.

Factory or plugin users must install `TracingInterceptor` on the client to get
the Temporal parent spans. Configure exporters, content capture, sampling, and
redaction in the [Observability guide](observability.md).
