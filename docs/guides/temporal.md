# Temporal Integration

HoloDeck exposes schema-gated agents as Temporal activities. You write the
Temporal workflow in Python and decide its control flow.

HoloDeck supplies the activity factory, typed payloads, deterministic policy
helpers, a worker plugin, and an activities-only worker CLI. It does not supply
a workflow engine, workflow YAML, a DAG runner, or a human-task UI.

## Install

```bash
pip install 'holodeck-ai[temporal]'
```

The extra pins `temporalio==1.32.0`. The workflow sandbox APIs used by this
integration are not yet stable, so HoloDeck uses an exact pin.

Importing `holodeck.temporal` without the extra raises a `ConfigError` that
names the install command. The core CLI keeps Temporal imports lazy, so
`holodeck --help` and `holodeck worker --help` still work without the extra.

## Python developer

### Bind an agent activity

`agent_activity()` binds an `EdgeNode` to one Temporal activity. The node ID
becomes the activity name and is part of replay history.

```python
from pathlib import Path

from holodeck.models.workflow import EdgeNode
from holodeck.temporal.activity import agent_activity

BASE_DIR = Path(__file__).parent
EVIDENCE_NODE = EdgeNode.model_validate(
    {
        "id": "evidence",
        "edge": {"agent": "agents/evidence.yaml"},
        "gate": {"schema": "gates/evidence.schema.json"},
        "source": "Hardship Policy v4.2 §72",
    }
)
evidence_activity = agent_activity(EVIDENCE_NODE, BASE_DIR)
```

The factory resolves the agent and gate paths, loads both files, and checks
that the agent has a `response_format`. These checks occur when the worker
binds the activity, before it polls or spends tokens.

Each activity call is one stateless `invoke_once()` call. The activity applies
the JSON Schema gate before it returns. Only the validated object enters
Temporal history.

### Use typed payloads

`AgentActivityInput` contains `message` and optional JSON-compatible `context`.
`AgentActivityResult` contains the gated `output` dictionary, `token_usage`,
`num_turns`, and `agent_id`. Raw model text is not present.

Use `output_as()` for typed access after the payload crosses the wire:

```python
from pydantic import BaseModel

from holodeck.temporal.models import AgentActivityResult


class Income(BaseModel):
    net: float
    expenses: float


class Residency(BaseModel):
    status: str


class EvidenceOutput(BaseModel):
    income: Income
    residency: Residency


def typed_output(result: AgentActivityResult) -> EvidenceOutput:
    return result.output_as(EvidenceOutput)
```

The wire value remains a dictionary. `output_as()` applies Pydantic validation
and returns your model.

All clients that encode or decode these payloads must use
`pydantic_data_converter`. `HoloDeckPlugin` and `holodeck worker` set it. If you
wire a client yourself, set the converter explicitly:

```python
from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter


async def connect_client() -> Client:
    return await Client.connect(
        "localhost:7233",
        namespace="default",
        data_converter=pydantic_data_converter,
    )
```

### Set timeouts and retries in the workflow

Temporal stores scheduling options on `execute_activity()`, not on the
activity definition. `ActivityParameters.to_activity_kwargs()` creates the
correct keyword arguments.

```python
from datetime import timedelta

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from holodeck.temporal.deterministic import ActivityParameters
    from holodeck.temporal.models import AgentActivityInput, AgentActivityResult

EVIDENCE_PARAMETERS = ActivityParameters(
    start_to_close=timedelta(minutes=3),
    maximum_attempts=3,
    initial_interval=timedelta(seconds=2),
)


async def extract(statement: str) -> AgentActivityResult:
    return await workflow.execute_activity(
        "evidence",
        AgentActivityInput(message=statement),
        result_type=AgentActivityResult,
        **EVIDENCE_PARAMETERS.to_activity_kwargs(),
    )
```

Set `start_to_close` or `schedule_to_close`. Temporal has no server default for
either closing timeout. Optional fields also cover `schedule_to_start`, retry
backoff, maximum intervals, maximum attempts, and non-retryable error types.

`heartbeat_timeout` is deliberately absent. The agent activity does not send
heartbeats. A heartbeat timeout could start a duplicate, billable model call
while the first call still runs.

### Understand retryable errors

HoloDeck separates model and transport faults from authoring faults.

| Fault | Temporal behavior |
| --- | --- |
| `GateValidationError` | Retryable. The model returned free text or an object that failed the gate. |
| `ExecutionError` | Retryable. The invocation raised or returned no object to judge. |
| `ConfigError` | Non-retryable when reached during a call. Most cases fail at bind time. |
| `GateSchemaError` | Non-retryable when reached during a call. Most cases fail at bind time. |
| `holodeck.lib.errors.FileNotFoundError` | Non-retryable authoring fault. |
| `BackendInitError` | Non-retryable authoring fault, such as missing credentials or Node.js. |

Per-call authoring faults become `ApplicationError(non_retryable=True)`. The
error type remains the original class name. Temporal retry policies match
`non_retryable_error_types` by that string.

### Choose plugin or factory wiring

If one client hosts your workflows and agent activities, use `HoloDeckPlugin`.
It sets the Pydantic converter and registers each activity. Pass it only to
`Client.connect()` because client plugins propagate to the worker:

```python
from temporalio.client import Client
from temporalio.contrib.opentelemetry import TracingInterceptor
from temporalio.worker import Worker

from holodeck.temporal.plugin import HoloDeckPlugin
from holodeck.temporal.worker_config import load_worker_config
from workflow import HardshipWorkflow


async def build_worker() -> Worker:
    config = load_worker_config("worker.yaml")
    client = await Client.connect(
        config.temporal.address,
        namespace=config.temporal.namespace,
        plugins=[
            HoloDeckPlugin(nodes=config.nodes, base_dir=config.base_dir)
        ],
        interceptors=[TracingInterceptor()],
        tls=config.temporal.tls,
    )
    return Worker(
        client,
        task_queue=config.temporal.task_queue,
        workflows=[HardshipWorkflow],
    )
```

If you want explicit registration, use the factory directly. Supply the
Pydantic converter and the activities to your client and worker:

```python
from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.worker import Worker

from holodeck.temporal.activity import agent_activity
from holodeck.temporal.worker_config import load_worker_config
from workflow import HardshipWorkflow


async def build_worker() -> Worker:
    config = load_worker_config("worker.yaml")
    activities = [agent_activity(node, config.base_dir) for node in config.nodes]
    client = await Client.connect(
        config.temporal.address,
        namespace=config.temporal.namespace,
        data_converter=pydantic_data_converter,
    )
    return Worker(
        client,
        task_queue=config.temporal.task_queue,
        workflows=[HardshipWorkflow],
        activities=activities,
    )
```

`holodeck worker` uses factory-direct wiring because it hosts activities only.

### Keep workflow code deterministic

Put workflows in their own Python module. Import the D3 surface inside
`workflow.unsafe.imports_passed_through()`:

- `evaluate()` evaluates a loaded [decision table](decision-tables.md).
- `check_gate()` applies a JSON Schema gate to a plain object.
- `ActivityParameters` creates scheduling arguments.
- `load_decision_table()` reads policy at import time only.

The [Decision Tables guide](decision-tables.md) covers the `.dmn.yaml` format,
S-FEEL subset, hit policies, loading validation, and `Verdict` fields. The
remaining rules in this section are specific to Temporal's workflow sandbox.

Load a decision table at module scope in a **sibling module**:

```python
# policy.py
from pathlib import Path

from holodeck.temporal.deterministic import DecisionTable, load_decision_table

TABLE_PATH = Path(__file__).parent / "tables" / "hardship.dmn.yaml"
TABLE: DecisionTable = load_decision_table(TABLE_PATH)
```

Then pass that sibling import through in the workflow module:

```python
# workflow.py
from typing import Any

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from holodeck.temporal.deterministic import Verdict, check_gate, evaluate
    from policy import TABLE


def decide(
    candidate: dict[str, Any], gate_schema: dict[str, Any]
) -> Verdict:
    validated = check_gate(candidate, gate_schema, node_id="evidence")
    return evaluate(TABLE, validated)
```

!!! warning "Load decision tables from a sibling module"
    The sandbox re-imports the workflow's defining module during validation.
    A table load there repeats file I/O inside the sandbox and fails. The
    passed-through sibling module is already loaded, so its file read does not
    run again.

!!! warning "A table is workflow code for versioning purposes"
    Sibling placement solves the sandbox problem only — it does not make
    table changes replay-safe. `evaluate(TABLE, ...)` runs in workflow code,
    so editing the YAML between deployments changes replay behavior for open
    workflows exactly as editing an `if` statement would. Apply the same
    discipline as for any workflow-definition change: gate the new policy
    behind `workflow.patched()` or deploy it with
    [Temporal Worker Versioning](https://docs.temporal.io/workers#worker-versioning),
    and bump the table's `version` field so completed histories stay
    auditable.

The hardship sample uses the
[gate-shape-equals-table-input-shape](decision-tables.md#use-a-table-with-an-agent-workflow)
pattern. As a result, `evaluate(TABLE, evidence.output)` needs no mapping
layer. This pattern keeps policy inputs explicit and gated.

## Workflow patterns

### Business-rule feedback retry

An activity retry repeats the same payload after a model or transport fault.
A business-rule retry is different. The workflow evaluates valid output,
adds deterministic feedback, and invokes the activity with a new payload.

```python
from datetime import timedelta
from typing import Any

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from holodeck.temporal.deterministic import ActivityParameters, evaluate
    from holodeck.temporal.models import AgentActivityInput, AgentActivityResult
    from policy import TABLE

MAX_BUSINESS_ATTEMPTS = 3
EVIDENCE_PARAMETERS = ActivityParameters(
    start_to_close=timedelta(minutes=3), maximum_attempts=1
)


def validate_evidence(evidence: dict[str, Any]) -> str | None:
    income = evidence["income"]
    if not isinstance(income, dict) or float(income["net"]) <= 0:
        return "income.net must be greater than zero for affordability policy"
    return None


@workflow.defn
class EvidenceFeedbackWorkflow:
    @workflow.run
    async def run(self, statement: str) -> dict[str, Any]:
        feedback: str | None = None
        for _ in range(MAX_BUSINESS_ATTEMPTS):
            context = None
            if feedback is not None:
                context = {
                    "business_rules_validation_outcome": {
                        "valid": False,
                        "reason": feedback,
                    }
                }
            evidence: AgentActivityResult = await workflow.execute_activity(
                "evidence",
                AgentActivityInput(message=statement, context=context),
                result_type=AgentActivityResult,
                **EVIDENCE_PARAMETERS.to_activity_kwargs(),
            )
            feedback = validate_evidence(evidence.output)
            if feedback is None:
                verdict = evaluate(TABLE, evidence.output)
                return {"status": "decided", "verdict": verdict.model_dump()}

        return {"status": "manual_review", "reason": feedback}
```

The bound keeps workflow history and model cost finite. Use this pattern for
reparable interpretation errors, not to force a preferred policy verdict.

When `context` is not `None`, the activity appends this exact prompt shape:

```text
<message>

Context (JSON):
<JSON object with sorted keys>
```

The renderer uses `json.dumps(..., sort_keys=True, default=str)`. A `None`
context adds no header. The `business_rules_validation_outcome` name is a
caller-owned contract, not a special HoloDeck field.

### Human approval with SLA escalation

HoloDeck ships no HITL implementation. Temporal signals, durable conditions,
and timers already supply the required behavior.

```python
import asyncio
from datetime import timedelta
from typing import Any

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from holodeck.temporal.deterministic import ActivityParameters
    from holodeck.temporal.models import AgentActivityInput, AgentActivityResult

PARAMETERS = ActivityParameters(start_to_close=timedelta(minutes=3))


@workflow.defn
class HardshipApprovalWorkflow:
    def __init__(self) -> None:
        self._approved: bool | None = None

    @workflow.signal
    async def set_approval(self, approved: bool) -> None:
        self._approved = approved

    @workflow.run
    async def run(self, statement: str) -> dict[str, Any]:
        evidence: AgentActivityResult = await workflow.execute_activity(
            "evidence",
            AgentActivityInput(message=statement),
            result_type=AgentActivityResult,
            **PARAMETERS.to_activity_kwargs(),
        )

        approval = asyncio.create_task(
            workflow.wait_condition(lambda: self._approved is not None)
        )
        sla = asyncio.create_task(workflow.sleep(timedelta(hours=24)))
        done, pending = await workflow.wait(
            {approval, sla}, return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()

        if approval not in done:
            return {"status": "escalated", "evidence": evidence.output}
        status = "approved" if self._approved else "declined"
        return {"status": status, "evidence": evidence.output}
```

A caller sends the decision with
`await handle.signal(HardshipApprovalWorkflow.set_approval, True)`. Use
`workflow.wait()` for the race. Temporal 1.32 rejects `asyncio.wait()` inside
the workflow sandbox because completion order can be non-deterministic.

Next, configure an activities-only host in the
[`worker.yaml` and CLI guide](temporal-worker.md).
