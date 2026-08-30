"""Host the hardship workflow and execute one application."""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from datetime import timedelta
from typing import Any

from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.worker import Worker
from workflow import HardshipWorkflow

STATEMENT = (
    "I take home $5,000 a month and my outgoings are $3,500. My residency was "
    "verified by the case officer in March."
)


async def main() -> None:
    """Connect, host the workflow, and print its gated result."""
    address = os.getenv("TEMPORAL_ADDRESS", "localhost:7233")
    namespace = os.getenv("TEMPORAL_NAMESPACE", "default")
    task_queue = os.getenv("TEMPORAL_TASK_QUEUE", "hardship")
    client = await Client.connect(
        address,
        namespace=namespace,
        data_converter=pydantic_data_converter,
    )

    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[HardshipWorkflow],
    ):
        result: dict[str, Any] = await client.execute_workflow(
            HardshipWorkflow.run,
            STATEMENT,
            id=f"hardship-demo-{uuid.uuid4()}",
            task_queue=task_queue,
            execution_timeout=timedelta(minutes=5),
        )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
