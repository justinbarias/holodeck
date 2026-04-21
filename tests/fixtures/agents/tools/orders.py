"""Order-lookup fixture referenced by tests/fixtures/agents/valid_agent.yaml.

Before Phase 0, this file did not exist and valid_agent.yaml was effectively
schema-only. With the FunctionTool runtime in place the fixture is now runnable.
"""


def get_order_status(order_id: str) -> str:
    """Return a canned status string for *order_id* (fixture stub).

    Args:
        order_id: Free-form order identifier.

    Returns:
        A one-line human-readable status for the order.
    """
    return f"Order {order_id} status: shipped"
