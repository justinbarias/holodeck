"""Deterministic function tools for the OpenAI Agents full-capability sample."""

from __future__ import annotations

import itertools

_INVENTORY: dict[str, dict[str, float | int]] = {
    "WIDGET-1": {"quantity": 120, "unit_price": 12.50},
    "GADGET-7": {"quantity": 0, "unit_price": 48.00},
    "COOLER-3": {"quantity": 14, "unit_price": 210.00},
}
_reservation_ids = itertools.count(1000)


def get_inventory(sku: str) -> str:
    """Return stock level and unit price for ``sku``.

    Args:
        sku: Stock-keeping unit, for example ``WIDGET-1``.
    """
    record = _INVENTORY.get(sku.upper())
    if record is None:
        return f"Unknown SKU '{sku}'."
    return (
        f"{sku.upper()}: {record['quantity']} units in stock at "
        f"${record['unit_price']:.2f} each."
    )


def reserve_stock(sku: str, quantity: int, customer_id: str) -> str:
    """Reserve ``quantity`` units of ``sku`` for ``customer_id``.

    Args:
        sku: Stock-keeping unit to reserve.
        quantity: Units to reserve (must not exceed stock).
        customer_id: Customer reference, for example ``C-42``.
    """
    record = _INVENTORY.get(sku.upper())
    if record is None:
        return f"Unknown SKU '{sku}'."
    if quantity > int(record["quantity"]):
        return f"Only {record['quantity']} units of {sku.upper()} available."
    record["quantity"] = int(record["quantity"]) - quantity
    reservation_id = f"R-{next(_reservation_ids)}"
    return (
        f"Reserved {quantity} x {sku.upper()} for {customer_id}; "
        f"reservation id {reservation_id}."
    )


def purge_inventory() -> str:
    """Delete every inventory record. Disallowed in agent.yaml; never callable."""
    _INVENTORY.clear()
    return "Inventory purged."
