"""Cost sink protocol for batched cost reporting.

This protocol defines how cost items (model calls, tools, subagents, etc.) are
reported and persisted. The default implementation batches costs in memory and
flushes periodically, with limited retries and graceful degradation on failure.
"""
from __future__ import annotations

from decimal import Decimal
from typing import Protocol, Any
from dataclasses import dataclass


@dataclass
class CostItem:
    """A single cost item to be recorded."""

    run_id: str  # session_id, job_id, or other run identifier
    kind: str  # "model", "tool", "fold", "subagent", "workflow"
    amount: Decimal  # cost amount
    currency: str  # "USD", "credits", etc.
    meta: dict[str, Any]  # additional metadata (model name, tool name, tokens, etc.)
    created_at: int  # millisecond timestamp


class CostSink(Protocol):
    """Protocol for batched cost recording.

    Implementations buffer costs in memory and flush periodically (e.g., every 5
    seconds or when buffer reaches 32 items / 1MB). Failed writes are retried a
    limited number of times (e.g., 3), then discarded with a log warning.

    Cost recording never blocks the main execution path. If the sink is unavailable,
    costs are dropped after retry exhaustion.
    """

    async def record_costs(self, items: list[CostItem]) -> None:
        """Record a batch of cost items.

        This may buffer items in memory and defer actual persistence. Call
        ``flush()`` to force immediate persistence of all buffered items.

        Args:
            items: List of cost items to record
        """
        ...

    async def flush(self) -> None:
        """Flush all buffered cost items to persistent storage.

        This is called automatically on periodic timer or when buffer is full.
        Also called explicitly at critical boundaries (end of send, session close).

        Failures are logged but do not raise exceptions.
        """
        ...

    async def close(self) -> None:
        """Close the sink, flushing any remaining buffered items.

        After close, further record_costs calls are ignored.
        """
        ...
