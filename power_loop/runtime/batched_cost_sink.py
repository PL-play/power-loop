"""Default in-memory batching cost sink implementation."""
from __future__ import annotations

import asyncio
import logging
from decimal import Decimal
from typing import Any

from .cost_sink import CostItem, CostSink

logger = logging.getLogger(__name__)


class BatchedCostSink:
    """Default cost sink with in-memory batching and periodic flush.

    Buffers cost items in memory and flushes them to the database in batches.
    This reduces SQL round-trips from O(n) cost items to O(1) batches.

    Features:
    - Batches up to 32 items or 1MB (whichever comes first)
    - Auto-flush every 5 seconds or on buffer full
    - Limited retries (3 attempts) on write failure
    - Graceful degradation: drops costs after retry exhaustion (logged as warning)
    - Non-blocking: never blocks main execution path

    Usage:
        sink = BatchedCostSink(store, max_items=32, flush_interval_s=5.0)
        await sink.record_costs([item1, item2, ...])
        await sink.flush()  # explicit flush at boundaries
        await sink.close()  # cleanup
    """

    def __init__(
        self,
        store: Any,  # TaskStore or similar with write_costs method
        *,
        max_items: int = 32,
        max_bytes: int = 1024 * 1024,  # 1MB
        flush_interval_s: float = 5.0,
        max_retries: int = 3,
    ):
        self._store = store
        self._max_items = max_items
        self._max_bytes = max_bytes
        self._flush_interval_s = flush_interval_s
        self._max_retries = max_retries

        self._buffer: list[CostItem] = []
        self._buffer_bytes = 0
        self._closed = False
        self._flush_task: asyncio.Task | None = None
        self._lock = asyncio.Lock()

    async def record_costs(self, items: list[CostItem]) -> None:
        """Buffer cost items and flush if buffer is full."""
        if self._closed:
            logger.warning("CostSink is closed, dropping %d cost items", len(items))
            return

        async with self._lock:
            self._buffer.extend(items)
            self._buffer_bytes += sum(self._estimate_size(item) for item in items)

            # Start auto-flush timer on first item
            if self._flush_task is None and self._buffer:
                self._flush_task = asyncio.create_task(self._auto_flush())

            # Flush immediately if buffer is full
            if len(self._buffer) >= self._max_items or self._buffer_bytes >= self._max_bytes:
                await self._flush_unlocked()

    async def flush(self) -> None:
        """Explicitly flush all buffered items."""
        if self._closed:
            return

        async with self._lock:
            await self._flush_unlocked()

    async def close(self) -> None:
        """Close sink and flush remaining items."""
        if self._closed:
            return

        self._closed = True

        # Cancel auto-flush task
        if self._flush_task is not None:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass

        # Final flush
        async with self._lock:
            await self._flush_unlocked()

    async def _flush_unlocked(self) -> None:
        """Flush buffer (must hold lock)."""
        if not self._buffer:
            return

        items = self._buffer[:]
        self._buffer.clear()
        self._buffer_bytes = 0

        # Cancel auto-flush task since buffer is now empty
        if self._flush_task is not None:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
            self._flush_task = None

        # Write with retries
        for attempt in range(self._max_retries):
            try:
                await self._store.write_costs(items)
                logger.debug("Flushed %d cost items (attempt %d)", len(items), attempt + 1)
                return
            except Exception as e:
                if attempt == self._max_retries - 1:
                    # Final retry failed, drop costs
                    logger.warning(
                        "Failed to write %d cost items after %d retries, dropping: %s",
                        len(items), self._max_retries, e
                    )
                else:
                    logger.debug("Cost write retry %d/%d failed: %s", attempt + 1, self._max_retries, e)
                    await asyncio.sleep(0.1 * (attempt + 1))  # exponential backoff

    async def _auto_flush(self) -> None:
        """Auto-flush task that runs periodically."""
        try:
            while not self._closed:
                await asyncio.sleep(self._flush_interval_s)
                async with self._lock:
                    await self._flush_unlocked()
        except asyncio.CancelledError:
            pass

    def _estimate_size(self, item: CostItem) -> int:
        """Estimate memory size of a cost item (bytes)."""
        # Rough estimate: 200 bytes base + metadata JSON size
        import json
        meta_size = len(json.dumps(item.meta)) if item.meta else 0
        return 200 + meta_size
