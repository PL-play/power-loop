"""Bounded, observable lifecycle notifications; never execution policy."""
import asyncio
import logging

from power_loop.contracts.hooks import HookContext

log = logging.getLogger(__name__)


async def notify(hooks, point, values, *, timeout_s=1.0):
    if hooks is None:
        return
    try:
        async with asyncio.timeout(timeout_s):
            await hooks.run_async(point, HookContext(values))
    except Exception:
        log.exception("execution notification failed: %s", point)
