"""Wait for tasks a native callback started before its return is recorded."""
from __future__ import annotations

import asyncio


async def join_children(slot, *, cancel=False):
    """The source callback cannot finish while its internal supplier tasks run."""
    while tasks := [task for task in slot.children if not task.done()]:
        if cancel:
            for task in tasks:
                task.cancel()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        if not cancel:
            for result in results:
                if isinstance(result, BaseException):
                    raise result
