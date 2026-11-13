"""Actual native callback returns become ordinary tool results, not job evidence."""
from __future__ import annotations

import json

from power_loop.agent.runtime_types import RuntimeRequestError


class NativeReturnPending(RuntimeRequestError):
    def __init__(self):
        super().__init__("native_return_checkpoint_pending")


async def invoke(pipeline, registered, arguments, slot=None, *, attempt=None, dispatch=None):
    from power_loop.contracts.tools import ToolResult

    from .tool_contracts import validate_registration

    validate_registration(pipeline.tool_registry, registered.definition.name, registered)
    result = (
        await dispatch() if dispatch is not None
        else await pipeline.tool_registry.invoke_async(registered.definition.name, arguments)
    )
    if isinstance(result, ToolResult):
        result = result.content
    return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)


__all__ = ["NativeReturnPending", "invoke"]
