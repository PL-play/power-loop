# Tool calling

SDK 11.0.0 · schema 15 · protocol 42 · descriptor 1.

Define the tool schema and handler, register it, and return the registry from ExecutionEnvironment. The runtime validates arguments, checks allowlists and invokes before/after hooks. Business permissions and budget remain in the authoritative host backend.

The following source is executable from this checkout. It imports the shared example composition helper, [examples/_runtime.py](../../../examples/_runtime.py); keep that file alongside the example. The helper starts and closes a real Runtime. For independent application setup, follow [Quickstart](../user-guide/quickstart.md).

```bash
.venv/bin/python examples/02_tool_calling.py
```

```python
"""02 · Custom free tool through the Runtime; the demo model is deterministic."""
from __future__ import annotations

import asyncio
import time

from _runtime import demo_runtime

from power_loop import (
    AgentLoopConfig,
    LLMResponse,
    LLMService,
    ToolDefinition,
    ToolRegistry,
)
from power_loop.runtime.pricing import Free

# ── 1. 定义工具 / Define tools ──────────────────────────────────────────

DISHES = {
    "tokyo": "sushi",
    "rome": "cacio e pepe",
    "lima": "ceviche",
    "bangkok": "pad thai",
}


def lookup_dish(**kwargs) -> str:
    """Sync handler. handler 也可以是 ``async def``，ToolRegistry 自动适配。
    / Sync handler. Handlers can also be ``async def`` — ToolRegistry adapts automatically."""
    city = str(kwargs.get("city") or "").strip().lower()
    return DISHES.get(city, f"No data for {city!r}")


LOOKUP_TOOL = ToolDefinition(
    pricing=Free("local example tool"),
    name="lookup_dish",
    description="Return the signature local dish for a given city.",
    input_schema={
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
    required_params=("city",),   # 校验在 handler 之前，缺参直接报错给模型
                                  # validation runs before handler; missing params → error to model
)


# ── 2. 注册并跑循环 / Register and run the loop ─────────────────────────


class DishModel(LLMService):
    async def complete(self, request, **kwargs):
        outputs = [m['content'] for m in request.messages if m['role'] == 'tool']
        if outputs:
            return LLMResponse(raw_text=outputs[-1])
        return LLMResponse(raw_text='', tool_calls=[{'id':'dish', 'type':'function', 'function':{
            'name':'lookup_dish', 'arguments':'{"city":"Bangkok"}'}}])

    async def close(self):
        pass


async def main():
    registry = ToolRegistry()
    registry.register(LOOKUP_TOOL, lookup_dish)
    async with demo_runtime(DishModel(), registry, config=AgentLoopConfig(max_rounds=3, compactor=None)) as (runtime, loop, target, actor):
        result = await loop.send(target, "What is Bangkok's signature dish?", message_id='dish-1', actor=actor,
                                 deadline_ms=int(time.time()*1000)+10000)
        assert result.status == 'completed' and result.final_text == 'pad thai'
        print(result.final_text)
        return result.final_text


if __name__ == '__main__':
    asyncio.run(main())
```

[tools](../user-guide/tools.md) · [API](../api/index.md)
