"""23 · RequestOptions limits visible tools and overrides this input's prompt.

An unbound registry resolves workspace from the current factory environment.
"""

from __future__ import annotations

import asyncio
import tempfile
import time
from pathlib import Path

from _runtime import demo_runtime

from power_loop import (
    LLMResponse,
    LLMService,
    RequestOptions,
    RuntimeEnv,
    ToolDefinition,
    ToolRegistry,
    create_default_tool_registry,
    runtime_env_context,
)
from power_loop.runtime.pricing import Free


def _weather(**kwargs) -> str:
    return f"{kwargs.get('city', '?')}: sunny, 24°C"


def _stock(**kwargs) -> str:
    return f"{kwargs.get('ticker', '?')}: 123.45"


def _build_registry() -> ToolRegistry:
    r = ToolRegistry()
    r.register(
        ToolDefinition(
            pricing=Free("local example tool"),
            name="get_weather",
            description="Get current weather for a city.",
            input_schema={"type": "object", "properties": {"city": {"type": "string"}},
                          "required": ["city"]},
            required_params=("city",),
        ),
        _weather,
    )
    r.register(
        ToolDefinition(
            pricing=Free("local example tool"),
            name="get_stock",
            description="Get the current price for a stock ticker.",
            input_schema={"type": "object", "properties": {"ticker": {"type": "string"}},
                          "required": ["ticker"]},
            required_params=("ticker",),
        ),
        _stock,
    )
    return r


class VisibleModel(LLMService):
    async def complete(self, request, **kwargs):
        names = [t['function']['name'] for t in request.tools if t['function']['name'] != 'pass_turn']
        assert len(names) == 1
        if names == ['get_stock']:
            assert 'Terse finance bot' in request.system_prompt
        return LLMResponse(raw_text=','.join(names))

    async def close(self):
        pass


async def main() -> None:
    async with demo_runtime(VisibleModel(), _build_registry()) as (runtime, loop, target, actor):
        for name in ['get_weather', 'get_stock']:
            result = await loop.send(target, 'Show available tools', message_id=name, actor=actor,
                options=RequestOptions(tools=[name], system_prompt='Terse finance bot' if name == 'get_stock' else None),
                deadline_ms=int(time.time()*1000)+10000)
            assert result.status == 'completed' and result.final_text == name
            print(name, result.final_text)

    # One unbound default registry can safely resolve a different workspace
    # for each invocation. The host owns the RuntimeEnv boundary.
    unbound = create_default_tool_registry(include=["read_file"], bind=False)
    with tempfile.TemporaryDirectory() as root:
        tenant_a = Path(root) / "tenant-a"
        tenant_b = Path(root) / "tenant-b"
        tenant_a.mkdir()
        tenant_b.mkdir()
        (tenant_a / "profile.txt").write_text("tenant A", encoding="utf-8")
        (tenant_b / "profile.txt").write_text("tenant B", encoding="utf-8")

        with runtime_env_context(RuntimeEnv(workspace_dir=tenant_a)):
            print("unbound A:", await unbound.invoke_async("read_file", {"path": "profile.txt"}))
        with runtime_env_context(RuntimeEnv(workspace_dir=tenant_b)):
            print("unbound B:", await unbound.invoke_async("read_file", {"path": "profile.txt"}))


if __name__ == "__main__":
    asyncio.run(main())
