"""23 · Configure session-scoped visible tools and system_prompt via RuntimeDescriptor.

``runtime.client.configure_session(..., descriptor=RuntimeDescriptor(..., config=...))``
sets ``tools`` and ``system_prompt`` on the session descriptor. When the session is
idle, the updated configuration applies immediately to subsequent ``send`` turns on
that session (when running, it is staged for the next claimed turn).

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
    RuntimeDescriptor,
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
    def __init__(self) -> None:
        self.seen_requests: list[tuple[list[str], str]] = []

    async def complete(self, request, **kwargs):
        names = [t['function']['name'] for t in request.tools if t['function']['name'] != 'pass_turn']
        prompt = request.system_prompt or ''
        self.seen_requests.append((names, prompt))
        assert len(names) == 1, names
        if names == ['get_weather']:
            assert 'Terse finance bot' not in prompt, prompt
        elif names == ['get_stock']:
            assert 'Terse finance bot' in prompt, prompt
        else:
            raise AssertionError(f'unexpected visible tools: {names!r}')
        return LLMResponse(raw_text=','.join(names))

    async def close(self):
        pass


def _session_descriptor(config: dict) -> RuntimeDescriptor:
    return RuntimeDescriptor(
        2,
        'agent',
        'example@215',
        None,
        'example.tools@215',
        'builtin.none@215',
        'builtin.none@215',
        config,
        {},
    )


async def main() -> None:
    model = VisibleModel()
    async with demo_runtime(model, _build_registry()) as (runtime, loop, target, actor):
        configs = [
            ('get_weather', {'tools': ['get_weather']}),
            ('get_stock', {'tools': ['get_stock'], 'system_prompt': 'Terse finance bot'}),
        ]
        for name, cfg in configs:
            await runtime.client.configure_session(
                target,
                actor=actor,
                descriptor=_session_descriptor(cfg),
            )
            result = await loop.send(
                target,
                'Show available tools',
                message_id=name,
                actor=actor,
                deadline_ms=int(time.time() * 1000) + 10000,
            )
            assert result.status == 'completed' and result.final_text == name
            print(name, result.final_text)

        assert len(model.seen_requests) == 2
        assert model.seen_requests[0][0] == ['get_weather']
        assert 'Terse finance bot' not in model.seen_requests[0][1]
        assert model.seen_requests[1][0] == ['get_stock']
        assert 'Terse finance bot' in model.seen_requests[1][1]

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
            out_a = await unbound.invoke_async("read_file", {"path": "profile.txt"})
            assert "tenant A" in str(out_a), out_a
            print("unbound A:", out_a)
        with runtime_env_context(RuntimeEnv(workspace_dir=tenant_b)):
            out_b = await unbound.invoke_async("read_file", {"path": "profile.txt"})
            assert "tenant B" in str(out_b), out_b
            print("unbound B:", out_b)


if __name__ == "__main__":
    asyncio.run(main())
