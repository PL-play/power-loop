"""Advanced runtime 03: hooks control a deployment tool.
Uses an offline model and the current Runtime factory; no real deployment occurs.
"""

from __future__ import annotations

import asyncio
import time

from _helpers import demo_runtime

from power_loop import (
    AgentHooks,
    HookPoint,
    LLMResponse,
    LLMService,
    ToolDefinition,
    ToolRegistry,
    get_tool_runtime_context,
)
from power_loop.runtime.pricing import Free


def deploy_service(service: str, target: str) -> str:
    return f"deployed {service} to {target}"


class ToolModel(LLMService):
    async def complete(self, request, **kwargs):
        results = [m['content'] for m in request.messages if m.get('role') == 'tool']
        if results:
            return LLMResponse(raw_text=results[-1])
        return LLMResponse(raw_text='', tool_calls=[{'id':'operation','type':'function','function':{
            'name':'deploy_service','arguments':'{"service":"checkout","target":"production"}'}}])
    async def close(self):
        pass

async def main() -> str:
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            pricing=Free("local simulated action"),
            name="deploy_service",
            description="Deploy a service to an environment.",
            input_schema={
                "type": "object",
                "properties": {
                    "service": {"type": "string"},
                    "target": {"type": "string"},
                },
                "required": ["service", "target"],
            },
            required_params=("service", "target"),
        ),
        deploy_service,
    )

    hooks = AgentHooks()

    def guard_production(ctx) -> None:
        if ctx.tool_name == "deploy_service" and ctx.tool_args.get("target") == "production":
            ctx.tool_args["target"] = "staging"

    async def audit_deploy(ctx) -> None:
        if ctx.tool_name == "deploy_service":
            runtime = get_tool_runtime_context(required=True)
            await runtime.store.set_runtime_state(
                runtime.session_id,
                "deploy_audit",
                {"args": dict(ctx.tool_args), "output": ctx.output},
            )

    hooks.register(HookPoint.TOOL_BEFORE, guard_production)
    hooks.register(HookPoint.TOOL_AFTER, audit_deploy)

    async with demo_runtime(ToolModel(), registry, hooks=hooks) as (runtime, loop, target, actor):
        result = await loop.send(target, 'Run simulated operation', message_id='operation', actor=actor,
            deadline_ms=int(time.time()*1000)+10000)
        assert result.status == 'completed'
        audit = await runtime.store.get_runtime_state(target.session_id, 'deploy_audit')
        assert audit['args']['target'] == 'staging' and 'staging' in result.final_text
        print(result.final_text, audit)
        return result.final_text

if __name__ == '__main__':
    asyncio.run(main())
