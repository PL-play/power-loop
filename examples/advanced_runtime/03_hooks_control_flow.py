"""Advanced runtime 03: hooks control a deployment tool.

Scenario
--------
An LLM tries to deploy to production. A `TOOL_BEFORE` hook rewrites the target
to staging unless a human approval flag is present. A `TOOL_AFTER` hook records
the deployment audit state in SQLite.

Run:
    python examples/advanced_runtime/03_hooks_control_flow.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _runtime_helpers import ScriptedLLM, tool_response

from llm_client.interface import LLMResponse
from power_loop import (
    AgentHooks,
    AgentLoopConfig,
    HookPoint,
    SessionStore,
    StatefulAgentLoop,
    ToolDefinition,
    ToolRegistry,
    get_tool_runtime_context,
)


def deploy_service(service: str, target: str) -> str:
    return f"deployed {service} to {target}"


async def main() -> str:
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
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

    def audit_deploy(ctx) -> None:
        if ctx.tool_name == "deploy_service":
            runtime = get_tool_runtime_context(required=True)
            runtime.store.set_runtime_state(
                runtime.session_id,
                "deploy_audit",
                {"args": dict(ctx.tool_args), "output": ctx.output},
            )

    hooks.register(HookPoint.TOOL_BEFORE, guard_production)
    hooks.register(HookPoint.TOOL_AFTER, audit_deploy)

    store = SessionStore.open(":memory:")
    llm = ScriptedLLM(
        responses=[
            tool_response("tc-deploy", "deploy_service", '{"service":"checkout","target":"production"}'),
            LLMResponse(raw_text="Deployment redirected to staging."),
        ]
    )
    loop = StatefulAgentLoop(
        llm=llm,
        store=store,
        tool_registry=registry,
        hooks=hooks,
        config=AgentLoopConfig(system_prompt="You operate deployments.", max_rounds=3, compactor=None),
    )
    sid = loop.new_session()

    result = await loop.send("Deploy checkout.", session_id=sid)
    print(result.final_text)
    print("audit:", store.get_runtime_state(sid, "deploy_audit"))
    return result.final_text


if __name__ == "__main__":
    asyncio.run(main())
