"""Advanced runtime 03: hooks control a deployment tool.

Scenario
--------
An LLM tries to deploy to production. A `TOOL_BEFORE` hook rewrites the target
to staging unless a human approval flag is present. A `TOOL_AFTER` hook records
the deployment audit state in SQLite.

Note
----
This example calls your configured real LLM and asks it to invoke a tool.
Make sure `.env` is configured and be mindful of provider usage/cost.

Run:
    python examples/advanced_runtime/03_hooks_control_flow.py
"""

from __future__ import annotations

import asyncio

from _helpers import make_llm

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
    loop = StatefulAgentLoop(
        llm=make_llm(max_tokens=512, temperature=0.0),
        store=store,
        tool_registry=registry,
        hooks=hooks,
        config=AgentLoopConfig(
            system_prompt=(
                "You operate deployments. Call deploy_service with service='checkout' "
                "and target='production'. After the tool returns, report the actual target used."
            ),
            max_rounds=4,
            compactor=None,
        ),
    )
    sid = loop.new_session()

    result = await loop.send("Deploy checkout.", session_id=sid)
    print(result.final_text)
    print("audit:", store.get_runtime_state(sid, "deploy_audit"))
    return result.final_text


if __name__ == "__main__":
    asyncio.run(main())
