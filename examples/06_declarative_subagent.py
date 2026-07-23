"""06 · 子代理覆盖与声明式 spec / Sub-agent overrides & declarative spec

What you learn / 你将学到
--------------------------
- 4.0 起 LLM 侧只有一个 meta-tool：``spawn_agent``（原 ``run_agent`` 已并入）
  / since 4.0 the LLM sees a single meta-tool: ``spawn_agent`` (the former
  ``run_agent`` was merged into it)
- ``spawn_agent`` 的可选覆盖参数：``system_prompt`` / ``tools`` / ``max_rounds``
  / optional per-call overrides: ``system_prompt`` / ``tools`` / ``max_rounds``
- ``tools`` 字段是父 registry 的白名单：父限定子可见的能力集，防过权
  / ``tools`` is a whitelist of parent registry: limits child's visible capabilities
- 声明式路径 = 宿主代码直接调 ``run_agent_spec(AgentSpec, ...)``（Python API，
  绕过 LLM-as-driver，适合测试 / 编排框架）
  / the declarative path = host code calls ``run_agent_spec(AgentSpec, ...)``
  directly (bypass LLM-as-driver, good for tests / orchestration)
- ``AgentSpec`` 是 strict-schema：未知字段 / 非法 lifecycle / max_rounds 越界
  → ``AgentSpecError``，调用直接报错而非静默忽略
  / strict-schema: unknown fields / invalid lifecycle / out-of-range max_rounds
  → ``AgentSpecError`` (raises, not silently ignored)

Run / 运行
----------
    python examples/06_declarative_subagent.py
"""

from __future__ import annotations

import asyncio
import json

from _helpers import make_llm

from power_loop import (
    AgentLoopConfig,
    AgentSpec,
    SessionStore,
    StatefulAgentLoop,
    SubagentLifecycle,
    ToolDefinition,
    ToolRegistry,
    register_spawn_agent,
)
from power_loop.core.agent_context import (
    reset_current_loop,
    reset_session_id,
    set_current_loop,
    set_session_id,
)
from power_loop.runtime.spec import run_agent_spec

# ── 1. 父 registry：提供 calculator + spawn meta-tool ─────────────────────
#    Parent registry: calculator + spawn meta-tool


def calculator(**kwargs) -> str:
    expr = str(kwargs.get("expr") or "").strip()
    # 用 ast 解析比 eval 安全；这里 demo 偷懒
    # ast parsing is safer than eval; this demo takes a shortcut
    try:
        return str(eval(expr, {"__builtins__": {}}, {}))     # noqa: S307
    except Exception as exc:
        return f"calc error: {exc}"


def _build_parent_registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(
        ToolDefinition(
            name="calc",
            description="Evaluate a simple arithmetic expression.",
            input_schema={
                "type": "object",
                "properties": {"expr": {"type": "string"}},
                "required": ["expr"],
            },
            required_params=("expr",),
        ),
        calculator,
    )
    # 注入 spawn_agent meta-tool（4.0 起 LLM 侧唯一的子代理入口）
    # Inject the spawn_agent meta-tool (the only LLM-facing entry since 4.0)
    register_spawn_agent(reg)
    return reg


# ── 2. 父 LLM 通过 spawn_agent meta-tool 委托（带覆盖参数）────────────────
#    Parent LLM delegates via the spawn_agent meta-tool (with overrides)


async def via_meta_tool() -> str:
    """让父 LLM 调 spawn_agent，并传 system_prompt / tools / max_rounds 覆盖。
    / Let the parent LLM call spawn_agent with system_prompt / tools /
    max_rounds overrides."""
    store = await SessionStore.open(":memory:")
    try:
        loop = StatefulAgentLoop(
            llm=make_llm(max_tokens=512, temperature=0),
            store=store,
            tool_registry=_build_parent_registry(),
            config=AgentLoopConfig(
                system_prompt=(
                    "You are an orchestrator. For math sub-tasks, call "
                    "`spawn_agent` with: a `system_prompt` telling the "
                    "sub-agent to compute and reply with the number only, "
                    "a `tools` list whitelisting ['calc'], and "
                    "`max_rounds` set to 3. Report the sub-agent's answer."
                ),
                max_rounds=6,
                compactor=None,
            ),
        )
        sid = await loop.new_session()
        r = await loop.send(
            "Delegate this to a sub-agent and report back: "
            "what is (17 + 25) × 3?",
            session_id=sid,
        )
        print("[via_meta_tool]")
        print(f"  status : {r.status}, rounds: {r.rounds}")
        print(f"  reply  : {r.final_text}\n")
        return r.final_text
    finally:
        await store.close()


# ── 3. 声明式：宿主代码直接调 run_agent_spec，自己拼 AgentSpec ────────────
#    Declarative: host code calls run_agent_spec with its own AgentSpec


async def direct_call() -> str:
    """从 Python 代码直接拼 AgentSpec 并跑——适合测试 / 编排框架。
    / Build AgentSpec from Python code and run — ideal for tests / orchestration."""
    store = await SessionStore.open(":memory:")
    try:
        parent_loop = StatefulAgentLoop(
            llm=make_llm(max_tokens=256, temperature=0),
            store=store,
            tool_registry=_build_parent_registry(),
            config=AgentLoopConfig(
                system_prompt="Reply briefly.", max_rounds=1, compactor=None,
            ),
        )
        # 先跑一轮父，建立 parent_session_id 给后面 run_agent_spec 用
        # Run one parent turn to establish parent_session_id for run_agent_spec
        parent_sid = await parent_loop.new_session()
        pr = await parent_loop.send("hi", session_id=parent_sid)

        spec = AgentSpec(
            name="math-helper",
            system_prompt="Compute the expression. Reply with the number only.",
            tools=["calc"],                                  # 只白名单 calc / whitelist only calc
            max_rounds=3,
            max_tokens=128,
            lifecycle=SubagentLifecycle.LINKED.value,         # 保留供审计 / preserve for audit
        )

        tok_loop = set_current_loop(parent_loop)
        tok_sid = set_session_id(pr.session_id)
        try:
            result = await run_agent_spec(spec, "What is 12 * 11?", parent_loop=parent_loop)
        finally:
            reset_current_loop(tok_loop)
            reset_session_id(tok_sid)

        print("[direct_call]")
        print(f"  child sid : {result['session_id']}  (LINKED → preserved)")
        print(f"  depth     : {result['depth']}")
        print(f"  status    : {result['status']}")
        print(f"  reply     : {result['final_text']}\n")
        print(f"  surviving children of parent: "
              f"{[c.session_id for c in await store.list_children(pr.session_id)]}")
        return result["final_text"]
    finally:
        await store.close()


# ── 4. AgentSpec strict-schema 演示 / Strict-schema demo ────────────────


def show_strict_schema() -> None:
    from power_loop import AgentSpecError

    print("[strict-schema]")

    # 未知字段 → 拒绝 / unknown field → rejected
    try:
        AgentSpec.from_json('{"name":"x","system_prompt":"y","evil":true}')
    except AgentSpecError as exc:
        print(f"  reject unknown field   : {exc}")

    # max_rounds 越界 → 拒绝 / out-of-range max_rounds → rejected
    try:
        AgentSpec(name="x", system_prompt="y", max_rounds=999)
    except AgentSpecError as exc:
        print(f"  reject max_rounds=999  : {exc}")

    # 合法 / valid
    spec = AgentSpec.from_dict({"name": "ok", "system_prompt": "p", "max_rounds": 3})
    print(f"  valid spec parsed      : name={spec.name!r}, rounds={spec.max_rounds}")
    print(f"  dump JSON              : {json.dumps({'name': spec.name, 'max_rounds': spec.max_rounds})}\n")


async def main() -> str:
    show_strict_schema()
    await direct_call()
    return await via_meta_tool()


if __name__ == "__main__":
    asyncio.run(main())
