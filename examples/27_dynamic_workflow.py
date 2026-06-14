"""27 · 动态工作流 / Dynamic workflow: declarative JSON DSL + deterministic engine

What you learn / 你将学到
--------------------------
- 用一份**声明式 WorkflowSpec JSON** 描述确定性控制流（sequence / foreach），
  叶子是普通子代理（每个一个 AgentSpec）
  / describe deterministic control flow (sequence / foreach) with a declarative
  ``WorkflowSpec`` JSON whose leaves are ordinary sub-agents (one ``AgentSpec`` each)
- ``create_workflow(spec, parent_loop=loop)`` 创建即校验（错误一次性聚合报出），
  ``await wf.run()`` 在进程内确定性执行
  / ``create_workflow`` validates on creation (all problems aggregated),
  ``await wf.run()`` interprets it deterministically in-process
- 用 ``output_schema`` 让上游 agent 产出结构化数据，下游用 ``items_from:"plan.subtopics"``
  引用它做 ``foreach`` 扇出
  / an upstream agent's ``output_schema`` lets a downstream ``foreach`` fan out over
  ``items_from: "plan.subtopics"``
- 这是 **code-driven** 确定性编排，区别于 ``spawn_agent``/``run_agent`` 的
  **model-driven** 即兴委派
  / this is *code-driven* deterministic orchestration, vs the *model-driven*
  ad-hoc delegation of ``spawn_agent`` / ``run_agent``

也可把 ``create_workflow`` 注册成 LLM 工具（``register_workflow_tools``），让主
agent 自己产出 WorkflowSpec JSON 提交执行——见文件末尾注释。

Run / 运行
----------
    python examples/27_dynamic_workflow.py        # needs POWER_LOOP_* env
"""

from __future__ import annotations

import asyncio

from _helpers import make_llm

from power_loop import AgentLoopConfig, StatefulAgentLoop
from power_loop.workflow import create_workflow

SPEC = {
    "name": "research_and_summarize",
    "input": "the Japanese tea ceremony",
    "budget": {"max_tokens": 200_000, "stop_at_remaining_pct": 5},
    "root": {
        "type": "sequence",
        "steps": [
            {
                "type": "agent",
                "id": "plan",
                "spec": {
                    "name": "planner",
                    "system_prompt": "Break the topic into exactly 3 short subtopics.",
                },
                "input": "Topic: {{input}}",
                # Structured output so downstream foreach can read plan.subtopics.
                "output_schema": {
                    "name": "Plan",
                    "schema": {
                        "type": "object",
                        "required": ["subtopics"],
                        "properties": {
                            "subtopics": {"type": "array", "items": {"type": "string"}}
                        },
                    },
                },
            },
            {
                "type": "foreach",
                "id": "research",
                "items_from": "plan.subtopics",
                "as": "subtopic",
                "parallel": True,
                "max_concurrency": 3,
                "body": {
                    "type": "agent",
                    "id": "researcher",
                    "spec": {
                        "name": "researcher",
                        "system_prompt": "Write 2-3 factual sentences on the given subtopic.",
                    },
                    "input": "Subtopic: {{subtopic}}",
                },
            },
            {
                "type": "agent",
                "id": "synthesize",
                "spec": {
                    "name": "writer",
                    "system_prompt": "Synthesize the research notes into one tight paragraph.",
                },
                "inputs_from": ["research"],
            },
        ],
    },
}


async def main() -> str:
    loop = StatefulAgentLoop(
        llm=make_llm(),
        db_path=":memory:",
        config=AgentLoopConfig(system_prompt="orchestrator", max_rounds=4),
    )

    # Validated on creation; an invalid spec raises WorkflowSpecError listing
    # every problem at once (good for an LLM to read and repair).
    wf = create_workflow(SPEC, parent_loop=loop)
    result = await wf.run()

    print(f"status     : {result.status}")
    print(f"subtopics  : {result.results['plan'].payload}")
    print(f"tokens     : {result.usage.get('total_tokens')}")
    print("\n--- final synthesis ---")
    print(result.results["synthesize"].text)
    print("\n--- run summary ---")
    print(result.summary())
    return result.results["synthesize"].text


# ── Letting the main agent author workflows itself (LLM-facing) ──────────────
#
#   from power_loop import create_default_tool_registry
#   from power_loop.workflow import register_workflow_tools
#
#   registry = create_default_tool_registry(preset="core")
#   register_workflow_tools(registry)        # adds create_workflow + workflow_status
#   loop = StatefulAgentLoop(llm=make_llm(), db_path=":memory:", tool_registry=registry)
#   # Now the model can emit a WorkflowSpec JSON and call create_workflow itself.
#   # (Don't grant this tool to a workflow's own leaf agents — nested runs are
#   #  refused in this tier.)


# ── Detached execution + completion wake (middle tier) ───────────────────────
#
#   from power_loop import TimerRunner
#   from power_loop.workflow import create_workflow, register_wake_guard, get_workflow
#
#   register_wake_guard(loop)                 # dedupe at-least-once wake timers
#   await TimerRunner(loop).start()           # REQUIRED: delivers the wake
#   parent_sid = loop.new_session()
#   wf = create_workflow(SPEC, parent_loop=loop, parent_session_id=parent_sid)
#   handle = await wf.start(detached=True)    # returns immediately; runs in background
#   # ... the parent agent can pass_turn now; it is woken with the result when done.
#   # Inspect any time:
#   #   get_workflow(loop, parent_sid, handle.run_id, detail=True)  -> status + steps + usage
#   #   list_workflows(loop, parent_sid)                            -> all runs


if __name__ == "__main__":
    asyncio.run(main())
