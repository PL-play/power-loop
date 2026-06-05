"""10 · 并发会话 + 异步审批队列

What you learn
--------------
- 一个 ``StatefulAgentLoop`` 实例**并发驱动多个 session**（每 session 一把
  ``asyncio.Lock``，互不阻塞）
- 用 ``asyncio.Queue`` 把工具审批从 hook 异步派发出去，由 ⼀个独立 worker
  消费 → 模拟生产里 "UI 等用户点击"，多个 session 并行排队
- 拒绝路径仍走 ``HookDirective.SKIP``——同 example 07
- 关键：审批 worker 决定速度，主循环**真的会等**，没有 timeout / 轮询

适用场景
--------
- 在线 Web 服务里多个用户同时聊天，每个的危险命令都要本人确认
- CLI / IDE 插件里只有一个用户界面，但后台跑多个 session

Run
---
    python examples/10_async_approval_queue.py
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from _helpers import make_llm

from power_loop import (
    AgentHooks,
    AgentLoopConfig,
    HookDirective,
    HookPoint,
    StatefulAgentLoop,
    ToolDefinition,
    ToolRegistry,
)
from power_loop.contracts.hook_contexts import ToolBeforeCtx
from power_loop.core.agent_context import get_session_id

# ── 1. 工具：一个能记录每个 session 在干什么的 bash 替身 ────────────────


def fake_bash(**kwargs) -> str:
    cmd = str(kwargs.get("command") or "")
    return f"(executed) {cmd}"


BASH_TOOL = ToolDefinition(
    name="bash",
    description="Run a shell command. Param: command (string).",
    input_schema={
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
    },
    required_params=("command",),
)


# ── 2. 审批请求 / 队列 ───────────────────────────────────────────────────


@dataclass
class ApprovalRequest:
    session_id: str
    command: str
    response: asyncio.Future[bool] = field(default_factory=asyncio.Future)


def build_approval_hook(queue: asyncio.Queue[ApprovalRequest]) -> AgentHooks:
    """每次 bash 调用都派一个 ApprovalRequest 到队列，等 future。"""
    hooks = AgentHooks()

    async def gate(ctx: ToolBeforeCtx) -> None:
        if ctx.tool_name != "bash":
            return
        cmd = str(ctx.tool_args.get("command") or "")
        sid = get_session_id() or "?"
        # 安全命令直接放行（避免无意义打扰审批人）
        if cmd.strip().startswith(("ls", "pwd", "echo", "cat ")):
            print(f"  [gate] auto-approve {sid[-6:]}: {cmd!r}")
            return
        req = ApprovalRequest(session_id=sid, command=cmd)
        await queue.put(req)
        approved = await req.response                 # 真等
        if not approved:
            ctx.output = f"[denied: {cmd!r}]"
            ctx.directive = HookDirective.SKIP

    hooks.register(HookPoint.TOOL_BEFORE, gate)
    return hooks


# ── 3. 审批 worker：决定每条请求是 yes / no ──────────────────────────────


_STOP = object()  # sentinel pushed into queue to tell the worker to exit


async def approval_worker(
    queue: asyncio.Queue,
    *,
    decide: Callable[[str], bool],
) -> int:
    """从队列里出请求 → 模拟人工决定 → fulfill future。

    ``decide`` 是策略：在真实 UI 里换成 "点同意按钮 → 返回 True"；
    本例用闭包决定行为。Worker 在收到 ``_STOP`` 时退出——比超时退出可靠，
    避免和并发 session 的 "尚未到达 gate" 期形成 race。
    """
    handled = 0
    while True:
        item = await queue.get()
        if item is _STOP:
            return handled
        req: ApprovalRequest = item
        await asyncio.sleep(0.05)                   # 模拟 UI 思考时间
        verdict = decide(req.command)
        print(f"  [worker] {req.session_id[-6:]} → {req.command!r}: "
              f"{'APPROVE' if verdict else 'DENY'}")
        req.response.set_result(verdict)
        handled += 1


# ── 4. 三个 session 并发跑 ──────────────────────────────────────────────


async def drive_session(
    loop: StatefulAgentLoop,
    label: str,
    user_input: str,
) -> dict[str, Any]:
    r = await loop.send(user_input)
    print(f"[{label}] done: status={r.status}, rounds={r.rounds}")
    return {"label": label, "sid": r.session_id, "text": r.final_text}


async def main() -> list[dict[str, Any]]:
    queue: asyncio.Queue[ApprovalRequest] = asyncio.Queue()

    registry = ToolRegistry()
    registry.register(BASH_TOOL, fake_bash)

    loop = StatefulAgentLoop(
        llm=make_llm(max_tokens=300, temperature=0),
        db_path=":memory:",
        tool_registry=registry,
        hooks=build_approval_hook(queue),
        config=AgentLoopConfig(
            system_prompt=(
                "You have a `bash` tool (param: `command`). Call it for any "
                "shell action the user asks. If a tool call is denied, "
                "summarize what failed and stop — do NOT retry."
            ),
            max_rounds=4,
            compactor=None,
        ),
    )

    # 决策策略：包含 'rm' 的命令一律拒绝，其它批准
    def decide(cmd: str) -> bool:
        return "rm" not in cmd

    # 启动 worker 与三个 session 并行
    worker = asyncio.create_task(approval_worker(queue, decide=decide))
    results = await asyncio.gather(
        drive_session(loop, "S1", "List the project files."),
        drive_session(loop, "S2", "Delete the file named cache.tmp."),
        drive_session(loop, "S3",
                       "Print the current working directory using bash."),
    )
    # 所有 session 跑完才停 worker，避免 "session 还没到 gate 就关 worker"
    # 的 race
    await queue.put(_STOP)
    handled = await worker
    print(f"\n[worker] handled {handled} approval request(s)")
    print(f"[sessions] {len(results)} sessions completed")
    return results


if __name__ == "__main__":
    asyncio.run(main())
