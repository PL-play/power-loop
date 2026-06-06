"""10 · 并发会话 + 异步审批队列 / Concurrent sessions + async approval queue

What you learn / 你将学到
--------------------------
- 一个 ``StatefulAgentLoop`` 实例**并发驱动多个 session**（每 session 一把
  ``asyncio.Lock``，互不阻塞）
  / a single ``StatefulAgentLoop`` instance **concurrently drives multiple sessions**
  (one ``asyncio.Lock`` per session, no mutual blocking)
- 用 ``asyncio.Queue`` 把工具审批从 hook 异步派发出去，由一个独立 worker
  消费 → 模拟生产里 "UI 等用户点击"，多个 session 并行排队
  / uses ``asyncio.Queue`` to dispatch tool approvals from hooks to an independent
  worker → simulates "UI waits for user click" in production, multiple sessions queue in parallel
- 拒绝路径仍走 ``HookDirective.SKIP``——同 example 07
  / denial path still uses ``HookDirective.SKIP`` — same as example 07
- 关键：审批 worker 决定速度，主循环**真的会等**，没有 timeout / 轮询
  / key point: the approval worker decides the pace; the main loop **actually waits** — no timeout / polling

适用场景 / Use cases
---------------------
- 在线 Web 服务里多个用户同时聊天，每个的危险命令都要本人确认
  / multiple users chatting simultaneously in a web service, each user's dangerous commands require their own confirmation
- CLI / IDE 插件里只有一个用户界面，但后台跑多个 session
  / CLI / IDE plugins with a single UI but multiple background sessions

Run / 运行
----------
    python examples/10_concurrent_sessions.py
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
#    Tool: a bash stub that records what each session is doing


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


# ── 2. 审批请求 / 队列 / Approval request / queue ─────────────────────────


@dataclass
class ApprovalRequest:
    session_id: str
    command: str
    response: asyncio.Future[bool] = field(default_factory=asyncio.Future)


@dataclass(frozen=True)
class _Stop:
    pass


ApprovalQueueItem = ApprovalRequest | _Stop


def build_approval_hook(queue: asyncio.Queue[ApprovalQueueItem]) -> AgentHooks:
    """每次 bash 调用都派一个 ApprovalRequest 到队列，等 future。
    / Every bash call dispatches an ApprovalRequest to the queue and awaits the future."""
    hooks = AgentHooks()

    async def gate(ctx: ToolBeforeCtx) -> None:
        if ctx.tool_name != "bash":
            return
        cmd = str(ctx.tool_args.get("command") or "")
        sid = get_session_id() or "?"
        # 安全命令直接放行（避免无意义打扰审批人）
        # Auto-approve safe commands (avoid unnecessarily bothering the approver)
        if cmd.strip().startswith(("ls", "pwd", "echo", "cat ")):
            print(f"  [gate] auto-approve {sid[-6:]}: {cmd!r}")
            return
        req = ApprovalRequest(session_id=sid, command=cmd)
        await queue.put(req)
        approved = await req.response                 # 真等 / actually waits
        if not approved:
            ctx.output = f"[denied: {cmd!r}]"
            ctx.directive = HookDirective.SKIP

    hooks.register(HookPoint.TOOL_BEFORE, gate)
    return hooks


# ── 3. 审批 worker：决定每条请求是 yes / no ──────────────────────────────
#    Approval worker: decides yes / no for each request


_STOP = _Stop()  # sentinel pushed into queue to tell the worker to exit


async def approval_worker(
    queue: asyncio.Queue[ApprovalQueueItem],
    *,
    decide: Callable[[str], bool],
) -> int:
    """从队列里出请求 → 模拟人工决定 → fulfill future。

    ``decide`` 是策略：在真实 UI 里换成 "点同意按钮 → 返回 True"；
    本例用闭包决定行为。Worker 在收到 ``_STOP`` 时退出——比超时退出可靠，
    避免和并发 session 的 "尚未到达 gate" 期形成 race。

    / Dequeues requests → simulates human decision → fulfills future.

    ``decide`` is the policy: in a real UI, replace with "click approve → return True".
    Worker exits on ``_STOP`` — more reliable than timeout, avoids races with
    concurrent sessions that haven't reached the gate yet.
    """
    handled = 0
    while True:
        item = await queue.get()
        if isinstance(item, _Stop):
            return handled
        req = item
        await asyncio.sleep(0.05)                   # 模拟 UI 思考时间 / simulate UI thinking time
        verdict = decide(req.command)
        print(f"  [worker] {req.session_id[-6:]} → {req.command!r}: "
              f"{'APPROVE' if verdict else 'DENY'}")
        req.response.set_result(verdict)
        handled += 1


# ── 4. 三个 session 并发跑 / Run three sessions concurrently ─────────────


async def drive_session(
    loop: StatefulAgentLoop,
    label: str,
    user_input: str,
) -> dict[str, Any]:
    sid = loop.new_session(metadata={"label": label})
    r = await loop.send(user_input, session_id=sid)
    print(f"[{label}] done: status={r.status}, rounds={r.rounds}")
    return {"label": label, "sid": r.session_id, "text": r.final_text}


async def main() -> list[dict[str, Any]]:
    queue: asyncio.Queue[ApprovalQueueItem] = asyncio.Queue()

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
    # Decision policy: deny commands containing 'rm', approve everything else
    def decide(cmd: str) -> bool:
        return "rm" not in cmd

    # 启动 worker 与三个 session 并行
    # Start worker and three sessions in parallel
    worker = asyncio.create_task(approval_worker(queue, decide=decide))
    results = await asyncio.gather(
        drive_session(loop, "S1", "List the project files."),
        drive_session(loop, "S2", "Delete the file named cache.tmp."),
        drive_session(loop, "S3",
                       "Print the current working directory using bash."),
    )
    # 所有 session 跑完才停 worker，避免 "session 还没到 gate 就关 worker" 的 race
    # Stop worker only after all sessions finish — avoids race where worker shuts down before a session reaches the gate
    await queue.put(_STOP)
    handled = await worker
    print(f"\n[worker] handled {handled} approval request(s)")
    print(f"[sessions] {len(results)} sessions completed")
    return list(results)


if __name__ == "__main__":
    asyncio.run(main())
