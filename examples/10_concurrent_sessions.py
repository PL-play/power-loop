"""10 · Three Runtime sessions and an independent asynchronous approval queue."""
from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from _runtime import demo_runtime

from power_loop import (
    AgentHooks,
    HookDirective,
    HookPoint,
    LLMResponse,
    LLMService,
    RuntimeDescriptor,
    ToolDefinition,
    ToolRegistry,
)
from power_loop.contracts.hook_contexts import ToolBeforeCtx
from power_loop.core.agent_context import get_session_id
from power_loop.runtime.pricing import Free

# ── 1. 工具：一个能记录每个 session 在干什么的 bash 替身 ────────────────
#    Tool: a bash stub that records what each session is doing


def fake_bash(**kwargs) -> str:
    cmd = str(kwargs.get("command") or "")
    return f"(executed) {cmd}"


BASH_TOOL = ToolDefinition(
    pricing=Free("local example tool"),
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


class CommandModel(LLMService):
    async def complete(self, request, **kwargs):
        outputs = [m['content'] for m in request.messages if m['role']=='tool']
        if outputs:
            return LLMResponse(raw_text=outputs[-1])
        command = next(m['content'] for m in request.messages if m['role']=='user')
        return LLMResponse(raw_text='', tool_calls=[{'id':'command','type':'function','function':{
            'name':'bash','arguments':json.dumps({'command':command})}}])

    async def close(self):
        pass


async def drive_session(runtime, loop, actor, label, command):
    target = await runtime.open_session(RuntimeDescriptor(1,'agent','example@1',None,
        'example.tools@1','builtin.none@1','builtin.none@1',{},{}), actor=actor)
    result = await loop.send(target, command, message_id=label, actor=actor,
                             deadline_ms=int(time.time()*1000)+10000)
    assert result.status == 'completed'
    return {'label':label, 'sid':target.session_id, 'text':result.final_text}


async def main():
    queue = asyncio.Queue()
    registry = ToolRegistry()
    registry.register(BASH_TOOL, fake_bash)
    approval = asyncio.create_task(approval_worker(queue, decide=lambda command: 'rm' not in command))
    try:
        async with demo_runtime(CommandModel(), registry, hooks=build_approval_hook(queue), capacity=3) as (runtime, loop, target, actor):
            results = await asyncio.gather(*(drive_session(runtime, loop, actor, f'session-{i}', command)
                for i, command in enumerate(['ls', 'rm cache.tmp', 'pwd'])))
            assert len({result['sid'] for result in results}) == 3
            assert 'denied' in results[1]['text']
            print(results)
            return results
    finally:
        await queue.put(_STOP)
        await approval


if __name__ == '__main__':
    asyncio.run(main())
