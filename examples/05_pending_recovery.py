"""05 · 悬挂态恢复 / Pending recovery: what if the process crashes mid-tool-call

What you learn / 你将学到
--------------------------
- 一轮工具调用的协议要求：assistant(tool_calls=[A,B]) 之后必须有 tool 消息对应每个 id
  / protocol requires: assistant(tool_calls=[A,B]) must be followed by a tool
  message for each id
- 如果进程在 assistant 已落库、tool 还没全部落库时挂掉 → 下次 send 会抛
  :class:`SessionPendingError`
  / if process crashes after assistant is persisted but before all tool messages →
  next send raises :class:`SessionPendingError`
- 两条恢复路径 / Two recovery paths:
  - ``await loop.resume(sid)`` —— 把剩余的 tool_calls 跑完，继续循环
    / finish executing remaining tool_calls, then continue the loop
  - ``loop.abort_pending(sid, reason=...)`` —— 给每个悬挂的 tool_call 写
    ``<aborted>`` 消息，恢复协议合法性，再 ``send`` 即可
    / write ``<aborted>`` messages for each pending tool_call, restoring protocol
    validity, then ``send`` normally

我们在这里直接往 store 里写一条 assistant(tool_calls) + ``set_pending`` 来模拟崩溃。
/ We simulate a crash by directly writing an assistant(tool_calls) + ``set_pending`` to the store.

Run / 运行
----------
    python examples/05_pending_recovery.py
"""

from __future__ import annotations

import asyncio

from _helpers import make_llm

from power_loop import (
    AgentLoopConfig,
    SessionPendingError,
    SessionStore,
    StatefulAgentLoop,
    ToolDefinition,
    ToolRegistry,
)


def _build_registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(
        ToolDefinition(
            name="echo",
            description="Echo input.",
            input_schema={"type": "object", "properties": {"text": {"type": "string"}}},
            required_params=(),
        ),
        lambda **kw: kw.get("text", ""),
    )
    return reg


def _simulate_crash_pending(store: SessionStore, sid: str) -> int:
    """直接往 store 里塞一条 assistant(tool_calls) + 把 pending 标上，
    等价于「LLM 已返回，但 tool 调用还没跑完进程就挂了」。
    / Write an assistant(tool_calls) to the store + mark pending — equivalent to
    "LLM returned but process crashed before tool calls finished"."""
    asst_seq = store.append_message(
        sid, role="assistant",
        tool_calls=[{
            "id": "tc-stuck",
            "function": {"name": "echo", "arguments": '{"text":"x"}'},
        }],
        round_index=0,
    )
    store.set_pending(sid, {
        "assistant_seq": asst_seq, "round_index": 0,
        "tool_call_ids": ["tc-stuck"],
        "tool_calls": [{
            "id": "tc-stuck",
            "function": {"name": "echo", "arguments": '{"text":"x"}'},
        }],
    })
    return asst_seq


async def main() -> str:
    store = SessionStore.open(":memory:")
    try:
        loop = StatefulAgentLoop(
            llm=make_llm(),
            store=store,
            tool_registry=_build_registry(),
            config=AgentLoopConfig(
                system_prompt="If a turn was aborted, acknowledge briefly and answer the new question.",
                max_rounds=2,
                compactor=None,
            ),
        )

        # 1. 准备一个被「半成品」打断的 session
        #    Prepare a session interrupted by a half-finished tool call
        sid = store.create_session(system_prompt="S")
        store.append_message(sid, role="user", content="initial prompt")
        _simulate_crash_pending(store, sid)

        # 2. 直接 send 会抛 — 协议禁止把悬挂态丢给 LLM
        #    Direct send raises — protocol forbids passing pending state to LLM
        try:
            await loop.send("anything", session_id=sid)
        except SessionPendingError as exc:
            print(f"[blocked] pending tool_calls: "
                  f"{[tc['id'] for tc in exc.pending_tool_calls]}")

        # 3. 选择 abort_pending（也可以 await loop.resume(sid)）
        #    Choose abort_pending (or alternatively: await loop.resume(sid))
        n = loop.abort_pending(sid, reason="user_cancelled")
        print(f"[abort_pending] aborted {n} tool_call(s); pending now {loop.get_pending(sid)}")

        # 4. 现在 send 可以正常往下走 / Now send proceeds normally
        r = await loop.send("In one sentence: what does HTML stand for?", session_id=sid)
        print(f"[send]  status={r.status}, reply={r.final_text}")
        return r.final_text
    finally:
        store.close()


if __name__ == "__main__":
    asyncio.run(main())
