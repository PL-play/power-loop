"""43 · 内置记忆 hook：默认 / 关闭 / 覆盖 / Built-in memory hook: default, disable, override

What you learn / 你将学到
--------------------------
- 设了 ``config.memory`` 后，循环会自动注册内置的 ``MemoryRecallHook``（一个
  ``LLM_BEFORE`` hook，名字 ``MemoryRecallHook.NAME``）。它把召回的记忆**临时拼到
  请求末尾**（不进 history、不落库），让 system + 历史前缀保持稳定、可被供应商前缀缓存。
  / setting ``config.memory`` auto-registers the built-in ``MemoryRecallHook`` (an
  ``LLM_BEFORE`` hook named ``MemoryRecallHook.NAME``). It appends recalled memory
  EPHEMERALLY at the request TAIL (never into history/store), keeping the system +
  history prefix stable and prefix-cacheable.
- 内置 hook 是**可覆盖/可关闭**的：``hooks.remove(name)`` 关闭，
  ``hooks.replace(point, fn, name=name)`` 替换为你自己的注入逻辑。
  / the built-in hook is overridable/disable-able: ``hooks.remove(name)`` to disable,
  ``hooks.replace(point, fn, name=name)`` to swap in your own injection.

Run / 运行
----------
    python examples/43_memory_hook_override.py
"""

from __future__ import annotations

import asyncio

from _helpers import make_llm

from power_loop import (
    AgentLoopConfig,
    HookPoint,
    MemoryRecallHook,
    NoteMemory,
    SessionStore,
    StatefulAgentLoop,
)


def _new_loop(store: SessionStore) -> StatefulAgentLoop:
    return StatefulAgentLoop(
        llm=make_llm(max_tokens=200, temperature=0),
        store=store,
        config=AgentLoopConfig(
            system_prompt="You are concise. Use the persistent notes you are given.",
            max_rounds=1,
            compactor=None,
            memory=NoteMemory(store),  # → auto-registers MemoryRecallHook
        ),
    )


async def main() -> str:
    store = await SessionStore.open(":memory:")
    try:
        # ── 1. default: the built-in hook is auto-registered ──
        loop = _new_loop(store)
        assert loop.hooks.has(MemoryRecallHook.NAME), "memory set → built-in hook registered"
        sid = await loop.new_session()
        await store.add_note(sid, "The user's project codename is 'Bluefin'.")
        r = await loop.send("What is my project's codename?", session_id=sid)
        print(f"[default]  {r.final_text}")  # should mention Bluefin

        # ── 2. disable it: hooks.remove(name) ──
        loop2 = _new_loop(store)
        loop2.hooks.remove(MemoryRecallHook.NAME)
        assert not loop2.hooks.has(MemoryRecallHook.NAME), "removed → no memory injection"
        print("[disabled] built-in memory hook removed (model would NOT see notes)")

        # ── 3. override it: replace under the same name ──
        loop3 = _new_loop(store)

        def my_injection(ctx) -> None:
            # your own injection logic — runs in place of the built-in
            ctx.messages.append(
                {"role": "system", "name": "memory_custom", "content": "Custom note: be playful."}
            )

        loop3.hooks.replace(HookPoint.LLM_BEFORE, my_injection, name=MemoryRecallHook.NAME)
        entries = [
            e for e in loop3.hooks._handlers[str(HookPoint.LLM_BEFORE)]
            if e.name == MemoryRecallHook.NAME
        ]
        assert len(entries) == 1 and entries[0].handler is my_injection
        print("[override] swapped the built-in hook for a custom LLM_BEFORE injector")

        return r.final_text
    finally:
        await store.close()


if __name__ == "__main__":
    asyncio.run(main())
