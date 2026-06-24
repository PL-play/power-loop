"""44 · Hook 注入审计 / Auditing ephemeral hook injection (record_hook_events)

What you learn / 你将学到
--------------------------
- `LLM_BEFORE` hooks inject EPHEMERAL context into a single LLM call (here: the built-in memory
  recall hook appends a recalled note at the request tail). It is sent to the model and then
  discarded — never written to history or the store (keeps the prompt prefix prefix-cacheable).
  / `LLM_BEFORE` hook 向单次调用注入**临时**上下文(这里是内置记忆 hook 把召回的笔记拼到请求尾部),
  发给模型后即丢弃,不进 history/存储(保持前缀可缓存)。
- Set `AgentLoopConfig.record_hook_events="full"` to AUDIT that injection into the `pl_hook_events`
  table — observability ONLY, written onto the assistant message's sink copy, so it can never
  re-enter history or the LLM request. Read it back with `store.list_hook_events(session_id)`.
  / 设 `record_hook_events="full"` 把注入审计进 `pl_hook_events` 表——仅可观测、绝不回流到上下文;
  用 `store.list_hook_events(session_id)` 读回。
- `"off"` (default) records nothing; `"metadata"` records name/source/chars/position but not the
  text; `"full"` also records the injected content.

Run / 运行
----------
    python examples/44_hook_events_audit.py
"""

from __future__ import annotations

import asyncio

from _helpers import make_llm

from power_loop import AgentLoopConfig, NoteMemory, SessionStore, StatefulAgentLoop


async def main() -> dict:
    store = await SessionStore.open(":memory:")
    try:
        loop = StatefulAgentLoop(
            llm=make_llm(max_tokens=200, temperature=0),
            store=store,
            config=AgentLoopConfig(
                system_prompt="You are concise. Use the persistent notes you are given.",
                max_rounds=1,
                compactor=None,
                memory=NoteMemory(store),       # → auto-registers the built-in MemoryRecallHook
                record_hook_events="full",       # ← audit what the hook injects each round
            ),
        )
        sid = await loop.new_session()
        await store.add_note(sid, "The user's project codename is 'Bluefin'.")

        r = await loop.send("What is my project's codename?", session_id=sid)
        print(f"[answer]   {r.final_text}")

        # The ephemeral memory injection — gone from the request after the call — is now auditable.
        events = await store.list_hook_events(sid)
        ev = events[0]
        injected = " ".join(str(it.get("content") or "") for it in ev.payload["items"])
        print(f"[audit]    {ev.hook} @ {ev.position}: {ev.payload['item_count']} item(s), "
              f"{ev.payload['total_chars']} chars → linked to message seq {ev.message_seq}")
        print(f"[injected] {injected!r}")

        # Leak check: the injected memory BLOCK is NOT a persisted message (it lives only in
        # hook_events). Match the block's distinctive header — the model's answer, which legitimately
        # says "Bluefin", does not contain it.
        msgs = await store.load_all_messages(sid)
        marker = "YOUR NOTES" if "YOUR NOTES" in injected else injected[:24]
        in_history = any(marker and marker in (m.content or "") for m in msgs)

        return {
            "event_count": len(events),
            "hook": ev.hook,
            "kind": ev.kind,
            "position": ev.position,
            "captured_codename": "Bluefin" in injected,
            "linked_to_assistant": ev.message_seq in {m.seq for m in msgs if m.role == "assistant"},
            "audit_text_not_in_history": not in_history,
            "final_text": r.final_text,
        }
    finally:
        await store.close()


if __name__ == "__main__":
    print(asyncio.run(main()))
