"""Use only the shipped tool catalog to create, nudge and recall a child."""
import asyncio
import json

import pytest

from power_loop import AgentLoopConfig, SessionStore, StatefulAgentLoop, create_default_tool_registry
from power_loop.runtime.delegation import _TASKS
from power_loop.runtime.representation import ProjectedRepresentation
from power_loop.tools.spawn_agent import register_spawn_agent

from ._llm import make_llm

pytestmark = [pytest.mark.real_llm, pytest.mark.asyncio]


async def drain():
    while _TASKS:
        await asyncio.wait_for(asyncio.gather(*list(_TASKS)), timeout=120)


async def test_catalog_alone_supports_create_nudge_and_child_recall(tmp_path):
    store = await SessionStore.open(":memory:")
    llm = make_llm(max_tokens=2048, temperature=0)
    registry = create_default_tool_registry(include=["recall_send"], workspace_dir=tmp_path)
    register_spawn_agent(registry)
    loop = StatefulAgentLoop(llm=llm, store=store, tool_registry=registry, config=AgentLoopConfig(
        system_prompt="Follow the user's requested tool actions. No additional manuals are available.",
        max_rounds=5, max_tokens=2048, representation=ProjectedRepresentation()))
    sid = await loop.new_session()
    try:
        await loop.send("Use spawn_agent to delegate calculating 7+5. Tell the child to reply with only the number. "
                        "After the acceptance receipt, end your turn.", session_id=sid)
        await drain()
        children = await store.list_children(sid)
        assert len(children) == 1
        child = children[0].session_id
        first = await store.history_bounds(child)
        assert any("12" in str(r.content) for r in await store.load_all_messages(child) if r.role == "assistant")
        await loop.send("Use spawn_agent nudge to ask the SAME child to add 3 to its previous answer. "
                        "Its existing context contains that answer. Do not create another child. "
                        "After the receipt, end your turn.", session_id=sid)
        await drain()
        assert len(await store.list_children(sid)) == 1
        latest = await store.history_bounds(child)
        assert latest[1] > first[1]
        answers = [r for r in await store.load_all_messages(child) if r.role == "assistant" and r.seq > first[1] and "15" in str(r.content)]
        assert answers
        answer_seq = answers[-1].seq
        await loop.send(f"Use recall_send to read original message seq={answer_seq} from child session {child}. "
                        "Then report its answer. Do not spawn or nudge.", session_id=sid)
        rows = await store.load_all_messages(sid)
        calls = [json.loads(tc["function"]["arguments"]) for row in rows
                 for tc in row.tool_calls or [] if tc["function"]["name"] == "recall_send"]
        assert any(c.get("sub_session") == child and c.get("seq") == answer_seq for c in calls)
    finally:
        await drain()
        await store.close()
        await llm.close()
