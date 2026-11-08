"""Use only the shipped tool catalog to create, nudge and recall a child."""
import json

import pytest

from power_loop import AgentLoopConfig, create_default_tool_registry
from power_loop.runtime.representation import ProjectedRepresentation
from power_loop.tools.spawn_agent import register_spawn_agent

from ._live_fixture import live_agent, settle
from ._llm import make_llm

pytestmark = [pytest.mark.real_llm, pytest.mark.asyncio]


async def test_catalog_alone_supports_create_nudge_and_child_recall(tmp_path):
    llm = make_llm(max_tokens=2048, temperature=0)
    registry = create_default_tool_registry(include=["recall_send"], workspace_dir=tmp_path)
    register_spawn_agent(registry)
    config=AgentLoopConfig(system_prompt="Follow the requested tool actions. On a child result report it without starting more work.",
        max_rounds=5,max_tokens=2048,representation=ProjectedRepresentation())
    async with live_agent(tmp_path,llm,registry,config) as runner:
        store=runner.runtime.store
        sid=runner.target.session_id
        await runner.send("Use spawn_agent to delegate calculating 7+5. Tell the child to reply only the number. After acceptance end your turn.")
        await settle(runner)
        children=await store.list_children(sid)
        assert len(children)==1
        child=children[0].session_id
        first=await store.history_bounds(child)
        assert any("12" in str(row.content) for row in await store.load_all_messages(child) if row.role=="assistant")
        await runner.send("Use spawn_agent nudge to ask the SAME child to add 3 to its previous answer. Do not create another child. After receipt end your turn.")
        await settle(runner)
        assert len(await store.list_children(sid))==1
        latest=await store.history_bounds(child)
        assert latest[1]>first[1]
        answers=[row for row in await store.load_all_messages(child) if row.role=="assistant" and row.seq>first[1] and "15" in str(row.content)]
        assert answers
        answer_seq=answers[-1].seq
        await runner.send(f"Use recall_send to read original message seq={answer_seq} from child session {child}. Report its answer. Do not spawn or nudge.")
        rows=await store.load_all_messages(sid)
        calls=[json.loads(call['function']['arguments']) for row in rows for call in row.tool_calls or [] if call['function']['name']=='recall_send']
        assert any(call.get('sub_session')==child and call.get('seq')==answer_seq for call in calls)
