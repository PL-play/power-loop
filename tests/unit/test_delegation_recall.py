"""Current child input control plus read-only original/projection recall."""
import asyncio
import json
from dataclasses import replace

import pytest

from power_loop import (
    AgentLoopConfig,
    AgentSpec,
    JobRef,
    LLMResponse,
    LLMService,
    SessionStore,
    ToolDefinition,
    ToolRegistry,
    get_tool_runtime_context,
)
from power_loop.agent.runtime_store import now_ms
from power_loop.agent.runtime_types import RuntimeRequestError
from power_loop.runtime import delegation
from power_loop.runtime.env import RuntimeEnv, runtime_env_context
from power_loop.runtime.pricing import Free
from power_loop.runtime.recall import recall
from power_loop.runtime.representation import ProjectedRepresentation

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.fixture
async def setup():
    store = await SessionStore.open(':memory:')
    parent = await store.create_session()
    try:
        yield store, None, None, parent
    finally:
        await store.close()


async def test_current_child_nudge_preserves_original_result_and_recall_authorization(runtime_agent):
    entered, release = asyncio.Event(), asyncio.Event()
    class Model(LLMService):
        requests = []
        async def complete(self, request, **kwargs):
            self.requests.append(request)
            entered.set()
            await release.wait()
            return LLMResponse(raw_text='CHILD-RESULT')
        async def close(self):
            pass
    launches = []
    async def launch():
        if launches:
            return 'Saved child completion received.'
        context = get_tool_runtime_context(required=True)
        receipt = await delegation.start(context.loop, AgentSpec(name='review', system_prompt='REVIEW',
            tools=[], lifecycle='linked', max_rounds=4), 'Review original material.')
        launches.append(receipt)
        return receipt
    registry = ToolRegistry()
    registry.register(ToolDefinition('launch','Launch one original child', pricing=Free('composite')), launch)
    model = Model()
    agent = await runtime_agent(model=model, registry=registry, command=('launch',{}),
        config=AgentLoopConfig(compactor=None, representation=ProjectedRepresentation()))
    try:
        assert (await agent.send('Launch review')).status == 'completed'
        await asyncio.wait_for(entered.wait(), 5)
        [accepted] = launches
        ref = JobRef(**accepted['job_ref'])
        running_ref = (await agent.runtime.client.inspect(ref, actor=agent.actor)).job_ref
        with pytest.raises(RuntimeRequestError, match='forbidden'):
            await agent.runtime.client.send(running_ref, 'forbidden', message_id='foreign',
                actor=replace(agent.actor, principal_id='stranger'))
        receipt = await agent.runtime.client.send(running_ref, 'Also review B.', message_id='nudge', actor=agent.actor)
        assert receipt.input_ref.message_id == 'nudge'
        release.set()
        result = await agent.runtime.client.wait(ref, actor=agent.actor, deadline_ms=now_ms()+10000)
        assert result.status == 'completed' and result.final_text == 'CHILD-RESULT'
        await agent.drain()
        child = accepted['session_id']
        rows = await agent.runtime.store.load_all_messages(child)
        assert any('Also review B.' in str(row.content) for row in rows)
        assert not any('forbidden' in str(row.content) for row in rows)
        page = json.loads(await recall(agent.runtime.store, agent.target.session_id, sub_session=child, ids=[1, rows[-1].seq]))
        assert 'CHILD-RESULT' in str(page['records'])
        stranger = await agent.runtime.store.create_session()
        denied = json.loads(await recall(agent.runtime.store, stranger, sub_session=child, seq=1))
        assert denied['status'] == 'error' and 'access_denied' in denied['error']
        assert (await agent.runtime.client.wait(ref, actor=agent.actor, deadline_ms=now_ms()+1000)) == result
        assert len(launches) == 1
    finally:
        release.set()
        await agent.aclose()


async def test_projection_id_stable_on_upsert_and_never_reused(setup):
    store, loop, _, parent = setup
    await store.upsert_project_message(parent, send_index=1, kind="project", content={"a": 1})
    one = (await store.load_project_messages(parent))[0]
    await store.upsert_project_message(parent, send_index=1, kind="project", content={"a": 2})
    two = (await store.load_project_messages(parent))[0]
    assert one.id == two.id and two.content == {"a": 2}
    other = await store.create_session()
    await store.upsert_project_message(other, send_index=1, kind="project", content={})
    assert (await store.load_project_messages(other))[0].id > one.id


async def test_recall_pages_long_record_without_send_and_exports_exclusively(setup, tmp_path):
    store, loop, _, parent = setup
    child = await store.create_session(parent_session_id=parent)
    for i in range(5):
        await store.append_message(child, role="tool", content="x" * 9000 + str(i),
                                   tool_calls=None, send_index=1)
    first = json.loads(await recall(store, parent, sub_session=child, ids=[1, 5], limit=2))
    assert len(first["records"]) == 2 and first["next_cursor"]
    page = json.loads(await recall(store, parent, sub_session=child, ids=[1, 5], limit=2,
                                   cursor=first["next_cursor"]))
    assert [r["seq"] for r in page["records"]] == [3, 4]
    bad = json.loads(await recall(store, parent, ids=[1, 5], cursor=first["next_cursor"]))
    assert bad["status"] == "error"
    single = json.loads(await recall(store, parent, sub_session=child, seq=1))
    assert len(json.dumps(single, ensure_ascii=False, separators=(",", ":"))) <= 8000
    assert single["next_char_offset"] == len(single["text"]) < 8000
    rest = json.loads(await recall(store, parent, sub_session=child, seq=1, char_offset=single["next_char_offset"]))
    assert json.loads(single["text"] + rest["text"])["content"].endswith("0")
    with runtime_env_context(RuntimeEnv(workspace_dir=tmp_path)):
        saved = json.loads(await recall(store, parent, sub_session=child, ids=[1, 5],
                                        output_file="records/data.jsonl"))
        assert saved["records"] == 5
        path = tmp_path / "records/data.jsonl"
        original = path.read_bytes()
        assert all(len(json.loads(line)["content"]) == 9001 for line in path.read_text().splitlines())
        refused = json.loads(await recall(store, parent, sub_session=child, ids=[1, 5],
                                          output_file="records/data.jsonl"))
        assert refused["status"] == "error" and path.read_bytes() == original
        conflict = json.loads(await recall(store, parent, sub_session=child, seq=1,
                                           output_file="other.jsonl", char_limit=8000))
        assert conflict["status"] == "error" and not (tmp_path / "other.jsonl").exists()


async def test_foreign_projection_ids_do_not_escape_session_filter(setup):
    store, loop, _, parent = setup
    child = await store.create_session(parent_session_id=parent)
    other = await store.create_session()
    for sid in [child, other, child]:
        count = len(await store.load_project_messages(sid)) + 1
        await store.upsert_project_message(sid, send_index=count, kind="project", content={"sid": sid},
                                           source_seq_lo=count, source_seq_hi=count)
    bounds = await store.history_bounds(child, source="projection")
    data = json.loads(await recall(store, parent, sub_session=child, source="projection", ids=bounds))
    assert len(data["records"]) == 2 and other not in json.dumps(data)
