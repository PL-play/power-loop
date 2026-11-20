"""P2 crash recovery: pending repair, no replay, owner fences, expired-run discovery."""
from __future__ import annotations

import json

import pytest

from power_loop.agent.runtime_store import RuntimeInputStore
from power_loop.contracts.errors import SessionLeaseLost

from .session_native_fixture import ACTOR, target

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def _assistant_with_pending(store, port, batch, tool_ids):
    """Simulate a crash right after the assistant tool-call append: the raw assistant
    row and pending_json are durable, but no tool result was written."""
    seq = batch.message_seq + 1
    tool_calls = [{"id": cid, "type": "function", "function": {"name": "echo", "arguments": "{}"}}
                  for cid in tool_ids]
    await store._db.execute(
        f"INSERT INTO {store.t.messages} (session_id,seq,role,content,tool_calls_json,round_index,"
        "state,meta_json,send_index,created_at) VALUES (?,?, 'assistant', '', ?, 0, 'active', ?, ?, 0)",
        (batch.target.session_id, seq, json.dumps(tool_calls), "{}", batch.send_index),
    )
    pending = {"assistant_seq": seq}
    await store._db.execute(
        f"UPDATE {store.t.session_state} SET pending_json=?,next_seq=? WHERE session_id=?",
        (json.dumps(pending), seq + 1, batch.target.session_id),
    )
    return seq


async def _expire(store, session_id):
    await store._db.execute(
        f"UPDATE {store.t.session_state} SET owner_lease_until_ms=1 WHERE session_id=?", (session_id,))


async def test_recover_repairs_missing_tools_without_replay(registry_store):
    store, connect = registry_store
    port = RuntimeInputStore(store)
    root = await target(port)
    await port.accept(root, "go", message_id="m1", actor=ACTOR)
    batch = await port.claim(root, node_id="a", boot_id="a")
    assistant_seq = await _assistant_with_pending(store, port, batch, ["t1", "t2"])
    await _expire(store, root.session_id)

    recovered = await port.recover(root, node_id="b", boot_id="b")
    assert recovered is not None
    assert recovered.results[0].status == "interrupted"
    # Both missing tools got a fixed unknown result stamped with the INTERRUPTED send_index.
    tools = await store._db.fetchall(
        f"SELECT seq,tool_call_id,content,send_index,meta_json FROM {store.t.messages} "
        "WHERE session_id=? AND role='tool' ORDER BY seq", (root.session_id,))
    assert {t["tool_call_id"] for t in tools} == {"t1", "t2"}
    assert all(t["send_index"] == batch.send_index for t in tools)
    assert all("执行中断" in t["content"] for t in tools)
    assert all(json.loads(t["meta_json"])["_assistant_seq"] == assistant_seq for t in tools)
    from power_loop.runtime.tool_archive import recall_original
    reader = await connect()
    for tool in tools:
        row = await reader.get_message(root.session_id, tool["seq"])
        original = json.loads(await recall_original(reader, root.session_id, batch.send_index, row))
        assert original["arguments"] == "{}" and original["result"] == tool["content"]
    state = await store.get_state(root.session_id)
    assert state.pending is None
    # The input is terminal interrupted; a cold wait sees it (no eternal wait).
    result = await port.get_result(batch.input_refs[0], actor=ACTOR)
    assert result.status == "interrupted"


async def test_existing_tool_result_is_not_duplicated(registry_store):
    store, _ = registry_store
    port = RuntimeInputStore(store)
    root = await target(port)
    await port.accept(root, "go", message_id="m1", actor=ACTOR)
    batch = await port.claim(root, node_id="a", boot_id="a")
    assistant_seq = await _assistant_with_pending(store, port, batch, ["t1", "t2"])
    # t1 already has a real result before the crash.
    await store._db.execute(
        f"INSERT INTO {store.t.messages} (session_id,seq,role,content,tool_call_id,round_index,"
        "state,meta_json,send_index,created_at) VALUES (?,?, 'tool', 'real result', 't1', 0, 'active', ?, ?, 0)",
        (root.session_id, assistant_seq + 1, "{}", batch.send_index),
    )
    await store._db.execute(
        f"UPDATE {store.t.session_state} SET next_seq=? WHERE session_id=?",
        (assistant_seq + 2, root.session_id),
    )
    before = await store.get_message(root.session_id, assistant_seq + 1)
    await _expire(store, root.session_id)
    await port.recover(root, node_id="b", boot_id="b")
    tools = await store._db.fetchall(
        f"SELECT tool_call_id,content FROM {store.t.messages} WHERE session_id=? AND role='tool' "
        "ORDER BY seq", (root.session_id,))
    assert [t["tool_call_id"] for t in tools] == ["t1", "t2"]  # t1 not duplicated
    assert tools[0]["content"] == "real result"
    assert "执行中断" in tools[1]["content"]
    assert await store.get_message(root.session_id, assistant_seq + 1) == before
    repaired = await store.get_message(root.session_id, assistant_seq + 2)
    assert repaired.meta["_assistant_seq"] == assistant_seq


async def test_claim_repairs_pending_before_new_user_message(registry_store):
    store, _ = registry_store
    port = RuntimeInputStore(store)
    root = await target(port)
    await port.accept(root, "first", message_id="m1", actor=ACTOR)
    batch = await port.claim(root, node_id="a", boot_id="a")
    await _assistant_with_pending(store, port, batch, ["t1"])
    # The interrupted run is finalized (pending left behind), session returns to idle.
    await port.final(batch, status="interrupted", error="crash")
    assert (await store.get_state(root.session_id)).pending is not None
    # A new send must repair the old pending BEFORE writing its merged user message.
    await port.accept(root, "second", message_id="m2", actor=ACTOR)
    second = await port.claim(root, node_id="b", boot_id="b")
    assert second.send_index == batch.send_index + 1
    tools = await store._db.fetchall(
        f"SELECT tool_call_id,send_index FROM {store.t.messages} WHERE session_id=? AND role='tool'",
        (root.session_id,))
    assert tools[0]["tool_call_id"] == "t1"
    assert tools[0]["send_index"] == batch.send_index  # repaired onto the OLD send
    assert (await store.get_state(root.session_id)).pending is None
    # The new merged user message carries the new send_index, after the repair.
    users = await store._db.fetchall(
        f"SELECT seq,send_index FROM {store.t.messages} WHERE session_id=? AND role='user' ORDER BY seq",
        (root.session_id,))
    assert users[-1]["send_index"] == second.send_index
    assert users[-1]["seq"] > tools[0]["send_index"]


async def test_running_expired_without_pending_input_terminates_wait(registry_store):
    store, _ = registry_store
    port = RuntimeInputStore(store)
    root = await target(port)
    await port.accept(root, "go", message_id="m1", actor=ACTOR)
    batch = await port.claim(root, node_id="a", boot_id="a")
    await _expire(store, root.session_id)
    # No pending inbox input, but the running lease expired: discovery must still find it.
    found = await port.discover_unfinished()
    assert root in found
    recovered = await port.recover(root, node_id="b", boot_id="b")
    assert recovered.results[0].status == "interrupted"
    state = await store._db.fetchone(
        f"SELECT execution_state,active_send_index FROM {store.t.session_state} WHERE session_id=?",
        (root.session_id,))
    assert state["execution_state"] == "idle" and state["active_send_index"] is None
    assert (await port.get_result(batch.input_refs[0], actor=ACTOR)).status == "interrupted"


async def test_stale_owner_final_is_rejected_after_takeover(registry_store):
    store, _ = registry_store
    port = RuntimeInputStore(store)
    root = await target(port)
    await port.accept(root, "go", message_id="m1", actor=ACTOR)
    batch = await port.claim(root, node_id="a", boot_id="a")
    await _expire(store, root.session_id)
    await port.recover(root, node_id="b", boot_id="b")  # new owner epoch
    with pytest.raises(SessionLeaseLost):
        await port.final(batch, final_text="stale result")
    # The stale owner's result message was never written.
    assistants = await store._db.fetchall(
        f"SELECT content FROM {store.t.messages} WHERE session_id=? AND role='assistant'",
        (root.session_id,))
    assert all(a["content"] != "stale result" for a in assistants)


async def test_final_ack_loss_recovers_without_duplicate_result(registry_store, monkeypatch):
    """design 215 §9.2: final commits but the ACK is lost — the retry reads the durable
    terminal state and returns the original result, never a second result message."""
    from .session_native_fixture import fail_next_commit
    store, _ = registry_store
    port = RuntimeInputStore(store)
    root = await target(port)
    await port.accept(root, "go", message_id="m1", actor=ACTOR)
    batch = await port.claim(root, node_id="a", boot_id="a")
    fail_next_commit(monkeypatch, store)
    recovered = await port.final(batch, consumed=batch.input_refs, final_text="the result",
                                 usage={"total_tokens": 3})
    assert recovered.results[0].status == "completed"
    assert recovered.results[0].final_text == "the result"
    assistants = await store._db.fetchall(
        f"SELECT content FROM {store.t.messages} WHERE session_id=? AND role='assistant'",
        (root.session_id,))
    assert [a["content"] for a in assistants] == ["the result"]  # exactly one
    assert (await port.get_result(batch.input_refs[0], actor=ACTOR)).final_text == "the result"


async def test_projection_write_failure_does_not_fail_send(registry_store, monkeypatch):
    """design 215 §8.2: the projection is derived/best-effort; a failed projection write
    never fails the send (raw messages remain the source of truth)."""
    from power_loop import AgentLoopConfig
    from power_loop.agent import stateful_loop as sl
    from tests.runtime_fixture import RunningAgent

    from .test_session_send_215 import EchoModel
    store, _ = registry_store

    async def broken_projection(self, *a, **k):
        raise RuntimeError("projection store unavailable")

    monkeypatch.setattr(sl._ExecutionLoop, "_write_session_projection", broken_projection)
    # Force projection mode so the end-of-send projection path is exercised.
    from power_loop.runtime.representation import ProjectedRepresentation
    agent = await RunningAgent(store, EchoModel(),
                               config=AgentLoopConfig(max_rounds=4, context_budget_tokens=10**9,
                                                      representation=ProjectedRepresentation())).start()
    try:
        result = await agent.send("go")
        assert result.status == "completed" and result.final_text == "done"
        rows = await store.load_all_messages(agent.target.session_id)
        assert [r.role for r in rows] == ["user", "assistant"]  # raw intact
    finally:
        await agent.aclose()


async def test_recovering_with_followup_terminates_old_then_claims_new(registry_store):
    store, _ = registry_store
    port = RuntimeInputStore(store)
    root = await target(port)
    await port.accept(root, "old", message_id="old", actor=ACTOR)
    batch = await port.claim(root, node_id="a", boot_id="a")
    await _expire(store, root.session_id)
    # A new input arrives while the old run is expired: accept turns it recovering.
    await port.accept(root, "new", message_id="new", actor=ACTOR)
    state = await store._db.fetchone(
        f"SELECT execution_state FROM {store.t.session_state} WHERE session_id=?", (root.session_id,))
    assert state["execution_state"] == "recovering"
    # Recovery terminates the old run, leaves the new input pending → queued.
    recovered = await port.recover(root, node_id="b", boot_id="b")
    assert recovered.results[0].status == "interrupted"
    assert recovered.execution_state == "queued"
    assert (await port.get_result(batch.input_refs[0], actor=ACTOR)).status == "interrupted"
    # The new input is claimed as a fresh send, not merged into the interrupted one.
    new_batch = await port.claim(root, node_id="b", boot_id="b")
    assert new_batch.send_index == batch.send_index + 1
    assert [r.message_id for r in new_batch.input_refs] == ["new"]


async def test_partial_result_and_abort_preserve_pending_source(registry_store):
    from power_loop import AgentLoopConfig
    from power_loop.agent.sink import SQLiteSink
    from power_loop.agent.stateful_loop import _ExecutionLoop
    from power_loop.runtime.tool_archive import recall_original
    from tests.runtime_fixture import ScriptedModel

    from .session_native_fixture import DESCRIPTOR

    store = registry_store[0]
    sid = await store.create_session(DESCRIPTOR, actor=ACTOR)
    calls = [{'id': cid, 'function': {'name': 'echo', 'arguments': '{"text":"original"}'}}
             for cid in ('completed', 'aborted')]
    sink = SQLiteSink(store, sid)
    await sink.on_message_appended({'role': 'assistant', 'tool_calls': calls, 'send_index': 1}, round_index=0)
    source = sink.persisted_assistant_seq
    await sink.on_assistant_tool_calls(assistant_seq=source, tool_calls=calls, round_index=0)
    await store.set_runtime_state(sid, 'send_index', 99)
    model = ScriptedModel()
    loop = _ExecutionLoop(llm=model, store=store, config=AgentLoopConfig(delegation_policy="single_level"))
    await sink.on_message_appended({'role': 'tool', 'tool_call_id': 'completed',
                                   'content': 'known result', 'send_index': 1}, round_index=0)
    assert (await store.get_state(sid)).pending == {'assistant_seq': source}
    assert len([r for r in await store.load_all_messages(sid) if r.role == 'tool']) == 1
    assert await loop.abort_pending(sid) == 1
    rows = await store.load_all_messages(sid)
    tools = [r for r in rows if r.role == 'tool']
    assert len(tools) == 2 and model._idx == 0
    assert all(r.meta['_assistant_seq'] == source for r in tools)
    assert (await store.get_state(sid)).pending is None
    for row in tools:
        value = json.loads(await recall_original(store, sid, 1, row))
        assert value['arguments'] == '{"text":"original"}'
        assert value['result'] == row.content


async def test_marker_and_last_result_rollback_preserve_batch(registry_store, monkeypatch):
    from contextlib import asynccontextmanager

    from power_loop.agent.sink import SQLiteSink
    from power_loop.runtime.tool_archive import recall_original

    store = registry_store[0]
    port = RuntimeInputStore(store)
    root = await target(port)
    await port.accept(root, 'go', message_id='m1', actor=ACTOR)
    await port.claim(root, node_id='old', boot_id='old')
    sid = root.session_id
    arguments = json.dumps({'text': 'CURRENT', 'large': 'x' * 65536})
    calls = [{'id': f'c{i}', 'function': {'name': 'echo', 'arguments': arguments}}
             for i in range(3)]
    # A completed earlier batch in the SAME send reused these ids.
    old = await store.append_message(sid, role='assistant', send_index=1, tool_calls=calls, pending_tool_calls=True)
    for i in range(3):
        options = {'pending': None} if i == 2 else {}
        await store.append_message(sid, role='tool', tool_call_id=f'c{i}', content='earlier',
                                   send_index=1, meta={'_assistant_seq': old}, **options)
    baseline = await store.load_all_messages(sid)
    transaction = store._db.transaction

    @asynccontextmanager
    async def rolled_back():
        async with transaction() as tx:
            yield tx
            raise RuntimeError('rollback before commit')

    with monkeypatch.context() as patch:
        patch.setattr(store._db, 'transaction', rolled_back)
        with pytest.raises(RuntimeError, match='rollback before commit'):
            await store.append_message(sid, role='assistant', tool_calls=calls,
                                       send_index=1, round_index=7, pending_tool_calls=True)
    assert (await store.get_state(sid)).pending is None
    assert await store.load_all_messages(sid) == baseline
    source = await store.append_message(sid, role='assistant', tool_calls=calls,
                                        send_index=1, round_index=7, pending_tool_calls=True)
    marker = {'assistant_seq': source}
    assert (await store.get_state(sid)).pending == marker
    sink = SQLiteSink(store, sid)
    await sink.on_assistant_tool_calls(assistant_seq=source, tool_calls=calls, round_index=7)
    # The call remains a valid recovery source after compaction.
    await store.record_compaction(sid, from_seq=source, to_seq=source, note_content='folded',
                                 before_tokens=30, after_tokens=2, round_index=7)
    for cid in ('c2', 'c0'):
        await sink.on_message_appended({'role': 'tool', 'tool_call_id': cid,
                                       'send_index': 1, 'content': cid}, round_index=7)
        assert (await store.get_state(sid)).pending == marker
    with monkeypatch.context() as patch:
        patch.setattr(store._db, 'transaction', rolled_back)
        with pytest.raises(RuntimeError, match='rollback before commit'):
            await sink.on_message_appended({'role': 'tool', 'tool_call_id': 'c1',
                                           'send_index': 1, 'content': 'rolled back'}, round_index=7)
    assert sink._unresolved == {'c1'} and sink.persisted_assistant_seq == source
    assert (await store.get_state(sid)).pending == marker
    await _expire(store, sid)
    await port.recover(root, node_id='new', boot_id='new')
    rows = await store.load_all_messages(sid)
    current = [r for r in rows if r.role == 'tool' and r.meta.get('_assistant_seq') == source]
    assert [r.tool_call_id for r in current] == ['c2', 'c0', 'c1']
    assert [r.content for r in current[:2]] == ['c2', 'c0']
    assert '结果未知' in current[-1].content and current[-1].round_index == 7
    for row in current:
        assert json.loads(await recall_original(store, sid, 1, row, char_limit=100000))['arguments'] == arguments
    assert (await store.get_state(sid)).pending is None


async def test_tool_result_ack_loss_is_not_replayed_or_duplicated(registry_store, monkeypatch):
    from power_loop.agent.sink import SQLiteSink

    from .session_native_fixture import UnknownCommit, fail_next_commit

    store = registry_store[0]
    port = RuntimeInputStore(store)
    root = await target(port)
    await port.accept(root, 'go', message_id='m1', actor=ACTOR)
    await port.claim(root, node_id='old', boot_id='old')
    sink = SQLiteSink(store, root.session_id)
    calls = [{'id': cid, 'function': {'name': 'echo', 'arguments': '{}'}} for cid in ('one', 'two')]
    await sink.on_message_appended({'role': 'assistant', 'tool_calls': calls, 'send_index': 1}, round_index=0)
    source = sink.persisted_assistant_seq
    await sink.on_assistant_tool_calls(assistant_seq=source, tool_calls=calls, round_index=0)
    fail_next_commit(monkeypatch, store)
    with pytest.raises(UnknownCommit):
        await sink.on_message_appended({'role': 'tool', 'tool_call_id': 'one',
                                       'content': 'committed original', 'send_index': 1}, round_index=0)
    assert sink._unresolved == {'one', 'two'}  # Stop this executor; do not retry the tool.
    await _expire(store, root.session_id)
    await port.recover(root, node_id='new', boot_id='new')
    tools = [r for r in await store.load_all_messages(root.session_id) if r.role == 'tool']
    assert [r.tool_call_id for r in tools] == ['one', 'two']
    assert tools[0].content == 'committed original' and '结果未知' in tools[1].content
    assert all(r.meta['_assistant_seq'] == source for r in tools)


@pytest.mark.parametrize('entry', ['recover', 'abort'])
async def test_all_results_present_only_clears_marker(registry_store, entry):
    from power_loop import AgentLoopConfig
    from power_loop.agent.stateful_loop import _ExecutionLoop
    from tests.runtime_fixture import ScriptedModel

    store = registry_store[0]
    port = RuntimeInputStore(store)
    root = await target(port)
    await port.accept(root, 'go', message_id='m1', actor=ACTOR)
    batch = await port.claim(root, node_id='old', boot_id='old')
    source = await _assistant_with_pending(store, port, batch, ['t1', 't2'])
    for cid in ('t1', 't2'):
        await store.append_message(root.session_id, role='tool', tool_call_id=cid,
                                   send_index=1, content=cid, meta={'_assistant_seq': source})
    before = [r for r in await store.load_all_messages(root.session_id) if r.role == 'tool']
    if entry == 'recover':
        await _expire(store, root.session_id)
        await port.recover(root, node_id='new', boot_id='new')
    else:
        loop = _ExecutionLoop(llm=ScriptedModel(), store=store, config=AgentLoopConfig(delegation_policy='single_level'))
        assert await loop.get_pending(root.session_id) == {'assistant_seq': source}
        assert await loop.abort_pending(root.session_id) == 0
    assert (await store.get_state(root.session_id)).pending is None
    assert [r for r in await store.load_all_messages(root.session_id) if r.role == 'tool'] == before


@pytest.mark.parametrize('entry', ['claim', 'abort'])
async def test_unavailable_pending_source_is_explicit_and_unchanged(registry_store, monkeypatch, entry):
    from power_loop import AgentLoopConfig
    from power_loop.agent.runtime_types import RuntimeRequestError
    from power_loop.agent.stateful_loop import _ExecutionLoop
    from tests.runtime_fixture import ScriptedModel

    store = registry_store[0]
    port = RuntimeInputStore(store)
    model = ScriptedModel()
    loop = _ExecutionLoop(llm=model, store=store, config=AgentLoopConfig(delegation_policy='single_level'))
    unknown_checks = []

    async def no_unknown(*args):
        unknown_checks.append(args)
        raise AssertionError('domain rejection must not check an unknown commit')

    monkeypatch.setattr(port, '_claim_after_unknown', no_unknown)
    for kind in ('missing', 'not-assistant', 'invalid-json', 'empty-calls'):
        root = await target(port, session_id=f'{entry}-{kind}')
        await port.accept(root, 'old', message_id='m1', actor=ACTOR)
        batch = await port.claim(root, node_id='old', boot_id='old')
        source = await _assistant_with_pending(store, port, batch, ['t1'])
        await port.final(batch, status='interrupted', error='crash')
        if kind == 'missing':
            await store._db.execute(f'DELETE FROM {store.t.messages} WHERE session_id=? AND seq=?', (root.session_id, source))
        elif kind == 'not-assistant':
            await store._db.execute(f"UPDATE {store.t.messages} SET role='user' WHERE session_id=? AND seq=?", (root.session_id, source))
        else:
            await store._db.execute(f'UPDATE {store.t.messages} SET tool_calls_json=? WHERE session_id=? AND seq=?',
                                    ('not-json' if kind == 'invalid-json' else '[]', root.session_id, source))
        if entry == 'claim':
            await port.accept(root, 'new', message_id='m2', actor=ACTOR)
        before = await store._db.fetchall(f'SELECT * FROM {store.t.messages} WHERE session_id=? ORDER BY seq', (root.session_id,))
        with pytest.raises(RuntimeRequestError) as error:
            if entry == 'claim':
                await port.claim(root, node_id='new', boot_id='new')
            else:
                await loop.abort_pending(root.session_id)
        assert error.value.code == 'pending_source_unavailable'
        assert (await store.get_state(root.session_id)).pending == {'assistant_seq': source}
        assert await store._db.fetchall(f'SELECT * FROM {store.t.messages} WHERE session_id=? ORDER BY seq', (root.session_id,)) == before
    assert model._idx == 0 and not unknown_checks


async def test_replaced_runtime_owner_rejects_old_sink(registry_store):
    import asyncio
    import contextvars
    import time

    from power_loop.agent.sink import SQLiteSink
    from power_loop.runtime.session_ownership import SessionOwnership

    store = registry_store[0]
    port = RuntimeInputStore(store)
    root = await target(port)
    await port.accept(root, 'go', message_id='m1', actor=ACTOR)
    batch = await port.claim(root, node_id='old', boot_id='old')
    lease = SessionOwnership((store.store_id, root.session_id), 'old', time.monotonic() + 60,
                             lambda: None, token=batch.owner_token, epoch=batch.owner_epoch,
                             node_id='old', boot_id='old', installs_fence=True)
    async with store.send_read_scope(root.session_id, ownership=lease):
        await store.get_state(root.session_id)
        sink = SQLiteSink(store, root.session_id)
        calls = [{'id': 'c', 'function': {'name': 'echo', 'arguments': '{}'}}]
        await sink.on_message_appended({'role': 'assistant', 'tool_calls': calls, 'send_index': 1}, round_index=0)
        await sink.on_assistant_tool_calls(assistant_seq=sink.persisted_assistant_seq,
                                           tool_calls=calls, round_index=0)

        async def replacement():
            await _expire(store, root.session_id)
            await RuntimeInputStore(store).recover(root, node_id='new', boot_id='new')

        await asyncio.create_task(replacement(), context=contextvars.Context())
        with pytest.raises(SessionLeaseLost):
            await sink.on_message_appended({'role': 'tool', 'tool_call_id': 'c',
                                           'content': 'stale', 'send_index': 1}, round_index=0)
        assert sink._unresolved == {'c'}
    rows = await store.load_all_messages(root.session_id)
    assert all(row.content != 'stale' for row in rows)
    assert len([r for r in rows if r.role == 'tool' and r.tool_call_id == 'c']) == 1
