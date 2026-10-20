import copy
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from power_loop.core.pipeline import AgentPipeline
from power_loop.runtime.store.store import SessionStore
from power_loop.runtime.tool_archive import manage_capacity, prepare_history, prepare_projection, recall_group
from power_loop.runtime.tool_context import ToolContextConfig, ToolContextEngine


async def fixture(overrides=None, n=3, dsn=None):
    from power_loop.runtime.store.factory import open_store
    store = await open_store(dsn) if dsn else await SessionStore.open(':memory:')
    sid = await store.create_session()
    history, seqs = [], []
    for i in range(n):
        args = {'command': 'x' * 1500, 'description': f'目的{i}', 'context_policy': 'keep'}
        call = {'id': str(i), 'type': 'function', 'function': {'name': 'bash', 'arguments': json.dumps(args)}}
        for row in ({'role': 'assistant', 'tool_calls': [call]}, {'role': 'tool', 'tool_call_id': str(i), 'content': 'r' * 1500}):
            seqs.append(await store.append_message(sid, send_index=1, **row))
            history.append(row)
    config = ToolContextConfig(schema_version=2, trigger_tokens=1000, target_tokens=500,
        window_tokens=5000, reserve_tokens=100, margin_tokens=100, overrides=overrides or {})
    engine = ToolContextEngine(config=config, store=store, owner_session_id=sid, owner_send_index=1,
        llm=None, model='test')
    await engine.start()
    p = SimpleNamespace(history=history, sink=SimpleNamespace(_history_seqs=seqs), send_index=1,
        config=SimpleNamespace(insend_distill_hot_tail=0, max_tokens=100), system_prompt='', runtime_tools=[],
        _tok_len=-1, _tok_total=0)
    p._estimate_history_tokens = lambda: AgentPipeline._estimate_history_tokens(p)
    return store, engine, p


async def test_two_levels_durable_restore_and_full_original():
    store, e, p = await fixture()
    try:
        original = copy.deepcopy(p.history)
        for _ in range(250):
            await prepare_history(e, p, 'age_retirement')
        assert p.history == original
        assert await prepare_history(e, p, 'insend_fuse', level=1) == 0  # keep soft
        await manage_capacity(e, p)
        assert not p._capacity_blocked
        assert e.state['archive_v2']['events'][-1]['levels'][-1]['level'] == 2
        assert '目的0' in p.history[1]['content']
        assert (await store.get_message(e.sid, 2)).content == 'r' * 1500
        view = copy.deepcopy(p.history)
        p.history = original
        p.sink._history_seqs = list(range(1, len(original) + 1))
        resumed = ToolContextEngine(config=replace(e.config, trigger_tokens=2000), store=store,
            owner_session_id=e.sid, owner_send_index=1, llm=None, model='test')
        await resumed.start()
        assert resumed.config.trigger_tokens == 1000
        await manage_capacity(resumed, p)
        assert p.history == view
    finally:
        await store.close()


async def test_hard_preserve_blocks_and_does_not_rescan():
    store, e, p = await fixture({'bash': {'hard_preserve': True}}, n=8)
    try:
        original = copy.deepcopy(p.history)
        await manage_capacity(e, p)
        assert p._capacity_blocked
        assert p.history == original
        events = copy.deepcopy(e.state['archive_v2']['events'])
        await manage_capacity(e, p)
        assert e.state['archive_v2']['events'] == events
    finally:
        await store.close()


async def test_general_groups_exact_recall_and_send_scope():
    store, e, p = await fixture(n=20)
    try:
        rows = await store.load_all_messages(e.sid)
        projected = SimpleNamespace(rows=[SimpleNamespace(kind='project', content={'tools': []}, rendered_text='old')])
        await prepare_projection(e, projected, rows, 1)
        entries = projected.rows[0].content['tools']
        assert len(entries) == 1
        assert '目的0' in entries[0]['s'] and '目的19' in entries[0]['s']
        gid = next(iter(e.state['archive_v2']['groups']))
        directory = json.loads(await recall_group(store, e.sid, 1, gid, limit=3))
        assert len(directory['items']) == 3 and directory['next_offset'] == 3
        restored = json.loads(await recall_group(store, e.sid, 1, gid, item=19))
        assert restored['result'] == 'r' * 1500
        assert json.loads(restored['arguments'])['description'] == '目的19'
        assert 'Unknown' in await recall_group(store, e.sid, 2, gid)
        stranger = await store.create_session()
        assert 'Unknown' in await recall_group(store, stranger, 1, gid)
    finally:
        await store.close()


async def test_failed_commit_leaves_view_and_state_unchanged(monkeypatch):
    store, e, p = await fixture()
    try:
        original, state = copy.deepcopy(p.history), copy.deepcopy(e.state)
        async def fail(*a, **k):
            raise OSError('disk unavailable')
        monkeypatch.setattr(e, '_save', fail)
        with pytest.raises(OSError):
            await prepare_history(e, p, 'insend_fuse', level=2)
        assert p.history == original
        assert e.state == state
    finally:
        await store.close()


@pytest.mark.parametrize('patch', [{'target_tokens': 500000}, {'window_tokens': 510000},
                                   {'trigger_tokens': True}, {'capacity_enabled': 1}])
def test_invalid_capacity(patch):
    with pytest.raises(ValueError):
        ToolContextConfig(schema_version=2, **patch)


async def test_stateful_two_sends_render_no_legacy_truncation():
    from power_loop import AgentLoopConfig, StatefulAgentLoop
    from power_loop._vendor.llm_client.interface import LLMResponse
    from power_loop.contracts.tools import ToolDefinition
    from power_loop.runtime.representation import ProjectedRepresentation
    from power_loop.tools.registry import ToolRegistry
    from tests.unit.test_stateful_loop import _Scripted, _tool_resp
    store = await SessionStore.open(':memory:')
    try:
        reg = ToolRegistry()
        reg.register(ToolDefinition('echo', ''), lambda **kw: 'UNIQUE-RESULT' * 5000)
        llm = _Scripted([_tool_resp('a','echo','{}'), LLMResponse(raw_text='done'), LLMResponse(raw_text='next')])
        loop = StatefulAgentLoop(llm=llm, store=store, tool_registry=reg,
            config=AgentLoopConfig(system_prompt='S', max_tokens=100, max_rounds=5,
                representation=ProjectedRepresentation(), tool_context=ToolContextConfig(schema_version=2)))
        sid = await loop.new_session()
        first = await loop.send('work', session_id=sid)
        assert first.status == 'completed'
        assert len(next(r.content for r in await store.load_all_messages(sid) if r.role=='tool')) == 65000
        assert any(m.get('content') == 'UNIQUE-RESULT' * 5000 for m in llm.calls[1])
        await loop.send('continue', session_id=sid)
        text = json.dumps(llm.calls[-1], ensure_ascii=False)
        assert '正文未展示' in text and 'UNIQUE-RESULT' not in text
        assert 'done' in text
    finally:
        await store.close()


async def test_group_replay_stable_and_reasoning_boundary():
    from power_loop.runtime.tool_archive import restore_history
    store, engine, p = await fixture(n=4)
    try:
        # Provider-required reasoning is a boundary, not something grouping may erase.
        p.history[2]['reasoning_content'] = 'provider-owned'
        await prepare_history(engine, p, 'insend_fuse', level=2)
        assert len(p.history) == 6
        assert p.history[2]['reasoning_content'] == 'provider-owned'
        ids = [c['id'] for r in p.history for c in r.get('tool_calls', [])]
        assert ids == [r['tool_call_id'] for r in p.history if r['role'] == 'tool']
        view = copy.deepcopy(p.history)
        p._tok_len = 99
        restore_history(engine, p)
        assert p.history == view and p._tok_len == 99
    finally:
        await store.close()


async def test_original_pagination_is_lossless():
    store, engine, p = await fixture(n=2)
    try:
        rows = await store.load_all_messages(engine.sid)
        projected = SimpleNamespace(rows=[SimpleNamespace(kind='project', content={'tools': []}, rendered_text=None)])
        await prepare_projection(engine, projected, rows, 1)
        gid = next(iter(engine.state['archive_v2']['groups']))
        chunks, offset = [], 0
        while offset is not None:
            page = json.loads(await recall_group(store, engine.sid, 1, gid, item=0,
                                                char_offset=offset, char_limit=111))
            chunks.append(page['text'])
            offset = page['next_char_offset']
        raw = json.loads(''.join(chunks))
        assert raw['result'] == 'r' * 1500
        assert json.loads(raw['arguments'])['command'] == 'x' * 1500
    finally:
        await store.close()


async def test_background_target_policy_and_original_delivery():
    from power_loop.contracts.tools import ToolDefinition
    from power_loop.runtime.background import encode_results, record_key, summarize_record
    from power_loop.runtime.tool_archive import _derive
    from power_loop.runtime.tool_context import ToolSource, active_engine
    from power_loop.tools.registry import ToolRegistry
    store, engine, p = await fixture({'bash': {'insend_fuse': 'truncate'}}, n=1)
    token = active_engine.set(engine)
    try:
        reg = ToolRegistry()
        reg.register(ToolDefinition('bash', ''), lambda **kw: '')
        engine.registry = reg
        record = {'task_id': 'bg1', 'tool': 'bash', 'status': 'completed',
                  'args': {'command': 'x' * 1500, 'description': 'target description', 'context_policy': 'keep'},
                  'result': 'z' * 5000}
        await store.set_runtime_state(engine.sid, record_key('bg1'), record)
        receipt = summarize_record(record)
        assert receipt['summary'] == record['result']
        source = ToolSource(engine.sid, 1, 2, 'bg-wrapper', 'background_run', {'action': 'check'},
                            encode_results([receipt]))
        assert await _derive(engine, source, 'insend_fuse', 1) is None
        result = await _derive(engine, source, 'insend_fuse', 2)
        assert 'target description' in result[1] and 'zzzz' not in result[1]
        engine.config = replace(engine.config, overrides={'bash': {'hard_preserve': True}})
        result = await _derive(engine, source, 'between_sends', 2)
        assert json.loads(result[1])['targets'][0]['result'] == record['result']
    finally:
        active_engine.reset(token)
        await store.close()


async def test_late_injection_admission_and_output_reserve():
    from power_loop.runtime.tool_archive import admit_request
    store, engine, p = await fixture(n=1)
    try:
        assert not await admit_request(engine, p, [{'role': 'user', 'content': 'z' * 100000}], '', [], 100)
        assert p._capacity_blocked.startswith('context capacity blocked')
        p.config.max_tokens = 4900
        await manage_capacity(engine, p)
        assert p._capacity_blocked
    finally:
        await store.close()


async def test_pg_projection_restart_and_raw_recall():
    import os
    dsn = os.getenv('POWER_LOOP_TEST_PG_DSN')
    if not dsn:
        pytest.skip('requires explicit isolated test database')
    from urllib.parse import urlparse
    assert urlparse(dsn).path.endswith('_test')
    store, engine, p = await fixture(n=4, dsn=dsn)
    try:
        raw, seqs = copy.deepcopy(p.history), list(p.sink._history_seqs)
        await manage_capacity(engine, p)
        view = copy.deepcopy(p.history)
        assert len(view) < len(raw)
        resumed = ToolContextEngine(config=engine.config, store=store,
            owner_session_id=engine.sid, owner_send_index=1, llm=None, model='test')
        await resumed.start()
        p.history, p.sink._history_seqs = raw, seqs
        await manage_capacity(resumed, p)
        assert p.history == view
        gid = next(iter(resumed.state['archive_v2']['groups']))
        restored = json.loads(await recall_group(store, engine.sid, 1, gid, item=0))
        assert restored['result'] == 'r' * 1500
    finally:
        await store.close()


def test_token_estimate_counts_provider_reasoning():
    from power_loop.runtime.budget import estimate_tokens
    bare = estimate_tokens([{'role': 'assistant', 'content': 'ok'}])
    reasoning = estimate_tokens([{'role': 'assistant', 'content': 'ok', 'reasoning_content': 'x' * 4000}])
    assert reasoning >= bare + 999


def test_v2_bash_full_capture_and_required_description(tmp_path):
    from power_loop.runtime.env import RuntimeEnv, runtime_env_context
    from power_loop.runtime.tool_context import active_engine
    from power_loop.tools import default_tools as dt
    token = active_engine.set(SimpleNamespace(config=ToolContextConfig(schema_version=2)))
    try:
        with runtime_env_context(RuntimeEnv(workspace_dir=tmp_path)):
            assert 'description is required' in dt.run_bash('touch should_not_exist')
            assert not (tmp_path / 'should_not_exist').exists()
            result = dt.run_bash('seq 1 12000', description='验证完整标准输出')
            assert '\n12000\n' in result and '\n6000\n' in result
            assert len(result) > 60000 and 'truncated' not in result
    finally:
        active_engine.reset(token)
        dt.close_all_bash_sessions()


async def test_pipeline_capacity_block_does_not_call_provider_again():
    from power_loop import AgentLoopConfig, StatefulAgentLoop
    from power_loop.contracts.tools import ToolDefinition
    from power_loop.runtime.representation import ProjectedRepresentation
    from power_loop.tools.registry import ToolRegistry
    from tests.unit.test_stateful_loop import _Scripted, _tool_resp
    store = await SessionStore.open(':memory:')
    try:
        reg = ToolRegistry()
        reg.register(ToolDefinition('speech', ''), lambda: 'SPEECH' * 10000)
        llm = _Scripted([_tool_resp('a', 'speech', '{}')])
        loop = StatefulAgentLoop(llm=llm, store=store, tool_registry=reg,
            config=AgentLoopConfig(system_prompt='S', max_tokens=100, max_rounds=5, context_checkpoint_tokens=10,
                representation=ProjectedRepresentation(), tool_context=ToolContextConfig(
                    schema_version=2, trigger_tokens=1000, target_tokens=500,
                    window_tokens=5000, reserve_tokens=100, margin_tokens=100,
                    overrides={'speech': {'hard_preserve': True}})))
        result = await loop.send('work', session_id=await loop.new_session())
        assert result.status == 'degraded'
        assert result.final_text.startswith('context capacity blocked')
        assert len(llm.calls) == 1
        assert any(r.content == 'SPEECH' * 10000 for r in await store.load_all_messages(result.session_id))
    finally:
        await store.close()
