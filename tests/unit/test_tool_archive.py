
import copy
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from power_loop.core.pipeline import AgentPipeline
from power_loop.runtime.pricing import Free
from power_loop.runtime.store.store import SessionStore
from power_loop.runtime.tool_archive import manage_capacity, prepare_history, prepare_projection, recall_group
from power_loop.runtime.tool_context import ToolContextConfig, ToolContextEngine
from power_loop.tools import create_default_tool_registry

ARCHIVE = {'bash': {'general': 'archive'}}


def host_entries(rows):
    """What a projector hands over: one entry per tool result, keyed by its seq."""
    return [{'name': 'bash', 'seq': r.seq, 's': 'host line'} for r in rows if r.role == 'tool']


async def fixture(overrides=None, n=3, dsn=None, keep=False):
    from power_loop.runtime.store.factory import open_store
    store = await open_store(dsn) if dsn else await SessionStore.open(':memory:')
    sid = await store.create_session()
    history, seqs = [], []
    for i in range(n):
        args = {'command': 'x' * 1500, 'description': f'目的{i}', 'context_policy': 'keep' if keep else 'retire'}
        call = {'id': str(i), 'type': 'function', 'function': {'name': 'bash', 'arguments': json.dumps(args)}}
        for row in ({'role': 'assistant', 'tool_calls': [call]}, {'role': 'tool', 'tool_call_id': str(i), 'content': 'r' * 1500}):
            seqs.append(await store.append_message(sid, send_index=1, **row))
            history.append(row)
    config = ToolContextConfig(trigger_tokens=1000, target_tokens=500, hot_tail=0,
        window_tokens=5000, reserve_tokens=100, margin_tokens=100,
        overrides=ARCHIVE if overrides is None else overrides)
    engine = ToolContextEngine(config=config, store=store, owner_session_id=sid, owner_send_index=1,
        llm=None, model='test', registry=create_default_tool_registry(bind=False))
    await engine.start()
    p = SimpleNamespace(history=history, sink=SimpleNamespace(_history_seqs=seqs), send_index=1,
        config=SimpleNamespace(max_tokens=100), system_prompt='', runtime_tools=[],
        _tok_len=-1, _tok_total=0)
    p._estimate_history_tokens = lambda: AgentPipeline._estimate_history_tokens(p)
    return store, engine, p


async def test_two_levels_durable_restore_and_full_original():
    store, e, p = await fixture({'bash': {'insend_fuse': 'preserve', 'general': 'archive'}})
    try:
        original = copy.deepcopy(p.history)
        for _ in range(250):
            await prepare_history(e, p, 'age_retirement')
        assert p.history == original
        assert await prepare_history(e, p, 'insend_fuse', level=1) == 0  # level 1 rule: preserve
        await manage_capacity(e, p)
        assert not p._capacity_blocked
        assert e.state['archive']['events'][-1]['levels'][-1]['level'] == 2
        assert '目的0' in p.history[1]['content']
        assert (await store.get_message(e.sid, 2)).content == 'r' * 1500
        view = copy.deepcopy(p.history)
        p.history = original
        p.sink._history_seqs = list(range(1, len(original) + 1))
        resumed = ToolContextEngine(config=replace(e.config, trigger_tokens=2000), store=store,
            owner_session_id=e.sid, owner_send_index=1, llm=None, model='test', registry=create_default_tool_registry(bind=False))
        await resumed.start()
        assert resumed.config.trigger_tokens == 1000
        await manage_capacity(resumed, p)
        assert p.history == view
    finally:
        await store.close()


async def test_hard_preserve_blocks_and_does_not_rescan():
    store, e, p = await fixture({'bash': {'general': 'archive', 'hard_preserve': True}}, n=8)
    try:
        original = copy.deepcopy(p.history)
        await manage_capacity(e, p)
        assert p._capacity_blocked
        assert p.history == original
        events = copy.deepcopy(e.state['archive']['events'])
        await manage_capacity(e, p)
        assert e.state['archive']['events'] == events
    finally:
        await store.close()


async def test_general_groups_exact_recall_and_send_scope():
    store, e, p = await fixture(n=20)
    try:
        rows = await store.load_all_messages(e.sid)
        projected = SimpleNamespace(rows=[SimpleNamespace(kind='project', content={'tools': host_entries(rows)}, rendered_text='old')])
        await prepare_projection(e, projected, rows, 1)
        entries = projected.rows[0].content['tools']
        assert len(entries) == 1 and entries[0]['n'] == 20
        assert '目的0' in entries[0]['s'] and '目的19' in entries[0]['s']
        assert f'group_id="{entries[0]["group_id"]}"' in entries[0]['s']
        gid = next(iter(e.state['archive']['groups']))
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
        ToolContextConfig(**patch)


async def test_stateful_two_sends_result_cap_and_archive_between_sends():
    from power_loop import AgentLoopConfig, StatefulAgentLoop
    from power_loop._vendor.llm_client.interface import LLMResponse
    from power_loop.contracts.tools import ToolDefinition
    from power_loop.runtime.representation import ProjectedRepresentation
    from power_loop.tools.registry import ToolRegistry
    from tests.unit.test_stateful_loop import _Scripted, _tool_resp
    store = await SessionStore.open(':memory:')
    try:
        reg = ToolRegistry()
        reg.register(ToolDefinition('echo', '', pricing=Free("test_fixture")), lambda **kw: 'UNIQUE-RESULT' * 5000)
        llm = _Scripted([_tool_resp('a','echo','{}'), LLMResponse(raw_text='done'), LLMResponse(raw_text='next')])
        loop = StatefulAgentLoop(llm=llm, store=store, tool_registry=reg,
            config=AgentLoopConfig(system_prompt='S', max_tokens=100, max_rounds=5,
                representation=ProjectedRepresentation(),
                tool_context=ToolContextConfig(overrides={'echo': {'general': 'archive'}})))
        sid = await loop.new_session()
        first = await loop.send('work', session_id=sid)
        assert first.status == 'completed'
        stored = next(r.content for r in await store.load_all_messages(sid) if r.role == 'tool')
        assert len(stored) == 20000 and 'chars omitted' in stored   # configurable cap, head + tail
        assert any(m.get('content') == stored for m in llm.calls[1])
        await loop.send('continue', session_id=sid)
        text = json.dumps(llm.calls[-1], ensure_ascii=False)
        assert '（原文 recall_send(' in text and 'UNIQUE-RESULT' not in text
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
        projected = SimpleNamespace(rows=[SimpleNamespace(kind='project', content={'tools': host_entries(rows)}, rendered_text=None)])
        await prepare_projection(engine, projected, rows, 1)
        gid = next(iter(engine.state['archive']['groups']))
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
    store, engine, p = await fixture({'bash': {'insend_fuse': 'truncate', 'general': 'archive'}}, n=1)
    token = active_engine.set(engine)
    try:
        reg = ToolRegistry()
        reg.register(ToolDefinition('bash', '', pricing=Free("test_fixture")), create_default_tool_registry(bind=False).get('bash').handler)
        from power_loop.tools.default_tools import DEFAULT_TOOL_HANDLERS
        reg.register(ToolDefinition('background_run', '', pricing=Free("test_fixture")), DEFAULT_TOOL_HANDLERS['background_run'])
        engine.registry = reg
        record = {'task_id': 'bg1', 'tool': 'bash', 'status': 'completed',
                  'args': {'command': 'x' * 1500, 'description': 'target description', 'context_policy': 'keep'},
                  'result': 'z' * 5000}
        await store.set_runtime_state(engine.sid, record_key('bg1'), record)
        receipt = summarize_record(record)
        assert receipt['summary'] == record['result']
        source = ToolSource(engine.sid, 1, 2, 'bg-wrapper', 'background_run', {'action': 'check'},
                            encode_results([receipt]))
        assert await _derive(engine, source, 'insend_fuse', 1) is None   # target keep: unchanged in-send
        assert await _derive(engine, source, 'insend_fuse', 2) is None
        record['args']['context_policy'] = 'retire'
        await store.set_runtime_state(engine.sid, record_key('bg1'), record)
        result = await _derive(engine, source, 'insend_fuse', 2)
        assert 'target description' in result[1] and 'zzzz' not in result[1]
        engine.config = replace(engine.config, overrides={'bash': {'hard_preserve': True}})
        result = await _derive(engine, source, 'general', 2)
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
            owner_session_id=engine.sid, owner_send_index=1, llm=None, model='test', registry=create_default_tool_registry(bind=False))
        await resumed.start()
        p.history, p.sink._history_seqs = raw, seqs
        await manage_capacity(resumed, p)
        assert p.history == view
        gid = next(iter(resumed.state['archive']['groups']))
        restored = json.loads(await recall_group(store, engine.sid, 1, gid, item=0))
        assert restored['result'] == 'r' * 1500
    finally:
        await store.close()


def test_token_estimate_counts_provider_reasoning():
    from power_loop.runtime.budget import estimate_tokens
    bare = estimate_tokens([{'role': 'assistant', 'content': 'ok'}])
    reasoning = estimate_tokens([{'role': 'assistant', 'content': 'ok', 'reasoning_content': 'x' * 4000}])
    assert reasoning >= bare + 999


def test_bash_full_capture_and_required_description(tmp_path):
    from power_loop.runtime.env import RuntimeEnv, runtime_env_context
    from power_loop.runtime.tool_context import active_engine
    from power_loop.tools import default_tools as dt
    token = active_engine.set(SimpleNamespace(config=ToolContextConfig()))
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
        reg.register(ToolDefinition('speech', '', pricing=Free("test_fixture")), lambda: 'SPEECH' * 10000)
        llm = _Scripted([_tool_resp('a', 'speech', '{}')])
        loop = StatefulAgentLoop(llm=llm, store=store, tool_registry=reg,
            config=AgentLoopConfig(system_prompt='S', max_tokens=100, max_rounds=5, context_checkpoint_tokens=10,
                tool_result_max_chars=None,
                representation=ProjectedRepresentation(), tool_context=ToolContextConfig(
                    trigger_tokens=1000, target_tokens=500,
                    window_tokens=5000, reserve_tokens=100, margin_tokens=100,
                    overrides={'speech': {'hard_preserve': True}})))
        result = await loop.send('work', session_id=await loop.new_session())
        assert result.status == 'degraded'
        assert result.final_text.startswith('context capacity blocked')
        assert len(llm.calls) == 1
        assert any(r.content == 'SPEECH' * 10000 for r in await store.load_all_messages(result.session_id))
    finally:
        await store.close()


async def _send_rows(store, sid, names, send_index=1):
    """One assistant row + result per (name) in order; '__user__' appends an injected user row."""
    for i, name in enumerate(names):
        if name == '__user__':
            await store.append_message(sid, send_index=send_index, role='user', content='[系统注入] 提醒')
            continue
        args = {'command': 'c' * 400, 'description': f'{name}{i}'}
        call = {'id': f'c{i}', 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(args)}}
        await store.append_message(sid, send_index=send_index, role='assistant', content='说明文字', tool_calls=[call])
        await store.append_message(sid, send_index=send_index, role='tool', tool_call_id=f'c{i}', content='r' * 400)
    return await store.load_all_messages(sid)


async def test_between_sends_merge_breakpoints_and_rule_entries_kept():
    store = await SessionStore.open(':memory:')
    try:
        sid = await store.create_session()
        rows = await _send_rows(store, sid, ['bash', 'bash', 'read', 'bash', '__user__', 'bash', 'bash', 'bash'])
        entries = []
        for r in rows:
            if r.role == 'tool':
                name = next(c['function']['name'] for a in rows if a.role == 'assistant'
                            for c in a.tool_calls or [] if c['id'] == r.tool_call_id)
                entries.append({'name': name, 'seq': r.seq, 's': f'{name} line s{r.seq}', 'd': 1})
            elif r.role == 'user':
                entries.append({'name': '__user__', 'seq': r.seq, 's': '[系统注入] 提醒'})
        engine = ToolContextEngine(config=ToolContextConfig(), store=store, owner_session_id=sid,
                                   owner_send_index=2, llm=None, model='test', registry=create_default_tool_registry(bind=False))
        await engine.start()
        projected = SimpleNamespace(rows=[SimpleNamespace(kind='project', content={'tools': entries}, rendered_text='x')])
        await prepare_projection(engine, projected, rows, 1)
        out = projected.rows[0].content['tools']
        # bash×2 | read | bash | injected user line | bash×3 — assistant text between calls never breaks a run
        assert [(e['name'], e.get('n', 1)) for e in out] == [
            ('bash', 2), ('read', 1), ('bash', 1), ('__user__', 1), ('bash', 3)]
        assert out[1] == entries[2] and out[1]['d'] == 1          # general=rule keeps the projector's entry
        group = out[4]
        assert group['seqs'] == [e['seq'] for e in entries[5:]]
        assert all(f'bash line s{s}' in group['s'] for s in group['seqs'])
        assert f'group_id="{group["group_id"]}"' in group['s']
        listing = json.loads(await recall_group(store, sid, 1, group['group_id']))
        assert [m['seq'] for m in listing['items']] == group['seqs']
        assert projected.rows[0].rendered_text is None
    finally:
        await store.close()


async def test_between_sends_parallel_same_tool_is_consecutive_and_merge_can_be_off():
    store = await SessionStore.open(':memory:')
    try:
        sid = await store.create_session()
        calls = [{'id': f'p{i}', 'type': 'function', 'function': {'name': 'read', 'arguments': json.dumps({'path': f'f{i}'})}}
                 for i in range(3)]
        await store.append_message(sid, send_index=1, role='assistant', tool_calls=calls)
        for i in range(3):
            await store.append_message(sid, send_index=1, role='tool', tool_call_id=f'p{i}', content='z' * 300)
        rows = await store.load_all_messages(sid)
        def entries():
            return [{'name': 'read', 'seq': r.seq, 's': f'read s{r.seq}'} for r in rows if r.role == 'tool']
        for send, (merge, expected) in enumerate(((True, 1), (False, 3)), start=2):
            engine = ToolContextEngine(config=ToolContextConfig(merge_consecutive=merge), store=store,
                                       owner_session_id=sid, owner_send_index=send, llm=None, model='test', registry=create_default_tool_registry(bind=False))
            await engine.start()
            projected = SimpleNamespace(rows=[SimpleNamespace(kind='project', content={'tools': entries()}, rendered_text=None)])
            await prepare_projection(engine, projected, rows, 1)
            assert len(projected.rows[0].content['tools']) == expected
    finally:
        await store.close()


async def test_hard_preserve_is_fuse_only_between_sends_uses_general():
    store = await SessionStore.open(':memory:')
    try:
        sid = await store.create_session()
        rows = await _send_rows(store, sid, ['say'])
        seq = next(r.seq for r in rows if r.role == 'tool')
        engine = ToolContextEngine(config=ToolContextConfig(overrides={'say': {'hard_preserve': True, 'general': 'archive'}}),
                                   store=store, owner_session_id=sid, owner_send_index=2, llm=None, model='test', registry=create_default_tool_registry(bind=False))
        await engine.start()
        projected = SimpleNamespace(rows=[SimpleNamespace(kind='project', content={'tools': [{'name': 'say', 'seq': seq, 's': 'host'}]}, rendered_text=None)])
        await prepare_projection(engine, projected, rows, 1)
        assert projected.rows[0].content['tools'] == [{'name': 'say', 'seq': seq, 's': 'host'}]
    finally:
        await store.close()


async def test_fuse_level_one_uses_insend_rule_level_two_falls_back_to_general():
    # load_skill-like tool: kept verbatim inside a send, archived only when level 1 cannot reach the target.
    store, e, p = await fixture({'bash': {'insend_fuse': 'preserve', 'general': 'archive'}}, n=6)
    try:
        original = copy.deepcopy(p.history)
        assert await prepare_history(e, p, 'insend_fuse', target_tokens=10, level=1) == 0
        assert p.history == original
        await manage_capacity(e, p)
        levels = [x['level'] for x in e.state['archive']['events'][-1]['levels']]
        assert levels == [1, 2]
        assert '（原文 recall_send(' in p.history[1]['content']
    finally:
        await store.close()


async def test_hot_tail_protects_newest_calls_at_level_one_only():
    store, e, p = await fixture(n=4, keep=False)
    try:
        e.config = replace(e.config, hot_tail=2, merge_consecutive=False)
        await prepare_history(e, p, 'insend_fuse', target_tokens=10**6, level=1)
        results = [r['content'] for r in p.history if r['role'] == 'tool']
        assert all(c == 'r' * 1500 for c in results[-2:]) and '（原文 recall_send(' in results[0]
        await prepare_history(e, p, 'insend_fuse', target_tokens=10**6, level=2)
        results = [r['content'] for r in p.history if r['role'] == 'tool']
        assert all('（原文 recall_send(' in c for c in results)
    finally:
        await store.close()


async def test_legacy_state_key_is_adopted_and_old_group_ids_recall():
    store = await SessionStore.open(':memory:')
    try:
        sid = await store.create_session()
        rows = await _send_rows(store, sid, ['bash', 'bash'])
        seqs = [r.seq for r in rows if r.role == 'tool']
        from power_loop.runtime.tool_context import TOOL_CONTEXT_STATE_KEY
        legacy = {'groups': {'gOLD': {'send_index': 1, 'members': [
            {'send_index': 1, 'seq': s, 'call_id': f'c{i}', 'name': 'bash'} for i, s in enumerate(seqs)]}}}
        await store.set_runtime_state(sid, TOOL_CONTEXT_STATE_KEY, {'operations': {}, 'attempts': {}, 'archive_v2': legacy})
        assert json.loads(await recall_group(store, sid, 1, 'gOLD'))['count'] == 2
        engine = ToolContextEngine(config=ToolContextConfig(), store=store, owner_session_id=sid,
                                   owner_send_index=2, llm=None, model='test', registry=create_default_tool_registry(bind=False))
        await engine.start()
        assert 'archive_v2' not in engine.state and 'gOLD' in engine.state['archive']['groups']
        restored = json.loads(await recall_group(store, sid, 1, 'gOLD', item=1))
        assert restored['result'] == 'r' * 400
    finally:
        await store.close()


def test_retired_policy_field_is_ignored_and_defaults():
    from power_loop.contracts.context_policy import ToolContextPolicy
    policy = ToolContextPolicy().override({'between_sends': 'archive', 'insend_fuse': 'truncate'})
    assert policy.insend_fuse == 'truncate' and policy.general == 'rule'
    assert ToolContextPolicy().strategy('insend_fuse') == 'rule'
    with pytest.raises(TypeError):
        ToolContextConfig(schema_version=2)


async def test_distilled_text_has_no_prefix_only_a_trailing_recall_note():
    from power_loop.runtime.tool_archive import archive_text
    from power_loop.runtime.tool_context import ToolSource
    bash = ToolSource('s', 1, 3, 'c', 'bash', {'command': 'pytest -q', 'description': '跑单元测试'}, 'ok')
    assert archive_text(bash, create_default_tool_registry(bind=False).get('bash')) == '跑单元测试\n（原文 recall_send(send_index=1, seq=3)）'
    skill = ToolSource('s', 1, 5, 'c', 'load_skill', {'name': 'pdf', 'doc': 'forms.md'}, 'x')
    assert archive_text(skill, create_default_tool_registry(bind=False).get('load_skill')).startswith('pdf/forms.md\n')
    store, e, p = await fixture(n=3, keep=False)
    try:
        await prepare_history(e, p, 'insend_fuse', target_tokens=10**6, level=2)
        text = p.history[1]['content']
        assert text.startswith('目的0') and text.endswith('）')
        assert not any(w in json.dumps(p.history, ensure_ascii=False) for w in ('distilled #', '已归档', '历史 bash'))
    finally:
        await store.close()


async def test_bash_keep_is_unchanged_in_send_at_both_levels_and_distilled_between_sends():
    store, e, p = await fixture(n=8, keep=True)
    try:
        original = copy.deepcopy(p.history)
        await manage_capacity(e, p)
        assert p.history == original and p._capacity_blocked
        rows = await store.load_all_messages(e.sid)
        projected = SimpleNamespace(rows=[SimpleNamespace(kind='project', content={'tools': host_entries(rows)}, rendered_text=None)])
        await prepare_projection(e, projected, rows, 1)
        assert '目的0' in projected.rows[0].content['tools'][0]['s']
    finally:
        await store.close()
