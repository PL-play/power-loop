from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from power_loop._vendor.llm_client.interface import LLMResponse, LLMTokenUsage
from power_loop.contracts.context_policy import ToolContextPolicy
from power_loop.contracts.tools import ToolDefinition
from power_loop.runtime.context_worker import SummaryWorker
from power_loop.runtime.store.store import SessionStore
from power_loop.runtime.tool_context import ToolContextConfig, ToolContextEngine, ToolSource
from power_loop.tools.registry import ToolRegistry
from power_loop.tools.shell_contract import ShellTool
from tests.unit.test_stateful_loop import _Scripted, _tool_resp


@pytest.fixture
async def env():
    store = await SessionStore.open(':memory:')
    sid = await store.create_session()
    registry = ToolRegistry()
    registry.register(ToolDefinition('bash', '', project=lambda a, r: a.get('description', 'ok')),
                      ShellTool(lambda **kw: 'result' * 300))
    llm = _Scripted([LLMResponse(raw_text='verified exact-file.txt', token_usage=LLMTokenUsage(
        prompt_tokens=100, completion_tokens=20, total_tokens=120, prompt_cached_tokens=50))])
    config = ToolContextConfig(summary_min_chars=0, overrides={'bash': {'general': 'llm_summary'}})
    engine = ToolContextEngine(config=config, store=store, owner_session_id=sid, owner_send_index=2,
                               registry=registry, llm=llm, model='test')
    await engine.start()
    source = ToolSource(sid, 1, 3, 'call', 'bash', {'command': 'long ' * 300, 'description': 'checks'}, 'ok ' * 300)
    yield store, sid, registry, llm, engine.config, engine, source
    await store.close()


async def generate(engine, source):
    await engine.submit(source)
    job = await engine.cache.claim(engine.executor_key)
    assert job is not None
    await SummaryWorker(engine.store).execute(job, engine)
    await engine.cache.reconcile()
    return job


def test_metadata_not_in_schema_and_field_override():
    policy = ToolContextPolicy(insend_fuse='preserve')
    tool = ToolDefinition('load_skill', '', context_policy=policy)
    assert 'context_policy' not in tool.to_openai_tool()['function']
    assert policy.override({'general': 'llm_summary'}).insend_fuse == 'preserve'
    with pytest.raises((ValueError, TypeError)):
        policy.override({'typo': 1})


async def test_read_pending_is_nonblocking_and_no_model_request(env):
    _, _, _, llm, _, engine, source = env
    assert await engine.submit(source)
    assert (await engine.distill(source, 'general')).startswith('checks')
    assert not llm.calls


async def test_completed_shared_cache_and_local_recall(env):
    store, sid, _, llm, _, engine, source = env
    await generate(engine, source)
    other = replace(source, seq=99, call_id='other')
    assert not await engine.submit(other)
    text = await engine.distill(other, 'general')
    assert 'verified exact-file' in text and 'seq=99' in text
    assert len(llm.calls) == 1
    stats = await store.get_session_stats(sid)
    assert stats.total_tokens == 120 and stats.llm_calls == 1 and stats.sends == 0


async def test_concurrent_enqueue_and_claim_are_single_flight(env):
    _, _, _, _, _, engine, source = env
    inserted = await asyncio.gather(*(engine.submit(source) for _ in range(10)))
    assert sum(inserted) == 1
    claims = await asyncio.gather(*(engine.cache.claim(engine.executor_key) for _ in range(10)))
    assert sum(job is not None for job in claims) == 1


async def test_hash_arguments_prompt_and_scope(env):
    _, _, _, _, _, engine, source = env
    policy = engine.policy(source)
    key = engine.cache_key(source, policy)
    assert key == engine.cache_key(replace(source, seq=9, call_id='new'), policy)
    assert key != engine.cache_key(replace(source, arguments={'command': 'different'}), policy)
    assert key != engine.cache_key(source, replace(policy, prompt='different'))
    engine.cache_scope = 'other-user'
    assert key != engine.cache_key(source, policy)


async def test_late_summary_does_not_rewrite_selected_view(env):
    _, _, _, _, _, engine, source = env
    before = await engine.distill(source, 'general')
    await generate(engine, source)
    assert await engine.distill(source, 'general') == before
    assert 'verified exact-file' in await engine.distill(replace(source, seq=4), 'general')


async def test_keep_only_current_send(env):
    _, _, _, _, _, engine, source = env
    source = replace(source, arguments={**source.arguments, 'context_policy': 'keep'})
    await generate(engine, source)
    assert await engine.distill(source, 'insend_fuse') is None
    assert await engine.distill(source, 'age_retirement') is None
    assert 'verified exact-file' in await engine.distill(source, 'general')


@pytest.mark.parametrize('trigger', ['age_retirement', 'insend_fuse'])
async def test_subtoken_rewrite_invalidates_working_context_estimate(env, monkeypatch, trigger):
    from power_loop.core.pipeline import AgentPipeline
    from power_loop.runtime.budget import estimate_tokens
    store, sid, _, _, _, engine, source = env
    raw = json.dumps(dict(source.arguments))
    call = {'id': source.call_id, 'type': 'function', 'function': {'name': 'bash', 'arguments': raw}}
    history = [{'role': 'assistant', 'tool_calls': [call]},
               {'role': 'tool', 'tool_call_id': source.call_id, 'content': source.result}]
    seqs = [await store.append_message(sid, send_index=1, **row) for row in history]
    source = replace(source, seq=seqs[1])
    engine.config.overrides['bash']['retention_rounds'] = 1
    engine.config.overrides['bash']['age_retirement'] = 'llm_summary'
    engine.state['archive'] = {'exposures': {f'1:{seqs[1]}': 1}}
    engine.config = replace(engine.config, hot_tail=0)
    arguments = json.dumps({'description': 'checks'}, ensure_ascii=False, separators=(',', ':'))
    summary = 'x' * (len(raw) + len(source.result) - len(arguments) - 2)   # saves 2 chars: < 1 token

    async def distill(*args, **kwargs):
        return summary

    monkeypatch.setattr(engine, 'distill', distill)
    pipeline = SimpleNamespace(history=history, sink=SimpleNamespace(_history_seqs=seqs), send_index=1,
                               config=SimpleNamespace(), _tok_len=-1, _tok_total=0)
    AgentPipeline._estimate_history_tokens(pipeline)
    assert pipeline._tok_len == len(pipeline.history)
    assert await engine.prepare_history(pipeline, trigger) == 0
    assert pipeline.history[1]['content'] == summary
    assert pipeline._tok_len == -1
    assert AgentPipeline._estimate_history_tokens(pipeline) == estimate_tokens(pipeline.history)


async def test_fuse_with_tool_context_is_capacity_only_and_uses_post_retirement_view(monkeypatch):
    from power_loop.agent.types import AgentLoopConfig
    from power_loop.runtime import tool_archive
    from power_loop.runtime.tool_context import active_engine
    from tests.unit.test_pipeline_tier2_fixes import _pipeline
    # The plain fuse threshold is set but must not run: with tool context the capacity fuse is the only one.
    p = _pipeline(config=AgentLoopConfig(insend_distill_tokens=10))
    await p._append_message({'role': 'tool', 'content': 'x' * 8000})
    original_estimate = p._estimate_history_tokens()
    p._tool_context_prompt_anchor = (81000, original_estimate)
    p.ctx.token_usage = {'prompt_tokens': 81000}
    triggers, capacity = [], []

    class Engine:
        async def prepare_history(self, pipeline, trigger, **kwargs):
            triggers.append(trigger)
            pipeline.history[0]['content'] = 'x' * 400
            pipeline._tok_len = -1
            return 0   # no caller may rely on a positive return to invalidate the cache

    async def manage_capacity(engine, pipeline):
        capacity.append(pipeline._context_prompt_tokens_estimate)

    monkeypatch.setattr(tool_archive, 'manage_capacity', manage_capacity)
    token = active_engine.set(Engine())
    try:
        await p.prepare_round(1)
    finally:
        active_engine.reset(token)
    expected = 81000 + p._estimate_history_tokens() - original_estimate
    assert triggers == ['age_retirement'] and capacity == [expected]
    assert p.ctx.token_usage == {'prompt_tokens': 81000}


async def test_discarded_summary_billed_once_and_fallback(env):
    store, sid, _, llm, _, engine, source = env
    llm.responses[0].raw_text = 'too large ' * 3000
    await generate(engine, source)
    assert (await engine.distill(source, 'general')).startswith('checks')
    assert not await engine.submit(source)
    await engine.cache.reconcile()
    stats = await store.get_session_stats(sid)
    assert stats.total_tokens == 120 and stats.llm_calls == 1


async def test_failed_summary_unknown_and_durable_outbox(env):
    _, _, _, _, _, engine, source = env
    class Broken:
        async def complete(self, request):
            raise TimeoutError()
    engine.llm = Broken()
    await generate(engine, source)
    deliveries = []
    async def receiver(record):
        deliveries.append(record)
        if len(deliveries) == 1:
            raise RuntimeError('offline')
    await engine.cache.reconcile(receiver)
    await engine.cache.reconcile(receiver)
    await engine.cache.reconcile(receiver)
    assert len(deliveries) == 2
    assert deliveries[0]['attempt_id'] == deliveries[1]['attempt_id']
    assert deliveries[0]['usage_unknown'] and deliveries[0]['owner_send_index'] == 2
    assert deliveries[0]['source_send_index'] == 1


async def test_expired_started_lease_is_uncertain_not_retried(env):
    _, _, _, _, _, engine, source = env
    await engine.submit(source)
    job = await engine.cache.claim(engine.executor_key)
    record = {'attempt_id': 'lost', 'owner_session_id': engine.sid, 'outcome': 'started', 'usage': None}
    await engine.cache.begin_attempt(job, record)
    await engine.cache.db.execute(f'UPDATE {engine.cache.cache} SET lease_until=0')
    assert await engine.cache.claim(engine.executor_key) is None
    await engine.cache.reconcile()
    rows = await engine.cache.db.fetchall(f'SELECT outcome,record_json FROM {engine.cache.attempts}')
    assert rows[0]['outcome'] == 'interrupted'
    assert json.loads(rows[0]['record_json'])['usage_unknown']


async def test_eviction_keeps_ledger_and_frozen_view(env):
    store, sid, _, _, _, engine, source = env
    await generate(engine, source)
    selected = await engine.distill(source, 'general')
    assert await engine.cache.prune(max_entries=0, max_bytes=0) == 1
    assert await engine.cache.get(engine.cache_key(source, engine.policy(source))) is None
    assert await engine.distill(source, 'general') == selected
    assert (await store.get_session_stats(sid)).total_tokens == 120
    assert len(await engine.cache.db.fetchall(f'SELECT * FROM {engine.cache.attempts}')) == 1


async def test_background_delegates_target_keep(env):
    from power_loop.runtime.background import encode_results, record_key
    store, sid, _, llm, _, engine, source = env
    record = {'task_id': 'task-1', 'tool': 'bash', 'args': {**source.arguments, 'context_policy': 'keep'},
              'result': source.result, 'status': 'completed'}
    await store.set_runtime_state(sid, record_key('task-1'), record)
    from power_loop.tools.default_tools import DEFAULT_TOOL_HANDLERS
    engine.registry.register(ToolDefinition('background_run', ''), DEFAULT_TOOL_HANDLERS['background_run'])
    wrapped = replace(source, name='background_run', arguments={'action': 'check', 'task_id': 'task-1'},
                      result=encode_results([record]))
    assert await engine.submit(wrapped)
    job = await engine.cache.claim(engine.executor_key)
    await SummaryWorker(store).execute(job, engine)
    assert await engine.distill(wrapped, 'insend_fuse') is None
    assert 'verified exact-file' in await engine.distill(wrapped, 'general')
    assert len(llm.calls) == 1


async def test_send_returns_while_summary_running(env):
    from power_loop.agent.stateful_loop import StatefulAgentLoop
    from power_loop.agent.types import AgentLoopConfig
    from power_loop.runtime.representation import ProjectedRepresentation
    store, _, registry, _, config, _, _ = env
    blocked = asyncio.Event()
    class Slow:
        async def complete(self, request):
            await blocked.wait()
            return LLMResponse(raw_text='summary')
    config.summary_llm = Slow()
    main = _Scripted([_tool_resp('c', 'bash', '{}'), LLMResponse(raw_text='done'), LLMResponse(raw_text='next')])
    loop = StatefulAgentLoop(llm=main, store=store, tool_registry=registry,
        config=AgentLoopConfig(tool_context=config, representation=ProjectedRepresentation()))
    sid = await loop.new_session()
    try:
        first = await asyncio.wait_for(loop.send('test', sid), 2)
        second = await asyncio.wait_for(loop.send('next', sid), 2)
        assert first.final_text == 'done' and second.final_text == 'next'
        assert not blocked.is_set()
    finally:
        blocked.set()
        await loop.aclose()


async def test_capacity_applies_backpressure_without_model(env):
    _, _, _, llm, _, engine, source = env
    engine.config.cache_max_entries = 1
    assert await engine.submit(source)
    assert not await engine.submit(replace(source, arguments={'command': 'second'}))
    assert not llm.calls


async def test_ttl_and_approximate_lru(env):
    _, _, _, _, _, engine, source = env
    await generate(engine, source)
    key = engine.cache_key(source, engine.policy(source))
    await engine.cache.db.execute(f'UPDATE {engine.cache.cache} SET expires_at=0 WHERE cache_key=?', (key,))
    assert await engine.cache.get(key) is None
    await engine.cache.prune(max_entries=100, max_bytes=1000000)
    assert await engine.submit(source)


async def test_explicit_503_backoff_distinct_attempts(env):
    _, _, _, _, _, engine, source = env
    class ServiceError(Exception):
        status_code = 503
    class Busy:
        async def complete(self, request):
            raise ServiceError()
    engine.llm = Busy()
    await generate(engine, source)
    assert await engine.cache.claim(engine.executor_key) is None
    await engine.cache.db.execute(f'UPDATE {engine.cache.cache} SET retry_at=0')
    job = await engine.cache.claim(engine.executor_key)
    await SummaryWorker(engine.store).execute(job, engine)
    assert await engine.cache.claim(engine.executor_key) is None
    rows = await engine.cache.db.fetchall(f'SELECT attempt_id FROM {engine.cache.attempts}')
    assert len(rows) == 2 and rows[0]['attempt_id'] != rows[1]['attempt_id']


async def test_cancelled_job_stays_uncertain_after_cache_eviction(env):
    _, _, _, _, _, engine, source = env
    class Cancel:
        async def complete(self, request):
            raise asyncio.CancelledError()
    engine.llm = Cancel()
    await engine.submit(source)
    job = await engine.cache.claim(engine.executor_key)
    with pytest.raises(asyncio.CancelledError):
        await SummaryWorker(engine.store).execute(job, engine)
    await engine.cache.prune(max_entries=0, max_bytes=0)
    assert not await engine.submit(source)


async def test_reconcile_stale_scan_cannot_overwrite_success(env, monkeypatch):
    store, sid, _, _, _, engine, source = env
    await engine.submit(source)
    job = await engine.cache.claim(engine.executor_key)
    record = {'attempt_id': 'race', 'owner_session_id': sid, 'outcome': 'started', 'usage': None}
    await engine.cache.begin_attempt(job, record)
    fetchall = engine.cache.db.fetchall

    async def stale_scan(sql, *args):
        rows = await fetchall(sql, *args)
        if sql.startswith('SELECT * FROM ' + engine.cache.attempts):
            await engine.cache.finish(job, {**record, 'outcome': 'success',
                'usage': {'prompt_tokens': 100, 'completion_tokens': 20, 'total_tokens': 120}}, None, None)
        return rows

    monkeypatch.setattr(engine.cache.db, 'fetchall', stale_scan)
    received = []
    async def receiver(record):
        received.append(record)
    await engine.cache.reconcile(receiver)
    assert received[0]['outcome'] == 'success'
    assert (await store.get_session_stats(sid)).total_tokens == 120


async def test_late_receipt_corrects_unknown_without_second_call(env):
    store, sid, _, _, _, engine, source = env
    await engine.submit(source)
    job = await engine.cache.claim(engine.executor_key)
    record = {'attempt_id': 'late', 'owner_session_id': sid, 'outcome': 'started', 'usage': None}
    await engine.cache.begin_attempt(job, record)
    await engine.cache.db.execute(f'UPDATE {engine.cache.cache} SET lease_until=0')
    received = []
    async def receiver(receipt):
        received.append(receipt.copy())
        if len(received) == 1:
            # Provider completes while the old interrupted receipt is being ACKed.
            await engine.cache.finish(job, {**record, 'outcome': 'success', 'usage_unknown': False,
                'usage': {'prompt_tokens': 100, 'completion_tokens': 20, 'total_tokens': 120}}, None, None)
    await engine.cache.reconcile(receiver)
    row = await engine.cache.db.fetchone(f'SELECT * FROM {engine.cache.attempts}')
    assert row['accounted'] == 0 and row['delivered'] == 0
    await engine.cache.reconcile(receiver)
    await engine.cache.reconcile(receiver)
    assert [r['revision'] for r in received] == [1, 2]
    assert [r['outcome'] for r in received] == ['interrupted', 'success']
    stats = await store.get_session_stats(sid)
    assert stats.total_tokens == 120 and stats.llm_calls == 1
    # An out-of-order accounting receipt cannot undo the correction.
    await store.record_auxiliary_usage(sid, 'late', {}, revision=1)
    assert (await store.get_session_stats(sid)).total_tokens == 120


async def test_durable_pending_and_receipt_survive_store_reopen(tmp_path):
    from power_loop.runtime.store.context_cache import ContextCache
    path = str(tmp_path / 'recovery.db')
    store = await SessionStore.open(path)
    sid = await store.create_session()
    cache = ContextCache(store)
    await cache.enqueue('pending', 'executor', {'owner_session_id': sid, 'timeout_s': 60}, 60)
    await store.close()
    store = await SessionStore.open(path)
    cache = ContextCache(store)
    job = await cache.claim('executor')
    assert job['cache_key'] == 'pending'
    record = {'attempt_id': 'recovery', 'owner_session_id': sid, 'outcome': 'started', 'usage': None}
    await cache.begin_attempt(job, record)
    await cache.db.execute(f'UPDATE {cache.cache} SET lease_until=0')
    await store.close()
    store = await SessionStore.open(path)
    cache = ContextCache(store)
    try:
        receipts = []
        async def receiver(record):
            receipts.append(record)
        await cache.reconcile(receiver)
        await cache.reconcile(receiver)
        assert len(receipts) == 1 and receipts[0]['outcome'] == 'interrupted'
        assert receipts[0]['usage_unknown']
        assert await cache.claim('executor') is None
        stats = await store.get_session_stats(sid)
        assert stats.llm_calls == 1 and stats.total_tokens == 0
    finally:
        await store.close()


async def test_stale_worker_does_not_issue_provider_request(env):
    _, _, _, llm, _, engine, source = env
    await engine.submit(source)
    job = await engine.cache.claim(engine.executor_key)
    await engine.cache.db.execute(f'UPDATE {engine.cache.cache} SET lease_until=0')
    await SummaryWorker(engine.store).execute(job, engine)
    assert not llm.calls
    assert not await engine.cache.db.fetchall(f'SELECT * FROM {engine.cache.attempts}')


async def test_hard_process_exit_preserves_started_attempt(tmp_path):
    import subprocess
    import sys

    from power_loop.runtime.store.context_cache import ContextCache

    path = str(tmp_path / 'hard-exit.db')
    script = '''
import asyncio, os, sys
from power_loop.runtime.store.store import SessionStore
from power_loop.runtime.store.context_cache import ContextCache
async def main():
    store = await SessionStore.open(sys.argv[1])
    sid = await store.create_session()
    cache = ContextCache(store)
    await cache.enqueue('crash', 'executor', {'owner_session_id': sid, 'timeout_s': 60}, 60)
    job = await cache.claim('executor')
    await cache.begin_attempt(job, {'attempt_id': 'hard-crash', 'owner_session_id': sid,
        'parent_session_id': 'parent', 'workflow_run_id': 'workflow', 'node_id': 'review',
        'outcome': 'started', 'usage': None})
    os._exit(23)
asyncio.run(main())
'''
    result = await asyncio.to_thread(subprocess.run, [sys.executable, '-c', script, path],
                                    capture_output=True, text=True, timeout=20)
    assert result.returncode == 23, result.stderr
    store = await SessionStore.open(path)
    cache = ContextCache(store)
    try:
        await cache.db.execute(f'UPDATE {cache.cache} SET lease_until=0')
        received = []
        async def receiver(record):
            received.append(record)
        await cache.reconcile(receiver)
        await cache.reconcile(receiver)
        assert len(received) == 1 and received[0]['usage_unknown']
        assert received[0]['parent_session_id'] == 'parent'
        assert received[0]['workflow_run_id'] == 'workflow' and received[0]['node_id'] == 'review'
        assert await cache.claim('executor') is None
    finally:
        await store.close()


async def test_summary_disables_deepseek_thinking_without_changing_main_config(env):
    store, sid, registry, llm, config, _, source = env
    requests = []
    complete = llm.complete
    async def capture(request, **kwargs):
        requests.append(request)
        return await complete(request, **kwargs)
    llm.complete = capture
    config = replace(config, overrides={'bash': {'general': 'llm_summary'}})
    engine = ToolContextEngine(config=config, store=store, owner_session_id=sid,
                               owner_send_index=3, llm=llm, model='deepseek-flash', registry=registry)
    await engine.start()
    job = await generate(engine, source)
    assert job['payload']['request_extra'] == {'extra_body': {'thinking': {'type': 'disabled'}}}
    assert requests[-1].extra == job['payload']['request_extra']
    inherited = ToolContextEngine(config=replace(config, summary_request_extra={}), store=store,
        owner_session_id=sid, owner_send_index=4, llm=llm, model='deepseek-flash', registry=registry)
    assert inherited.cache_key(source, inherited.policy(source)) != engine.cache_key(source, engine.policy(source))
    assert inherited.executor_key != engine.executor_key
    assert config.summary_request_extra is None


async def test_default_small_summary_preserves_original_without_enqueue_or_rule(env):
    store, sid, registry, llm, _, _, source = env
    cfg = ToolContextConfig(overrides={'bash': {'general': 'llm_summary',
                                               'insend_fuse': 'llm_summary'}})
    engine = ToolContextEngine(config=cfg, store=store, owner_session_id=sid,
                               owner_send_index=5, registry=registry, llm=llm, model='test')
    await engine.start()
    below = replace(source, result='汉' * 7999)
    assert not await engine.submit(below)
    assert await engine.distill(below, 'general') is None
    assert await engine.distill(below, 'insend_fuse') is None
    assert not llm.calls
    assert await engine.submit(replace(source, result='汉' * 8000))
    # Rule strategies ignore the LLM admission threshold.
    assert await engine.distill(below, 'age_retirement') is None
    cfg.overrides['bash']['age_retirement'] = 'rule'
    # Engine snapshots overrides at start, so build the rule engine explicitly.
    rule_engine = ToolContextEngine(config=cfg, store=store, owner_session_id=sid,
        owner_send_index=6, registry=registry, llm=llm, model='test')
    await rule_engine.start()
    assert 'checks' in await rule_engine.distill(below, 'age_retirement')


@pytest.mark.parametrize('candidate', ['<preserve>', '<preserve>summary</preserve>',
                                      '<PRESERVE>summary</PRESERVE>', '</preserve>',
                                      '<preserve reason="constraints">summary'])
async def test_preservation_markers_never_become_cached_summaries(env, candidate):
    store, sid, _, llm, _, engine, source = env
    llm.responses[0].raw_text = candidate
    await generate(engine, source)
    assert await engine.cache.get(engine.cache_key(source, engine.policy(source))) is None
    assert not engine._valid(candidate, 'x' * 10000, source)
    # Rejected candidates still incur a real accounted request.
    assert (await store.get_session_stats(sid)).total_tokens == 120


async def test_old_operation_with_preserve_marker_is_not_reused(env):
    _, _, _, _, _, engine, source = env
    await generate(engine, source)
    assert await engine.distill(source, 'general')
    for operation in engine.state['operations'].values():
        operation['text'] = '<preserve>previously accepted candidate</preserve>'
    assert await engine.distill(source, 'general') is None
