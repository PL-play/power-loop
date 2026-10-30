from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from power_loop._vendor.llm_client.interface import LLMResponse, LLMTokenUsage
from power_loop.contracts.context_policy import ToolContextPolicy
from power_loop.contracts.tools import ToolDefinition
from power_loop.runtime.pricing import Free
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
    registry.register(ToolDefinition('bash', '', project=lambda a, r: a.get('description', 'ok'), pricing=Free("test_fixture")),
                      ShellTool(lambda **kw: 'result' * 300))
    llm = _Scripted([LLMResponse(raw_text='unexpected model request', token_usage=LLMTokenUsage(
        prompt_tokens=100, completion_tokens=20, total_tokens=120))])
    config = ToolContextConfig(overrides={'bash': {'general': 'rule'}})
    engine = ToolContextEngine(config=config, store=store, owner_session_id=sid, owner_send_index=2,
                               registry=registry, llm=llm, model='test')
    await engine.start()
    source = ToolSource(sid, 1, 3, 'call', 'bash', {'command': 'long ' * 300, 'description': 'checks'}, 'ok ' * 300)
    yield store, sid, registry, llm, engine.config, engine, source
    await store.close()


def test_metadata_not_in_schema_and_field_override():
    policy = ToolContextPolicy(insend_fuse='preserve')
    tool = ToolDefinition('load_skill', '', context_policy=policy, pricing=Free("test_fixture"))
    assert 'context_policy' not in tool.to_openai_tool()['function']
    assert policy.override({'general': 'rule'}).insend_fuse == 'preserve'
    with pytest.raises((ValueError, TypeError)):
        policy.override({'typo': 1})


def test_background_summary_entry_points_are_retired():
    import importlib.util

    assert importlib.util.find_spec('power_loop.runtime.context_worker') is None
    assert not hasattr(ToolContextEngine, 'submit')
    assert not hasattr(ToolContextEngine, 'usage_snapshot')
    for retired in ('summary_llm', 'summary_min_chars', 'deliver_usage', 'cache_scope', 'summary_budget_s'):
        assert retired not in ToolContextConfig.__dataclass_fields__


@pytest.mark.parametrize('strategy', ['rule', 'llm_summary'])
async def test_rule_distill_needs_no_model_request(env, strategy):
    store, sid, registry, llm, _, _, source = env
    cfg = ToolContextConfig(overrides={'bash': {'general': strategy, 'insend_fuse': strategy}})
    engine = ToolContextEngine(config=cfg, store=store, owner_session_id=sid, owner_send_index=2,
                               registry=registry, llm=llm, model='test')
    await engine.start()
    text = await engine.distill(source, 'general')
    assert text.startswith('checks') and 'recall_send(send_index=1, seq=3)' in text
    assert (await engine.distill(source, 'insend_fuse')).startswith('checks')
    # A short result below the old summary threshold is distilled by the rule, too.
    assert (await engine.distill(replace(source, seq=4, result='汉' * 7999), 'general')).startswith('checks')
    assert not llm.calls


def test_model_identity_keeps_existing_projection_versions():
    class Transport:
        def __init__(self, model):
            self._cfg = SimpleNamespace(model=model, base_url='https://x/v1')
    Transport.__module__, Transport.__qualname__ = 'fake_transport', 'FakeLLM'
    # Values computed by the release before the background summaries were retired.
    expected = {
        ('deepseek-v3.2', ''): '8a796589c8b8664e9b20ea6524b08feaf22b2bd9f44223eafda9c77538cfa43b',
        ('deepseek-v3.2', 'deepseek-v3.2:1'): 'f2aa0967db101039d0a1466c2c066d5107405f72e1291d77c6b8d7363112ef87',
        ('gpt-5', ''): '97913991d8b8e90adb7c05094bca764204013d2776d43c2de09083151b5fd965',
        (None, ''): '8dd26548aff96942f08003d6cf883ac7654a0501d3e5549c26512f8d5a8f5168',
    }
    for (model, declared), identity in expected.items():
        engine = ToolContextEngine(config=ToolContextConfig(model_identity=declared), store=None,
            owner_session_id='s', owner_send_index=1, llm=Transport(model or 'qwen-max'), model=model)
        assert engine.model_identity == identity


async def test_keep_only_current_send(env):
    _, _, _, _, _, engine, source = env
    source = replace(source, arguments={**source.arguments, 'context_policy': 'keep'})
    assert await engine.distill(source, 'insend_fuse') is None
    assert await engine.distill(source, 'age_retirement') is None
    assert (await engine.distill(source, 'general')).startswith('checks')


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
    engine.config.overrides['bash']['age_retirement'] = 'rule'
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


async def test_background_delegates_target_keep(env):
    from power_loop.runtime.background import encode_results, record_key
    store, sid, _, llm, _, engine, source = env
    record = {'task_id': 'task-1', 'tool': 'bash', 'args': {**source.arguments, 'context_policy': 'keep'},
              'result': source.result, 'status': 'completed'}
    await store.set_runtime_state(sid, record_key('task-1'), record)
    from power_loop.tools.default_tools import DEFAULT_TOOL_HANDLERS
    engine.registry.register(ToolDefinition('background_run', '', pricing=Free("test_fixture")), DEFAULT_TOOL_HANDLERS['background_run'])
    wrapped = replace(source, name='background_run', arguments={'action': 'check', 'task_id': 'task-1'},
                      result=encode_results([record]))
    assert await engine.distill(wrapped, 'insend_fuse') is None
    assert 'task=task-1: checks' in await engine.distill(wrapped, 'general')
    assert not llm.calls


async def test_send_with_tool_context_makes_only_main_model_requests(env):
    from power_loop.agent.stateful_loop import StatefulAgentLoop
    from power_loop.agent.types import AgentLoopConfig
    from power_loop.runtime.representation import ProjectedRepresentation
    store, _, registry, _, config, _, _ = env
    main = _Scripted([_tool_resp('c', 'bash', '{"description": "checks"}'), LLMResponse(raw_text='done'),
                      LLMResponse(raw_text='next')])
    loop = StatefulAgentLoop(llm=main, store=store, tool_registry=registry,
        config=AgentLoopConfig(tool_context=config, representation=ProjectedRepresentation()))
    sid = await loop.new_session()
    try:
        first = await loop.send('test', sid)
        second = await loop.send('next', sid)
        assert first.final_text == 'done' and second.final_text == 'next'
        assert len(main.calls) == 3
        assert not {'auxiliary_pending', 'tool_distill_calls'} & set(first.usage)
        cache = await store._db.fetchone(f'SELECT COUNT(*) AS n FROM {store.t.sessions.removesuffix("sessions")}context_cache')
        assert int(cache['n']) == 0
    finally:
        await loop.aclose()


@pytest.mark.parametrize('candidate', ['<preserve>', '<preserve>summary</preserve>',
                                      '<PRESERVE>summary</PRESERVE>', '</preserve>',
                                      '<preserve reason="constraints">summary'])
def test_preservation_markers_are_never_valid_views(candidate):
    source = ToolSource('s', 1, 3, 'call', 'bash', {}, 'x')
    assert not ToolContextEngine._valid(candidate, 'x' * 10000, source)


async def test_old_operation_with_preserve_marker_is_not_reused(env):
    _, _, _, _, _, engine, source = env
    assert await engine.distill(source, 'general')
    for operation in engine.state['operations'].values():
        operation['text'] = '<preserve>previously accepted candidate</preserve>'
    assert await engine.distill(source, 'general') is None
