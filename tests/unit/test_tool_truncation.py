import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from power_loop.contracts.context_policy import ToolContextPolicy
from power_loop.core.pipeline import AgentPipeline
from power_loop.runtime.budget import estimate_tokens
from power_loop.runtime.store.store import SessionStore
from power_loop.runtime.tool_context import ToolContextConfig, ToolContextEngine, ToolSource
from power_loop.runtime.tool_truncation import truncate_pair


def args(n):
    return json.dumps({'command': 'x' * n, 'description': '完整目的'}, ensure_ascii=False)


@pytest.mark.parametrize('a,r', [(200, 300), (201, 300), (200, 301), (201, 301)])
def test_independent_limits(a, r):
    raw = args(1)
    raw = raw.replace('"x"', '"' + 'x' * (a - len(raw) + 1) + '"')
    assert len(raw) == a
    pair = truncate_pair(raw, 'y' * r, ToolContextPolicy(), 'recall_send(1,3)')
    if a == 200 and r == 300:
        assert pair is None
        return
    assert (pair[0] == raw) == (a == 200)
    assert (pair[1] == 'y' * r) == (r == 300)
    for original, view, limit in [(raw, pair[0], 200), ('y' * r, pair[1], 300)]:
        if original != view:
            value = json.loads(view)
            assert value['preview'] == original[:limit]
            assert value['description'] == '完整目的'
            assert f'前{limit}字符' in value['notice']
            assert value['_distilled'] == 'recall_send(1,3)'


@pytest.mark.parametrize('field', ['argument_max_chars', 'result_max_chars'])
@pytest.mark.parametrize('bad', [0, -1, True, 1.5, '200'])
def test_invalid_limit(field, bad):
    with pytest.raises(ValueError):
        ToolContextPolicy(**{field: bad})


def test_side_disabled_without_losing_other_limit():
    policy = ToolContextPolicy(argument_max_chars=None)
    raw = args(500)
    pair = truncate_pair(raw, 'y' * 500, policy, 'r')
    assert pair[0] == raw
    assert json.loads(pair[1])['preview'] == 'y' * 300
    assert replace(policy, result_max_chars=None).argument_max_chars is None
    assert truncate_pair(raw, 'y' * 500, replace(policy, result_max_chars=None), 'r') is None


@pytest.mark.parametrize('trigger', ['age_retirement', 'insend_fuse'])
async def test_off_on_keep_idempotence_and_zero_saving_recount(trigger):
    store = await SessionStore.open(':memory:')
    try:
        sid = await store.create_session()
        cfg = ToolContextConfig(overrides={'bash': {'age_retirement': 'preserve', 'insend_fuse': 'preserve',
            'retention_rounds': 16, 'argument_max_chars': 200, 'result_max_chars': 300}})
        engine = ToolContextEngine(config=cfg, store=store, owner_session_id=sid, owner_send_index=1, llm=None, model='test')
        await engine.start()
        raw = args(170)  # near threshold: notices can make the working view larger
        p = SimpleNamespace(history=[{'role': 'assistant', 'tool_calls': [
            {'id': 'c', 'function': {'name': 'bash', 'arguments': raw}}]},
            {'role': 'tool', 'tool_call_id': 'c', 'content': 'y' * 301}],
            sink=SimpleNamespace(_history_seqs=[2, 3]), send_index=1,
            config=SimpleNamespace(insend_distill_hot_tail=0, insend_distill_keep_tools=()),
            _tok_len=-1, _tok_total=0)
        engine.state.setdefault('exposures', {})['3'] = 16
        original = json.dumps(p.history)
        await engine.prepare_history(p, trigger)
        assert json.dumps(p.history) == original
        cfg.overrides['bash'][trigger] = 'truncate'
        cfg.overrides['bash']['between_sends'] = 'truncate'
        # In-flight sends retain their snapshot; the switch takes effect next send.
        assert getattr(engine.policy(ToolSource(sid, 1, 3, 'c', 'bash', {}, '')), trigger) == 'preserve'
        engine = ToolContextEngine(config=cfg, store=store, owner_session_id=sid, owner_send_index=2, llm=None, model='test')
        await engine.start()
        engine.state.setdefault('exposures', {})['3'] = 16
        p.send_index = 2
        AgentPipeline._estimate_history_tokens(p)
        await engine.prepare_history(p, trigger)
        assert p._tok_len == -1
        assert AgentPipeline._estimate_history_tokens(p) == estimate_tokens(p.history)
        changed = json.dumps(p.history)
        await engine.prepare_history(p, trigger)
        assert json.dumps(p.history) == changed
        source = ToolSource(sid, 1, 3, 'keep', 'bash', {'command': 'x' * 500, 'description': 'rules', 'context_policy': 'keep'}, 'y' * 500)
        assert await engine.distill(source, trigger) is None
        assert await engine.distill(source, 'between_sends') is not None
        assert cfg.overrides['bash']['retention_rounds'] == 16
    finally:
        await store.close()


async def test_background_uses_target_limits_and_keep():
    from power_loop.contracts.tools import ToolDefinition
    from power_loop.runtime.background import encode_results, record_key
    from power_loop.tools.registry import ToolRegistry
    store = await SessionStore.open(':memory:')
    try:
        sid = await store.create_session()
        registry = ToolRegistry()
        registry.register(ToolDefinition('bash', ''), lambda **kw: '')
        config = ToolContextConfig(overrides={'bash': {'insend_fuse': 'truncate',
            'between_sends': 'truncate', 'argument_max_chars': 200, 'result_max_chars': 300}})
        engine = ToolContextEngine(config=config, store=store, owner_session_id=sid,
                                   owner_send_index=1, llm=None, model='test', registry=registry)
        await engine.start()
        record = {'task_id': 'job', 'tool': 'bash', 'args': {'command': 'x' * 2000,
                  'description': 'background check'}, 'result': 'y' * 3000, 'status': 'completed'}
        await store.set_runtime_state(sid, record_key('job'), record)
        wrapped = ToolSource(sid, 1, 5, 'check', 'background_run', {'action': 'check'}, encode_results([record]))
        text = await engine.distill(wrapped, 'insend_fuse')
        assert 'background check' in text and 'recall_send' in text
        assert 'x' * 201 not in text and 'y' * 301 not in text
        record['args']['context_policy'] = 'keep'
        await store.set_runtime_state(sid, record_key('job'), record)
        assert await engine.distill(wrapped, 'insend_fuse') is None
        assert await engine.distill(wrapped, 'between_sends') is not None
    finally:
        await store.close()


def test_side_switch_off_on_retains_custom_values():
    policy = ToolContextPolicy(argument_max_chars=777, result_max_chars=888,
                               argument_truncation_enabled=False, result_truncation_enabled=False)
    raw, result = args(2000), 'r' * 3000
    assert truncate_pair(raw, result, policy, 'recall') is None
    enabled = replace(policy, argument_truncation_enabled=True, result_truncation_enabled=True)
    pair = truncate_pair(raw, result, enabled, 'recall')
    assert len(json.loads(pair[0])['preview']) == 777
    assert len(json.loads(pair[1])['preview']) == 888
    assert policy.argument_max_chars == 777 and policy.result_max_chars == 888


@pytest.mark.parametrize('key', ['argument_truncation_enabled', 'result_truncation_enabled'])
@pytest.mark.parametrize('value', [0, 1, None, 'false'])
def test_side_switch_is_strict_boolean(key, value):
    with pytest.raises(ValueError):
        ToolContextPolicy(**{key: value})


def test_incomplete_archive_is_preserved():
    result = json.dumps({'capture_complete': True, 'truncated': False,
                         'output': 'x' * 350, 'output_chars': 3000})
    assert truncate_pair(args(500), result, ToolContextPolicy(), 'recall') is None
