
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from power_loop.contracts.context_policy import ToolContextPolicy
from power_loop.core.pipeline import AgentPipeline
from power_loop.runtime.budget import estimate_tokens
from power_loop.runtime.pricing import Free
from power_loop.runtime.store.store import SessionStore
from power_loop.runtime.tool_context import ToolContextConfig, ToolContextEngine, ToolSource
from power_loop.runtime.tool_truncation import truncate_pair
from power_loop.tools import create_default_tool_registry


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
async def test_off_on_keep_idempotence_and_recount(trigger):
    store = await SessionStore.open(':memory:')
    try:
        sid = await store.create_session()
        cfg = ToolContextConfig(hot_tail=0, overrides={'bash': {'age_retirement': 'preserve', 'insend_fuse': 'preserve',
            'retention_rounds': 16, 'argument_max_chars': 200, 'result_max_chars': 300}})
        engine = ToolContextEngine(config=cfg, store=store, owner_session_id=sid, owner_send_index=1, llm=None, model='test', registry=create_default_tool_registry(bind=False))
        await engine.start()
        raw = args(170)
        history = [{'role': 'assistant', 'tool_calls': [{'id': 'c', 'function': {'name': 'bash', 'arguments': raw}}]},
                   {'role': 'tool', 'tool_call_id': 'c', 'content': 'y' * 1000}]
        seqs = [await store.append_message(sid, send_index=2, **row) for row in history]
        p = SimpleNamespace(history=history, sink=SimpleNamespace(_history_seqs=seqs), send_index=2,
                            config=SimpleNamespace(), _tok_len=-1, _tok_total=0)
        engine.state['archive'] = {'exposures': {f'2:{seqs[1]}': 16}}
        original = json.dumps(p.history)
        await engine.prepare_history(p, trigger)
        assert json.dumps(p.history) == original
        cfg.overrides['bash'][trigger] = 'truncate'
        cfg.overrides['bash']['general'] = 'truncate'
        # In-flight sends retain their snapshot; the switch takes effect next send.
        assert getattr(engine.policy(ToolSource(sid, 2, seqs[1], 'c', 'bash', {}, '')), trigger) == 'preserve'
        engine = ToolContextEngine(config=cfg, store=store, owner_session_id=sid, owner_send_index=2, llm=None, model='test', registry=create_default_tool_registry(bind=False))
        await engine.start()
        engine.state['archive'] = {'exposures': {f'2:{seqs[1]}': 16}}
        AgentPipeline._estimate_history_tokens(p)
        await engine.prepare_history(p, trigger)
        assert json.dumps(p.history) != original
        content = p.history[1]['content']
        assert content.startswith('y' * 300 + '…（输出共 1000 字，只留前 300 字）')
        assert content.endswith(f'（原文 recall_send(send_index=2, seq={seqs[1]})）')
        assert json.loads(p.history[0]['tool_calls'][0]['function']['arguments']) == {
            'description': '完整目的', 'command': 'x' * 170}
        assert p._tok_len == -1
        assert AgentPipeline._estimate_history_tokens(p) == estimate_tokens(p.history)
        changed = json.dumps(p.history)
        await engine.prepare_history(p, trigger)
        assert json.dumps(p.history) == changed
        source = ToolSource(sid, 1, 3, 'keep', 'bash', {'command': 'x' * 500, 'description': 'rules', 'context_policy': 'keep'}, 'y' * 500)
        assert await engine.distill(source, trigger) is None
        assert await engine.distill(source, 'general') is not None
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
        registry.register(ToolDefinition('bash', '', pricing=Free("test_fixture")), create_default_tool_registry(bind=False).get('bash').handler)
        from power_loop.tools.default_tools import DEFAULT_TOOL_HANDLERS
        registry.register(ToolDefinition('background_run', '', pricing=Free("test_fixture")), DEFAULT_TOOL_HANDLERS['background_run'])
        config = ToolContextConfig(overrides={'bash': {'insend_fuse': 'truncate',
            'general': 'truncate', 'argument_max_chars': 200, 'result_max_chars': 300}})
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
        assert await engine.distill(wrapped, 'general') is not None
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


def test_bash_insend_view_keeps_description_exit_and_cuts_command_and_output():
    from power_loop.runtime.tool_truncation import truncate_bash
    policy = ToolContextPolicy(argument_max_chars=20, result_max_chars=50)
    arguments = {'command': 'python -m pytest ' + 'tests/x.py ' * 20, 'description': '跑测试'}
    result = 'exit_code=1\n' + 'F' * 400
    args_text, text = truncate_bash(arguments, result, policy, 'recall_send(send_index=3, seq=9)')
    args = json.loads(args_text)
    assert args['description'] == '跑测试'
    assert args['command'].startswith(arguments['command'][:20] + '…（命令共')
    lines = text.split('\n')
    assert lines[0] == 'exit_code=1'
    assert lines[1] == 'F' * 50 + '…（输出共 400 字，只留前 50 字）'
    assert lines[-1] == '（原文 recall_send(send_index=3, seq=9)）'
    assert truncate_bash({'command': 'ls', 'description': 'd'}, 'exit_code=0\nok', policy, 'r') is None


def test_bash_general_is_description_exit_code_and_recall_only():
    from power_loop.runtime.tool_archive import archive_text, slim_arguments
    source = ToolSource('s', 2, 7, 'c', 'bash', {'command': 'rm -rf build && make', 'description': '重新构建'},
                        'exit_code=2\nmake: *** error')
    assert archive_text(source, create_default_tool_registry(bind=False).get('bash')) == '重新构建（exit 2）\n（原文 recall_send(send_index=2, seq=7)）'
    assert json.loads(slim_arguments(source, create_default_tool_registry(bind=False).get('bash'))) == {'description': '重新构建'}
