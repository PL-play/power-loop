"""Context behavior follows the actual registered implementation, never its name."""

import json
import os
from dataclasses import replace
from functools import wraps

import pytest

from power_loop.contracts.context_policy import ToolContextPolicy
from power_loop.contracts.tools import ToolDefinition
from power_loop.runtime.pricing import Free
from power_loop.runtime.store.factory import open_store
from power_loop.runtime.store.store import SessionStore
from power_loop.runtime.tool_archive import _derive, archive_label, slim_arguments
from power_loop.runtime.tool_context import ToolContextConfig, ToolContextEngine, ToolSource
from power_loop.runtime.tool_contracts import (
    RuntimeTool,
    ToolContextPresentation,
    ToolContractError,
    ToolRuntimeContract,
    context_presentation,
    policy,
    truncate_context,
)
from power_loop.runtime.tool_truncation import truncate_view
from power_loop.tools import create_default_tool_registry
from power_loop.tools.registry import ToolRegistry


@pytest.fixture(params=['sqlite', 'postgres'])
async def store(request):
    dsn = os.environ.get('POWER_LOOP_TEST_PG_DSN')
    if request.param == 'postgres':
        if not dsn:
            pytest.skip('requires explicit isolated PostgreSQL test database')
        from urllib.parse import urlparse
        assert urlparse(dsn).path.endswith('_test')
        value = await open_store(dsn)
    else:
        value = await SessionStore.open(':memory:')
    try:
        yield value
    finally:
        await value.close()


def source(name, *, keep=False):
    return ToolSource('original', 1, 3, 'call', name,
        {'command': 'x' * 500, 'description': 'Check original evidence', 'context_policy': 'keep' if keep else 'retire'},
        'exit_code=7\n' + 'y' * 500)


def registry_for(builtin, name, *, wrapped=False, subset=False):
    defaults = create_default_tool_registry(bind=False)
    original = defaults.get(builtin)
    handler = original.handler
    if wrapped:
        @wraps(original.handler)
        def handler(**args):
            raise AssertionError('presentation must never execute a tool')
    registry = ToolRegistry()
    registry.register(replace(original.definition, name=name), handler)
    return registry.subset([name]) if subset else registry


@pytest.mark.parametrize('name', ['bash', 'run_original', 'send_message'])
@pytest.mark.parametrize('wrapped,subset', [(False, False), (True, False), (True, True)])
async def test_shell_alias_views_and_preservation_need_actual_binding(name, wrapped, subset, store):
    registry = registry_for('bash', name, wrapped=wrapped, subset=subset)
    registered, original = registry.get(name), source(name, keep=True)
    assert context_presentation(registered, original).keep_in_send
    assert archive_label(original, registered) == 'Check original evidence（exit 7）'
    assert json.loads(slim_arguments(original, registered)) == {'description': 'Check original evidence'}
    pair = truncate_context(registered, original, ToolContextPolicy(), raw_arguments=json.dumps(dict(original.arguments)))
    assert pair[1].startswith('exit_code=7\n' + 'y' * 300 + '…')
    assert json.loads(pair[0])['command'].startswith('x' * 200 + '…')
    assert not policy(registered, {}).native_admitted
    engine = ToolContextEngine(config=ToolContextConfig(overrides={name: {'general': 'archive', 'insend_fuse': 'truncate'}}),
                               store=store, owner_session_id='original', owner_send_index=1, llm=None, model='test', registry=registry)
    assert await _derive(engine, original, 'insend_fuse', 2) is None
    assert await engine.distill(original, 'insend_fuse') is None
    assert await _derive(engine, original, 'general', 2) == (
        slim_arguments(original, registered), 'Check original evidence（exit 7）\n（原文 recall_send(send_index=1, seq=3)）')


@pytest.mark.parametrize('name', ['bash', 'load_skill', 'send_message', 'workflow'])
@pytest.mark.parametrize('fake_attribute', [False, True])
async def test_plain_same_named_handler_receives_no_context_special_behavior(name, fake_attribute, store):
    def handler(**arguments):
        raise AssertionError('not a presentation operation')
    if fake_attribute:
        handler.context_presentation = lambda _: ToolContextPresentation(keep_in_send=True, archive_label='FORGED')
        handler.runtime_contract = ToolRuntimeContract('forged')
    registry = ToolRegistry()
    registry.register(ToolDefinition(name, '', pricing=Free("test_fixture")), handler)
    original, registered = source(name, keep=True), registry.get(name)
    assert context_presentation(registered, original) == ToolContextPresentation()
    assert slim_arguments(original, registered) == '{}'
    assert not archive_label(original, registered).startswith('Check original evidence（exit')
    engine = ToolContextEngine(config=ToolContextConfig(overrides={name: {'insend_fuse': 'truncate'}}),
                               store=store, owner_session_id='original', owner_send_index=1, llm=None, model='test', registry=registry)
    assert not engine.preserves_context(original, 'insend_fuse')
    pair = await _derive(engine, original, 'insend_fuse', 1)
    assert json.loads(pair[1])['preview'].startswith('exit_code=7\n')
    assert not policy(registered, {}).native_admitted


@pytest.mark.parametrize('name', ['load_skill', 'read_handbook', 'bash'])
def test_skill_label_is_owned_by_reader_implementation(name):
    registry = registry_for('load_skill', name, wrapped=True, subset=True)
    original = ToolSource('original', 1, 3, 'call', name, {'name': 'example', 'doc': 'guide.md'}, 'original text')
    assert archive_label(original, registry.get(name)) == 'example/guide.md'
    assert not context_presentation(registry.get(name), original).keep_in_send


def test_unregistered_and_legacy_helpers_use_generic_view_even_for_builtin_name():
    original = source('bash', keep=True)
    assert slim_arguments(original) == '{}'
    pair = truncate_view('bash', json.dumps(dict(original.arguments)), dict(original.arguments),
                         original.result, ToolContextPolicy(), original.recall)
    assert json.loads(pair[1])['preview'].startswith('exit_code=7\n')


@pytest.mark.parametrize('bad', [1, None, 'true'])
def test_presentation_keep_is_strict_boolean(bad):
    with pytest.raises(ToolContractError):
        ToolContextPresentation(keep_in_send=bad)


@pytest.mark.parametrize('bad', ['[]', 'null', '{', 2])
def test_archival_arguments_must_be_json_object(bad):
    with pytest.raises(ToolContractError):
        ToolContextPresentation(archive_arguments=bad)


@pytest.mark.parametrize('bad', [False, 1, {}])
def test_archival_label_must_be_string(bad):
    with pytest.raises(ToolContractError):
        ToolContextPresentation(archive_label=bad)


class HostView(RuntimeTool):
    @property
    def runtime_contract(self):
        return ToolRuntimeContract('example.context-view')

    async def __call__(self, **arguments):
        raise AssertionError('views cannot execute')

    def context_presentation(self, source):
        return ToolContextPresentation(keep_in_send=True, archive_arguments='{"label":"original"}', archive_label='host view')

    def truncate_context(self, source, policy, *, raw_arguments):
        return ('{}', 'bounded host preview')


def test_host_view_extension_needs_no_sdk_business_class_and_rechecks_binding():
    registry = ToolRegistry()
    definition = ToolDefinition('custom_view', '', pricing=Free("test_fixture"))
    registry.register(definition, HostView())
    registered, original = registry.get('custom_view'), source('custom_view')
    assert archive_label(original, registered) == 'host view'
    assert truncate_context(registered, original, ToolContextPolicy(), raw_arguments='{}') == ('{}', 'bounded host preview')
    registered.definition.input_schema['properties']['late'] = {'type': 'string'}
    for read in [lambda: context_presentation(registered, original),
                 lambda: truncate_context(registered, original, ToolContextPolicy(), raw_arguments='{}')]:
        with pytest.raises(ToolContractError, match='binding_changed'):
            read()


@pytest.mark.parametrize('kind', ['presentation', 'truncation'])
def test_invalid_host_view_return_is_rejected(kind):
    class Invalid(HostView):
        def context_presentation(self, source):
            return {'keep_in_send': True}

        def truncate_context(self, source, policy, *, raw_arguments):
            return ['{}', 'preview']
    registry = ToolRegistry()
    registry.register(ToolDefinition('invalid', '', pricing=Free("test_fixture")), Invalid())
    with pytest.raises(ToolContractError):
        if kind == 'presentation':
            context_presentation(registry.get('invalid'), source('invalid'))
        else:
            truncate_context(registry.get('invalid'), source('invalid'), ToolContextPolicy(), raw_arguments='{}')


async def test_alias_archive_restart_and_group_recall_preserve_originals(store):
    from types import SimpleNamespace

    from power_loop.runtime.tool_archive import recall_group
    registry = registry_for('bash', 'inspect_original', wrapped=True)
    sid = await store.create_session()
    history, seqs = [], []
    original = source('inspect_original')
    for index in range(2):
        for row in [
            {'role': 'assistant', 'tool_calls': [{'id': f'call{index}', 'type': 'function',
                'function': {'name': original.name, 'arguments': json.dumps(dict(original.arguments))}}]},
            {'role': 'tool', 'tool_call_id': f'call{index}', 'content': original.result},
        ]:
            seqs.append(await store.append_message(sid, send_index=1, **row))
            history.append(row)
    config = ToolContextConfig(hot_tail=0, overrides={original.name: {'general': 'archive'}})
    engine = ToolContextEngine(config=config, store=store, owner_session_id=sid, owner_send_index=1,
                               llm=None, model='test', registry=registry)
    await engine.start()
    proxy = SimpleNamespace(history=history, sink=SimpleNamespace(_history_seqs=seqs), send_index=1,
                            config=SimpleNamespace(), _tok_len=-1, _tok_total=0)
    assert await engine.prepare_history(proxy, 'general', level=2) > 0
    assert len(proxy.history) == 2
    assert 'Check original evidence（exit 7）' in proxy.history[1]['content']
    groups = engine.state['archive']['groups']
    assert len(groups) == 1
    group_id = next(iter(groups))
    original_group = json.loads(await recall_group(store, sid, 1, group_id, item=1))
    assert original_group['result'] == original.result
    assert original_group['arguments'] == json.dumps(dict(original.arguments))
    restarted = ToolContextEngine(config=config, store=store, owner_session_id=sid, owner_send_index=1,
                                  llm=None, model='test', registry=registry.subset([original.name]))
    await restarted.start()
    assert restarted.state['archive']['groups'] == groups
    assert json.loads(await recall_group(store, sid, 1, group_id, item=1)) == original_group
    rows = await store.load_all_messages(sid)
    assert [row.content for row in rows if row.role == 'tool'] == [original.result] * 2


def test_sdk_skill_registration_helper_preserves_implemented_alias(tmp_path):
    from power_loop.runtime.skills import register_skill_tools
    skill = tmp_path / 'example'
    skill.mkdir()
    (skill / 'SKILL.md').write_text('---\nname: example\n---\nOriginal instructions')
    registry = ToolRegistry()
    register_skill_tools(registry, skills_dir=tmp_path)
    original = registry.get('load_skill')
    registry.register(replace(original.definition, name='read_instructions'), original.handler)
    view = ToolSource('original', 1, 3, 'call', 'read_instructions', {'name': 'example'}, 'Original instructions')
    assert archive_label(view, registry.get('read_instructions')) == 'example'
    assert 'Original instructions' in registry.invoke('read_instructions', {'name': 'example'})
