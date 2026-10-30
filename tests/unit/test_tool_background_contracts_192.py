"""Background policy and presentation follow registered implementations."""

import asyncio
import copy
from dataclasses import replace
from functools import wraps
from types import SimpleNamespace

import pytest

from power_loop import ToolDefinition
from power_loop.core import agent_context
from power_loop.runtime.background import encode_results, record_key
from power_loop.runtime.pricing import Free
from power_loop.runtime.tool_context import ToolContextConfig, ToolContextEngine, ToolSource
from power_loop.runtime.tool_contracts import (
    FunctionRuntimeTool,
    ToolContractError,
    ToolRuntimeContract,
    background_allowed,
    context_spec,
    result_container,
)
from power_loop.tools import create_default_tool_registry
from power_loop.tools import default_tools as dt
from power_loop.tools.registry import ToolRegistry

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.mark.parametrize('name', ['background_run', 'workflow', 'spawn_agent', 'ordinary'])
async def test_name_alone_does_not_change_background_policy(monkeypatch, name):
    registry, calls = ToolRegistry(), []
    registry.register(ToolDefinition(name, '', async_capable=True, pricing=Free("test_fixture")), lambda **args: calls.append(args) or 'OK')
    monkeypatch.setattr(agent_context, 'get_current_loop', lambda: SimpleNamespace(tool_registry=registry))
    monkeypatch.setattr(dt, '_current_store_and_session', lambda: (None, None))
    monkeypatch.setattr(dt, '_TOOL_TASK_ON_COMPLETE', None)
    manager = dt.BackgroundManager()
    try:
        receipt = await manager.run_tool(name, {'x': 1})
        await asyncio.gather(*manager._workers.values())
        assert 'task_id=' in receipt and calls == [{'x': 1}]
    finally:
        await asyncio.gather(*manager._workers.values(), return_exceptions=True)


@pytest.mark.parametrize('name', ['renamed_scheduler', 'research', 'send_message'])
async def test_contract_forbids_background_before_any_task_is_created(monkeypatch, name):
    registry, calls = ToolRegistry(), []
    implementation = FunctionRuntimeTool(
        lambda **args: calls.append(args), ToolRuntimeContract('example.orchestrator', background_allowed=False))
    registry.register(ToolDefinition(name, '', async_capable=True, pricing=Free("test_fixture")), implementation)
    monkeypatch.setattr(agent_context, 'get_current_loop', lambda: SimpleNamespace(tool_registry=registry))
    manager = dt.BackgroundManager()
    with pytest.raises(ValueError, match='注册契约禁止'):
        await manager.run_tool(name, {})
    assert not manager.tasks and not calls


@pytest.mark.parametrize('change', ['replace', 'revoke', 'contract'])
async def test_queued_call_cannot_use_replaced_or_mutated_registration(monkeypatch, change):
    started, release, calls = asyncio.Event(), asyncio.Event(), []
    registry = ToolRegistry()

    async def original(**args):
        calls.append(args['i'])
        if len(calls) == 8:
            started.set()
        await release.wait()
        return 'OK'

    implementation = FunctionRuntimeTool(original, ToolRuntimeContract('example.worker'))
    registry.register(ToolDefinition('work', '', async_capable=True, pricing=Free("test_fixture")), implementation)
    monkeypatch.setattr(agent_context, 'get_current_loop', lambda: SimpleNamespace(tool_registry=registry))
    monkeypatch.setattr(dt, '_current_store_and_session', lambda: (None, None))
    monkeypatch.setattr(dt, '_TOOL_TASK_ON_COMPLETE', None)
    manager = dt.BackgroundManager()
    try:
        for i in range(9):
            await manager.run_tool('work', {'i': i})
        await started.wait()
        if change == 'replace':
            registry.register(ToolDefinition('work', '', async_capable=True, pricing=Free("test_fixture")),
                              lambda **args: calls.append('replacement'), overwrite=True)
        elif change == 'revoke':
            registry.unregister('work')
        else:
            object.__setattr__(implementation, 'contract', replace(implementation.contract, background_allowed=False))
        release.set()
        await asyncio.gather(*manager._workers.values())
        assert calls == list(range(8))
        rows = list(manager.tasks.values())
        assert rows[-1]['status'] == 'failed' and 'binding_changed' in rows[-1]['result']
    finally:
        release.set()
        await asyncio.gather(*manager._workers.values(), return_exceptions=True)


@pytest.mark.parametrize('name', ['background_run', 'bash', 'custom'])
async def test_plain_names_do_not_modify_model_schema_or_add_scheduler_hints(name):
    registry = ToolRegistry()
    definition = ToolDefinition(name, 'original', input_schema={
        'type': 'object', 'properties': {'action': {'enum': ['run', 'tool']}, 'command': {'type': 'string'}}}, pricing=Free("test_fixture"))
    registry.register(definition, lambda **args: 'unused')
    registry.register(ToolDefinition('image', '', async_capable=True, pricing=Free("test_fixture")), lambda **args: '')
    original = definition.to_openai_tool()
    assert context_spec(registry.get(name), copy.deepcopy(original)) == original
    assert '可异步' not in str(registry.to_openai_tools())
    assert not result_container(registry.get(name))


async def test_renamed_builtins_keep_schema_hints_wrappers_and_sync_shell(tmp_path):
    defaults = create_default_tool_registry(workspace_dir=tmp_path)
    registry = ToolRegistry()
    for source, alias in [('background_run', 'defer_work'), ('bash', 'terminal')]:
        registered = defaults.get(source)
        @wraps(registered.handler)
        def wrapper(**args):
            raise AssertionError('presentation must never invoke a handler')
        registry.register(replace(registered.definition, name=alias), wrapper)
    registry.register(ToolDefinition('image', '', async_capable=True, pricing=Free("test_fixture")), lambda **args: '')
    registry.register(ToolDefinition('recursive', '', async_capable=True, pricing=Free("test_fixture")), FunctionRuntimeTool(
        lambda **args: '', ToolRuntimeContract('example.recursive', background_allowed=False)))
    specs = {s['function']['name']: s['function'] for s in registry.to_openai_tools()}
    assert 'defer_work(action="tool", tool="image"' in specs['image']['description']
    assert '可异步' not in specs['recursive']['description']
    for name in ['defer_work', 'terminal']:
        spec = context_spec(registry.get(name), copy.deepcopy(registry.get(name).definition.to_openai_tool()))
        if name == 'defer_work':
            assert 'command' not in spec['function']['parameters']['properties']
        else:
            assert 'description' in spec['function']['parameters']['required']
    assert not defaults.get('bash').is_async
    assert 'shell-contract-ok' in defaults.invoke('bash', {'command': 'printf shell-contract-ok'})
    registry.register(replace(defaults.get('background_run').definition, name='other_scheduler'),
                      defaults.get('background_run').handler)
    assert '可异步' not in str(registry.to_openai_tools())  # no arbitrary choice between two schedulers


async def test_renamed_result_container_reads_originals_but_plain_same_name_does_not():
    from power_loop import SessionStore
    store = await SessionStore.open(':memory:')
    sid = await store.create_session()
    registry = ToolRegistry()
    defaults = create_default_tool_registry(bind=False)
    registry.register(replace(defaults.get('background_run').definition, name='defer_work'),
                      defaults.get('background_run').handler)
    registry.register(ToolDefinition('work', '', pricing=Free("test_fixture")), lambda **args: '')
    registry.register(ToolDefinition('background_run', '', pricing=Free("test_fixture")), lambda **args: '')
    engine = ToolContextEngine(config=ToolContextConfig(), store=store, owner_session_id=sid,
                               owner_send_index=1, registry=registry, llm=None, model='unused')
    record = {'task_id': 'original', 'tool': 'work', 'args': {}, 'result': 'EXACT', 'status': 'completed'}
    await store.set_runtime_state(sid, record_key('original'), record)
    source = ToolSource(sid, 1, 2, 'call', 'defer_work', {}, encode_results([record]))
    try:
        assert engine.manages('defer_work') and not engine.manages('background_run')
        assert (await engine._background_sources(source))[0].result == 'EXACT'
        assert await engine._background_sources(replace(source, name='background_run')) == []
        record['tool'] = 'defer_work'
        await store.set_runtime_state(sid, record_key('original'), record)
        assert await engine._background_sources(source) == []  # nested containers are not expanded
    finally:
        await store.close()


@pytest.mark.parametrize('field', ['background_allowed', 'result_container'])
@pytest.mark.parametrize('value', [0, 1, None, 'yes'])
async def test_capability_flags_are_strict_booleans(field, value):
    with pytest.raises(ToolContractError):
        ToolRuntimeContract('example.invalid', **{field: value})


async def test_contract_change_after_registration_is_rejected_by_every_consumer():
    registry = ToolRegistry()
    implementation = FunctionRuntimeTool(lambda **args: '', ToolRuntimeContract('example.change'))
    registry.register(ToolDefinition('work', '', pricing=Free("test_fixture")), implementation)
    object.__setattr__(implementation, 'contract', replace(implementation.contract, result_container=True))
    for consumer in [background_allowed, result_container]:
        with pytest.raises(ToolContractError, match='binding_changed'):
            consumer(registry.get('work'))
    with pytest.raises(ToolContractError, match='binding_changed'):
        registry.to_openai_tools()


async def test_original_receipt_recall_survives_alias_and_registration_removal():
    import json

    from power_loop import SessionStore
    from power_loop.runtime.tool_archive import recall_original

    store = await SessionStore.open(':memory:')
    sid = await store.create_session()
    record = {'task_id': 'old', 'tool': 'image', 'args': {'x': 1},
              'result': 'FULL ORIGINAL', 'status': 'completed'}
    try:
        await store.set_runtime_state(sid, record_key('old'), record)
        await store.append_message(sid, send_index=1, role='assistant', tool_calls=[{
            'id': 'old-call', 'type': 'function', 'function': {'name': 'retired_alias', 'arguments': '{}'}}])
        seq = await store.append_message(sid, send_index=1, role='tool', tool_call_id='old-call',
                                         content=encode_results([record]))
        row = await store.get_message(sid, seq)
        original = json.loads(await recall_original(store, sid, 1, row))
        assert original['background_originals'] == [record]
        assert (await store.get_message(sid, seq)).content == row.content
    finally:
        await store.close()
