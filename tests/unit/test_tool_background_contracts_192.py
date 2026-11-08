"""Background policy and presentation follow registered implementations."""

import asyncio
import copy
import json
from dataclasses import replace
from functools import wraps

import pytest

from power_loop import AgentLoopConfig, JobRef, ToolDefinition
from power_loop.agent.runtime_store import now_ms
from power_loop.runtime.background import encode_results, load_record
from tests.runtime_fixture import RunningAgent

from .test_job_registry_192 import registry_store
from .test_timers import _Scripted, _tool_resp

__all__ = ["registry_store"]
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
from power_loop.tools.registry import ToolRegistry

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def start_background(store, registry, name, arguments=None, *, runtime_env=None):
    default = create_default_tool_registry(bind=False).get('background_run')
    registry.register(replace(default.definition, name='defer_work'), default.handler)
    model = _Scripted([_tool_resp('background','defer_work',json.dumps({
        'action':'tool','tool':name,'args':arguments or {}}))])
    runner = await RunningAgent(store,model,registry,AgentLoopConfig(compactor=None), runtime_env=runtime_env).start()
    await runner.send('start')
    rows = await store.load_all_messages(runner.target.session_id)
    return runner, next(row for row in rows if row.tool_call_id=='background')


@pytest.mark.parametrize('name',['background_run','workflow','spawn_agent','ordinary'])
async def test_name_alone_does_not_change_background_policy(registry_store,name):
    store,_ = registry_store
    registry,calls = ToolRegistry(),[]
    registry.register(ToolDefinition(name,'',async_capable=True,pricing=Free('fixture')),lambda **args:calls.append(args) or 'OK')
    runner,row = await start_background(store,registry,name,{'x':1})
    try:
        result = await runner.runtime.client.wait(JobRef(**json.loads(row.content)['job_ref']),actor=runner.actor,deadline_ms=now_ms()+10000)
        assert result.status=='completed' and result.final_text=='OK' and calls==[{'x':1}]
    finally:
        await runner.aclose()


@pytest.mark.parametrize('name',['renamed_scheduler','research','send_message'])
async def test_contract_forbids_background_before_any_task_is_created(registry_store,name):
    store,_ = registry_store
    registry,calls = ToolRegistry(),[]
    implementation=FunctionRuntimeTool(lambda **args:calls.append(args),ToolRuntimeContract('example.orchestrator',background_allowed=False))
    registry.register(ToolDefinition(name,'',async_capable=True,pricing=Free('fixture')),implementation)
    runner,row = await start_background(store,registry,name)
    try:
        assert 'does not allow background execution' in row.content
        assert not calls and not await store._db.fetchall(f"SELECT job_id FROM {store.table_prefix}jobs WHERE kind='background_tool'")
    finally:
        await runner.aclose()


@pytest.mark.parametrize('change',['replace','revoke','contract'])
async def test_queued_call_cannot_use_replaced_or_mutated_registration(registry_store,change):
    store,_ = registry_store
    began,release,calls=asyncio.Event(),asyncio.Event(),[]
    async def original(i):
        calls.append(i)
        began.set()
        await release.wait()
        return 'OK'
    registry=create_default_tool_registry(bind=False)
    implementation=FunctionRuntimeTool(original,ToolRuntimeContract('example.worker'))
    registry.register(ToolDefinition('work','',async_capable=True,pricing=Free('fixture')),implementation)
    model=_Scripted([_tool_resp(f'bg{i}','background_run',json.dumps({'action':'tool','tool':'work','args':{'i':i}})) for i in range(2)])
    runner=await RunningAgent(store,model,registry,AgentLoopConfig(compactor=None)).start()
    try:
        await runner.send('queue two')
        await asyncio.wait_for(began.wait(),10)
        if change=='replace':
            registry.register(ToolDefinition('work','new',async_capable=True,pricing=Free('fixture')),lambda **args:calls.append('replacement'),overwrite=True)
        elif change=='revoke':
            registry.unregister('work')
        else:
            registry.register(ToolDefinition('work','',async_capable=True,pricing=Free('fixture')),
                FunctionRuntimeTool(original,replace(implementation.contract,background_allowed=False)),overwrite=True)
        release.set()
        rows=await store.load_all_messages(runner.target.session_id)
        receipt=json.loads(next(row.content for row in rows if row.tool_call_id=='bg1'))
        result=await runner.runtime.client.wait(JobRef(**receipt['job_ref']),actor=runner.actor,deadline_ms=now_ms()+10000)
        assert result.status=='failed' and calls==[0]
    finally:
        release.set()
        await runner.aclose()


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
            assert 'command' in spec['function']['parameters']['properties']
            assert 'run' in spec['function']['parameters']['properties']['action']['enum']
        else:
            assert 'description' in spec['function']['parameters']['required']
    assert not defaults.get('bash').is_async
    assert 'shell-contract-ok' in defaults.invoke('bash', {'command': 'printf shell-contract-ok'})
    registry.register(replace(defaults.get('background_run').definition, name='other_scheduler'),
                      defaults.get('background_run').handler)
    assert '可异步' not in str(registry.to_openai_tools())  # no arbitrary choice between two schedulers


async def test_renamed_result_container_reads_originals_but_plain_same_name_does_not(registry_store):
    store,_=registry_store
    registry=ToolRegistry()
    registry.register(ToolDefinition('work','',async_capable=True,pricing=Free('fixture')),lambda **args:'EXACT')
    registry.register(ToolDefinition('background_run','',pricing=Free('fixture')),lambda **args:'')
    runner,row=await start_background(store,registry,'work')
    try:
        ref=JobRef(**json.loads(row.content)['job_ref'])
        await runner.runtime.client.wait(ref,actor=runner.actor,deadline_ms=now_ms()+10000)
        sid=runner.target.session_id
        record=await load_record(store,sid,ref.job_id)
        engine=ToolContextEngine(config=ToolContextConfig(),store=store,owner_session_id=sid,
            owner_send_index=1,registry=registry,llm=None,model='unused')
        source=ToolSource(sid,1,row.seq,'background','defer_work',{},encode_results([record]))
        assert engine.manages('defer_work') and not engine.manages('background_run')
        assert (await engine._background_sources(source))[0].result=='EXACT'
        assert await engine._background_sources(replace(source,name='background_run'))==[]
    finally:
        await runner.aclose()


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


async def test_original_receipt_recall_survives_alias_and_registration_removal(registry_store):
    from power_loop.runtime.tool_archive import recall_original
    store,_=registry_store
    registry=ToolRegistry()
    registry.register(ToolDefinition('image','',async_capable=True,pricing=Free('fixture')),lambda **args:'FULL ORIGINAL')
    runner,row=await start_background(store,registry,'image',{'x':1})
    try:
        ref=JobRef(**json.loads(row.content)['job_ref'])
        await runner.runtime.client.wait(ref,actor=runner.actor,deadline_ms=now_ms()+10000)
        sid=runner.target.session_id
        record=await load_record(store,sid,ref.job_id)
        registry.unregister('image')
        await store.append_message(sid,send_index=1,role='assistant',tool_calls=[{
            'id':'old-call','type':'function','function':{'name':'retired_alias','arguments':'{}'}}])
        seq=await store.append_message(sid,send_index=1,role='tool',tool_call_id='old-call',content=encode_results([record]))
        row=await store.get_message(sid,seq)
        original=json.loads(await recall_original(store,sid,1,row))
        assert original['background_originals']==[record]
        assert (await store.get_message(sid,seq)).content==row.content
    finally:
        await runner.aclose()
