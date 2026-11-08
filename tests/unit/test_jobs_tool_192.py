"""Jobs tools use the current runtime and retain permissions, cancellation and error privacy."""
import asyncio
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from power_loop import (
    JOBS_DEFINITION,
    AgentLoopConfig,
    JobRef,
    LLMResponse,
    create_default_tool_registry,
    register_jobs_tools,
)
from power_loop.agent.runtime_store import now_ms
from power_loop.core.agent_context import (
    get_session_id,
    reset_current_loop,
    reset_session_id,
    set_current_loop,
    set_session_id,
)
from power_loop.runtime.jobs import JobManager
from power_loop.runtime.task_runtime import TaskRuntime
from power_loop.tools.default_manifest import CORE_TOOL_NAMES, EXPLORE_TOOL_NAMES, FULL_TOOL_NAMES
from power_loop.tools.jobs import run_jobs
from power_loop.tools.registry import ToolRegistry
from power_loop.tools.spawn_agent import register_spawn_agent

from .test_job_registry_192 import registry_store
from .test_task_runtime_192 import AdapterModel

__all__ = ['registry_store']
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.fixture
async def case(registry_store, runtime_agent):
    store, _ = registry_store
    manager = JobManager(store)
    runtime = TaskRuntime(manager, tenant_id='jobs', principal_id='host', executor_id='worker', boot_id='boot')
    registry = create_default_tool_registry(include=['jobs'])
    register_spawn_agent(registry)
    model = AdapterModel()
    agent = await runtime_agent(model=model, store=store, registry=registry, task_runtime=runtime,
        config=AgentLoopConfig(max_rounds=5, compactor=None), capacity=2)
    model.root = agent.target.session_id
    try:
        yield agent, model, registry, runtime
    finally:
        model.release.set()
        await agent.aclose()


async def invoke(runtime, sid, **kwargs):
    # Read tools require the trusted host TaskRuntime; writes additionally
    # require an actual persisted model tool call and current execution.
    a, b = set_current_loop(SimpleNamespace(task_runtime=runtime)), set_session_id(sid)
    try:
        return json.loads(await run_jobs(**kwargs))
    finally:
        reset_session_id(b)
        reset_current_loop(a)


async def spawned(case):
    agent, model, _, runtime = case
    await agent.send('delegate')
    await asyncio.wait_for(model.entered.wait(), 5)
    rows = await agent.runtime.store.load_all_messages(agent.target.session_id)
    return next(json.loads(m.content) for m in rows if m.role=='tool' and m.name=='spawn_agent')


async def test_explicit_builtin_registration_does_not_expand_default_presets():
    registry = create_default_tool_registry(include=["jobs"])
    assert registry.get("jobs").definition == JOBS_DEFINITION
    assert all("jobs" not in names for names in (CORE_TOOL_NAMES, EXPLORE_TOOL_NAMES, FULL_TOOL_NAMES))
    second = ToolRegistry()
    register_jobs_tools(second, definition=replace(JOBS_DEFINITION, description="Localized description"))
    assert second.get("jobs").definition.input_schema == JOBS_DEFINITION.input_schema
    with pytest.raises(ValueError):
        register_jobs_tools(second, definition=replace(JOBS_DEFINITION, input_schema={}))
    assert JOBS_DEFINITION.interrupt == "finish"


async def test_jobs_list_get_stop_controls_original_child_and_binds_tool_receipt(case, monkeypatch):
    agent, model, _, runtime = case
    accepted = await spawned(case)
    sid = agent.target.session_id
    actions = [dict(action='list', kinds=['subagent']), dict(action='get', job_id=accepted['job_id']),
        dict(action='stop', job_id=accepted['job_id'], expected_generation=1, reason='user stop')]
    original = model.complete
    phase = 0
    async def complete(request, **kwargs):
        nonlocal phase
        if get_session_id()==sid and phase<len(actions):
            params=actions[phase]
            phase+=1
            return LLMResponse(raw_text='', tool_calls=[{'id':'jobs-'+str(phase),'type':'function','function':{
                'name':'jobs','arguments':json.dumps(params)}}])
        return await original(request, **kwargs)
    monkeypatch.setattr(model, 'complete', complete)
    assert (await agent.send('inspect and stop')).status=='completed'
    result=await agent.runtime.client.wait(JobRef(**accepted['job_ref']), actor=agent.actor, deadline_ms=now_ms()+10000)
    assert result.status=='cancelled' and model.child_calls==1
    await agent.drain()
    rows=[m for m in await agent.runtime.store.load_all_messages(sid) if m.role=='tool' and m.name=='jobs']
    listed, detail, stopped=[json.loads(m.content) for m in rows]
    assert listed['matched_count']==1 and detail['node']['job_id']==accepted['job_id']
    assert stopped['status']=='accepted' and stopped['job_id']==accepted['job_id']
    assert stopped['receipt']['job_ref']['job_id']==accepted['job_id']
    assert rows[-1].tool_call_id=='jobs-3'
    assert (await runtime.manager.get_job(runtime.actor(sid), accepted['job_id'])).cancel_requested_at is not None


async def test_read_aliases_require_permission_and_stop_requires_model_invocation(case):
    agent, model, _, runtime=case
    accepted=await spawned(case)
    sid=agent.target.session_id
    for alias in (accepted['job_id'], 'sub:'+accepted['run_id'], accepted['run_id'], accepted['operation_id']):
        assert (await invoke(runtime, sid, action='get', job_id=alias))['node']['job_id']==accepted['job_id']
    assert (await invoke(runtime, sid, action='stop', job_id=accepted['job_id']))['error']['code']=='task_send_required'
    foreign=await agent.runtime.open_session(agent.descriptor, actor=agent.actor)
    assert (await invoke(runtime, foreign.session_id, action='get', job_id=accepted['job_id']))['error']['code']=='access_denied'
    assert (await runtime.manager.get_job(runtime.actor(sid), accepted['job_id'])).cancel_requested_at is None
    model.release.set()


@pytest.mark.parametrize(
    "params",
    [
        {"action": "get"},
        {"action": "list", "max_chars": True},
        {"action": "list", "max_chars": 32001},
        {"action": "get", "job_id": "missing"},
        {"action": "list", "ids": ["job_x"]},
        {"action": "stop", "ids": ["one", "two"]},
        {"action": "stop", "ids": []},
        {"action": "get", "job_id": "job_x", "root_job_id": "job_y"},
    ],
)
async def test_invalid_parameters_are_structured_and_never_write_control(case, params):
    agent, _, _, runtime=case
    assert 'error' in await invoke(runtime, agent.target.session_id, **params)
    assert not await runtime.manager.db.fetchall(f"SELECT event_id FROM {runtime.manager.events} WHERE action='stop'")


async def test_jobs_outside_session_is_a_structured_error():
    assert json.loads(await run_jobs(action='list'))['error']['code']=='active_session_required'


@pytest.mark.parametrize('action', ['list','get','stop'])
@pytest.mark.parametrize('exception', [RuntimeError,ValueError])
async def test_backend_failure_does_not_expose_private_message_or_forged_code(case, monkeypatch, action, exception):
    from power_loop.runtime import task_query
    agent, _, _, runtime=case
    accepted=await spawned(case)
    async def unavailable(*args, **kwargs):
        err=exception('PRIVATE-CONNECTION-SECRET')
        err.code='PRIVATE-BACKEND-CODE'
        raise err
    if action=='list':
        monkeypatch.setattr(task_query,'list_jobs',unavailable)
    else:
        monkeypatch.setattr(runtime.manager,'get_job',unavailable)
    result=await invoke(runtime,agent.target.session_id,action=action,**({'job_id':accepted['job_id']} if action!='list' else {}))
    assert result['error']['code']==('stop_requires_reconciliation' if action=='stop' else 'jobs_backend_unavailable')
    assert 'PRIVATE' not in json.dumps(result)


@pytest.mark.parametrize('name',['jobs','inspect_tasks','send_message'])
async def test_registered_jobs_alias_keeps_authorized_contract(case, monkeypatch, name):
    from power_loop import ToolDefinition
    from power_loop.runtime.pricing import Free
    from power_loop.runtime.tool_contracts import policy
    agent,model,registry,_=case
    register_jobs_tools(registry,definition=replace(JOBS_DEFINITION,name=name),overwrite=True)
    calls=[]
    async def complete(request,**kwargs):
        calls.append(request)
        return LLMResponse(raw_text='inspected' if len(calls)>1 else '',tool_calls=None if len(calls)>1 else [
            {'id':'alias-list','type':'function','function':{'name':name,'arguments':'{"action":"list"}'}}])
    monkeypatch.setattr(model,'complete',complete)
    assert (await agent.send('inspect')).status=='completed' and len(calls)==2
    [reply]=[m for m in await agent.runtime.store.load_all_messages(agent.target.session_id) if m.role=='tool']
    assert reply.tool_call_id=='alias-list' and not json.loads(reply.content).get('error')
    assert policy(registry.get(name),{'action':'list'}).control_only
    forged=ToolRegistry()
    forged.register(ToolDefinition(name,'ordinary callable',pricing=Free('fixture')),lambda **kwargs:'ordinary')
    assert not policy(forged.get(name),{'action':'list'}).native_admitted
