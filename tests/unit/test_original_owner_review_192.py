"""The original live child controller handles its abort before releasing output."""
import asyncio
import json
from collections import Counter
from types import SimpleNamespace

import pytest

from power_loop import (
    AgentLoopConfig,
    AgentSpec,
    LLMResponse,
    LLMService,
    LLMTokenUsage,
    StatefulAgentLoop,
    ToolDefinition,
)
from power_loop.core.agent_context import get_current_loop, get_session_id
from power_loop.runtime.budget_contracts import BudgetConfig, LocalBudgetScope
from power_loop.runtime.jobs import JobManager
from power_loop.runtime.managed_effects import FunctionManagedTool, ManagedToolContract
from power_loop.runtime.task_runtime import _OWNER, _SEND, TaskRuntime
from power_loop.runtime.tool_contracts import FunctionRuntimeTool, ToolDispatchPolicy, ToolRuntimeContract
from power_loop.tools.registry import ToolRegistry
from power_loop.workflow import create_workflow

from ._tool_plan_faults_192 import fail_plan_once
from .test_job_registry_192 import registry_store
from .test_model_budgets_192 import Host

__all__ = ['registry_store']
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def original_owner_case(registry_store, kind, *, allowed_calls=2, stage='seal', hide=False, provider=None, fail_review=False, shared_calls=12, max_tokens=2048, effect_calls=1, native_tail=None, native_first=False):
    store, _ = registry_store
    manager, host = JobManager(store), Host()
    if provider is not None:
        from tests.real.test_real_budget_tree_192 import RealBudgetHost
        host = RealBudgetHost()
    runtime = TaskRuntime(manager, tenant_id='original-owner', principal_id='host', executor_id='original', boot_id='boot',
        lease_ms=300000, boot_contract_hash='c'*64, budget_config=BudgetConfig('original-owner-v1', (
            LocalBudgetScope('shared', max_calls=shared_calls),
            LocalBudgetScope('child', max_calls=allowed_calls, binding='owner', owner_kinds=('workflow_node','subagent'))),host))
    state = SimpleNamespace(counts=Counter(), effects=[], entered=asyncio.Event(), release=asyncio.Event(),
        handles=[], notifications=[], source=None, root=None, review_requests=[], original=asyncio.Event())
    class Model(LLMService):
        async def complete(self, request, **kwargs):
            sid = get_session_id()
            role = 'MAIN' if sid == state.root else 'AFTER' if 'ROLE-AFTER' in request.system_prompt else 'BUILD'
            state.counts[role] += 1
            calls, text = None, 'DONE'
            if role == 'MAIN' and state.counts[role] == 1:
                text, calls = '', [{'id':'launch-original','type':'function','function':{'name':'launch_owned','arguments':'{}'}}]
            elif role == 'BUILD' and state.counts[role] == 1:
                state.source = _SEND.get().lease.job_id
                state.source_binding, state.owner_binding = _SEND.get(), _OWNER.get()
                state.original.set()
                if stage is not None:
                    fail_plan_once(manager,stage)
                text, calls = '', [{'id':'original-effect' if index == 0 else f'original-effect-{index+1}',
                    'type':'function','function':{'name':'touch_once','arguments':'{}'}} for index in range(effect_calls)]
                if native_tail is not None:
                    call = {'id':'original-native','type':'function','function':{'name':native_tail,'arguments':'{}'}}
                    calls.insert(0 if native_first else len(calls), call)
            elif role == 'BUILD':
                assert 'ORIGINAL OUTCOME REVIEW' in request.system_prompt
                assert not request.tools
                assert not any(m.get('tool_calls') or m.get('role')=='tool' for m in request.messages)
                state.review_requests.append(request)
                state.entered.set()
                await state.release.wait()
                if fail_review:
                    raise RuntimeError('test_original_review_provider_failed')
                text = 'REVIEWED: original action was not dispatched.'
            if provider is not None:
                return await provider.complete(request,**kwargs)
            return LLMResponse(raw_text=text,tool_calls=calls,token_usage=LLMTokenUsage(prompt_tokens=5,completion_tokens=5,total_tokens=10))
        def stream(self, request):
            raise AssertionError('complete selected')
        async def close(self):
            pass
    model, registry = Model(), ToolRegistry()
    host.service = model
    async def forbidden(*args):
        state.effects.append(args)
        raise AssertionError('original aborted effect cannot execute')
    definition = ToolDefinition('touch_once','Original finite effect',input_schema={'type':'object'})
    adapter = FunctionManagedTool(ManagedToolContract('example.original-owner','d'*64),forbidden,forbidden,forbidden)
    registry.register(definition,adapter.bind(runtime,definition))
    if native_tail is not None:
        state.native_calls = 0
        async def native():
            state.native_calls += 1
            return 'Original native completion'
        registry.register(ToolDefinition(native_tail,'Original native operation',input_schema={'type':'object'}),
            FunctionRuntimeTool(native,ToolRuntimeContract('example.original-native'),policy=lambda args:ToolDispatchPolicy(True)))
    build_tools = ['touch_once'] + ([native_tail] if native_tail is not None else [])
    async def notify(*args):
        state.notifications.append(args)
    async def launch():
        if kind == 'subagent':
            from power_loop.runtime.delegation import start
            result = await start(get_current_loop(),AgentSpec(name='build',system_prompt='ROLE-BUILD',tools=build_tools,max_rounds=3,lifecycle='linked'),
                'Perform the original task once.',on_complete=notify)
            state.handles.append(result)
            return json.dumps(result)
        workflow = create_workflow({'name':'original-owner','root':{'type':'sequence','steps':[
            {'type':'agent','id':'BUILD','input':'Perform original action once.','continue_on_error':True,
             'spec':{'name':'build','system_prompt':'ROLE-BUILD','tools':build_tools,'max_rounds':3}},
            {'type':'agent','id':'AFTER','input':'Only after BUILD is settled.','spec':{'name':'after','system_prompt':'ROLE-AFTER','tools':[],'max_rounds':2}}
        ]}},parent_loop=get_current_loop(),parent_session_id=get_session_id())
        handle = await workflow.start(detached=True,on_complete=notify)
        state.handles.append(handle)
        return json.dumps({'job_id':handle.job_id,'run_id':handle.run_id,'status':'accepted'})
    registry.register(ToolDefinition('launch_owned','Launch original child',input_schema={'type':'object'}),
        FunctionRuntimeTool(launch,ToolRuntimeContract('example.launch-owned',allowed_roles=('main',)),policy=lambda args:ToolDispatchPolicy(True)))
    loop = StatefulAgentLoop(llm=model,store=store,task_runtime=runtime,tool_registry=registry,
        config=AgentLoopConfig(max_tokens=max_tokens if provider is not None else 20,max_rounds=4,delegation_policy='single_level',compactor=None))
    from power_loop.runtime.retry import LLMRetryPolicy
    loop.config.retry_policy = LLMRetryPolicy(max_attempts=1)
    state.root = await loop.new_session()
    if provider is not None:
        from power_loop import HookPoint
        async def phase(ctx):
            sid = get_session_id()
            role = 'MAIN' if sid==state.root else 'AFTER' if 'ROLE-AFTER' in ctx.system_prompt else 'BUILD'
            name = 'launch_owned' if role=='MAIN' and not state.counts[role] else 'touch_once' if role=='BUILD' and not state.counts[role] else None
            ctx.tools = [registry.get(name).definition.to_openai_tool()] if name else []
            if role == 'BUILD':
                ctx.system_prompt += ('\nAcknowledge only the supplied negative outcome in one short sentence. The action was not dispatched. Do not solve or retry the task.'
                    if state.counts[role] else '\nCall the offered touch_once function once using native function calling. Do not write textual or XML calls.')
                if native_tail is not None and not state.counts[role]:
                    ctx.tools.append(registry.get(native_tail).definition.to_openai_tool())
                    order = f'{native_tail}, then touch_once' if native_first else f'touch_once, then {native_tail}'
                    ctx.system_prompt += (f' In this same response return two native function calls in order: {order}. '
                        'Both use empty arguments. Do not wait for the first result before requesting the second call.')
            elif role == 'AFTER':
                ctx.system_prompt += '\nThe failed previous node was deliberately tolerated. Return AFTER as plain text; do no other work.'
            if role=='MAIN':
                ctx.system_prompt += ('\nRead and acknowledge the supplied original completion without starting more work.' if state.counts[role]>=2 else '\nLaunch the requested child once with launch_owned, then return QUEUED as plain text. Do not launch additional tasks.')
        loop.hooks.register(HookPoint.LLM_BEFORE,phase,name='test.owner.actual-phase-schema')
    if hide:
        from power_loop import HookPoint
        async def hide_notice(ctx):
            ctx.messages = [m for m in ctx.messages if 'Original tool outcome:' not in str(m.get('content'))]
        loop.hooks.register(HookPoint.LLM_BEFORE,hide_notice,name='test.owner.hide-notice')
    return SimpleNamespace(store=store,manager=manager,runtime=runtime,loop=loop,model=model,host=host,state=state,provider=provider)


@pytest.mark.parametrize('kind',['workflow_node','subagent'])
async def test_live_original_owner_reviews_aborted_batch_before_delivering_output_or_advancing(registry_store,kind):
    from power_loop.runtime import delegation
    c = await original_owner_case(registry_store,kind)
    initial = set(delegation._TASKS)
    try:
        result = await c.loop.send('Launch the original child task.',c.state.root)
        assert result.status == 'completed'
        await asyncio.wait_for(c.state.original.wait(),3)
        await asyncio.wait_for(c.state.entered.wait(),3)
        assert c.state.counts['AFTER'] == 0 and not c.state.notifications and not c.state.effects
        original = await c.manager.get_job(c.runtime.actor(c.state.root),c.state.source)
        assert original.execution_role == kind and original.status == 'model_closed' and original.result_status == 'pending_delivery'
        owner = await c.manager.get_job(c.runtime.actor(c.state.root),original.control_owner_job_id)
        assert owner.status == 'running' and owner.result_status == 'none'
        c.state.release.set()
        tasks = [c.state.handles[0].task] if kind=='workflow_node' else list(set(delegation._TASKS)-initial)
        await asyncio.wait_for(asyncio.gather(*tasks),10)
        assert c.state.counts['BUILD'] == 2 and c.state.counts['AFTER'] == (1 if kind=='workflow_node' else 0)
        original = await c.manager.get_job(c.runtime.actor(c.state.root),c.state.source)
        assert original.result_status == 'handled' and original.supervision_status == 'closed'
        assert (await c.store.get_state(original.session_id)).pending is None
        assert not await c.manager.db.fetchall(f"SELECT id FROM {c.store.t.messages} WHERE session_id=? AND role='tool'",(original.session_id,))
        assert not c.state.effects
        owner = await c.manager.get_job(c.runtime.actor(c.state.root),original.control_owner_job_id)
        assert owner.outcome == 'failed' and owner.result['usage']['total_tokens'] == 20
        assert c.state.notifications
    finally:
        c.state.release.set()
        await c.runtime.shutdown(timeout_s=2)
        await c.loop.aclose(drain_timeout_s=.1)


async def wait_owned(c,kind,initial):
    from power_loop.runtime import delegation
    tasks = [c.state.handles[0].task] if kind=='workflow_node' else list(set(delegation._TASKS)-initial)
    # This checks recovery/credit invariants, not a 12-second latency SLA.
    # Independent full PG shards can take >12s for the same indexed transaction
    # sequence; the execution's actual lease remains 300s and is not extended.
    await asyncio.wait_for(asyncio.gather(*tasks,return_exceptions=True),150 if c.provider is not None else 30)


@pytest.mark.parametrize('kind',['workflow_node','subagent'])
@pytest.mark.parametrize('failure',['quota','hidden','provider','cancel'])
async def test_unprocessed_original_owner_outcome_never_releases_its_result_or_downstream(registry_store,kind,failure,monkeypatch):
    from power_loop.runtime import abort_outcomes, delegation
    c = await original_owner_case(registry_store,kind,allowed_calls=1 if failure=='quota' else 2,hide=failure=='hidden',fail_review=failure=='provider')
    c.state.release.set()
    initial = set(delegation._TASKS)
    if failure=='cancel':
        reconcile = abort_outcomes._reconcile
        async def cancel_after_publication(loop,actor,source,**kwargs):
            value = await reconcile(loop,actor,source,**kwargs)
            original = await c.manager.get_job(actor,source)
            await c.manager.request_cancel(c.runtime.actor(c.state.root),original.control_owner_job_id,
                system_key='stop-original-owner',reason='Stop before review model admission')
            return value
        monkeypatch.setattr(abort_outcomes,'_reconcile',cancel_after_publication)
    try:
        await c.loop.send('Launch original child.',c.state.root)
        await wait_owned(c,kind,initial)
        original = await c.manager.get_job(c.runtime.actor(c.state.root),c.state.source)
        assert original.result_status == 'pending_delivery' and original.supervision_status == 'open'
        assert c.state.counts['AFTER'] == 0 and not c.state.effects
        assert c.state.counts['BUILD'] == (1 if failure in ('quota','cancel') else 2)
        owner = await c.manager.get_job(c.runtime.actor(c.state.root),original.control_owner_job_id)
        assert owner.result_status == 'none' and owner.supervision_status == 'open'
        if kind=='subagent':
            assert not c.state.notifications
    finally:
        c.state.release.set()
        await c.runtime.shutdown(timeout_s=2)
        await c.loop.aclose(drain_timeout_s=.1)


@pytest.mark.parametrize('kind',['workflow_node','subagent'])
@pytest.mark.parametrize('stage',['child','result','notice','inbox','disposition','pending','commit','commit_ack'])
async def test_original_controller_confirms_one_atomic_recovery_without_more_model_credit(registry_store,kind,stage,monkeypatch):
    from power_loop.runtime import abort_outcomes, delegation

    from ._abort_outcome_faults_192 import fail_outcome_once
    c = await original_owner_case(registry_store,kind,stage='commit_ack')
    c.state.release.set()
    initial = set(delegation._TASKS)
    reconcile = abort_outcomes._reconcile
    faults=[]
    async def fault_once(loop,actor,source,**kwargs):
        if not faults:
            faults.append(fail_outcome_once(c.manager,source,stage))
        return await reconcile(loop,actor,source,**kwargs)
    monkeypatch.setattr(abort_outcomes,'_reconcile',fault_once)
    try:
        await c.loop.send('Launch original child.',c.state.root)
        await wait_owned(c,kind,initial)
        assert faults and faults[0].fired and c.state.counts['BUILD'] == 2 and not c.state.effects
        original = await c.manager.get_job(c.runtime.actor(c.state.root),c.state.source)
        assert original.result_status == 'handled' and original.supervision_status == 'closed'
        notices = await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE event_type='source_abort_delivery'")
        assert len(notices)==1
        children = await c.manager.db.fetchall(f"SELECT generation,result_status,supervision_status FROM {c.manager.jobs} WHERE kind='background_tool'")
        assert len(children)==1 and children[0]['generation']==0 and children[0]['result_status']=='suppressed' and children[0]['supervision_status']=='closed'
    finally:
        c.state.release.set()
        await c.runtime.shutdown(timeout_s=2)
        await c.loop.aclose(drain_timeout_s=.1)


@pytest.mark.parametrize('kind',['workflow_node','subagent'])
async def test_copied_owner_same_fields_and_inherited_context_in_a_new_task_are_not_execution_authority(registry_store,kind,monkeypatch):
    from dataclasses import replace

    from power_loop.runtime import abort_outcomes, delegation
    from power_loop.runtime.jobs import JobError
    from power_loop.runtime.original_owner_review import _Failure, require_live
    from power_loop.runtime.task_runtime import Execution
    c = await original_owner_case(registry_store,kind)
    c.state.release.set()
    initial = set(delegation._TASKS)
    reconcile = abort_outcomes._reconcile
    denied=[]
    async def inspect(loop,actor,source,*,permit):
        if permit is None:
            return await reconcile(loop,actor,source,permit=None)
        async with c.manager._write() as tx:
            row = await c.manager._authorize(tx,actor,source)
            clone = replace(permit,owner=Execution(permit.owner.runtime,permit.owner.actor,permit.owner.lease))
            assert type(clone) is _Failure
            with pytest.raises(JobError,match='original_owner_controller_required'):
                await require_live(c.manager,tx,actor,row,clone)
            async def different_task():
                with pytest.raises(JobError,match='original_owner_controller_required'):
                    await require_live(c.manager,tx,actor,row,permit)
            await asyncio.create_task(different_task())
            denied.extend(['copied_owner','different_task'])
        with pytest.raises(JobError,match='managed_completion_requires_host'):
            await abort_outcomes.reconcile(loop,actor,source)
        return await reconcile(loop,actor,source,permit=permit)
    monkeypatch.setattr(abort_outcomes,'_reconcile',inspect)
    try:
        await c.loop.send('Launch original child.',c.state.root)
        await wait_owned(c,kind,initial)
        assert denied==['copied_owner','different_task'] and c.state.counts['BUILD']==2 and not c.state.effects
        assert (await c.manager.get_job(c.runtime.actor(c.state.root),c.state.source)).result_status=='handled'
    finally:
        c.state.release.set()
        await c.runtime.shutdown(timeout_s=2)
        await c.loop.aclose(drain_timeout_s=.1)


@pytest.mark.parametrize('kind',['workflow_node','subagent'])
@pytest.mark.parametrize('part',['admission','creation_flag','both_owner_markers','scope_boolean','physical_role','member'])
async def test_owner_evidence_damage_cannot_clear_pending_or_promote_the_child(registry_store,kind,part,monkeypatch):
    from power_loop.runtime import abort_outcomes, delegation
    from power_loop.runtime.execution_membership import member_id, scope_id
    from power_loop.runtime.jobs import _json
    from power_loop.runtime.original_owner_review import admission_id
    c = await original_owner_case(registry_store,kind)
    c.state.release.set()
    initial = set(delegation._TASKS)
    reconcile = abort_outcomes._reconcile
    async def damaged(loop,actor,source,**kwargs):
        row = await c.manager._row(c.manager.db,source)
        owner = await c.manager._row(c.manager.db,row['control_owner_job_id'])
        if part in ('admission','both_owner_markers'):
            await c.manager.db.execute(f'DELETE FROM {c.manager.events} WHERE event_id=?',(admission_id(c.manager,owner),))
        if part in ('creation_flag','both_owner_markers'):
            event = await c.manager.db.fetchone(f"SELECT event_id,payload_json FROM {c.manager.events} WHERE job_id=? AND action='create'",(owner['job_id'],))
            data = json.loads(event['payload_json'])
            data['request'].pop('owner_abort_review_protocol')
            await c.manager.db.execute(f'UPDATE {c.manager.events} SET payload_json=? WHERE event_id=?',(_json(data),event['event_id']))
        elif part=='physical_role':
            await c.store.merge_session_metadata(row['session_id'],{'execution_role':'main'})
        elif part=='member':
            await c.manager.db.execute(f'DELETE FROM {c.manager.events} WHERE event_id=?',(member_id(c.manager,source,row['generation']),))
        elif part=='scope_boolean':
            key = scope_id(c.manager,source,row['generation'])
            event = await c.manager.db.fetchone(f'SELECT payload_json FROM {c.manager.events} WHERE event_id=?',(key,))
            data = json.loads(event['payload_json'])
            data['owner']['generation']=True
            await c.manager.db.execute(f'UPDATE {c.manager.events} SET payload_json=? WHERE event_id=?',(_json(data),key))
        return await reconcile(loop,actor,source,**kwargs)
    monkeypatch.setattr(abort_outcomes,'_reconcile',damaged)
    try:
        await c.loop.send('Launch original child.',c.state.root)
        await wait_owned(c,kind,initial)
        source = await c.manager.get_job(c.runtime.actor(c.state.root),c.state.source)
        owner = await c.manager.get_job(c.runtime.actor(c.state.root),source.control_owner_job_id)
        assert source.result_status=='none' and source.supervision_status=='open'
        assert (await c.store.get_state(source.session_id)).pending is not None
        assert owner.result_status=='none' and c.state.counts['AFTER']==0 and not c.state.effects
        assert c.state.counts['BUILD']==1
    finally:
        c.state.release.set()
        await c.runtime.shutdown(timeout_s=2)
        await c.loop.aclose(drain_timeout_s=.1)


@pytest.mark.parametrize('kind',['workflow_node','subagent'])
async def test_closed_original_source_keeps_its_child_owner_proof_in_queries_and_finalization(registry_store,kind):
    from power_loop.runtime import delegation
    from power_loop.runtime.jobs import JobError
    from power_loop.runtime.original_owner_review import admission_id
    from power_loop.runtime.task_query import get_job
    c = await original_owner_case(registry_store,kind)
    c.state.release.set()
    initial = set(delegation._TASKS)
    try:
        await c.loop.send('Launch original child.',c.state.root)
        await wait_owned(c,kind,initial)
        actor = c.runtime.actor(c.state.root)
        source = await c.manager.get_job(actor,c.state.source)
        assert source.result_status=='handled' and source.supervision_status=='closed'
        owner = await c.manager._row(c.manager.db,source.control_owner_job_id)
        await c.manager.db.execute(f'DELETE FROM {c.manager.events} WHERE event_id=?',(admission_id(c.manager,owner),))
        view = await get_job(c.manager,actor,source.job_id)
        assert view['node']['processing_obligation']['state']=='unknown'
        with pytest.raises(JobError):
            await c.manager.finalize_supervision(actor,source.job_id,generation=1)
        assert (await c.manager.get_job(actor,source.job_id)).result_status=='handled' and c.state.counts['BUILD']==2
    finally:
        c.state.release.set()
        await c.runtime.shutdown(timeout_s=2)
        await c.loop.aclose(drain_timeout_s=.1)


@pytest.mark.parametrize('kind',['workflow_node','subagent'])
async def test_actual_per_call_round_limit_cannot_be_replaced_by_the_larger_child_config(registry_store,kind,monkeypatch):
    from power_loop.runtime import delegation
    c = await original_owner_case(registry_store,kind)
    c.state.release.set()
    initial = set(delegation._TASKS)
    send = StatefulAgentLoop.send
    async def one_round(self,*args,**kwargs):
        sid = kwargs.get('session_id')
        session = await (await self.ensure_store()).get_session(sid) if sid else None
        if session is not None and session.metadata.get('execution_role')==kind:
            assert self.config.max_rounds==3
            kwargs['max_rounds']=1
        return await send(self,*args,**kwargs)
    monkeypatch.setattr(StatefulAgentLoop,'send',one_round)
    try:
        await c.loop.send('Launch original child.',c.state.root)
        await wait_owned(c,kind,initial)
        source = await c.manager.get_job(c.runtime.actor(c.state.root),c.state.source)
        assert source.result['aborted_call_batch']['review_owner']['round_limit']==1
        assert source.result_status=='pending_delivery' and c.state.counts['BUILD']==1
        assert c.state.counts['AFTER']==0 and not c.state.effects
        owner = await c.manager.get_job(c.runtime.actor(c.state.root),source.control_owner_job_id)
        assert owner.result_status=='none'
    finally:
        c.state.release.set()
        await c.runtime.shutdown(timeout_s=2)
        await c.loop.aclose(drain_timeout_s=.1)


async def test_original_completion_window_with_no_remaining_supplements_does_not_get_a_review_round(registry_store,monkeypatch):
    from power_loop.runtime import abort_outcomes, delegation
    from power_loop.runtime.completion import WINDOW
    c = await original_owner_case(registry_store,'workflow_node')
    c.state.release.set()
    initial = set(delegation._TASKS)
    reconcile = abort_outcomes._reconcile
    async def exhausted(loop,actor,source,**kwargs):
        window = WINDOW.get()
        assert window is not None
        window.remaining=0
        return await reconcile(loop,actor,source,**kwargs)
    monkeypatch.setattr(abort_outcomes,'_reconcile',exhausted)
    try:
        await c.loop.send('Launch original child.',c.state.root)
        await wait_owned(c,'workflow_node',initial)
        source = await c.manager.get_job(c.runtime.actor(c.state.root),c.state.source)
        assert source.result_status=='pending_delivery' and c.state.counts['BUILD']==1
        assert c.state.counts['AFTER']==0 and not c.state.effects
    finally:
        c.state.release.set()
        await c.runtime.shutdown(timeout_s=2)
        await c.loop.aclose(drain_timeout_s=.1)


@pytest.mark.parametrize('kind',['workflow_node','subagent'])
async def test_host_can_publish_original_child_negative_but_cannot_start_a_replacement_controller(registry_store,kind,monkeypatch):
    from power_loop.runtime import delegation, original_owner_review
    from power_loop.runtime.jobs import JobError
    from power_loop.runtime.tool_call_plans import reconcile_pending
    c = await original_owner_case(registry_store,kind)
    initial = set(delegation._TASKS)
    async def unavailable(*args,**kwargs):
        return None
    monkeypatch.setattr(original_owner_review,'try_handle',unavailable)
    try:
        await c.loop.send('Launch original child.',c.state.root)
        await wait_owned(c,kind,initial)
        source = await c.manager.get_job(c.runtime.actor(c.state.root),c.state.source)
        actor = c.runtime.actor(source.session_id)
        value = await reconcile_pending(c.loop,actor,source.job_id)
        assert value['notices'][0]['review_session_id']==source.session_id
        assert 'main_session_id' not in value['notices'][0] and not value['execution_authority'] and not value['acknowledgement_created']
        assert c.state.counts['BUILD']==1 and not c.state.effects
        with pytest.raises(JobError,match='task_owner_required'):
            await c.loop.send('Read original negative only.',source.session_id)
        assert c.state.counts['BUILD']==1 and c.state.counts['AFTER']==0
        assert (await c.manager.get_job(actor,source.job_id)).result_status=='pending_delivery'
    finally:
        c.state.release.set()
        await c.runtime.shutdown(timeout_s=2)
        await c.loop.aclose(drain_timeout_s=.1)
