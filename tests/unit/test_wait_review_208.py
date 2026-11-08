"""Original finite child waits release capacity and use explicit main decisions."""
import json

import pytest

from power_loop import AgentLoopConfig, JobRef, LLMResponse, LLMService, ToolRegistry
from power_loop.agent.runtime_store import now_ms
from power_loop.runtime.wait_review import WaitPolicy
from power_loop.tools.spawn_agent import register_spawn_agent
from power_loop.tools.wait_handoff import register_wait_handoff_tool
from tests.runtime_fixture import RunningAgent

from .test_job_registry_192 import registry_store

__all__=['registry_store']
pytestmark=[pytest.mark.unit,pytest.mark.asyncio]


def call(name,args,identity):
    return LLMResponse(raw_text='',tool_calls=[{'id':identity,'type':'function',
        'function':{'name':name,'arguments':json.dumps(args)}}])


@pytest.mark.parametrize('choice',['continue','cancel','defer','expire','reopen'])
async def test_finite_wait_preserves_original_child_and_capacity_one(registry_store,choice, *, measure=None):
    store,_=registry_store
    class Model(LLMService):
        parent_calls=0
        child_calls=0
        reviewed=set()
        decisions=[]
        async def complete(self,request,**kwargs):
            if request.system_prompt.startswith('CHILD'):
                self.child_calls+=1
                if self.child_calls==1:
                    return call('wait_handoff',{'action':'request','due_at_ms':now_ms()+500,
                        'note':'Review original computation','continuation':'Continue the original computation: 42'},'child-wait')
                assert 'Continue the original computation: 42' in str(request.messages)
                return LLMResponse(raw_text='42')
            self.parent_calls+=1
            if self.parent_calls==1:
                return call('spawn_agent',{'task':'compute','system_prompt':'CHILD','tools':['wait_handoff'],
                    'max_rounds':4},'spawn')
            for message in reversed(request.messages):
                body=str(message.get('content',''))
                if 'Task data:\n' not in body:
                    continue
                data=json.JSONDecoder().raw_decode(body.split('Task data:\n',1)[1])[0]
                if data['event_id'] in self.reviewed or choice=='expire':
                    continue
                self.reviewed.add(data['event_id'])
                decision='defer' if choice=='defer' and not self.decisions else 'cancel' if choice=='cancel' else 'continue'
                self.decisions.append(decision)
                args={k:data[k] for k in ('job_id','generation','event_id')}
                args.update(action='decide',decision=decision)
                if decision=='defer':
                    args['next_review_at_ms']=now_ms()+500
                return call('wait_handoff',args,'decide-'+str(len(self.decisions)))
            return LLMResponse(raw_text='Acknowledged')
        async def close(self):pass
    model,registry=Model(),ToolRegistry()
    register_spawn_agent(registry)
    register_wait_handoff_tool(registry,policy=WaitPolicy(now_ms()+(1500 if choice=='expire' else 15000)))
    agent=await RunningAgent(store,model,registry,AgentLoopConfig(compactor=None,max_rounds=4),capacity=1).start()
    target=agent.target
    from contextlib import nullcontext
    async with measure(store) if measure else nullcontext():
        try:
            first=await agent.send('Start original child')
            assert first.status=='completed'
            await agent.drain()
            [child]=await store._db.fetchall(f"SELECT * FROM {store.table_prefix}jobs WHERE kind='subagent'")
            assert child['status']=='waiting' and child['outcome'] is None, [(m.role,m.content,m.tool_calls) for m in await store.load_all_messages(child['session_id'])]
            assert model.child_calls==1
            assert not agent.runtime.worker._tasks
            assert len(agent.runtime.worker._wait_deadlines)==1
            ref=JobRef(store.store_id,child['job_id'],child['generation'])
            if choice=='reopen':
                await agent.aclose()
                agent=await RunningAgent(store,model,registry,AgentLoopConfig(compactor=None,max_rounds=4),capacity=1).start(target=target)
                assert len(agent.runtime.worker._wait_deadlines)==1
            result=await agent.runtime.client.wait(ref,actor=agent.actor,deadline_ms=now_ms()+15000)
            assert result.status==('cancelled' if choice in {'cancel','expire'} else 'completed')
            await agent.drain()
            assert model.child_calls==(1 if choice in {'cancel','expire'} else 2)
            assert (await store._db.fetchone(f"SELECT COUNT(*) AS n FROM {store.table_prefix}jobs WHERE kind='subagent'"))['n']==1
            assert result.job_ref==ref
            if choice=='defer':
                assert model.decisions==['defer','continue']
            assert not agent.runtime.worker._wait_deadlines
        finally:
            await agent.aclose()



@pytest.fixture
async def paused_review(registry_store,request):
    from dataclasses import replace
    from types import SimpleNamespace

    from power_loop.runtime.wait_review import event_id
    store,_=registry_store
    continuation='Original detail: '+('计算42。'*2500)
    duration=getattr(request,"param",120000)
    deadline=now_ms()+duration
    class Model(LLMService):
        child_calls=0
        parent_calls=0
        async def complete(self,request,**kwargs):
            if request.system_prompt.startswith('CHILD'):
                self.child_calls+=1
                if self.child_calls==1:
                    return call('wait_handoff',dict(action='request',due_at_ms=min(now_ms()+60000,deadline-100),
                        note='Original review',continuation=continuation),'request')
                assert continuation in str(request.messages)
                return LLMResponse(raw_text='42')
            self.parent_calls+=1
            if self.parent_calls==1:
                return call('spawn_agent',dict(task='compute',system_prompt='CHILD',tools=['wait_handoff'],max_rounds=4),'spawn')
            return LLMResponse(raw_text='Parent finished')
        async def close(self):pass
    model,registry=Model(),ToolRegistry()
    register_spawn_agent(registry)
    register_wait_handoff_tool(registry,policy=WaitPolicy(deadline))
    agent=await RunningAgent(store,model,registry,AgentLoopConfig(compactor=None),capacity=1).start()
    try:
        await agent.send('start')
        await agent.drain()
        [job]=await store._db.fetchall(f"SELECT * FROM {store.table_prefix}jobs WHERE kind='subagent'")
        state=await store.get_state(job['session_id'])
        item=state.pending['pending_interactions'][0]
        messages=await store.load_all_messages(agent.target.session_id)
        source=next(m for m in messages if m.tool_calls)
        call_ref=await store.message_ref(agent.target.session_id,source.seq)
        yield SimpleNamespace(agent=agent,store=store,model=model,job=job,
            ref=JobRef(store.store_id,job['job_id'],job['generation']),pending=state.pending,
            item=item,event_id=event_id(item),continuation=continuation,deadline=deadline,
            actor=replace(agent.actor,session_id=agent.target.session_id,call_ref=call_ref,tool_call_id="spawn"))
    finally:
        await agent.aclose()


async def test_wait_original_authority_and_full_text_paging(paused_review):
    from dataclasses import replace

    from power_loop import RuntimeRequestError
    from power_loop.runtime.wait_review import decide, inspect
    c=paused_review
    for actor in [replace(c.actor,tenant_id='other'),replace(c.actor,principal_id='other'),
                  replace(c.actor,session_id=c.job['session_id'])]:
        with pytest.raises(RuntimeRequestError):
            await decide(c.agent.runtime.worker,c.ref,actor,c.event_id,decision='continue')
    with pytest.raises(RuntimeRequestError,match='stale_wait_review'):
        await decide(c.agent.runtime.worker,c.ref,c.actor,c.event_id+'wrong',decision='continue')
    with pytest.raises(RuntimeRequestError):
        await c.agent.runtime.client.answer(c.ref,actor=c.actor,
            interaction_id=c.item['interaction_id'],wait_round=c.pending['round_index'],
            value='bypass',message_id='bypass')
    summary=await inspect(c.agent.runtime.client,c.ref,c.actor,c.event_id)
    assert summary['continuation_length']==len(c.continuation) and len(summary['continuation'])==1000
    [timer]=await c.store.list_timers(c.agent.target.session_id)
    assert len(timer.note)<8000 and c.continuation not in timer.note
    chunks=[]
    offset=0
    while True:
        page=await inspect(c.agent.runtime.client,c.ref,c.actor,c.event_id,field='continuation',char_offset=offset,max_chars=4000)
        chunks.append(page['text'])
        offset=page['next_offset']
        if offset is None:
            break
    assert ''.join(chunks)==c.continuation
    for options in [{'char_offset':-1},{'max_chars':0},{'max_chars':True},{'max_chars':16001},{'field':'private'}]:
        with pytest.raises(RuntimeRequestError,match='invalid_options'):
            await inspect(c.agent.runtime.client,c.ref,c.actor,c.event_id,**{'field':'continuation',**options})
    assert c.model.child_calls==1


async def test_competing_main_decisions_select_one_original_action(paused_review):
    import asyncio

    from power_loop import RuntimeRequestError
    from power_loop.runtime.wait_review import decide
    c=paused_review
    results=await asyncio.gather(*(decide(c.agent.runtime.worker,c.ref,c.actor,c.event_id,decision=x)
        for x in ['continue','cancel']),return_exceptions=True)
    assert sum(isinstance(r,dict) for r in results)==1,results
    [rejected]=[r for r in results if isinstance(r,Exception)]
    assert isinstance(rejected,RuntimeRequestError)
    result=await c.agent.runtime.client.wait(c.ref,actor=c.actor,deadline_ms=now_ms()+10000)
    [accepted]=[r for r in results if isinstance(r,dict)]
    assert result.status==('completed' if accepted['decision']=='continue' else 'cancelled')
    await c.agent.drain()
    assert c.model.child_calls==(2 if accepted['decision']=='continue' else 1)


async def test_repeated_defer_retains_deadline_and_rejects_old_notice(paused_review):
    from power_loop import RuntimeRequestError
    from power_loop.runtime.wait_review import decide, inspect
    c=paused_review
    key=c.event_id
    for i in range(2):
        next_time=now_ms()+70000+i*1000
        response=await decide(c.agent.runtime.worker,c.ref,c.actor,key,decision='defer',next_review_at_ms=next_time)
        with pytest.raises(RuntimeRequestError,match='stale_wait_review'):
            await decide(c.agent.runtime.worker,c.ref,c.actor,key,decision='cancel')
        key=response['event_id']
        summary=await inspect(c.agent.runtime.client,c.ref,c.actor,key)
        assert summary['deadline_at_ms']==c.deadline and summary['due_at_ms']==next_time
    with pytest.raises(RuntimeRequestError,match='invalid_options'):
        await decide(c.agent.runtime.worker,c.ref,c.actor,key,decision='defer',next_review_at_ms=c.deadline)
    await decide(c.agent.runtime.worker,c.ref,c.actor,key,decision='cancel')
    result=await c.agent.runtime.client.wait(c.ref,actor=c.actor,deadline_ms=now_ms()+10000)
    assert result.status=='cancelled' and c.model.child_calls==1


@pytest.mark.parametrize('paused_review',[2000],indirect=True)
@pytest.mark.parametrize('late_response',[False,True])
async def test_original_deadline_stops_inflight_continuation(paused_review,monkeypatch,late_response):
    import asyncio

    from power_loop.runtime.wait_review import decide
    c=paused_review
    started,stopped,release=asyncio.Event(),asyncio.Event(),asyncio.Event()
    complete=c.model.complete
    async def blocked(request,**kwargs):
        response=await complete(request,**kwargs)
        if request.system_prompt.startswith('CHILD'):
            started.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                stopped.set()
                if not late_response:
                    raise
                await release.wait()
        return response
    monkeypatch.setattr(c.model,'complete',blocked)
    try:
        await decide(c.agent.runtime.worker,c.ref,c.actor,c.event_id,decision='continue')
        await asyncio.wait_for(started.wait(),1)
        await asyncio.wait_for(stopped.wait(),3)
        result=await c.agent.runtime.client.wait(c.ref,actor=c.actor,deadline_ms=now_ms()+3000)
        assert result.status=='cancelled'
        assert c.model.child_calls==2
    finally:
        release.set()
        await c.agent.drain()


async def test_unknown_timer_creation_is_not_replayed_and_new_decision_can_continue(paused_review,monkeypatch):
    from power_loop import RuntimeRequestError
    from power_loop.runtime.wait_review import arm, decide, event_id, inspect
    c=paused_review
    backend=c.agent.runtime.timer_backend
    calls=[]
    async def unknown(source,**options):
        calls.append(source.operation_id)
        raise RuntimeRequestError('timer_create_unknown')
    monkeypatch.setattr(backend,'create',unknown)
    with pytest.raises(RuntimeRequestError,match='timer_create_unknown'):
        await decide(c.agent.runtime.worker,c.ref,c.actor,c.event_id,decision='defer',next_review_at_ms=now_ms()+65000)
    state=await c.store.get_state(c.job['session_id'])
    key=event_id(state.pending['pending_interactions'][0])
    summary=await inspect(c.agent.runtime.client,c.ref,c.actor,key)
    assert summary['schedule_error']=='timer_create_unknown'
    await arm(c.agent.runtime.worker,c.job['session_id'],recover=True)
    await arm(c.agent.runtime.worker,c.job['session_id'])
    assert len(calls)==1
    await decide(c.agent.runtime.worker,c.ref,c.actor,key,decision='continue')
    result=await c.agent.runtime.client.wait(c.ref,actor=c.actor,deadline_ms=now_ms()+10000)
    assert result.status=='completed' and c.model.child_calls==2 and len(calls)==1


@pytest.mark.parametrize('options',[{'deadline_at_ms':True},{'deadline_at_ms':0},
    {'deadline_at_ms':1,'max_wait_ms':False},{'deadline_at_ms':1,'max_wait_ms':30*86400000+1}])
async def test_trusted_wait_policy_rejects_invalid_bounds(options):
    with pytest.raises(ValueError):
        WaitPolicy(**options)


async def test_workflow_leaf_wait_uses_original_main_and_capacity_one(registry_store):
    store,_=registry_store
    class Model(LLMService):
        node_calls=0
        decided=False
        async def complete(self,request,**kwargs):
            if request.system_prompt.startswith('NODE'):
                self.node_calls+=1
                if self.node_calls==1:
                    return call('wait_handoff',dict(action='request',due_at_ms=now_ms()+500,
                        note='Review workflow arithmetic',continuation='Finish original node: 42'),'node-wait')
                assert 'Finish original node: 42' in str(request.messages)
                return LLMResponse(raw_text='42')
            if not self.decided:
                for message in reversed(request.messages):
                    content=str(message.get('content',''))
                    if 'Task data:\n' in content:
                        value=json.JSONDecoder().raw_decode(content.split('Task data:\n',1)[1])[0]
                        self.decided=True
                        return call('wait_handoff',dict(action='decide',decision='continue',
                            **{k:value[k] for k in ['job_id','generation','event_id']}),'main-decision')
            return LLMResponse(raw_text='acknowledged')
        async def close(self):pass
    model,registry=Model(),ToolRegistry()
    register_wait_handoff_tool(registry,policy=WaitPolicy(now_ms()+20000))
    runner=await RunningAgent(store,model,registry,AgentLoopConfig(compactor=None),capacity=1).start()
    try:
        workflow=await runner.runtime.client.workflow(runner.target,{'name':'wait','root':{
            'type':'agent','id':'calculate','input':'calculate','spec':{'name':'calculate',
                'system_prompt':'NODE','tools':['wait_handoff'],'max_rounds':4}}},message_id='workflow',actor=runner.actor)
        result=await workflow.wait(deadline_ms=now_ms()+15000)
        assert result.status=='completed' and model.node_calls==2 and model.decided
        await runner.drain()
        assert not runner.runtime.worker._wait_deadlines
        [node]=await store._db.fetchall(f"SELECT generation FROM {store.table_prefix}jobs WHERE kind='workflow_node'")
        assert node['generation']==1
        [timer]=await store.list_timers(runner.target.session_id,statuses=('fired',))
        assert timer.session_id==runner.target.session_id
    finally:
        await runner.aclose()
