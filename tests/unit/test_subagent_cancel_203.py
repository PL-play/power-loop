"""Root cancellation serializes with real child admission and late delivery."""
import asyncio
import json

import pytest

from power_loop import AgentLoopConfig
from power_loop.agent.runtime_store import now_ms
from power_loop.agent.runtime_types import JobRef
from power_loop.runtime import child_tasks
from power_loop.tools.registry import ToolRegistry
from power_loop.tools.spawn_agent import register_spawn_agent
from tests.runtime_fixture import RunningAgent

from .test_job_registry_192 import registry_store
from .test_subagent_runtime_203 import ParentAndChild

__all__=['registry_store']
pytestmark=[pytest.mark.unit,pytest.mark.asyncio]


@pytest.mark.parametrize('create_first',[False,True])
async def test_subtree_close_orders_real_child_creation(registry_store,monkeypatch,create_first):
    store,_=registry_store
    model,registry=ParentAndChild(),ToolRegistry()
    register_spawn_agent(registry)
    runner=await RunningAgent(store,model,registry,AgentLoopConfig(compactor=None)).start()
    entered,release=asyncio.Event(),asyncio.Event()
    original=child_tasks.admit
    async def blocked(*args,**kwargs):
        if create_first:
            result=await original(*args,**kwargs)
        entered.set()
        await release.wait()
        return result if create_first else await original(*args,**kwargs)
    monkeypatch.setattr(child_tasks,'admit',blocked)
    acquire=runner.runtime.coordinator.acquire
    async def main_only(key,**kwargs):
        return await acquire(key,**kwargs) if key[1]==runner.target.session_id else None
    monkeypatch.setattr(runner.runtime.coordinator,'acquire',main_only)
    try:
        receipt=await runner.runtime.client.send(runner.target,'delegate',message_id='run',actor=runner.actor)
        await asyncio.wait_for(entered.wait(),5)
        parent=await runner.runtime.client.inspect(receipt.input_ref,actor=runner.actor)
        await runner.runtime.client.cancel(parent.job_ref,actor=runner.actor,scope='subtree')
        release.set()
        result=await runner.runtime.client.wait(receipt.input_ref,actor=runner.actor,deadline_ms=now_ms()+10000)
        assert result.status=='cancelled'
        children=await store._db.fetchall(f"SELECT job_id,generation,cancel_requested_at FROM {store.table_prefix}jobs WHERE kind='subagent'")
        assert len(children)==int(create_first)
        for child in children:
            terminal=await runner.runtime.client.wait(JobRef(store.store_id,child['job_id'],0),actor=runner.actor,deadline_ms=now_ms()+1000)
            assert terminal.status=='cancelled' and terminal.job_ref.generation==0
        assert not model.child_started.is_set()
    finally:
        release.set()
        await runner.aclose()


async def test_saved_result_without_notify_is_readable_and_cancelled_scope_does_not_wake(registry_store,monkeypatch):
    store,_=registry_store
    model,registry=ParentAndChild(),ToolRegistry()
    register_spawn_agent(registry)
    model.child_gate=asyncio.Event()
    runner=await RunningAgent(store,model,registry,AgentLoopConfig(compactor=None)).start()
    saved,notify=asyncio.Event(),asyncio.Event()
    original=runner.runtime.client._notify
    async def hold_result(target,message_id,input_ref,channels,payload=None):
        if target.session_id==runner.target.session_id and channels==['work','activation']:
            record=await runner.runtime.worker.inputs.input(store._db,input_ref)
            if record and record.meta['operation']=='result_ready':
                saved.set()
                await notify.wait()
        return await original(target,message_id,input_ref,channels,payload)
    monkeypatch.setattr(runner.runtime.client,'_notify',hold_result)
    try:
        parent=await runner.send('delegate')
        rows=await store.load_all_messages(runner.target.session_id)
        receipt=json.loads(next(r.content for r in rows if r.tool_call_id=='spawn'))
        model.child_gate.set()
        await asyncio.wait_for(saved.wait(),5)
        result=await runner.runtime.client.wait(JobRef(**receipt['job_ref']),actor=runner.actor,deadline_ms=now_ms()+1000)
        assert result.final_text=='child result'
        await runner.runtime.client.cancel(parent.job_ref,actor=runner.actor,scope='subtree')
        notify.set()
        await runner.drain()
        await asyncio.sleep(.05)
        await runner.drain()
        assert len(model.requests)==3
        notices=await store._db.fetchall(f"SELECT status,meta_json FROM {store.table_prefix}follow_up_queue WHERE session_id=? AND kind='task_done'",(runner.target.session_id,))
        assert len(notices)==1 and notices[0]['status']=='void'
        assert json.loads(notices[0]['meta_json'])['reason']['detail']=='origin_cancelled'
        assert not await store._db.fetchall(f"SELECT * FROM {store.table_prefix}session_obligations WHERE session_id=?", (runner.target.session_id,))
    finally:
        notify.set()
        model.child_gate.set()
        await runner.aclose()
