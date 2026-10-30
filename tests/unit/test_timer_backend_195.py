"""A host backend owns every timer surface; native work cannot use store timers."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from power_loop import (
    AgentLoopConfig,
    StatefulAgentLoop,
    TimerRunner,
    create_default_tool_registry,
    get_tool_definitions,
)
from power_loop.runtime.jobs import JobError
from power_loop.runtime.store.types import TimerRow
from power_loop.tools import DEFAULT_TOOL_HANDLERS

from .test_timers import _Scripted, _tool_resp


class Backend:
    def __init__(self):
        self.calls = []
        self.timer = TimerRow(session_id='',timer_id=195,due_at=5000,note='',status='armed',
            interval_s=None,fire_count=0,last_fired_at=None,created_at=0,updated_at=0)
    async def create(self, sid, **kwargs):
        self.calls.append(('create',sid,kwargs))
        self.timer = replace(self.timer,session_id=sid,note=kwargs['note'],interval_s=kwargs['interval_s'])
        return self.timer
    async def list(self, sid, *, statuses):
        self.calls.append(('list',sid,statuses))
        return [self.timer] if self.timer.session_id==sid and self.timer.status in statuses else []
    async def cancel(self, sid, tid, **kwargs):
        self.calls.append(('cancel',sid,tid,kwargs))
        if (self.timer.session_id,self.timer.timer_id,self.timer.status)!=(sid,tid,'armed'):
            return False
        self.timer = replace(self.timer,status='cancelled')
        return True


async def test_model_tool_schedule_list_cancel_share_host_backend_and_original_call_ids(tmp_path):
    backend = Backend()
    registry = create_default_tool_registry(preset='core',bind=False)
    for definition in get_tool_definitions(include=['schedule_wakeup']):
        registry.register(definition,DEFAULT_TOOL_HANDLERS[definition.name],overwrite=True)
    loop = StatefulAgentLoop(llm=_Scripted(responses=[
        _tool_resp('schedule','schedule_wakeup','{"action":"schedule","delay_seconds":10,"note":"check"}'),
        _tool_resp('list','schedule_wakeup','{"action":"list"}'),
        _tool_resp('cancel','schedule_wakeup','{"action":"cancel","timer_id":195}'),
    ]),db_path=str(tmp_path/'timers.sqlite'),timer_backend=backend,
        config=AgentLoopConfig(max_rounds=5),
        tool_registry=registry)
    try:
        sid = await loop.new_session()
        await loop.send('schedule, list, then cancel',session_id=sid)
        creates = [c for c in backend.calls if c[0]=='create']
        cancels = [c for c in backend.calls if c[0]=='cancel']
        assert len(creates)==len(cancels)==1
        body = creates[0][2]
        assert body['delay_s']==10 and body['due_at_ms'] is None
        assert body['operation_id'].startswith('timer-tool-')
        assert cancels[0][3]['operation_id'] != body['operation_id']
        assert backend.timer.status=='cancelled'
        assert await loop.store.list_timers(sid,statuses=('armed','cancelled','fired'))==[]
    finally:
        await loop.aclose()


async def test_external_sdk_and_task_list_use_the_same_host_backend(tmp_path):
    backend = Backend()
    loop = StatefulAgentLoop(llm=_Scripted(),db_path=str(tmp_path/'timers.sqlite'),timer_backend=backend)
    try:
        sid = await loop.new_session()
        timer = await loop.schedule_timer(sid,due_at_ms=123456,note=' check ',operation_id='original')
        assert backend.calls[-1][2]==dict(due_at_ms=123456,delay_s=None,note='check',interval_s=None,operation_id='original')
        assert [t.timer_id for t in await loop.list_timers(sid)]==[timer.timer_id]
        assert any(row['id']=='timer:195' for row in await loop.list_jobs(sid))
        assert await loop.cancel_timer(sid,195,operation_id='cancel-original')
        assert await loop.list_timers(sid)==[]
        assert await loop.store.list_timers(sid,statuses=('armed','cancelled'))==[]
    finally:
        await loop.aclose()


@pytest.mark.parametrize('action',['create','list','cancel'])
async def test_native_runtime_requires_a_host_backend_on_every_timer_surface(tmp_path,action):
    loop = StatefulAgentLoop(llm=_Scripted(),db_path=str(tmp_path/'timers.sqlite'),
        config=AgentLoopConfig(delegation_policy='single_level'))
    try:
        sid = await loop.new_session()
        loop.task_runtime = SimpleNamespace(manager=SimpleNamespace(store=loop.store))
        with pytest.raises(JobError,match='task_adapter_unavailable:timer'):
            if action=='create':
                await loop.schedule_timer(sid,delay_s=10,note='check')
            elif action=='list':
                await loop.list_timers(sid)
            else:
                await loop.cancel_timer(sid,195)
        assert await loop.store.list_timers(sid)==[]
    finally:
        loop.task_runtime = None
        await loop.aclose()


async def test_finite_child_is_rejected_before_host_timer_io(tmp_path):
    backend = Backend()
    loop = StatefulAgentLoop(llm=_Scripted(),db_path=str(tmp_path/'timers.sqlite'),timer_backend=backend,
        config=AgentLoopConfig(delegation_policy='single_level'))
    try:
        parent = await loop.new_session()
        child = await loop.store.create_session(parent_session_id=parent,metadata={'execution_role':'subagent'})
        with pytest.raises(ValueError,match='execution_role_forbids_timer'):
            await loop.schedule_timer(child,delay_s=10,note='check',operation_id='child')
        assert backend.calls==[]
    finally:
        await loop.aclose()


async def test_store_timer_runner_cannot_own_host_reminders(tmp_path):
    loop=StatefulAgentLoop(llm=_Scripted(),db_path=str(tmp_path/'timers.sqlite'),timer_backend=Backend())
    try:
        with pytest.raises(ValueError,match='host timer backend requires its own delivery driver'):
            await TimerRunner(loop).start()
    finally:
        await loop.aclose()
