"""The SDK tool passes trusted identity to the explicitly selected backend."""
import time
from dataclasses import replace

from power_loop import (
    AgentLoopConfig,
    ExecutionEnvironment,
    RegistryRuntimeFactory,
    Runtime,
    RuntimeDescriptor,
    TimerDriverStatus,
    TrustedActor,
    create_default_tool_registry,
    get_tool_definitions,
)
from power_loop.agent.runtime_types import plain
from power_loop.runtime.store.factory import open_store
from power_loop.runtime.store.types import TimerRow
from power_loop.tools import DEFAULT_TOOL_HANDLERS

from .test_timers import _Scripted, _tool_resp


class Backend:
    def __init__(self):
        self.calls = []
        self.timer = TimerRow(session_id='',timer_id=195,due_at=5000,note='',status='armed',
            interval_s=None,fire_count=0,last_fired_at=None,created_at=0,updated_at=0)
    async def create(self, source, **kwargs):
        self.calls.append(('create',source,kwargs))
        self.timer = replace(self.timer,session_id=source.session_id,note=kwargs['note'],interval_s=kwargs['interval_s'])
        return self.timer
    async def list(self, source, *, statuses):
        self.calls.append(('list',source,statuses))
        return [self.timer] if self.timer.session_id==source.session_id and self.timer.status in statuses else []
    async def cancel(self, source, tid):
        self.calls.append(('cancel',source,tid))
        self.timer = replace(self.timer,status='cancelled')
        return True


class Driver:
    def __init__(self): self.started = False
    async def start(self): self.started = True
    async def stop(self): self.started = False
    def status(self): return TimerDriverStatus('external', self.started)


async def test_model_tool_schedule_list_cancel_share_backend_and_original_identity(tmp_path):
    backend, driver = Backend(), Driver()
    store = await open_store(str(tmp_path/'timers.db'), schema='tasks')
    registry = create_default_tool_registry(preset='core',bind=False)
    for definition in get_tool_definitions(include=['schedule_wakeup']):
        registry.register(definition,DEFAULT_TOOL_HANDLERS[definition.name],overwrite=True)
    model = _Scripted(responses=[
        _tool_resp('schedule','schedule_wakeup','{"action":"schedule","delay_seconds":10,"note":"check"}'),
        _tool_resp('list','schedule_wakeup','{"action":"list"}'),
        _tool_resp('cancel','schedule_wakeup','{"action":"cancel","timer_id":195}'),
    ])
    factory = RegistryRuntimeFactory()
    async def build(descriptor, context):
        return ExecutionEnvironment(model,registry,AgentLoopConfig(max_rounds=5))
    factory.register('timer@1', build)
    runtime = Runtime(store=store, factory=factory, node_id='timer', timer_backend=backend, timer_driver=driver)
    try:
        await runtime.worker.start()
        actor = TrustedActor('tenant','principal',None)
        descriptor = RuntimeDescriptor(1,'agent','timer@1',None,'tools@1','hooks@1','resources@1',{}, {'source':{'original':'run'}})
        target = await runtime.open_session(descriptor,actor=actor)
        receipt = await runtime.client.send(target,'schedule, list, cancel',message_id='original',actor=actor)
        result = await runtime.client.wait(receipt.input_ref,actor=actor,deadline_ms=int(time.time()*1000)+10000)
        assert result.status == 'completed', result
        creates = [c for c in backend.calls if c[0]=='create']
        cancels = [c for c in backend.calls if c[0]=='cancel']
        assert len(creates)==len(cancels)==1
        source = creates[0][1]
        assert source.store_id == store.store_id and source.session_id == target.session_id
        assert source.actor.tenant_id == 'tenant' and source.job_ref == result.job_ref
        assert source.tool_call_id == 'schedule' and source.operation_id.startswith('timer-tool-')
        assert cancels[0][1].operation_id != source.operation_id
        assert plain(source.host_context) == {'source':{'original':'run'}}
        assert backend.timer.status == 'cancelled'
        assert await store.list_timers(target.session_id,statuses=('armed','fired','cancelled')) == []
    finally:
        await runtime.aclose()
    assert not driver.started
