"""Concurrency through current workers and independently owned task sessions."""
import asyncio
import json
import os
import sys
from dataclasses import asdict

import pytest

from power_loop import AgentLoopConfig, ExecutionEnvironment, LLMResponse, LLMService, RegistryRuntimeFactory, Runtime
from power_loop.agent.runtime_store import now_ms
from power_loop.runtime.jobs import JobError, JobManager

from .test_job_registry_192 import registry_store, runtime
from .test_runtime_messaging_203 import fixture

__all__=['registry_store','runtime']


async def test_separate_default_processes_never_dispatch_two_models(registry_store):
    store,_=registry_store
    _,target,actor=await fixture(store)
    kind=store._db.dialect.name
    dsn=store._db.path if kind=='sqlite' else os.environ['POWER_LOOP_TASK_PG_DSN' if kind=='postgres' else 'POWER_LOOP_TASK_MYSQL_DSN']
    env={**os.environ,'P7A_DB_DSN':dsn,'P7A_TABLE_PREFIX':store.table_prefix,
         'P7A_ACTOR':json.dumps(asdict(actor)),'P7A_SESSION':target.session_id}
    processes=[]
    async def spawn():
        p=await asyncio.create_subprocess_exec(sys.executable,'-m','tests.concurrent_send_worker_208',env=env,
            stdin=asyncio.subprocess.PIPE,stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.PIPE)
        processes.append(p)
        return p
    try:
        first=await spawn()
        async with asyncio.timeout(20):
            assert await first.stdout.readline()==b'model_entered\n'
            second=await spawn()
            out,error=await second.communicate()
            assert second.returncode==0 and out==b'invalid_options\n',error.decode()
            first.stdin.write(b'release\n')
            await first.stdin.drain()
            out,error=await first.communicate()
            assert first.returncode==0 and out==b'completed\n',error.decode()
        assert [r.content for r in await store.load_all_messages(target.session_id)]==['first','first']
    finally:
        for p in processes:
            if p.returncode is None:
                p.kill()
                await p.wait()


async def test_different_sessions_enter_models_concurrently_and_close_without_deadlock(registry_store):
    store,_=registry_store
    _,first,actor=await fixture(store)
    _,second,_=await fixture(store)
    entered,release=asyncio.Event(),asyncio.Event()
    arrivals=[]
    class Model(LLMService):
        def __init__(self,sid): self.sid=sid
        async def complete(self,request,**kwargs):
            arrivals.append(self.sid)
            if len(arrivals)==2:
                entered.set()
            await release.wait()
            return LLMResponse(raw_text=self.sid)
        async def close(self): pass
    factory=RegistryRuntimeFactory()
    async def build(descriptor,context):
        return ExecutionEnvironment(Model(context.session_id),None,AgentLoopConfig(compactor=None))
    factory.register('test.agent@1',build)
    worker=Runtime(store=store,factory=factory,node_id='concurrent',capacity=2,timers_enabled=False)
    await worker.worker.start()
    try:
        receipts=await asyncio.gather(*(worker.client.send(t,'start',message_id='start',actor=actor) for t in (first,second)))
        await asyncio.wait_for(entered.wait(),10)
        assert set(arrivals)=={first.session_id,second.session_id}
        release.set()
        results=await asyncio.gather(*(worker.client.wait(r.input_ref,actor=actor,deadline_ms=now_ms()+10000) for r in receipts))
        assert [r.final_text for r in results]==[first.session_id,second.session_id]
    finally:
        release.set()
        await worker.worker.shutdown()
        await worker.client.aclose()
        await worker.transport.aclose()


@pytest.mark.parametrize("order", ["stop_first", "results_first", "interleaved"])
async def test_same_tree_stop_new_send_and_duplicate_child_completions_do_not_deadlock(runtime, order):
    _store, manager, actor, root, parent, connect, _now = runtime
    others = [JobManager(await connect(schema="verify"), now_ms=manager.now) for _ in range(3)]
    children = []
    for i in range(2):
        sid = await _store.create_session(parent_session_id=actor.session_id)
        registered = await manager.register_job(actor, kind="subagent", session_id=sid, parent=parent, system_key=f"child-{i}")
        lease = await manager.claim_job(actor, registered.job.job_id, executor_id=f"worker-{i}",
                                        boot_id="boot", expected_version=1)
        children.append(lease)
    barrier = asyncio.Event()

    async def finish(i, m):
        await barrier.wait()
        return await m.record_execution_result(actor, children[i], outcome="succeeded",
            result={"value": i}, resource_state="stopped")

    async def stop(m):
        await barrier.wait()
        return await m.request_cancel(actor, root.job.job_id, system_key="stop-tree", reason="stop original tree")

    async def new_send():
        await barrier.wait()
        try:
            return await others[2].register_job(actor, kind="send", session_id=actor.session_id,
                parent=parent, system_key="concurrent-send", request={"model_attempt_protocol": 1})
        except JobError as exc:
            assert exc.code == "cancel_requested"
            return None

    completions = [finish(i, m) for i, m in ((0, manager), (1, others[0]), (0, others[1]), (1, others[2]))]
    stops = [stop(manager), stop(others[1])]
    actions = stops + completions + [new_send()] if order == "stop_first" else (
        completions + [new_send()] + stops if order == "results_first" else [completions[0], stops[0], *completions[1:], new_send(), stops[1]])
    tasks = [asyncio.create_task(action) for action in actions]
    barrier.set()
    results = await asyncio.wait_for(asyncio.gather(*tasks), 15)
    assert (await manager.get_job(actor, root.job.job_id)).cancel_requested_at is not None
    for i, lease in enumerate(children):
        row = await manager.get_job(actor, lease.job_id)
        assert row.result == {"value": i} and row.resource_state == "stopped"
        assert row.outcome in ("succeeded", "cancelled")
        events = await manager.db.fetchall(f"SELECT event_id FROM {manager.events} WHERE job_id=? AND event_type='result'",
                                         (lease.job_id,))
        assert len(events) == 1
        assert await others[2].record_execution_result(actor, lease, outcome="succeeded",
            result={"value": i}, resource_state="stopped") == events[0]["event_id"]
    stops = [value for value in results if getattr(value, "action", None) == "stop"]
    assert len(stops) == 2 and stops[0] == stops[1]
    for registered in (value for value in results if getattr(value, "job", None) is not None):
        with pytest.raises(JobError, match="cancel_requested"):
            await manager.claim_job(actor, registered.job.job_id, executor_id="too-late", boot_id="boot", expected_version=1)
