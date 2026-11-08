"""Another process can cancel the original child even without a parent tool receipt."""
import asyncio
import json
import os
import sys
from dataclasses import asdict
from urllib.parse import urlparse

from power_loop import AgentLoopConfig, DatabasePollingTransport, JobRef, ToolRegistry
from power_loop.agent.runtime_store import RuntimeInputStore, now_ms
from power_loop.runtime.jobs import JobManager
from power_loop.runtime.task_runtime import TaskRuntime
from power_loop.runtime.tool_contracts import RuntimeTool
from power_loop.tools.spawn_agent import register_spawn_agent
from tests.runtime_fixture import RunningAgent


async def exercise(registry_store, model, record_property, *, target='child', real=False):
    store,_=registry_store
    manager=JobManager(store)
    managed=TaskRuntime(manager,tenant_id='no-return',principal_id='host',executor_id='original',boot_id='boot')
    registry=ToolRegistry()
    register_spawn_agent(registry)
    original=registry.get('spawn_agent')
    callbacks=[]
    entered,release=(model.child_entered,model.release_children) if real else (model.entered,model.release)
    class LostReturn(RuntimeTool):
        @property
        def runtime_contract(self):
            return original.handler.runtime_contract
        def dispatch_policy(self,arguments,*,runtime=None):
            return original.handler.dispatch_policy(arguments,runtime=runtime)
        async def __call__(self,**arguments):
            callbacks.append(await original.handler(**arguments))
            await asyncio.wait_for(entered.wait(),10)
            raise asyncio.CancelledError('callback interrupted before returning receipt')
    registry.register(original.definition,LostReturn(),overwrite=True)
    transport=DatabasePollingTransport(RuntimeInputStore(store),interval_s=.03,control_interval_s=.03)
    agent=await RunningAgent(store,model,registry,AgentLoopConfig(compactor=None,max_rounds=3,max_tokens=2048),
        task_runtime=managed,capacity=2,transport=transport).start()
    sid=agent.target.session_id
    if real:
        model.root_sid,model.hold_children=sid,True
    else:
        model.root=sid
    process=None
    try:
        parent=await agent.send("Use spawn_agent exactly once with tools=[] to calculate 19+23, then wait for its result.")
        assert parent.status=='cancelled' and len(callbacks)==1
        [child]=await manager.db.fetchall(f"SELECT job_id,parent_job_id,generation,resource_state FROM {manager.jobs} WHERE kind='subagent'")
        assert child['resource_state']=='running'
        assert not any(r.role=='tool' and r.name=='spawn_agent' for r in await store.load_all_messages(sid))
        calls=model.calls if real else model.child_calls
        engine=manager.db.dialect.name
        if engine=='sqlite':
            dsn=next(r['file'] for r in await manager.db.fetchall('PRAGMA database_list') if r['name']=='main')
        else:
            dsn=os.environ['POWER_LOOP_TASK_PG_DSN' if engine=='postgres' else 'POWER_LOOP_TASK_MYSQL_DSN']
            assert urlparse(dsn).path.endswith('_test')
        code='''import asyncio,json,sys
from power_loop import RuntimeClient,DatabasePollingTransport,JobRef,TrustedActor
from power_loop.runtime.store.factory import open_store
from power_loop.agent.runtime_store import RuntimeInputStore
async def main():
 d=json.load(sys.stdin)
 store=await open_store(d['dsn'],table_prefix=d['prefix'],schema='verify')
 transport=DatabasePollingTransport(RuntimeInputStore(store),interval_s=.03)
 client=RuntimeClient(store=store,transport=transport)
 try:
  ref=JobRef(store.store_id,d['target'],1)
  actor=TrustedActor(**d['actor'])
  one=await client.cancel(ref,actor=actor,scope='subtree',reason='stop original work')
  two=await client.cancel(ref,actor=actor,scope='subtree',reason='stop original work')
  assert one.target.job_id==two.target.job_id==d['target']
  print(json.dumps({'job_id':one.target.job_id,'state':one.state}))
 finally:
  await client.aclose()
  await transport.aclose()
  await store.close()
asyncio.run(main())'''
        process=await asyncio.create_subprocess_exec(sys.executable,'-c',code,stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.PIPE)
        record_property('original_no_return_stop_pid',process.pid)
        output,error=await asyncio.wait_for(process.communicate(json.dumps({'dsn':dsn,'prefix':store.table_prefix,
            'actor':asdict(agent.actor),'target':child['parent_job_id'] if target=='root' else child['job_id']}).encode()),30)
        assert process.returncode==0,error.decode()[-2000:]
        assert json.loads(output)['state']=='applied'
        result=await agent.runtime.client.wait(JobRef(store.store_id,child['job_id'],1),actor=agent.actor,deadline_ms=now_ms()+10000)
        assert result.status=='cancelled'
        await agent.drain()
        assert not release.is_set() and len(callbacks)==1
        assert (model.calls if real else model.child_calls)==calls
        assert (await manager.db.fetchone(f"SELECT COUNT(*) AS n FROM {manager.jobs} WHERE kind='subagent'"))['n']==1
    finally:
        if process is not None and process.returncode is None:
            process.kill()
            await process.wait()
        release.set()
        await agent.aclose()
        await transport.aclose()
