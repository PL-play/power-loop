"""A exits, B retains children, C controls and consumes the original results."""
import asyncio
import json
import os
import sys
from pathlib import Path

import pytest

from power_loop import DatabasePollingTransport, RuntimeClient
from power_loop.agent.runtime_store import RuntimeInputStore, now_ms
from power_loop.agent.runtime_types import JobRef, TargetRef, TrustedActor

from .test_job_registry_192 import registry_store

__all__ = ['registry_store']
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def test_parent_a_child_b_parent_c_with_original_budget(registry_store, tmp_path, parent_lost=False):
    store, _ = registry_store
    descriptor = dict(schema_version=1,kind='agent',definition_ref='p3.agent@1', model_profile_ref=None,
        tool_profile_ref='p3.tools@1', hook_profile_ref='p3.hooks@1',resource_profile_ref='p3.resources@1',config={},host_context={})
    sid = await store.create_session(metadata=dict(tenant_id='tenant',principal_id='principal',execution_role='main',runtime_descriptor=descriptor))
    backend = store._db.dialect.name
    dsn = str(tmp_path/'registry.db') if backend == 'sqlite' else os.environ['POWER_LOOP_TASK_PG_DSN' if backend == 'postgres' else 'POWER_LOOP_TASK_MYSQL_DSN']
    processes = []
    async def launch(role):
        process = await asyncio.create_subprocess_exec(sys.executable,'-m','tests.unit._p3_runtime_worker',dsn,store.table_prefix,sid,str(tmp_path/'coordinator.db'),role,str(tmp_path),
            cwd=Path(__file__).resolve().parents[2],stdin=asyncio.subprocess.PIPE,stdout=asyncio.subprocess.PIPE,stderr=(tmp_path/(role+'-stderr.log')).open('wb'))
        process._stderr_path=tmp_path/(role+'-stderr.log')
        processes.append(process)
        return process
    async def event(process, kind):
        try:
            line = await asyncio.wait_for(process.stdout.readline(),20)
        except TimeoutError:
            process.kill()
            await process.wait()
            raise AssertionError(process._stderr_path.read_bytes().decode()[-8000:]) from None
        assert line, process._stderr_path.read_bytes().decode()[-20000:]
        value = json.loads(line)
        assert value['event']==kind,value
        return value
    async def command(process, value):
        process.stdin.write(((value if isinstance(value,str) else json.dumps(value))+'\n').encode())
        await process.stdin.drain()
    transport = DatabasePollingTransport(RuntimeInputStore(store),interval_s=.03)
    client = RuntimeClient(store=store,transport=transport)
    actor = TrustedActor('tenant','principal',None)
    try:
        a = await launch('A-parent_lost' if parent_lost else 'A')
        created = await event(a,'parent_finished')
        if not parent_lost:
            await asyncio.wait_for(a.wait(),10)
            assert a.returncode==0, a._stderr_path.read_bytes().decode()[-20000:]
        assert len(created['children'])==2
        one,two = created['children']
        b = await launch('B')
        await event(b,'ready')
        for _ in range(2):
            await event(b,'child_running')
        if parent_lost:
            assert a.returncode is None
            a.kill()
            await a.wait()
        c = await launch('C')
        await event(c,'ready')
        await command(c,dict(id='discover',tool='jobs',args=dict(action='list',kinds=['subagent'])))
        listing = json.loads((await event(c,'controlled'))['output'])
        assert {j['job_id'] for j in listing['nodes'] if j['kind']=='subagent'} == {one['job_id'],two['job_id']}
        await command(c,dict(id='nudge',tool='spawn_agent',args=dict(action='send',sub_session=one['session_id'],message='new direction')))
        assert json.loads((await event(c,'controlled'))['output'])['state']=='accepted'
        await command(c,dict(id='stop',tool='jobs',args=dict(action='stop',job_id=two['job_id'],expected_generation=1)))
        assert json.loads((await event(c,'controlled'))['output'])['status']=='accepted'
        stopped = await client.wait(JobRef(**two['job_ref']),actor=actor,deadline_ms=now_ms()+15000)
        assert stopped.status=='cancelled'
        await command(b,'release')
        await event(b,'released')
        result = await client.wait(JobRef(**one['job_ref']),actor=actor,deadline_ms=now_ms()+15000)
        assert result.status=='completed' and 'new direction' in result.final_text
        for _ in range(500):
            rows = await store._db.fetchall(f'SELECT job_id,result_status,supervision_status FROM {store.table_prefix}jobs WHERE job_id IN (?,?)',(one['job_id'],two['job_id']))
            if all(row['result_status']=='handled'
                   and row['supervision_status']=='closed' for row in rows):
                break
            await asyncio.sleep(.02)
        assert {row['job_id']:row['result_status'] for row in rows} == {one['job_id']:'handled',two['job_id']:'handled'},rows
        assert all(row['supervision_status']=='closed' for row in rows)
        roots = await store._db.fetchall(f'SELECT root_job_id,generation FROM {store.table_prefix}jobs WHERE job_id IN (?,?)',(one['job_id'],two['job_id']))
        assert all(r==dict(root_job_id=created['job_id'],generation=1) for r in roots)
        for process in (b,c):
            await command(process,'quit')
            await asyncio.wait_for(process.wait(),10)
            assert process.returncode==0, process._stderr_path.read_bytes().decode()[-20000:]
    finally:
        await client.aclose()
        await transport.aclose()
        for process in processes:
            if process.returncode is None:
                process.kill()
            await process.wait()


async def test_background_query_cancel_and_lost_shell_never_replays(registry_store, tmp_path):
    store, _ = registry_store
    descriptor = dict(schema_version=1,kind='agent',definition_ref='p3.agent@1',model_profile_ref=None,
        tool_profile_ref='p3.tools@1',hook_profile_ref='p3.hooks@1',resource_profile_ref='p3.resources@1',config={},host_context={})
    sid = await store.create_session(metadata=dict(tenant_id='tenant',principal_id='principal',execution_role='main',runtime_descriptor=descriptor))
    backend = store._db.dialect.name
    dsn = str(tmp_path/'registry.db') if backend=='sqlite' else os.environ['POWER_LOOP_TASK_PG_DSN' if backend=='postgres' else 'POWER_LOOP_TASK_MYSQL_DSN']
    processes=[]
    async def launch(role):
        process=await asyncio.create_subprocess_exec(sys.executable,'-m','tests.unit._p3_runtime_worker',dsn,store.table_prefix,sid,str(tmp_path/'coordinator.db'),role+'-background',str(tmp_path),
            cwd=Path(__file__).resolve().parents[2],stdin=asyncio.subprocess.PIPE,stdout=asyncio.subprocess.PIPE,stderr=(tmp_path/(role+'-stderr.log')).open('wb'))
        process._stderr_path=tmp_path/(role+'-stderr.log')
        processes.append(process)
        return process
    async def read(process):
        try:
            line=await asyncio.wait_for(process.stdout.readline(),20)
        except TimeoutError:
            process.kill()
            await process.wait()
            raise AssertionError(process._stderr_path.read_bytes().decode()[-8000:]) from None
        assert line,process._stderr_path.read_bytes().decode()[-20000:]
        return json.loads(line)
    async def control(c,action,**options):
        data=dict(id=action,tool='background_run',args=dict(action=action,**options))
        c.stdin.write((json.dumps(data)+'\n').encode())
        await c.stdin.drain()
        response=await read(c)
        assert response['event']=='controlled',response
        return response['output']
    transport=DatabasePollingTransport(RuntimeInputStore(store),interval_s=.03)
    client=RuntimeClient(store=store,transport=transport)
    actor=TrustedActor('tenant','principal',None)
    try:
        a=await launch('A')
        created=await read(a)
        await asyncio.wait_for(a.wait(),10)
        assert a.returncode==0,a._stderr_path.read_bytes().decode()[-20000:]
        one,two=created['children']
        b=await launch('B')
        assert (await read(b))['event']=='ready'
        for _ in range(500):
            if all((tmp_path/('marker-'+name)).exists() for name in ('one','two')):
                break
            await asyncio.sleep(.02)
        assert all((tmp_path/('marker-'+name)).exists() for name in ('one','two'))
        c=await launch('C')
        assert (await read(c))['event']=='ready'
        listed=await control(c,'list')
        assert one['job_id'] in listed and two['job_id'] in listed
        checked=await control(c,'check',task_id=one['job_id'])
        assert 'running' in checked
        timed=await control(c,'wait',task_id=one['job_id'],timeout_s=.05)
        assert 'running' in timed
        cancelled=await control(c,'cancel',task_id=one['job_id'])
        assert json.loads(cancelled)['state']=='applied'
        first=await client.wait(JobRef(**one['job_ref']),actor=actor,deadline_ms=now_ms()+15000)
        assert first.status=='cancelled' and (tmp_path/'marker-one').read_text()=='started\n'
        b.kill()
        await b.wait()
        # Another B starts with the same definitions. Recovery reports unknown
        # resource state; it must never launch the original shell again.
        replacement=await launch('B')
        assert (await read(replacement))['event']=='ready'
        lost=await client.wait(JobRef(**two['job_ref']),actor=actor,deadline_ms=now_ms()+15000)
        assert lost.status=='interrupted'
        assert (await client.inspect(JobRef(**two['job_ref']),actor=actor)).resource_state=='unknown'
        await asyncio.sleep(.2)
        assert (tmp_path/'marker-two').read_text()=='started\n'
        following = await client.send(TargetRef(store.store_id, sid), 'after unknown background',
                                      message_id='after-unknown-background', actor=actor)
        continued = await client.wait(following.input_ref, actor=actor, deadline_ms=now_ms()+15000)
        assert continued.status == 'completed', continued
        assert (tmp_path/'marker-two').read_text()=='started\n'
        for process in (c,replacement):
            process.stdin.write(b'quit\n')
            await process.stdin.drain()
            await asyncio.wait_for(process.wait(),10)
            assert process.returncode==0,process._stderr_path.read_bytes().decode()[-20000:]
    finally:
        await client.aclose()
        await transport.aclose()
        for process in processes:
            if process.returncode is None:
                process.kill()
            await process.wait()
