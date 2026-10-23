"""Kill only an owned child process; startup must retain its original unknown work."""
from __future__ import annotations

import asyncio
import json
import signal
import sys

import pytest

from power_loop import SessionStore
from power_loop.runtime import executor_registry as registry
from power_loop.runtime.jobs import JobActor, JobError, JobManager
from power_loop.runtime.model_attempt_query import list_attempts

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]

SCRIPT = r'''
import asyncio,json,sys
from power_loop import AgentLoopConfig,LLMService,SessionStore,StatefulAgentLoop
from power_loop.core.agent_context import get_session_id
from power_loop.runtime.jobs import JobManager
from power_loop.runtime.task_runtime import TaskRuntime
from power_loop.runtime import executor_registry as registry
class HeldModel(LLMService):
    async def complete(self,request,**kwargs):
        print('READY:'+json.dumps({'sid':get_session_id(),'store_id':store.store_id}),flush=True)
        await asyncio.Event().wait()
    def stream(self,request):
        async def empty():
            if False: yield None
        return empty()
    async def close(self): pass
async def main():
    global store
    store=await SessionStore.open(sys.argv[1],schema='tasks')
    manager=JobManager(store)
    identity=registry.ExecutorIdentity('crash-test','host','executor','old-boot')
    await registry.register(manager,identity,contract_hash='a'*64)
    await registry.activate(manager,identity,expected_version=1,system_key='startup')
    runtime=TaskRuntime(manager,tenant_id='crash-test',principal_id='host',executor_id='executor',boot_id='old-boot')
    loop=StatefulAgentLoop(llm=HeldModel(),store=store,task_runtime=runtime,
        config=AgentLoopConfig(delegation_policy='single_level',max_rounds=2))
    await loop.send('Wait for the external test stop.',await loop.new_session())
asyncio.run(main())
'''


@pytest.mark.parametrize("stop_signal", [signal.SIGTERM, signal.SIGKILL])
async def test_process_death_preserves_dispatched_attempt_and_blocks_unconfirmed_replacement(tmp_path, stop_signal):
    path = tmp_path / "killed-executor.db"
    process = await asyncio.create_subprocess_exec(sys.executable, "-c", SCRIPT, str(path),
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, start_new_session=True)
    try:
        async with asyncio.timeout(20):
            while True:
                line = await process.stdout.readline()
                if not line:
                    pytest.fail((await process.stderr.read()).decode())
                if line.startswith(b"READY:"):
                    ready = json.loads(line[len(b"READY:"):])
                    break
            process.send_signal(stop_signal)
            assert await process.wait() == -stop_signal
        store = await SessionStore.open(str(path), schema="tasks")
        try:
            assert store.store_id == ready["store_id"]
            manager = JobManager(store)
            old = registry.ExecutorIdentity("crash-test", "host", "executor", "old-boot")
            record = await registry.get(manager, old)
            # Deterministic expiry after an actual process death; expiry is not
            # evidence that a remote model request stopped or cost nothing.
            future = JobManager(store, now_ms=lambda: record["lease_until_ms"]+1)
            actor = JobActor(old.tenant_id, old.principal_id, ready["sid"])
            [job] = await future.db.fetchall(f"SELECT * FROM {future.jobs}")
            page = await list_attempts(future, actor, job["job_id"])
            assert page["items"][0]["status"] == "dispatched_unconfirmed"
            assert not page["items"][0]["response_received"] and page["billing_status"] == "unavailable"
            assert job["resource_state"] == "running" and job["outcome"] is None
            newer = registry.ExecutorIdentity("crash-test", "host", "executor", "new-boot")
            await registry.register(future, newer, contract_hash="a"*64)
            with pytest.raises(JobError, match="executor_prior_execution_unsettled"):
                await registry.activate(future, newer, expected_version=1, system_key="startup")
            stopped = await registry.drain(future, old, system_key="recover-stop", reason="owned process is gone; remote request remains unknown")
            assert stopped["effects"]["executions"][0]["job_id"] == job["job_id"]
            assert not stopped["physical_stop_confirmed"]
            after = await future.get_job(actor, job["job_id"])
            assert after.cancel_requested_at is not None and after.supervision_status == "open"
        finally:
            await store.close()
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()
