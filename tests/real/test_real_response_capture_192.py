"""Real response captured through the SDK port survives loss of its original host."""
import asyncio
import json
import signal
import sqlite3
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

from power_loop import SessionStore
from power_loop.runtime import model_attempts, model_budgets
from power_loop.runtime.budget_contracts import BudgetGrant, BudgetUsageEvidence
from power_loop.runtime.jobs import JobError, JobManager
from power_loop.runtime.response_capture import ModelResponseCapture
from power_loop.runtime.task_runtime import TaskRuntime
from tests.real._initial_evidence_worker_192 import RecoveryHost, config

pytestmark = pytest.mark.asyncio


@pytest.mark.parametrize('real', [False, pytest.param(True, marks=pytest.mark.real_llm)])
async def test_sdk_captured_response_survives_process_kill_before_observation(tmp_path, real, record_property):
    path, evidence_path = str(tmp_path / 'original.db'), str(tmp_path / 'captured.db')
    worker = await asyncio.create_subprocess_exec(sys.executable,
        str(Path(__file__).with_name('_response_capture_worker_192.py')),
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    record_property('original_worker_pid', worker.pid)
    runtime = store = None
    try:
        worker.stdin.write(json.dumps(dict(path=path, evidence_path=evidence_path, real=real)).encode())
        await worker.stdin.drain()
        worker.stdin.close()
        async def packet():
            line = await asyncio.wait_for(worker.stdout.readline(), 90)
            if not line:
                _, errors = await worker.communicate()
                pytest.fail(errors.decode()[-4000:])
            return json.loads(line)
        started = await packet()
        assert started['kind'] == 'request_started'
        record_property('real_provider_requests', started['real_provider_requests'])
        captured = await packet()
        assert captured['kind'] == 'capture_committed'
        record_property('reported_tokens', captured['reported_tokens'] if real else 0)
        worker.kill()
        await asyncio.wait_for(worker.communicate(), 5)
        assert worker.returncode == -signal.SIGKILL
        record_property('original_worker_exit', worker.returncode)
        with sqlite3.connect(evidence_path) as db:
            rows = db.execute('SELECT id,fingerprint,packet FROM captures').fetchall()
        assert len(rows) == 1
        identity, fingerprint, encoded = rows[0]
        raw = json.loads(encoded)
        raw['grant'] = BudgetGrant(**raw['grant'])
        capture = ModelResponseCapture(**raw)
        assert capture.capture_id == identity == captured['receipt']['capture_id']
        assert capture.fingerprint == fingerprint == captured['receipt']['fingerprint']
        assert capture.usage['total_tokens'] > 0
        store = await SessionStore.open(path, schema='compatible')
        manager, host = JobManager(store), RecoveryHost()
        runtime = TaskRuntime(manager, tenant_id='test', principal_id='host', executor_id='control',
            boot_id='control', boot_contract_hash='a' * 64, budget_config=config(host))
        sid, attempt_id, job_id = (capture.context[k] for k in ('session_id', 'attempt_id', 'job_id'))
        actor = runtime.actor(sid)
        original = await model_budgets.get_budget(manager, actor, attempt_id)
        assert original['context'] == capture.context and original['grant'] == asdict(capture.grant)
        assert original['phase'] == 'dispatch_authorized' and original['observation'] is None
        assert await model_attempts._read(manager, manager.db, 'finished', attempt_id, required=False) is None
        messages = await store.load_all_messages(sid)
        job = await manager.get_job(actor, job_id)
        evidence = BudgetUsageEvidence(identity, 'trusted-test-response-store', identity, 'actual', capture.usage)
        receipt = await runtime.stage_budget_usage_revision(sid, job_id=job_id, attempt_id=attempt_id,
            expected_revision=0, evidence=evidence)
        ack = await runtime.reconcile_budget_usage(sid, job_id=job_id, attempt_id=attempt_id)
        assert ack['usage_acknowledged'] and not ack['execution_authority']
        assert receipt == await runtime.stage_budget_usage_revision(sid, job_id=job_id, attempt_id=attempt_id,
            expected_revision=0, evidence=evidence)
        saved = await model_budgets.get_budget(manager, actor, attempt_id)
        assert saved['observation'] is None and 'finished_at_ms' not in saved
        assert await store.load_all_messages(sid) == messages and await manager.get_job(actor, job_id) == job
        assert host.reserve_calls == 0 and len(host.revisions) == 1 and not host.released
        assert runtime._lifecycle._start_task is None
        async with manager._write() as tx:
            row = await manager._authorize(tx, actor, job_id)
            with pytest.raises(JobError):
                await model_attempts.require_quiescent(manager, tx, row)
        record_property('capture_id', identity)
    finally:
        if worker.returncode is None:
            worker.kill()
            await worker.communicate()
        if runtime:
            await runtime.shutdown(timeout_s=3)
        if store:
            await store.close()


@pytest.mark.real_llm
async def test_real_capture_failure_preserves_actual_observation_without_retry(tmp_path, record_property):
    from power_loop import AgentLoopConfig, StatefulAgentLoop
    from power_loop.runtime.retry import LLMRetryPolicy
    from tests.real.test_real_async_boundaries_192 import BoundedProvider
    service, host = BoundedProvider(limit=1), RecoveryHost()
    host.service = service
    captures = []
    async def unavailable(packet):
        captures.append(packet)
        raise OSError('independent capture endpoint unavailable')
    host.capture_response = unavailable
    store = await SessionStore.open(str(tmp_path / 'failed-capture.db'), schema='tasks')
    manager = JobManager(store)
    runtime = TaskRuntime(manager, tenant_id='test', principal_id='host', executor_id='original',
        boot_id='original', boot_contract_hash='a' * 64, budget_config=config(host))
    loop = StatefulAgentLoop(llm=service, store=store, task_runtime=runtime,
        config=AgentLoopConfig(delegation_policy='single_level', max_rounds=1, max_tokens=2048,
                              retry_policy=LLMRetryPolicy(max_attempts=3)))
    try:
        sid = await loop.new_session()
        await loop.send('Reply only with the number 42.', sid)
        await runtime.flush_control_checkpoints()
        [capture] = captures
        original = await model_budgets.get_budget(manager, runtime.actor(sid), capture.grant.attempt_id)
        assert original['observation']['status'] == 'returned' and not original['pending']
        assert original['observation']['usage'] == capture.usage
        assert service.calls == host.reserve_calls == host.apply_calls == 1
        assert not runtime.budget_control_tasks()
    finally:
        service.report(record_property)
        await runtime.shutdown(timeout_s=3)
        await service.close()
        await store.close()
