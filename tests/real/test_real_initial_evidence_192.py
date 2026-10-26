"""SIGKILL after a real response cannot turn late accounting into execution exit."""
import asyncio
import json
import signal
import sys
from pathlib import Path

import pytest

from power_loop import SessionStore
from power_loop.runtime import model_attempts, model_budgets
from power_loop.runtime.budget_contracts import BudgetUsageEvidence
from power_loop.runtime.jobs import JobError, JobManager
from power_loop.runtime.task_runtime import TaskRuntime
from tests.real._initial_evidence_worker_192 import RecoveryHost, config

pytestmark = pytest.mark.asyncio


@pytest.mark.parametrize('real', [False, pytest.param(True, marks=pytest.mark.real_llm)])
async def test_killed_original_process_has_accounting_evidence_but_no_exit(tmp_path, real, record_property):
    path = str(tmp_path / 'killed.db')
    process = await asyncio.create_subprocess_exec(sys.executable,
        str(Path(__file__).with_name('_initial_evidence_worker_192.py')),
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    record_property('original_worker_pid', process.pid)
    store = None
    controls = []
    try:
        process.stdin.write(json.dumps({'path': path, 'real': real}).encode())
        await process.stdin.drain()
        process.stdin.close()
        packet = await asyncio.wait_for(process.stdout.readline(), 90)
        if not packet:
            _, errors = await process.communicate()
            pytest.fail(errors.decode()[-4000:])
        started = json.loads(packet)
        assert started['kind'] == 'request_started'
        # Retain the real request count even when the supplier errors before a
        # captured response. Missing reported tokens are not invented as usage.
        record_property('real_provider_requests', started['real_provider_requests'])
        packet = await asyncio.wait_for(process.stdout.readline(), 90)
        if not packet:
            _, errors = await process.communicate()
            pytest.fail(errors.decode()[-4000:])
        facts = json.loads(packet)
        assert facts['real_provider_requests'] == started['real_provider_requests']
        record_property('reported_tokens', facts['reported_tokens'] if real else 0)
        assert facts['kind'] == 'captured_before_exit' and '42' in facts['text']
        assert process.returncode is None
        process.kill()
        await asyncio.wait_for(process.communicate(), 5)
        assert process.returncode == -signal.SIGKILL
        record_property('original_worker_exit', process.returncode)
        store = await SessionStore.open(path, schema='compatible')
        manager, host = JobManager(store), RecoveryHost()
        def control():
            runtime = TaskRuntime(manager, tenant_id='test', principal_id='host', executor_id='control',
                boot_id='unused', boot_contract_hash='a' * 64, budget_config=config(host))
            controls.append(runtime)
            return runtime
        first = control()
        sid, job_id, attempt_id = (facts[k] for k in ('sid','job_id','attempt_id'))
        actor = first.actor(sid)
        original = await model_budgets.get_budget(manager, actor, attempt_id)
        assert original['phase'] == 'dispatch_authorized' and original['observation'] is None
        assert await model_attempts._read(manager, manager.db, 'finished', attempt_id, required=False) is None
        messages = await store.load_all_messages(sid)
        await manager.request_cancel(actor, job_id, system_key='stop-after-crash', reason='test')
        job = await manager.get_job(actor, job_id)
        evidence = BudgetUsageEvidence('captured-provider-return', 'trusted-test-collector',
            facts['provider_response_id'] or attempt_id, 'actual', facts['usage'],
            details={'source': 'Actual response captured outside the killed SDK process'})
        receipt = await first.stage_budget_usage_revision(sid, job_id=job_id, attempt_id=attempt_id,
            expected_revision=0, evidence=evidence)
        host.lost_revision = 1
        with pytest.raises(OSError, match='ACK lost'):
            await first.reconcile_budget_usage(sid, job_id=job_id, attempt_id=attempt_id)
        pending = await model_budgets.get_budget(manager, actor, attempt_id)
        await first.shutdown(timeout_s=3)
        host.lost_revision = None
        second = control()
        assert receipt == await second.stage_budget_usage_revision(sid, job_id=job_id, attempt_id=attempt_id,
            expected_revision=0, evidence=evidence)
        ack = await second.reconcile_budget_usage(sid, job_id=job_id, attempt_id=attempt_id)
        assert ack['usage_acknowledged'] and not ack['execution_authority']
        saved = await model_budgets.get_budget(manager, actor, attempt_id)
        assert saved['observation'] is None and 'finished_at_ms' not in saved
        assert saved['submission'] == pending['submission'] and not saved['pending']
        assert saved['context'] == original['context'] and saved['grant'] == original['grant']
        assert await store.load_all_messages(sid) == messages and await manager.get_job(actor, job_id) == job
        assert host.reserve_calls == 0 and len(host.revisions) == 1 and not host.released
        assert all(r._lifecycle._start_task is None for r in controls)
        [row] = await manager.db.fetchall(f'SELECT counters_json FROM {manager.prefix}task_budget_scopes')
        counts = json.loads(row['counters_json'])
        assert counts['used_calls'] == counts['inflight'] == 1
        assert counts['held_calls'] == counts['held_tokens'] == 0
        async with manager._write() as tx:
            raw_job = await manager._authorize(tx, actor, job_id)
            with pytest.raises(JobError):
                await model_attempts.require_quiescent(manager, tx, raw_job)
        record_property('accounting_receipt', json.dumps(receipt))
    finally:
        if process.returncode is None:
            process.kill()
            await process.communicate()
        for runtime in controls:
            await runtime.shutdown(timeout_s=3)
        if store is not None:
            await store.close()
