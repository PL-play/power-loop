"""Actual SDK capture port persists before the owned process is killed."""
import asyncio
import json
import sqlite3
import sys
from dataclasses import asdict

from power_loop import AgentLoopConfig, LLMResponse, LLMService, LLMTokenUsage, SessionStore, StatefulAgentLoop
from power_loop.runtime.jobs import JobManager
from power_loop.runtime.response_capture import ResponseCaptureReceipt
from power_loop.runtime.task_runtime import TaskRuntime
from tests.real._initial_evidence_worker_192 import RecoveryHost, config
from tests.real.test_real_async_boundaries_192 import BoundedProvider


async def main():
    options = json.load(sys.stdin)
    store = await SessionStore.open(options['path'], schema='tasks')
    manager = JobManager(store)
    class Source(LLMService):
        async def complete(self, request, **kwargs):
            print(json.dumps({'kind': 'request_started', 'real_provider_requests': int(options['real'])}), flush=True)
            if options['real']:
                return await BoundedProvider(limit=1).complete(request, **kwargs)
            return LLMResponse(raw_text='42', content_text='42',
                token_usage=LLMTokenUsage(prompt_tokens=7, completion_tokens=3, total_tokens=10))
    class CapturingHost(RecoveryHost):
        async def capture_response(self, capture):
            # Independent disposable test evidence store, never a business DB.
            with sqlite3.connect(options['evidence_path']) as db:
                db.execute('CREATE TABLE IF NOT EXISTS captures (id TEXT PRIMARY KEY, fingerprint TEXT NOT NULL, packet TEXT NOT NULL)')
                prior = db.execute('SELECT fingerprint FROM captures WHERE id=?', (capture.capture_id,)).fetchone()
                if prior and prior[0] != capture.fingerprint:
                    raise ValueError('capture identity conflict')
                db.execute('INSERT OR IGNORE INTO captures VALUES (?,?,?)',
                    (capture.capture_id, capture.fingerprint, json.dumps(asdict(capture))))
            # Only report after actual COMMIT; the parent kills us before ACK
            # and before the SDK invokes normal model observation persistence.
            proof = ResponseCaptureReceipt(capture.capture_id, capture.fingerprint, capture.capture_id)
            print(json.dumps({'kind': 'capture_committed', 'receipt': asdict(proof),
                'reported_tokens': capture.usage.get('total_tokens', 0)}), flush=True)
            await asyncio.Event().wait()
            return proof
    service, host = Source(), CapturingHost()
    host.service = service
    runtime = TaskRuntime(manager, tenant_id='test', principal_id='host', executor_id='original',
        boot_id='original', boot_contract_hash='a' * 64, budget_config=config(host))
    loop = StatefulAgentLoop(llm=service, store=store, task_runtime=runtime,
        config=AgentLoopConfig(delegation_policy='single_level', max_rounds=1, max_tokens=2048))
    sid = await loop.new_session()
    await loop.send('Reply only with the number 42.', sid)
    raise AssertionError('parent must kill worker at the recorded capture boundary')


if __name__ == '__main__':
    asyncio.run(main())
