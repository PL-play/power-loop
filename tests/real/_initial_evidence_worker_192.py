"""Owned test worker: expose the response, then stop before SDK exit persistence."""
import asyncio
import json
import sys

from power_loop import AgentLoopConfig, LLMResponse, LLMService, LLMTokenUsage, SessionStore, StatefulAgentLoop
from power_loop.runtime import model_attempts
from power_loop.runtime.budget_contracts import BudgetConfig, BudgetPlan, LocalBudgetScope
from power_loop.runtime.jobs import JobManager
from power_loop.runtime.task_runtime import TaskRuntime
from tests.real.test_real_async_boundaries_192 import BoundedProvider
from tests.unit.test_budget_initial_evidence_192 import InitialHost


class RecoveryHost(InitialHost):
    async def prepare(self, context, logical_request, *, service):
        assert service is self.service
        return BudgetPlan(16384, {'attempt_id': context['attempt_id'], 'origin': context})


def config(host):
    return BudgetConfig('initial-evidence-host', (
        LocalBudgetScope('shared', max_calls=1, max_tokens=16384, max_inflight=1),), host)


async def main():
    options = json.load(sys.stdin)
    store = await SessionStore.open(options['path'], schema='tasks')
    manager = JobManager(store)
    class BeforeObservation(LLMService):
        async def complete(self, request, **kwargs):
            if options['real']:
                inner = BoundedProvider(limit=1)
                print(json.dumps({'kind': 'request_started', 'real_provider_requests': 1}), flush=True)
                response = await inner.complete(request, **kwargs)
            else:
                print(json.dumps({'kind': 'request_started', 'real_provider_requests': 0}), flush=True)
                response = LLMResponse(raw_text='42', content_text='42',
                    token_usage=LLMTokenUsage(prompt_tokens=7, completion_tokens=3, total_tokens=10))
            [row] = await manager.db.fetchall(f'SELECT attempt_id,job_id FROM {manager.prefix}task_budget_attempts')
            facts = model_attempts.response_observation(response.token_usage, response)
            print(json.dumps({'kind': 'captured_before_exit', **row, 'sid': sid,
                'text': response.raw_text, 'usage': facts['usage'],
                'reported_tokens': response.token_usage.total_tokens or 0,
                'provider_response_id': facts['response_id'], 'real_provider_requests': int(options['real'])}), flush=True)
            await asyncio.Event().wait()  # Parent SIGKILLs this exact owned process.
            raise AssertionError('test boundary must never return')
    service, host = BeforeObservation(), RecoveryHost()
    host.service = service
    runtime = TaskRuntime(manager, tenant_id='test', principal_id='host', executor_id='original',
        boot_id='original', boot_contract_hash='a' * 64, budget_config=config(host))
    loop = StatefulAgentLoop(llm=service, store=store, task_runtime=runtime,
        config=AgentLoopConfig(delegation_policy='single_level', max_rounds=1, max_tokens=1024))
    sid = await loop.new_session()
    await loop.send('Reply only with the number 42.', sid)
    raise AssertionError('worker unexpectedly completed')


if __name__ == '__main__':
    asyncio.run(main())
