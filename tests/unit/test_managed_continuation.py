"""Host continuation keeps the answer attached to the original logical send."""

import pytest

from power_loop import (
    AgentLoopConfig,
    LLMResponse,
    LLMTokenUsage,
    create_default_tool_registry,
)
from power_loop.runtime.budget_contracts import BudgetConfig, LocalBudgetScope
from power_loop.runtime.jobs import JobManager
from power_loop.runtime.task_runtime import TaskRuntime
from tests.unit.test_job_registry_192 import registry_store
from tests.unit.test_model_budgets_192 import Host
from tests.unit.test_request_sanitize import StrictModel

__all__ = ["registry_store"]

def response(text='', calls=None):
    return LLMResponse(raw_text=text, tool_calls=calls or [],
        token_usage=LLMTokenUsage(prompt_tokens=3, completion_tokens=2, total_tokens=5))


def question():
    return {'id': 'question', 'type': 'function', 'function': {
        'name': 'request_user_input', 'arguments': '{"prompt":"Which option?"}'}}


def runtime(store, model, budgeted=False):
    host = Host()
    host.service = model

    budget = BudgetConfig('continuation', (LocalBudgetScope('shared', max_calls=20, max_tokens=10000),), host)
    return TaskRuntime(JobManager(store), tenant_id='continuation', principal_id='host',
        executor_id='executor', boot_id='boot', boot_contract_hash='a' * 64,
        budget_config=budget if budgeted else None), host


@pytest.mark.asyncio
async def test_new_input_marks_stale_call_unknown(registry_store, runtime_agent):
    store, _ = registry_store
    model = StrictModel([response('continued')])
    managed, _ = runtime(store, model)
    agent = await runtime_agent(model=model, store=store, task_runtime=managed,
        registry=create_default_tool_registry(include=['current_time']), config=AgentLoopConfig(compactor=None))
    sid = agent.target.session_id
    seq = await store.append_message(sid, role='assistant', tool_calls=[question()])
    await store.set_pending(sid, {'assistant_seq':seq, 'tool_calls':[question()], 'tool_call_ids':['question']})
    result = await agent.send('Continue explicitly')
    assert result.status == 'completed'
    assert (await store.get_state(sid)).pending is None
    [tool] = [row for row in model.requests[0] if row['role'] == 'tool']
    assert '结果未知' in tool['content']
