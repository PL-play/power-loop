"""Host continuation keeps the answer attached to the original logical send."""
import asyncio
import json

import pytest

from power_loop import (
    AgentLoopConfig,
    LLMResponse,
    LLMTokenUsage,
    ProjectedRepresentation,
    RuntimeRequestError,
    VerbatimRepresentation,
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
@pytest.mark.parametrize('budgeted', [False, True])
@pytest.mark.parametrize('projected', [False, True])
async def test_waiting_answer_and_resume_after_reopen(registry_store, budgeted, projected, runtime_agent):  # noqa: F811
    store, connect = registry_store
    first_model = StrictModel([response(calls=[question()])])
    first_runtime, _ = runtime(store, first_model, budgeted)
    config = AgentLoopConfig(delegation_policy='single_level', compactor=None, max_tokens=20,
        representation=ProjectedRepresentation() if projected else VerbatimRepresentation())
    registry = create_default_tool_registry(include=['request_user_input'])
    first = await runtime_agent(model=first_model, store=store, task_runtime=first_runtime, registry=registry, config=config)
    sid = first.target.session_id
    receipt = await first.runtime.client.send(first.target, 'ask', message_id='ask', actor=first.actor)
    async with asyncio.timeout(10):
        while True:
            status = await first.runtime.client.inspect(receipt.input_ref, actor=first.actor)
            if status.state == 'waiting_input':
                break
            await asyncio.sleep(.01)
    pending = (await store.get_state(sid)).pending
    interaction = pending['pending_interactions'][0]['interaction_id']
    await first.aclose()
    reopened = await connect()
    model = StrictModel([response('answered'), response('continued')])
    managed, host = runtime(reopened, model, budgeted)
    agent = await runtime_agent(model=model, store=reopened, task_runtime=managed, registry=registry, config=config, target=first.target)
    assert (await agent.runtime.client.inspect(receipt.input_ref, actor=agent.actor)).state == 'waiting_input'
    assert not model.requests
    args = dict(wait_round=pending['round_index'], value={'choice':'yes'}, actor=agent.actor)
    with pytest.raises(RuntimeRequestError, match='interaction_expired'):
        await agent.runtime.client.answer(status.job_ref, interaction_id='wrong-id', message_id='wrong', **args)
    await agent.runtime.client.answer(status.job_ref, interaction_id=interaction, message_id='answer', **args)
    from power_loop.agent.runtime_store import now_ms
    result = await agent.runtime.client.wait(receipt.input_ref, actor=agent.actor, deadline_ms=now_ms()+10000)
    assert result.status == 'completed' and result.final_text == 'answered'
    assert result.input_ref == receipt.input_ref and result.job_ref == status.job_ref
    assert (await reopened.get_state(sid)).pending is None
    assert (await agent.send('Continue explicitly', message_id='continue')).final_text == 'continued'
    rows = await reopened.load_all_messages(sid)
    [tool] = [row for row in rows if row.role == 'tool']
    assert tool.send_index == 1 and json.loads(tool.content) == {'choice':'yes'}
    assert [row.send_index for row in rows if row.role == 'user'] == [1, 2]
    if budgeted:
        assert host.reserve_calls == host.apply_calls == 2


@pytest.mark.asyncio
async def test_new_input_marks_stale_call_unknown(registry_store, runtime_agent):
    store, _ = registry_store
    model = StrictModel([response('continued')])
    managed, _ = runtime(store, model)
    agent = await runtime_agent(model=model, store=store, task_runtime=managed,
        registry=create_default_tool_registry(include=['request_user_input']), config=AgentLoopConfig(compactor=None))
    sid = agent.target.session_id
    seq = await store.append_message(sid, role='assistant', tool_calls=[question()])
    await store.set_pending(sid, {'assistant_seq':seq, 'tool_calls':[question()], 'tool_call_ids':['question']})
    result = await agent.send('Continue explicitly')
    assert result.status == 'completed'
    assert (await store.get_state(sid)).pending is None
    [tool] = [row for row in model.requests[0] if row['role'] == 'tool']
    assert '结果未知' in tool['content']
