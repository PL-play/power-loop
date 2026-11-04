"""Host continuation keeps the answer attached to the original logical send."""
import json

import pytest

from power_loop import AgentLoopConfig, LLMResponse, LLMTokenUsage, StatefulAgentLoop, create_default_tool_registry
from power_loop.runtime.budget_contracts import BudgetConfig, LocalBudgetScope
from power_loop.runtime.history_projector import IdentityProjector
from power_loop.runtime.jobs import JobManager
from power_loop.runtime.response_capture import ResponseCaptureReceipt
from power_loop.runtime.task_runtime import TaskRuntime
from tests.unit.test_job_registry_192 import registry_store  # noqa: F401
from tests.unit.test_model_budgets_192 import Host
from tests.unit.test_request_sanitize import StrictModel


def response(text='', calls=None):
    return LLMResponse(raw_text=text, tool_calls=calls or [],
        token_usage=LLMTokenUsage(prompt_tokens=3, completion_tokens=2, total_tokens=5))


def question():
    return {'id': 'question', 'type': 'function', 'function': {
        'name': 'request_user_input', 'arguments': '{"prompt":"Which option?"}'}}


def runtime(store, model, budgeted=False):
    host = Host()
    host.service = model

    async def capture(packet):
        return ResponseCaptureReceipt(packet.capture_id, packet.fingerprint, 'captured')
    host.capture_response = capture
    budget = BudgetConfig('continuation', (LocalBudgetScope('shared', max_calls=20, max_tokens=10000),), host)
    return TaskRuntime(JobManager(store), tenant_id='continuation', principal_id='host',
        executor_id='executor', boot_id='boot', boot_contract_hash='a' * 64,
        budget_config=budget if budgeted else None), host


@pytest.mark.asyncio
@pytest.mark.parametrize('budgeted', [False, True])
@pytest.mark.parametrize('projected', [False, True])
async def test_waiting_answer_and_resume_after_reopen(registry_store, budgeted, projected):  # noqa: F811
    store, connect = registry_store
    first_model = StrictModel([response(calls=[question()])])
    first_runtime, _ = runtime(store, first_model, budgeted)
    config = AgentLoopConfig(delegation_policy='single_level', compactor=None, max_tokens=20,
        history_projector=IdentityProjector() if projected else None)
    registry = create_default_tool_registry(include=['request_user_input'])
    first = StatefulAgentLoop(llm=first_model, store=store, task_runtime=first_runtime,
        tool_registry=registry, config=config)
    sid = await first.new_session()
    waiting = await first.send('ask', sid)
    assert waiting.status == 'waiting_for_input', '\n'.join(str(m.get('content')) for m in await first.get_messages(sid))
    interaction = waiting.pending_interactions[0]['interaction_id']
    await first.aclose()
    reopened = await connect()
    model = StrictModel([response('answered'), response('continued')])
    managed, host = runtime(reopened, model, budgeted)
    loop = StatefulAgentLoop(llm=model, store=reopened, task_runtime=managed,
        tool_registry=registry, config=config)
    try:
        assert (await loop.resume(sid)).pending_interactions[0]['interaction_id'] == interaction
        assert not model.requests
        with pytest.raises(ValueError, match='pending interaction not found'):
            await loop.submit_input(sid, 'wrong-id', 'no')
        answer = await loop.submit_input(sid, interaction, {'choice': 'yes'})
        assert answer.status == 'completed' and answer.final_text == 'answered'
        assert await loop.get_pending(sid) is None
        assert (await loop.resume(sid)).final_text == 'continued'
        rows = await reopened.load_all_messages(sid)
        [tool] = [r for r in rows if r.role == 'tool']
        assert tool.send_index == 1 and json.loads(tool.content) == {'choice': 'yes'}
        assert all(r.send_index == 1 for r in rows)
        jobs = await managed.manager.db.fetchall(f'SELECT job_id FROM {managed.manager.jobs}')
        assert len(jobs) == 3
        if budgeted:
            assert host.reserve_calls == host.apply_calls == 2
    finally:
        await loop.aclose()


@pytest.mark.asyncio
async def test_managed_resume_marks_stale_call_unknown(registry_store):  # noqa: F811
    store, _ = registry_store
    model = StrictModel([response('continued')])
    managed, _ = runtime(store, model)
    loop = StatefulAgentLoop(llm=model, store=store, task_runtime=managed,
        tool_registry=create_default_tool_registry(include=['request_user_input']),
        config=AgentLoopConfig(delegation_policy='single_level', compactor=None))
    try:
        sid = await loop.new_session()
        seq = await store.append_message(sid, role='assistant', tool_calls=[question()])
        await store.set_pending(sid, {'assistant_seq': seq, 'tool_calls': [question()], 'tool_call_ids': ['question']})
        result = await loop.resume(sid)
        assert result.status == 'completed' and not result.pending_interactions
        assert await loop.get_pending(sid) is None
        [tool] = [m for m in model.requests[0] if m['role'] == 'tool']
        assert '结果未知' in tool['content']
    finally:
        await loop.aclose()
