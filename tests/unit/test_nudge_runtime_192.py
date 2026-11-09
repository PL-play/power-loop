"""Nudges use the original task, current ownership and immutable message identity."""
import asyncio
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from power_loop import InputRef, JobRef, LLMResponse, LLMTokenUsage, RuntimeRequestError
from power_loop.agent.runtime_store import now_ms
from power_loop.core.agent_context import get_session_id
from power_loop.runtime import child_tasks

from .test_job_registry_192 import registry_store, runtime
from .test_task_runtime_192 import finish, jobs, managed, receipt

__all__ = ['registry_store', 'runtime', 'managed']
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.fixture
async def nudge_case(runtime):
    store, manager, actor, _, supervisor, connect, now = runtime
    sid = await store.create_session(parent_session_id=actor.session_id,
        metadata={"execution_role": "subagent", "standalone_subagent": True})
    reg = await manager.register_job(actor, kind="subagent", session_id=sid,
        parent=supervisor, independent=True, system_key="nudge-target")
    source = await manager.register_job(actor, kind="send", session_id=actor.session_id,
        system_key="nudge-caller", request={"send_index": 1})
    caller = await manager.claim_job(actor, source.job.job_id, executor_id="parent", boot_id="boot", expected_version=1)
    child_actor = replace(actor, session_id=sid)
    target = await manager.claim_job(child_actor, reg.job.job_id, executor_id="child", boot_id="boot", expected_version=reg.job.version)
    seq = await store.append_message(actor.session_id, role="assistant", send_index=1, round_index=0,
        tool_calls=[{"id": "nudge-call", "type": "function", "function": {"name": "spawn_agent", "arguments": "{}"}}])
    origin = await store.message_ref(actor.session_id, seq)
    send = await manager.register_job(child_actor, kind="send", session_id=sid, parent=target,
                                     system_key="receiver-send", request={"send_index": 1})
    receiver = await manager.claim_job(child_actor, send.job.job_id, executor_id="child", boot_id="boot", expected_version=1)

    async def accept(**changes):
        params = {"caller": caller, "call_ref": origin, "tool_call_id": "nudge-call", "session_id": sid, "message": "NUDGED-42"}
        params.update(changes)
        return await manager.accept_nudge(actor, target, **params)

    from power_loop.runtime.result_consumption import finish as finish_consumption
    from power_loop.runtime.result_consumption import observe
    from power_loop.runtime.task_runtime import Execution
    binding = Execution(SimpleNamespace(manager=manager), child_actor, receiver)
    binding.resource_evidence["_control_source"] = await manager._row(manager.db, receiver.job_id)

    async def consume(operation, *, complete=True):
        rows = await store.inbox_claim(sid)
        seq = await store.append_message(sid, role="user", content="NUDGED-42", send_index=1,
            inbox={"claim_token": rows[0]["claim_token"], "ids": [r["id"] for r in rows]})
        await observe(binding, [{"seq": seq, "role": "user", "content": "NUDGED-42",
            "item_ids": [r["item_id"] for r in rows]}], round_index=0, send_index=1)
        await store.append_message(sid, role="assistant", content="NUDGED-42", send_index=1, round_index=0)
        if complete:
            await finish_consumption(binding)
            await manager.close_model_execution(child_actor, receiver, evidence={"exit": "completed"})
        return tuple(binding.consumed_inputs)

    return SimpleNamespace(store=store, manager=manager, actor=actor, caller=caller, supervisor=supervisor, target=target, sid=sid,
        child_actor=child_actor, receiver=receiver, origin=origin, accept=accept, consume=consume, binding=binding, connect=connect, now=now)


async def issue_nudge(fixture, monkeypatch, *, message="NUDGED-42", repeat=False):
    _, _, _, loop, model, parent, _ = fixture
    accepted = await receipt(fixture)
    original = model.complete
    sent = False

    async def complete(request, **kwargs):
        nonlocal sent
        if get_session_id() == parent and not sent:
            sent = True
            model.root_calls += 1
            return LLMResponse(raw_text="", tool_calls=[{"id": "actual-nudge", "type": "function", "function": {
                "name": "spawn_agent", "arguments": json.dumps({"action": "send", "sub_session": accepted["session_id"], "message": message})}}],
                token_usage=LLMTokenUsage(prompt_tokens=7, completion_tokens=3, total_tokens=10))
        result = await original(request, **kwargs)
        if get_session_id() != parent and "NUDGED-42" in json.dumps(request.messages, default=str):
            result.raw_text = "NUDGED-42"
        return result

    monkeypatch.setattr(model, "complete", complete)
    receipts = []
    if repeat:
        original_nudge = child_tasks.send_input

        async def twice(*args, **kwargs):
            receipts.append(await original_nudge(*args, **kwargs))
            receipts.append(await original_nudge(*args, **kwargs))
            return receipts[0]

        monkeypatch.setattr(child_tasks, "send_input", twice)
    result = await loop.send("add a correction to the same child")
    return accepted, result, receipts


@pytest.mark.parametrize('managed', [{'capacity':2}], indirect=True)
@pytest.mark.parametrize('message', ['NUDGED-42', '  NUDGED-42\n    keep indentation\n'])
@pytest.mark.parametrize('repeat', [False,True])
async def test_actual_nudge_is_deduplicated_preserves_text_and_never_replaces_running_child(managed, monkeypatch, message, repeat):
    store, manager, runtime, agent, model, sid, connect=managed
    await agent.send('delegate')
    await asyncio.wait_for(model.entered.wait(),5)
    accepted,result,repeated=await issue_nudge(managed,monkeypatch,message=message,repeat=repeat)
    assert result.status=='completed'
    tools=[m for m in await store.load_all_messages(sid) if m.role=='tool']
    sent=json.loads(tools[-1].content)
    assert sent['target']['job_id']==accepted['job_id']
    assert sent['input_ref']['session_id']==accepted['session_id']
    if repeat:
        assert repeated[0]['input_ref']==repeated[1]['input_ref']
        assert repeated[0]['newly_accepted'] and not repeated[1]['newly_accepted']
    queued=await agent.runtime.client.inputs.input(agent.runtime.client.inputs.db,InputRef(**sent['input_ref']))
    assert json.loads(queued.row['content'])==message
    assert (await manager.get_job(runtime.actor(sid),accepted['job_id'])).result is None
    await finish(managed)
    saved=await agent.runtime.client.wait(JobRef(**accepted['job_ref']),actor=agent.actor,deadline_ms=now_ms()+1000)
    assert saved.status=='completed' and saved.final_text=='NUDGED-42'
    assert saved.job_ref.generation==1
    assert saved.usage['total_tokens']==20 and model.child_calls==2
    assert len([j for j in await jobs(managed) if j.kind=='subagent'])==1
    assert sum(m.name=='follow_up' and message.strip() in m.content for m in await store.load_all_messages(accepted['session_id']))==1


@pytest.mark.parametrize('barrier',['foreign_actor','old_generation','cancelled','closed'])
async def test_control_refuses_wrong_actor_or_generation_and_does_not_reopen_closed_child(managed, barrier):
    _,_,_,agent,model,_,_=managed
    await agent.send('delegate')
    await asyncio.wait_for(model.entered.wait(),5)
    accepted=await receipt(managed)
    ref=(await agent.runtime.client.inspect(JobRef(**accepted['job_ref']),actor=agent.actor)).job_ref
    actor=agent.actor
    if barrier=='foreign_actor':
        actor=replace(actor,principal_id='foreign')
    elif barrier=='old_generation':
        ref=replace(ref,generation=0)
    elif barrier=='cancelled':
        await agent.runtime.client.cancel(ref,actor=actor)
        await agent.drain()
    else:
        await finish(managed)
    with pytest.raises(RuntimeRequestError):
        await agent.runtime.client.send(ref,'forbidden correction',message_id='wrong',actor=actor)
    model.release.set()
    await agent.drain()
    assert model.child_calls==1
    assert not any('forbidden correction' in (m.content or '') for m in await agent.runtime.store.load_all_messages(accepted['session_id']))


@pytest.mark.parametrize('message',['', None, 42, '   '])
async def test_tool_invalid_nudge_is_an_explicit_error_without_input(managed, monkeypatch, message):
    store,_,_,agent,_,sid,_=managed
    await agent.send('delegate')
    await finish(managed)
    accepted,_,_=await issue_nudge(managed,monkeypatch,message=message)
    tool=[m for m in await store.load_all_messages(sid) if m.role=='tool'][-1]
    assert 'error' in tool.content.lower()
    assert len(await store.list_children(sid))==1
    assert not (await store.inbox_counts(accepted['session_id']))['pending']
