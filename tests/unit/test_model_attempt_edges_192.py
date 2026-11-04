"""Persistent model-call bounds, corrupt evidence and original owner barriers."""
from __future__ import annotations

from dataclasses import replace

import pytest

from power_loop.runtime import model_attempts as attempts
from power_loop.runtime.jobs import JobError

from .test_job_registry_192 import registry_store
from .test_model_attempts_192 import attempt_case, completed, prepare

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]
__all__ = ["attempt_case", "registry_store"]


async def test_active_model_call_limit(attempt_case, monkeypatch):
    c = attempt_case
    await completed(c)
    monkeypatch.setattr(attempts, "_LIMIT", 1)
    with pytest.raises(JobError, match="model_attempt_limit"):
        await prepare(c, nonce="next")

@pytest.mark.parametrize("field,value",[("error_type",False),("error_type",""),("response_id","bad\nidentifier"),("response_id","x"*256),("estimated",1),("status","done"),("usage",{"unknown_dimension":1})])
async def test_invalid_observation_never_creates_a_finished_event(attempt_case,field,value):
    c=attempt_case
    attempt=await prepare(c)
    await attempts.dispatch(attempt)
    args=dict(status="returned",usage={"total_tokens":10},estimated=False)
    args[field]=value
    with pytest.raises(JobError,match="invalid_model_attempt_(observation|usage)"):
        await attempts.finish(attempt,**args)
    assert not await c.manager.db.fetchall(f"SELECT * FROM {c.manager.events} WHERE event_type='model_attempt_finished'")

async def test_known_finished_attempt_refuses_another_dispatch_and_lost_ack_recall(attempt_case):
    c=attempt_case
    attempt=await completed(c)
    with pytest.raises(JobError,match="model_attempt_dispatch_already_admitted"):
        await attempts.dispatch(attempt)
    with pytest.raises(JobError,match="model_attempt_dispatch_conflict"):
        await attempts._recall_dispatch(attempt)
    from power_loop.runtime.control_runtime import current
    assert current(c.binding).get("attempt", attempt.attempt_id)["phase"] == "finished"

async def test_prepared_without_dispatch_has_no_ack_to_recall(attempt_case):
    c=attempt_case
    attempt=await prepare(c)
    assert await attempts._recall_dispatch(attempt) is False

async def test_changed_original_preparation_cannot_authorize_dispatch_or_late_observation(attempt_case):
    c=attempt_case
    attempt=await prepare(c)
    wrong=replace(attempt,prepared={**attempt.prepared,"model":"substituted"})
    with pytest.raises(JobError,match="model_attempt_owner_mismatch"):
        await attempts.dispatch(wrong)
    await attempts.dispatch(attempt)
    with pytest.raises(JobError,match="model_attempt_owner_mismatch"):
        await attempts.finish(wrong,status="returned",usage={},estimated=False)


async def test_new_physical_send_cannot_evade_same_owner_unknown_request(attempt_case):
    c=attempt_case
    sid=await c.store.create_session(parent_session_id=c.sid,metadata={"execution_role":"subagent"})
    child=await c.manager.register_job(c.actor,kind="subagent",parent=c.binding.lease,system_key="original-owner",independent=True)
    bound=await c.manager.bind_session(c.actor,child.job.job_id,sid)
    lease=await c.manager.claim_job(c.actor,child.job.job_id,executor_id="executor",boot_id="boot",expected_version=bound.version)
    actor=c.runtime.actor(sid)
    first=await c.manager.register_job(actor,kind="send",parent=lease,session_id=sid,system_key="first",request={"model_attempt_protocol":1,"send_index":1})
    binding=await c.runtime.claim(actor,first)
    await attempts.enable(binding)
    args=dict(nonce="same-owner-call",round_index=0,retry_index=1,request_hash="a"*64,model="test",max_tokens=100)
    original=await attempts.prepare(binding,**args)
    await attempts.dispatch(original)
    await attempts.finish(original,status="unknown",usage={"total_tokens":11},estimated=True,error_type="TimeoutError")
    await c.manager.close_model_execution(actor,binding.lease,evidence={"exit":"degraded"},resource_state="unknown")
    second=await c.manager.register_job(actor,kind="send",parent=lease,session_id=sid,system_key="second",request={"model_attempt_protocol":1,"send_index":2})
    replacement=await c.runtime.claim(actor,second)
    with pytest.raises(JobError,match="unresolved_prior_attempt"):
        await attempts.enable(replacement)
        await attempts.prepare(replacement,**(args|{"nonce":"replacement-must-not-dispatch"}))
    rows=await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE job_id=? AND event_type='model_attempt_prepared'",(second.job.job_id,))
    assert rows==[]
