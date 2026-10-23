"""Persistent model-call bounds, corrupt evidence and original owner barriers."""
from __future__ import annotations

from dataclasses import replace

import pytest

from power_loop.runtime import model_attempts as attempts
from power_loop.runtime.jobs import JobError
from power_loop.runtime.model_attempt_query import list_attempts

from .test_job_registry_192 import registry_store
from .test_model_attempts_192 import attempt_case, closed, completed, prepare

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]
__all__ = ["attempt_case", "registry_store"]

@pytest.mark.parametrize("raw", ["[]", "{", '{"version":true}', '{"version":2}', '{"version":1,"bad":NaN}'])
async def test_corrupt_serialized_phase_cannot_be_used_as_control_evidence(attempt_case, raw):
    c = attempt_case
    attempt = await completed(c)
    await c.manager.db.execute(f"UPDATE {c.manager.events} SET payload_json=? WHERE event_id=?", (raw, attempts.event_id(c.manager,"prepared",attempt.attempt_id)))
    with pytest.raises(JobError, match="model_attempt_evidence_corrupt"):
        await list_attempts(c.manager,c.actor,c.binding.lease.job_id)
    assert (await c.manager.get_job(c.actor,c.binding.lease.job_id)).supervision_status == "open"

async def test_corrupt_tracking_delivery_marker_refuses_dispatch(attempt_case):
    c=attempt_case
    attempt=await prepare(c)
    await c.manager.db.execute(f"UPDATE {c.manager.events} SET delivery_status='handled' WHERE event_type='model_attempt_tracking'")
    with pytest.raises(JobError,match="model_attempt_evidence_corrupt"):
        await attempts.dispatch(attempt)
    assert not await c.manager.db.fetchall(f"SELECT * FROM {c.manager.events} WHERE event_type='model_attempt_dispatched'")

async def test_new_protocol_no_calls_does_not_treat_lost_tracking_as_legacy(attempt_case):
    c=attempt_case
    await c.manager.db.execute(f"DELETE FROM {c.manager.events} WHERE event_type='model_attempt_tracking'")
    with pytest.raises(JobError,match="model_attempt_evidence_missing"):
        await closed(c)

@pytest.mark.parametrize("case", ["new_admission","phase_bound","prepared_bound","query_bound"])
async def test_phase_and_query_limits_are_enforced_against_real_saved_rows(attempt_case,monkeypatch,case):
    c=attempt_case
    await completed(c)
    if case=="new_admission":
        monkeypatch.setattr(attempts,"_LIMIT",1)
        with pytest.raises(JobError,match="model_attempt_limit"):
            await prepare(c,nonce="next")
        assert len(await c.manager.db.fetchall(f"SELECT * FROM {c.manager.events} WHERE event_type='model_attempt_prepared'"))==1
        return
    await completed(c,await prepare(c,nonce="second"))
    if case=="query_bound":
        from power_loop.runtime import model_attempt_query
        monkeypatch.setattr(model_attempt_query,"_MAX_EVENTS",1)
        error="task_event_scope_too_large"
    else:
        monkeypatch.setattr(attempts,"_LIMIT",1)
        error="model_attempt_limit"
        if case=="prepared_bound":
            await c.manager.db.execute(f"DELETE FROM {c.manager.events} WHERE event_type IN ('model_attempt_dispatched','model_attempt_finished')")
    with pytest.raises(JobError,match=error):
        await list_attempts(c.manager,c.actor,c.binding.lease.job_id)

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
    with pytest.raises(JobError,match="model_attempt_already_finished"):
        await attempts.dispatch(attempt)
    with pytest.raises(JobError,match="model_attempt_dispatch_conflict"):
        await attempts._recall_dispatch(attempt)
    assert len(await c.manager.db.fetchall(f"SELECT * FROM {c.manager.events} WHERE event_type='model_attempt_dispatched'"))==1

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

async def test_attempt_query_requires_send_kind_and_a_whole_item_budget(attempt_case):
    c=attempt_case
    child=await c.manager.register_job(c.actor,kind="subagent",parent=c.binding.lease,system_key="query-kind")
    with pytest.raises(JobError,match="model_attempts_require_send_job"):
        await list_attempts(c.manager,c.actor,child.job.job_id)
    attempt=await prepare(c,model="model-"+"x"*240)
    await attempts.dispatch(attempt)
    await attempts.finish(attempt,status="returned",usage={"total_tokens":10},estimated=False,response_id="r"*255)
    with pytest.raises(JobError,match="query_result_too_small"):
        await list_attempts(c.manager,c.actor,c.binding.lease.job_id,max_chars=1024)

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
    await attempts.enable(replacement)
    with pytest.raises(JobError,match="unresolved_prior_attempt"):
        await attempts.prepare(replacement,**(args|{"nonce":"replacement-must-not-dispatch"}))
    rows=await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE job_id=? AND event_type='model_attempt_prepared'",(second.job.job_id,))
    assert rows==[]
