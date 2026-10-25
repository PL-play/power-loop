"""Original completed managed sends survive reopen without a new provider call."""
import json
import tracemalloc
from dataclasses import asdict, replace

import pytest

from power_loop.agent.stateful_loop import StatefulResult
from power_loop.runtime import send_receipt
from power_loop.runtime.jobs import JobError, JobManager, _hash

from .test_job_registry_192 import registry_store
from .test_root_context_192 import NAMESPACE, root_case

__all__ = ["registry_store", "root_case"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def original(c):
    source = {"namespace": NAMESPACE, "payload": {"origin_run_id": 101}}
    with c.runtime.root_context(c.sid, **source):
        result = await c.loop.send("original input", c.sid)
    return c.model.calls[-1], result, source


async def test_completed_send_reopens_same_full_result_and_raw_reference(root_case):
    c = root_case
    job, result, source = await original(c)
    first = await send_receipt.read(c.manager, c.actor, job, expected_host_context=source)
    assert first["result"] == asdict(result)
    assert first["result"]["usage"]["total_tokens"] == 10
    assert first["last_assistant_ref"]["id"] and first["last_assistant_ref"]["seq"]
    assert all(first[k] is False for k in ("execution_authority", "resource_stop_confirmed", "billing_settled"))
    reopened = await c.connect()
    try:
        assert await send_receipt.read(JobManager(reopened), c.actor, job, expected_host_context=source) == first
    finally:
        await reopened.close()
    assert len(c.model.calls) == 1
    first["result"]["usage"]["total_tokens"] = 999
    assert (await send_receipt.read(c.manager, c.actor, job))["result"]["usage"]["total_tokens"] == 10


@pytest.mark.parametrize("fault", ["hash", "result", "usage", "generation", "reference", "scope", "missing"])
async def test_missing_or_conflicting_original_result_never_becomes_replay_authority(root_case, fault):
    c = root_case
    job, _, _ = await original(c)
    [event] = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.events} WHERE event_type='model_closed'")
    payload = json.loads(event["payload_json"])
    evidence = payload["evidence"]
    receipt = evidence[send_receipt.KEY]
    if fault == "hash":
        receipt["result_hash"] = "0" * 64
    elif fault == "result":
        receipt["result"]["rounds"] = True
    elif fault == "usage":
        receipt["result"]["usage"]["total_tokens"] += 1
        receipt["result_hash"] = _hash(receipt["result"])
    elif fault == "generation":
        payload["generation"] += 1
    elif fault == "reference":
        evidence["last_assistant_ref"]["seq"] = "999999"
    elif fault == "scope":
        evidence["session_id"] = "foreign"
    else:
        del evidence[send_receipt.KEY]
    await c.manager.db.execute(f"UPDATE {c.manager.events} SET payload_json=? WHERE event_id=?",
        (json.dumps(payload), event["event_id"]))
    with pytest.raises(JobError, match="send_receipt"):
        await send_receipt.read(c.manager, c.actor, job)
    assert len(c.model.calls) == 1


async def test_host_context_and_actor_cannot_relabel_original_send(root_case):
    c = root_case
    job, _, source = await original(c)
    with pytest.raises(JobError, match="context_mismatch"):
        await send_receipt.read(c.manager, c.actor, job,
            expected_host_context={**source, "payload": {"origin_run_id": 102}})
    with pytest.raises(JobError, match="access_denied"):
        await send_receipt.read(c.manager, replace(c.actor, tenant_id="foreign"), job)
    assert len(c.model.calls) == 1


async def test_open_provider_or_missing_exit_is_explicit(root_case):
    c = root_case
    async def before(binding):
        with pytest.raises(JobError, match="exit_unavailable"):
            await send_receipt.read(c.manager, c.actor, binding.lease.job_id)
    c.model.before = before
    job, _, _ = await original(c)
    await c.manager.db.execute(f"UPDATE {c.manager.jobs} SET resource_state='unknown' WHERE job_id=?", (job,))
    with pytest.raises(JobError, match="model_exit_unconfirmed"):
        await send_receipt.read(c.manager, c.actor, job)


async def test_utf8_preflight_rejects_oversized_original_exit(root_case):
    c = root_case
    job, _, _ = await original(c)
    await c.manager.db.execute(f"UPDATE {c.manager.events} SET payload_json=? WHERE event_id=?",
        (json.dumps({"wide": "汉" * 3000}, ensure_ascii=False), "model_closed_" + job))
    with pytest.raises(JobError, match="too_large"):
        await send_receipt.read(c.manager, c.actor, job, max_bytes=8000)


async def test_capture_is_a_snapshot_and_invalid_or_oversized_evidence_is_explicit():
    result = StatefulResult(session_id="original", status="completed", usage={"calls": 1})
    snapshot = send_receipt.capture(result)
    result.usage["calls"] = 2
    assert snapshot["result"]["usage"]["calls"] == 1
    with pytest.raises(JobError, match="result_required"):
        send_receipt.capture(object())
    with pytest.raises(JobError, match="too_large"):
        send_receipt.capture(StatefulResult(session_id="original", status="completed", final_text="汉" * 400000))


async def test_receipt_capture_failure_preserves_original_success_not_retry(root_case, monkeypatch):
    c = root_case
    def unavailable(result):
        raise JobError("send_receipt_too_large")
    monkeypatch.setattr(send_receipt, "capture", unavailable)
    job, result, _ = await original(c)
    assert result.status == "completed" and len(c.model.calls) == 1
    with pytest.raises(JobError, match="unavailable_or_corrupt"):
        await send_receipt.read(c.manager, c.actor, job)


@pytest.mark.parametrize("fault", ["continuation", "pending_tools", "attempt_snapshot", "response_id", "response_send"])
async def test_continuation_or_uncertain_evidence_cannot_be_used_as_completed_result(root_case, fault):
    c = root_case
    job, _, _ = await original(c)
    [event] = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.events} WHERE event_type='model_closed'")
    payload = json.loads(event["payload_json"])
    evidence = payload["evidence"]
    receipt = evidence[send_receipt.KEY]
    if fault == "continuation":
        receipt["result"]["status"] = evidence["exit"] = "waiting"
    elif fault == "pending_tools":
        receipt["result"]["pending_tool_calls"] = [{"id": "not-handled"}]
    elif fault == "attempt_snapshot":
        evidence["model_attempt_snapshot"]["phases"] = []
    elif fault == "response_id":
        evidence["last_assistant_ref"]["id"] = "999999"
    else:
        evidence["last_assistant_ref"]["send_index"] = "999999"
    receipt["result_hash"] = _hash(receipt["result"])
    await c.manager.db.execute(f"UPDATE {c.manager.events} SET payload_json=? WHERE event_id=?",
        (json.dumps(payload), event["event_id"]))
    with pytest.raises(JobError, match="requires_continuation|snapshot_mismatch|response_mismatch"):
        await send_receipt.read(c.manager, c.actor, job)
    assert len(c.model.calls) == 1


async def test_later_send_does_not_replace_original_result_or_reference(root_case):
    c = root_case
    job, _, source = await original(c)
    saved = await send_receipt.read(c.manager, c.actor, job, expected_host_context=source)
    with c.runtime.root_context(c.sid, namespace=NAMESPACE, payload={"origin_run_id": 102}):
        await c.loop.send("second independent input", c.sid)
    assert len(c.model.calls) == 2
    assert await send_receipt.read(c.manager, c.actor, job, expected_host_context=source) == saved


async def test_raw_reference_does_not_materialize_unrelated_large_message_body(root_case):
    c = root_case
    job, _, _ = await original(c)
    receipt = await send_receipt.read(c.manager, c.actor, job)
    await c.manager.db.execute(f"UPDATE {c.store.t.messages} SET content=? WHERE session_id=? AND seq=?",
        ("x" * (16 * 1024 * 1024), c.sid, int(receipt["last_assistant_ref"]["seq"])))
    tracemalloc.start()
    try:
        assert await send_receipt.read(c.manager, c.actor, job) == receipt
        _, peak = tracemalloc.get_traced_memory()
        assert peak < 8 * 1024 * 1024, "Reference lookup materialized an unrelated 16MiB message body"
    finally:
        tracemalloc.stop()
