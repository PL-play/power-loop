"""Persistent task directory authorization, bounds and honest active semantics."""
from __future__ import annotations

import base64
import json
from dataclasses import replace

import pytest

from power_loop.runtime import task_query
from power_loop.runtime.jobs import JobActor, JobError, JobManager, _json

from .test_job_registry_192 import call, child, claimed, registry_store, runtime

__all__ = ["registry_store", "runtime"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def tree(r):
    _, m, actor, root, proof, _, now = r
    one, p1 = await claimed(m, actor, "background_command", proof, "one")
    now[0] += 1
    two, p2 = await claimed(m, actor, "background_command", p1, "two")
    now[0] += 1
    other = await child(r, key="other", kind="timer")
    return root, one, two, other, p2


async def test_directory_persists_across_store_reopen_and_never_dispatches(runtime):
    store, m, actor, _, _, connect, _ = runtime
    root, one, two, other, _ = await tree(runtime)
    before = await m.db.fetchall(f"SELECT * FROM {m.events} ORDER BY event_id")
    reader = JobManager(await connect(), now_ms=m.now)
    result = await task_query.list_jobs(reader, actor)
    assert {n["job_id"] for n in result["nodes"]} == {x.job.job_id for x in (root, one, two, other)}
    assert result["matched_count"] == 4 and not result["truncated"] and result["next_cursor"] is None
    assert await m.db.fetchall(f"SELECT * FROM {m.events} ORDER BY event_id") == before
    assert not await store.load_all_messages(actor.session_id)


async def test_result_get_returns_saved_excerpt_and_distinct_call_receipt_ids(runtime):
    store, m, actor, _, proof, _, _ = runtime
    ref = await call(store, actor)
    reg = await m.register_job(actor, kind="background_tool", parent=proof, call_ref=ref, tool_call_id="call-1")
    seq = await store.append_message(actor.session_id, role="tool", tool_call_id="call-1", content="accepted")
    receipt = await store.message_ref(actor.session_id, seq)
    await m.bind_receipt(actor, reg.operation.operation_id, receipt, caller=proof)
    lease = await m.claim_job(actor, reg.job.job_id, executor_id="SECRET-EXECUTOR", boot_id="SECRET-BOOT", expected_version=1)
    await m.record_execution_result(actor, lease, outcome="succeeded", result={"final_text": "Saved output", "system_prompt": "PRIVATE-PROMPT"}, resource_state="stopped")
    result = await task_query.get_job(m, actor, reg.job.job_id)
    node = result["node"]
    assert node["output_excerpt"] == "Saved output"
    assert node["call_ref"] == ref.to_dict() and node["receipt_ref"] == receipt.to_dict()
    assert node["call_ref"]["id"] != node["receipt_ref"]["id"]
    assert node["result_ref"]["event_id"] == "result_"+reg.job.job_id
    assert node["result_ref"]["store_id"] == store.store_id and node["billing_status"] == "unavailable"
    assert all(x not in _json(result) for x in ("SECRET-EXECUTOR", "SECRET-BOOT", "PRIVATE-PROMPT"))


async def test_terminal_outcome_can_remain_active_and_unknown_is_never_stopped(runtime):
    _, m, actor, _, _, _, _ = runtime
    reg = await child(runtime, independent=True)
    lease = await m.claim_job(actor, reg.job.job_id, executor_id="worker", boot_id="boot", expected_version=1)
    await m.record_execution_result(actor, lease, outcome="failed", result={"final_text": "failed"}, resource_state="unknown")
    active = await task_query.list_jobs(m, actor, status="active", kinds=["subagent"])
    terminal = await task_query.list_jobs(m, actor, status="terminal")
    node = next(n for n in active["nodes"] if n["job_id"] == reg.job.job_id)
    assert node["active_reason"] == ["resource_unknown", "result_pending_delivery", "supervision_open"]
    assert node["outcome"] == "failed" and not node["controls"]["resource_stopped"]
    assert reg.job.job_id in {n["job_id"] for n in terminal["nodes"] if not n["context_only"]}


async def test_filters_keep_ancestors_as_context_without_spending_match_limit(runtime):
    _, m, actor, _, _, _, _ = runtime
    root, one, two, _, _ = await tree(runtime)
    page = await task_query.list_jobs(m, actor, root_job_id=one.job.job_id, kinds=["background_command"], limit=1)
    assert page["matched_count"] == 1 and len(page["nodes"]) == 2
    assert page["nodes"][0]["job_id"] == one.job.job_id and page["nodes"][0]["context_only"]
    assert page["nodes"][1]["job_id"] == two.job.job_id and not page["nodes"][1]["context_only"]
    assert root.job.job_id not in {n["job_id"] for n in page["nodes"]}
    own = await task_query.list_jobs(m, actor, root_job_id=one.job.job_id, scope="self", max_depth=0)
    assert own["matched_count"] == 1 and own["nodes"][0]["job_id"] == one.job.job_id
    assert not own["truncated"] and not own["tree_truncated"]


async def test_cursor_bounds_new_inserts_and_query_identity(runtime):
    _, m, actor, _, _, _, now = runtime
    await tree(runtime)
    page = await task_query.list_jobs(m, actor, limit=1)
    seen = [n["job_id"] for n in page["nodes"] if not n["context_only"]]
    now[0] += 1
    new = await child(runtime, key="new")
    while page["next_cursor"]:
        page = await task_query.list_jobs(m, actor, limit=1, cursor=page["next_cursor"])
        seen.extend(n["job_id"] for n in page["nodes"] if not n["context_only"])
    assert len(seen) == len(set(seen)) == 4 and new.job.job_id not in seen
    first = await task_query.list_jobs(m, actor, limit=1)
    with pytest.raises(JobError, match="query_cursor_mismatch"):
        await task_query.list_jobs(m, actor, status="all", cursor=first["next_cursor"])
    with pytest.raises(JobError, match="query_cursor_mismatch"):
        await task_query.list_jobs(m, replace(actor, principal_id="stranger"), cursor=first["next_cursor"])


@pytest.mark.parametrize("bad", ["tenant", "principal", "session"])
async def test_unrelated_actor_cannot_read_get_or_root(runtime, bad):
    store, m, actor, root, _, _, _ = runtime
    reg = await child(runtime)
    intruder = replace(actor, **{bad+"_id": await store.create_session() if bad == "session" else "stranger"})
    with pytest.raises(JobError, match="access_denied"):
        await task_query.get_job(m, intruder, reg.job.job_id)
    with pytest.raises(JobError, match="access_denied"):
        await task_query.list_jobs(m, intruder, root_job_id=root.job.job_id)
    assert (await task_query.list_jobs(m, intruder))["nodes"] == []


async def test_child_actor_cannot_expand_to_ancestor_or_sibling(runtime):
    store, m, actor, root, _, _, _ = runtime
    reg = await child(runtime)
    sid = await store.create_session(parent_session_id=actor.session_id, metadata={"execution_role": "subagent"})
    await m.bind_session(actor, reg.job.job_id, sid)
    sibling = await child(runtime, key="sibling")
    leaf = JobActor(actor.tenant_id, actor.principal_id, sid)
    assert (await task_query.list_jobs(m, leaf))["nodes"][0]["job_id"] == reg.job.job_id
    for target in (root.job.job_id, sibling.job.job_id):
        with pytest.raises(JobError, match="access_denied"):
            await task_query.list_jobs(m, leaf, root_job_id=target)


@pytest.mark.parametrize("fault", ["cycle", "missing_parent", "cross_tenant"])
async def test_bad_lineage_fails_closed_without_an_empty_success(runtime, fault):
    _, m, actor, root, _, _, _ = runtime
    reg = await child(runtime)
    if fault == "cycle":
        await m.db.execute(f"UPDATE {m.jobs} SET parent_job_id=? WHERE job_id=?", (reg.job.job_id, root.job.job_id))
    elif fault == "missing_parent":
        await m.db.execute(f"UPDATE {m.jobs} SET parent_job_id='missing' WHERE job_id=?", (reg.job.job_id,))
    else:
        await m.db.execute(f"UPDATE {m.jobs} SET tenant_id='stranger' WHERE job_id=?", (reg.job.job_id,))
    with pytest.raises(JobError, match="lineage|access_denied"):
        await task_query.list_jobs(m, actor)


async def test_depth_and_scan_limits_are_explicit(runtime, monkeypatch):
    _, m, actor, root, _, _, _ = runtime
    await tree(runtime)
    shallow = await task_query.list_jobs(m, actor, root_job_id=root.job.job_id, max_depth=0)
    assert shallow["truncated"] and shallow["tree_truncated"] and shallow["matched_count"] == 0
    assert shallow["next_cursor"] is None
    monkeypatch.setattr(task_query, "_SCAN_LIMIT", 2)
    with pytest.raises(JobError, match="query_scope_too_large"):
        await task_query.list_jobs(m, actor)
    bounded = await task_query.list_jobs(m, actor, root_job_id=root.job.job_id)
    assert bounded["tree_truncated"] and bounded["truncated"]
    assert len(bounded["nodes"]) <= 2 and bounded["matched_count"] <= 1


async def test_current_status_filter_time_and_output_limits(runtime):
    _, m, actor, _, _, _, now = runtime
    _, _, two, other, _ = await tree(runtime)
    page = await task_query.list_jobs(m, actor, status=["queued"], created_after=now[0], created_before=now[0], limit=1)
    assert {n["job_id"] for n in page["nodes"] if not n["context_only"]} == {other.job.job_id}
    assert two.job.job_id not in {n["job_id"] for n in page["nodes"]}
    lease = await m.claim_job(actor, other.job.job_id, executor_id="worker", boot_id="boot", expected_version=1)
    await m.record_execution_result(actor, lease, outcome="succeeded", result={"final_text": "X" * 400}, resource_state="stopped")
    with pytest.raises(JobError, match="query_result_too_small"):
        await task_query.get_job(m, actor, other.job.job_id, max_chars=1024)
    page = await task_query.list_jobs(m, actor, max_chars=3000)
    assert len(_json(page)) <= 3000 and page["truncated"] and page["next_cursor"]


@pytest.mark.parametrize("kwargs", [
    {"limit": True}, {"limit": 0}, {"limit": 101}, {"max_chars": 1023}, {"max_chars": 32001},
    {"max_depth": True}, {"max_depth": 64}, {"root_job_id": 123}, {"scope": "self"}, {"scope": "all"},
    {"kinds": []}, {"kinds": ["unknown"]}, {"status": ["unknown"]}, {"status": []}, {"status": True},
    {"created_after": -1}, {"created_after": 2, "created_before": 1}, {"billing_status": "settled"},
    {"cursor": "not-json"}, {"cursor": 1},
])
async def test_query_parameter_errors_never_mutate_state(runtime, kwargs):
    _, m, actor, _, _, _, _ = runtime
    before = await m.db.fetchall(f"SELECT * FROM {m.jobs}")
    with pytest.raises(JobError):
        await task_query.list_jobs(m, actor, **kwargs)
    assert await m.db.fetchall(f"SELECT * FROM {m.jobs}") == before


async def test_malformed_cursor_shape_and_result_data_are_explicit_errors(runtime):
    _, m, actor, root, _, _, _ = runtime
    bad = base64.urlsafe_b64encode(json.dumps({"query": "x" * 64, "membership": "y" * 64,
        "upper": [True, "id"], "after": [0, "id"], "snapshot_at": 0}).encode()).decode()
    with pytest.raises(JobError, match="invalid_query_cursor"):
        await task_query.list_jobs(m, actor, cursor=bad)
    await m.db.execute(f"UPDATE {m.jobs} SET result_json='[]' WHERE job_id=?", (root.job.job_id,))
    with pytest.raises(JobError, match="query_result_corrupt"):
        await task_query.get_job(m, actor, root.job.job_id)


async def test_same_timestamp_insert_below_upper_requires_snapshot_restart(runtime):
    _, m, actor, _, _, _, _ = runtime
    await tree(runtime)
    first = await task_query.list_jobs(m, actor, limit=1)
    saved = task_query._decode(first["next_cursor"])
    # Model a frozen/coarse wall clock: an actually admitted new task has an
    # earlier timestamp than the cursor bound. No membership is fabricated.
    new = await child(runtime, key="clock-drift")
    await m.db.execute(f"UPDATE {m.jobs} SET created_at=? WHERE job_id=?", (saved["upper"][0] - 1, new.job.job_id))
    with pytest.raises(JobError, match="query_snapshot_changed"):
        await task_query.list_jobs(m, actor, limit=1, cursor=first["next_cursor"])


@pytest.mark.parametrize("fault", ["store", "extra_field", "role"])
async def test_message_pointers_cannot_leak_unvalidated_payload_or_cross_store(runtime, fault):
    store, m, actor, _, proof, _, _ = runtime
    ref = await call(store, actor)
    reg = await m.register_job(actor, kind="background_tool", parent=proof, call_ref=ref, tool_call_id="call-1")
    data = ref.to_dict()
    if fault == "store":
        data["store_id"] = "other-store"
    elif fault == "role":
        data["role"] = "tool"
    else:
        data["system_prompt"] = "PRIVATE-CONTENT"
    await m.db.execute(f"UPDATE {m.events} SET call_ref_json=? WHERE event_id=?", (json.dumps(data), reg.operation.operation_id))
    with pytest.raises(JobError, match="query_message_ref_corrupt"):
        await task_query.get_job(m, actor, reg.job.job_id)


async def test_directory_reads_immutable_output_instead_of_mutable_result_cache(runtime):
    _, m, actor, _, _, _, _ = runtime
    reg = await child(runtime, independent=True)
    lease = await m.claim_job(actor, reg.job.job_id, executor_id="worker", boot_id="boot", expected_version=1)
    await m.record_execution_result(actor, lease, outcome="succeeded", result={"final_text": "Actual output"}, resource_state="stopped")
    await m.db.execute(f"UPDATE {m.jobs} SET result_json=? WHERE job_id=?", (json.dumps({"final_text": "Tampered cache"}), reg.job.job_id))
    assert (await task_query.get_job(m, actor, reg.job.job_id))["node"]["output_excerpt"] == "Actual output"
