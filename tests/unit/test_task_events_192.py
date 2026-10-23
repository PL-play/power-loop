"""Canonical results and each operation's real references are read-only views."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from power_loop.core.agent_context import reset_current_loop, reset_session_id, set_current_loop, set_session_id
from power_loop.runtime import task_control, task_events
from power_loop.runtime.env import RuntimeEnv, runtime_env_context
from power_loop.runtime.jobs import JobActor, JobError, JobManager
from power_loop.runtime.result_revisions import result_id
from power_loop.tools.jobs import run_jobs

from .test_job_registry_192 import registry_store, runtime
from .test_nudge_runtime_192 import nudge_case
from .test_workflow_restore_runtime_192 import managed_workflow, native_restore

__all__ = ["registry_store", "runtime", "nudge_case", "managed_workflow", "native_restore"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.fixture
async def events_case(registry_store, tmp_path):
    store, _ = registry_store
    manager = JobManager(store, now_ms=lambda: 1000000)
    sid = await store.create_session()
    actor = JobActor("events", "host", sid)
    source = await manager.register_job(actor, kind="send", session_id=sid, system_key="source", request={"send_index": 1})
    lease = await manager.claim_job(actor, source.job.job_id, executor_id="SOURCE-EXECUTOR-SECRET", boot_id="BOOT-SECRET", expected_version=1)
    seq = await store.append_message(sid, role="assistant", send_index=1, round_index=0, tool_calls=[{
        "id": "create-call", "type": "function", "function": {"name": "spawn_agent", "arguments": "{}"}}])
    child = await manager.register_job(actor, kind="subagent", parent=lease, independent=True,
        call_ref=await store.message_ref(sid, seq), tool_call_id="create-call")
    receipt = await store.append_message(sid, role="tool", send_index=1, round_index=0, tool_call_id="create-call", content="CREATED")
    await manager.bind_receipt(actor, child.operation.operation_id, await store.message_ref(sid, receipt), caller=lease)
    child_sid = await store.create_session(parent_session_id=sid, metadata={"execution_role": "subagent"})
    bound = await manager.bind_session(actor, child.job.job_id, child_sid)
    child_lease = await manager.claim_job(actor, child.job.job_id, executor_id="CHILD-EXECUTOR-SECRET", boot_id="BOOT-SECRET", expected_version=bound.version)
    event = await manager.record_execution_result(actor, child_lease, outcome="succeeded", result={
        "status": "completed", "final_text": "RESULT-42", "usage": {"total_tokens": 19}}, resource_state="stopped")
    await manager.close_model_execution(actor, lease)
    other_source = await manager.register_job(actor, kind="send", session_id=sid, system_key="controller", request={"send_index": 2})
    caller = await manager.claim_job(actor, other_source.job.job_id, executor_id="CONTROL-EXECUTOR-SECRET", boot_id="BOOT-SECRET", expected_version=1)
    stop_seq = await store.append_message(sid, role="assistant", send_index=2, round_index=0, tool_calls=[{
        "id": "stop-call", "type": "function", "function": {"name": "jobs", "arguments": json.dumps({"action": "stop", "job_id": child.job.job_id})}}])
    stopped, repeated = await task_control.request_stop(manager, actor, child.job.job_id, caller=caller,
        call_ref=await store.message_ref(sid, stop_seq), tool_call_id="stop-call", reason="stop original", expected_generation=1)
    assert not repeated
    stop_receipt = await store.append_message(sid, role="tool", send_index=2, round_index=0, tool_call_id="stop-call", content="STOP-ACCEPTED")
    await manager.bind_receipt(actor, stopped.operation_id, await store.message_ref(sid, stop_receipt), caller=caller)
    loop = SimpleNamespace(task_runtime=SimpleNamespace(manager=manager, actor=lambda _: actor))
    return SimpleNamespace(store=store, manager=manager, actor=actor, sid=sid, child_sid=child_sid,
        job=child.job.job_id, create=child.operation.operation_id, stop=stopped.operation_id, result=event,
        loop=loop, workspace=tmp_path, caller=caller)


async def get(c, **options):
    return await task_events.read_events(c.manager, c.actor, c.job, **options)


async def tool(c, **options):
    lt, st = set_current_loop(c.loop), set_session_id(c.sid)
    try:
        return json.loads(await run_jobs(action="get", **{"job_id": c.job, **options}))
    finally:
        reset_session_id(st)
        reset_current_loop(lt)


async def test_each_operation_has_its_own_original_call_and_receipt_and_no_control_secrets(events_case):
    c = events_case
    before = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.events} ORDER BY event_id")
    response = await tool(c, view="operations")
    assert response["status"] == "ok" and response["read_only"] and not response["execution_authority"]
    operations = {row["operation_id"]: row for row in response["events"]}
    assert set(operations) == {c.create, c.stop}
    assert operations[c.create]["call_ref"]["send_index"] == "1" and operations[c.stop]["call_ref"]["send_index"] == "2"
    assert all(row["call_ref"]["id"] != row["receipt_ref"]["id"] and row["receipt_status"] == "available" for row in operations.values())
    assert "SECRET" not in json.dumps(response)
    single = await tool(c, view="operations", event_id=c.stop)
    assert json.loads(single["text"])["action"] == "stop" and "SECRET" not in single["text"]
    assert [dict(r) for r in before] == [dict(r) for r in await c.manager.db.fetchall(f"SELECT * FROM {c.manager.events} ORDER BY event_id")]


async def test_canonical_result_remains_original_after_job_cache_changes_and_session_deletion(events_case):
    c = events_case
    await c.manager.db.execute(f"UPDATE {c.manager.jobs} SET result_json=? WHERE job_id=?", (json.dumps({"final_text": "WRONG-CACHE"}), c.job))
    await c.store.close_session(c.child_sid)
    listing = await get(c, view="results")
    assert listing["events"][0]["event_id"] == c.result and listing["events"][0]["generation"] == "1"
    result = await get(c, view="results", event_id=c.result)
    assert json.loads(result["text"])["result"]["final_text"] == "RESULT-42" and "WRONG-CACHE" not in result["text"]


@pytest.mark.parametrize("view", ["operations", "results"])
@pytest.mark.parametrize("identity", ["job", "operation", "sub_alias"])
async def test_model_event_views_work_when_mutable_result_cache_is_invalid_json(events_case, view, identity):
    c = events_case
    await c.manager.db.execute(f"UPDATE {c.manager.jobs} SET result_json='invalid-json' WHERE job_id=?", (c.job,))
    target = {"job": c.job, "operation": c.create, "sub_alias": "sub:"+c.create[3:]}[identity]
    response = await tool(c, job_id=target, view=view, **({"event_id": c.result} if view == "results" else {}))
    assert response["status"] == "ok"
    if view == "results":
        assert json.loads(response["text"])["result"]["final_text"] == "RESULT-42"
    else:
        assert len(response["events"]) == 2
    with pytest.raises(ValueError):
        await c.manager.get_job(c.actor, c.job)
    assert (await c.manager.get_job(c.actor, c.job, include_result=False)).result is None


async def test_metadata_read_option_requires_a_boolean(events_case):
    c = events_case
    with pytest.raises(JobError, match="invalid_job_read_options"):
        await c.manager.get_job(c.actor, c.job, include_result=0)


@pytest.mark.parametrize("view", ["operations", "results"])
async def test_workspace_jsonl_contains_exact_public_events_and_no_synthetic_message_ids(events_case, view):
    c = events_case
    with runtime_env_context(RuntimeEnv(workspace_dir=c.workspace)):
        response = await tool(c, view=view, output_file=view+".jsonl")
    assert response["status"] == "exported"
    rows = [json.loads(line) for line in (c.workspace/(view+".jsonl")).read_text().splitlines()]
    assert len(rows) == response["records"] and all("id" not in row and row["source"] == "task_events" for row in rows)
    assert "SECRET" not in str(rows)
    if view == "results":
        assert rows[0]["result"]["final_text"] == "RESULT-42"


async def test_pages_retain_all_operations_and_reject_changed_membership_or_selection(events_case):
    c = events_case
    first = await get(c, view="operations", limit=1)
    second = await get(c, view="operations", cursor=first["next_cursor"], limit=1)
    assert len(first["events"]) == len(second["events"]) == 1 and not second["next_cursor"]
    assert first["events"][0]["event_id"] != second["events"][0]["event_id"]
    with pytest.raises(JobError, match="invalid_task_event_cursor"):
        await get(c, view="results", cursor=first["next_cursor"])
    await c.manager.db.execute(f"DELETE FROM {c.manager.events} WHERE event_id=?", (second["events"][0]["event_id"],))
    with pytest.raises(JobError, match="task_event_snapshot_changed"):
        await get(c, view="operations", cursor=first["next_cursor"])


@pytest.mark.parametrize("fault", ["owner", "tenant", "store", "foreign_event", "private_event_type"])
async def test_event_read_authority_rejects_foreign_scopes_without_payload(events_case, fault):
    c = events_case
    actor, options = c.actor, {"view": "results", "event_id": c.result}
    if fault == "owner":
        actor = JobActor(c.actor.tenant_id, c.actor.principal_id, await c.store.create_session())
    elif fault == "tenant":
        actor = JobActor("foreign", c.actor.principal_id, c.sid)
    elif fault == "store":
        await c.manager.db.execute(f"UPDATE {c.manager.prefix}task_store SET protocol_version=999 WHERE id=1")
    elif fault == "foreign_event":
        options["event_id"] = result_id("job_foreign", 1)
    else:
        options["event_id"] = "control_metadata_private"
        await c.manager.db.execute(f"INSERT INTO {c.manager.events} (event_id,job_id,event_type,job_version,payload_json,delivery_status,created_at) VALUES (?,?,?,?,?,?,?)",
            (options["event_id"], c.job, "execution", 1, '{"secret":"SECRET"}', "not_applicable", 1))
    if fault in ("foreign_event", "private_event_type"):
        result = await task_events.read_events(c.manager, actor, c.job, **options)
        assert result["status"] == "unavailable" and "SECRET" not in str(result)
    else:
        with pytest.raises(JobError):
            await task_events.read_events(c.manager, actor, c.job, **options)


@pytest.mark.parametrize("options", [{"view": "wrong"}, {"view": "results", "char_offset": 1},
    {"view": "results", "limit": True}, {"view": "operations", "cursor": "bad"},
    {"view": "results", "event_id": "", "char_offset": 0}, {"view": "results", "max_chars": 100},
    {"view": "operations", "output_file": "out.jsonl", "cursor": "bad"}])
async def test_invalid_inputs_are_explicit_and_never_write(events_case, options):
    with pytest.raises(JobError):
        await get(events_case, **options)
    assert not list(events_case.workspace.rglob("*.jsonl"))


async def test_full_result_chunks_are_bounded_and_reconstruct_original_json(events_case):
    c = events_case
    row = await c.manager.db.fetchone(f"SELECT payload_json FROM {c.manager.events} WHERE event_id=?", (c.result,))
    payload = json.loads(row["payload_json"])
    text = '完整输出"\\'*1200
    payload["result"]["final_text"] = text
    await c.manager.db.execute(f"UPDATE {c.manager.events} SET payload_json=? WHERE event_id=?", (json.dumps(payload), c.result))
    offset, pieces = 0, []
    while offset is not None:
        response = await tool(c, view="results", event_id=c.result, char_offset=offset, max_chars=1024)
        assert response["status"] == "ok" and len(json.dumps(response, ensure_ascii=False, separators=(",", ":"))) <= 1024
        pieces.append(response["text"])
        offset = response["next_char_offset"]
    assert json.loads("".join(pieces))["result"]["final_text"] == text


@pytest.mark.parametrize("fault", ["existing", "outside", "small_receipt", "partial", "authority", "scope_change", "revision_changed"])
async def test_export_revalidates_and_cleans_failures_without_overwriting(events_case, monkeypatch, fault):
    c = events_case
    target = c.workspace/"events.jsonl"
    options = {"view": "results", "output_file": "events.jsonl"}
    if fault == "existing":
        target.write_text("KEEP")
    elif fault == "outside":
        options["output_file"] = str(c.workspace.parent/"outside-private.jsonl")
    elif fault == "small_receipt":
        options.update(max_chars=1024, output_file="segment/"*100+"events.jsonl")
    else:
        original = task_events._export
        async def failed(*args, **kwargs):
            if fault == "authority":
                await c.manager.db.execute(f"UPDATE {c.manager.jobs} SET principal_id='foreign' WHERE job_id=?", (c.job,))
            elif fault == "scope_change":
                await c.manager.db.execute(f"UPDATE {c.manager.jobs} SET created_by_session_id='foreign' WHERE job_id=?", (c.job,))
            elif fault == "revision_changed":
                await c.manager.db.execute(f"UPDATE {c.manager.events} SET job_version=job_version+1 WHERE event_id=?", (c.result,))
            else:
                async def bad(*_):
                    raise OSError("SECRET FILESYSTEM")
                kwargs["read_records"] = bad
            return await original(*args, **kwargs)
        monkeypatch.setattr(task_events, "_export", failed)
    with runtime_env_context(RuntimeEnv(workspace_dir=c.workspace)):
        result = await tool(c, **options)
    assert "error" in result and "SECRET" not in str(result)
    assert not list(c.workspace.rglob(".recall-*"))
    assert target.read_text() == "KEEP" if fault == "existing" else not target.exists()


@pytest.mark.parametrize("which", ["call", "receipt", "both"])
async def test_later_stop_messages_recall_only_exact_refs_from_its_original_send(events_case, which):
    from power_loop.runtime.task_history import read_history

    c = events_case
    operation = await c.manager.get_operation(c.actor, c.stop)
    ids = [operation.call_ref["id"], operation.receipt_ref["id"]]
    options = {"message_ids": ids} if which == "both" else {"message_id": ids[0 if which == "call" else 1]}
    result = await read_history(c.manager, c.actor, c.job, **options)
    assert result["status"] == "ok" and result["send_index"] == "2"
    assert result["bindings"][0]["source_operation_ids"] == [c.stop]
    assert "CREATED" not in json.dumps(result) and not result["execution_authority"]
    if which == "receipt":
        assert json.loads(result["text"])["content"] == "STOP-ACCEPTED"


@pytest.mark.parametrize("fault", ["child_actor", "mixed_sends", "unreferenced", "wrong_source", "changed_call"])
async def test_operation_origin_refusal_never_expands_to_caller_transcript(events_case, fault):
    from power_loop.runtime.task_history import read_history

    c = events_case
    stop = await c.manager.get_operation(c.actor, c.stop)
    actor, options = c.actor, {"message_id": stop.call_ref["id"]}
    if fault == "child_actor":
        actor = JobActor(c.actor.tenant_id, c.actor.principal_id, c.child_sid)
    elif fault == "mixed_sends":
        create = await c.manager.get_operation(c.actor, c.create)
        options = {"message_ids": [create.call_ref["id"], stop.receipt_ref["id"]]}
    elif fault == "unreferenced":
        seq = await c.store.append_message(c.sid, role="user", send_index=2, content="UNREFERENCED-PRIVATE")
        options = {"message_id": (await c.store.message_ref(c.sid, seq)).to_dict()["id"]}
    elif fault == "wrong_source":
        create = await c.manager.get_job(c.actor, c.job)
        raw = await c.manager.db.fetchone(f"SELECT payload_json FROM {c.manager.events} WHERE event_id=?", (c.stop,))
        payload = json.loads(raw["payload_json"])
        payload["request"]["caller"]["job_id"] = create.parent_job_id
        await c.manager.db.execute(f"UPDATE {c.manager.events} SET payload_json=? WHERE event_id=?", (json.dumps(payload), c.stop))
    else:
        await c.manager.db.execute(f"UPDATE {c.store.t.messages} SET tool_calls_json='[]' WHERE session_id=? AND id=?", (c.sid, int(stop.call_ref["id"])))
    if fault == "unreferenced":
        result = await read_history(c.manager, actor, c.job, **options)
        assert result["status"] == "unavailable" and "UNREFERENCED-PRIVATE" not in str(result)
    else:
        with pytest.raises(JobError):
            await read_history(c.manager, actor, c.job, **options)


async def test_cancelled_unstarted_result_preserves_generation_zero(events_case):
    c = events_case
    queued = await c.manager.register_job(c.actor, kind="subagent", parent=c.caller, system_key="never-dispatched")
    await c.manager.record_unstarted_result(c.actor, queued.job.job_id, outcome="cancelled", result={"status": "cancelled", "final_text": "not dispatched"})
    listing = await task_events.read_events(c.manager, c.actor, queued.job.job_id, view="results")
    assert listing["events"][0]["generation"] == "0"
    result = await task_events.read_events(c.manager, c.actor, queued.job.job_id, view="results", event_id=listing["events"][0]["event_id"])
    assert json.loads(result["text"])["result"]["final_text"] == "not dispatched"


@pytest.mark.parametrize("fault", ["list", "generation", "missing_result", "malformed_id", "bad_call"])
async def test_corrupt_event_fields_do_not_expose_control_payload(events_case, fault):
    c = events_case
    if fault == "bad_call":
        await c.manager.db.execute(f"UPDATE {c.manager.events} SET tool_call_id=NULL WHERE event_id=?", (c.stop,))
        result = await tool(c, view="operations")
    else:
        row = await c.manager.db.fetchone(f"SELECT payload_json FROM {c.manager.events} WHERE event_id=?", (c.result,))
        payload = json.loads(row["payload_json"])
        if fault == "list":
            payload = []
        elif fault == "generation":
            payload["generation"] = True
        elif fault == "missing_result":
            payload.pop("result")
        if fault == "malformed_id":
            await c.manager.db.execute(f"UPDATE {c.manager.events} SET event_id='malformed-result' WHERE event_id=?", (c.result,))
            result = await tool(c, view="results")
        else:
            await c.manager.db.execute(f"UPDATE {c.manager.events} SET payload_json=? WHERE event_id=?", (json.dumps(payload), c.result))
            result = await tool(c, view="results", event_id=c.result)
    assert "error" in result and "SECRET" not in str(result)


async def test_newer_operations_do_not_enter_an_existing_page(events_case):
    c = events_case
    first = await get(c, view="operations", limit=1)
    c.manager.now = lambda: 1000100
    await c.manager.request_cancel(c.actor, c.job, system_key="newer-op", reason="later request")
    second = await get(c, view="operations", cursor=first["next_cursor"], limit=1)
    assert len(second["events"]) == 1 and not second["next_cursor"]
    assert {first["events"][0]["event_id"], second["events"][0]["event_id"]} == {c.create, c.stop}


async def test_event_scope_limit_and_workspace_requirement_are_explicit(events_case, monkeypatch):
    c = events_case
    monkeypatch.setattr(task_events, "_MAX_EVENTS", 1)
    with pytest.raises(JobError, match="scope_too_large"):
        await get(c, view="operations")
    with runtime_env_context(RuntimeEnv(workspace_dir=None)):
        with pytest.raises(JobError, match="workspace_required"):
            await get(c, view="results", output_file="never-written.jsonl")


@pytest.mark.parametrize("fault", ["none", "duplicate", "child_actor"])
async def test_actual_active_nudge_without_caller_payload_uses_unique_send_authority(nudge_case, fault):
    from power_loop.runtime.task_history import read_history

    c = nudge_case
    operation, _ = await c.accept()
    assert "caller" not in operation.result["request"]
    seq = await c.store.append_message(c.actor.session_id, role="tool", send_index=1, round_index=0,
        tool_call_id="nudge-call", content="NUDGE-ACCEPTED")
    ref = await c.store.message_ref(c.actor.session_id, seq)
    await c.manager.bind_receipt(c.actor, operation.operation_id, ref, caller=c.caller)
    await c.manager.close_model_execution(c.actor, c.caller)
    if fault == "duplicate":
        duplicate = await c.manager.register_job(c.actor, kind="send", session_id=c.actor.session_id,
            system_key="ambiguous-send", request={"send_index": 1})
        await c.manager.claim_job(c.actor, duplicate.job.job_id, executor_id="duplicate", boot_id="boot", expected_version=1)
    actor = c.child_actor if fault == "child_actor" else c.actor
    if fault == "none":
        result = await read_history(c.manager, actor, c.target.job_id, message_id=ref.to_dict()["id"])
        assert result["status"] == "ok" and json.loads(result["text"])["content"] == "NUDGE-ACCEPTED"
        assert result["bindings"][0]["source_operation_ids"] == [operation.operation_id]
    else:
        with pytest.raises(JobError):
            await read_history(c.manager, actor, c.target.job_id, message_id=ref.to_dict()["id"])


async def test_small_list_budget_preserves_every_operation_across_pages(events_case):
    c = events_case
    seen, cursor = [], None
    while True:
        response = await get(c, view="operations", max_chars=2000, **({"cursor": cursor} if cursor else {}))
        assert len(json.dumps(response, ensure_ascii=False, separators=(",", ":"))) <= 2000
        seen.extend(e["event_id"] for e in response["events"])
        cursor = response["next_cursor"]
        if not cursor:
            break
    assert len(seen) == 2 and set(seen) == {c.create, c.stop}
    with pytest.raises(JobError, match="budget_too_small"):
        await get(c, view="operations", max_chars=1024)


async def test_actual_restored_workflow_keeps_both_canonical_result_versions(native_restore):
    import asyncio

    c = native_restore
    handle = await c.restore()
    await asyncio.wait_for(handle.task, 10)
    first = await c.manager.get_result_revision(c.actor, handle.job_id, 1)
    second = await c.manager.get_result_revision(c.actor, handle.job_id, 2)
    counts = c.model.counts.copy()
    listing = await task_events.read_events(c.manager, c.actor, handle.job_id, view="results")
    assert {event["generation"] for event in listing["events"]} == {"1", "2"}
    for revision in (first, second):
        response = await task_events.read_events(c.manager, c.actor, handle.job_id, view="results", event_id=revision["event_id"], max_chars=32000)
        assert not response["truncated"]
        saved = json.loads(response["text"])
        assert saved["generation"] == str(revision["generation"]) and saved["result"] == revision["payload"]["result"]
    assert c.model.counts == counts and await c.manager.get_result_revision(c.actor, handle.job_id, 1) == first
    assert await c.manager.get_result_revision(c.actor, handle.job_id, 2) == second
