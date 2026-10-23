"""Shared workflow control selects durable nodes and keeps the actual call identity."""
from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager, nullcontext
from types import SimpleNamespace

import pytest

from power_loop.core.agent_context import reset_session_id, set_session_id
from power_loop.runtime.invocation import invocation_context
from power_loop.runtime.jobs import JobError
from power_loop.runtime.task_runtime import _SEND
from power_loop.workflow import control, runner
from power_loop.workflow.facade import run_workflow_tool, workflow_tool_definition

from .test_job_registry_192 import registry_store
from .test_node_nudge_runtime_192 import client, live_node, target
from .test_node_resume_192 import ready_node
from .test_workflow_facade_192 import legacy_context
from .test_workflow_task_runtime_192 import leaf, managed_workflow, settle

__all__ = ["registry_store", "managed_workflow", "live_node", "ready_node", "legacy_context"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def issue(c, **options):
    await c.issue(handler=lambda: run_workflow_tool({"action": "nudge", "run_id": c.run_id,
        "node": "BUILD", "message": "NUDGED-43", **options}))
    return json.loads(c.replies[-1])


@pytest.mark.parametrize("action", ["nudge", "steer"])
@pytest.mark.parametrize("selector", ["node", "node_job_id", "sid", "node_sid", "job_sid"])
async def test_shared_actions_use_original_node_and_real_operation_refs(live_node, managed_workflow, action, selector):
    c = live_node
    options = {"action": action, "run_id": c.run_id, "message": "NUDGED-43"}
    if selector in ("node", "node_sid"):
        options["node"] = "BUILD"
    if selector in ("node_job_id", "job_sid"):
        options["node_job_id"] = c.job_id
    if selector in ("sid", "node_sid", "job_sid"):
        options["sid"] = c.sid
    await c.issue(handler=lambda: run_workflow_tool(options))
    receipt = json.loads(c.replies[-1])
    assert receipt["status"] == "accepted" and receipt["mode"] == "steer", receipt
    assert receipt["run_id"] == c.run_id and receipt["job_id"] == c.job_id and receipt["session_id"] == c.sid
    assert not c.state["handles"][0].task.done() and c.model.counts["REVIEW"] == 0
    operation = await c.manager.get_operation(c.runtime.actor(c.parent), receipt["operation_id"])
    messages = await c.store.load_all_messages(c.parent)
    [tool] = [m for m in messages if str(m.id) == operation.receipt_ref["id"]]
    assert operation.receipt_ref["id"] == str(tool.id) and operation.call_ref["id"] != operation.receipt_ref["id"]
    assert operation.receipt_ref["tool_call_id"] == tool.tool_call_id
    assert operation.result["request"]["message"] == "NUDGED-43"  # steer adds no synthetic prefix
    await settle(managed_workflow)
    assert (await c.manager.get_job(c.runtime.actor(c.parent), c.job_id)).result["text"] == "NUDGED-43"
    assert len(await c.manager.db.fetchall(f"SELECT job_id FROM {c.manager.jobs} WHERE kind='workflow'")) == 1


@pytest.mark.parametrize("options,code", [
    ({"message": ""}, "invalid_nudge_message"), ({"message": "  "}, "invalid_nudge_message"),
    ({"message": None}, "invalid_nudge_message"), ({"message": True}, "invalid_nudge_message"),
    ({"message": "x\x00y"}, "invalid_nudge_message"), ({"message": "x"*64001}, "invalid_nudge_message"),
    ({"sid": ""}, "invalid_task_session_id"), ({"sid": False}, "invalid_task_session_id"),
    ({"sid": "x"*256}, "invalid_task_session_id"),
    ({"sid": "sess_unrelated"}, "workflow_node_session_not_found"),
    ({"node": "unknown"}, "workflow_node_not_found"), ({"node": False}, "invalid_workflow_node"),
    ({"node_job_id": "unknown"}, "ambiguous_workflow_node_selector"),
    ({"rounds": 3}, "invalid_workflow_parameters"), ({"force": True}, "invalid_workflow_parameters"),
    ({"expected_generation": 1}, "invalid_workflow_parameters"),
])
async def test_invalid_control_never_admits_or_widens_scope(live_node, options, code):
    c = live_node
    result = await issue(c, **options)
    assert result["error"]["code"] == code, result
    assert not await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE action='nudge'")
    assert (await c.manager.get_job(c.runtime.actor(c.parent), c.job_id)).generation == 1
    assert not c.state["handles"][0].task.done()


@pytest.mark.parametrize("selector", ["missing", "workflow_job", "sid_not_found", "sid_lookup_limit"])
async def test_exact_selector_errors_never_fall_back_to_whole_workflow(live_node, monkeypatch, selector):
    c = live_node
    options = {"action": "steer", "run_id": c.run_id, "message": "NUDGED-43"}
    code = "workflow_node_required"
    if selector == "workflow_job":
        options["node_job_id"] = c.state["handles"][0].job_id
        code = "workflow_node_outside_run"
    elif selector == "sid_not_found":
        options["sid"] = "sess_unrelated"
        code = "workflow_node_not_found"
    elif selector == "sid_lookup_limit":
        options["sid"] = c.sid
        monkeypatch.setattr(control, "_MAX_BINDINGS", 0)
        code = "workflow_node_scope_too_large"
    await c.issue(handler=lambda: run_workflow_tool(options))
    assert json.loads(c.replies[-1])["error"]["code"] == code
    assert not await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE action='nudge'")


@pytest.mark.parametrize("selector", ["node", "node_job_id", "sid"])
async def test_original_call_replay_survives_deleted_session_and_engine_removal(live_node, managed_workflow, selector):
    c = live_node
    options = {"action": "nudge", "run_id": c.run_id, "message": "NUDGED-43",
        selector: {"node": "BUILD", "node_job_id": c.job_id, "sid": c.sid}[selector]}
    async def handler():
        first = json.loads(await run_workflow_tool(options))
        assert first["status"] == "accepted", first
        await settle(managed_workflow)
        assert c.run_id not in runner._ENGINES
        await c.manager.db.execute(f"DELETE FROM {c.store.t.sessions} WHERE session_id=?", (c.sid,))
        again = json.loads(await run_workflow_tool(options))
        assert again["operation_id"] == first["operation_id"] and again["replayed"]
        return json.dumps(again)
    await c.issue(handler=handler)
    assert json.loads(c.replies[-1])["replayed"]
    assert len(await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE action='nudge'")) == 1


@pytest.mark.parametrize("change", ["message", "node", "node_job_id", "sid"])
async def test_replayed_call_cannot_change_target_or_message(live_node, change):
    c = live_node
    async def handler():
        options = {"action": "nudge", "run_id": c.run_id, "node": "BUILD", "message": "NUDGED-43"}
        first = json.loads(await run_workflow_tool(options))
        assert first["status"] == "accepted"
        if change == "node_job_id":
            options.pop("node")
        options[change] = "different"
        second = json.loads(await run_workflow_tool(options))
        assert second["error"]["code"] == "operation_conflict", second
        return json.dumps(second)
    await c.issue(handler=handler)
    assert len(await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE action='nudge'")) == 1


@pytest.mark.parametrize("fault", ["missing_binding", "deleted_session", "metadata", "malformed_binding", "cache"])
async def test_physical_evidence_and_cache_boundary(live_node, fault):
    c = live_node
    if fault == "missing_binding":
        await c.manager.db.execute(f"DELETE FROM {c.manager.events} WHERE job_id=? AND event_type='node_session'", (c.job_id,))
    elif fault == "deleted_session":
        await c.manager.db.execute(f"DELETE FROM {c.store.t.sessions} WHERE session_id=?", (c.sid,))
    elif fault == "metadata":
        await c.store.merge_session_metadata(c.sid, {"workflow_node_id": "IMPOSTER"})
    elif fault == "malformed_binding":
        await c.manager.db.execute(f"UPDATE {c.manager.events} SET payload_json='[]' WHERE job_id=? AND event_type='node_session'", (c.job_id,))
    else:
        await c.manager.db.execute(f"UPDATE {c.manager.jobs} SET result_json='invalid-json' WHERE job_id=?", (c.state["handles"][0].job_id,))
    result = await issue(c)
    if fault == "cache":
        assert result["status"] == "accepted", result
        await c.manager.db.execute(f"UPDATE {c.manager.jobs} SET result_json=NULL WHERE job_id=?", (c.state["handles"][0].job_id,))
    else:
        assert "error" in result, result
        assert not await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE action='nudge'")


async def test_finished_parallel_branch_uses_original_sid_and_existing_async_admission(ready_node, managed_workflow):
    c = ready_node
    result = await issue(c)
    assert result["status"] == "accepted" and result["mode"] == "continue" and result["generation"] == 2, result
    assert result["session_id"] == c.sid and result["job_id"] == c.job_id
    assert not c.state["handles"][0].task.done() and c.model.counts["REVIEW"] == 0
    await settle(managed_workflow)
    assert (await c.manager.get_job(c.runtime.actor(c.parent), c.job_id)).result["text"] == "NUDGED-43"


async def test_new_call_after_workflow_exit_does_not_launch_replacement(live_node, managed_workflow):
    c = live_node
    await settle(managed_workflow)
    before = c.model.counts.copy()
    result = await issue(c)
    assert result["error"]["code"] == "task_adapter_unavailable:workflow_node_resume", result
    assert c.model.counts["BUILD"] == before["BUILD"] and c.model.counts["REVIEW"] == before["REVIEW"]


async def test_current_main_and_persisted_invocation_are_required(live_node):
    c = live_node
    async def handler():
        other = await c.loop.new_session()
        token = set_session_id(other)
        try:
            result = json.loads(await run_workflow_tool({"action": "steer", "run_id": c.run_id,
                "node": "BUILD", "message": "NUDGED-43"}))
            assert "error" in result
            return json.dumps(result)
        finally:
            reset_session_id(token)
    await c.issue(handler=handler)
    assert not await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE action='nudge'")


@pytest.mark.parametrize("error_type", [RuntimeError, ValueError, TypeError])
@pytest.mark.parametrize("timing", ["before", "after"])
async def test_unknown_adapter_error_is_private_and_requires_reconciliation(live_node, monkeypatch, error_type, timing):
    c = live_node
    original = runner.steer_node
    async def fail(*args, **kwargs):
        if timing == "after":
            await original(*args, **kwargs)
        raise error_type("private database secret")
    monkeypatch.setattr(runner, "steer_node", fail)
    async def handler():
        options = {"action": "nudge", "run_id": c.run_id, "node": "BUILD", "message": "NUDGED-43"}
        result = json.loads(await run_workflow_tool(options))
        assert result == {"version": 1, "error": {"code": "workflow_control_requires_reconciliation"}, "control_state": "unknown"}
        if timing == "after":
            again = json.loads(await run_workflow_tool(options))
            assert again["status"] == "accepted" and again["replayed"]
        return json.dumps(result)
    await c.issue(handler=handler)
    saved = await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE action='nudge'")
    assert len(saved) == (1 if timing == "after" else 0)


async def test_shared_schema_preserves_contract_and_legacy_requires_host(legacy_context):
    props = workflow_tool_definition().input_schema["properties"]
    assert {"nudge", "steer"} <= set(props["action"]["enum"])
    assert "message" in props and "rounds" not in props
    for action in ("nudge", "steer"):
        result = json.loads(await run_workflow_tool({"action": action, "run_id": "a"*16, "node": "BUILD", "message": "correct"}))
        assert result["error"]["code"] == "legacy_workflow_control_requires_host"


async def assert_stale_candidate(c, monkeypatch, change):
    original = runner.steer_node
    async def stale(*args, expected_job=None):
        assert expected_job == (c.job_id, 1)
        changed = ("job_wrong", 1) if change == "job" else (c.job_id, 2)
        return await original(*args, expected_job=changed)
    monkeypatch.setattr(runner, "steer_node", stale)
    result = await issue(c)
    assert result["error"]["code"] == "workflow_node_generation_changed", result
    assert not await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE action='nudge'")
    assert (await c.manager.get_job(c.runtime.actor(c.parent), c.job_id)).generation == 1


@pytest.mark.parametrize("change", ["job", "generation"])
async def test_live_candidate_expectation_is_checked_at_actual_admission(live_node, monkeypatch, change):
    await assert_stale_candidate(live_node, monkeypatch, change)


@pytest.mark.parametrize("change", ["job", "generation"])
async def test_settled_candidate_expectation_is_checked_at_actual_admission(ready_node, monkeypatch, change):
    await assert_stale_candidate(ready_node, monkeypatch, change)


@pytest.mark.parametrize("expected", [("job_wrong", True), ("job_wrong", 0), ["job_wrong", 1], ("job_wrong",), (False, 1)])
async def test_invalid_internal_expectations_cannot_be_coerced(live_node, expected):
    c = live_node
    async def handler():
        from power_loop.runtime.jobs import JobError
        with pytest.raises(JobError, match="invalid_node_control_expectation"):
            await runner.steer_node(c.run_id, "BUILD", c.sid, "NUDGED-43", expected_job=expected)
        return "{}"
    await c.issue(handler=handler)
    assert not await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE action='nudge'")


@pytest.mark.parametrize("selector", ["node_job_id", "sid"])
async def test_foreach_ambiguity_requires_exact_route_and_keeps_other_iteration(managed_workflow, monkeypatch, selector):
    c = await client(managed_workflow, monkeypatch)
    c.state["spec"]["root"] = {"type": "foreach", "id": "items", "as": "item", "items": [1, 2],
        "body": leaf("BUILD"), "parallel": True}
    await c.loop.send("start foreach", c.parent)
    async with asyncio.timeout(4):
        while len(await target(c)) != 2:
            await asyncio.sleep(.01)
    c.run_id = c.state["handles"][0].run_id
    bindings = await target(c)
    c.job_id, c.sid = bindings[0][0]["job_id"], bindings[0][1]["session_id"]
    ambiguous = await issue(c)
    assert ambiguous["error"]["code"] == "workflow_node_ambiguous_choose_job", ambiguous
    value = c.job_id if selector == "node_job_id" else c.sid
    await c.issue(handler=lambda: run_workflow_tool({"action": "steer", "run_id": c.run_id,
        "message": "NUDGED-43", selector: value}))
    assert json.loads(c.replies[-1])["job_id"] == c.job_id
    await settle(managed_workflow)
    for row, data in bindings:
        job = await c.manager.get_job(c.runtime.actor(c.parent), row["job_id"])
        assert job.result["text"] == ("NUDGED-43" if data["session_id"] == c.sid else "PROOF-42")


async def test_node_name_chooses_latest_actual_retry_not_old_session(managed_workflow, monkeypatch):
    c = await client(managed_workflow, monkeypatch)
    c.state["spec"]["root"] = leaf("BUILD", retry={"max_attempts": 2, "on": ["empty"]})
    c.model.hold = set()
    original = c.model.complete
    entered, release = asyncio.Event(), asyncio.Event()
    async def retry(request, **kwargs):
        build = "ROLE-BUILD" in (request.system_prompt or "")
        if build and c.model.counts["BUILD"] == 1:
            entered.set()
            await release.wait()
        result = await original(request, **kwargs)
        if build and c.model.counts["BUILD"] == 1:
            result.raw_text = ""
            result.tool_calls = [{"id": "empty-pass", "type": "function", "function": {"name": "pass_turn", "arguments": '{"reason":"no output"}'}}]
        return result
    monkeypatch.setattr(c.model, "complete", retry)
    try:
        await c.loop.send("start retry", c.parent)
        await asyncio.wait_for(entered.wait(), 4)
        bindings = sorted(await target(c), key=lambda pair: pair[1]["attempt"])
        assert [data["attempt"] for _, data in bindings] == [1, 2]
        c.run_id = c.state["handles"][0].run_id
        c.job_id, c.sid = bindings[1][0]["job_id"], bindings[1][1]["session_id"]
        result = await issue(c)
        assert result["status"] == "accepted" and result["session_id"] == c.sid, result
    finally:
        release.set()
    await settle(managed_workflow)
    assert (await c.manager.get_job(c.runtime.actor(c.parent), c.job_id)).result["text"] == "NUDGED-43"


async def test_workflow_cancel_barrier_cannot_be_bypassed_by_node_selection(live_node):
    c = live_node
    await c.manager.request_cancel(c.runtime.actor(c.parent), c.state["handles"][0].job_id,
        system_key="node-facade-cancel", reason="stop original workflow")
    result = await issue(c)
    assert "error" in result, result
    assert not await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE action='nudge'")


async def assert_no_input(c):
    assert not await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE action='nudge'")
    assert (await c.manager.get_job(c.runtime.actor(c.parent), c.job_id)).generation == 1


@pytest.mark.parametrize("value", [None, False, "", "x"*65])
async def test_exact_job_selector_rejects_invalid_values_without_name_fallback(live_node, value):
    c = live_node
    options = {"action": "steer", "run_id": c.run_id, "node_job_id": value, "message": "NUDGED-43"}
    await c.issue(handler=lambda: run_workflow_tool(options))
    assert json.loads(c.replies[-1])["error"]["code"] == "invalid_workflow_node_job"
    await assert_no_input(c)


@pytest.mark.parametrize("payload", ["{", "[]", "{}", '{"request":null}'])
async def test_name_lookup_rejects_corrupt_original_admission(live_node, payload):
    c = live_node
    [row] = await c.manager.db.fetchall(f"SELECT event_id,payload_json FROM {c.manager.events} WHERE job_id=? AND action='create'", (c.job_id,))
    try:
        await c.manager.db.execute(f"UPDATE {c.manager.events} SET payload_json=? WHERE event_id=?", (payload, row["event_id"]))
        result = await issue(c)
        assert result["error"]["code"] == "workflow_node_binding_corrupt", result
        await assert_no_input(c)
    finally:
        await c.manager.db.execute(f"UPDATE {c.manager.events} SET payload_json=? WHERE event_id=?", (row["payload_json"], row["event_id"]))


@pytest.mark.parametrize("payload", ["{", "[]", "null"])
async def test_sid_lookup_rejects_malformed_saved_bindings(live_node, payload):
    c = live_node
    [row] = await c.manager.db.fetchall(f"SELECT event_id,payload_json FROM {c.manager.events} WHERE job_id=? AND event_type='node_session'", (c.job_id,))
    try:
        await c.manager.db.execute(f"UPDATE {c.manager.events} SET payload_json=? WHERE event_id=?", (payload, row["event_id"]))
        await c.issue(handler=lambda: run_workflow_tool({"action": "nudge", "run_id": c.run_id, "sid": c.sid, "message": "NUDGED-43"}))
        assert json.loads(c.replies[-1])["error"]["code"] == "workflow_node_binding_corrupt"
        await assert_no_input(c)
    finally:
        await c.manager.db.execute(f"UPDATE {c.manager.events} SET payload_json=? WHERE event_id=?", (row["payload_json"], row["event_id"]))


@pytest.mark.parametrize("source", ["missing", "foreign_runtime", "foreign_session", "missing_call"])
async def test_internal_control_requires_current_runtime_send_and_actual_call(live_node, source):
    c = live_node
    job = await c.manager.get_job(c.runtime.actor(c.parent), c.state["handles"][0].job_id)
    async def handler():
        current = _SEND.get()
        changed = None if source == "missing" else SimpleNamespace(runtime=object(), actor=current.actor) if source == "foreign_runtime" else SimpleNamespace(runtime=c.runtime, actor=c.runtime.actor("foreign"))
        token = _SEND.set(changed) if source != "missing_call" else None
        try:
            with invocation_context(c.parent, None, None) if source == "missing_call" else nullcontext():
                with pytest.raises(JobError, match="persisted_tool_call_required" if source == "missing_call" else "task_send_required"):
                    await control.nudge(c.loop, c.parent, job, {"run_id": c.run_id, "node": "BUILD", "message": "NUDGED-43"})
            return "{}"
        finally:
            if token is not None:
                _SEND.reset(token)
    await c.issue(handler=handler)
    await assert_no_input(c)


async def test_internal_selector_does_not_accidentally_select_whole_run(live_node):
    c = live_node
    actor = c.runtime.actor(c.parent)
    workflow = await c.manager.get_job(actor, c.state["handles"][0].job_id)
    assert await control.select_native_target(c.runtime, c.parent, workflow, {}) == workflow.job_id
    with pytest.raises(JobError, match="ambiguous_workflow_node_selector"):
        await control.select_native_target(c.runtime, c.parent, workflow, {"node": "BUILD", "node_job_id": c.job_id})
    with pytest.raises(JobError, match="workflow_node_required"):
        await control._selected(c.runtime, actor, workflow, {})
    await assert_no_input(c)


async def test_selector_snapshot_cannot_silently_change_generation(live_node, monkeypatch):
    from power_loop.runtime import task_sessions
    c = live_node
    original = task_sessions.list_sessions
    async def changed(*args, **kwargs):
        saved = await original(*args, **kwargs)
        # Inject a mismatched second read; actual admission must stay untouched.
        return {**saved, "job_generation": saved["job_generation"]+1}
    monkeypatch.setattr(task_sessions, "list_sessions", changed)
    result = await issue(c)
    assert result["error"]["code"] == "workflow_node_generation_changed", result
    await assert_no_input(c)


async def test_same_attempt_multiple_physical_sessions_requires_explicit_sid(live_node):
    c = live_node
    original = await c.store.get_session(c.sid)
    other = await c.store.create_session(parent_session_id=original.parent_session_id, metadata=original.metadata)
    key = c.manager._node_session_event_id(c.job_id, other, 1)
    data = {**c.data, "session_id": other}
    try:
        await c.manager.db.execute(f"INSERT INTO {c.manager.events} (event_id,job_id,event_type,job_version,payload_json,delivery_status,created_at) VALUES (?,?,?,?,?,?,?)",
            (key, c.job_id, "node_session", 1, json.dumps(data), "not_applicable", c.manager.now()))
        result = await issue(c)
        assert result["error"]["code"] == "workflow_node_session_ambiguous_choose_sid", result
        await assert_no_input(c)
        selected = await issue(c, sid=c.sid)
        assert selected["status"] == "accepted" and selected["session_id"] == c.sid
    finally:
        await c.manager.db.execute(f"DELETE FROM {c.manager.events} WHERE event_id=?", (key,))
        await c.manager.db.execute(f"DELETE FROM {c.store.t.sessions} WHERE session_id=?", (other,))


async def test_saved_receipt_cannot_be_replayed_after_task_lineage_is_changed(live_node):
    c = live_node
    async def handler():
        options = {"action": "nudge", "run_id": c.run_id, "node": "BUILD", "message": "NUDGED-43"}
        first = json.loads(await run_workflow_tool(options))
        assert first["status"] == "accepted"
        row = await c.manager.db.fetchone(f"SELECT parent_job_id FROM {c.manager.jobs} WHERE job_id=?", (c.job_id,))
        try:
            await c.manager.db.execute(f"UPDATE {c.manager.jobs} SET parent_job_id=? WHERE job_id=?", (_SEND.get().lease.job_id, c.job_id))
            repeated = json.loads(await run_workflow_tool(options))
            assert repeated["error"]["code"] == "workflow_node_outside_run", repeated
        finally:
            await c.manager.db.execute(f"UPDATE {c.manager.jobs} SET parent_job_id=? WHERE job_id=?", (row["parent_job_id"], c.job_id))
        return json.dumps(first)
    await c.issue(handler=handler)
    assert len(await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE action='nudge'")) == 1


@asynccontextmanager
async def duplicate_node_records(c, count, *, node_id=None, same_sid=False):
    """Inject saved conflicting records; clean them before the held engine exits."""
    job = dict(await c.manager.db.fetchone(f"SELECT * FROM {c.manager.jobs} WHERE job_id=?", (c.job_id,)))
    event = dict(await c.manager.db.fetchone(f"SELECT * FROM {c.manager.events} WHERE job_id=? AND action='create'", (c.job_id,)))
    ids = ["job_boundary_"+str(i) for i in range(count)]
    try:
        async with c.manager._write() as tx:
            for target_id in ids:
                saved = {**job, "job_id": target_id}
                keys = list(saved)
                await tx.execute(f"INSERT INTO {c.manager.jobs} ({','.join(keys)}) VALUES ({','.join('?' for _ in keys)})", tuple(saved.values()))
                payload = json.loads(event["payload_json"])
                if node_id is not None:
                    payload["request"]["node"]["id"] = node_id
                saved = {**event, "event_id": "op_"+target_id, "job_id": target_id,
                    "operation_key": "boundary:"+target_id, "payload_json": json.dumps(payload)}
                keys = list(saved)
                await tx.execute(f"INSERT INTO {c.manager.events} ({','.join(keys)}) VALUES ({','.join('?' for _ in keys)})", tuple(saved.values()))
                if same_sid:
                    await tx.execute(f"INSERT INTO {c.manager.events} (event_id,job_id,event_type,job_version,payload_json,delivery_status,created_at) VALUES (?,?,?,?,?,?,?)",
                        (c.manager._node_session_event_id(target_id, c.sid, 1), target_id, "node_session", 1, json.dumps(c.data), "not_applicable", c.manager.now()))
        yield ids
    finally:
        async with c.manager._write() as tx:
            for target_id in ids:
                await tx.execute(f"DELETE FROM {c.manager.events} WHERE job_id=?", (target_id,))
                await tx.execute(f"DELETE FROM {c.manager.jobs} WHERE job_id=?", (target_id,))


@pytest.mark.parametrize("total,code", [(100, "workflow_node_ambiguous_choose_job"), (101, "workflow_node_scope_too_large")])
async def test_real_query_cardinality_bound_is_100_and_not_101(live_node, monkeypatch, total, code):
    c = live_node
    # This is a cardinality/corruption test, not a wall-clock lease-expiry test.
    # Keep actual lease checks enabled while loading 100 records under one lock.
    now = c.manager.now()
    monkeypatch.setattr(c.manager, "now", lambda: now)
    async with duplicate_node_records(c, total-1):
        result = await issue(c)
        assert result["error"]["code"] == code, result
        await assert_no_input(c)


async def test_name_lookup_checks_original_node_id_beyond_display_title(live_node):
    c = live_node
    async with duplicate_node_records(c, 1, node_id="different-original-id"):
        workflow = await c.manager.get_job(c.runtime.actor(c.parent), c.state["handles"][0].job_id)
        chosen = await control.select_native_target(c.runtime, c.parent, workflow, {"node": "BUILD"})
        assert chosen == c.job_id
        await assert_no_input(c)


async def test_same_sid_recorded_under_two_jobs_is_not_execution_authority(live_node):
    c = live_node
    async with duplicate_node_records(c, 1, same_sid=True):
        await c.issue(handler=lambda: run_workflow_tool({"action": "steer", "run_id": c.run_id, "sid": c.sid, "message": "NUDGED-43"}))
        assert json.loads(c.replies[-1])["error"]["code"] == "workflow_node_ambiguous_choose_job"
        await assert_no_input(c)
