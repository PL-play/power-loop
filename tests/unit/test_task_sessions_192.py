"""Persistent task authority outlives physical driver cleanup."""
from __future__ import annotations

import json
from dataclasses import replace

import pytest

from power_loop.runtime import task_sessions
from power_loop.runtime.jobs import JobError

from .test_job_registry_192 import registry_store
from .test_job_registry_192 import runtime as registry_runtime
from .test_workflow_restore_plan_192 import finished
from .test_workflow_restore_runtime_192 import native_restore
from .test_workflow_task_runtime_192 import leaf, managed_workflow

__all__ = ["registry_store", "registry_runtime", "managed_workflow", "native_restore"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.fixture
async def history(managed_workflow):
    _, manager, runtime, _, model, sid, state, _ = managed_workflow
    model.spawn_from_build = True
    state["spec"]["root"]["steps"][0] = leaf("BUILD", tools=["spawn_agent"])
    await finished(managed_workflow)
    return managed_workflow


async def test_owned_node_and_nested_child_bindings_survive_driver_deletion_without_execution(history):
    store, manager, runtime, _, model, sid, state, connect = history
    actor = runtime.actor(sid)
    before = model.counts.copy()
    events = await manager.db.fetchall(f"SELECT event_id FROM {manager.events} ORDER BY event_id")
    reopened = type(manager)(await connect())
    workflow = await task_sessions.list_sessions(reopened, actor, state["handles"][0].job_id)
    assert workflow["bindings"] == [] and workflow["reason"] == "logical_task"
    nodes = await manager.db.fetchall(f"SELECT job_id FROM {manager.jobs} WHERE kind='workflow_node'")
    for node in nodes:
        result = await task_sessions.list_sessions(reopened, actor, node["job_id"])
        [binding] = result["bindings"]
        assert binding["generation"] == 1 and binding["available"] and not binding["execution_authority"]
        session = await store.get_session(binding["session_id"])
        assert session.parent_session_id is None
        assert (await task_sessions.authorize_session(reopened, actor, node["job_id"], session.session_id))["bindings"] == [binding]
    [child] = await manager.db.fetchall(f"SELECT job_id,session_id FROM {manager.jobs} WHERE kind='subagent'")
    child_binding = await task_sessions.authorize_session(reopened, actor, child["job_id"], child["session_id"])
    assert child_binding["bindings"][0]["available"]
    assert events == await manager.db.fetchall(f"SELECT event_id FROM {manager.events} ORDER BY event_id")
    assert model.counts == before


async def test_send_binding_keeps_original_send_index_and_never_grants_whole_session(history):
    _, manager, runtime, _, _, sid, _, _ = history
    rows = await manager.db.fetchall(f"SELECT job_id,session_id FROM {manager.jobs} WHERE kind='send'")
    for row in rows:
        result = await task_sessions.list_sessions(manager, runtime.actor(sid), row["job_id"])
        [binding] = result["bindings"]
        assert binding["session_id"] == row["session_id"] and binding["history_scope"] == "send"
        assert type(binding["send_index"]) is int and binding["send_index"] >= 1


@pytest.mark.parametrize("fault", ["other_actor", "other_tenant", "other_principal", "leaf_reads_sibling", "wrong_sid"])
async def test_cross_scope_and_unbound_session_denied_even_with_matching_workflow_metadata(history, fault):
    store, manager, runtime, _, _, sid, _, _ = history
    actor = runtime.actor(sid)
    nodes = await manager.db.fetchall(f"SELECT job_id FROM {manager.jobs} WHERE kind='workflow_node' ORDER BY title")
    source = (await task_sessions.list_sessions(manager, actor, nodes[0]["job_id"]))["bindings"][0]["session_id"]
    target = (await task_sessions.list_sessions(manager, actor, nodes[1]["job_id"]))["bindings"][0]["session_id"]
    if fault == "other_actor":
        actor = replace(actor, session_id=await store.create_session())
    elif fault == "other_tenant":
        actor = replace(actor, tenant_id="other")
    elif fault == "other_principal":
        actor = replace(actor, principal_id="other")
    elif fault == "leaf_reads_sibling":
        actor = replace(actor, session_id=source)
    else:
        target = source
    with pytest.raises(JobError):
        await task_sessions.authorize_session(manager, actor, nodes[1]["job_id"], target)


@pytest.mark.parametrize("fault", ["event_id", "generation_bool", "generation_future", "node_metadata", "delete_session"])
async def test_corrupt_binding_is_not_authority_and_deleted_history_is_explicit(history, fault):
    store, manager, runtime, _, _, sid, _, _ = history
    actor = runtime.actor(sid)
    [event] = (await manager.db.fetchall(f"SELECT event_id,job_id,payload_json FROM {manager.events} WHERE event_type='node_session' ORDER BY event_id"))[:1]
    data = json.loads(event["payload_json"])
    target = data["session_id"]
    if fault == "event_id":
        await manager.db.execute(f"UPDATE {manager.events} SET event_id=? WHERE event_id=?", ("forged", event["event_id"]))
    elif fault in ("generation_bool", "generation_future"):
        data["generation"] = True if fault == "generation_bool" else 99
        await manager.db.execute(f"UPDATE {manager.events} SET payload_json=? WHERE event_id=?", (json.dumps(data), event["event_id"]))
    elif fault == "node_metadata":
        await store.merge_session_metadata(target, {"task_owner_job_id": "forged"})
    else:
        await store.close_session(target)
        result = await task_sessions.authorize_session(manager, actor, event["job_id"], target)
        assert result["bindings"][0]["available"] is False
        assert result["bindings"][0]["reason"] == "session_deleted"
        return
    with pytest.raises(JobError):
        await task_sessions.authorize_session(manager, actor, event["job_id"], target)


@pytest.mark.parametrize("fault", ["scope_missing", "session_replaced", "send_index_bool", "child_role", "child_parent"])
async def test_bound_send_and_child_cannot_be_repointed_to_unrelated_history(history, fault):
    store, manager, runtime, _, _, sid, _, _ = history
    kind = "send" if fault in ("scope_missing", "session_replaced", "send_index_bool") else "subagent"
    [job] = (await manager.db.fetchall(f"SELECT * FROM {manager.jobs} WHERE kind=? ORDER BY job_id", (kind,)))[:1]
    if fault == "scope_missing":
        await manager.db.execute(f"DELETE FROM {manager.events} WHERE job_id=? AND event_type='execution_scope'", (job["job_id"],))
    elif fault == "session_replaced":
        other = await store.create_session()
        await manager.db.execute(f"UPDATE {manager.jobs} SET session_id=? WHERE job_id=?", (other, job["job_id"]))
    elif fault == "send_index_bool":
        [event] = await manager.db.fetchall(f"SELECT event_id,payload_json FROM {manager.events} WHERE job_id=? AND action='create'", (job["job_id"],))
        data = json.loads(event["payload_json"])
        data["request"]["send_index"] = True
        await manager.db.execute(f"UPDATE {manager.events} SET payload_json=? WHERE event_id=?", (json.dumps(data), event["event_id"]))
    elif fault == "child_role":
        await store.merge_session_metadata(job["session_id"], {"execution_role": "main"})
    else:
        await manager.db.execute(f"UPDATE {store.t.sessions} SET parent_session_id=NULL WHERE session_id=?", (job["session_id"],))
    with pytest.raises(JobError):
        await task_sessions.list_sessions(manager, runtime.actor(sid), job["job_id"])


@pytest.mark.parametrize("fault", ["extra_field", "node_id", "route", "attempt_bool", "nonfinite", "scope_limit"])
async def test_deleted_session_still_requires_valid_frozen_binding_and_scope_is_bounded(history, fault):
    store, manager, runtime, _, _, sid, _, _ = history
    [event] = (await manager.db.fetchall(f"SELECT * FROM {manager.events} WHERE event_type='node_session' ORDER BY event_id"))[:1]
    data = json.loads(event["payload_json"])
    await store.close_session(data["session_id"])
    if fault == "scope_limit":
        for i in range(100):
            await manager.db.execute(f"INSERT INTO {manager.events} (event_id,job_id,event_type,job_version,payload_json,delivery_status,created_at) VALUES (?,?,?,?,?,?,?)",
                ("extra-"+str(i), event["job_id"], "node_session", event["job_version"], event["payload_json"], "not_applicable", event["created_at"]))
    else:
        if fault == "extra_field":
            data["unexpected"] = "value"
        elif fault == "node_id":
            data["node_id"] = "other-node"
        elif fault == "route":
            data["route"] = ["other-route"]
        elif fault == "attempt_bool":
            data["attempt"] = True
        else:
            data["route"] = [float("nan")]
        await manager.db.execute(f"UPDATE {manager.events} SET payload_json=? WHERE event_id=?", (json.dumps(data), event["event_id"]))
    with pytest.raises(JobError):
        await task_sessions.list_sessions(manager, runtime.actor(sid), event["job_id"])


@pytest.mark.parametrize("value", [None, "", True, 123, "a"*256])
async def test_invalid_task_session_coordinates_fail_before_any_store_access(value):
    with pytest.raises(JobError):
        await task_sessions.authorize_session(None, None, "job-original", value)


@pytest.mark.parametrize("value", [None, "", True, 123, "a"*65])
async def test_invalid_task_id_fails_before_any_store_access(value):
    with pytest.raises(JobError):
        await task_sessions.list_sessions(None, None, value)


async def test_retained_physical_session_lists_actual_three_generation_descriptors_without_dispatch(native_restore):
    c = native_restore
    second = await c.restore()
    await second.task
    await c.loop.flush_follow_ups(c.parent)
    third = await c.restore(node="REVIEW")
    await third.task
    await c.loop.flush_follow_ups(c.parent)
    before = c.model.counts.copy()
    for node in c.node_jobs:
        expected = [1, 2] if node["title"] == "BUILD" else [1, 2, 3]
        result = await task_sessions.list_sessions(c.manager, c.actor, node["job_id"])
        assert sorted(binding["generation"] for binding in result["bindings"]) == expected
        assert len({binding["session_id"] for binding in result["bindings"]}) == 1
        physical_sid = result["bindings"][0]["session_id"]
        assert (await c.store.get_session(physical_sid)).parent_session_id is None
        assert (await task_sessions.authorize_session(c.manager, c.actor, node["job_id"], physical_sid))["bindings"] == result["bindings"]
    assert c.model.counts == before


@pytest.mark.parametrize("kind", ["subagent", "workflow", "background_command", "timer"])
async def test_queued_or_logical_task_does_not_grant_access_to_its_callers_history(registry_runtime, kind):
    _, manager, actor, _, parent, _, _ = registry_runtime
    registration = await manager.register_job(actor, kind=kind, parent=parent, system_key="unstarted-"+kind)
    result = await task_sessions.list_sessions(manager, actor, registration.job.job_id)
    assert result["bindings"] == [] and result["execution_authority"] is False
    assert result["reason"] == ("not_created" if kind == "subagent" else "logical_task")
    with pytest.raises(JobError, match="access_denied"):
        await task_sessions.authorize_session(manager, actor, registration.job.job_id, actor.session_id)
