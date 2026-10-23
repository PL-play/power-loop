"""Legacy workflow history uses owned journals/contracts, not metadata-only SQL."""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from power_loop import AgentLoopConfig, StatefulAgentLoop
from power_loop.runtime.env import RuntimeEnv, runtime_env_context
from power_loop.runtime.jobs import JobError
from power_loop.runtime.representation import ProjectedRepresentation
from power_loop.tools.registry import ToolRegistry
from power_loop.workflow import create_workflow, journal, legacy_history
from power_loop.workflow.facade import register_workflow_tool

from .test_job_registry_192 import registry_store
from .test_workflow_control_192 import invoke
from .test_workflow_task_runtime_192 import WorkflowModel, leaf

__all__ = ["registry_store"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.fixture
async def legacy_case(registry_store, tmp_path):
    _, connect = registry_store
    store = await connect(schema=None, suffix="old_")
    model = WorkflowModel()
    loop = StatefulAgentLoop(llm=model, store=store, tool_registry=ToolRegistry(),
        config=AgentLoopConfig(delegation_policy="single_level", max_rounds=3, representation=ProjectedRepresentation()))
    owner = await loop.new_session()
    model.root = owner
    register_workflow_tool(loop.tool_registry)
    c = SimpleNamespace(store=store, connect=connect, loop=loop, model=model, owner=owner, handles=[], workspace=tmp_path)
    async def start(root=None):
        spec = {"name": "legacy-history", "root": root or {"type": "sequence", "steps": [leaf("BUILD"), leaf("REVIEW")]}}
        async def completed(_):
            pass
        handle = await create_workflow(spec, parent_loop=loop, parent_session_id=owner).start(detached=True, on_complete=completed)
        c.handles.append(handle)
        c.handle = handle
        return handle
    c.start = start
    try:
        yield c
    finally:
        for event in model.release.values():
            event.set()
        for role in model.hold:
            model.release[role].set()
        await asyncio.wait_for(asyncio.gather(*(h.task for h in c.handles), return_exceptions=True), 8)


async def settled(c, root=None):
    c.model.hold = set()
    handle = await c.start(root)
    await asyncio.wait_for(handle.task, 8)
    return handle


async def view(c, action, **kwargs):
    return await invoke(c.loop, c.owner, {"action": action, "run_id": c.handle.run_id, **kwargs})


async def test_actual_foreach_binding_precedes_provider_and_survives_driver_cleanup(legacy_case):
    c = legacy_case
    root = {"type": "foreach", "id": "many", "as": "item", "items": ["a", "b", "c"], "body": leaf("BUILD"), "max_concurrency": 3}
    handle = await c.start(root)
    async def all_started():
        while c.model.counts["BUILD"] < 3:
            await asyncio.sleep(.01)
    await asyncio.wait_for(all_started(), 4)
    first = await view(c, "leaves")
    assert len(first["nodes"]) == 3 and first["history_complete"]
    assert {n["iteration"] for n in first["nodes"]} == {0, 1, 2}
    assert all(n["available"] and n["binding_source"] == "owned_history_registry" for n in first["nodes"])
    saved = await journal.read(c.store, c.owner, handle.run_id)
    assert not saved["steps"] and not saved.get("live_sessions")
    c.model.release["BUILD"].set()
    await asyncio.wait_for(handle.task, 8)
    before = c.model.counts.copy()
    after = await view(c, "leaves")
    for node in after["nodes"]:
        session = await c.store.get_session(node["session_id"])
        assert session.parent_session_id is None
        result = await view(c, "logs", sid=session.session_id)
        assert result["status"] == "ok" and result["format"] == "round_preview"
        assert "PROOF-42" in str(result["rounds"])
    assert c.model.counts == before and not after["physical_state_confirmed"]


@pytest.mark.parametrize("mode", ["node", "sid", "sid_only", "projection", "paged", "export"])
async def test_legacy_logs_use_shared_recall_and_original_owned_selection(legacy_case, mode):
    c = legacy_case
    await settled(c)
    listing = await view(c, "leaves")
    node = next(n for n in listing["nodes"] if n["node_id"] == "BUILD")
    before = c.model.counts.copy()
    options = {"node": "BUILD"} if mode == "node" else {"sid": node["session_id"]}
    if mode == "projection":
        options.update(source="projection", limit=1)
    elif mode == "paged":
        options.update(limit=1)
    elif mode == "export":
        options.update(output_file="legacy/history.jsonl")
    with runtime_env_context(RuntimeEnv(workspace_dir=c.workspace)):
        result = await (invoke(c.loop, c.owner, {"action": "logs", **options}) if mode == "sid_only" else view(c, "logs", **options))
    assert result["backend"] == "legacy" and not result["execution_authority"]
    assert result["raw_message_id_status"] == "unavailable"
    if mode == "export":
        assert result["status"] == "exported" and result["records"] >= 2
        values = [json.loads(line) for line in (c.workspace/"legacy/history.jsonl").read_text().splitlines()]
        assert all(v["session_id"] == node["session_id"] and v["id"] is None for v in values)
    else:
        assert result["status"] == "ok"
    assert c.model.counts == before


async def test_metadata_only_forged_session_is_not_discovered_or_readable(legacy_case):
    c = legacy_case
    await settled(c)
    forged = await c.store.create_session(metadata={"workflow_run_id": c.handle.run_id, "workflow_node_id": "BUILD"})
    await c.store.append_message(forged, role="assistant", content="FOREIGN-PRIVATE")
    result = await view(c, "leaves")
    assert forged not in str(result)
    denied = await view(c, "logs", sid=forged)
    assert denied["error"]["code"] == "legacy_workflow_history_session_not_owned"
    assert "FOREIGN-PRIVATE" not in str(denied)


async def test_existing_contracts_remain_readable_without_new_history_registry(legacy_case):
    c = legacy_case
    await settled(c)
    await c.store.delete_runtime_state(c.owner, legacy_history.history_key(c.handle.run_id))
    result = await view(c, "leaves")
    assert len(result["nodes"]) == 2 and result["history_complete"] is False
    assert all(n["available"] and n["binding_source"] == "owned_journal_and_contract" for n in result["nodes"])
    assert (await view(c, "logs", node="BUILD"))["status"] == "ok"


@pytest.mark.parametrize("fault", ["contract_hash", "driver", "owner", "route", "attempt", "missing_contract", "deleted"])
async def test_registered_history_refuses_corruption_and_reports_deleted_sessions(legacy_case, fault):
    c = legacy_case
    await settled(c)
    key = legacy_history.history_key(c.handle.run_id)
    state = await c.store.get_runtime_state(c.owner, key)
    sid = next(iter(state["sessions"]))
    if fault == "missing_contract":
        await c.store.delete_runtime_state(sid, "workflow_execution_contract")
    elif fault == "deleted":
        await c.store.close_session(sid)
    else:
        if fault == "contract_hash":
            state["sessions"][sid]["contract_hash"] = "0"*64
        elif fault == "driver":
            state["drivers"][next(iter(state["drivers"]))]["owner_session_id"] = "foreign"
        elif fault == "owner":
            state["owner_session_id"] = "foreign"
        elif fault == "route":
            state["sessions"][sid]["route"] = [{"foreach": "foreign", "index": 0}]
        else:
            state["sessions"][sid]["attempt"] = True
        await c.store.set_runtime_state(c.owner, key, state)
    result = await view(c, "logs", sid=sid)
    if fault == "deleted":
        assert result["status"] == "unavailable" and result["reason"] == "session_deleted"
    else:
        assert "error" in result and "PROOF-42" not in str(result)


async def test_wrong_owner_never_uses_global_run_id_or_session_metadata(legacy_case):
    c = legacy_case
    await settled(c)
    other = await c.loop.new_session()
    denied = await invoke(c.loop, other, {"action": "leaves", "run_id": c.handle.run_id})
    assert denied["error"]["code"] == "workflow_run_not_found"


@pytest.mark.parametrize("where", ["driver", "session"])
async def test_persistent_admission_failure_never_dispatches_child_provider(legacy_case, monkeypatch, where):
    c = legacy_case
    c.model.hold = set()
    async def unavailable(*args, **kwargs):
        raise JobError("test_history_binding_unavailable")
    monkeypatch.setattr(legacy_history, "register_"+where, unavailable)
    handle = await c.start()
    await asyncio.wait_for(handle.task, 8)
    assert c.model.counts["BUILD"] == 0 and c.model.counts["REVIEW"] == 0
    assert (await journal.read(c.store, c.owner, handle.run_id))["status"] == "failed"


async def test_owner_history_pages_do_not_skip_sessions_and_cursor_binds_owner_scope(legacy_case):
    c = legacy_case
    await settled(c)
    first = await view(c, "leaves", limit=1)
    assert len(first["nodes"]) == 1 and first["next_cursor"]
    second = await view(c, "leaves", limit=1, cursor=first["next_cursor"])
    assert len(second["nodes"]) == 1 and second["nodes"][0]["session_id"] != first["nodes"][0]["session_id"]
    state = await c.store.get_runtime_state(c.owner, legacy_history.history_key(c.handle.run_id))
    node_sid = first["nodes"][0]["session_id"]
    await c.store.close_session(node_sid)
    changed = await view(c, "leaves", cursor=first["next_cursor"])
    assert changed["error"]["code"] == "invalid_legacy_workflow_history_cursor"
    assert state["sessions"]


async def test_legacy_log_export_uses_current_authority_and_never_overwrites(legacy_case):
    c = legacy_case
    await settled(c)
    target = c.workspace/"existing.jsonl"
    target.write_text("KEEP")
    with runtime_env_context(RuntimeEnv(workspace_dir=c.workspace)):
        result = await view(c, "logs", node="BUILD", output_file="existing.jsonl")
    assert "error" in result and target.read_text() == "KEEP" and not list(c.workspace.glob(".recall-*"))


@pytest.mark.parametrize("owner_marker", [None, "foreign"])
async def test_pre_registry_contracts_without_new_owner_fields_remain_scoped(legacy_case, owner_marker):
    c = legacy_case
    await settled(c)
    state = await c.store.get_runtime_state(c.owner, legacy_history.history_key(c.handle.run_id))
    sid = next(iter(state["sessions"]))
    await c.store.delete_runtime_state(c.owner, legacy_history.history_key(c.handle.run_id))
    contract = await c.store.get_runtime_state(sid, "workflow_execution_contract")
    contract["spec"]["metadata"].pop("legacy_workflow_owner_sid", None)
    if owner_marker:
        contract["spec"]["metadata"]["legacy_workflow_owner_sid"] = owner_marker
    await c.store.set_runtime_state(sid, "workflow_execution_contract", contract)
    await c.store.merge_session_metadata(sid, {"legacy_workflow_owner_sid": None, "execution_role": None})
    result = await view(c, "logs", sid=sid)
    assert result["history_complete"] is False
    if owner_marker is None:
        assert result["status"] == "ok" and "PROOF-42" in str(result["rounds"])
    else:
        assert result["status"] == "unavailable" and result["reason"] == "execution_contract_mismatch"


async def test_old_foreach_contract_discovery_works_without_step_or_live_session_witnesses(legacy_case):
    c = legacy_case
    await settled(c, {"type": "foreach", "as": "item", "items": ["a", "b"], "body": leaf("BUILD")})
    saved = await journal.read(c.store, c.owner, c.handle.run_id)
    assert not saved["steps"] and not saved.get("live_sessions")
    await c.store.delete_runtime_state(c.owner, legacy_history.history_key(c.handle.run_id))
    listing = await view(c, "leaves")
    assert len(listing["nodes"]) == 2 and not listing["history_complete"]
    assert all(node["available"] for node in listing["nodes"])
    assert (await view(c, "logs", node="BUILD"))["error"]["code"] == "legacy_workflow_history_session_required"
    assert (await view(c, "logs", sid=listing["nodes"][0]["session_id"]))["status"] == "ok"


@pytest.mark.parametrize("fault", ["steps", "step_type", "step_node", "live", "live_sid", "clamp"])
async def test_corrupt_journal_never_becomes_metadata_authority_or_leaks(legacy_case, fault):
    c = legacy_case
    await settled(c)
    saved = await journal.read(c.store, c.owner, c.handle.run_id)
    key, value = {"steps": ("steps", {}), "step_type": ("steps", [None]), "step_node": ("steps", [{"node_id": []}]),
        "live": ("live_sessions", []), "live_sid": ("live_sessions", {"BUILD": True}), "clamp": ("allowed_tools", "*")}[fault]
    saved[key] = value
    await c.store.set_runtime_state(c.owner, journal.run_key(c.handle.run_id), saved)
    result = await view(c, "leaves")
    assert result["error"]["code"] == "legacy_workflow_history_journal_corrupt"
    assert "PROOF-42" not in str(result)


@pytest.mark.parametrize("source", ["messages", "projection"])
async def test_history_cursor_keeps_original_upper_bound_during_live_growth(legacy_case, source):
    c = legacy_case
    await settled(c)
    sid = (await view(c, "leaves"))["nodes"][0]["session_id"]
    first = await view(c, "logs", sid=sid, source=source, limit=1)
    assert first["next_cursor"]
    if source == "messages":
        await c.store.append_message(sid, role="assistant", content="ADDED-AFTER-CURSOR")
    else:
        await c.store.upsert_project_message(sid, send_index=999, kind="test-growth",
            content={"text": "ADDED-AFTER-CURSOR"}, rendered_text="ADDED-AFTER-CURSOR")
    seen, cursor = list(first["records"]), first["next_cursor"]
    while cursor:
        result = await view(c, "logs", sid=sid, source=source, limit=1, cursor=cursor)
        assert result["status"] == "ok"
        seen.extend(result["records"])
        cursor = result["next_cursor"]
    assert "ADDED-AFTER-CURSOR" not in str(seen)
    coord = "seq" if source == "messages" else "id"
    assert len({row[coord] for row in seen}) == len(seen)


@pytest.mark.parametrize("fault", ["authority_change", "foreign_sid", "selection"])
async def test_logs_cursor_cannot_cross_binding_session_or_selection(legacy_case, fault):
    c = legacy_case
    await settled(c)
    listing = await view(c, "leaves")
    sid, other = [n["session_id"] for n in listing["nodes"]]
    first = await view(c, "logs", sid=sid, limit=1)
    options = {"sid": sid, "limit": 1, "cursor": first["next_cursor"]}
    if fault == "authority_change":
        # The binding list changes even though the selected physical session survives.
        await c.store.close_session(other)
    elif fault == "foreign_sid":
        options["sid"] = other
    else:
        options["send_index"] = "1"
    result = await view(c, "logs", **options)
    assert result["error"]["code"] == "invalid_legacy_workflow_history_selection"


@pytest.mark.parametrize("source", ["messages", "projection"])
async def test_decimal_coordinate_reads_original_record_and_projection_export(legacy_case, source):
    c = legacy_case
    await settled(c)
    sid = (await view(c, "leaves"))["nodes"][0]["session_id"]
    bounds = await c.store.history_bounds(sid, source=source)
    coord = "seq" if source == "messages" else "id"
    result = await view(c, "logs", sid=sid, source=source, **{coord: str(bounds[1])})
    assert result["status"] == "ok" and result["format"] == "record_json_text"
    row = json.loads(result["text"])
    assert row[coord] == bounds[1]
    with runtime_env_context(RuntimeEnv(workspace_dir=c.workspace)):
        export = await view(c, "logs", sid=sid, source=source, output_file=source+".jsonl")
    assert export["status"] == "exported"
    records = [json.loads(line) for line in (c.workspace/(source+".jsonl")).read_text().splitlines()]
    assert len(records) == export["records"] and all(r["source"] == source and r["session_id"] == sid for r in records)


@pytest.mark.parametrize("fault", ["backend", "partial_export", "receipt_budget", "workspace"])
async def test_history_failures_are_sanitized_and_publish_no_partial_file(legacy_case, monkeypatch, fault):
    c = legacy_case
    await settled(c)
    options = {"node": "BUILD", "output_file": "private/final.jsonl"}
    if fault == "backend":
        async def unavailable(*args, **kwargs):
            raise ValueError("SECRET DATABASE DSN")
        monkeypatch.setattr(legacy_history, "collect", unavailable)
    elif fault == "partial_export":
        original = c.store.history_records
        async def interrupted(*args, **kwargs):
            if kwargs.get("_fetchall"):
                raise OSError("SECRET FILESYSTEM")
            return await original(*args, **kwargs)
        monkeypatch.setattr(c.store, "history_records", interrupted)
    elif fault == "receipt_budget":
        options.update(max_chars=1024, output_file="segment/"*100+"final.jsonl")
    with runtime_env_context(RuntimeEnv(workspace_dir=None if fault == "workspace" else c.workspace)):
        result = await view(c, "logs", **options)
    assert "error" in result and "SECRET" not in str(result)
    assert not list(c.workspace.rglob("*.jsonl")) and not list(c.workspace.rglob(".recall-*"))


async def test_admission_chains_existing_callback_and_restores_context(legacy_case):
    from power_loop.runtime.spec import _CHILD_ADMISSION

    c = legacy_case
    observed = []
    async def existing(sid):
        observed.append(sid)
    token = _CHILD_ADMISSION.set(existing)
    try:
        await settled(c)
        assert _CHILD_ADMISSION.get() is existing
    finally:
        _CHILD_ADMISSION.reset(token)
    registered = (await view(c, "leaves"))["nodes"]
    assert set(observed) == {node["session_id"] for node in registered}


async def test_nested_loop_foreach_preserves_every_actual_route(legacy_case):
    c = legacy_case
    root = {"type": "loop", "id": "outer", "max_iterations": 2,
        "until": {"ref": "batch", "op": "eq", "value": "never"}, "body": {
            "type": "foreach", "id": "batch", "as": "item", "items": ["a", "b"], "body": leaf("BUILD")}}
    await settled(c, root)
    result = await view(c, "leaves")
    assert len(result["nodes"]) == 4 and result["history_complete"]
    assert {(n["route"][0]["iteration"], n["route"][1]["index"]) for n in result["nodes"]} == {(1, 0), (1, 1), (2, 0), (2, 1)}


@pytest.mark.parametrize("strategy", ["retry", "fallback"])
async def test_retry_and_fallback_sessions_are_bound_before_actual_provider(legacy_case, monkeypatch, strategy):
    c = legacy_case
    original = c.model.complete
    async def limited(request, **kwargs):
        value = await original(request, **kwargs)
        if "ROLE-BUILD" in (request.system_prompt or "") and c.model.counts["BUILD"] <= 3:
            value.raw_text = ""
        return value
    monkeypatch.setattr(c.model, "complete", limited)
    root = leaf("BUILD", **({"retry": {"max_attempts": 2, "on": ["hit_round_limit"]}} if strategy == "retry" else {"fallback": leaf("FALLBACK")}))
    await settled(c, root)
    result = await view(c, "leaves")
    assert len(result["nodes"]) == 2 and result["history_complete"]
    if strategy == "retry":
        assert {n["attempt"] for n in result["nodes"]} == {1, 2}
        assert (await view(c, "logs", node="BUILD"))["error"]["code"] == "legacy_workflow_history_session_required"
    else:
        assert {n["node_id"] for n in result["nodes"]} == {"BUILD", "FALLBACK"}
    for node in result["nodes"]:
        assert (await view(c, "logs", sid=node["session_id"]))["status"] == "ok"


@pytest.mark.parametrize("options,code", [
    ({"node_job_id": "job_fake"}, "legacy_workflow_history_native_ids_unavailable"),
    ({"message_id": "1"}, "legacy_workflow_history_native_ids_unavailable"),
    ({"source": "wrong"}, "invalid_legacy_workflow_history_source"),
    ({"tail": True}, "invalid_legacy_workflow_history_tail"),
    ({"verbose": "yes"}, "invalid_legacy_workflow_history_verbose"),
    ({"tail": 2, "limit": 1}, "ambiguous_legacy_workflow_history_view"),
    ({"seq": True}, "invalid_legacy_workflow_history_seq"),
    ({"ids": ["1"]}, "invalid_legacy_workflow_history_ids"),
    ({"seq": "9007199254740993"}, None),
])
async def test_legacy_history_input_boundaries_preserve_explicit_semantics(legacy_case, options, code):
    c = legacy_case
    await settled(c)
    result = await view(c, "logs", node="BUILD", **options)
    if code:
        assert result["error"]["code"] == code
    else:
        assert result["status"] == "unavailable" and result["reason"] == "records_not_found"


async def test_bounded_preview_marks_clipping_and_single_record_chunks_reassemble(legacy_case):
    c = legacy_case
    await settled(c)
    sid = (await view(c, "leaves"))["nodes"][0]["session_id"]
    text = '详细原始内容"\\'*1700
    seq = await c.store.append_message(sid, role="assistant", content=text)
    preview = await view(c, "logs", sid=sid, tail=1, max_chars=1024)
    assert preview["preview_truncated"] and preview["rounds"][0]["truncated"]
    assert len(json.dumps(preview, ensure_ascii=False, separators=(",", ":"))) <= 1024
    offset, pieces = 0, []
    while offset is not None:
        result = await view(c, "logs", sid=sid, seq=str(seq), char_offset=offset, max_chars=1024)
        assert result["status"] == "ok"
        assert len(json.dumps(result, ensure_ascii=False, separators=(",", ":"))) <= 1024
        pieces.append(result["text"])
        offset = result["next_char_offset"]
    assert json.loads("".join(pieces))["content"] == text


@pytest.mark.parametrize("fault", ["missing_index", "too_many", "missing_contract", "deleted"])
async def test_unknown_old_history_is_explicit_and_sid_only_requires_owned_index(legacy_case, fault):
    c = legacy_case
    await settled(c)
    sid = (await view(c, "leaves"))["nodes"][0]["session_id"]
    await c.store.delete_runtime_state(c.owner, legacy_history.history_key(c.handle.run_id))
    if fault in ("missing_index", "too_many"):
        await c.store.set_runtime_state(c.owner, journal.INDEX_KEY, [] if fault == "missing_index" else [c.handle.run_id]*101)
        result = await invoke(c.loop, c.owner, {"action": "logs", "sid": sid})
        assert "error" in result
    else:
        if fault == "missing_contract":
            await c.store.delete_runtime_state(sid, "workflow_execution_contract")
        else:
            await c.store.close_session(sid)
        result = await view(c, "logs", sid=sid)
        assert result["status"] == "unavailable" and result["reason"] in ("execution_contract_missing", "session_unavailable")
    assert "PROOF-42" not in str(result)


@pytest.mark.parametrize("run_id", ["run-readable", "中文 场景001", "sdk-"+"x"*300])
async def test_sdk_opaque_run_ids_restore_and_do_not_poison_sid_only_lookup(legacy_case, run_id):
    from power_loop.workflow import WorkflowSpec
    from power_loop.workflow.engine import WorkflowEngine
    from power_loop.workflow.resume import resume_run
    from power_loop.workflow.runner import make_on_step

    c = legacy_case
    c.model.hold = set()
    spec = WorkflowSpec.from_json({"name": "old-sdk", "root": leaf("BUILD")})
    await journal.seed(c.store, c.owner, run_id, spec.name, spec=spec.to_dict())
    # Emulate the old SDK's direct engine, which had no history index/owner marker.
    result = await WorkflowEngine(c.loop, run_id=run_id, on_step=make_on_step(c.store, c.owner, run_id)).run(spec)
    assert result.status == "completed"
    sid = result.results["BUILD"].session_id
    counts = c.model.counts.copy()
    restored = await resume_run(c.loop, c.owner, run_id)
    assert restored.status == "completed" and c.model.counts == counts
    history = await invoke(c.loop, c.owner, {"action": "logs", "sid": sid})
    assert history["status"] == "ok" and history["run_id"] == run_id and "PROOF-42" in str(history["rounds"])
    # Model-facing explicit run selectors retain their existing shape checks.
    denied = await invoke(c.loop, c.owner, {"action": "leaves", "run_id": run_id})
    assert denied["error"]["code"] == "workflow_run_id_required"
