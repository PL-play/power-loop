"""Task-qualified history: coordinates, authorization, bounded reads and exports."""
from __future__ import annotations

import asyncio
import base64
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from power_loop.core.agent_context import (
    reset_current_loop,
    reset_session_id,
    set_current_loop,
    set_session_id,
)
from power_loop.runtime import task_history
from power_loop.runtime.env import RuntimeEnv, runtime_env_context
from power_loop.runtime.jobs import JobActor, JobError, JobManager
from power_loop.tools.default_tools import run_recall_send

from .test_job_registry_192 import registry_store

__all__ = ['registry_store']
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.fixture
async def audit(registry_store, tmp_path):
    store, connect = registry_store
    sid = await store.create_session()
    manager, actor = JobManager(store), JobActor("history", "host", sid)
    reg = await manager.register_job(actor, kind="send", session_id=sid, system_key="original-send", request={"send_index": 1})
    await manager.claim_job(actor, reg.job.job_id, executor_id="executor", boot_id="boot", expected_version=1)
    await store.append_message(sid, role="user", content="原请求", send_index=1)
    await store.append_message(sid, role="assistant", content="ORIGINAL-42", send_index=1)
    await store.append_message(sid, role="tool", name="saved", content="SAVED-RESULT", tool_call_id="saved-call", send_index=1)
    await store.append_message(sid, role="user", content="FUTURE-PRIVATE", send_index=2)
    await store.upsert_project_message(sid, send_index=1, kind="user", content={"text": "原请求"})
    await store.upsert_project_message(sid, send_index=1, kind="project", content={"text": "SUMMARY-42"}, source_seq_lo=1, source_seq_hi=3)
    await store.upsert_project_message(sid, send_index=1, kind="compact", content={"text": "COMPACT-PRIVATE"}, compact_from_send=1, compact_to_send=2)
    other = await store.create_session()
    await store.append_message(other, role="user", content="OTHER-PRIVATE", send_index=1)
    return SimpleNamespace(store=store, manager=manager, actor=actor, job=reg.job.job_id, sid=sid, other=other,
        workspace=tmp_path, connect=connect)


async def read(c, **kwargs):
    return await task_history.read_history(c.manager, c.actor, c.job, **kwargs)


async def test_exact_raw_bigint_id_preserves_both_id_and_seq_without_rounding(audit):
    c = audit
    identity = 2**53+123
    await c.manager.db.execute(f"UPDATE {c.store.t.messages} SET id=? WHERE session_id=? AND seq=2", (identity, c.sid))
    result = await read(c, message_id=str(identity))
    value = json.loads(result["text"])
    assert value["id"] == str(identity) and value["seq"] == "2"
    assert value["message_ref"]["id"] == str(identity) and value["message_ref"]["seq"] == "2"
    assert result["execution_authority"] is False and value["content"] == "ORIGINAL-42"


@pytest.mark.parametrize("source", ["messages", "projection"])
async def test_original_send_scope_never_reads_next_send_or_multisend_compact(audit, source):
    c = audit
    result = await read(c, source=source)
    text = json.dumps(result, ensure_ascii=False)
    assert result["status"] == "ok" and result["send_index"] == "1"
    assert "FUTURE-PRIVATE" not in text and "COMPACT-PRIVATE" not in text
    assert len(result["records"]) == (3 if source == "messages" else 2)


async def test_exact_message_ids_are_not_seq_ranges_and_ignore_unowned_records(audit):
    c = audit
    raw = await c.store.get_message(c.sid, 2)
    foreign = (await c.store.load_all_messages(c.other))[0]
    result = await read(c, message_ids=[str(raw.id), str(foreign.id)])
    assert [r["seq"] for r in result["records"]] == ["2"]
    assert "OTHER-PRIVATE" not in json.dumps(result)
    ranged = await read(c, ids=["1", "2"])
    assert [r["seq"] for r in ranged["records"]] == ["1", "2"]


async def test_raw_compacted_and_dropped_originals_remain_readable(audit):
    c = audit
    await c.manager.db.execute(f"UPDATE {c.store.t.messages} SET state='compacted_out' WHERE session_id=? AND seq=2", (c.sid,))
    await c.manager.db.execute(f"UPDATE {c.store.t.messages} SET state='dropped' WHERE session_id=? AND seq=3", (c.sid,))
    assert json.loads((await read(c, seq=2))["text"])["state"] == "compacted_out"
    assert json.loads((await read(c, seq=3))["text"])["state"] == "dropped"


async def test_pagination_keeps_captured_upper_bound_and_rechecks_selection(audit):
    c = audit
    first = await read(c, limit=1)
    assert first["upper"] == "3" and first["next_cursor"]
    await c.store.append_message(c.sid, role="assistant", content="LATE-SAME-SEND", send_index=1)
    pages, cursor = [first], first["next_cursor"]
    while cursor:
        page = await read(c, limit=1, cursor=cursor)
        pages.append(page)
        cursor = page["next_cursor"]
    assert [r["seq"] for p in pages for r in p["records"]] == ["1", "2", "3"]
    assert "LATE-SAME-SEND" not in str(pages)
    with pytest.raises(JobError, match="cursor"):
        await read(c, cursor=first["next_cursor"], source="projection")


@pytest.mark.parametrize("mode", ["other_actor", "tenant", "principal", "sid", "send_index"])
async def test_scope_cannot_expand_to_other_principal_session_or_send(audit, mode):
    c = audit
    actor, options = c.actor, {}
    if mode == "other_actor":
        actor = replace(actor, session_id=c.other)
    elif mode in ("tenant", "principal"):
        actor = replace(actor, **{mode+"_id": "foreign"})
    elif mode == "sid":
        options["session_id"] = c.other
    else:
        options["send_index"] = 2
    with pytest.raises(JobError):
        await task_history.read_history(c.manager, actor, c.job, **options)


@pytest.mark.parametrize("options", [{"message_id": True}, {"message_id": "1e5"}, {"message_id": str(2**63)},
    {"message_ids": []}, {"ids": [2, 1]}, {"seq": 2, "message_id": "2"}, {"source": "projection", "seq": 2},
    {"id": 2}, {"limit": True}, {"max_chars": 100}, {"char_offset": 0}, {"cursor": "[]"}])
async def test_malformed_selection_and_cursor_are_explicit_rejections(audit, options):
    with pytest.raises(JobError):
        await read(audit, **options)


async def test_forged_projection_source_cannot_leak_another_send(audit):
    c = audit
    await c.store.upsert_project_message(c.sid, send_index=1, kind="project", content={"text": "FUTURE-PRIVATE"},
        source_seq_lo=4, source_seq_hi=4)
    with pytest.raises(JobError, match="projection_scope_mismatch"):
        await read(c, source="projection")


async def test_single_record_unicode_chunks_are_bounded_and_inline_media_is_omitted(audit):
    c = audit
    body = ('中文 "\\\n'*600)+"data:image/png;base64,QUJDRA=="
    await c.manager.db.execute(f"UPDATE {c.store.t.messages} SET content=? WHERE session_id=? AND seq=2", (body, c.sid))
    offset, pieces = 0, []
    while True:
        result = await read(c, seq=2, char_offset=offset, max_chars=2400)
        assert len(json.dumps(result, ensure_ascii=False, separators=(",", ":"))) <= 2400
        pieces.append(result["text"])
        offset = result["next_char_offset"]
        if offset is None:
            break
    value = json.loads("".join(pieces))
    assert "[inline media omitted]" in value["content"] and "QUJDRA==" not in value["content"]
    assert value["message_ref"]["seq"] == "2"


async def test_export_writes_complete_authorized_originals_as_jsonl_without_model_body(audit):
    c = audit
    with runtime_env_context(RuntimeEnv(workspace_dir=c.workspace)):
        result = await read(c, output_file="audit/original.jsonl")
    values = [json.loads(line) for line in (c.workspace/"audit/original.jsonl").read_text().splitlines()]
    assert result["status"] == "exported" and result["records"] == 3 and "text" not in result
    assert all(v["source"] == "messages" and v["send_index"] == "1" and isinstance(v["id"], str) for v in values)
    assert "FUTURE-PRIVATE" not in str(values)


@pytest.mark.parametrize("fault", ["existing", "symlink", "outside", "records", "bytes", "cancel", "binding"])
async def test_export_refuses_overwrite_scope_escape_limits_and_cleans_partial_files(audit, monkeypatch, fault):
    c = audit
    target = c.workspace/"audit.jsonl"
    output = "audit.jsonl"
    if fault == "existing":
        target.write_text("KEEP")
    elif fault == "symlink":
        target.symlink_to(c.workspace/"missing.jsonl")
    elif fault == "outside":
        output = str(c.workspace.parent/"outside-audit.jsonl")
    elif fault in ("records", "bytes"):
        monkeypatch.setattr(task_history, "_EXPORT_MAX_RECORDS" if fault == "records" else "_EXPORT_MAX_BYTES", 1)
    else:
        original = task_history._export
        async def changed(*args, **kwargs):
            if fault == "binding":
                await c.store.close_session(c.sid)
            else:
                async def cancelled(*args):
                    raise asyncio.CancelledError()
                kwargs["read_records"] = cancelled
            return await original(*args, **kwargs)
        monkeypatch.setattr(task_history, "_export", changed)
    with runtime_env_context(RuntimeEnv(workspace_dir=c.workspace)):
        with pytest.raises(asyncio.CancelledError if fault == "cancel" else JobError):
            await read(c, output_file=output)
    assert not list(c.workspace.glob(".recall-*"))
    if fault == "existing":
        assert target.read_text() == "KEEP"
    elif fault == "symlink":
        assert target.is_symlink() and not target.exists()
    else:
        assert not target.exists()




async def test_model_entry_sanitizes_unknown_backend_valueerror_and_legacy_rejects_task_mode(audit, monkeypatch):
    c = audit
    loop = SimpleNamespace(store=c.store, config=None, task_runtime=SimpleNamespace(manager=c.manager, actor=lambda sid: c.actor))
    lt, st = set_current_loop(loop), set_session_id(c.sid)
    original = task_history.read_history
    async def unavailable(*args, **kwargs):
        raise ValueError("PRIVATE-CONNECTION-SECRET")
    try:
        monkeypatch.setattr(task_history, "read_history", unavailable)
        result = json.loads(await run_recall_send(job_id=c.job))
        assert result == {"version": 1, "error": {"code": "task_history_backend_unavailable"}}
        assert "PRIVATE" not in str(result)
        monkeypatch.setattr(task_history, "read_history", original)
        loop.task_runtime = None
        assert json.loads(await run_recall_send(job_id=c.job))["error"]["code"] == "task_registry_required"
    finally:
        reset_session_id(st)
        reset_current_loop(lt)














@pytest.mark.parametrize("field", ["query", "binding", "upper", "after", "snapshot_at", "extra"])
async def test_cursor_tampering_never_expands_read_authority(audit, field):
    c = audit
    page = await read(c, limit=1)
    state = json.loads(base64.urlsafe_b64decode(page["next_cursor"]+"="*(-len(page["next_cursor"])%4)))
    if field in ("query", "binding"):
        state[field] = "0"*64
    elif field in ("upper", "after"):
        state[field] = str(2**63)
    elif field == "snapshot_at":
        state[field] = True
    else:
        state[field] = "unexpected"
    cursor = base64.urlsafe_b64encode(json.dumps(state).encode()).decode()
    with pytest.raises(JobError):
        await read(c, cursor=cursor)


async def test_projection_pages_report_mutability_and_follow_same_coordinate_upper_bound(audit):
    c = audit
    first = await read(c, source="projection", limit=1)
    await c.store.upsert_project_message(c.sid, send_index=1, kind="project", content={"text": "UPDATED-SUMMARY"},
        source_seq_lo=1, source_seq_hi=3)
    second = await read(c, source="projection", limit=1, cursor=first["next_cursor"])
    assert first["upper"] == second["upper"] and second["projection_content_mutable"]
    assert second["snapshot_kind"] == "coordinate_upper_bound" and "UPDATED-SUMMARY" in str(second["records"])


async def test_deleted_session_is_unavailable_not_an_empty_successful_export(audit):
    c = audit
    page = await read(c, limit=1)
    await c.store.close_session(c.sid)
    deleted = await read(c, cursor=page["next_cursor"])
    assert deleted["status"] == "unavailable" and deleted["reason"] == "session_deleted"
    with runtime_env_context(RuntimeEnv(workspace_dir=c.workspace)):
        result = await read(c, output_file="deleted.jsonl")
    assert result["status"] == "unavailable" and not (c.workspace/"deleted.jsonl").exists()


async def test_export_receipt_budget_is_checked_before_publishing_any_file(audit):
    c = audit
    with runtime_env_context(RuntimeEnv(workspace_dir=c.workspace)):
        with pytest.raises(JobError, match="budget_too_small"):
            await read(c, max_chars=1024, output_file=("segment/"*100)+"audit.jsonl")
    assert not (c.workspace/"segment").exists() and not list(c.workspace.rglob("*.jsonl"))


async def test_export_rechecks_store_identity_and_does_not_publish_on_protocol_change(audit, monkeypatch):
    c = audit
    original = task_history._export
    async def replaced(*args, **kwargs):
        await c.manager.db.execute(f"UPDATE {c.manager.prefix}task_store SET protocol_version=999 WHERE id=1")
        return await original(*args, **kwargs)
    monkeypatch.setattr(task_history, "_export", replaced)
    try:
        with runtime_env_context(RuntimeEnv(workspace_dir=c.workspace)):
            with pytest.raises(JobError, match="store_identity_changed"):
                await read(c, output_file="store-change.jsonl")
        assert not (c.workspace/"store-change.jsonl").exists() and not list(c.workspace.glob(".recall-*"))
    finally:
        await c.manager.db.execute(f"UPDATE {c.manager.prefix}task_store SET protocol_version=? WHERE id=1", (task_history.TASK_PROTOCOL_VERSION,))


async def test_export_link_race_keeps_the_other_writers_file_and_cleans_temp(audit, monkeypatch):
    from power_loop.runtime import recall

    c = audit
    original = recall.os.link
    target = c.workspace/"raced.jsonl"
    def race(source, dest):
        target.write_text("OTHER-WRITER")
        return original(source, dest)
    monkeypatch.setattr(recall.os, "link", race)
    with runtime_env_context(RuntimeEnv(workspace_dir=c.workspace)):
        with pytest.raises(JobError, match="export_failed"):
            await read(c, output_file="raced.jsonl")
    assert target.read_text() == "OTHER-WRITER" and not list(c.workspace.glob(".recall-*"))




@pytest.mark.parametrize("options", [{"session_id": ""}, {"source": "bogus"}, {"ids": []},
    {"output_file": "x", "seq": 1, "char_offset": 0}, {"cursor": ""}])
async def test_invalid_history_inputs_are_rejected_before_query_or_export(audit, options):
    with pytest.raises(JobError):
        await read(audit, **options)


async def test_export_requires_an_explicit_workspace(audit):
    with runtime_env_context(RuntimeEnv()):
        with pytest.raises(JobError, match="workspace_required"):
            await read(audit, output_file="missing.jsonl")


async def test_deleted_page_upper_bound_reports_changed_records(audit):
    c = audit
    first = await read(c, limit=1)
    await c.manager.db.execute(f"DELETE FROM {c.store.t.messages} WHERE session_id=? AND seq=3", (c.sid,))
    result = await read(c, cursor=first["next_cursor"])
    assert result["status"] == "unavailable" and result["reason"] == "records_changed"


@pytest.mark.parametrize("budget", [1024, 1400, 2200])
async def test_list_budget_never_skips_a_record_when_previews_are_trimmed(audit, budget):
    c = audit
    cursor, seen = None, []
    try:
        while True:
            result = await read(c, cursor=cursor, max_chars=budget)
            assert len(json.dumps(result, ensure_ascii=False, separators=(",", ":"))) <= budget
            seen.extend(row["seq"] for row in result["records"])
            cursor = result["next_cursor"]
            if cursor is None:
                break
    except JobError as exc:
        assert exc.code == "task_history_budget_too_small" and not seen
    else:
        assert seen == ["1", "2", "3"]
