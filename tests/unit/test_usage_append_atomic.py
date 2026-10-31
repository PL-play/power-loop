"""Round accounting shares the first response append without losing originals."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from power_loop import AgentLoopConfig, LLMResponse, LLMService, LLMTokenUsage, StatefulAgentLoop
from power_loop.agent.sink import SQLiteSink
from power_loop.runtime.jobs import JobManager, _hash
from power_loop.runtime.task_runtime import TaskRuntime

from ._budget_checkpoint_crash_worker import config
from ._usage_append_crash_worker import UsageHost
from .test_job_registry_192 import registry_store
from .test_send_sql_profile import Profiled

__all__ = ["registry_store"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def usage_rows(store, sid):
    return await store._db.fetchall(f"SELECT * FROM {store.t.usage_rounds} WHERE session_id=?", (sid,))


@pytest.mark.parametrize("stage", ["after_message", "after_usage", "before_round_end"])
async def test_process_crash_keeps_message_usage_atomic_and_supplier_charge_original(registry_store, tmp_path, stage):
    store, connect = registry_store
    sid = await store.create_session()
    backend = store._db.dialect.name
    dsn = str(tmp_path / "registry.db") if backend == "sqlite" else os.environ[
        "POWER_LOOP_TASK_PG_DSN" if backend == "postgres" else "POWER_LOOP_TASK_MYSQL_DSN"]
    evidence = tmp_path / "usage-crash.json"
    process = await asyncio.create_subprocess_exec(sys.executable, "-m", "tests.unit._usage_append_crash_worker",
        dsn, store.table_prefix, sid, str(evidence), stage, cwd=Path(__file__).resolve().parents[2],
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    try:
        output, _ = await asyncio.wait_for(process.communicate(), 30)
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()
    assert process.returncode == {"after_message": 93, "after_usage": 94, "before_round_end": 95}[stage], output.decode()
    saved = json.loads(evidence.read_text())
    reader = await connect(schema="compatible")
    rows = await reader.load_all_messages(sid)
    accounted = await usage_rows(reader, sid)
    committed = stage == "before_round_end"
    assert [row.role for row in rows] == (["user", "assistant"] if committed else ["user"])
    assert len(accounted) == int(committed)
    if committed:
        assert (accounted[0]["send_index"], accounted[0]["round_index"], accounted[0]["total_tokens"]) == (1, 0, 12)
    charges = Path(str(evidence) + ".charges")
    assert len(json.loads(charges.read_text())) == 1
    submission = json.loads(Path(str(charges) + ".submission").read_text())
    assert submission["request"]["evidence"]["usage"]["total_tokens"] == 12
    capture = json.loads(Path(str(charges) + ".capture").read_text())
    assert capture["usage"]["total_tokens"] == 12
    assert capture["context"]["attempt_id"] == submission["attempt_id"]
    prepared = await reader._db.fetchone(f"SELECT payload_json FROM {reader.table_prefix}job_events WHERE event_id=?",
        (capture["context"]["prepared_event_id"],))
    assert _hash(json.loads(prepared["payload_json"])) == capture["context"]["prepared_hash"]
    host = UsageHost(charges)
    runtime = TaskRuntime(JobManager(reader), tenant_id="budget-tenant", principal_id="host",
        executor_id="reconciler", boot_id="new-boot", boot_contract_hash="a" * 64, budget_config=config(host))
    try:
        for _ in range(2):
            result = await runtime.repair_execution_checkpoint(sid, saved["job_id"], generation=saved["generation"])
            assert result["needs_reconciliation"] and not result["replay_allowed"]
            # The last control checkpoint can precede the supplier's return.
            # Preserve its entire grant; the supplier capture/charge above is
            # the original usage authority, never a guessed control summary.
            [allocation] = await reader._db.fetchall(f"SELECT state_json FROM {reader.table_prefix}execution_allocations")
            booked = json.loads(allocation["state_json"])["booked"]
            assert booked == dict(used_calls=0, used_tokens=0, held_calls=1, held_tokens=100, inflight=1)
        assert len(json.loads(charges.read_text())) == 1 and host.reserve_calls == 0
        assert saved["calls"] == 1
    finally:
        await runtime.shutdown(timeout_s=1)


@pytest.mark.parametrize("fault", ["rollback", "ack_loss"])
async def test_failed_or_unknown_message_commit_has_the_same_usage_outcome(registry_store, monkeypatch, fault):
    store, connect = registry_store
    sid = await store.create_session()
    sink = SQLiteSink(store, sid)
    sink.send_index = 1
    transaction = store._db.transaction

    @asynccontextmanager
    async def fail():
        async with transaction() as tx:
            yield tx
            if fault == "rollback":
                raise OSError("before_commit")
        raise OSError("committed_ack_lost")

    with monkeypatch.context() as patch:
        patch.setattr(store._db, "transaction", fail)
        with pytest.raises(OSError, match="before_commit|committed_ack_lost"):
            await sink.on_message_appended({"role": "assistant", "content": "original", "send_index": 1,
                "_round_usage": {"prompt_tokens": 7, "completion_tokens": 5, "total_tokens": 12}}, round_index=0)
    cold = await connect(schema="compatible")
    assert len(await cold.load_all_messages(sid)) == len(await usage_rows(cold, sid)) == int(fault == "ack_loss")


async def test_round_end_is_idempotent_after_usage_commits_with_message(registry_store):
    store, _ = registry_store
    sid = await store.create_session()
    sink = SQLiteSink(store, sid)
    sink.send_index = 1
    usage = {"prompt_tokens": 7, "completion_tokens": 5, "total_tokens": 12}
    original = store._db
    store._db = counted = Profiled(original)
    try:
        await sink.on_message_appended({"role": "assistant", "content": "original", "send_index": 1,
                                       "_round_usage": usage}, round_index=0)
        assert counted.transactions == 1
        before = len(counted.records), counted.transactions
        for _ in range(3):
            await sink.on_round_ended(0, usage=usage)
        assert (len(counted.records), counted.transactions) == before
        [row] = await usage_rows(store, sid)
        assert row["total_tokens"] == 12
    finally:
        store._db = original


async def test_empty_response_rounds_keep_usage_and_private_accounting_never_reaches_the_model(registry_store):
    store, _ = registry_store

    class Model(LLMService):
        calls = 0
        async def complete(self, request, **kwargs):
            self.calls += 1
            assert all("_round_usage" not in row for row in request.messages)
            return LLMResponse(raw_text="" if self.calls == 1 else "done", token_usage=LLMTokenUsage(
                prompt_tokens=7, completion_tokens=5, total_tokens=12))
        def stream(self, request):
            raise AssertionError("nonstream fixture")
        async def close(self):
            pass

    model = Model()
    loop = StatefulAgentLoop(llm=model, store=store, config=AgentLoopConfig(max_rounds=3, compactor=None))
    try:
        sid = await loop.new_session()
        assert (await loop.send("respond", sid)).status == "completed"
        rows = await usage_rows(store, sid)
        assert {(row["round_index"], row["total_tokens"]) for row in rows} == {(0, 12), (1, 12)}
        assert model.calls == 2
    finally:
        await loop.aclose(drain_timeout_s=.1)
