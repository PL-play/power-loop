"""Explicit release rehearsal against a real historical SDK checkout.

Not in default testpaths: run explicitly with POWER_LOOP_BRIDGE_LEGACY_SDK set
to a clean protocol-6 checkout. Server DSNs must name disposable *_test DBs.
There is no version monkeypatch, production DSN, implicit migration or skip.
"""
from __future__ import annotations

import asyncio
import json
import os
import secrets
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

import pytest

from power_loop.runtime import model_budgets
from power_loop.runtime.jobs import JobActor, JobManager
from power_loop.runtime.store import task_schema
from power_loop.runtime.store.factory import _make_database, open_store
from power_loop.runtime.store.schema import _STORE_TABLES, StoreSchemaError
from power_loop.runtime.task_runtime import TaskRuntime
from tests.real.test_real_budget_tree_192 import RealBudgetHost

pytestmark = pytest.mark.asyncio
WORKER = Path(__file__).with_name("_schema_bridge_worker.py")
LEGACY_PROTOCOL = int(os.environ.get("POWER_LOOP_BRIDGE_LEGACY_PROTOCOL", "6"))


async def old_process(config):
    root = Path(os.environ["POWER_LOOP_BRIDGE_LEGACY_SDK"]).resolve()
    assert (root / "power_loop/runtime/store/task_schema.py").is_file()
    env = {**os.environ, "PYTHONPATH": str(root)}
    result = await asyncio.to_thread(subprocess.run, [sys.executable, str(WORKER)],
        cwd=root, env=env, input=json.dumps({**config, "protocol": LEGACY_PROTOCOL}),
        capture_output=True, text=True, timeout=110)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


async def held_writer(config):
    root = Path(os.environ["POWER_LOOP_BRIDGE_LEGACY_SDK"]).resolve()
    proc = await asyncio.create_subprocess_exec(sys.executable, str(WORKER), cwd=root,
        env={**os.environ, "PYTHONPATH": str(root)}, stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    try:
        proc.stdin.write((json.dumps({**config, "protocol": LEGACY_PROTOCOL, "action": "held_writer"}) + "\n").encode())
        await proc.stdin.drain()
        ready = json.loads(await asyncio.wait_for(proc.stdout.readline(), 15))
        assert ready == {"ready": True, "store_id": config["store_id"]}
        return proc
    except BaseException:
        if proc.returncode is None:
            proc.kill()
        await proc.communicate()
        raise


async def snapshot(db, prefix):
    # Full reads are deliberate acceptance checks on a private, tiny fixture.
    result = {}
    for table in (*_STORE_TABLES, *task_schema.TASK_TABLES):
        rows = [dict(r) for r in await db.fetchall(f"SELECT * FROM {prefix}{table}")]
        result[table] = sorted(rows, key=lambda r: json.dumps(r, sort_keys=True, default=str))
    return result


class ControlHost(RealBudgetHost):
    async def prepare(self, *args, **kwargs):
        raise AssertionError("control bridge must not prepare another paid request")

    async def reserve(self, *args, **kwargs):
        raise AssertionError("control bridge must not reserve again")

    async def lookup(self, *args, **kwargs):
        raise AssertionError("recorded original grant already exists")

    async def release(self, *args, **kwargs):
        raise AssertionError("recovery is not permission to refund unknown resources")


@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
@pytest.mark.parametrize("real", [False, pytest.param(True, marks=pytest.mark.real_llm)])
async def test_actual_legacy_protocol_upgrade_original_control_and_accounting(
        tmp_path, monkeypatch, record_property, backend, real, *, backup_restore=False):
    dsn = str(tmp_path / "bridge.db") if backend == "sqlite" else os.environ["POWER_LOOP_TASK_PG_DSN"]
    if backend != "sqlite":
        assert urlparse(dsn).path.endswith("_test"), "release rehearsal refuses non-test database"
    prefix = "bridge_" + secrets.token_hex(4) + "_"
    db, store, runtime, stale = None, None, None, None
    config = {"dsn": dsn, "prefix": prefix, "real": real, "action": "seed", "large_ids": backup_restore}
    try:
        original = await old_process(config)
        record_property("original_writer_pid", original["worker_pid"])
        record_property("original_identity", json.dumps(original))
        record_property("real_provider_requests", original["requests"])
        record_property("reported_tokens", original["reported_tokens"])
        db = await _make_database(dsn)
        before = await snapshot(db, prefix)
        assert before["task_store"][0]["protocol_version"] == LEGACY_PROTOCOL
        if backup_restore:
            from scripts.acceptance._backup_restore import restore_copy
            source_dsn = dsn
            destination = (str(tmp_path / 'restored-legacy.db') if backend == 'sqlite'
                else os.environ['POWER_LOOP_TASK_RESTORE_PG_DSN'])
            await db.close()
            db = None
            record = await restore_copy(dsn, destination, prefix=prefix,
                archive=tmp_path / 'legacy.backup', backend=backend)
            record_property('legacy_backup', json.dumps(record))
            for pid in record['utility_pids']:
                record_property('backup_utility_pid', pid)
            dsn, config['dsn'] = destination, destination
            db = await _make_database(dsn)
            assert await snapshot(db, prefix) == before
            assert int(original['call']['id']) > 2**53 and int(original['receipt']['id']) > 2**53
        stale = await held_writer({**config, **original})
        record_property("stale_writer_pid", stale.pid)
        for mode in ("compatible", "tasks"):
            with pytest.raises(StoreSchemaError):
                await open_store(dsn, table_prefix=prefix, schema=mode)
        with pytest.raises(StoreSchemaError, match="maintenance_required"):
            await task_schema.migrate_task_schema(db, prefix)

        # DDL/stamp must roll back together on supported transactional backends.
        original_ddl = task_schema._run_task_ddl
        async def failed_ddl(*args):
            await original_ddl(*args)
            raise OSError("injected migration failure after DDL")
        with monkeypatch.context() as patch:
            patch.setattr(task_schema, "_run_task_ddl", failed_ddl)
            with pytest.raises(OSError, match="injected migration failure"):
                await task_schema.migrate_task_schema(db, prefix, maintenance=True)
        assert await snapshot(db, prefix) == before
        await task_schema.migrate_task_schema(db, prefix, maintenance=True)
        await task_schema.migrate_task_schema(db, prefix, maintenance=True)
        await task_schema.verify_task_schema(db, prefix)
        after = await snapshot(db, prefix)
        assert after["task_store"] == [{**before["task_store"][0], "protocol_version": task_schema.TASK_PROTOCOL_VERSION}]
        assert {k: v for k, v in after.items() if k != "task_store"} == {
            k: v for k, v in before.items() if k != "task_store"}
        stdout, stderr = await asyncio.wait_for(stale.communicate(b"try-write\n"), 15)
        assert stale.returncode == 0, stderr.decode()
        assert json.loads(stdout) == {"stale_write_refused": True}
        assert await old_process({**config, "action": "reopen"}) == {"refused": ["tasks", "compatible"]}
        assert await snapshot(db, prefix) == after
        if backup_restore:
            destination = (str(tmp_path / 'restored-current.db') if backend == 'sqlite'
                else os.environ['POWER_LOOP_TASK_RESTORE_CURRENT_PG_DSN'])
            await db.close()
            db = None
            record = await restore_copy(dsn, destination, prefix=prefix,
                archive=tmp_path / 'current.backup', backend=backend)
            record_property('current_backup', json.dumps(record))
            for pid in record['utility_pids']:
                record_property('backup_utility_pid', pid)
            dsn, config['dsn'] = destination, destination
            db = await _make_database(dsn)
            assert await snapshot(db, prefix) == after
            # The original backup source remains untouched at its old protocol.
            source = await _make_database(source_dsn)
            try:
                assert await snapshot(source, prefix) == before
            finally:
                await source.close()

        store = await open_store(dsn, table_prefix=prefix, schema="compatible")
        assert store.store_id == original["store_id"]
        manager = JobManager(store)
        actor = JobActor("bridge-test", "host", original["queued_sid"])
        op = await manager.get_operation(actor, original["operation_id"])
        assert op.call_ref == original["call"] and op.receipt_ref == original["receipt"]
        cancelled = await manager.request_cancel(actor, original["queued_job"], system_key="bridge-stop", reason="stop")
        assert await manager.request_cancel(actor, original["queued_job"], system_key="bridge-stop", reason="stop") == cancelled
        assert (await manager.get_job(actor, original["queued_job"])).cancel_requested_at is not None

        host = ControlHost()
        usage_actor = JobActor("bridge-test", "host", original["sid"])
        budget = await model_budgets.recovery_config(manager, usage_actor, original["job_id"], original["attempt_id"],
            provider_key="original-budget", provider=host)
        runtime = TaskRuntime(manager, tenant_id="bridge-test", principal_id="host",
            executor_id="control", boot_id="control", boot_contract_hash="a" * 64, budget_config=budget)
        for _ in range(2):
            result = await runtime.reconcile_budget_usage(original["sid"],
                job_id=original["job_id"], attempt_id=original["attempt_id"])
            assert result["usage_acknowledged"] and not result["execution_authority"]
        assert list(host.applied) == [original["attempt_id"]]
        state = await model_budgets.get_budget(manager, usage_actor, original["attempt_id"])
        assert state["ack"] and not state["pending"]
        final = await snapshot(db, prefix)
        for table in ("messages", "task_budget_policies", "task_budget_scopes", "task_executors", "task_executor_events"):
            assert final[table] == before[table], table
        assert len(final["jobs"]) == len(before["jobs"])
        assert len(final["task_budget_attempts"]) == 1
        record_property("migration_and_control", "original messages/IDs/grants/counters retained; no new boot or model")
        if backup_restore:
            previous_max = max(row['id'] for row in final['messages'])
            seq = await store.append_message(original['queued_sid'], role='user', content='after restored control')
            next_ref = await store.message_ref(original['queued_sid'], seq)
            assert next_ref.id > previous_max
            assert next_ref.to_dict()['id'] == str(next_ref.id)
            assert (await store.message_by_id(original['queued_sid'], original['call']['id'])).tool_calls
            exported = await store.export_session(original['queued_sid'])
            assert all('id' not in row for row in exported['tables']['messages'])
            record_property('restored_new_message_id', str(next_ref.id))
            from power_loop.runtime import task_history
            from power_loop.runtime.env import RuntimeEnv, runtime_env_context
            from scripts.acceptance._backup_restore import command
            ids = [original['call']['id'], original['receipt']['id']]
            with runtime_env_context(RuntimeEnv(workspace_dir=tmp_path)):
                report = await task_history.read_history(manager, actor, original['queued_job'],
                    message_ids=ids, output_file='restored/original.jsonl')
            exported = (tmp_path / 'restored/original.jsonl').read_bytes()
            values = [json.loads(line) for line in exported.splitlines()]
            assert report['status'] == 'exported' and report['records'] == 2
            assert [v['id'] for v in values] == ids
            assert [v['message_ref'] for v in values] == [original['call'], original['receipt']]
            # Exercise a real JavaScript reader, not Python's unbounded integers.
            js = "const fs=require('fs');const ids=JSON.parse(process.argv[1]);" + \
                "const rows=fs.readFileSync(0,'utf8').trim().split('\\n').map(JSON.parse);" + \
                "if(rows.some((r,i)=>typeof r.id!=='string'||r.id!==ids[i]))throw Error('precision lost');" + \
                "process.stdout.write(JSON.stringify(rows.map(r=>r.id)));"
            output, pid = await command(['node','-e',js,json.dumps(ids)], input_data=exported)
            assert json.loads(output) == ids
            record_property('javascript_reader_pid', pid)
            record_property('restored_jsonl_ids', json.dumps(ids))
    finally:
        if stale and stale.returncode is None:
            stale.kill()
            await stale.communicate()
        if runtime:
            await runtime.shutdown(timeout_s=5)
        if store:
            await store.close()
        if db:
            if backend == "postgres":
                for table in (*task_schema.TASK_TABLES, *_STORE_TABLES, "schema_migrations"):
                    await db.execute(f"DROP TABLE IF EXISTS {prefix}{table}")
            await db.close()
