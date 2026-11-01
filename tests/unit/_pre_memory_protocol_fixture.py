"""Construct an offline pre-memory accounting fixture for explicit migration."""
from __future__ import annotations

import json

from power_loop.runtime.control_runtime import flush_binding
from power_loop.runtime.jobs import JobManager, _hash, _json
from power_loop.runtime.task_runtime import TaskRuntime


async def retain_pre_memory_format(c, protocol):
    # A maintenance fixture has no live writer. Keep original fees, grants,
    # messages and notices; remove only the later memory-journal metadata.
    original_calls = c.host.reserve_calls, c.host.apply_calls
    for binding in tuple(c.manager._execution_controls.values()):
        await flush_binding(binding, close=True)
    await c.runtime.shutdown(timeout_s=1)
    db, prefix = c.manager.db, c.manager.prefix
    rows = await db.fetchall(f"SELECT attempt_id,state_json FROM {prefix}task_budget_attempts")
    assert rows
    c._migration_original_budgets = {}
    removed = 0
    for row in rows:
        value = json.loads(row["state_json"])
        removed += "allocation_id" in value
        value.pop("allocation_id", None)
        c._migration_original_budgets[row["attempt_id"]] = value
        await db.execute(f"UPDATE {prefix}task_budget_attempts SET state_json=?,state_hash=? WHERE attempt_id=?",
                         (_json(value), _hash(value), row["attempt_id"]))
    assert removed > 0 and (c.host.reserve_calls, c.host.apply_calls) == original_calls
    c._migration_original_calls = original_calls
    c._migration_original_counters = await db.fetchall(
        f"SELECT scope_key,counters_json,counters_hash FROM {prefix}task_budget_scopes ORDER BY scope_key")
    events = await db.fetchall(f"SELECT event_id,payload_json FROM {prefix}job_events WHERE event_type='model_budget_state'")
    for event in events:
        value = json.loads(event["payload_json"])
        value.pop("allocation_id", None)
        await db.execute(f"UPDATE {prefix}job_events SET payload_json=?,request_hash=? WHERE event_id=?",
                         (_json(value), _hash(value), event["event_id"]))
    for table in ("execution_batches", "execution_checkpoints", "execution_allocations"):
        await db.execute(f"DELETE FROM {prefix}{table}")
    await db.execute(f"UPDATE {prefix}task_store SET protocol_version=? WHERE id=1", (protocol,))


async def reopen_after_maintenance(c, connect):
    previous = c.runtime
    c.store = await connect()
    c.manager = JobManager(c.store)
    c.runtime = TaskRuntime(c.manager, tenant_id=previous.tenant_id, principal_id=previous.principal_id,
        executor_id="maintenance-recovery", boot_id="after-maintenance",
        boot_contract_hash=previous._lifecycle.contract_hash,
        budget_config=previous._original_budget_config)
    for key, original in c._migration_original_budgets.items():
        row = await c.manager.db.fetchone(f"SELECT state_json FROM {c.manager.prefix}task_budget_attempts WHERE attempt_id=?", (key,))
        value = json.loads(row["state_json"])
        assert value["allocation_id"] and value["version"] == original["version"] + 1
        assert {k: v for k, v in value.items() if k not in ("allocation_id", "version")} == {
            k: v for k, v in original.items() if k != "version"}
    assert await c.manager.db.fetchall(
        f"SELECT scope_key,counters_json,counters_hash FROM {c.manager.prefix}task_budget_scopes ORDER BY scope_key") == c._migration_original_counters
    assert (c.host.reserve_calls, c.host.apply_calls) == c._migration_original_calls
