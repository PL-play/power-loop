"""Explicit import preserves old fees/holds; it is never a runtime fallback."""
from __future__ import annotations

import pytest

from power_loop.runtime import model_budgets as budgets
from power_loop.runtime.control_runtime import flush_binding
from power_loop.runtime.jobs import _hash, _json
from power_loop.runtime.store.schema import StoreSchemaError
from power_loop.runtime.store.task_schema import migrate_task_schema

from .test_budget_reconcile_192 import reopened
from .test_job_registry_192 import registry_store
from .test_model_budgets_192 import budget_case, counters, observed, start

__all__ = ["registry_store", "budget_case"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def protocol21(c, attempt):
    await flush_binding(attempt.binding, close=True)
    value = await budgets.get_budget(c.manager, c.actor, attempt.attempt_id)
    await c.runtime.shutdown(timeout_s=1)
    value.pop("allocation_id")
    db, p = c.manager.db, c.manager.prefix
    await db.execute(f"UPDATE {p}task_budget_attempts SET state_json=?,state_hash=? WHERE attempt_id=?",
                     (_json(value), _hash(value), attempt.attempt_id))
    await db.execute(f"UPDATE {p}job_events SET payload_json=?,request_hash=? WHERE event_id=?",
                     (_json(value), _hash(value), "mb_" + _hash([attempt.attempt_id, value["version"]])[:48]))
    for table in ("execution_batches", "execution_checkpoints", "execution_allocations"):
        await db.execute(f"DELETE FROM {p}{table}")
    await db.execute(f"UPDATE {p}task_store SET protocol_version=21 WHERE id=1")
    return value


@pytest.mark.parametrize("outcome", ["returned", "pending_ack", "unknown"])
async def test_import_retains_original_fees_and_shared_balances(budget_case, outcome):
    c = budget_case
    c.host.apply_lost_ack = outcome == "pending_ack"
    attempt = await start(c)
    await observed(c, attempt, error=OSError("unknown") if outcome == "unknown" else None)
    original = await protocol21(c, attempt)
    before = await counters(c)
    calls = c.host.reserve_calls, c.host.apply_calls
    await migrate_task_schema(c.manager.db, c.manager.prefix, maintenance=True)
    fresh = await reopened(c)
    migrated = await budgets.get_budget(fresh.manager, c.actor, attempt.attempt_id)
    assert migrated["allocation_id"] and migrated["version"] == original["version"] + 1
    assert {k: v for k, v in migrated.items() if k not in ("allocation_id", "version")} == {
        k: v for k, v in original.items() if k != "version"}
    assert await counters(c) == before and (c.host.reserve_calls, c.host.apply_calls) == calls
    c.host.apply_lost_ack = False
    result = await fresh.reconcile_budget_usage(c.sid, job_id=c.binding.lease.job_id, attempt_id=attempt.attempt_id)
    assert result["usage_acknowledged"] and not result["execution_authority"]
    assert await counters(c) == before and c.host.reserve_calls == 1
    # Re-running explicit maintenance is harmless and doesn't reimport fees.
    await migrate_task_schema(c.manager.db, c.manager.prefix, maintenance=True)
    assert await counters(c) == before
    await fresh.shutdown(timeout_s=1)


async def test_import_rejects_inconsistent_original_balance(budget_case):
    c = budget_case
    attempt = await start(c)
    await observed(c, attempt)
    await protocol21(c, attempt)
    altered = {**await counters(c), "used_tokens": 999}
    await c.manager.db.execute(f"UPDATE {c.manager.prefix}task_budget_scopes SET counters_json=?,counters_hash=?",
                              (_json(altered), _hash(altered)))
    with pytest.raises(StoreSchemaError, match="balance_mismatch"):
        await migrate_task_schema(c.manager.db, c.manager.prefix, maintenance=True)
    [stamp] = await c.manager.db.fetchall(f"SELECT protocol_version FROM {c.manager.prefix}task_store")
    assert stamp["protocol_version"] == 21
    assert not await c.manager.db.fetchall(f"SELECT * FROM {c.manager.prefix}execution_checkpoints")
