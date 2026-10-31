"""Die between a durable usage checkpoint and allocation settlement."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

from power_loop.runtime.budget_contracts import BudgetAck, BudgetConfig, LocalBudgetScope
from power_loop.runtime.control_runtime import current, flush_binding
from power_loop.runtime.jobs import JobManager
from power_loop.runtime.store.factory import open_store
from power_loop.runtime.task_runtime import TaskRuntime

from .test_model_budgets_192 import Host, new_send, observed, start


class FileHost(Host):
    def __init__(self, path):
        super().__init__()
        self.path = Path(path)

    async def apply_usage(self, submission):
        charges = json.loads(self.path.read_text()) if self.path.exists() else {}
        key = submission.attempt_id
        assert charges.setdefault(key, submission.request_hash) == submission.request_hash
        self.path.write_text(json.dumps(charges))
        if self.apply_lost_ack:
            raise OSError("durable host charge ACK lost")
        return BudgetAck(key, submission.revision, submission.request_hash)


def config(host):
    return BudgetConfig("host-budget-v1", (LocalBudgetScope("shared", max_calls=2,
                        max_tokens=200, max_inflight=2),), host)


async def main(dsn, prefix, evidence, stage):
    store = await open_store(dsn, table_prefix=prefix, schema="tasks")
    host = FileHost(evidence + ".charges")
    host.apply_lost_ack = True
    manager = JobManager(store)
    runtime = TaskRuntime(manager, tenant_id="budget-tenant", principal_id="host",
        executor_id="executor", boot_id="boot", lease_ms=300000,
        boot_contract_hash="a" * 64, budget_config=config(host))
    case = SimpleNamespace(store=store, manager=manager, runtime=runtime, host=host)
    await new_send(case)
    attempt = await start(case)
    await observed(case, attempt, tokens=12)
    await flush_binding(case.binding)
    state = current(case.binding)
    Path(evidence).write_text(json.dumps({"sid": case.sid, "job_id": case.binding.lease.job_id,
        "generation": case.binding.lease.generation, "attempt_id": attempt.attempt_id,
        "batch_no": state.cursor}))
    if stage == "open":
        os._exit(81)
    if stage == "during_close":
        transaction = manager.db.transaction

        @asynccontextmanager
        async def interrupted():
            async with transaction() as tx:
                execute = tx.execute

                async def die(sql, params=()):
                    result = await execute(sql, params)
                    if sql.startswith(f"UPDATE {manager.prefix}task_budget_scopes "):
                        os._exit(82)
                    return result

                tx.execute = die
                yield tx

        manager.db.transaction = interrupted
    await flush_binding(case.binding, close=True)
    os._exit(83)


if __name__ == "__main__":
    asyncio.run(main(*sys.argv[1:]))
