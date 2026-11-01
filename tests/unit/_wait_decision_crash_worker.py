"""Exit between the original response checkpoint and wait-decision publication."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

from power_loop import AgentLoopConfig, StatefulAgentLoop
from power_loop.runtime import wait_decisions
from power_loop.runtime.execution_state import CheckpointPolicy
from power_loop.runtime.jobs import JobManager
from power_loop.runtime.store.factory import open_store
from power_loop.runtime.task_runtime import TaskRuntime
from power_loop.tools.registry import ToolRegistry
from power_loop.tools.wait_handoff import register_wait_handoff_tool

from .test_wait_handoff_192 import MainModel


async def main(dsn, prefix, evidence, config, stage):
    saved = json.loads(Path(config).read_text())
    store = await open_store(dsn, table_prefix=prefix, schema="tasks")
    manager = JobManager(store, now_ms=lambda: saved["now"])
    runtime = TaskRuntime(manager, tenant_id=saved["tenant"], principal_id=saved["principal"],
                          executor_id="main-executor", boot_id="main-boot",
                          checkpoint_policy=CheckpointPolicy(interval_s=300))
    model, registry = MainModel(), ToolRegistry()
    model.options = {"event_id": saved["event_id"], "decision": "continue"}
    register_wait_handoff_tool(registry)
    loop = StatefulAgentLoop(llm=model, store=store, task_runtime=runtime, tool_registry=registry,
                             config=AgentLoopConfig(delegation_policy="single_level", max_rounds=3, max_tokens=20))
    record, observation = wait_decisions.record_decision, wait_decisions._observation

    async def retain(*args, **kwargs):
        proof = {**kwargs, "caller": asdict(kwargs["caller"]), "call_ref": kwargs["call_ref"].to_dict(),
                 "model_calls": model.calls}
        Path(evidence).write_text(json.dumps(proof))
        return await record(*args, **kwargs)

    async def after_checkpoint(*args, **kwargs):
        result = await observation(*args, **kwargs)
        if stage == "after_checkpoint":
            os._exit(91)
        return result

    decision_written = False
    transaction = store._db.transaction

    class CrashTransaction:
        def __init__(self, tx):
            self.tx = tx

        def __getattr__(self, name):
            return getattr(self.tx, name)

        async def execute(self, sql, params=()):
            nonlocal decision_written
            result = await self.tx.execute(sql, params)
            if sql.startswith("INSERT INTO " + manager.events) and len(params) > 2 and params[2] == "wait_decision":
                decision_written = True
                if stage == "after_decision":
                    os._exit(92)
            return result

    @asynccontextmanager
    async def crash_after_commit():
        async with transaction() as tx:
            yield CrashTransaction(tx)
        if decision_written and stage == "after_commit":
            os._exit(93)

    store._db.transaction = crash_after_commit
    wait_decisions.record_decision, wait_decisions._observation = retain, after_checkpoint
    await loop.flush_follow_ups(saved["sid"])
    raise AssertionError("wait decision crash boundary was not reached")


if __name__ == "__main__":
    asyncio.run(main(*sys.argv[1:]))
