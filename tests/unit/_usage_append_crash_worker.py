"""Persist original supplier usage, then die inside/beyond its message commit."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

from power_loop import AgentLoopConfig, LLMResponse, LLMService, LLMTokenUsage, StatefulAgentLoop
from power_loop.agent.sink import SQLiteSink
from power_loop.runtime.jobs import JobManager
from power_loop.runtime.response_capture import ResponseCaptureReceipt
from power_loop.runtime.store.factory import open_store
from power_loop.runtime.task_runtime import _SEND, TaskRuntime

from ._budget_checkpoint_crash_worker import FileHost, config


class UsageHost(FileHost):
    async def capture_response(self, packet):
        Path(str(self.path) + ".capture").write_text(json.dumps(asdict(packet)))
        return ResponseCaptureReceipt(packet.capture_id, packet.fingerprint, "original-supplier-response")

    async def apply_usage(self, submission):
        Path(str(self.path) + ".submission").write_text(json.dumps(asdict(submission)))
        return await super().apply_usage(submission)


async def main(dsn, prefix, sid, evidence, stage):
    store = await open_store(dsn, table_prefix=prefix, schema="tasks")
    host = UsageHost(evidence + ".charges")

    class Model(LLMService):
        async def complete(self, request, **kwargs):
            binding = _SEND.get()
            Path(evidence).write_text(json.dumps({"sid": sid, "job_id": binding.lease.job_id,
                "generation": binding.lease.generation, "calls": 1}))
            return LLMResponse(raw_text="done", token_usage=LLMTokenUsage(
                prompt_tokens=7, completion_tokens=5, total_tokens=12))

        def stream(self, request):
            raise AssertionError("nonstream fixture")

        async def close(self):
            pass

    model = host.service = Model()
    runtime = TaskRuntime(JobManager(store), tenant_id="budget-tenant", principal_id="host",
        executor_id="executor", boot_id="boot", boot_contract_hash="a" * 64, budget_config=config(host))
    loop = StatefulAgentLoop(llm=model, store=store, task_runtime=runtime,
        config=AgentLoopConfig(max_rounds=2, max_tokens=20, compactor=None, delegation_policy="single_level"))
    transaction = store._db.transaction

    @asynccontextmanager
    async def interrupted():
        async with transaction() as tx:
            execute, fetchone = tx.execute, tx.fetchone
            assistant = False

            def stop(sql, params):
                nonlocal assistant
                if sql.startswith(f"INSERT INTO {store.t.messages} ") and params[2] == "assistant":
                    assistant = True
                    if stage == "after_message":
                        os._exit(93)
                if assistant and store.t.usage_rounds in sql and stage == "after_usage":
                    os._exit(94)

            async def write(sql, params=()):
                result = await execute(sql, params)
                stop(sql, params)
                return result

            async def read(sql, params=()):
                result = await fetchone(sql, params)
                stop(sql, params)
                return result

            tx.execute, tx.fetchone = write, read
            yield tx

    store._db.transaction = interrupted
    ended = SQLiteSink.on_round_ended

    async def before_end(self, *args, **kwargs):
        if self.session_id == sid and stage == "before_round_end":
            os._exit(95)
        return await ended(self, *args, **kwargs)

    SQLiteSink.on_round_ended = before_end
    await loop.send("record once", sid)
    raise AssertionError("usage crash boundary was not reached")


if __name__ == "__main__":
    asyncio.run(main(*sys.argv[1:]))
