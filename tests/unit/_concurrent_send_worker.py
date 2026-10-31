"""A genuinely separate process competing for the same session turn."""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from power_loop import AgentLoopConfig, LLMResponse, LLMService, LLMTokenUsage, SessionBusy, StatefulAgentLoop
from power_loop.runtime.jobs import JobManager
from power_loop.runtime.store.factory import open_store
from power_loop.runtime.task_runtime import TaskRuntime


async def main(dsn, prefix, sid, evidence, release, role):
    store = await open_store(dsn, table_prefix=prefix, schema="compatible")
    path, gate = Path(evidence), Path(release)
    calls = 0

    def record(status):
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps({"status": status, "calls": calls}))
        temporary.replace(path)

    class Model(LLMService):
        async def complete(self, request, **kwargs):
            nonlocal calls
            calls += 1
            record("model_entered")
            async with asyncio.timeout(20):
                while not gate.exists():
                    await asyncio.sleep(.01)
            return LLMResponse(raw_text=role, token_usage=LLMTokenUsage(
                prompt_tokens=2, completion_tokens=1, total_tokens=3))

        def stream(self, request):
            raise AssertionError("nonstream fixture")

        async def close(self):
            pass

    runtime = TaskRuntime(JobManager(store), tenant_id="lock-test", principal_id="host",
        executor_id="executor", boot_id=role, lease_ms=300000, boot_contract_hash="a" * 64)
    loop = StatefulAgentLoop(llm=Model(), store=store, task_runtime=runtime,
        config=AgentLoopConfig(max_rounds=1, compactor=None, distributed_sessions=True,
                              session_lease_ttl_s=300, delegation_policy="single_level"))
    try:
        try:
            result = await loop.send(role, sid)
        except SessionBusy:
            record("busy")
            return
        assert result.status == "completed"
        record("completed")
    finally:
        await runtime.shutdown(timeout_s=1)
        await loop.aclose(drain_timeout_s=.1)
        await store.close()


if __name__ == "__main__":
    asyncio.run(main(*sys.argv[1:]))
