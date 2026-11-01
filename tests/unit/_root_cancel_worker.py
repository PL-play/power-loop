"""Actual process for context-stop crash and admission race boundaries."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from urllib.parse import urlparse

from power_loop import AgentLoopConfig, LLMResponse, LLMService, StatefulAgentLoop
from power_loop.runtime import root_control
from power_loop.runtime.jobs import JobActor, JobError, JobManager
from power_loop.runtime.store.factory import open_store
from power_loop.runtime.task_runtime import TaskRuntime


async def main():
    dsn, prefix, tenant, principal, sid, namespace, payload, mode, ready, release, output = sys.argv[1:]
    if "://" in dsn:
        address = urlparse(dsn)
        assert address.hostname == "127.0.0.1" and address.path.endswith("_test")
    payload = json.loads(payload)
    store = await open_store(dsn, table_prefix=prefix, schema="compatible")
    manager = JobManager(store)
    actor = JobActor(tenant, principal, sid)
    try:
        if mode == "new-send":
            class Model(LLMService):
                calls = 0
                async def complete(self, request, **kwargs):
                    self.calls += 1
                    return LLMResponse(raw_text="must not dispatch")
                def stream(self, request):
                    raise AssertionError("nonstream fixture")
                async def close(self):
                    pass
            model = Model()
            runtime = TaskRuntime(manager, tenant_id=tenant, principal_id=principal,
                                  executor_id="executor", boot_id="boot")
            loop = StatefulAgentLoop(llm=model, store=store, task_runtime=runtime,
                config=AgentLoopConfig(delegation_policy="single_level", max_rounds=2, compactor=None))
            try:
                try:
                    with runtime.root_context(sid, namespace=namespace, payload=payload):
                        result = await loop.send("new send during context stop", sid)
                    packet = {"status": result.status, "calls": model.calls}
                except JobError as exc:
                    packet = {"error": exc.code, "calls": model.calls}
                Path(output).write_text(json.dumps(packet))
            finally:
                await loop.aclose(drain_timeout_s=.1)
            return
        original = root_control._cancel_tree
        calls = 0

        async def tree(*args, **kwargs):
            nonlocal calls
            if mode == "hold-before-first" and calls == 0:
                Path(ready).write_text(json.dumps({"state": "barrier-and-scope-committed"}))
                async with asyncio.timeout(30):
                    while not Path(release).exists():
                        await asyncio.sleep(.01)
            result = await original(*args, **kwargs)
            calls += 1
            if mode == "crash-after-first":
                Path(ready).write_text(json.dumps(result))
                os._exit(74)
            return result

        root_control._cancel_tree = tree
        receipt = await root_control.cancel(manager, actor, namespace=namespace, payload=payload,
                                           system_key="process-context-stop", reason="user stop")
        Path(output).write_text(json.dumps(receipt))
    finally:
        await store.close()


asyncio.run(main())
