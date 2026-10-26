"""Executed in a fresh interpreter against an explicitly selected SDK checkout."""
from __future__ import annotations

import asyncio
import json
import sys

from power_loop import AgentLoopConfig, LLMResponse, LLMService, LLMTokenUsage, StatefulAgentLoop
from power_loop.runtime.budget_contracts import BudgetConfig, LocalBudgetScope
from power_loop.runtime.jobs import JobActor, JobError, JobManager
from power_loop.runtime.store.factory import open_store
from power_loop.runtime.store.schema import StoreSchemaError
from power_loop.runtime.store.task_schema import TASK_PROTOCOL_VERSION
from power_loop.runtime.task_runtime import TaskRuntime
from tests.real.test_real_budget_tree_192 import RealBudgetHost


class Once(LLMService):
    def __init__(self, real):
        self.calls, self.tokens = 0, 0
        self.inner = None
        if real:
            from tests.real._llm import make_llm
            self.inner = make_llm(max_tokens=512, temperature=0)

    async def complete(self, request, **kwargs):
        assert self.calls == 0, "bridge rehearsal must not repeat a model request"
        self.calls += 1
        result = (await self.inner.complete(request, **kwargs) if self.inner else
                  LLMResponse(raw_text="42", token_usage=LLMTokenUsage(
                      prompt_tokens=7, completion_tokens=3, total_tokens=10)))
        self.tokens += result.token_usage.total_tokens
        return result

    def stream(self, request):
        raise AssertionError("streaming is outside this rehearsal")

    async def close(self):
        if self.inner:
            await self.inner.close()


class UnavailableUsage(RealBudgetHost):
    async def apply_usage(self, submission):
        raise OSError("isolated original accounting endpoint unavailable")


async def run(config):
    assert TASK_PROTOCOL_VERSION == config["protocol"], "wrong legacy SDK selected"
    if config["action"] == "reopen":
        refused = []
        for mode in ("tasks", "compatible"):
            try:
                store = await open_store(config["dsn"], table_prefix=config["prefix"], schema=mode)
            except StoreSchemaError:
                refused.append(mode)
            else:
                await store.close()
                raise AssertionError("old SDK admitted upgraded task store")
        return {"refused": refused}

    if config["action"] == "held_writer":
        store = await open_store(config["dsn"], table_prefix=config["prefix"], schema="compatible")
        try:
            manager = JobManager(store)
            print(json.dumps({"ready": True, "store_id": store.store_id}), flush=True)
            # Deliberately idle old process; the parent owns its bounded lifetime.
            assert sys.stdin.readline().strip() == "try-write"
            try:
                await manager.request_cancel(JobActor("bridge-test", "host", config["queued_sid"]),
                    config["queued_job"], system_key="stale-writer", reason="must be refused")
            except JobError as error:
                assert str(error) == "store_identity_changed"
            else:
                raise AssertionError("already-open old JobManager wrote after upgrade")
            return {"stale_write_refused": True}
        finally:
            await store.close()

    store = await open_store(config["dsn"], table_prefix=config["prefix"], schema="tasks")
    provider, host = Once(config["real"]), UnavailableUsage()
    host.service = provider
    manager = JobManager(store)
    runtime = TaskRuntime(manager, tenant_id="bridge-test", principal_id="host",
        executor_id="old-executor", boot_id="old-boot", boot_contract_hash="a" * 64,
        budget_config=BudgetConfig("original-budget", (LocalBudgetScope("authority", max_calls=1),), host))
    try:
        loop = StatefulAgentLoop(llm=provider, store=store, task_runtime=runtime,
            config=AgentLoopConfig(delegation_policy="single_level", max_rounds=1, max_tokens=512))
        sid = await loop.new_session()
        result = await loop.send("Reply with the number 42 only. Do not call tools.", sid)
        assert result.status == "completed" and "42" in result.final_text and provider.calls == 1
        await runtime.shutdown(timeout_s=5)
        [attempt] = await manager.db.fetchall(f"SELECT * FROM {manager.prefix}task_budget_attempts")
        assert attempt["pending"] == 1
        state = json.loads(attempt["state_json"])
        assert state["submission"] and state["ack"] is None
        # A second original task remains queued, with a real call/receipt pair.
        queued_sid = await store.create_session()
        actor = JobActor("bridge-test", "host", queued_sid)
        root = await manager.register_job(actor, kind="send", session_id=queued_sid, system_key="original-root")
        lease = await manager.claim_job(actor, root.job.job_id, executor_id="queued-owner", boot_id="owner",
            expected_version=root.job.version)
        seq = await store.append_message(queued_sid, role="assistant", tool_calls=[{
            "id": "original-call", "type": "function", "function": {"name": "work", "arguments": "{}"}}])
        call = await store.message_ref(queued_sid, seq)
        reg = await manager.register_job(actor, kind="background_tool", parent=lease,
            call_ref=call, tool_call_id="original-call", independent=True)
        receipt_seq = await store.append_message(queued_sid, role="tool", tool_call_id="original-call",
            content="accepted original task")
        receipt = await store.message_ref(queued_sid, receipt_seq)
        await manager.reconcile_receipt(actor, reg.operation.operation_id, receipt)
        await manager.close_model_execution(actor, lease)
        return {"protocol": TASK_PROTOCOL_VERSION, "store_id": store.store_id, "sid": sid,
                "attempt_id": attempt["attempt_id"], "job_id": attempt["job_id"],
                "queued_sid": queued_sid, "queued_job": reg.job.job_id,
                "operation_id": reg.operation.operation_id, "call": call.to_dict(), "receipt": receipt.to_dict(),
                "requests": provider.calls if config["real"] else 0,
                "reported_tokens": provider.tokens if config["real"] else 0}
    finally:
        print(json.dumps({"real_provider_requests": provider.calls if config["real"] else 0,
                          "reported_tokens": provider.tokens if config["real"] else 0}), file=sys.stderr)
        await runtime.shutdown(timeout_s=5)
        await provider.close()
        await store.close()


if __name__ == "__main__":
    async def main():
        async with asyncio.timeout(100):
            value = await run(json.loads(sys.stdin.readline()))
        print(json.dumps(value))
    asyncio.run(main())
