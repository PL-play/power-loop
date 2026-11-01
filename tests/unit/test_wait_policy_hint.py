"""New finite policies reach a live owner without per-dispatch confirmation reads."""
from __future__ import annotations

from contextlib import asynccontextmanager

import pytest

from power_loop.runtime import model_attempts, owner_waits, timer_waits
from power_loop.runtime.control_runtime import flush_binding
from power_loop.runtime.task_runtime import Execution, TaskRuntime

from .test_timer_waits_192 import registry_store, runtime, waiting

__all__ = ["registry_store", "runtime", "waiting"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def owned(c):
    runtime = TaskRuntime(c.manager, tenant_id=c.actor.tenant_id, principal_id=c.actor.principal_id,
                          executor_id=c.caller.executor_id, boot_id=c.caller.boot_id)
    binding = Execution(runtime, c.actor, c.caller)
    await model_attempts.enable(binding)
    assert binding.resource_evidence["_control_wait_deadline"] is None
    return binding


async def prepare(binding, nonce):
    return await model_attempts.prepare(binding, nonce=nonce, round_index=0, retry_index=1,
        request_hash="a" * 64, model="test", max_tokens=20)


async def finish(attempt):
    await model_attempts.finish(attempt, status="returned", usage={"total_tokens": 0}, estimated=False)


@pytest.mark.parametrize("fault", [None, "rollback", "ack_loss"])
async def test_committed_policy_is_reused_and_unknown_policy_is_resolved_once(waiting, monkeypatch, fault):
    c = waiting
    binding = await owned(c)
    transaction, injected = c.store._db.transaction, False

    @asynccontextmanager
    async def fail_once():
        nonlocal injected
        inject = False
        async with transaction() as tx:
            yield tx
            found = await tx.fetchone(f"SELECT job_id FROM {c.manager.jobs} WHERE kind='timer'")
            if found is not None and not injected:
                injected = inject = True
                if fault == "rollback":
                    raise OSError("wait policy rolled back")
        if inject:
            raise OSError("wait policy commit ACK lost")

    if fault is None:
        await c.register()
        assert binding.resource_evidence["_control_wait_deadline"] == c.policy.deadline_at_ms
    else:
        with monkeypatch.context() as patch:
            patch.setattr(c.store._db, "transaction", fail_once)
            with pytest.raises(OSError):
                await c.register()
        assert injected and "_control_wait_deadline" not in binding.resource_evidence
    reads, policy = [], owner_waits.require_dispatch

    async def resolve(*args):
        reads.append(True)
        return await policy(*args)

    monkeypatch.setattr(owner_waits, "require_dispatch", resolve)
    first = await prepare(binding, "first")
    deadline = await model_attempts.dispatch(first)
    assert len(reads) == int(fault is not None)
    assert (deadline is not None) == (fault != "rollback")
    assert binding.resource_evidence["_control_wait_deadline"] == (
        None if fault == "rollback" else c.policy.deadline_at_ms)
    await finish(first)
    second = await prepare(binding, "second")

    def forbidden():
        pytest.fail("a confirmed original policy must not reopen a control transaction")

    with monkeypatch.context() as patch:
        patch.setattr(c.manager, "_write", forbidden)
        assert (await model_attempts.dispatch(second) is not None) == (fault != "rollback")
    await finish(second)
    await flush_binding(binding, close=True)


async def test_unknown_short_policy_is_not_overwritten_by_a_later_long_commit(waiting, monkeypatch):
    c = waiting
    binding = await owned(c)
    transaction, injected = c.store._db.transaction, False
    short = timer_waits.WaitPolicy(c.now[0] + 4000)

    @asynccontextmanager
    async def lose_once():
        nonlocal injected
        inject = False
        async with transaction() as tx:
            yield tx
            if not injected:
                found = await tx.fetchone(f"SELECT job_id FROM {c.manager.jobs} WHERE kind='timer'")
                if found is not None:
                    injected = inject = True
        if inject:
            raise OSError("short policy ACK lost")

    with monkeypatch.context() as patch:
        patch.setattr(c.store._db, "transaction", lose_once)
        with pytest.raises(OSError, match="ACK lost"):
            await c.register(policy=short)
    assert "_control_wait_deadline" not in binding.resource_evidence
    seq = await c.store.append_message(c.sid, role="assistant", send_index=1, round_index=0,
        tool_calls=[{"id": "second-wait", "type": "function", "function": {"name": "wait", "arguments": "{}"}}])
    await c.register(call_ref=await c.store.message_ref(c.sid, seq), tool_call_id="second-wait")
    assert "_control_wait_deadline" not in binding.resource_evidence
    attempt = await prepare(binding, "after-both-policies")
    assert await model_attempts.dispatch(attempt) is not None
    assert binding.resource_evidence["_control_wait_deadline"] == short.deadline_at_ms
    await finish(attempt)
    await flush_binding(binding, close=True)
