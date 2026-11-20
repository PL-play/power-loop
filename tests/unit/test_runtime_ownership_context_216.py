"""Real worker claims bind one handle and retire it before business cleanup."""
import asyncio

import pytest

from power_loop import (
    AgentHooks,
    AgentLoopConfig,
    ExecutionEnvironment,
    HookPoint,
    RegistryRuntimeFactory,
    Runtime,
    SessionLeaseLost,
)
from power_loop.agent.runtime_store import RuntimeInputStore
from power_loop.agent.runtime_types import RuntimeRequestError
from power_loop.core.execution_context import independent_execution_scope
from power_loop.runtime.session_ownership import (
    MemorySessionOwnershipCoordinator,
    current_ownership,
    require_current_ownership,
)

from .session_native_fixture import ACTOR, target
from .test_execution_renewal import WaitingModel
from .test_session_multinode_215 import EchoModel

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.mark.parametrize("exit_kind", [
    "normal", "build_failure", "cancel", "loss", "close_failure", "release_failure", "empty_claim",
])
async def test_worker_retires_actual_external_handle_before_cleanup(registry_store, monkeypatch, exit_kind):
    store, _ = registry_store
    port = RuntimeInputStore(store)
    root = await target(port)
    await port.accept(root, "go", message_id="one", actor=ACTOR)
    seen, released, acquired, finished, lost, reset = [], [], [], [], [], []
    late_release = asyncio.Event()
    children = []
    model = WaitingModel() if exit_kind in {"cancel", "loss"} else EchoModel()

    class Coordinator(MemorySessionOwnershipCoordinator):
        async def acquire(self, *args, **kwargs):
            lease = await super().acquire(*args, **kwargs)
            acquired.append(lease)
            return lease

        async def release(self, lease):
            released.append(lease)
            assert not lease.released  # A valid coordinator may early-return otherwise.
            assert current_ownership() is (None if exit_kind == "empty_claim" else lease)
            if exit_kind == "release_failure":
                raise OSError("external release failed")
            await super().release(lease)

    class Resources:
        closed = False

        async def aclose(self):
            assert current_ownership() is None
            assert seen[-1].released
            await store.set_runtime_state(root.session_id, "cleanup", {"count": 1})
            value = await store.mutate_runtime_state(
                root.session_id, "cleanup", lambda old: {"count": old["count"] + 1})
            assert value == {"count": 2}
            self.closed = True
            if exit_kind == "close_failure":
                raise OSError("resource close failed")

    resources = Resources()
    hooks, factory, coordinator = AgentHooks(), RegistryRuntimeFactory(), Coordinator()

    async def on_acquired(context):
        lease = current_ownership()
        lease.require_owned()
        execution = context.values["execution"]
        assert (lease.epoch, lease.token) == (execution.owner_epoch, execution.owner_token)

    async def on_finished(context):
        assert current_ownership() is None
        finished.append(context.values["execution"])

    hooks.register(HookPoint.EXECUTION_ACQUIRED, on_acquired)
    hooks.register(HookPoint.EXECUTION_FINISHED, on_finished)
    hooks.register(HookPoint.EXECUTION_LOST, lambda context: lost.append(context.values["execution"]))

    async def late(lease, sid):
        await late_release.wait()
        assert current_ownership(store, sid) is lease
        assert current_ownership(store, "another-session") is None
        with pytest.raises(SessionLeaseLost):
            require_current_ownership(store)
        with pytest.raises(SessionLeaseLost):
            await store.append_message(sid, role="assistant", content="stale child")

    async def build(descriptor, execution):
        lease = current_ownership(store, execution.session_id)
        assert lease is acquired[-1] is runtime.worker._active[execution.session_id].ownership
        lease.require_owned()
        assert lease.key == (execution.store_id, execution.session_id)
        assert (lease.epoch, lease.token) == (execution.owner_epoch, execution.owner_token)
        seen.append(lease)
        with independent_execution_scope():
            assert current_ownership() is None
        assert current_ownership() is lease
        if len(seen) == 2:
            # Another session, same Runtime owner_id, cannot authorize the old task.
            late_release.set()
            await children[0]
        elif exit_kind == "normal":
            children.append(asyncio.create_task(late(lease, execution.session_id)))
        if exit_kind in {"build_failure", "release_failure"}:
            raise RuntimeRequestError("resource_unavailable", "build failed")
        return ExecutionEnvironment(model, None, AgentLoopConfig(context_budget_tokens=10**9),
                                    resources=resources)

    factory.register("test.agent@215", build)
    runtime = Runtime(store=store, factory=factory, node_id="same-owner", boot_id="same-boot",
                      coordinator=coordinator, recovery_interval_s=None, timers_enabled=False,
                      hooks=hooks)
    original_final = runtime.worker.inputs.final

    async def final(batch, **kwargs):
        assert current_ownership() is runtime.worker._active[batch.target.session_id].ownership
        current_ownership().require_owned()
        return await original_final(batch, **kwargs)

    monkeypatch.setattr(runtime.worker.inputs, "final", final)
    if exit_kind == "empty_claim":
        async def no_batch(*args, **kwargs):
            return None
        monkeypatch.setattr(runtime.worker.inputs, "claim", no_batch)

    async def run(sid):
        try:
            await runtime.worker._run_session(sid)
        finally:
            reset.append(current_ownership())

    task = asyncio.create_task(run(root.session_id))
    try:
        if exit_kind in {"cancel", "loss"}:
            await asyncio.wait_for(model.entered.wait(), 5)
            if exit_kind == "cancel":
                task.cancel()
            else:
                acquired[0].lose()
        if exit_kind == "cancel":
            with pytest.raises(asyncio.CancelledError):
                await task
        elif exit_kind in {"close_failure", "release_failure"}:
            with pytest.raises(OSError, match="failed"):
                await task
        else:
            await asyncio.wait_for(task, 10)
        assert reset == [None]
        assert len(acquired) == len(released) == 1 and acquired[0] is released[0]
        lease = acquired[0]
        assert lease.released and lease.lost == (exit_kind == "loss")
        with pytest.raises(SessionLeaseLost):
            lease.require_owned()
        assert len(lost) == (1 if exit_kind == "loss" else 0)
        assert len(finished) == (0 if exit_kind in {"empty_claim", "release_failure"} else 1)
        assert resources.closed == (exit_kind not in {"build_failure", "release_failure", "empty_claim"})
        if exit_kind in {"build_failure", "empty_claim"}:
            state = await store._db.fetchone(
                f"SELECT execution_state FROM {store.t.session_state} WHERE session_id=?",
                (root.session_id,),
            )
            assert state["execution_state"] == "queued"
            assert not await store.load_all_messages(root.session_id)
        if exit_kind == "normal":
            other = await target(port, session_id="another-session")
            await port.accept(other, "next", message_id="two", actor=ACTOR)
            await asyncio.wait_for(run(other.session_id), 10)
            assert len(seen) == 2 and seen[0] is not seen[1]
            assert seen[0].owner_id == seen[1].owner_id
            assert all(lease.released and not lease.lost for lease in seen)
            assert len(released) == 2 and reset == [None, None]
        assert not [row for row in await store.load_all_messages(root.session_id)
                    if row.content == "stale child"]
    finally:
        late_release.set()
        if isinstance(model, WaitingModel):
            model.release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, *children, return_exceptions=True)
        await runtime.client.aclose()
        await runtime.transport.aclose()
