"""R03 regressions: parent waiting on run_agent_spec yields worker capacity safely."""
from __future__ import annotations

import asyncio
import json

import pytest

from power_loop import (
    AgentLoopConfig,
    AgentSpec,
    ExecutionEnvironment,
    LLMResponse,
    LLMService,
    RegistryRuntimeFactory,
    Runtime,
    RuntimeDescriptor,
    ToolDefinition,
    ToolRegistry,
    TrustedActor,
)
from power_loop.agent.runtime_store import now_ms
from power_loop.core.agent_context import get_current_loop
from power_loop.runtime.pricing import Free
from power_loop.runtime.spec import run_agent_spec
from power_loop.runtime.tool_contracts import FunctionRuntimeTool, ToolRuntimeContract

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


def _live_session_tasks(worker) -> set[asyncio.Task]:
    """Independently count all live event-loop tasks executing worker sessions."""
    run_code = type(worker)._run_session.__code__
    return {
        t for t in asyncio.all_tasks()
        if not t.done() and getattr(t.get_coro(), "cr_code", None) is run_code
    }


def _make_descriptor() -> RuntimeDescriptor:
    return RuntimeDescriptor(
        2,
        "agent",
        "r03.agent@215",
        None,
        "r03.tools@215",
        "r03.hooks@215",
        "r03.resources@215",
        {"context_budget_tokens": 10**9},
        {},
    )


async def test_r03_ordinary_tool_run_agent_spec_completes_at_capacity_1_with_peak_concurrency_1(
    registry_store,
) -> None:
    store, _ = registry_store
    in_flight = 0
    peak_in_flight = 0
    parent_round = 0
    child_entered = asyncio.Event()
    parent_active_during_child = {"present": False, "suspended": False}

    class _DeterministicModel(LLMService):
        async def complete(self, request, **kwargs):
            nonlocal in_flight, peak_in_flight, parent_round
            in_flight += 1
            peak_in_flight = max(peak_in_flight, in_flight)
            try:
                if request.system_prompt.startswith("CHILD"):
                    child_entered.set()
                    ctx = rt.worker._active.get(parent_target.session_id)
                    parent_active_during_child["present"] = ctx is not None
                    parent_active_during_child["suspended"] = bool(ctx and ctx.slot_suspended)
                    await asyncio.sleep(0.03)
                    return LLMResponse(raw_text="child done")
                if request.messages and "peer" in str(request.messages[-1].get("content", "")):
                    await asyncio.sleep(0.03)
                    return LLMResponse(raw_text="peer done")
                parent_round += 1
                await asyncio.sleep(0.02)
                if parent_round == 1:
                    return LLMResponse(
                        raw_text="",
                        tool_calls=[{
                            "id": "call-delegate",
                            "type": "function",
                            "function": {"name": "delegate", "arguments": "{}"},
                        }],
                    )
                return LLMResponse(raw_text="parent done")
            finally:
                in_flight -= 1

        async def close(self):
            return None

    registry = ToolRegistry()

    async def delegate():
        return await run_agent_spec(
            AgentSpec(name="kid", system_prompt="CHILD", lifecycle="linked"),
            "child work",
            parent_loop=get_current_loop(),
        )

    registry.register(
        ToolDefinition("delegate", "delegate", {"type": "object", "properties": {}}, pricing=Free("test")),
        FunctionRuntimeTool(delegate, ToolRuntimeContract("test.delegate", allowed_roles=("main",))),
    )

    model = _DeterministicModel()
    factory = RegistryRuntimeFactory()

    async def _build(_descriptor, _ctx):
        return ExecutionEnvironment(
            model, registry, AgentLoopConfig(context_budget_tokens=10**9),
        )

    factory.register("r03.agent@215", _build)
    rt = Runtime(
        store=store,
        factory=factory,
        node_id="r03-node",
        boot_id="r03-boot",
        capacity=1,
        fuse_local_send=True,
        recovery_interval_s=None,
        timers_enabled=False,
    )
    actor = TrustedActor("r03", "host", None)
    await rt.worker.start()
    try:
        parent_target = await rt.open_session(_make_descriptor(), actor=actor)
        peer_target = await rt.open_session(_make_descriptor(), actor=actor)

        parent_rcpt = await rt.client.send(
            parent_target, "start parent", message_id="m-parent", actor=actor,
        )
        await asyncio.wait_for(child_entered.wait(), timeout=5.0)
        # Concurrent runnable session arrives while child holds the single slot.
        peer_rcpt = await rt.client.send(
            peer_target, "peer work", message_id="m-peer", actor=actor,
        )

        parent_res, peer_res = await asyncio.gather(
            rt.client.wait(parent_rcpt.input_ref, actor=actor, deadline_ms=now_ms() + 10000),
            rt.client.wait(peer_rcpt.input_ref, actor=actor, deadline_ms=now_ms() + 10000),
        )
        assert parent_res.status == "completed" and parent_res.final_text == "parent done"
        assert peer_res.status == "completed" and peer_res.final_text == "peer done"
        assert peak_in_flight == 1, f"model concurrency exceeded capacity=1: peak={peak_in_flight}"
        assert parent_active_during_child == {"present": True, "suspended": True}

        children = await rt.client.list_children(parent_target, actor=actor)
        assert len(children.items) == 1
        rows = await store.load_all_messages(parent_target.session_id)
        tool_msg = next(r for r in rows if r.tool_call_id == "call-delegate")
        payload = json.loads(tool_msg.content)
        assert payload["status"] == "completed"
        assert payload["final_text"] == "child done"

        while rt.worker._tasks:
            await asyncio.wait(tuple(rt.worker._tasks))
        assert rt.worker._active == {}
        assert not rt.worker._slot_waiters
    finally:
        await rt.worker.shutdown(timeout_s=2, cancel_running=True)
        await rt.client.aclose()
        await rt.transport.aclose()


async def test_r03_cancel_parent_while_waiting_for_child_frees_capacity_for_next_send(
    registry_store,
) -> None:
    store, _ = registry_store
    child_entered = asyncio.Event()
    in_flight = 0
    peak_in_flight = 0

    class _BlockingChildModel(LLMService):
        async def complete(self, request, **kwargs):
            nonlocal in_flight, peak_in_flight
            in_flight += 1
            peak_in_flight = max(peak_in_flight, in_flight)
            try:
                if request.system_prompt.startswith("CHILD"):
                    child_entered.set()
                    await asyncio.sleep(30)
                    return LLMResponse(raw_text="child unreachable")
                if request.messages and "second send" in str(request.messages[-1].get("content", "")):
                    return LLMResponse(raw_text="second ok")
                return LLMResponse(
                    raw_text="",
                    tool_calls=[{
                        "id": "call-delegate",
                        "type": "function",
                        "function": {"name": "delegate", "arguments": "{}"},
                    }],
                )
            finally:
                in_flight -= 1

        async def close(self):
            return None

    registry = ToolRegistry()

    async def delegate():
        return await run_agent_spec(
            AgentSpec(name="kid", system_prompt="CHILD", lifecycle="linked"),
            "slow child",
            parent_loop=get_current_loop(),
        )

    registry.register(
        ToolDefinition("delegate", "delegate", {"type": "object", "properties": {}}, pricing=Free("test")),
        FunctionRuntimeTool(delegate, ToolRuntimeContract("test.delegate", allowed_roles=("main",))),
    )

    factory = RegistryRuntimeFactory()
    model = _BlockingChildModel()

    async def _build(_descriptor, _ctx):
        return ExecutionEnvironment(
            model, registry, AgentLoopConfig(context_budget_tokens=10**9),
        )

    factory.register("r03.agent@215", _build)
    rt = Runtime(
        store=store,
        factory=factory,
        node_id="r03-cancel-node",
        boot_id="r03-cancel-boot",
        capacity=1,
        fuse_local_send=True,
        recovery_interval_s=None,
        timers_enabled=False,
    )
    actor = TrustedActor("r03", "host", None)
    await rt.worker.start()
    try:
        parent_target = await rt.open_session(_make_descriptor(), actor=actor)
        first_rcpt = await rt.client.send(
            parent_target, "first send", message_id="m-first", actor=actor,
        )
        await asyncio.wait_for(child_entered.wait(), timeout=5.0)

        await rt.client.cancel(parent_target, actor=actor, scope="subtree", reason="user_stop")
        first_res = await rt.client.wait(
            first_rcpt.input_ref, actor=actor, deadline_ms=now_ms() + 10000,
        )
        assert first_res.status == "cancelled"

        while rt.worker._tasks:
            await asyncio.wait(tuple(rt.worker._tasks))
        children = await rt.client.list_children(parent_target, actor=actor)
        assert len(children.items) == 1
        child_session = await rt.worker.inputs._session(rt.worker.inputs.db, children.items[0].session_id)
        assert child_session["execution_state"] == "idle"
        from power_loop.agent.runtime_types import InputRef
        child_result = await rt.worker.inputs.get_result(
            InputRef(store.store_id, children.items[0].session_id, child_session["start_input_message_id"]),
            actor=actor,
        )
        assert child_result.status == "cancelled"
        assert rt.worker._active == {}
        assert not rt.worker._slot_waiters

        # Next send must acquire the freed capacity slot and complete normally.
        second_rcpt = await rt.client.send(
            parent_target, "second send", message_id="m-second", actor=actor,
        )
        second_res = await rt.client.wait(
            second_rcpt.input_ref, actor=actor, deadline_ms=now_ms() + 10000,
        )
        assert second_res.status == "completed" and second_res.final_text == "second ok"
        assert peak_in_flight == 1
    finally:
        await rt.worker.shutdown(timeout_s=2, cancel_running=True)
        await rt.client.aclose()
        await rt.transport.aclose()


async def test_r03_cancel_parent_self_with_independent_child_keeps_concurrency_bounded(
    registry_store,
) -> None:
    store, _ = registry_store
    child_entered = asyncio.Event()
    release_child = asyncio.Event()
    in_flight = 0
    peak_in_flight = 0

    class _IndependentChildModel(LLMService):
        async def complete(self, request, **kwargs):
            nonlocal in_flight, peak_in_flight
            in_flight += 1
            peak_in_flight = max(peak_in_flight, in_flight)
            try:
                if request.system_prompt.startswith("CHILD"):
                    child_entered.set()
                    await asyncio.wait_for(release_child.wait(), timeout=10.0)
                    return LLMResponse(raw_text="independent child done")
                if request.messages and "next send" in str(request.messages[-1].get("content", "")):
                    return LLMResponse(raw_text="next ok")
                return LLMResponse(
                    raw_text="",
                    tool_calls=[{
                        "id": "call-delegate",
                        "type": "function",
                        "function": {"name": "delegate", "arguments": "{}"},
                    }],
                )
            finally:
                in_flight -= 1

        async def close(self):
            return None

    registry = ToolRegistry()

    async def delegate():
        return await run_agent_spec(
            AgentSpec(name="kid", system_prompt="CHILD", lifecycle="ephemeral"),
            "independent child",
            parent_loop=get_current_loop(),
        )

    registry.register(
        ToolDefinition("delegate", "delegate", {"type": "object", "properties": {}}, pricing=Free("test")),
        FunctionRuntimeTool(delegate, ToolRuntimeContract("test.delegate", allowed_roles=("main",))),
    )

    factory = RegistryRuntimeFactory()
    model = _IndependentChildModel()

    async def _build(_descriptor, _ctx):
        return ExecutionEnvironment(
            model, registry, AgentLoopConfig(context_budget_tokens=10**9),
        )

    factory.register("r03.agent@215", _build)
    rt = Runtime(
        store=store,
        factory=factory,
        node_id="r03-indep-node",
        boot_id="r03-indep-boot",
        capacity=1,
        fuse_local_send=True,
        recovery_interval_s=None,
        timers_enabled=False,
    )
    actor = TrustedActor("r03", "host", None)
    await rt.worker.start()
    try:
        parent_target = await rt.open_session(_make_descriptor(), actor=actor)
        first_rcpt = await rt.client.send(
            parent_target, "first send", message_id="m-first", actor=actor,
        )
        await asyncio.wait_for(child_entered.wait(), timeout=5.0)

        # Cancel only parent: parent ends cancelled promptly while independent child keeps the slot.
        await rt.client.cancel(parent_target, actor=actor, scope="self", reason="user_stop")
        first_res = await rt.client.wait(
            first_rcpt.input_ref, actor=actor, deadline_ms=now_ms() + 10000,
        )
        assert first_res.status == "cancelled"

        # A follow-up send queues while independent child still occupies capacity=1.
        next_rcpt = await rt.client.send(
            parent_target, "next send", message_id="m-next", actor=actor,
        )
        await asyncio.sleep(0.05)
        assert peak_in_flight == 1, f"next send overlapped independent child: peak={peak_in_flight}"

        release_child.set()
        next_res = await rt.client.wait(
            next_rcpt.input_ref, actor=actor, deadline_ms=now_ms() + 10000,
        )
        assert next_res.status == "completed" and next_res.final_text == "next ok"
        assert peak_in_flight == 1
        while rt.worker._tasks:
            await asyncio.wait(tuple(rt.worker._tasks))
        assert rt.worker._active == {}
        assert not rt.worker._slot_waiters
    finally:
        release_child.set()
        await rt.worker.shutdown(timeout_s=2, cancel_running=True)
        await rt.client.aclose()
        await rt.transport.aclose()


async def test_r03_backlog_parents_prioritize_awaited_children_across_backends(
    registry_store,
) -> None:
    """Queued parents must not starve awaited children or accumulate suspended tasks."""
    from power_loop.core.execution_context import get_execution_context

    store, _ = registry_store
    in_flight = 0
    peak_in_flight = 0
    max_tasks = 0
    child_calls = 0
    delegated_parents: set[str] = set()
    release_parent = asyncio.Event()

    class _BacklogModel(LLMService):
        async def complete(self, request, **kwargs):
            nonlocal in_flight, peak_in_flight, max_tasks, child_calls
            in_flight += 1
            peak_in_flight = max(peak_in_flight, in_flight)
            max_tasks = max(max_tasks, len(rt.worker._tasks))
            sid = get_execution_context().session_id
            try:
                if request.system_prompt.startswith("CHILD"):
                    child_calls += 1
                    return LLMResponse(raw_text="child done")
                if sid not in delegated_parents:
                    delegated_parents.add(sid)
                    await release_parent.wait()
                    return LLMResponse(
                        raw_text="",
                        tool_calls=[{
                            "id": "call-delegate",
                            "type": "function",
                            "function": {"name": "delegate", "arguments": "{}"},
                        }],
                    )
                return LLMResponse(raw_text="parent done")
            finally:
                in_flight -= 1

        async def close(self):
            return None

    registry = ToolRegistry()

    async def delegate():
        return await run_agent_spec(
            AgentSpec(name="kid", system_prompt="CHILD", lifecycle="linked"),
            "child work",
            parent_loop=get_current_loop(),
        )

    registry.register(
        ToolDefinition("delegate", "delegate", {"type": "object", "properties": {}}, pricing=Free("test")),
        FunctionRuntimeTool(delegate, ToolRuntimeContract("test.delegate", allowed_roles=("main",))),
    )

    model = _BacklogModel()
    factory = RegistryRuntimeFactory()

    async def _build(_descriptor, _ctx):
        return ExecutionEnvironment(
            model, registry, AgentLoopConfig(context_budget_tokens=10**9),
        )

    factory.register("r03.agent@215", _build)
    rt = Runtime(
        store=store,
        factory=factory,
        node_id="r03-backlog-node",
        boot_id="r03-backlog-boot",
        capacity=1,
        fuse_local_send=True,
        recovery_interval_s=None,
        timers_enabled=False,
    )
    actor = TrustedActor("r03", "host", None)
    await rt.worker.start()
    try:
        count = 8
        receipts = []
        for i in range(count):
            target = await rt.open_session(_make_descriptor(), actor=actor)
            receipts.append(
                await rt.client.send(target, "start parent", message_id=f"bp-{i}", actor=actor)
            )
        release_parent.set()
        results = await asyncio.gather(*(
            rt.client.wait(r.input_ref, actor=actor, deadline_ms=now_ms() + 15000)
            for r in receipts
        ))
        assert all(res.status == "completed" and res.final_text == "parent done" for res in results)
        assert child_calls == count
        assert peak_in_flight == 1
        assert max_tasks <= 4
        while rt.worker._tasks:
            await asyncio.wait(tuple(rt.worker._tasks))
        assert rt.worker._active == {}
        assert rt.worker._pending == {}
        assert rt.worker._ready_awaited == {}
        assert rt.worker._awaited_children == {}
        assert not rt.worker._slot_waiters
    finally:
        release_parent.set()
        await rt.worker.shutdown(timeout_s=2, cancel_running=True)
        await rt.client.aclose()
        await rt.transport.aclose()


async def test_r03_1024_parent_backlog_completes_within_task_limit_at_capacity_1() -> None:
    """Rework counterexample: 1024 queued parents at capacity=1 complete without saturating _tasks."""
    from power_loop import SessionStore
    from power_loop.core.execution_context import get_execution_context

    store = await SessionStore.open(":memory:")
    in_flight = 0
    peak_in_flight = 0
    max_tasks = 0
    child_calls = 0
    parent_r2 = 0
    delegated_parents: set[str] = set()
    release_parent = asyncio.Event()

    class _BacklogModel(LLMService):
        async def complete(self, request, **kwargs):
            nonlocal in_flight, peak_in_flight, max_tasks, child_calls, parent_r2
            in_flight += 1
            peak_in_flight = max(peak_in_flight, in_flight)
            max_tasks = max(max_tasks, len(rt.worker._tasks))
            sid = get_execution_context().session_id
            try:
                if request.system_prompt.startswith("CHILD"):
                    child_calls += 1
                    return LLMResponse(raw_text="child done")
                if sid not in delegated_parents:
                    delegated_parents.add(sid)
                    await release_parent.wait()
                    return LLMResponse(
                        raw_text="",
                        tool_calls=[{
                            "id": "call-delegate",
                            "type": "function",
                            "function": {"name": "delegate", "arguments": "{}"},
                        }],
                    )
                parent_r2 += 1
                return LLMResponse(raw_text="parent done")
            finally:
                in_flight -= 1

        async def close(self):
            return None

    registry = ToolRegistry()

    async def delegate():
        return await run_agent_spec(
            AgentSpec(name="kid", system_prompt="CHILD", lifecycle="linked"),
            "child work",
            parent_loop=get_current_loop(),
        )

    registry.register(
        ToolDefinition("delegate", "delegate", {"type": "object", "properties": {}}, pricing=Free("test")),
        FunctionRuntimeTool(delegate, ToolRuntimeContract("test.delegate", allowed_roles=("main",))),
    )

    model = _BacklogModel()
    factory = RegistryRuntimeFactory()

    async def _build(_descriptor, _ctx):
        return ExecutionEnvironment(
            model, registry, AgentLoopConfig(context_budget_tokens=10**9),
        )

    factory.register("r03.agent@215", _build)
    rt = Runtime(
        store=store,
        factory=factory,
        node_id="r03-1024-node",
        boot_id="r03-1024-boot",
        capacity=1,
        fuse_local_send=True,
        recovery_interval_s=None,
        timers_enabled=False,
    )
    actor = TrustedActor("r03", "host", None)
    await rt.worker.start()
    try:
        total = 1024
        last_target = None
        for i in range(total):
            last_target = await rt.open_session(_make_descriptor(), actor=actor)
            await rt.client.send(last_target, "start parent", message_id=f"p-{i}", actor=actor)
        release_parent.set()
        async with asyncio.timeout(120):
            while rt.worker._tasks or rt.worker._pending or rt.worker._ready_awaited:
                await asyncio.sleep(0.05)

        assert len(delegated_parents) == total
        assert child_calls == total
        assert parent_r2 == total
        assert peak_in_flight == 1
        assert max_tasks < 1024
        assert rt.worker._active == {}
        assert rt.worker._pending == {}
        assert rt.worker._ready_awaited == {}
        assert rt.worker._awaited_children == {}
        assert not rt.worker._slot_waiters

        # Subsequent send after draining the 1024-parent backlog still succeeds.
        assert last_target is not None
        follow_rcpt = await rt.client.send(last_target, "after backlog", message_id="p-after", actor=actor)
        follow_res = await rt.client.wait(follow_rcpt.input_ref, actor=actor, deadline_ms=now_ms() + 10000)
        assert follow_res.status == "completed" and follow_res.final_text == "parent done"
    finally:
        release_parent.set()
        await rt.worker.shutdown(timeout_s=2, cancel_running=True)
        await rt.client.aclose()
        await rt.transport.aclose()
        await store.close()


@pytest.mark.parametrize("mode", ["parent_reacquire", "child_running"])
async def test_r03_workflow_preclaim_race_keeps_peak_concurrency_1(
    registry_store, monkeypatch, mode: str,
) -> None:
    """Rework 2: pre-claimed workflow leaf racing with resuming parent or running child stays at peak=1."""
    from power_loop.agent.runtime_store import RuntimeInputStore

    store, _ = registry_store
    sampled, proceed, child_ev, leaf_ev, finish = (asyncio.Event() for _ in range(5))
    result_ready, deliver, resumed = (asyncio.Event() for _ in range(3))
    real_create_many = RuntimeInputStore.create_many

    async def _delayed_create_many(self, **kw):
        sampled.set()
        await proceed.wait()
        return await real_create_many(self, **kw)

    monkeypatch.setattr(RuntimeInputStore, "create_many", _delayed_create_many)

    in_flight = 0
    peak_in_flight = 0
    rounds = 0

    class _RaceModel(LLMService):
        async def complete(self, request, **kwargs):
            nonlocal in_flight, peak_in_flight, rounds
            in_flight += 1
            peak_in_flight = max(peak_in_flight, in_flight)
            try:
                if request.system_prompt.startswith("CHILD"):
                    child_ev.set()
                    if mode == "child_running":
                        await finish.wait()
                    return LLMResponse(raw_text="CHILD done")
                if request.system_prompt.startswith("LEAF"):
                    leaf_ev.set()
                    await finish.wait()
                    return LLMResponse(raw_text="LEAF done")
                rounds += 1
                if rounds == 1:
                    return LLMResponse(
                        raw_text="",
                        tool_calls=[{
                            "id": "call-delegate",
                            "type": "function",
                            "function": {"name": "delegate", "arguments": "{}"},
                        }],
                    )
                resumed.set()
                if mode == "parent_reacquire":
                    await finish.wait()
                return LLMResponse(raw_text="parent done")
            finally:
                in_flight -= 1

        async def close(self):
            return None

    registry = ToolRegistry()

    async def delegate():
        return await run_agent_spec(
            AgentSpec(name="kid", system_prompt="CHILD", lifecycle="linked"),
            "child work",
            parent_loop=get_current_loop(),
        )

    registry.register(
        ToolDefinition("delegate", "delegate", {"type": "object", "properties": {}}, pricing=Free("test")),
        FunctionRuntimeTool(delegate, ToolRuntimeContract("test.delegate", allowed_roles=("main",))),
    )

    model = _RaceModel()
    factory = RegistryRuntimeFactory()

    async def _build(_descriptor, _ctx):
        return ExecutionEnvironment(
            model, registry, AgentLoopConfig(context_budget_tokens=10**9),
        )

    factory.register("r03.agent@215", _build)
    rt = Runtime(
        store=store,
        factory=factory,
        node_id="r03-wf-race-node",
        boot_id="r03-wf-race-boot",
        capacity=1,
        fuse_local_send=True,
        recovery_interval_s=None,
        timers_enabled=False,
    )
    actor = TrustedActor("r03", "host", None)
    await rt.worker.start()
    try:
        wf_target = await rt.open_session(_make_descriptor(), actor=actor)
        parent_target = await rt.open_session(_make_descriptor(), actor=actor)
        wf_spec = {
            "name": "race",
            "root": {
                "type": "agent",
                "id": "leaf",
                "input": "go",
                "spec": {"name": "leaf", "system_prompt": "LEAF"},
            },
        }
        if mode == "parent_reacquire":
            real_wait = rt.client.wait

            async def _delayed_wait(ref, **kw):
                res = await real_wait(ref, **kw)
                if ref.session_id != parent_target.session_id and res.final_text == "CHILD done":
                    result_ready.set()
                    await deliver.wait()
                return res

            monkeypatch.setattr(rt.client, "wait", _delayed_wait)
            parent_rcpt = await rt.client.send(parent_target, "parent", message_id="p", actor=actor)
            await asyncio.wait_for(result_ready.wait(), timeout=5.0)
            wf_handle = await rt.client.workflow(
                wf_target, wf_spec, message_id="wf", actor=actor, descriptor=_make_descriptor(),
            )
            await asyncio.wait_for(sampled.wait(), timeout=5.0)
            deliver.set()
            await asyncio.wait_for(resumed.wait(), timeout=5.0)
            proceed.set()
            await asyncio.sleep(0.05)
            assert not leaf_ev.is_set(), "workflow leaf must wait while resumed parent holds capacity=1"
        else:
            wf_handle = await rt.client.workflow(
                wf_target, wf_spec, message_id="wf", actor=actor, descriptor=_make_descriptor(),
            )
            await asyncio.wait_for(sampled.wait(), timeout=5.0)
            parent_rcpt = await rt.client.send(parent_target, "parent", message_id="p", actor=actor)
            await asyncio.wait_for(child_ev.wait(), timeout=5.0)
            proceed.set()
            await asyncio.sleep(0.05)
            assert not leaf_ev.is_set(), "workflow leaf must wait while child holds capacity=1"

        finish.set()
        parent_res = await rt.client.wait(parent_rcpt.input_ref, actor=actor, deadline_ms=now_ms() + 10000)
        wf_res = await wf_handle.wait(deadline_ms=now_ms() + 10000)
        assert parent_res.status == "completed" and parent_res.final_text == "parent done"
        assert wf_res.status == "completed"
        assert peak_in_flight == 1
        while rt.worker._tasks:
            await asyncio.wait(tuple(rt.worker._tasks))
        assert rt.worker._active == {}
        assert rt.worker._claimed_pending == {}
        assert not rt.worker._slot_waiters
    finally:
        deliver.set()
        finish.set()
        proceed.set()
        await rt.worker.shutdown(timeout_s=2, cancel_running=True)
        await rt.client.aclose()
        await rt.transport.aclose()


async def test_r03_worker_task_limit_1024_holds_when_parent_suspends_at_full_capacity() -> None:
    """Rework 2: worker._tasks never exceeds 1024 even when a parent suspends at capacity=1024."""
    from power_loop import SessionStore

    store = await SessionStore.open(":memory:")
    release_parent = asyncio.Event()
    release_peers = asyncio.Event()
    child_entered = asyncio.Event()
    parent_calls = 0

    class _LimitModel(LLMService):
        async def complete(self, request, **kwargs):
            nonlocal parent_calls
            if request.system_prompt.startswith("CHILD"):
                child_entered.set()
                return LLMResponse(raw_text="child done")
            if request.messages[-1]["role"] == "user" and not request.messages[-1]["content"].startswith("["):
                parent_calls += 1
                if parent_calls != 1:
                    await release_peers.wait()
                    return LLMResponse(raw_text="peer done")
                await release_parent.wait()
                return LLMResponse(
                    raw_text="",
                    tool_calls=[{
                        "id": "call-delegate",
                        "type": "function",
                        "function": {"name": "delegate", "arguments": "{}"},
                    }],
                )
            return LLMResponse(raw_text="parent done")

        async def close(self):
            return None

    registry = ToolRegistry()

    async def delegate():
        return await run_agent_spec(
            AgentSpec(name="kid", system_prompt="CHILD", lifecycle="linked"),
            "child work",
            parent_loop=get_current_loop(),
        )

    registry.register(
        ToolDefinition("delegate", "delegate", {"type": "object", "properties": {}}, pricing=Free("test")),
        FunctionRuntimeTool(delegate, ToolRuntimeContract("test.delegate", allowed_roles=("main",))),
    )

    model = _LimitModel()
    factory = RegistryRuntimeFactory()

    async def _build(_descriptor, _ctx):
        return ExecutionEnvironment(
            model, registry, AgentLoopConfig(context_budget_tokens=10**9),
        )

    factory.register("r03.agent@215", _build)
    rt = Runtime(
        store=store,
        factory=factory,
        node_id="r03-limit-node",
        boot_id="r03-limit-boot",
        capacity=1024,
        fuse_local_send=True,
        recovery_interval_s=None,
        timers_enabled=False,
    )
    actor = TrustedActor("r03", "host", None)
    await rt.worker.start()
    try:
        first_rcpt = None
        for i in range(1024):
            target = await rt.open_session(_make_descriptor(), actor=actor)
            rcpt = await rt.client.send(target, "start parent", message_id=f"lim-{i}", actor=actor)
            if i == 0:
                first_rcpt = rcpt
        async with asyncio.timeout(60):
            while parent_calls < 1024:
                await asyncio.sleep(0.01)
        assert len(rt.worker._tasks) == 1024
        release_parent.set()
        await asyncio.wait_for(child_entered.wait(), timeout=5.0)
        assert child_entered.is_set()
        assert len(rt.worker._tasks) == 1024
        actual_live = _live_session_tasks(rt.worker)
        assert len(actual_live) == 1024
        assert actual_live <= rt.worker._tasks

        # Once peers finish, parent completes without ever exceeding 1024 live session tasks.
        release_peers.set()
        assert first_rcpt is not None
        first_res = await rt.client.wait(first_rcpt.input_ref, actor=actor, deadline_ms=now_ms() + 30000)
        assert first_res.status == "completed" and first_res.final_text == "parent done"
        assert child_entered.is_set()
        assert len(_live_session_tasks(rt.worker)) <= 1024
    finally:
        release_peers.set()
        release_parent.set()
        await rt.worker.shutdown(timeout_s=2, cancel_running=True)
        await rt.client.aclose()
        await rt.transport.aclose()
        await store.close()


async def test_r03_claimed_pending_lease_renewal_completes_after_full_ttl(
    registry_store, monkeypatch,
) -> None:
    """Round 3: a queued pre-claimed workflow leaf renewed across >1 TTL runs and completes."""
    from power_loop.agent.runtime_store import RuntimeInputStore

    store, _ = registry_store
    sampled, proceed, child_ev, leaf_ev, finish = (asyncio.Event() for _ in range(5))
    real_create_many = RuntimeInputStore.create_many

    async def _delayed_create_many(self, **kw):
        sampled.set()
        await proceed.wait()
        return await real_create_many(self, **kw)

    monkeypatch.setattr(RuntimeInputStore, "create_many", _delayed_create_many)

    in_flight = 0
    peak_in_flight = 0
    rounds = 0

    class _LeaseModel(LLMService):
        async def complete(self, request, **kwargs):
            nonlocal in_flight, peak_in_flight, rounds
            in_flight += 1
            peak_in_flight = max(peak_in_flight, in_flight)
            try:
                if request.system_prompt.startswith("CHILD"):
                    child_ev.set()
                    await finish.wait()
                    return LLMResponse(raw_text="CHILD done")
                if request.system_prompt.startswith("LEAF"):
                    leaf_ev.set()
                    return LLMResponse(raw_text="LEAF done")
                rounds += 1
                if rounds == 1:
                    return LLMResponse(
                        raw_text="",
                        tool_calls=[{
                            "id": "call-delegate",
                            "type": "function",
                            "function": {"name": "delegate", "arguments": "{}"},
                        }],
                    )
                return LLMResponse(raw_text="parent done")
            finally:
                in_flight -= 1

        async def close(self):
            return None

    registry = ToolRegistry()

    async def delegate():
        return await run_agent_spec(
            AgentSpec(name="kid", system_prompt="CHILD", lifecycle="linked"),
            "child work",
            parent_loop=get_current_loop(),
        )

    registry.register(
        ToolDefinition("delegate", "delegate", {"type": "object", "properties": {}}, pricing=Free("test")),
        FunctionRuntimeTool(delegate, ToolRuntimeContract("test.delegate", allowed_roles=("main",))),
    )

    model = _LeaseModel()
    factory = RegistryRuntimeFactory()

    async def _build(_descriptor, _ctx):
        return ExecutionEnvironment(
            model, registry, AgentLoopConfig(context_budget_tokens=10**9),
        )

    factory.register("r03.agent@215", _build)
    rt = Runtime(
        store=store,
        factory=factory,
        node_id="r03-lease-node",
        boot_id="r03-lease-boot",
        capacity=1,
        session_lease_ttl_s=1.5,
        fuse_local_send=True,
        recovery_interval_s=None,
        timers_enabled=False,
    )
    actor = TrustedActor("r03", "host", None)
    await rt.worker.start()
    try:
        wf_target = await rt.open_session(_make_descriptor(), actor=actor)
        parent_target = await rt.open_session(_make_descriptor(), actor=actor)
        wf_spec = {
            "name": "lease-wait",
            "root": {
                "type": "agent",
                "id": "leaf",
                "input": "go",
                "spec": {"name": "leaf", "system_prompt": "LEAF"},
            },
        }
        wf_handle = await rt.client.workflow(
            wf_target, wf_spec, message_id="wf", actor=actor, descriptor=_make_descriptor(),
        )
        await asyncio.wait_for(sampled.wait(), timeout=5.0)
        parent_rcpt = await rt.client.send(parent_target, "parent", message_id="p", actor=actor)
        await asyncio.wait_for(child_ev.wait(), timeout=5.0)
        proceed.set()
        async with asyncio.timeout(5.0):
            while not rt.worker._claimed_pending:
                await asyncio.sleep(0.01)
        leaf_sid = next(iter(rt.worker._claimed_pending))
        first_row = await rt.worker.inputs._session(rt.worker.inputs.db, leaf_sid)
        # Wait past one full 1.5s lease TTL so batch_renew must refresh local claimed_at_mono.
        await asyncio.sleep(2.2)
        renewed_row = await rt.worker.inputs._session(rt.worker.inputs.db, leaf_sid)
        assert int(renewed_row["owner_lease_until_ms"]) > int(first_row["owner_lease_until_ms"])
        assert not leaf_ev.is_set()

        finish.set()
        parent_res = await rt.client.wait(parent_rcpt.input_ref, actor=actor, deadline_ms=now_ms() + 10000)
        wf_res = await wf_handle.wait(deadline_ms=now_ms() + 10000)
        assert parent_res.status == "completed" and parent_res.final_text == "parent done"
        assert wf_res.status == "completed"
        assert leaf_ev.is_set()
        assert peak_in_flight == 1
    finally:
        finish.set()
        proceed.set()
        await rt.worker.shutdown(timeout_s=2, cancel_running=True)
        await rt.client.aclose()
        await rt.transport.aclose()


async def test_r03_all_1024_parents_delegating_at_full_limit_progress_within_1024_tasks() -> None:
    """Round 3: 1024 active parents all delegating at capacity=1024 progress & complete with <=1024 tasks."""
    from power_loop import SessionStore
    from power_loop.core.execution_context import get_execution_context

    store = await SessionStore.open(":memory:")
    release_parent = asyncio.Event()
    delegated_parents: set[str] = set()
    child_calls = 0
    parent_r2 = 0
    max_tasks = 0
    max_actual_tasks = 0

    class _FullLimitModel(LLMService):
        async def complete(self, request, **kwargs):
            nonlocal child_calls, parent_r2, max_tasks, max_actual_tasks
            max_tasks = max(max_tasks, len(rt.worker._tasks))
            sid = get_execution_context().session_id
            if request.system_prompt.startswith("CHILD"):
                child_calls += 1
                if child_calls in (1, 16, 256, 512, 1024):
                    max_actual_tasks = max(max_actual_tasks, len(_live_session_tasks(rt.worker)))
                return LLMResponse(raw_text="child done")
            if sid not in delegated_parents:
                delegated_parents.add(sid)
                await release_parent.wait()
                return LLMResponse(
                    raw_text="",
                    tool_calls=[{
                        "id": "call-delegate",
                        "type": "function",
                        "function": {"name": "delegate", "arguments": "{}"},
                    }],
                )
            parent_r2 += 1
            return LLMResponse(raw_text="parent done")

        async def close(self):
            return None

    registry = ToolRegistry()

    async def delegate():
        return await run_agent_spec(
            AgentSpec(name="kid", system_prompt="CHILD", lifecycle="linked"),
            "child work",
            parent_loop=get_current_loop(),
        )

    registry.register(
        ToolDefinition("delegate", "delegate", {"type": "object", "properties": {}}, pricing=Free("test")),
        FunctionRuntimeTool(delegate, ToolRuntimeContract("test.delegate", allowed_roles=("main",))),
    )

    model = _FullLimitModel()
    factory = RegistryRuntimeFactory()

    async def _build(_descriptor, _ctx):
        return ExecutionEnvironment(
            model, registry, AgentLoopConfig(context_budget_tokens=10**9),
        )

    factory.register("r03.agent@215", _build)
    rt = Runtime(
        store=store,
        factory=factory,
        node_id="r03-full-node",
        boot_id="r03-full-boot",
        capacity=1024,
        fuse_local_send=True,
        recovery_interval_s=None,
        timers_enabled=False,
    )
    actor = TrustedActor("r03", "host", None)
    await rt.worker.start()
    try:
        total = 1024
        for i in range(total):
            target = await rt.open_session(_make_descriptor(), actor=actor)
            await rt.client.send(target, "start parent", message_id=f"full-{i}", actor=actor)
        async with asyncio.timeout(60):
            while len(delegated_parents) < total:
                await asyncio.sleep(0.01)
        assert len(rt.worker._tasks) == total
        release_parent.set()
        async with asyncio.timeout(180):
            while rt.worker._tasks or rt.worker._pending or rt.worker._ready_awaited:
                max_tasks = max(max_tasks, len(rt.worker._tasks))
                max_actual_tasks = max(max_actual_tasks, len(_live_session_tasks(rt.worker)))
                await asyncio.sleep(0.25)
        assert child_calls == total
        assert parent_r2 == total
        assert max_tasks <= 1024
        assert max_actual_tasks <= 1024
        assert rt.worker._active == {}
        assert rt.worker._awaited_handoffs == {}
        assert rt.worker._deferred_handoffs == {}
        assert not rt.worker._slot_waiters
    finally:
        release_parent.set()
        await rt.worker.shutdown(timeout_s=2, cancel_running=True)
        await rt.client.aclose()
        await rt.transport.aclose()
        await store.close()


async def test_r03_reserved_fuse_slot_and_dispatch_respect_1024_task_limit(monkeypatch) -> None:
    """Round 3: _dispatch counts reserved local fuse slots so interleaving never exceeds 1024 tasks."""
    from power_loop import SessionStore
    from power_loop.agent.runtime_store import RuntimeInputStore

    store = await SessionStore.open(":memory:")
    release_peers = asyncio.Event()
    parked = asyncio.Event()
    proceed = asyncio.Event()
    peer_calls = 0

    class _PeerModel(LLMService):
        async def complete(self, request, **kwargs):
            nonlocal peer_calls
            peer_calls += 1
            await release_peers.wait()
            return LLMResponse(raw_text="peer done")

        async def close(self):
            return None

    factory = RegistryRuntimeFactory()
    model = _PeerModel()

    async def _build(_descriptor, _ctx):
        return ExecutionEnvironment(
            model, ToolRegistry(), AgentLoopConfig(context_budget_tokens=10**9),
        )

    factory.register("r03.agent@215", _build)
    rt = Runtime(
        store=store,
        factory=factory,
        node_id="r03-resv-node",
        boot_id="r03-resv-boot",
        capacity=1024,
        fuse_local_send=True,
        recovery_interval_s=None,
        timers_enabled=False,
    )
    actor = TrustedActor("r03", "host", None)
    await rt.worker.start()
    try:
        for i in range(1023):
            target = await rt.open_session(_make_descriptor(), actor=actor)
            await rt.client.send(target, "hold", message_id=f"r-{i}", actor=actor)
        async with asyncio.timeout(60):
            while peer_calls < 1023:
                await asyncio.sleep(0.01)
        assert len(rt.worker._tasks) == 1023

        orig_aac = RuntimeInputStore.accept_and_claim

        async def _slow_aac(self, target, body, **kw):
            if body == "slow":
                parked.set()
                await proceed.wait()
            return await orig_aac(self, target, body, **kw)

        monkeypatch.setattr(RuntimeInputStore, "accept_and_claim", _slow_aac)
        slow_target = await rt.open_session(_make_descriptor(), actor=actor)
        other_target = await rt.open_session(_make_descriptor(), actor=actor)
        slow_task = asyncio.create_task(
            rt.client.send(slow_target, "slow", message_id="r-slow", actor=actor)
        )
        await asyncio.wait_for(parked.wait(), timeout=5.0)
        await rt.client.send(other_target, "other", message_id="r-other", actor=actor)
        await asyncio.sleep(0.1)
        assert len(rt.worker._tasks) == 1023
        assert rt.worker._reserved_fuse_slots() == 1

        proceed.set()
        await slow_task
        await asyncio.sleep(0.1)
        assert len(rt.worker._tasks) <= 1024
        assert len(_live_session_tasks(rt.worker)) <= 1024
        assert rt.worker._active_llm_count() <= 1024
    finally:
        proceed.set()
        release_peers.set()
        await rt.worker.shutdown(timeout_s=0, cancel_running=True)
        await rt.client.aclose()
        await rt.transport.aclose()
        await store.close()


async def test_r03_full_capacity_1024_cancel_subtree_and_self_preserve_limit_and_contract() -> None:
    """Round 4: at full 1024 tasks, subtree cancel frees slot for next send while peers stay busy;
    parent-only cancel lets independent child continue without exceeding 1024 live session tasks."""
    from power_loop import SessionStore
    from power_loop.agent.runtime_types import InputRef

    store = await SessionStore.open(":memory:")
    release_peers = asyncio.Event()
    linked_child_entered = asyncio.Event()
    indep_child_entered = asyncio.Event()
    release_indep_child = asyncio.Event()
    peer_calls = 0

    class _CancelFullModel(LLMService):
        async def complete(self, request, **kwargs):
            nonlocal peer_calls
            if request.system_prompt.startswith("LINKED_CHILD"):
                linked_child_entered.set()
                await asyncio.sleep(30)
                return LLMResponse(raw_text="unreachable")
            if request.system_prompt.startswith("INDEP_CHILD"):
                indep_child_entered.set()
                await asyncio.wait_for(release_indep_child.wait(), timeout=15.0)
                return LLMResponse(raw_text="INDEP done")
            last_text = str(request.messages[-1].get("content", "")) if request.messages else ""
            if "hold-peer" in last_text:
                peer_calls += 1
                await release_peers.wait()
                return LLMResponse(raw_text="peer done")
            if "delegate-linked" in last_text:
                return LLMResponse(
                    raw_text="",
                    tool_calls=[{
                        "id": "call-linked",
                        "type": "function",
                        "function": {"name": "delegate_linked", "arguments": "{}"},
                    }],
                )
            if "delegate-indep" in last_text:
                return LLMResponse(
                    raw_text="",
                    tool_calls=[{
                        "id": "call-indep",
                        "type": "function",
                        "function": {"name": "delegate_indep", "arguments": "{}"},
                    }],
                )
            if "second-send" in last_text:
                return LLMResponse(raw_text="second ok")
            return LLMResponse(raw_text="parent done")

        async def close(self):
            return None

    registry = ToolRegistry()

    async def delegate_linked():
        return await run_agent_spec(
            AgentSpec(name="linked-kid", system_prompt="LINKED_CHILD", lifecycle="linked"),
            "work",
            parent_loop=get_current_loop(),
        )

    async def delegate_indep():
        return await run_agent_spec(
            AgentSpec(name="indep-kid", system_prompt="INDEP_CHILD", lifecycle="ephemeral"),
            "work",
            parent_loop=get_current_loop(),
        )

    registry.register(
        ToolDefinition("delegate_linked", "delegate", {"type": "object", "properties": {}}, pricing=Free("test")),
        FunctionRuntimeTool(delegate_linked, ToolRuntimeContract("test.delegate_linked", allowed_roles=("main",))),
    )
    registry.register(
        ToolDefinition("delegate_indep", "delegate", {"type": "object", "properties": {}}, pricing=Free("test")),
        FunctionRuntimeTool(delegate_indep, ToolRuntimeContract("test.delegate_indep", allowed_roles=("main",))),
    )

    model = _CancelFullModel()
    factory = RegistryRuntimeFactory()

    async def _build(_descriptor, _ctx):
        return ExecutionEnvironment(
            model, registry, AgentLoopConfig(context_budget_tokens=10**9),
        )

    factory.register("r03.agent@215", _build)
    rt = Runtime(
        store=store,
        factory=factory,
        node_id="r03-full-cancel-node",
        boot_id="r03-full-cancel-boot",
        capacity=1024,
        fuse_local_send=True,
        recovery_interval_s=None,
        timers_enabled=False,
    )
    actor = TrustedActor("r03", "host", None)
    await rt.worker.start()
    try:
        for i in range(1023):
            peer = await rt.open_session(_make_descriptor(), actor=actor)
            await rt.client.send(peer, "hold-peer", message_id=f"hp-{i}", actor=actor)
        async with asyncio.timeout(60):
            while peer_calls < 1023:
                await asyncio.sleep(0.01)
        assert len(rt.worker._tasks) == 1023

        # Part A: subtree cancel at full 1024 tasks -> both cancelled, next send completes while 1023 peers run.
        parent_target = await rt.open_session(_make_descriptor(), actor=actor)
        first_rcpt = await rt.client.send(parent_target, "delegate-linked", message_id="fc-1", actor=actor)
        await asyncio.wait_for(linked_child_entered.wait(), timeout=5.0)
        live_during_linked = _live_session_tasks(rt.worker)
        assert len(rt.worker._tasks) == 1024
        assert len(live_during_linked) == 1024
        assert live_during_linked <= rt.worker._tasks

        await rt.client.cancel(parent_target, actor=actor, scope="subtree", reason="user_stop")
        first_res = await rt.client.wait(first_rcpt.input_ref, actor=actor, deadline_ms=now_ms() + 10000)
        assert first_res.status == "cancelled"
        children = await rt.client.list_children(parent_target, actor=actor)
        assert len(children.items) == 1
        linked_sid = children.items[0].session_id
        async with asyncio.timeout(5.0):
            while linked_sid in rt.worker._active or parent_target.session_id in rt.worker._active:
                await asyncio.sleep(0.01)
        linked_row = await rt.worker.inputs._session(rt.worker.inputs.db, linked_sid)
        linked_res = await rt.worker.inputs.get_result(
            InputRef(store.store_id, linked_sid, linked_row["start_input_message_id"]),
            actor=actor,
        )
        assert linked_res.status == "cancelled"

        # 1023 peers still occupy slots; same parent's next send must take the freed 1024th slot and complete.
        second_rcpt = await rt.client.send(parent_target, "second-send", message_id="fc-2", actor=actor)
        second_res = await rt.client.wait(second_rcpt.input_ref, actor=actor, deadline_ms=now_ms() + 10000)
        assert second_res.status == "completed" and second_res.final_text == "second ok"
        assert len(_live_session_tasks(rt.worker)) <= 1024

        # Part B: parent-only cancel at full 1024 tasks -> independent child continues and completes, <=1024 tasks.
        third_rcpt = await rt.client.send(parent_target, "delegate-indep", message_id="fc-3", actor=actor)
        await asyncio.wait_for(indep_child_entered.wait(), timeout=5.0)
        live_during_indep = _live_session_tasks(rt.worker)
        assert len(rt.worker._tasks) == 1024
        assert len(live_during_indep) == 1024
        assert live_during_indep <= rt.worker._tasks

        await rt.client.cancel(parent_target, actor=actor, scope="self", reason="user_stop")
        third_res = await rt.client.wait(third_rcpt.input_ref, actor=actor, deadline_ms=now_ms() + 10000)
        assert third_res.status == "cancelled"
        await asyncio.sleep(0.05)
        assert len(rt.worker._tasks) <= 1024
        assert len(_live_session_tasks(rt.worker)) <= 1024

        all_children = await rt.client.list_children(parent_target, actor=actor)
        indep_sid = next(c.session_id for c in all_children.items if c.session_id != linked_sid)
        indep_row = await rt.worker.inputs._session(rt.worker.inputs.db, indep_sid)
        release_indep_child.set()
        indep_res = await rt.client.wait(
            InputRef(store.store_id, indep_sid, indep_row["start_input_message_id"]),
            actor=actor,
            deadline_ms=now_ms() + 10000,
        )
        assert indep_res.status == "completed" and indep_res.final_text == "INDEP done"

        release_peers.set()
        async with asyncio.timeout(30.0):
            while rt.worker._tasks:
                await asyncio.wait(tuple(rt.worker._tasks))
        assert rt.worker._active == {}
        assert rt.worker._deferred_handoffs == {}
    finally:
        release_indep_child.set()
        release_peers.set()
        await rt.worker.shutdown(timeout_s=2, cancel_running=True)
        await rt.client.aclose()
        await rt.transport.aclose()
        await store.close()


@pytest.mark.parametrize("count", [1, 1024])
async def test_r03_child_internal_asyncio_timeout_completes_under_light_and_full_load(
    registry_store, count: int,
) -> None:
    """Round 5: child's own asyncio.timeout(0.05) raises TimeoutError inside child without
    detaching/stalling parent or child, under both light (1) and full (1024) worker load."""
    from power_loop.agent.runtime_types import InputRef

    store, _ = registry_store
    release_parent = asyncio.Event()
    release_peers = asyncio.Event()
    entered_factory = asyncio.Event()
    caught_timeout = asyncio.Event()
    child_calls = 0
    parent_calls = 0

    class _TimeoutModel(LLMService):
        async def complete(self, request, **kwargs):
            nonlocal child_calls, parent_calls
            if request.system_prompt.startswith("CHILD"):
                child_calls += 1
                return LLMResponse(raw_text="child done")
            if "start parent" in str(request.messages[-1].get("content", "")):
                parent_calls += 1
                if parent_calls != 1:
                    await release_peers.wait()
                    return LLMResponse(raw_text="peer done")
                await release_parent.wait()
                return LLMResponse(
                    raw_text="",
                    tool_calls=[{
                        "id": "call-delegate",
                        "type": "function",
                        "function": {"name": "delegate", "arguments": "{}"},
                    }],
                )
            return LLMResponse(raw_text="parent done")

        async def close(self):
            return None

    registry = ToolRegistry()

    async def delegate():
        return await run_agent_spec(
            AgentSpec(name="kid", system_prompt="CHILD", lifecycle="linked"),
            "child work",
            parent_loop=get_current_loop(),
        )

    registry.register(
        ToolDefinition("delegate", "delegate", {"type": "object", "properties": {}}, pricing=Free("test")),
        FunctionRuntimeTool(delegate, ToolRuntimeContract("test.delegate", allowed_roles=("main",))),
    )

    model = _TimeoutModel()
    factory = RegistryRuntimeFactory()

    async def _build(_descriptor, _ctx):
        if "child_spec" in _descriptor.config:
            entered_factory.set()
            try:
                async with asyncio.timeout(0.05):
                    await release_peers.wait()
            except TimeoutError:
                caught_timeout.set()
        return ExecutionEnvironment(
            model, registry, AgentLoopConfig(context_budget_tokens=10**9),
        )

    factory.register("r03.agent@215", _build)
    rt = Runtime(
        store=store,
        factory=factory,
        node_id="r03-timeout-node",
        boot_id="r03-timeout-boot",
        capacity=1024,
        fuse_local_send=True,
        recovery_interval_s=None,
        timers_enabled=False,
    )
    actor = TrustedActor("r03", "host", None)
    await rt.worker.start()
    try:
        parent_target = None
        first_rcpt = None
        for i in range(count):
            t = await rt.open_session(_make_descriptor(), actor=actor)
            rcpt = await rt.client.send(t, "start parent", message_id=f"to-{i}", actor=actor)
            if i == 0:
                parent_target, first_rcpt = t, rcpt
        async with asyncio.timeout(60):
            while parent_calls < count:
                await asyncio.sleep(0.01)
        assert len(rt.worker._tasks) == count
        release_parent.set()
        await asyncio.wait_for(entered_factory.wait(), timeout=5.0)
        assert len(_live_session_tasks(rt.worker)) <= 1024

        assert first_rcpt is not None and parent_target is not None
        parent_res = await rt.client.wait(first_rcpt.input_ref, actor=actor, deadline_ms=now_ms() + 10000)
        assert parent_res.status == "completed" and parent_res.final_text == "parent done"
        assert caught_timeout.is_set()
        assert child_calls == 1
        assert sum(map(len, rt.worker._deferred_handoffs.values())) == 0
        assert not rt.worker._slot_waiters

        children = await rt.client.list_children(parent_target, actor=actor)
        assert len(children.items) == 1
        child_sid = children.items[0].session_id
        child_row = await rt.worker.inputs._session(rt.worker.inputs.db, child_sid)
        child_res = await rt.worker.inputs.get_result(
            InputRef(store.store_id, child_sid, child_row["start_input_message_id"]),
            actor=actor,
        )
        assert child_res.status == "completed" and child_res.final_text == "child done"
        assert len(_live_session_tasks(rt.worker)) <= 1024
    finally:
        release_peers.set()
        release_parent.set()
        await rt.worker.shutdown(timeout_s=2, cancel_running=True)
        await rt.client.aclose()
        await rt.transport.aclose()


@pytest.mark.parametrize("count", [1, 1024])
@pytest.mark.parametrize(
    ("interrupt_mode", "wrapped"),
    [("finish", False), ("abort", False), ("abort", True)],
)
async def test_r03_independent_child_timeout_fires_after_parent_only_cancel(
    registry_store, count: int, interrupt_mode: str, wrapped: bool,
) -> None:
    """Round 6/7: after parent cancel(scope='self'), parent tool (interrupt='abort' or
    'finish', direct await or asyncio.gather) stops executing post-await code while the
    already-started independent child keeps its original asyncio.timeout deadline, calls
    the model once, completes, and keeps live session tasks <= 1024 at every step."""
    from power_loop.agent.runtime_types import InputRef

    store, _ = registry_store
    release_parent = asyncio.Event()
    release_peers = asyncio.Event()
    entered_factory = asyncio.Event()
    caught_timeout = asyncio.Event()
    deadlines: list[asyncio.Timeout] = []
    resumed_tool: list[object] = []
    child_calls = 0
    parent_calls = 0

    class _FixedTimeoutModel(LLMService):
        async def complete(self, request, **kwargs):
            nonlocal child_calls, parent_calls
            if request.system_prompt.startswith("CHILD"):
                child_calls += 1
                return LLMResponse(raw_text="child done")
            if "start parent" in str(request.messages[-1].get("content", "")):
                parent_calls += 1
                if parent_calls != 1:
                    await release_peers.wait()
                    return LLMResponse(raw_text="peer done")
                await release_parent.wait()
                return LLMResponse(
                    raw_text="",
                    tool_calls=[{
                        "id": "call-delegate",
                        "type": "function",
                        "function": {"name": "delegate", "arguments": "{}"},
                    }],
                )
            return LLMResponse(raw_text="parent done")

        async def close(self):
            return None

    registry = ToolRegistry()

    async def delegate():
        call = run_agent_spec(
            AgentSpec(name="kid", system_prompt="CHILD", lifecycle="ephemeral"),
            "child work",
            parent_loop=get_current_loop(),
        )
        if wrapped:
            (result,) = await asyncio.gather(call)
        else:
            result = await call
        resumed_tool.append(result)
        return result

    registry.register(
        ToolDefinition(
            "delegate",
            "delegate",
            {"type": "object", "properties": {}},
            pricing=Free("test"),
            interrupt=interrupt_mode,
        ),
        FunctionRuntimeTool(delegate, ToolRuntimeContract("test.delegate", allowed_roles=("main",))),
    )

    model = _FixedTimeoutModel()
    factory = RegistryRuntimeFactory()

    async def _build(_descriptor, _ctx):
        if "child_spec" in _descriptor.config:
            entered_factory.set()
            try:
                async with asyncio.timeout(1.0) as deadline:
                    deadlines.append(deadline)
                    await release_peers.wait()
            except TimeoutError:
                caught_timeout.set()
        return ExecutionEnvironment(
            model, registry, AgentLoopConfig(context_budget_tokens=10**9),
        )

    factory.register("r03.agent@215", _build)
    rt = Runtime(
        store=store,
        factory=factory,
        node_id="r03-fixed-to-node",
        boot_id="r03-fixed-to-boot",
        capacity=1024,
        fuse_local_send=True,
        recovery_interval_s=None,
        timers_enabled=False,
    )
    actor = TrustedActor("r03", "host", None)
    await rt.worker.start()
    try:
        parent_target = None
        first_rcpt = None
        for i in range(count):
            t = await rt.open_session(_make_descriptor(), actor=actor)
            rcpt = await rt.client.send(t, "start parent", message_id=f"ft-{i}", actor=actor)
            if i == 0:
                parent_target, first_rcpt = t, rcpt
        async with asyncio.timeout(60):
            while parent_calls < count:
                await asyncio.sleep(0.01)
        assert len(rt.worker._tasks) == count
        release_parent.set()
        await asyncio.wait_for(entered_factory.wait(), timeout=5.0)

        # Observation point 1: child has entered factory timeout while peers still run.
        live_1 = _live_session_tasks(rt.worker)
        assert len(live_1) <= 1024
        assert live_1 <= rt.worker._tasks
        if count == 1024:
            assert len(live_1) == 1024

        assert parent_target is not None and first_rcpt is not None
        children = await rt.client.list_children(parent_target, actor=actor)
        assert len(children.items) == 1
        child_sid = children.items[0].session_id
        child_row = await rt.worker.inputs._session(rt.worker.inputs.db, child_sid)
        child_ref = InputRef(store.store_id, child_sid, child_row["start_input_message_id"])

        await rt.client.cancel(parent_target, scope="self", reason="review", actor=actor)
        parent_res = await rt.client.wait(first_rcpt.input_ref, actor=actor, deadline_ms=now_ms() + 5000)
        assert parent_res.status == "cancelled"
        async with asyncio.timeout(5.0):
            while parent_target.session_id in rt.worker._active:
                await asyncio.sleep(0.01)
        await asyncio.sleep(0.02)

        # Observation point 2: parent exited, handoff completed before 1.0s deadline expires.
        assert not deadlines[0].expired()
        live_2 = _live_session_tasks(rt.worker)
        assert len(live_2) <= 1024
        assert live_2 <= rt.worker._tasks
        if count == 1024:
            assert len(live_2) == 1024

        # Observation point 3: independent child continues, hits original 1.0s timeout, calls model once, completes.
        child_res = await rt.client.wait(child_ref, actor=actor, deadline_ms=now_ms() + 3000)
        live_3 = _live_session_tasks(rt.worker)
        assert len(live_3) <= 1024
        assert live_3 <= rt.worker._tasks
        assert deadlines[0].expired()
        assert caught_timeout.is_set()
        assert child_calls == 1
        assert child_res.status == "completed" and child_res.final_text == "child done"
        final_child_row = await rt.worker.inputs._session(rt.worker.inputs.db, child_sid)
        assert final_child_row["execution_state"] == "idle"
        await asyncio.sleep(0.1)
        assert resumed_tool == [], f"parent tool resumed after cancel: {resumed_tool!r}"
    finally:
        release_peers.set()
        release_parent.set()
        await rt.worker.shutdown(timeout_s=2, cancel_running=True)
        await rt.client.aclose()
        await rt.transport.aclose()


@pytest.mark.parametrize("count", [1, 1024])
@pytest.mark.parametrize(
    "parent_timeout_site",
    [
        "finish_hook",
        "resource_aclose",
        "resource_exitstack",
        "resource_reschedule",
        "resource_anyio",
    ],
)
async def test_r03_parent_teardown_timeout_does_not_cancel_independent_child(
    registry_store, count: int, parent_timeout_site: str,
) -> None:
    """Round 8/9/10/11: under light (1) and full (1024) load, a timeout in the cancelled
    parent's EXECUTION_FINISHED notification or resource aclose() (direct, via
    AsyncExitStack, externally rescheduled from a host coroutine, or anyio.fail_after)
    is routed to the parent fiber only, leaving the independent child running to
    completion with <= 1024 live tasks."""
    from contextlib import AsyncExitStack

    from anyio import fail_after

    from power_loop.agent.runtime_types import InputRef
    from power_loop.contracts.hooks import HookPoint
    from power_loop.core.hooks import AgentHooks

    store, _ = registry_store
    release_parent = asyncio.Event()
    release_peers = asyncio.Event()
    release_child = asyncio.Event()
    entered_factory = asyncio.Event()
    hook_entered = asyncio.Event()
    close_entered = asyncio.Event()
    close_release = asyncio.Event()
    parent_cleanup_caught = asyncio.Event()
    child_timeout_caught = asyncio.Event()
    cleanup_deadline: asyncio.Future[asyncio.Timeout] = asyncio.get_running_loop().create_future()
    deadlines: list[asyncio.Timeout] = []
    first_sid: list[str | None] = [None]
    child_calls = 0
    parent_calls = 0

    class _TeardownModel(LLMService):
        async def complete(self, request, **kwargs):
            nonlocal child_calls, parent_calls
            if request.system_prompt.startswith("CHILD"):
                child_calls += 1
                return LLMResponse(raw_text="child done")
            if "start parent" in str(request.messages[-1].get("content", "")):
                parent_calls += 1
                if parent_calls != 1:
                    await release_peers.wait()
                    return LLMResponse(raw_text="peer done")
                await release_parent.wait()
                return LLMResponse(
                    raw_text="",
                    tool_calls=[{
                        "id": "call-delegate",
                        "type": "function",
                        "function": {"name": "delegate", "arguments": "{}"},
                    }],
                )
            return LLMResponse(raw_text="parent done")

        async def close(self):
            return None

    registry = ToolRegistry()

    async def delegate():
        return await run_agent_spec(
            AgentSpec(name="kid", system_prompt="CHILD", lifecycle="ephemeral"),
            "child work",
            parent_loop=get_current_loop(),
        )

    registry.register(
        ToolDefinition("delegate", "delegate", {"type": "object", "properties": {}}, pricing=Free("test")),
        FunctionRuntimeTool(delegate, ToolRuntimeContract("test.delegate", allowed_roles=("main",))),
    )

    class _SlowCloseResources:
        async def aclose(self):
            close_entered.set()
            try:
                if parent_timeout_site == "resource_exitstack":
                    async with AsyncExitStack() as stack:
                        await stack.enter_async_context(asyncio.timeout(0.1))
                        await close_release.wait()
                elif parent_timeout_site == "resource_reschedule":
                    async with AsyncExitStack() as stack:
                        deadline = await stack.enter_async_context(asyncio.timeout(None))
                        if not cleanup_deadline.done():
                            cleanup_deadline.set_result(deadline)
                        await close_release.wait()
                elif parent_timeout_site == "resource_anyio":
                    with fail_after(0.1):
                        await close_release.wait()
                else:
                    async with asyncio.timeout(0.1):
                        await close_release.wait()
            except TimeoutError:
                parent_cleanup_caught.set()

    model = _TeardownModel()
    factory = RegistryRuntimeFactory()

    async def _build(_descriptor, _ctx):
        if "child_spec" in _descriptor.config:
            entered_factory.set()
            if parent_timeout_site == "finish_hook":
                await release_child.wait()
            else:
                try:
                    async with asyncio.timeout(60) as deadline:
                        deadlines.append(deadline)
                        await release_child.wait()
                        deadline.reschedule(asyncio.get_running_loop().time())
                        await release_peers.wait()
                except TimeoutError:
                    child_timeout_caught.set()
        resources = (
            _SlowCloseResources()
            if parent_timeout_site in (
                "resource_aclose",
                "resource_exitstack",
                "resource_reschedule",
                "resource_anyio",
            )
            and _ctx.session_id == first_sid[0]
            else None
        )
        return ExecutionEnvironment(
            model, registry, AgentLoopConfig(context_budget_tokens=10**9), resources=resources,
        )

    factory.register("r03.agent@215", _build)

    hooks = None
    hook_timeout_s = 1.0
    if parent_timeout_site == "finish_hook":
        hooks = AgentHooks()

        async def _finish_hook(context):
            if context.values["execution"].session_id == first_sid[0]:
                hook_entered.set()
                await asyncio.sleep(0.3)

        hooks.register(HookPoint.EXECUTION_FINISHED, _finish_hook)
        hook_timeout_s = 0.05

    rt = Runtime(
        store=store,
        factory=factory,
        hooks=hooks,
        hook_timeout_s=hook_timeout_s,
        node_id="r03-teardown-node",
        boot_id="r03-teardown-boot",
        capacity=1024,
        fuse_local_send=True,
        recovery_interval_s=None,
        timers_enabled=False,
    )
    actor = TrustedActor("r03", "host", None)
    await rt.worker.start()
    try:
        parent_target = None
        first_rcpt = None
        for i in range(count):
            t = await rt.open_session(_make_descriptor(), actor=actor)
            if i == 0:
                first_sid[0] = t.session_id
            rcpt = await rt.client.send(t, "start parent", message_id=f"td-{i}", actor=actor)
            if i == 0:
                parent_target, first_rcpt = t, rcpt
        async with asyncio.timeout(60):
            while parent_calls < count:
                await asyncio.sleep(0.01)
        assert len(rt.worker._tasks) == count
        release_parent.set()
        await asyncio.wait_for(entered_factory.wait(), timeout=5.0)
        assert len(_live_session_tasks(rt.worker)) <= 1024

        assert parent_target is not None and first_rcpt is not None
        children = await rt.client.list_children(parent_target, actor=actor)
        assert len(children.items) == 1
        child_sid = children.items[0].session_id
        child_row = await rt.worker.inputs._session(rt.worker.inputs.db, child_sid)
        child_ref = InputRef(store.store_id, child_sid, child_row["start_input_message_id"])

        await rt.client.cancel(parent_target, scope="self", reason="review", actor=actor)
        if parent_timeout_site == "resource_reschedule":
            timer = await asyncio.wait_for(cleanup_deadline, timeout=3.0)
            timer.reschedule(asyncio.get_running_loop().time() + 0.1)
        parent_res = await rt.client.wait(first_rcpt.input_ref, actor=actor, deadline_ms=now_ms() + 5000)
        assert parent_res.status == "cancelled"

        if parent_timeout_site == "finish_hook":
            await asyncio.wait_for(hook_entered.wait(), timeout=2.0)
            await asyncio.sleep(0.15)
            async with asyncio.timeout(5.0):
                while parent_target.session_id in rt.worker._active:
                    await asyncio.sleep(0.01)
            live_mid = _live_session_tasks(rt.worker)
            assert len(live_mid) <= 1024 and live_mid <= rt.worker._tasks
            release_child.set()
        else:
            await asyncio.wait_for(close_entered.wait(), timeout=2.0)
            await asyncio.wait_for(parent_cleanup_caught.wait(), timeout=2.0)
            async with asyncio.timeout(5.0):
                while parent_target.session_id in rt.worker._active:
                    await asyncio.sleep(0.01)
            assert not deadlines[0].expired()
            live_mid = _live_session_tasks(rt.worker)
            assert len(live_mid) <= 1024 and live_mid <= rt.worker._tasks
            release_child.set()

        child_res = await rt.client.wait(child_ref, actor=actor, deadline_ms=now_ms() + 3000)
        assert child_res.status == "completed" and child_res.final_text == "child done"
        assert child_calls == 1
        if parent_timeout_site in (
            "resource_aclose",
            "resource_exitstack",
            "resource_reschedule",
            "resource_anyio",
        ):
            assert deadlines[0].expired()
            assert child_timeout_caught.is_set()
        final_child_row = await rt.worker.inputs._session(rt.worker.inputs.db, child_sid)
        assert final_child_row["execution_state"] == "idle"
        assert len(_live_session_tasks(rt.worker)) <= 1024
    finally:
        close_release.set()
        release_child.set()
        release_peers.set()
        release_parent.set()
        await rt.worker.shutdown(timeout_s=0, cancel_running=True)
        await rt.client.aclose()
        await rt.transport.aclose()


async def test_r03_child_factory_anyio_fail_after_completes_at_capacity_1(
    registry_store,
) -> None:
    """Round 11: child factory using anyio.fail_after catches TimeoutError and completes
    at capacity=1 with peak concurrency=1."""
    from anyio import fail_after

    store, _ = registry_store
    in_flight = 0
    peak_in_flight = 0
    parent_round = 0
    child_calls = 0

    class _AnyioFactoryModel(LLMService):
        async def complete(self, request, **kwargs):
            nonlocal in_flight, peak_in_flight, parent_round, child_calls
            in_flight += 1
            peak_in_flight = max(peak_in_flight, in_flight)
            try:
                if request.system_prompt.startswith("CHILD"):
                    child_calls += 1
                    return LLMResponse(raw_text="child done")
                parent_round += 1
                if parent_round == 1:
                    return LLMResponse(
                        raw_text="",
                        tool_calls=[{
                            "id": "call-delegate",
                            "type": "function",
                            "function": {"name": "delegate", "arguments": "{}"},
                        }],
                    )
                return LLMResponse(raw_text="parent done")
            finally:
                in_flight -= 1

        async def close(self):
            return None

    registry = ToolRegistry()

    async def delegate():
        return await run_agent_spec(
            AgentSpec(name="kid", system_prompt="CHILD", lifecycle="linked"),
            "child work",
            parent_loop=get_current_loop(),
        )

    registry.register(
        ToolDefinition("delegate", "delegate", {"type": "object", "properties": {}}, pricing=Free("test")),
        FunctionRuntimeTool(delegate, ToolRuntimeContract("test.delegate", allowed_roles=("main",))),
    )

    model = _AnyioFactoryModel()
    factory = RegistryRuntimeFactory()
    child_timeout_caught = asyncio.Event()

    async def _build(_descriptor, _ctx):
        if "child_spec" in _descriptor.config:
            try:
                with fail_after(0.03):
                    await asyncio.sleep(0.3)
            except TimeoutError:
                child_timeout_caught.set()
            else:
                raise RuntimeError("child factory anyio.fail_after deadline missed")
        return ExecutionEnvironment(
            model, registry, AgentLoopConfig(context_budget_tokens=10**9),
        )

    factory.register("r03.agent@215", _build)
    rt = Runtime(
        store=store,
        factory=factory,
        node_id="r03-anyio-node",
        boot_id="r03-anyio-boot",
        capacity=1,
        fuse_local_send=True,
        recovery_interval_s=None,
        timers_enabled=False,
    )
    actor = TrustedActor("r03", "host", None)
    await rt.worker.start()
    try:
        parent_target = await rt.open_session(_make_descriptor(), actor=actor)
        parent_rcpt = await rt.client.send(
            parent_target, "start parent", message_id="m-anyio-parent", actor=actor,
        )
        parent_res = await rt.client.wait(
            parent_rcpt.input_ref, actor=actor, deadline_ms=now_ms() + 10000,
        )
        assert parent_res.status == "completed" and parent_res.final_text == "parent done"
        assert child_timeout_caught.is_set()
        assert child_calls == 1
        assert peak_in_flight == 1
        while rt.worker._tasks:
            await asyncio.wait(tuple(rt.worker._tasks))
        assert rt.worker._active == {}
    finally:
        await rt.worker.shutdown(timeout_s=2, cancel_running=True)
        await rt.client.aclose()
        await rt.transport.aclose()


async def test_r03_fiber_task_handle_propagates_session_exceptions(
    monkeypatch, caplog,
) -> None:
    """R03: fiber task handles reflect escaping exceptions for both primary and handed-off
    sessions without breaking normal completion, cancellation, sibling execution, or error logs."""
    import logging

    from power_loop import SessionStore
    from power_loop.agent.runtime_worker import _FiberTaskHandle

    store = await SessionStore.open(":memory:")
    factory = RegistryRuntimeFactory()
    rt = Runtime(
        store=store,
        factory=factory,
        node_id="r03-exc-node",
        boot_id="r03-exc-boot",
        capacity=1,
        fuse_local_send=True,
        recovery_interval_s=None,
        timers_enabled=False,
    )
    loop = asyncio.get_running_loop()

    try:
        # 1. Primary session raises ValueError("r03-probe"): outer task and saved handle
        #    must both receive the exact same exception instance.
        primary_err = ValueError("r03-probe")
        saved_primary: list[asyncio.Future] = []

        async def _primary_fail(sid, *, preclaimed_batch=None):
            h = asyncio.current_task()
            assert isinstance(h, _FiberTaskHandle)
            saved_primary.append(h)
            await asyncio.sleep(0)
            raise primary_err

        monkeypatch.setattr(rt.worker, "_run_one_session", _primary_fail)
        outer_primary = asyncio.create_task(rt.worker._run_session("primary-fail"))
        with pytest.raises(ValueError, match="r03-probe") as outer_exc:
            await outer_primary
        assert outer_exc.value is primary_err
        assert len(saved_primary) == 1
        primary_handle = saved_primary[0]
        assert primary_handle.done() and not primary_handle.cancelled()
        with pytest.raises(ValueError, match="r03-probe") as handle_exc:
            await primary_handle
        assert handle_exc.value is primary_err

        # 2. Handed-off sessions on the same worker task:
        #    - failing handed-off session reports original exception on its handle & logs error,
        #    - cancelled handed-off session stays cancelled() and raises CancelledError,
        #    - sibling handed-off session and primary session still complete with None.
        handoff_err = ValueError("r03-handoff-probe")
        handles: dict[str, asyncio.Future] = {}
        handoff_handle_ready = asyncio.Event()
        release_handoff_fail = asyncio.Event()
        handoff_failed = asyncio.Event()
        cancel_entered = asyncio.Event()
        sibling_done = asyncio.Event()
        outer_holder: list[asyncio.Task] = []

        async def _multiplexed(sid, *, preclaimed_batch=None):
            h = asyncio.current_task()
            assert isinstance(h, _FiberTaskHandle)
            handles[sid] = h
            if sid == "primary-ok":
                host = rt.worker._deferred_handoffs[outer_holder[0]]
                host.enqueue("handoff-fail-awaited", None)
                host.enqueue("handoff-cancel", None)
                host.enqueue("handoff-ok", None)
                await asyncio.wait_for(cancel_entered.wait(), timeout=2.0)
                handles["handoff-cancel"].cancel("r03-cancel")
                await asyncio.wait_for(sibling_done.wait(), timeout=2.0)
                return None
            if sid == "handoff-fail-awaited":
                handoff_handle_ready.set()
                await asyncio.wait_for(release_handoff_fail.wait(), timeout=2.0)
                handoff_failed.set()
                raise handoff_err
            if sid == "handoff-cancel":
                cancel_entered.set()
                await loop.create_future()
                return None
            if sid == "handoff-ok":
                await asyncio.wait_for(handoff_failed.wait(), timeout=2.0)
                await asyncio.sleep(0)
                sibling_done.set()
                return None
            raise AssertionError(f"unexpected sid: {sid}")

        monkeypatch.setattr(rt.worker, "_run_one_session", _multiplexed)
        with caplog.at_level(logging.ERROR, logger="power_loop.agent.runtime_worker"):
            outer_ok = asyncio.create_task(rt.worker._run_session("primary-ok"))
            outer_holder.append(outer_ok)
            await asyncio.wait_for(handoff_handle_ready.wait(), timeout=2.0)

            async def _await_handle(fut: asyncio.Future):
                return await fut

            live_waiter = asyncio.create_task(_await_handle(handles["handoff-fail-awaited"]))
            await asyncio.sleep(0)
            release_handoff_fail.set()
            assert await outer_ok is None

        with pytest.raises(ValueError, match="r03-handoff-probe") as live_exc:
            await live_waiter
        assert live_exc.value is handoff_err

        assert await handles["primary-ok"] is None
        assert await handles["handoff-ok"] is None

        cancel_handle = handles["handoff-cancel"]
        assert cancel_handle.done() and cancel_handle.cancelled()
        with pytest.raises(asyncio.CancelledError, match="r03-cancel"):
            await cancel_handle

        fail_handle = handles["handoff-fail-awaited"]
        assert fail_handle.done() and not fail_handle.cancelled()
        assert fail_handle.exception() is handoff_err
        with pytest.raises(ValueError, match="r03-handoff-probe") as handoff_exc:
            await fail_handle
        assert handoff_exc.value is handoff_err

        assert any(
            "handed-off runtime execution failed for handoff-fail-awaited" in rec.getMessage()
            and rec.exc_info is not None
            and rec.exc_info[1] is handoff_err
            for rec in caplog.records
        )
    finally:
        await rt.worker.shutdown(timeout_s=2, cancel_running=True)
        await rt.client.aclose()
        await rt.transport.aclose()
        await store.close()


@pytest.mark.parametrize("role", ["primary", "handoff"])
async def test_r03_unawaited_failed_fiber_handle_emits_no_unretrieved_warning(
    monkeypatch, caplog, role: str,
) -> None:
    """R03: unawaited failing primary and handed-off _FiberTaskHandle instances emit no
    '_FiberTaskHandle exception was never retrieved' warning once actually garbage-collected."""
    import gc
    import logging
    import traceback
    import weakref

    from power_loop import SessionStore
    from power_loop.agent.runtime_worker import _FiberTaskHandle

    store = await SessionStore.open(":memory:")
    factory = RegistryRuntimeFactory()
    rt = Runtime(
        store=store,
        factory=factory,
        node_id=f"r03-unawaited-{role}-node",
        boot_id=f"r03-unawaited-{role}-boot",
        capacity=1,
        fuse_local_send=True,
        recovery_interval_s=None,
        timers_enabled=False,
    )
    loop = asyncio.get_running_loop()
    loop_errors: list[dict[str, object]] = []
    prev_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, ctx: loop_errors.append(ctx))

    def _capture_handle_weakref() -> weakref.ReferenceType[_FiberTaskHandle]:
        cur = asyncio.current_task()
        assert isinstance(cur, _FiberTaskHandle)
        return weakref.ref(cur)

    def _unretrieved_fiber_warnings() -> list[dict[str, object]]:
        return [
            ctx
            for ctx in loop_errors
            if isinstance(ctx.get("future"), _FiberTaskHandle)
            and str(ctx.get("message", "")).endswith("exception was never retrieved")
        ]

    try:
        handle_ref: list[weakref.ReferenceType[_FiberTaskHandle] | None] = [None]

        if role == "primary":
            async def _primary_fail_unawaited(sid, *, preclaimed_batch=None):
                handle_ref[0] = _capture_handle_weakref()
                await asyncio.sleep(0)
                raise ValueError("r03-primary-unawaited")

            monkeypatch.setattr(rt.worker, "_run_one_session", _primary_fail_unawaited)

            async def _run_primary() -> str:
                task = asyncio.create_task(rt.worker._run_session("primary-fail-unawaited"))
                try:
                    await task
                except ValueError as exc:
                    msg = str(exc)
                    if exc.__traceback__ is not None:
                        traceback.clear_frames(exc.__traceback__)
                    return msg
                raise AssertionError("expected ValueError from primary session")

            assert await _run_primary() == "r03-primary-unawaited"
        else:
            handoff_raised = asyncio.Event()
            outer_holder: list[asyncio.Task] = []

            async def _handoff_fail_unawaited(sid, *, preclaimed_batch=None):
                if sid == "primary-ok":
                    rt.worker._deferred_handoffs[outer_holder[0]].enqueue("handoff-fail-unawaited", None)
                    await asyncio.wait_for(handoff_raised.wait(), timeout=2.0)
                    await asyncio.sleep(0)
                    return None
                if sid == "handoff-fail-unawaited":
                    handle_ref[0] = _capture_handle_weakref()
                    await asyncio.sleep(0)
                    handoff_raised.set()
                    raise ValueError("r03-handoff-unawaited")
                raise AssertionError(f"unexpected sid: {sid}")

            monkeypatch.setattr(rt.worker, "_run_one_session", _handoff_fail_unawaited)

            async def _run_handoff() -> None:
                outer = asyncio.create_task(rt.worker._run_session("primary-ok"))
                outer_holder.append(outer)
                try:
                    assert await outer is None
                finally:
                    outer_holder.clear()

            with caplog.at_level(logging.ERROR, logger="power_loop.agent.runtime_worker"):
                await _run_handoff()

            assert any(
                "handed-off runtime execution failed for handoff-fail-unawaited" in rec.getMessage()
                for rec in caplog.records
            )
            for rec in caplog.records:
                if rec.exc_info and rec.exc_info[2] is not None:
                    traceback.clear_frames(rec.exc_info[2])
                rec.exc_info = None
            caplog.clear()

        assert handle_ref[0] is not None
        gc.collect()
        await asyncio.sleep(0)
        assert handle_ref[0]() is None, f"{role} _FiberTaskHandle was not garbage-collected"
        unretrieved = _unretrieved_fiber_warnings()
        assert unretrieved == [], f"unexpected unretrieved _FiberTaskHandle warnings: {unretrieved!r}"
    finally:
        loop.set_exception_handler(prev_handler)
        await rt.worker.shutdown(timeout_s=2, cancel_running=True)
        await rt.client.aclose()
        await rt.transport.aclose()
        await store.close()
