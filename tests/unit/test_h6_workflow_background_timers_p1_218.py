"""Design 218 H6 regressions (P1-14 ~ P1-19): workflow, background, subagent, and timers."""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import replace

import pytest

from power_loop import (
    AgentEventBus,
    AgentEventType,
    AgentHooks,
    AgentLoopConfig,
    ExecutionEnvironment,
    HookPoint,
    InputRef,
    LLMResponse,
    LLMService,
    RegistryRuntimeFactory,
    Runtime,
    RuntimeRequestError,
    TargetRef,
    TimerContext,
    TimerRunner,
    ToolRegistry,
    TrustedActor,
    create_default_tool_registry,
)
from power_loop.agent.runtime_store import ChildSpec, RuntimeInputStore, now_ms
from power_loop.contracts.tools import ToolDefinition
from power_loop.runtime.env import RuntimeEnv
from power_loop.runtime.pricing import Free
from power_loop.runtime.representation import ProjectedRepresentation
from power_loop.runtime.stop_policy import StopPolicy
from power_loop.tools.spawn_agent import register_spawn_agent
from power_loop.workflow import WorkflowSpec
from power_loop.workflow.engine import WorkflowEngine
from power_loop.workflow.facade import register_workflow_tool, workflow_runtime_tool
from tests.runtime_fixture import RunningAgent

from .session_native_fixture import ACTOR, DESCRIPTOR, fixture, target
from .test_stateful_loop import _Scripted, _tool_resp

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


# ── P1-14: create_many partial-batch idempotency ────────────────────────────


async def test_h6_p1_14_create_many_partial_batch_idempotent_and_conflict(registry_store):
    """P1-14: partial batch collision verifies identity, inserts missing subset, rejects mismatch."""
    store, _ = registry_store
    port = RuntimeInputStore(store)
    parent = await target(port)
    drv_desc = replace(DESCRIPTOR, kind="workflow_driver")
    cm_drv = await port.create_many(
        parent=parent,
        actor=ACTOR,
        specs=[
            ChildSpec(
                session_id="drv_h6_p14",
                message_id="inp_drv_h6_p14",
                body={"run": 1},
                descriptor=drv_desc,
                result_kind="workflow_result",
                creation_identity_hash="a" * 64,
            ),
        ],
        node_id="n1",
        boot_id="b1",
        can_execute=True,
        free_slots=1,
    )
    assert len(cm_drv.preclaimed) == 1
    driver_target = cm_drv.preclaimed[0].target
    driver_actor = TrustedActor(ACTOR.tenant_id, ACTOR.principal_id, driver_target.session_id)

    spec_a = ChildSpec(
        session_id="leaf_h6_p14_a",
        message_id="inp_leaf_h6_p14_a",
        body="work A",
        descriptor=DESCRIPTOR,
        result_kind="workflow_node_result",
        creation_identity_hash="1" * 64,
    )
    spec_b = ChildSpec(
        session_id="leaf_h6_p14_b",
        message_id="inp_leaf_h6_p14_b",
        body="work B",
        descriptor=DESCRIPTOR,
        result_kind="workflow_node_result",
        creation_identity_hash="2" * 64,
    )
    spec_c = ChildSpec(
        session_id="leaf_h6_p14_c",
        message_id="inp_leaf_h6_p14_c",
        body="work C",
        descriptor=DESCRIPTOR,
        result_kind="workflow_node_result",
        creation_identity_hash="3" * 64,
    )

    # First create [A] alone.
    cm1 = await port.create_many(
        parent=driver_target,
        actor=driver_actor,
        specs=[spec_a],
        node_id="n1",
        boot_id="b1",
        can_execute=False,
        free_slots=0,
        parent_owner_epoch=1,
    )
    assert not cm1.already_created
    assert [t.session_id for t in cm1.queued_targets] == ["leaf_h6_p14_a"]

    # Partial batch [A, B] with identical identity for A: reuses A and inserts missing B.
    cm2 = await port.create_many(
        parent=driver_target,
        actor=driver_actor,
        specs=[spec_a, spec_b],
        node_id="n1",
        boot_id="b1",
        can_execute=True,
        free_slots=2,
        parent_owner_epoch=1,
    )
    assert cm2.already_created is True
    assert [b.target.session_id for b in cm2.preclaimed] == ["leaf_h6_p14_b"]
    assert cm2.queued_targets == ()

    rows = await store._db.fetchall(
        f"SELECT session_id FROM {store.t.sessions} WHERE parent_session_id=? ORDER BY session_id",
        (driver_target.session_id,),
    )
    assert [r["session_id"] for r in rows] == ["leaf_h6_p14_a", "leaf_h6_p14_b"]

    # Full replay [A, B] with identical identities returns already_created=True without re-inserting.
    cm3 = await port.create_many(
        parent=driver_target,
        actor=driver_actor,
        specs=[spec_a, spec_b],
        node_id="n1",
        boot_id="b1",
        can_execute=True,
        free_slots=2,
        parent_owner_epoch=1,
    )
    assert cm3.already_created is True
    assert cm3.preclaimed == () and cm3.queued_targets == ()

    # Conflicting creation_identity_hash on existing A must reject and must NOT insert C.
    spec_a_conflict = replace(spec_a, creation_identity_hash="9" * 64)
    with pytest.raises(RuntimeRequestError, match="creation_conflict"):
        await port.create_many(
            parent=driver_target,
            actor=driver_actor,
            specs=[spec_a_conflict, spec_c],
            node_id="n1",
            boot_id="b1",
            can_execute=False,
            free_slots=0,
            parent_owner_epoch=1,
        )
    c_row = await store._db.fetchone(
        f"SELECT session_id FROM {store.t.sessions} WHERE session_id=?",
        ("leaf_h6_p14_c",),
    )
    assert c_row is None


async def test_h6_p1_14_workflow_runner_recovers_partial_create_many_batch(registry_store, monkeypatch):
    """P1-14: runner._flush_pending_creates converges when one leaf of a parallel batch pre-exists."""
    store, _ = registry_store

    class BranchLLM(LLMService):
        async def complete(self, request, **kwargs):
            text = str(request.messages)
            if "task-a" in text:
                return LLMResponse(raw_text="done-a")
            if "task-b" in text:
                return LLMResponse(raw_text="done-b")
            return LLMResponse(raw_text="ok")

        async def close(self):
            pass

    agent = await RunningAgent(
        store, BranchLLM(), ToolRegistry(), AgentLoopConfig(context_budget_tokens=10**9), capacity=6,
    ).start()
    orig_create_many = agent.runtime.worker.inputs.create_many
    injected = False

    async def partial_precreate(*args, **kwargs):
        nonlocal injected
        specs = kwargs.get("specs") or ()
        if not injected and len(specs) == 2 and all(s.descriptor.kind == "agent" for s in specs):
            injected = True
            # Simulate a prior crash where only specs[0] had committed before journal persistence.
            await orig_create_many(
                parent=kwargs["parent"],
                actor=kwargs["actor"],
                specs=[specs[0]],
                node_id=kwargs["node_id"],
                boot_id=kwargs["boot_id"],
                ttl_s=kwargs.get("ttl_s", 30.0),
                can_execute=False,
                free_slots=0,
                parent_owner_epoch=kwargs.get("parent_owner_epoch"),
                parent_cancel_seq=kwargs.get("parent_cancel_seq"),
            )
        return await orig_create_many(*args, **kwargs)

    monkeypatch.setattr(agent.runtime.worker.inputs, "create_many", partial_precreate)
    try:
        spec = {
            "name": "partial-batch",
            "root": {
                "type": "parallel",
                "branches": [
                    {"type": "agent", "id": "a", "input": "task-a",
                     "spec": {"name": "a", "system_prompt": "a", "tools": [], "max_rounds": 2}},
                    {"type": "agent", "id": "b", "input": "task-b",
                     "spec": {"name": "b", "system_prompt": "b", "tools": [], "max_rounds": 2}},
                ],
            },
        }
        handle = await agent.runtime.client.workflow(
            agent.target, spec, message_id="wf-p14", actor=agent.actor,
        )
        result = await handle.wait(deadline_ms=now_ms() + 10000)
        assert injected is True
        assert result.status == "completed", result
        assert result.results["a"].text == "done-a"
        assert result.results["b"].text == "done-b"
    finally:
        await agent.aclose()


# ── P1-15: serial ForeachNode halt / continue semantics ─────────────────────


class _StubForeachExec:
    def __init__(self, fail_bodies: set[str]):
        self.fail_bodies = fail_bodies
        self.calls: list[tuple[str, str]] = []

    async def run_agent(self, request, *, allow_start=True):
        self.calls.append((request.node_path, request.body))
        if request.body in self.fail_bodies or request.node_path in self.fail_bodies:
            return {"status": "failed", "final_text": "", "session_id": None, "usage": {}, "error": "boom"}
        return {"status": "completed", "final_text": f"ok:{request.body}", "session_id": None, "usage": {}}

    async def todo_remaining(self, session_id: str) -> list[str]:
        return []


async def test_h6_p1_15_serial_foreach_halt_stops_remaining_items_and_downstream():
    """P1-15: Foreach(parallel=False, on_error='halt') stops on first failed item and blocks downstream."""
    ex = _StubForeachExec(fail_bodies={"item-1"})
    spec = WorkflowSpec.from_json({
        "name": "serial-halt",
        "root": {
            "type": "sequence",
            "steps": [
                {
                    "type": "foreach",
                    "id": "fe",
                    "items": ["item-1", "item-2", "item-3"],
                    "as": "x",
                    "parallel": False,
                    "on_error": "halt",
                    "body": {
                        "type": "agent",
                        "id": "work",
                        "input": "{{x}}",
                        "spec": {"name": "work", "system_prompt": "p"},
                    },
                },
                {
                    "type": "agent",
                    "id": "downstream",
                    "input": "after",
                    "spec": {"name": "downstream", "system_prompt": "p"},
                },
            ],
        },
    })
    res = await WorkflowEngine(ex, run_id="r-halt", driver_sid="drv").run(spec)
    assert res.status == "failed"
    assert ex.calls == [("work", "item-1")]
    assert "downstream" not in res.results
    assert ("fe" not in res.results) or res.results["fe"].status != "completed"


async def test_h6_p1_15_serial_foreach_continue_runs_all_items_and_downstream():
    """P1-15: Foreach(parallel=False, on_error='continue') tolerates failures and continues."""
    ex = _StubForeachExec(fail_bodies={"item-1"})
    spec = WorkflowSpec.from_json({
        "name": "serial-continue",
        "root": {
            "type": "sequence",
            "steps": [
                {
                    "type": "foreach",
                    "id": "fe",
                    "items": ["item-1", "item-2", "item-3"],
                    "as": "x",
                    "parallel": False,
                    "on_error": "continue",
                    "body": {
                        "type": "agent",
                        "id": "work",
                        "input": "{{x}}",
                        "spec": {"name": "work", "system_prompt": "p"},
                    },
                },
                {
                    "type": "agent",
                    "id": "downstream",
                    "input": "after",
                    "spec": {"name": "downstream", "system_prompt": "p"},
                },
            ],
        },
    })
    res = await WorkflowEngine(ex, run_id="r-cont", driver_sid="drv").run(spec)
    assert res.status == "completed"
    assert [body for _, body in ex.calls] == ["item-1", "item-2", "item-3", "after"]
    assert res.results["fe"].status == "completed"
    assert res.results["downstream"].status == "completed"
    assert any("fe[0].work" in err for err in res.errors)


async def test_h6_p1_15_serial_foreach_nested_sequence_halt():
    """P1-15: Foreach(parallel=False, on_error='halt') with sequence body halts on step 1 failure."""
    ex = _StubForeachExec(fail_bodies={"s1-a"})
    spec = WorkflowSpec.from_json({
        "name": "serial-seq-halt",
        "root": {
            "type": "sequence",
            "steps": [
                {
                    "type": "foreach",
                    "id": "fe",
                    "items": ["a", "b"],
                    "as": "x",
                    "parallel": False,
                    "on_error": "halt",
                    "body": {
                        "type": "sequence",
                        "steps": [
                            {"type": "agent", "id": "s1", "input": "s1-{{x}}",
                             "spec": {"name": "s1", "system_prompt": "p"}},
                            {"type": "agent", "id": "s2", "input": "s2-{{x}}",
                             "spec": {"name": "s2", "system_prompt": "p"}},
                        ],
                    },
                },
                {"type": "agent", "id": "tail", "input": "tail",
                 "spec": {"name": "tail", "system_prompt": "p"}},
            ],
        },
    })
    res = await WorkflowEngine(ex, run_id="r-seq", driver_sid="drv").run(spec)
    assert res.status == "failed"
    assert ex.calls == [("s1", "s1-a")]
    assert "tail" not in res.results


# ── P1-16: spawn_agent(send) validation & background_execution single-send ──


async def test_h6_p1_16_spawn_agent_send_rejects_self_background_and_workflow_driver(registry_store, tmp_path):
    """P1-16: spawn_agent(action='send') only targets direct 'agent' child sessions."""
    store, _ = registry_store
    bg_calls = 0

    async def slow_tool():
        nonlocal bg_calls
        bg_calls += 1
        return "bg-ok"

    registry = create_default_tool_registry(include=["background_run"], workspace_dir=str(tmp_path))
    register_spawn_agent(registry)
    registry.register(
        ToolDefinition(
            name="slow_tool", description="slow", input_schema={"type": "object"},
            pricing=Free("read"), async_capable=True, interrupt="finish",
        ),
        slow_tool,
    )

    class ScriptedMainAndChild(LLMService):
        def __init__(self):
            self.plan: list[LLMResponse] = []

        async def complete(self, request, **kwargs):
            if request.system_prompt.startswith("CHILD"):
                return LLMResponse(raw_text="child-ack")
            if self.plan:
                return self.plan.pop(0)
            return LLMResponse(raw_text="done")

        async def close(self):
            pass

    model = ScriptedMainAndChild()
    runner = await RunningAgent(
        store, model, registry, AgentLoopConfig(context_budget_tokens=10**9), capacity=6,
    ).start()
    try:
        # 1) Create one legitimate agent child and one background child.
        model.plan.extend([
            _tool_resp("mk-child", "spawn_agent", json.dumps({
                "task": "hello", "system_prompt": "CHILD", "tools": [],
            })),
            _tool_resp("mk-bg", "background_run", json.dumps({
                "action": "tool", "tool": "slow_tool", "args": {},
            })),
            LLMResponse(raw_text="created"),
        ])
        assert (await runner.send("setup")).status == "completed"
        rows = await store.load_all_messages(runner.target.session_id)
        child_sid = json.loads(next(r.content for r in rows if r.tool_call_id == "mk-child"))["session_id"]
        bg_receipt = json.loads(next(r.content for r in rows if r.tool_call_id == "mk-bg"))
        bg_sid = bg_receipt["session_id"]
        bg_res = await runner.runtime.client.wait(
            InputRef(**bg_receipt["input_ref"]), actor=runner.actor, deadline_ms=now_ms() + 10000,
        )
        assert bg_res.status == "completed" and bg_calls == 1

        # Create a workflow_driver child session directly owned by runner.target.
        wf_handle = await runner.runtime.client.workflow(
            runner.target,
            {
                "name": "wf-driver",
                "root": {"type": "agent", "id": "leaf", "input": "hi",
                         "spec": {"name": "leaf", "system_prompt": "CHILD", "tools": [], "max_rounds": 2}},
            },
            message_id="wf-p16",
            actor=runner.actor,
        )
        wf_res = await wf_handle.wait(deadline_ms=now_ms() + 10000)
        assert wf_res.status == "completed"
        wf_driver_sid = wf_handle.target.session_id
        await runner.drain()

        # 2) Attempt spawn_agent(action='send') to self, background, workflow_driver, and valid agent child.
        model.plan.extend([
            _tool_resp("send-self", "spawn_agent", json.dumps({
                "action": "send", "sub_session": runner.target.session_id, "message": "self-inject",
            })),
            _tool_resp("send-bg", "spawn_agent", json.dumps({
                "action": "send", "sub_session": bg_sid, "message": "rerun-bg",
            })),
            _tool_resp("send-wf", "spawn_agent", json.dumps({
                "action": "send", "sub_session": wf_driver_sid, "message": "inject-wf",
            })),
            _tool_resp("send-child", "spawn_agent", json.dumps({
                "action": "send", "sub_session": child_sid, "message": "follow-up",
            })),
            LLMResponse(raw_text="checked"),
        ])
        assert (await runner.send("test sends")).status == "completed"
        msgs = await store.load_all_messages(runner.target.session_id)
        by_call = {r.tool_call_id: r.content for r in msgs if r.role == "tool"}

        assert by_call["send-self"].startswith("Error:") and "forbidden" in by_call["send-self"]
        assert by_call["send-bg"].startswith("Error:") and "forbidden" in by_call["send-bg"]
        assert by_call["send-wf"].startswith("Error:") and "forbidden" in by_call["send-wf"]

        ok_receipt = json.loads(by_call["send-child"])
        assert ok_receipt["target"]["session_id"] == child_sid
        child_follow = await runner.runtime.client.wait(
            InputRef(**ok_receipt["input_ref"]), actor=runner.actor, deadline_ms=now_ms() + 10000,
        )
        assert child_follow.status == "completed" and child_follow.send_index == 2
        assert bg_calls == 1
    finally:
        await runner.aclose()


async def test_h6_p1_16_background_run_session_rejects_second_send(registry_store, tmp_path):
    """P1-16: direct RuntimeClient.send() to a background session refuses send_index != 1."""
    store, _ = registry_store
    calls = 0

    async def once_only():
        nonlocal calls
        calls += 1
        return f"call-{calls}"

    registry = create_default_tool_registry(include=["background_run"], workspace_dir=str(tmp_path))
    registry.register(
        ToolDefinition(
            name="once_only", description="once", input_schema={"type": "object"},
            pricing=Free("read"), async_capable=True, interrupt="finish",
        ),
        once_only,
    )
    model = _Scripted([
        _tool_resp("bg1", "background_run", json.dumps({"action": "tool", "tool": "once_only", "args": {}})),
        LLMResponse(raw_text="started"),
    ])
    runner = await RunningAgent(
        store, model, registry, AgentLoopConfig(context_budget_tokens=10**9), capacity=4,
    ).start()
    try:
        assert (await runner.send("go")).status == "completed"
        rows = await store.load_all_messages(runner.target.session_id)
        receipt = json.loads(next(r.content for r in rows if r.tool_call_id == "bg1"))
        first = await runner.runtime.client.wait(
            InputRef(**receipt["input_ref"]), actor=runner.actor, deadline_ms=now_ms() + 10000,
        )
        assert first.status == "completed" and first.final_text == "call-1" and calls == 1

        # Directly send a second input to the background session.
        bg_target = TargetRef(**receipt["target_ref"])
        second_receipt = await runner.runtime.client.send(
            bg_target, "replay attempt", message_id="bg-second-send", actor=runner.actor,
        )
        second = await runner.runtime.client.wait(
            second_receipt.input_ref, actor=runner.actor, deadline_ms=now_ms() + 10000,
        )
        assert second.status == "failed"
        assert second.error_code == "background_replay_forbidden"
        assert calls == 1
    finally:
        await runner.aclose()


# ── P1-17: background_execution.action batch cancel & missing task_id ───────


async def test_h6_p1_17_background_action_batch_cancel_and_missing_task_id_guard(registry_store, tmp_path):
    """P1-17: cancel without task_id errors without cancelling tasks; batch cancel cancels all ids."""
    store, _ = registry_store
    started: dict[str, asyncio.Event] = {"a": asyncio.Event(), "b": asyncio.Event()}
    release = asyncio.Event()

    async def hold(tag: str = "a"):
        started[tag].set()
        await release.wait()
        return f"finished-{tag}"

    registry = create_default_tool_registry(include=["background_run"], workspace_dir=str(tmp_path))
    registry.register(
        ToolDefinition(
            name="hold",
            description="block until cancelled",
            input_schema={"type": "object", "properties": {"tag": {"type": "string"}}},
            pricing=Free("local"),
            async_capable=True,
            interrupt="abort",
        ),
        hold,
    )

    class ScriptedController(LLMService):
        def __init__(self):
            self.plan: list[LLMResponse] = []

        async def complete(self, request, **kwargs):
            return self.plan.pop(0) if self.plan else LLMResponse(raw_text="done")

        async def close(self):
            pass

    model = ScriptedController()
    runner = await RunningAgent(
        store,
        model,
        registry,
        AgentLoopConfig(context_budget_tokens=10**9, stop_policy=StopPolicy(abort_tool_s=0.01)),
        capacity=6,
    ).start()
    try:
        model.plan.extend([
            _tool_resp("bg-a", "background_run", json.dumps({"action": "tool", "tool": "hold", "args": {"tag": "a"}})),
            _tool_resp("bg-b", "background_run", json.dumps({"action": "tool", "tool": "hold", "args": {"tag": "b"}})),
            LLMResponse(raw_text="launched"),
        ])
        assert (await runner.send("launch two")).status == "completed"
        await asyncio.wait_for(asyncio.gather(started["a"].wait(), started["b"].wait()), 5)

        msgs = await store.load_all_messages(runner.target.session_id)
        rec_a = json.loads(next(m.content for m in msgs if m.tool_call_id == "bg-a"))
        rec_b = json.loads(next(m.content for m in msgs if m.tool_call_id == "bg-b"))
        tid_a, tid_b = rec_a["task_id"], rec_b["task_id"]

        # Calling cancel / check / wait without task_id must fail with task_id is required,
        # and must NOT cancel running tasks; list without task_id must still list both tasks.
        model.plan.extend([
            _tool_resp("cancel-empty", "background_run", json.dumps({"action": "cancel"})),
            _tool_resp("check-empty", "background_run", json.dumps({"action": "check"})),
            _tool_resp("list-all", "background_run", json.dumps({"action": "list"})),
            LLMResponse(raw_text="checked-empty"),
        ])
        assert (await runner.send("guard check")).status == "completed"
        msgs = await store.load_all_messages(runner.target.session_id)
        by_call = {m.tool_call_id: m.content for m in msgs if m.role == "tool"}

        assert "task_id is required" in by_call["cancel-empty"]
        assert "task_id is required" in by_call["check-empty"]
        listed = json.loads(by_call["list-all"].split("\n", 1)[1])["tasks"]
        assert {item["task_id"] for item in listed} == {tid_a, tid_b}

        # Verify both background tasks are still running (not killed by cancel-empty).
        for rec in (rec_a, rec_b):
            st = await runner.runtime.client.inspect(InputRef(**rec["input_ref"]), actor=runner.actor)
            assert st.terminal_status is None

        # Batch cancel both task_ids in one call.
        model.plan.extend([
            _tool_resp("cancel-both", "background_run", json.dumps({
                "action": "cancel", "task_ids": [tid_a, tid_b],
            })),
            LLMResponse(raw_text="cancelled-both"),
        ])
        assert (await runner.send("cancel both")).status == "completed"
        msgs = await store.load_all_messages(runner.target.session_id)
        cancel_out = json.loads(next(m.content for m in msgs if m.tool_call_id == "cancel-both"))
        assert len(cancel_out["cancelled"]) == 2

        res_a = await runner.runtime.client.wait(
            InputRef(**rec_a["input_ref"]), actor=runner.actor, deadline_ms=now_ms() + 10000,
        )
        res_b = await runner.runtime.client.wait(
            InputRef(**rec_b["input_ref"]), actor=runner.actor, deadline_ms=now_ms() + 10000,
        )
        assert res_a.status == "cancelled" and res_b.status == "cancelled"
    finally:
        release.set()
        await runner.aclose()


# ── P1-18: workflow logs pagination, parameter validation, control_only ─────


async def test_h6_p1_18_workflow_logs_projection_pagination_and_validation(registry_store, tmp_path):
    """P1-18: interleaved multi-session projections paginate cleanly; bad inputs yield structured errors."""
    store, _ = registry_store
    model = _Scripted([LLMResponse(raw_text="LEAF-OK")])
    registry = ToolRegistry()
    register_workflow_tool(registry)
    agent = await RunningAgent(
        store,
        model,
        registry,
        AgentLoopConfig(context_budget_tokens=10**9, representation=ProjectedRepresentation()),
        runtime_env=RuntimeEnv(workspace_dir=tmp_path),
    ).start()
    try:
        handle = await agent.runtime.client.workflow(
            agent.target,
            {
                "name": "logs-p18",
                "root": {
                    "type": "agent", "id": "leaf", "input": "work",
                    "spec": {"name": "leaf", "system_prompt": "ok", "tools": [], "max_rounds": 2},
                },
            },
            message_id="wf-p18",
            actor=agent.actor,
        )
        result = await handle.wait(deadline_ms=now_ms() + 10000)
        assert result.status == "completed"
        await agent.drain()
        leaf_sid = result.results["leaf"].session_id

        # Create a sibling session and interleave projection writes so leaf_sid's auto-increment IDs have wide gaps.
        other_target = await agent.runtime.open_session(DESCRIPTOR, actor=agent.actor)
        other_sid = other_target.session_id
        for send_idx in (2, 3, 4):
            # Write several rows to other_sid between each leaf_sid projection row.
            for noise_idx in range(1, 5):
                noise_send = send_idx * 10 + noise_idx
                await store.write_send_projection_rows(
                    other_sid,
                    send_index=noise_send,
                    rows=[("user", f"noise-{noise_send}", f"noise-{noise_send}")],
                    source_seq_lo=1,
                    source_seq_hi=1,
                    projector_version=1,
                    load_snapshot=False,
                )
            await store.write_send_projection_rows(
                leaf_sid,
                send_index=send_idx,
                rows=[("project", {"marker": f"leaf-send-{send_idx}"}, f"leaf-send-{send_idx}")],
                source_seq_lo=1,
                source_seq_hi=2,
                projector_version=1,
                load_snapshot=False,
            )

        all_proj = await store._db.fetchall(
            f"SELECT id, send_index FROM {store.t.project_messages} WHERE session_id=? ORDER BY id ASC",
            (leaf_sid,),
        )
        assert len(all_proj) >= 4
        # Confirm wide gap between last two projection IDs (where MAX(id) - 2 would miss the second-to-last row).
        assert int(all_proj[-1]["id"]) - int(all_proj[-2]["id"]) > 2
        expected_latest_two = [str(all_proj[-2]["id"]), str(all_proj[-1]["id"])]

        base = {"action": "logs", "run_id": handle.run_id, "node": "leaf"}
        cases = [
            ("latest-proj", {**base, "source": "projection", "limit": 2}),
            ("page1-proj", {**base, "source": "projection", "cursor": "0", "limit": 2}),
            ("latest-msg", {**base, "source": "messages", "limit": 2}),
            ("empty-mids", {**base, "message_ids": []}),
            ("proj-seq", {**base, "source": "projection", "seq": "1"}),
            ("bad-cursor", {**base, "cursor": "not-a-number"}),
            ("bad-limit", {**base, "limit": "nope"}),
            ("bool-limit", {**base, "limit": True}),
            ("bad-seq", {**base, "source": "messages", "seq": "xyz"}),
            ("bad-mid-item", {**base, "message_ids": ["1", "bad"]}),
            ("empty-output", {**base, "output_file": ""}),
        ]
        tool_responses = [_tool_resp(cid, "workflow", json.dumps(opts)) for cid, opts in cases]
        model.responses.extend([*tool_responses, LLMResponse(raw_text="all-read")])
        model._idx = len(model.responses) - len(tool_responses) - 1
        reply = await agent.send("query logs")
        assert reply.status == "completed"

        msgs = await store.load_all_messages(agent.target.session_id)
        by_call = {r.tool_call_id: json.loads(r.content) for r in msgs if r.role == "tool" and r.name == "workflow"}

        latest_proj = by_call["latest-proj"]
        assert latest_proj["status"] == "ok"
        assert [r["id"] for r in latest_proj["records"]] == expected_latest_two
        assert latest_proj["next_cursor"] is None

        page1 = by_call["page1-proj"]
        assert page1["status"] == "ok"
        assert [r["id"] for r in page1["records"]] == [str(all_proj[0]["id"]), str(all_proj[1]["id"])]
        assert page1["next_cursor"] == str(all_proj[1]["id"])

        latest_msg = by_call["latest-msg"]
        assert latest_msg["status"] == "ok" and len(latest_msg["records"]) == 2

        for bad_call in ("empty-mids", "proj-seq", "bad-cursor", "bad-limit", "bool-limit", "bad-seq", "bad-mid-item", "empty-output"):
            assert "error" in by_call[bad_call], (bad_call, by_call[bad_call])
            assert by_call[bad_call]["error"]["code"]
    finally:
        await agent.aclose()


async def test_h6_p1_18_workflow_logs_output_file_not_control_only():
    """P1-18: workflow logs with output_file writes to disk and must not be marked control_only."""
    async def noop(**kwargs):
        return "{}"

    rt_tool = workflow_runtime_tool(noop)
    policy_read = rt_tool.dispatch_policy({"action": "logs", "run_id": "inp_" + "0" * 56})
    policy_export = rt_tool.dispatch_policy({
        "action": "logs", "run_id": "inp_" + "0" * 56, "output_file": "audit/out.jsonl",
    })
    assert policy_read.native_admitted is True and policy_read.control_only is True
    assert policy_export.native_admitted is True and policy_export.control_only is False


# ── P1-19: Timer operation_id idempotency, cancel before/after send, pruning ─


async def _timer_runtime(store):
    _, tgt, actor = await fixture(store)
    factory = RegistryRuntimeFactory()

    async def build(descriptor, context):
        return ExecutionEnvironment(
            _Scripted([LLMResponse(raw_text="woke")]),
            create_default_tool_registry(preset="core", bind=False),
            AgentLoopConfig(),
        )

    factory.register("test.agent@215", build)
    bus = AgentEventBus()
    runtime = Runtime(store=store, factory=factory, node_id="timers-h6", timers_enabled=False)
    await runtime.worker.start()
    return runtime, tgt, actor, bus


async def test_h6_p1_19_timer_create_operation_id_idempotent(registry_store):
    """P1-19: StoreTimerBackend.create persists operation_id and is idempotent even at the 10-timer cap."""
    store, _ = registry_store
    runtime, tgt, actor, _ = await _timer_runtime(store)
    try:
        from power_loop import StoreTimerBackend

        backend = StoreTimerBackend(store)
        ctx1 = TimerContext(store.store_id, tgt.session_id, actor, operation_id="op-h6-1")
        t1 = await backend.create(ctx1, delay_s=60, note="first")
        t1_retry = await backend.create(ctx1, delay_s=120, note="first")
        assert t1_retry.timer_id == t1.timer_id
        assert t1_retry.due_at == t1.due_at
        assert len(await backend.list(ctx1)) == 1

        # Fill up to the 10-timer limit with distinct operation_ids.
        for idx in range(2, 11):
            ctx_n = TimerContext(store.store_id, tgt.session_id, actor, operation_id=f"op-h6-{idx}")
            await backend.create(ctx_n, delay_s=60, note=f"n-{idx}")
        assert len(await backend.list(ctx1)) == 10

        # Retrying an already-created operation_id still succeeds idempotently at the limit.
        t1_at_limit = await backend.create(ctx1, delay_s=60, note="first")
        assert t1_at_limit.timer_id == t1.timer_id

        # A genuinely new operation_id is still rejected by timer_limit_reached.
        ctx11 = TimerContext(store.store_id, tgt.session_id, actor, operation_id="op-h6-11")
        with pytest.raises(RuntimeRequestError, match="timer_limit_reached"):
            await backend.create(ctx11, delay_s=60, note="overflow")
    finally:
        await runtime.worker.shutdown()
        await runtime.client.aclose()
        await runtime.transport.aclose()


async def test_h6_p1_19_timer_cancel_before_send_keeps_armed_and_after_send_marks_unknown(registry_store, monkeypatch):
    """P1-19: cancellation before client.send restores armed; after client.send marks unknown."""
    store, _ = registry_store
    runtime, tgt, actor, bus = await _timer_runtime(store)
    events = []
    bus.subscribe(AgentEventType.TIMER_FIRED, lambda e: events.append(e))
    try:
        from power_loop import StoreTimerBackend

        backend = StoreTimerBackend(store)
        source = TimerContext(store.store_id, tgt.session_id, actor)

        # Case A: Cancel inside TIMER_FIRE hook (BEFORE client.send).
        hook_entered = asyncio.Event()
        hook_release = asyncio.Event()
        block_hook = True
        hooks = AgentHooks()

        async def gating_hook(ctx):
            if block_hook:
                hook_entered.set()
                await hook_release.wait()

        hooks.register(HookPoint.TIMER_FIRE, gating_hook)
        runner = TimerRunner(store=store, client=runtime.client, hooks=hooks, event_bus=bus)
        send_calls = 0
        orig_send = runtime.client.send

        async def counting_send(*args, **kwargs):
            nonlocal send_calls
            send_calls += 1
            return await orig_send(*args, **kwargs)

        monkeypatch.setattr(runtime.client, "send", counting_send)

        timer_a = await backend.create(source, due_at_ms=1, note="before-send-cancel")
        scan_task = asyncio.create_task(runner.scan_once())
        await asyncio.wait_for(hook_entered.wait(), 5)
        scan_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await scan_task

        assert send_calls == 0
        row_a = await store.get_timer(tgt.session_id, timer_a.timer_id)
        assert row_a is not None and row_a.status == "armed" and row_a.fire_count == 0
        assert events == []

        # Re-running scan_once after unblocking fires the restored timer normally.
        block_hook = False
        assert await runner.scan_once() == 1
        assert send_calls == 1
        row_a2 = await store.get_timer(tgt.session_id, timer_a.timer_id)
        assert row_a2 is not None and row_a2.status == "fired" and row_a2.fire_count == 1

        # Case B: Cancel AFTER client.send has been initiated -> must mark 'unknown'.
        send_entered = asyncio.Event()
        send_hold = asyncio.Event()

        async def hanging_send(*args, **kwargs):
            send_entered.set()
            await send_hold.wait()
            return await orig_send(*args, **kwargs)

        monkeypatch.setattr(runtime.client, "send", hanging_send)
        timer_b = await backend.create(source, due_at_ms=1, note="after-send-cancel")
        scan_task_b = asyncio.create_task(runner.scan_once())
        await asyncio.wait_for(send_entered.wait(), 5)
        scan_task_b.cancel()
        with pytest.raises(asyncio.CancelledError):
            await scan_task_b

        row_b = await store.get_timer(tgt.session_id, timer_b.timer_id)
        assert row_b is not None and row_b.status == "unknown" and row_b.fire_count == 1
        assert any(e.data.timer_id == timer_b.timer_id and e.data.outcome == "unknown" for e in events)
    finally:
        await runtime.worker.shutdown()
        await runtime.client.aclose()
        await runtime.transport.aclose()


async def test_h6_p1_19_timer_runner_periodically_prunes_terminal_timers(registry_store):
    """P1-19: TimerRunner periodically prunes terminal timers while preserving armed timers and non-timer state."""
    store, _ = registry_store
    runtime, tgt, actor, _ = await _timer_runtime(store)
    try:
        from power_loop import StoreTimerBackend

        backend = StoreTimerBackend(store)
        source = TimerContext(store.store_id, tgt.session_id, actor)

        # Store non-timer runtime state alongside timers to verify due_timers / prune_timers ignore it cleanly.
        await store.set_runtime_state(tgt.session_id, "custom_blob", {"arbitrary": [1, 2, 3]})

        fired_timer = await backend.create(source, due_at_ms=1, note="old-fired")
        cancelled_timer = await backend.create(source, delay_s=3600, note="old-cancelled")
        armed_timer = await backend.create(source, delay_s=3600, note="keep-armed")

        assert await backend.cancel(source, cancelled_timer.timer_id)
        await store.transition_timer(tgt.session_id, fired_timer.timer_id, from_status="armed", to_status="firing")
        assert await store.finish_firing_timer(tgt.session_id, fired_timer.timer_id)

        # Backdate terminal timer rows to simulate expired retention.
        old_ts = int(time.time() * 1000) - 100_000
        for tid in (fired_timer.timer_id, cancelled_timer.timer_id):
            await store._db.execute(
                f"UPDATE {store.t.session_runtime_state} SET updated_at=? WHERE session_id=? AND state_key=?",
                (old_ts, tgt.session_id, f"timer:{tid}"),
            )

        runner = TimerRunner(
            store=store,
            client=runtime.client,
            prune_interval_s=0.01,
            prune_older_than_ms=10_000,
        )
        assert await runner.scan_once() == 0

        assert await store.get_timer(tgt.session_id, fired_timer.timer_id) is None
        assert await store.get_timer(tgt.session_id, cancelled_timer.timer_id) is None
        kept = await store.get_timer(tgt.session_id, armed_timer.timer_id)
        assert kept is not None and kept.status == "armed"
        assert await store.get_runtime_state(tgt.session_id, "custom_blob") == {"arbitrary": [1, 2, 3]}
    finally:
        await runtime.worker.shutdown()
        await runtime.client.aclose()
        await runtime.transport.aclose()


# ── H6 Rework Regressions (Reviewer Counterexamples) ────────────────────────


async def test_h6_p1_14_create_many_five_item_interleaved_progress_completes_e(registry_store, monkeypatch):
    """P1-14 rework: 5 identical-identity items (A-E) with interleaved B/C/D collisions still create E."""
    store, _ = registry_store
    port = RuntimeInputStore(store)
    parent = await target(port)
    drv_desc = replace(DESCRIPTOR, kind="workflow_driver")
    cm_drv = await port.create_many(
        parent=parent,
        actor=ACTOR,
        specs=[
            ChildSpec(
                session_id="drv_h6_p14_5",
                message_id="inp_drv_h6_p14_5",
                body={"run": 1},
                descriptor=drv_desc,
                result_kind="workflow_result",
                creation_identity_hash="d" * 64,
            ),
        ],
        node_id="n1",
        boot_id="b1",
        can_execute=True,
        free_slots=1,
    )
    driver_target = cm_drv.preclaimed[0].target
    driver_actor = TrustedActor(ACTOR.tenant_id, ACTOR.principal_id, driver_target.session_id)

    specs = [
        ChildSpec(
            session_id=f"leaf_h6_p14_5_{name}",
            message_id=f"inp_leaf_h6_p14_5_{name}",
            body=f"work {name}",
            descriptor=DESCRIPTOR,
            result_kind="workflow_node_result",
            creation_identity_hash=ch * 64,
        )
        for name, ch in (("a", "1"), ("b", "2"), ("c", "3"), ("d", "4"), ("e", "5"))
    ]

    # Pre-create A before the 5-item batch starts.
    await port.create_many(
        parent=driver_target,
        actor=driver_actor,
        specs=[specs[0]],
        node_id="n1",
        boot_id="b1",
        can_execute=False,
        free_slots=0,
        parent_owner_epoch=1,
    )

    # After each duplicate-key slow-path lookup, commit the next legal item (B, then C, then D)
    # so the batch collides 4 times while strictly shrinking missing: 5 -> 4 -> 3 -> 2 -> 1 (E).
    to_interleave = [specs[1], specs[2], specs[3]]
    orig_fetchall = port.db.fetchall
    in_interleave = False

    async def interleaved_fetchall(sql, params=()):
        nonlocal in_interleave
        rows = await orig_fetchall(sql, params)
        if not in_interleave and to_interleave and "creation_identity_hash" in sql:
            nxt = to_interleave.pop(0)
            in_interleave = True
            try:
                await port.create_many(
                    parent=driver_target,
                    actor=driver_actor,
                    specs=[nxt],
                    node_id="n1",
                    boot_id="b1",
                    can_execute=False,
                    free_slots=0,
                    parent_owner_epoch=1,
                )
            finally:
                in_interleave = False
        return rows

    monkeypatch.setattr(port.db, "fetchall", interleaved_fetchall)
    cm = await port.create_many(
        parent=driver_target,
        actor=driver_actor,
        specs=specs,
        node_id="n1",
        boot_id="b1",
        can_execute=True,
        free_slots=2,
        parent_owner_epoch=1,
    )
    assert not to_interleave
    assert cm.already_created is True
    assert [b.target.session_id for b in cm.preclaimed] == ["leaf_h6_p14_5_e"]
    assert cm.queued_targets == ()

    rows = await store._db.fetchall(
        f"SELECT session_id FROM {store.t.sessions} WHERE parent_session_id=? ORDER BY session_id",
        (driver_target.session_id,),
    )
    assert [r["session_id"] for r in rows] == [sp.session_id for sp in specs]


async def test_h6_p1_15_parallel_continue_sibling_serial_foreach_halt_isolation():
    """P1-15 rework: Parallel(continue) containing failing Foreach(halt) completes sibling, downstream, and run."""
    good_started = asyncio.Event()
    bad_halted = asyncio.Event()
    calls: list[tuple[str, str]] = []

    class CoExec:
        async def run_agent(self, request, *, allow_start=True):
            calls.append((request.node_path, request.body))
            if request.body == "bad-1":
                await asyncio.wait_for(good_started.wait(), 5)
                return {"status": "failed", "final_text": "", "session_id": None, "usage": {}, "error": "bad boom"}
            if request.body == "good-1":
                good_started.set()
                await asyncio.wait_for(bad_halted.wait(), 5)
            return {"status": "completed", "final_text": f"ok:{request.body}", "session_id": None, "usage": {}}

        async def todo_remaining(self, session_id: str) -> list[str]:
            return []

    spec = WorkflowSpec.from_json({
        "name": "parallel-continue-foreach-isolation",
        "root": {
            "type": "sequence",
            "steps": [
                {
                    "type": "parallel",
                    "on_error": "continue",
                    "branches": [
                        {
                            "type": "foreach",
                            "id": "bad",
                            "items": ["bad-1", "bad-2"],
                            "as": "x",
                            "parallel": False,
                            "on_error": "halt",
                            "body": {
                                "type": "agent",
                                "id": "err",
                                "input": "{{x}}",
                                "spec": {"name": "err", "system_prompt": "p"},
                            },
                        },
                        {
                            "type": "foreach",
                            "id": "good",
                            "items": ["good-1", "good-2"],
                            "as": "y",
                            "parallel": False,
                            "on_error": "halt",
                            "body": {
                                "type": "agent",
                                "id": "ok_node",
                                "input": "{{y}}",
                                "spec": {"name": "ok_node", "system_prompt": "p"},
                            },
                        },
                    ],
                },
                {
                    "type": "agent",
                    "id": "tail",
                    "input": "after-parallel",
                    "spec": {"name": "tail", "system_prompt": "p"},
                },
            ],
        },
    })
    eng = WorkflowEngine(CoExec(), run_id="r-iso", driver_sid="drv")
    orig_exec_foreach = eng._exec_foreach

    async def observed_exec_foreach(node, env, driver_sid):
        try:
            return await orig_exec_foreach(node, env, driver_sid)
        finally:
            if node.id == "bad":
                bad_halted.set()

    eng._exec_foreach = observed_exec_foreach  # type: ignore[method-assign]
    res = await eng.run(spec)
    assert res.status == "completed"
    assert ("err", "bad-1") in calls
    assert ("err", "bad-2") not in calls
    assert ("ok_node", "good-1") in calls
    assert ("ok_node", "good-2") in calls
    assert ("tail", "after-parallel") in calls
    assert res.results["good"].status == "completed"
    assert res.results["good"].payload == {"items": ["ok:good-1", "ok:good-2"]}
    assert res.results["tail"].status == "completed"
    assert any("err" in err or "bad" in err for err in res.errors)


async def test_h6_p1_15_nested_foreach_inner_continue_isolates_iterations():
    """P1-15 rework 2: outer foreach re-entering inner foreach(continue) must skip step2 and record every failure."""
    ex = _StubForeachExec(fail_bodies={"s1-a", "s1-b"})
    spec = WorkflowSpec.from_json({
        "name": "nested-foreach-continue-isolation",
        "root": {
            "type": "foreach",
            "id": "outer",
            "items": ["a", "b"],
            "as": "x",
            "parallel": False,
            "on_error": "halt",
            "body": {
                "type": "foreach",
                "id": "inner",
                "items": ["once"],
                "as": "y",
                "parallel": False,
                "on_error": "continue",
                "body": {
                    "type": "sequence",
                    "steps": [
                        {"type": "agent", "id": "step1", "input": "s1-{{x}}",
                         "spec": {"name": "step1", "system_prompt": "p"}},
                        {"type": "agent", "id": "step2", "input": "s2-{{x}}",
                         "spec": {"name": "step2", "system_prompt": "p"}},
                    ],
                },
            },
        },
    })
    res = await WorkflowEngine(ex, run_id="r-nested", driver_sid="drv").run(spec)
    assert res.status == "completed"
    assert ex.calls == [("step1", "s1-a"), ("step1", "s1-b")]
    step1_errors = [err for err in res.errors if "step1" in err]
    assert len(step1_errors) == 2, res.errors


async def test_h6_p1_15_parallel_continue_leaf_with_continue_on_error_records_error_and_not_ok():
    """P1-15 rework 3: Parallel(continue) still records failure and ok=False for continue_on_error leaf."""
    ex = _StubForeachExec(fail_bodies={"bad-body"})
    spec = WorkflowSpec.from_json({
        "name": "parallel-continue-leaf-coe",
        "root": {
            "type": "sequence",
            "steps": [
                {
                    "type": "parallel",
                    "on_error": "continue",
                    "branches": [
                        {
                            "type": "agent",
                            "id": "bad",
                            "input": "bad-body",
                            "continue_on_error": True,
                            "spec": {"name": "bad", "system_prompt": "p"},
                        },
                        {
                            "type": "agent",
                            "id": "good",
                            "input": "good-body",
                            "spec": {"name": "good", "system_prompt": "p"},
                        },
                    ],
                },
                {
                    "type": "agent",
                    "id": "tail",
                    "input": "tail-body",
                    "spec": {"name": "tail", "system_prompt": "p"},
                },
            ],
        },
    })
    res = await WorkflowEngine(ex, run_id="r-pcoe", driver_sid="drv").run(spec)
    assert res.status == "completed"
    assert res.ok is False
    assert "parallel branch bad: failed" in res.errors
    assert res.results["bad"].status == "failed"
    assert res.results["good"].status == "completed"
    assert res.results["tail"].status == "completed"


async def test_h6_p1_15_serial_foreach_continue_sequence_skips_step2_every_item():
    """P1-15 rework: Foreach(parallel=False, on_error='continue') skips step2 on every failed step1 and records all errors."""
    ex = _StubForeachExec(fail_bodies={"s1-a", "s1-b"})
    spec = WorkflowSpec.from_json({
        "name": "serial-seq-continue",
        "root": {
            "type": "sequence",
            "steps": [
                {
                    "type": "foreach",
                    "id": "fe",
                    "items": ["a", "b"],
                    "as": "x",
                    "parallel": False,
                    "on_error": "continue",
                    "body": {
                        "type": "sequence",
                        "steps": [
                            {"type": "agent", "id": "step1", "input": "s1-{{x}}",
                             "spec": {"name": "step1", "system_prompt": "p"}},
                            {"type": "agent", "id": "step2", "input": "s2-{{x}}",
                             "spec": {"name": "step2", "system_prompt": "p"}},
                        ],
                    },
                },
                {
                    "type": "agent",
                    "id": "downstream",
                    "input": "after",
                    "spec": {"name": "downstream", "system_prompt": "p"},
                },
            ],
        },
    })
    res = await WorkflowEngine(ex, run_id="r-seq-cont", driver_sid="drv").run(spec)
    assert res.status == "completed"
    assert ex.calls == [("step1", "s1-a"), ("step1", "s1-b"), ("downstream", "after")]
    assert "fe[0].step1" in res.errors
    assert "fe[1].step1" in res.errors
    assert res.results["downstream"].status == "completed"


async def test_h6_p1_18_workflow_logs_rejects_int64_overflow_and_array_source(registry_store, tmp_path):
    """P1-18 rework: int64-overflow coordinates and unhashable array source return structured input errors."""
    store, _ = registry_store
    model = _Scripted([LLMResponse(raw_text="LEAF-OK")])
    registry = ToolRegistry()
    register_workflow_tool(registry)
    agent = await RunningAgent(
        store,
        model,
        registry,
        AgentLoopConfig(context_budget_tokens=10**9, representation=ProjectedRepresentation()),
        runtime_env=RuntimeEnv(workspace_dir=tmp_path),
    ).start()
    try:
        handle = await agent.runtime.client.workflow(
            agent.target,
            {
                "name": "logs-overflow",
                "root": {
                    "type": "agent", "id": "leaf", "input": "work",
                    "spec": {"name": "leaf", "system_prompt": "ok", "tools": [], "max_rounds": 2},
                },
            },
            message_id="wf-p18-ovf",
            actor=agent.actor,
        )
        result = await handle.wait(deadline_ms=now_ms() + 10000)
        assert result.status == "completed"
        await agent.drain()

        overflow = str((1 << 63))  # 9223372036854775808 (> signed int64 max)
        base = {"action": "logs", "run_id": handle.run_id, "node": "leaf"}
        cases = [
            ("ovf-cursor", {**base, "cursor": overflow}),
            ("ovf-seq", {**base, "seq": overflow}),
            ("ovf-id", {**base, "id": overflow}),
            ("ovf-mid", {**base, "message_ids": [overflow]}),
            ("arr-source", {**base, "source": ["messages"]}),
            ("obj-source", {**base, "source": {"k": "messages"}}),
        ]
        tool_responses = [_tool_resp(cid, "workflow", json.dumps(opts)) for cid, opts in cases]
        model.responses.extend([*tool_responses, LLMResponse(raw_text="checked")])
        model._idx = len(model.responses) - len(tool_responses) - 1
        reply = await agent.send("check overflow and array source")
        assert reply.status == "completed"

        msgs = await store.load_all_messages(agent.target.session_id)
        by_call = {r.tool_call_id: json.loads(r.content) for r in msgs if r.role == "tool" and r.name == "workflow"}
        for cid, _ in cases:
            assert "error" in by_call[cid], (cid, by_call[cid])
            assert by_call[cid]["error"]["code"], (cid, by_call[cid])
    finally:
        await agent.aclose()


async def test_h6_p1_18_workflow_list_and_status_support_limit_100(registry_store, tmp_path):
    """P1-18 rework: workflow list and run_id-less status both return all 100 runs when limit=100."""
    from power_loop.workflow import journal

    store, _ = registry_store
    model = _Scripted([])
    registry = ToolRegistry()
    register_workflow_tool(registry)
    agent = await RunningAgent(
        store,
        model,
        registry,
        AgentLoopConfig(context_budget_tokens=10**9),
        runtime_env=RuntimeEnv(workspace_dir=tmp_path),
    ).start()
    try:
        port = agent.runtime.worker.inputs
        drv_desc = replace(DESCRIPTOR, kind="workflow_driver")
        specs = [
            ChildSpec(
                session_id=f"drv_list100_{idx:03d}",
                message_id=f"inp_{idx:056x}",
                body={"run": idx},
                descriptor=drv_desc,
                result_kind="workflow_result",
                creation_identity_hash=f"{idx:064x}",
            )
            for idx in range(100)
        ]
        await port.create_many(
            parent=agent.target,
            actor=agent.actor,
            specs=specs,
            node_id="n1",
            boot_id="b1",
            can_execute=False,
            free_slots=0,
        )
        async with store._db.transaction() as tx:
            for idx, sp in enumerate(specs):
                await journal.put(
                    store,
                    tx,
                    sp.session_id,
                    journal.run_key(sp.message_id),
                    {"run_id": sp.message_id, "name": f"wf-{idx}", "status": "completed", "nodes": []},
                )

        model.responses.extend([
            _tool_resp("list-100", "workflow", json.dumps({"action": "list", "limit": 100})),
            _tool_resp("status-100", "workflow", json.dumps({"action": "status", "limit": 100})),
            LLMResponse(raw_text="listed-100"),
        ])
        reply = await agent.send("list 100 runs")
        assert reply.status == "completed"

        msgs = await store.load_all_messages(agent.target.session_id)
        by_call = {r.tool_call_id: json.loads(r.content) for r in msgs if r.role == "tool" and r.name == "workflow"}
        for cid in ("list-100", "status-100"):
            assert len(by_call[cid]["runs"]) == 100, (cid, len(by_call[cid]["runs"]))
            assert by_call[cid]["next_cursor"] is None, (cid, by_call[cid]["next_cursor"])
    finally:
        await agent.aclose()


async def test_h6_p1_19_timer_cancel_before_send_releases_claim_once(registry_store, monkeypatch):
    """P1-19 rework: cancelling worker 1 before send releases claim once and cannot clobber worker 2's firing claim."""
    store, _ = registry_store
    runtime, tgt, actor, _ = await _timer_runtime(store)
    try:
        from power_loop import StoreTimerBackend

        backend = StoreTimerBackend(store)
        source = TimerContext(store.store_id, tgt.session_id, actor)
        timer = await backend.create(source, due_at_ms=1, note="single-release")

        r1_in_hook = asyncio.Event()
        r1_hold = asyncio.Event()
        r2_in_hook = asyncio.Event()
        r2_release = asyncio.Event()
        r1_after_release = asyncio.Event()
        r1_finish_cleanup = asyncio.Event()

        hooks1 = AgentHooks()

        async def hook1(ctx):
            r1_in_hook.set()
            await r1_hold.wait()

        hooks1.register(HookPoint.TIMER_FIRE, hook1)

        hooks2 = AgentHooks()

        async def hook2(ctx):
            r2_in_hook.set()
            await r2_release.wait()

        hooks2.register(HookPoint.TIMER_FIRE, hook2)

        runner1 = TimerRunner(store=store, client=runtime.client, hooks=hooks1)
        runner2 = TimerRunner(store=store, client=runtime.client, hooks=hooks2)

        orig_transition = store.transition_timer
        release_calls = 0

        async def observed_transition(session_id, timer_id, *, from_status, to_status, **kwargs):
            nonlocal release_calls
            res = await orig_transition(
                session_id, timer_id, from_status=from_status, to_status=to_status, **kwargs,
            )
            if from_status == "firing" and to_status == "armed":
                release_calls += 1
                if release_calls == 1:
                    r1_after_release.set()
                    await r1_finish_cleanup.wait()
            return res

        monkeypatch.setattr(store, "transition_timer", observed_transition)

        send_calls = 0
        orig_send = runtime.client.send

        async def counting_send(*args, **kwargs):
            nonlocal send_calls
            send_calls += 1
            return await orig_send(*args, **kwargs)

        monkeypatch.setattr(runtime.client, "send", counting_send)

        task1 = asyncio.create_task(runner1.scan_once())
        await asyncio.wait_for(r1_in_hook.wait(), 5)
        task1.cancel()
        await asyncio.wait_for(r1_after_release.wait(), 5)

        # Worker 2 now claims the restored armed timer and enters its hook (status='firing').
        task2 = asyncio.create_task(runner2.scan_once())
        await asyncio.wait_for(r2_in_hook.wait(), 5)

        # Resume worker 1's outer cleanup; it MUST NOT issue a second firing->armed transition.
        r1_finish_cleanup.set()
        with pytest.raises(asyncio.CancelledError):
            await task1

        mid_row = await store.get_timer(tgt.session_id, timer.timer_id)
        assert mid_row is not None and mid_row.status == "firing"

        # Let worker 2 finish delivery and settle cleanly to 'fired'.
        r2_release.set()
        assert await task2 == 1
        assert release_calls == 1
        assert send_calls == 1
        final_row = await store.get_timer(tgt.session_id, timer.timer_id)
        assert final_row is not None and final_row.status == "fired" and final_row.fire_count == 1
    finally:
        await runtime.worker.shutdown()
        await runtime.client.aclose()
        await runtime.transport.aclose()


async def test_h6_p1_19_timer_timeout_before_send_keeps_armed_and_can_refire(registry_store, monkeypatch):
    """P1-19 rework: acceptance_timeout in TIMER_FIRE before send keeps timer armed with fire_count=0."""
    store, _ = registry_store
    runtime, tgt, actor, bus = await _timer_runtime(store)
    events = []
    bus.subscribe(AgentEventType.TIMER_FIRED, lambda e: events.append(e))
    try:
        from power_loop import StoreTimerBackend

        backend = StoreTimerBackend(store)
        source = TimerContext(store.store_id, tgt.session_id, actor)
        timer = await backend.create(source, due_at_ms=1, note="pre-send-timeout")

        block = True
        never_set = asyncio.Event()
        hooks = AgentHooks()

        async def slow_hook(ctx):
            if block:
                await never_set.wait()

        hooks.register(HookPoint.TIMER_FIRE, slow_hook)
        send_calls = 0
        orig_send = runtime.client.send

        async def counting_send(*args, **kwargs):
            nonlocal send_calls
            send_calls += 1
            return await orig_send(*args, **kwargs)

        monkeypatch.setattr(runtime.client, "send", counting_send)

        runner = TimerRunner(
            store=store, client=runtime.client, hooks=hooks, event_bus=bus, acceptance_timeout_s=0.05,
        )
        assert await runner.scan_once() == 0
        assert send_calls == 0
        row = await store.get_timer(tgt.session_id, timer.timer_id)
        assert row is not None and row.status == "armed" and row.fire_count == 0
        assert events == []

        # Once the hook stops timing out, the timer fires normally.
        block = False
        assert await runner.scan_once() == 1
        assert send_calls == 1
        row2 = await store.get_timer(tgt.session_id, timer.timer_id)
        assert row2 is not None and row2.status == "fired" and row2.fire_count == 1
    finally:
        await runtime.worker.shutdown()
        await runtime.client.aclose()
        await runtime.transport.aclose()


async def test_h6_p1_19_timer_prune_does_not_reuse_id_and_old_cancel_cannot_cancel_new(registry_store):
    """P1-19 rework: pruning terminal timers never reuses timer_id so old receipts cannot cancel new timers."""
    store, _ = registry_store
    runtime, tgt, actor, _ = await _timer_runtime(store)
    try:
        from power_loop import StoreTimerBackend

        backend = StoreTimerBackend(store)
        src_old = TimerContext(store.store_id, tgt.session_id, actor, operation_id="op-old-1")
        old_timer = await backend.create(src_old, due_at_ms=1, note="old")
        assert old_timer.timer_id == 1

        # Settle old timer to terminal 'fired' and prune with retention=0.
        await store.transition_timer(tgt.session_id, old_timer.timer_id, from_status="armed", to_status="firing")
        assert await store.finish_firing_timer(tgt.session_id, old_timer.timer_id)
        runner = TimerRunner(store=store, client=runtime.client, prune_interval_s=0.01, prune_older_than_ms=0)
        assert await runner.scan_once() == 0
        assert await store.get_timer(tgt.session_id, old_timer.timer_id) is None

        # Create a new timer with a new operation_id: timer_id must strictly advance (no reuse of 1).
        src_new = TimerContext(store.store_id, tgt.session_id, actor, operation_id="op-new-2")
        new_timer = await backend.create(src_new, delay_s=3600, note="new")
        assert new_timer.timer_id > old_timer.timer_id

        # Cancelling with the stale old receipt ID must return False and leave new_timer armed.
        assert await backend.cancel(src_old, old_timer.timer_id) is False
        kept = await store.get_timer(tgt.session_id, new_timer.timer_id)
        assert kept is not None and kept.status == "armed"

        # operation_id idempotency for new_timer still works while retained.
        new_retry = await backend.create(src_new, delay_s=3600, note="new")
        assert new_retry.timer_id == new_timer.timer_id
    finally:
        await runtime.worker.shutdown()
        await runtime.client.aclose()
        await runtime.transport.aclose()
