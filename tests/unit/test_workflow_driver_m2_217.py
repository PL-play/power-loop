"""Design 217 M2: Plan B Resident workflow_driver on Session-Core (Probes 1/2/6/9, crash replay, 5 features)."""
from __future__ import annotations

import asyncio
import json

import pytest

from power_loop import AgentLoopConfig, LLMResponse, LLMTokenUsage, ToolDefinition
from power_loop.agent.runtime_store import now_ms
from power_loop.agent.runtime_types import InputRef, TrustedActor, result_message_id
from power_loop.core.agent_context import get_session_id
from power_loop.runtime.pricing import Free
from power_loop.tools.registry import ToolRegistry
from power_loop.workflow import journal, register_workflow_tool
from tests.runtime_fixture import RunningAgent

from ._workflow_fixture_203 import Model, leaf

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def test_probe1_inspect_input_midflight_settle_and_halt_zero_ghost_rewake(registry_store):
    """Probe 1: inspect_input mid-flight None -> explicit settle on Final; halt uses cancel_by='subtree' (wake=0)."""
    store, _ = registry_store
    gate = asyncio.Event()
    b_started = asyncio.Event()
    release_b = asyncio.Event()

    class Probe1Model(Model):
        async def complete(self, req, **kwargs):
            if req.system_prompt.startswith("N1"):
                await gate.wait()
                return LLMResponse(raw_text="N1 done")
            if req.system_prompt.startswith("A"):
                await b_started.wait()
                raise RuntimeError("A boom")
            if req.system_prompt.startswith("B"):
                b_started.set()
                await release_b.wait()
                return LLMResponse(raw_text="B late")
            return LLMResponse(raw_text="main ack")

    runner = await RunningAgent(
        store, Probe1Model(), config=AgentLoopConfig(context_budget_tokens=10**9), capacity=2,
    ).start()
    try:
        # 1) Normal workflow: inspect_input mid-flight is None, settled with result_json on Final
        h1 = await runner.runtime.client.workflow(
            runner.target, {"name": "p1-normal", "root": leaf("N1")},
            message_id="p1-normal", actor=runner.actor,
        )
        async with asyncio.timeout(5):
            while True:
                st = await store._db.fetchone(
                    f"SELECT execution_state FROM {store.t.sessions} WHERE session_id=?",
                    (h1.target.session_id,),
                )
                if st and st["execution_state"] == "running":
                    break
                await asyncio.sleep(0.01)
        mid_info = await runner.runtime.client.inspect(h1.input_ref, actor=runner.actor)
        assert mid_info.terminal_status is None

        gate.set()
        res1 = await h1.wait(deadline_ms=now_ms() + 10000)
        assert res1.status == "completed" and res1.final.text == "N1 done"
        settled_row = await store._db.fetchone(
            f"SELECT terminal_status,result_json,completed_at FROM {store.t.follow_up_queue} "
            "WHERE session_id=? AND item_id=?",
            (h1.target.session_id, h1.input_ref.message_id),
        )
        assert settled_row["terminal_status"] == "completed"
        assert settled_row["result_json"] is not None and settled_row["completed_at"] is not None
        await runner.drain()

        # 2) on_error='halt': stops sibling B with cancel_by='subtree' & wake=0 -> 0 Driver re-wake
        h2 = await runner.runtime.client.workflow(
            runner.target,
            {"name": "p1-halt", "root": {"type": "parallel", "on_error": "halt", "branches": [leaf("A"), leaf("B")]}},
            message_id="p1-halt", actor=runner.actor,
        )
        await asyncio.wait_for(b_started.wait(), 5)
        res2 = await h2.wait(deadline_ms=now_ms() + 10000)
        assert res2.status == "failed"
        await runner.finish(h2.input_ref)

        release_b.set()
        await runner.drain()

        drv_state = await store._db.fetchone(
            f"SELECT execution_state,send_index,wake_pending_count FROM {store.t.sessions} WHERE session_id=?",
            (h2.target.session_id,),
        )
        assert dict(drv_state) == {"execution_state": "idle", "send_index": 1, "wake_pending_count": 0}
        late_b = await store._db.fetchall(
            f"SELECT wake,content_json FROM {store.t.follow_up_queue} "
            "WHERE session_id=? AND kind='workflow_node_result' AND status='pending'",
            (h2.target.session_id,),
        )
        assert len(late_b) == 1 and int(late_b[0]["wake"]) == 0
        assert json.loads(late_b[0]["content_json"])["cancel_by"] == "subtree"
    finally:
        gate.set()
        release_b.set()
        await runner.aclose()


async def test_probe2_stop_driver_subtree_vs_stop_main_subtree(registry_store):
    """Probe 2: stop(driver, subtree) -> 0 dead-wait + wake=1 to Main; stop(main, subtree) -> wake=0 everywhere."""
    store, _ = registry_store
    l1_started = asyncio.Event()
    release_l1 = asyncio.Event()
    l2_started = asyncio.Event()
    release_l2 = asyncio.Event()

    class Probe2Model(Model):
        def __init__(self):
            self.main_wakes = 0

        async def complete(self, req, **kwargs):
            if req.system_prompt.startswith("L1"):
                l1_started.set()
                await release_l1.wait()
                return LLMResponse(raw_text="L1")
            if req.system_prompt.startswith("L2"):
                l2_started.set()
                await release_l2.wait()
                return LLMResponse(raw_text="L2")
            self.main_wakes += 1
            return LLMResponse(raw_text="main handled")

    model = Probe2Model()
    runner = await RunningAgent(
        store, model, config=AgentLoopConfig(context_budget_tokens=10**9), capacity=3,
    ).start()
    try:
        # Case A: stop(driver, scope='subtree') -> 0 dead-wait, wakes Main with workflow_result(cancelled, wake=1)
        h_a = await runner.runtime.client.workflow(
            runner.target, {"name": "stop-drv", "root": leaf("L1")},
            message_id="stop-drv", actor=runner.actor,
        )
        await asyncio.wait_for(l1_started.wait(), 5)
        rcpt_a = await runner.runtime.client.cancel(
            h_a.target, actor=runner.actor, scope="subtree", reason="cancel driver subtree",
        )
        assert rcpt_a.state == "cancel_requested"
        # Finishes BEFORE release_l1 is set (0 dead-wait!)
        res_a = await h_a.wait(deadline_ms=now_ms() + 5000)
        assert res_a.status == "cancelled" and res_a.cancel_by == "user"
        parent_ref_a = InputRef(
            store.store_id, runner.target.session_id,
            result_message_id(h_a.target, h_a.input_ref, 1, "workflow_result"),
        )
        main_a = await runner.runtime.client.wait(parent_ref_a, actor=runner.actor, deadline_ms=now_ms() + 10000)
        assert main_a.status == "completed" and model.main_wakes == 1
        release_l1.set()
        await runner.drain()

        # Case B: stop(main, scope='subtree') -> cascades cancel_by='subtree' to Driver + Leaf, wake=0 to Main!
        main_wakes_before = model.main_wakes
        h_b = await runner.runtime.client.workflow(
            runner.target, {"name": "stop-main", "root": leaf("L2")},
            message_id="stop-main", actor=runner.actor,
        )
        await asyncio.wait_for(l2_started.wait(), 5)
        await runner.runtime.client.cancel(
            runner.target, actor=runner.actor, scope="subtree", reason="cancel whole conversation",
        )
        res_b = await h_b.wait(deadline_ms=now_ms() + 5000)
        assert res_b.status == "cancelled" and res_b.cancel_by == "subtree"
        release_l2.set()
        await runner.drain()

        # Main MUST NOT be woken by stop(main, subtree)!
        assert model.main_wakes == main_wakes_before
        main_state = await store._db.fetchone(
            f"SELECT execution_state,wake_pending_count FROM {store.t.sessions} WHERE session_id=?",
            (runner.target.session_id,),
        )
        assert dict(main_state) == {"execution_state": "idle", "wake_pending_count": 0}
        wf_b_row = await store._db.fetchone(
            f"SELECT wake,content_json FROM {store.t.follow_up_queue} "
            "WHERE session_id=? AND item_id=?",
            (
                runner.target.session_id,
                result_message_id(h_b.target, h_b.input_ref, 1, "workflow_result"),
            ),
        )
        assert wf_b_row is not None and int(wf_b_row["wake"]) == 0
        assert json.loads(wf_b_row["content_json"])["cancel_by"] == "subtree"
    finally:
        release_l1.set()
        release_l2.set()
        await runner.aclose()


async def test_probe6_and_probe9_inflight_poke_immunity_close_if_idle_and_purge(registry_store):
    """Probe 6 (in-flight Driver ignores duplicate claim on wake=1 poke) + Probe 9 (close_if_idle & purge)."""
    store, _ = registry_store
    leaf_gate = asyncio.Event()
    leaf_started = asyncio.Event()

    class Gated(Model):
        async def complete(self, req, **kwargs):
            if req.system_prompt.startswith("A"):
                leaf_started.set()
                await leaf_gate.wait()
                return LLMResponse(raw_text="A ok")
            return LLMResponse(raw_text="main ok")

    runner = await RunningAgent(
        store, Gated(), config=AgentLoopConfig(context_budget_tokens=10**9), capacity=2,
    ).start()
    try:
        handle = await runner.runtime.client.workflow(
            runner.target, {"name": "p6-p9", "root": leaf("A")},
            message_id="p6-p9", actor=runner.actor,
        )
        await asyncio.wait_for(leaf_started.wait(), 5)
        children = await runner.runtime.client.list_children(handle.target, actor=runner.actor)
        leaf_sid = children.items[0].session_id

        # Probe 9a: while Driver & Leaf are running, close_sessions_if_idle reports 'busy'
        assert await store.close_sessions_if_idle([handle.target.session_id]) == "busy"
        assert await store.close_sessions_if_idle([leaf_sid]) == "busy"

        # Probe 6: while Resident Driver is running, a concurrent claim() MUST return None
        dup_claim = await runner.runtime.worker.inputs.claim(
            handle.target, node_id="other-node", boot_id="other-boot", ttl_s=30.0,
        )
        assert dup_claim is None

        # Finish workflow + parent turn
        leaf_gate.set()
        wf_res = await handle.wait(deadline_ms=now_ms() + 10000)
        assert wf_res.status == "completed"
        parent_ref = InputRef(
            store.store_id, runner.target.session_id,
            result_message_id(handle.target, handle.input_ref, 1, "workflow_result"),
        )
        await runner.runtime.client.wait(parent_ref, actor=runner.actor, deadline_ms=now_ms() + 10000)
        await runner.drain()

        # Probe 9b: once idle, close_sessions_if_idle closes Driver + linked Leaf cleanly, and purge cleans them
        assert await store.close_sessions_if_idle([handle.target.session_id]) == "closed"
        cutoff_far_future = now_ms() + 31 * 86_400_000
        # First pass purges leaf; second pass purges driver (after child session row is gone)
        await store.purge_expired_page(cutoff_far_future, limit=50)
        # Settle start_input retention (>30d cutoff) and purge driver
        await store._db.execute(
            f"DELETE FROM {store.t.follow_up_queue} WHERE session_id=?",
            (handle.target.session_id,),
        )
        await store.purge_expired_page(cutoff_far_future, limit=50)
        rem = await store._db.fetchall(
            f"SELECT session_id FROM {store.t.sessions} WHERE session_id IN (?,?)",
            (leaf_sid, handle.target.session_id),
        )
        assert rem == []
    finally:
        leaf_gate.set()
        await runner.aclose()


async def test_crash_takeover_replay_does_not_rerun_completed_leaves(registry_store):
    """Driver crash via periodic_tick after Step A completes: Step A is NOT re-run; Step B finishes cleanly."""
    store, _ = registry_store
    calls = {"A": 0, "B": 0}
    b_started = asyncio.Event()
    release_b = asyncio.Event()

    class ReplayModel(Model):
        async def complete(self, req, **kwargs):
            if req.system_prompt.startswith("A"):
                calls["A"] += 1
                return LLMResponse(raw_text="A-once")
            if req.system_prompt.startswith("B"):
                calls["B"] += 1
                b_started.set()
                await release_b.wait()
                return LLMResponse(raw_text="B-once")
            return LLMResponse(raw_text="main ack")

    runner = await RunningAgent(
        store, ReplayModel(), config=AgentLoopConfig(context_budget_tokens=10**9), capacity=2,
    ).start()
    try:
        spec = {
            "name": "crash-replay",
            "root": {"type": "sequence", "steps": [leaf("A"), leaf("B")]},
        }
        handle = await runner.runtime.client.workflow(
            runner.target, spec, message_id="crash-replay", actor=runner.actor,
        )
        await asyncio.wait_for(b_started.wait(), 5)
        assert calls == {"A": 1, "B": 1}

        # Simulate sudden Driver crash (kill local task + expire lease) while Leaf B stays healthy!
        drv_sid = handle.target.session_id
        active_ctx = runner.runtime.worker._active.get(drv_sid)
        assert active_ctx is not None
        active_ctx.shutdown_interrupted = True
        drv_task = runner.executions[handle.input_ref]
        drv_task.cancel("simulated crash")
        await asyncio.gather(drv_task, return_exceptions=True)
        runner.runtime.worker._active.pop(drv_sid, None)
        await store._db.execute(
            f"UPDATE {store.t.sessions} SET owner_lease_until_ms=0 WHERE session_id=?",
            (drv_sid,),
        )

        # Node periodic_tick discovers expired workflow_driver and invokes non-terminating recover()!
        await runner.runtime.worker.periodic_tick(limit=32)
        release_b.set()

        wf_res = await handle.wait(deadline_ms=now_ms() + 15000)
        assert wf_res.status == "completed"
        assert wf_res.results["A"].text == "A-once"
        assert wf_res.results["B"].text == "B-once"
        # CRITICAL: neither A nor B was re-executed!
        assert calls == {"A": 1, "B": 1}
    finally:
        release_b.set()
        await runner.aclose()


async def test_post_completion_human_direct_chat_with_leaf_writes_wake0_notice_only(registry_store):
    """Design 217 §5.3: human direct chat with finished workflow leaf -> human reply + Main system_notice(wake=0), 0 Driver wake."""
    store, _ = registry_store
    main_calls = 0

    class ChatModel(Model):
        async def complete(self, req, **kwargs):
            nonlocal main_calls
            if req.system_prompt.startswith("A"):
                last_user = next(
                    (m["content"] for m in reversed(req.messages) if m["role"] == "user"), "",
                )
                if "direct human follow-up" in str(last_user):
                    return LLMResponse(raw_text="reply directly to human")
                return LLMResponse(raw_text="initial leaf output")
            main_calls += 1
            return LLMResponse(raw_text="main handled")

    runner = await RunningAgent(
        store, ChatModel(), config=AgentLoopConfig(context_budget_tokens=10**9), capacity=2,
    ).start()
    try:
        handle = await runner.runtime.client.workflow(
            runner.target, {"name": "leaf-chat", "root": leaf("A")},
            message_id="leaf-chat", actor=runner.actor,
        )
        wf_res = await handle.wait(deadline_ms=now_ms() + 10000)
        assert wf_res.status == "completed"
        parent_ref = InputRef(
            store.store_id, runner.target.session_id,
            result_message_id(handle.target, handle.input_ref, 1, "workflow_result"),
        )
        await runner.runtime.client.wait(parent_ref, actor=runner.actor, deadline_ms=now_ms() + 10000)
        await runner.drain()
        assert main_calls == 1

        children = await runner.runtime.client.list_children(handle.target, actor=runner.actor)
        leaf_target = children.items[0]

        # Human actor (actor.session_id is None -> reply_route is None!) chats directly with finished leaf
        human_actor = TrustedActor(runner.actor.tenant_id, runner.actor.principal_id, None)
        rcpt = await runner.runtime.client.send(
            leaf_target, "direct human follow-up", message_id="human-msg-1", actor=human_actor,
        )
        human_res = await runner.runtime.client.wait(
            rcpt.input_ref, actor=human_actor, deadline_ms=now_ms() + 10000,
        )
        assert human_res.status == "completed" and human_res.final_text == "reply directly to human"
        await runner.drain()

        # 1) Driver stayed idle at send_index=1 with 0 new workflow_node_result rows!
        drv_state = await store._db.fetchone(
            f"SELECT execution_state,send_index,wake_pending_count FROM {store.t.sessions} WHERE session_id=?",
            (handle.target.session_id,),
        )
        assert dict(drv_state) == {"execution_state": "idle", "send_index": 1, "wake_pending_count": 0}
        drv_results = await store._db.fetchall(
            f"SELECT item_id FROM {store.t.follow_up_queue} "
            "WHERE session_id=? AND kind='workflow_node_result'",
            (handle.target.session_id,),
        )
        assert len(drv_results) == 1  # only Turn 1's workflow_node_result!

        # 2) Main received system_notice(wake=0) and was NOT woken!
        assert main_calls == 1
        notice = await store._db.fetchone(
            f"SELECT item_id,wake,status,content_json FROM {store.t.follow_up_queue} "
            "WHERE session_id=? AND kind='system_notice'",
            (runner.target.session_id,),
        )
        assert notice is not None
        assert len(notice["item_id"]) <= 64 and notice["item_id"].startswith("sys_chat_")
        assert int(notice["wake"]) == 0 and notice["status"] == "pending"
        assert json.loads(notice["content_json"])["event"] == "human_child_interaction"
    finally:
        await runner.aclose()


async def test_five_advanced_features_and_finish_late_on_session_core(registry_store):
    """Verify all 5 advanced features (Loop+continue_nodes, ContinuationPolicy, SharedBudget, on_error/fallback/retry, timeout_s) + _finish_late."""
    store, _ = registry_store
    registry = ToolRegistry()

    async def mark_todo():
        await store.set_runtime_state(
            get_session_id(), "todo", {"items": [{"text": "step", "status": "pending"}]},
        )
        return "todo_set"

    registry.register(
        ToolDefinition("mark_todo", "mark todo", input_schema={"type": "object"}, pricing=Free("test")),
        mark_todo,
    )
    register_workflow_tool(registry)

    class FeatureModel(Model):
        def __init__(self):
            self.loop_calls = 0
            self.cont_calls = 0
            self.retry_calls = 0

        async def complete(self, req, **kwargs):
            usage = LLMTokenUsage(prompt_tokens=2, completion_tokens=2, total_tokens=4)
            sp = req.system_prompt
            if sp.startswith("LOOP"):
                self.loop_calls += 1
                return LLMResponse(raw_text="done" if self.loop_calls >= 2 else "again", token_usage=usage)
            if sp.startswith("CONT"):
                self.cont_calls += 1
                last_u = next((str(m.get("content")) for m in reversed(req.messages) if m.get("role") == "user"), "")
                if "[continuation" in last_u:
                    return LLMResponse(raw_text="continued-ok", token_usage=usage)
                return LLMResponse(raw_text="", token_usage=usage, tool_calls=[{
                    "id": f"tc-{self.cont_calls}", "type": "function",
                    "function": {"name": "mark_todo", "arguments": "{}"},
                }])
            if sp.startswith("FLAKY"):
                self.retry_calls += 1
                if self.retry_calls == 1:
                    raise RuntimeError("transient")
                return LLMResponse(raw_text="retried-ok", token_usage=usage)
            if sp.startswith("FAIL_FB"):
                raise RuntimeError("always fails")
            if sp.startswith("FB"):
                return LLMResponse(raw_text="fallback-ok", token_usage=usage)
            if sp.startswith("SLOW"):
                await asyncio.sleep(5)
                return LLMResponse(raw_text="never", token_usage=usage)
            return LLMResponse(raw_text="main ack", token_usage=usage)

    model = FeatureModel()
    runner = await RunningAgent(
        store, model, registry, config=AgentLoopConfig(context_budget_tokens=10**9), capacity=4,
    ).start()
    try:
        # Features 1, 2, 3, 4 in one composite workflow:
        loop_leaf = leaf("LOOP")
        cont_leaf = leaf("CONT")
        cont_leaf["spec"].update(max_rounds=1, tools=["mark_todo"])
        cont_leaf["continuation"] = {"max_continuations": 1, "extra_rounds": 2, "gate": "todo_remaining"}
        retry_leaf = leaf("FLAKY")
        retry_leaf["retry"] = {"max_attempts": 2, "on": ["failed"]}
        fb_leaf = leaf("FAIL_FB")
        fb_leaf["fallback"] = leaf("FB")

        spec = {
            "name": "features-1-to-4",
            "budget": {"max_tokens": 100, "stop_at_remaining_pct": 10},
            "root": {
                "type": "sequence",
                "steps": [
                    {
                        "type": "loop",
                        "id": "L",
                        "max_iterations": 3,
                        "continue_nodes": ["LOOP"],
                        "body": loop_leaf,
                        "until": {"ref": "LOOP", "op": "eq", "value": "done"},
                    },
                    {
                        "type": "parallel",
                        "on_error": "continue",
                        "branches": [cont_leaf, retry_leaf, fb_leaf],
                    },
                ],
            },
        }
        h = await runner.runtime.client.workflow(
            runner.target, spec, message_id="feat-1-4", actor=runner.actor,
        )
        res = await h.wait(deadline_ms=now_ms() + 15000)
        assert res.status == "completed", res
        assert res.results["CONT"].text == "continued-ok"
        assert res.results["FLAKY"].text == "retried-ok"
        assert res.results["FB"].text == "fallback-ok"

        # Verify LoopNode(continue_nodes=True) and ContinuationPolicy reused original leaf sessions via kind='workflow_continue'!
        saved = await journal.read(store, runner.target.session_id, h.run_id)
        loop_nodes = [n for n in saved["nodes"] if n["path"].endswith("LOOP")]
        assert len(loop_nodes) == 2 and len({n["session_id"] for n in loop_nodes}) == 1
        cont_sids = {n["session_id"] for n in saved["nodes"] if n["path"] in {"CONT", "L/LOOP"}}
        wc_rows = await store._db.fetchall(
            f"SELECT session_id,kind FROM {store.t.follow_up_queue} WHERE kind='workflow_continue'",
        )
        assert {r["session_id"] for r in wc_rows} >= cont_sids

        # Feature 5: workflow-level timeout_s -> failed('workflow timeout') + sibling stopped with cancel_by='subtree' (wake=0)
        h_to = await runner.runtime.client.workflow(
            runner.target,
            {"name": "feat-timeout", "timeout_s": 1, "root": leaf("SLOW")},
            message_id="feat-timeout", actor=runner.actor,
        )
        res_to = await h_to.wait(deadline_ms=now_ms() + 10000)
        assert res_to.status == "failed" and "workflow timeout" in res_to.errors
        await runner.drain()

        # Verify _finish_late: sending a wake=1 poke after aggregate_delivered settles cleanly without duplicate workflow_result
        agg_before = await store._db.fetchall(
            f"SELECT item_id FROM {store.t.follow_up_queue} WHERE session_id=? AND kind='workflow_result'",
            (runner.target.session_id,),
        )
        poke = await runner.runtime.client.send(
            h.target, "late-poke", message_id="late-poke-1", actor=runner.actor,
        )
        poke_res = await runner.runtime.client.wait(
            poke.input_ref, actor=runner.actor, deadline_ms=now_ms() + 10000,
        )
        assert poke_res.status == "completed"
        agg_after = await store._db.fetchall(
            f"SELECT item_id FROM {store.t.follow_up_queue} WHERE session_id=? AND kind='workflow_result'",
            (runner.target.session_id,),
        )
        assert len(agg_after) == len(agg_before)
    finally:
        await runner.aclose()
