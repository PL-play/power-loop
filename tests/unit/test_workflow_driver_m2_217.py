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

        # Verify _finish_late (B2): sending a wake=1 poke after aggregate_delivered settles cleanly
        # with write_driver_message=False (0 new assistant messages, unchanged next_seq, 0 dup workflow_result).
        agg_before = await store._db.fetchall(
            f"SELECT item_id FROM {store.t.follow_up_queue} WHERE session_id=? AND kind='workflow_result'",
            (runner.target.session_id,),
        )
        asst_before = await store._db.fetchall(
            f"SELECT seq FROM {store.t.messages} WHERE session_id=? AND role='assistant'",
            (h.target.session_id,),
        )
        seq_before = await store._db.fetchone(
            f"SELECT next_seq FROM {store.t.sessions} WHERE session_id=?",
            (h.target.session_id,),
        )
        poke = await runner.runtime.client.send(
            h.target, "late-poke", message_id="late-poke-1", actor=runner.actor,
        )
        poke_res = await runner.runtime.client.wait(
            poke.input_ref, actor=runner.actor, deadline_ms=now_ms() + 10000,
        )
        assert poke_res.status == "completed" and poke_res.result_seq is None
        agg_after = await store._db.fetchall(
            f"SELECT item_id FROM {store.t.follow_up_queue} WHERE session_id=? AND kind='workflow_result'",
            (runner.target.session_id,),
        )
        asst_after = await store._db.fetchall(
            f"SELECT seq FROM {store.t.messages} WHERE session_id=? AND role='assistant'",
            (h.target.session_id,),
        )
        seq_after = await store._db.fetchone(
            f"SELECT next_seq FROM {store.t.sessions} WHERE session_id=?",
            (h.target.session_id,),
        )
        assert len(agg_after) == len(agg_before)
        assert len(asst_after) == len(asst_before) == 1
        assert int(seq_after["next_seq"]) == int(seq_before["next_seq"])
    finally:
        await runner.aclose()


# ── M1a Rework Regressions (B1–B7 + M1a-6 5-leaf multi-value create_many) ─────


async def _assert_session_counts_match_pending(db, store, sid: str) -> dict:
    srow = await db.fetchone(
        f"SELECT execution_state,lifecycle,wake_pending_count,pending_count,pending_bytes "
        f"FROM {store.t.sessions} WHERE session_id=?",
        (sid,),
    )
    assert srow is not None
    qagg = await db.fetchone(
        f"SELECT COALESCE(SUM(wake),0) AS w,COUNT(*) AS c,COALESCE(SUM(content_bytes),0) AS b "
        f"FROM {store.t.follow_up_queue} WHERE session_id=? AND status='pending'",
        (sid,),
    )
    assert int(srow["wake_pending_count"]) == int(qagg["w"]), (dict(srow), dict(qagg))
    assert int(srow["pending_count"]) == int(qagg["c"]), (dict(srow), dict(qagg))
    assert int(srow["pending_bytes"]) == int(qagg["b"]), (dict(srow), dict(qagg))
    if srow["execution_state"] != "running":
        expected_state = "queued" if (srow["lifecycle"] == "open" and int(qagg["w"]) > 0) else "idle"
        assert srow["execution_state"] == expected_state, (dict(srow), dict(qagg))
    return dict(srow)


async def _setup_driver_and_leaf(port, prefix: str, *, with_second_leaf: bool = False):
    from dataclasses import replace

    from power_loop.agent.runtime_store import ChildSpec

    from .session_native_fixture import ACTOR, DESCRIPTOR, target

    main = await target(port, f"{prefix}-main")
    drv_desc = replace(DESCRIPTOR, kind="workflow_driver")
    cm_drv = await port.create_many(
        parent=main, actor=ACTOR,
        specs=[ChildSpec(
            session_id=f"{prefix}-drv", message_id=f"{prefix}-dm1", body={"wf": "start"},
            descriptor=drv_desc, result_kind="workflow_result", cancel_propagation="independent",
        )],
        node_id="n1", boot_id="b1", can_execute=True, free_slots=1,
    )
    drv_batch = cm_drv.preclaimed[0]
    leaf_desc = replace(DESCRIPTOR, config={**dict(DESCRIPTOR.config), "workflow_node": {"id": "L1"}})
    leaf_specs = [ChildSpec(
        session_id=f"{prefix}-leaf1", message_id=f"{prefix}-lm1", body={"text": "work-1"},
        descriptor=leaf_desc, result_kind="workflow_node_result",
    )]
    if with_second_leaf:
        leaf_specs.append(ChildSpec(
            session_id=f"{prefix}-leaf2", message_id=f"{prefix}-lm2", body={"text": "work-2"},
            descriptor=replace(DESCRIPTOR, config={**dict(DESCRIPTOR.config), "workflow_node": {"id": "L2"}}),
            result_kind="workflow_node_result",
        ))
    cm_leaves = await port.create_many(
        parent=drv_batch.target, actor=ACTOR, specs=leaf_specs,
        node_id="n1", boot_id="b1", can_execute=True, free_slots=len(leaf_specs),
        parent_owner_epoch=drv_batch.owner_epoch, parent_cancel_seq=drv_batch.cancel_seq,
    )
    return main, drv_batch, cm_leaves.preclaimed


async def test_b1_driver_final_drain_no_deadlock_vs_recover_and_close(registry_store):
    """B1: Two-connection deterministic interleavings of Driver Final drain vs recover & set_lifecycle('closed')."""
    from power_loop.agent.runtime_store import RuntimeInputStore
    from power_loop.contracts.errors import SessionLeaseLost

    from .session_native_fixture import ACTOR

    store, connect = registry_store
    dialect = store._db.dialect.name
    p1 = RuntimeInputStore(store)
    peer = await connect(schema="verify")
    p2 = RuntimeInputStore(peer)
    try:
        # 1a) recover holds sessions(driver) lock FIRST, Driver Final starts concurrently -> SessionLeaseLost, 0 deadlock
        _, drv_batch, (leaf1_batch,) = await _setup_driver_and_leaf(p1, "b1-rec-first")
        await p1.final(leaf1_batch, status="completed", final_text="leaf1-ok")
        await store._db.execute(
            f"UPDATE {store.t.sessions} SET owner_lease_until_ms=0 WHERE session_id=?",
            (drv_batch.target.session_id,),
        )
        if dialect in ("postgres", "mysql"):
            recover_locked = asyncio.Event()
            final_started = asyncio.Event()
            orig_recover_wf = p1._recover_workflow_tx

            async def hooked_recover_wf(tx, own_state, tgt, *args, **kwargs):
                recover_locked.set()
                await asyncio.wait_for(final_started.wait(), 5)
                await asyncio.sleep(0.05)
                return await orig_recover_wf(tx, own_state, tgt, *args, **kwargs)

            p1._recover_workflow_tx = hooked_recover_wf  # type: ignore[method-assign]

            async def run_final_2():
                await asyncio.wait_for(recover_locked.wait(), 5)
                final_started.set()
                return await p2.final(
                    drv_batch, status="completed", final_text="stale-drv",
                    drain_driver_node_results=True, settle_start_input=True, write_driver_message=True,
                )

            try:
                rec_res, fin_res = await asyncio.gather(
                    p1.recover(drv_batch.target, node_id="n2", boot_id="b2"),
                    run_final_2(),
                    return_exceptions=True,
                )
            finally:
                p1._recover_workflow_tx = orig_recover_wf  # type: ignore[method-assign]
            assert not isinstance(rec_res, BaseException), rec_res
            assert isinstance(fin_res, SessionLeaseLost), repr(fin_res)
        else:
            rec_res = await p1.recover(drv_batch.target, node_id="n2", boot_id="b2")
            assert rec_res is not None
            with pytest.raises(SessionLeaseLost):
                await p1.final(
                    drv_batch, status="completed", final_text="stale-drv",
                    drain_driver_node_results=True, settle_start_input=True, write_driver_message=True,
                )
        await _assert_session_counts_match_pending(store._db, store, drv_batch.target.session_id)

        # 1b) Driver Final holds sessions(driver) lock FIRST (paused in drain), recover runs concurrently -> 0 deadlock
        _, drv_b2, (leaf_b2,) = await _setup_driver_and_leaf(p1, "b1-fin-first")
        await p1.final(leaf_b2, status="completed", final_text="leaf1-ok")
        await store._db.execute(
            f"UPDATE {store.t.sessions} SET owner_lease_until_ms=0 WHERE session_id=?",
            (drv_b2.target.session_id,),
        )
        if dialect in ("postgres", "mysql"):
            final_locked = asyncio.Event()
            recover_started = asyncio.Event()
            orig_drain = p1._drain_pending_driver_node_results_tx

            async def hooked_drain(tx, sid, send_index):
                res = await orig_drain(tx, sid, send_index)
                final_locked.set()
                await asyncio.wait_for(recover_started.wait(), 5)
                await asyncio.sleep(0.05)
                return res

            p1._drain_pending_driver_node_results_tx = hooked_drain  # type: ignore[method-assign]

            async def run_recover_2():
                await asyncio.wait_for(final_locked.wait(), 5)
                recover_started.set()
                return await p2.recover(drv_b2.target, node_id="n2", boot_id="b2")

            try:
                fin_res2, rec_res2 = await asyncio.gather(
                    p1.final(
                        drv_b2, status="completed", final_text="drv-ok",
                        drain_driver_node_results=True, settle_start_input=True, write_driver_message=True,
                    ),
                    run_recover_2(),
                )
            finally:
                p1._drain_pending_driver_node_results_tx = orig_drain  # type: ignore[method-assign]
            assert fin_res2.execution_state == "idle"
            assert rec_res2 is None
        else:
            fin_res2 = await p1.final(
                drv_b2, status="completed", final_text="drv-ok",
                drain_driver_node_results=True, settle_start_input=True, write_driver_message=True,
            )
            assert fin_res2.execution_state == "idle"
            assert await p1.recover(drv_b2.target, node_id="n2", boot_id="b2") is None
        st_b2 = await _assert_session_counts_match_pending(store._db, store, drv_b2.target.session_id)
        assert st_b2["execution_state"] == "idle" and st_b2["pending_count"] == 0

        # 2a) set_lifecycle('closed') holds sessions(driver) lock FIRST, Driver Final starts concurrently -> 0 deadlock
        _, drv_b3, (leaf_b3,) = await _setup_driver_and_leaf(p1, "b1-close-first")
        await p1.final(leaf_b3, status="completed", final_text="leaf1-ok")
        if dialect in ("postgres", "mysql"):
            close_locked = asyncio.Event()
            final3_started = asyncio.Event()
            orig_lock_state = p1.db.dialect.lock_state

            async def hooked_lock_state(tx, table, sid):
                row = await orig_lock_state(tx, table, sid)
                if sid == drv_b3.target.session_id:
                    close_locked.set()
                    await asyncio.wait_for(final3_started.wait(), 5)
                    await asyncio.sleep(0.05)
                return row

            p1.db.dialect.lock_state = hooked_lock_state  # type: ignore[method-assign]

            async def run_final_3():
                await asyncio.wait_for(close_locked.wait(), 5)
                final3_started.set()
                return await p2.final(
                    drv_b3, status="completed", final_text="drv-raced-close",
                    drain_driver_node_results=True, settle_start_input=True, write_driver_message=True,
                )

            try:
                close_res, fin_res3 = await asyncio.gather(
                    p1.set_lifecycle(
                        drv_b3.target, "closed", actor=ACTOR, inflight="abort", subtree="none",
                    ),
                    run_final_3(),
                )
            finally:
                p1.db.dialect.lock_state = orig_lock_state  # type: ignore[method-assign]
            assert close_res.lifecycle == "closed"
            assert fin_res3.results[0].status == "cancelled"
        else:
            await p1.set_lifecycle(drv_b3.target, "closed", actor=ACTOR, inflight="abort", subtree="none")
            fin_res3 = await p1.final(
                drv_b3, status="completed", final_text="drv-raced-close",
                drain_driver_node_results=True, settle_start_input=True, write_driver_message=True,
            )
            assert fin_res3.results[0].status == "cancelled"
        st_b3 = await _assert_session_counts_match_pending(store._db, store, drv_b3.target.session_id)
        assert st_b3["execution_state"] == "idle" and st_b3["pending_count"] == 0

        # 2b) Driver Final holds sessions(driver) lock FIRST, set_lifecycle('closed') starts concurrently -> 0 deadlock
        _, drv_b4, (leaf_b4,) = await _setup_driver_and_leaf(p1, "b1-fin-close")
        await p1.final(leaf_b4, status="completed", final_text="leaf1-ok")
        if dialect in ("postgres", "mysql"):
            fin4_locked = asyncio.Event()
            close4_started = asyncio.Event()
            orig_drain4 = p1._drain_pending_driver_node_results_tx

            async def hooked_drain4(tx, sid, send_index):
                res = await orig_drain4(tx, sid, send_index)
                fin4_locked.set()
                await asyncio.wait_for(close4_started.wait(), 5)
                await asyncio.sleep(0.05)
                return res

            p1._drain_pending_driver_node_results_tx = hooked_drain4  # type: ignore[method-assign]

            async def run_close_4():
                await asyncio.wait_for(fin4_locked.wait(), 5)
                close4_started.set()
                return await p2.set_lifecycle(
                    drv_b4.target, "closed", actor=ACTOR, inflight="abort", subtree="none",
                )

            try:
                fin_res4, close_res4 = await asyncio.gather(
                    p1.final(
                        drv_b4, status="completed", final_text="drv-ok-4",
                        drain_driver_node_results=True, settle_start_input=True, write_driver_message=True,
                    ),
                    run_close_4(),
                )
            finally:
                p1._drain_pending_driver_node_results_tx = orig_drain4  # type: ignore[method-assign]
            assert fin_res4.execution_state == "idle"
            assert close_res4.lifecycle == "closed"
        else:
            fin_res4 = await p1.final(
                drv_b4, status="completed", final_text="drv-ok-4",
                drain_driver_node_results=True, settle_start_input=True, write_driver_message=True,
            )
            assert fin_res4.execution_state == "idle"
            await p1.set_lifecycle(drv_b4.target, "closed", actor=ACTOR, inflight="abort", subtree="none")
        st_b4 = await _assert_session_counts_match_pending(store._db, store, drv_b4.target.session_id)
        assert st_b4["execution_state"] == "idle" and st_b4["pending_count"] == 0
    finally:
        await peer.close()


@pytest.mark.parametrize("leaf2_subtree_cancel", [False, True])
async def test_b1_driver_drain_concurrent_second_leaf_commit_counts_exact(registry_store, leaf2_subtree_cancel):
    """B1: Second leaf commits while Driver Final is draining -> 0 deadlock, exact row-matched counts, no spurious queue."""
    from power_loop.agent.runtime_store import RuntimeInputStore

    from .session_native_fixture import ACTOR

    store, connect = registry_store
    dialect = store._db.dialect.name
    p1 = RuntimeInputStore(store)
    peer = await connect(schema="verify")
    p2 = RuntimeInputStore(peer)
    try:
        suffix = f"b1-leaf2-{'sub' if leaf2_subtree_cancel else 'norm'}"
        _, drv_batch, (leaf1_batch, leaf2_batch) = await _setup_driver_and_leaf(
            p1, suffix, with_second_leaf=True,
        )
        # Leaf 1 commits first so Driver has 1 pending workflow_node_result before Final starts
        await p1.final(leaf1_batch, status="completed", final_text="leaf1-done")
        if leaf2_subtree_cancel:
            await p1.cancel(
                leaf2_batch.target, actor=ACTOR, scope="self",
                reason="halt sibling", cancel_by="subtree",
            )

        if dialect in ("postgres", "mysql"):
            drain_selected = asyncio.Event()
            leaf2_started = asyncio.Event()
            orig_drain = p1._drain_pending_driver_node_results_tx
            captured_sql: list[str] = []

            async def hooked_drain(tx, sid, send_index):
                class _SpyTx:
                    def __getattr__(self, name):
                        return getattr(tx, name)

                    async def fetchall(self, sql, params=()):
                        captured_sql.append(sql)
                        rows = await tx.fetchall(sql, params)
                        # Pause right after SELECT ... FOR UPDATE (MySQL) / UPDATE ... RETURNING (PG)
                        drain_selected.set()
                        await asyncio.wait_for(leaf2_started.wait(), 5)
                        await asyncio.sleep(0.06)
                        return rows

                    async def execute(self, sql, params=()):
                        captured_sql.append(sql)
                        return await tx.execute(sql, params)

                return await orig_drain(_SpyTx(), sid, send_index)

            p1._drain_pending_driver_node_results_tx = hooked_drain  # type: ignore[method-assign]

            async def run_leaf2():
                await asyncio.wait_for(drain_selected.wait(), 5)
                leaf2_started.set()
                return await p2.final(
                    leaf2_batch,
                    status="cancelled" if leaf2_subtree_cancel else "completed",
                    final_text="" if leaf2_subtree_cancel else "leaf2-done",
                    error="halt sibling" if leaf2_subtree_cancel else None,
                    error_code="cancelled" if leaf2_subtree_cancel else None,
                )

            try:
                drv_fin, leaf2_fin = await asyncio.gather(
                    p1.final(
                        drv_batch, status="completed", final_text="drv-done",
                        drain_driver_node_results=True, settle_start_input=True, write_driver_message=True,
                    ),
                    run_leaf2(),
                )
            finally:
                p1._drain_pending_driver_node_results_tx = orig_drain  # type: ignore[method-assign]

            assert drv_fin.execution_state == "idle"
            assert leaf2_fin.result_delivery_error is None
            if dialect == "mysql":
                assert any("FOR UPDATE" in s for s in captured_sql), captured_sql
                assert any("id IN (" in s for s in captured_sql), captured_sql
        else:
            drv_fin = await p1.final(
                drv_batch, status="completed", final_text="drv-done",
                drain_driver_node_results=True, settle_start_input=True, write_driver_message=True,
            )
            assert drv_fin.execution_state == "idle"
            leaf2_fin = await p1.final(
                leaf2_batch,
                status="cancelled" if leaf2_subtree_cancel else "completed",
                final_text="" if leaf2_subtree_cancel else "leaf2-done",
                error="halt sibling" if leaf2_subtree_cancel else None,
                error_code="cancelled" if leaf2_subtree_cancel else None,
            )
            assert leaf2_fin.result_delivery_error is None

        # Leaf 1 was drained; Leaf 2 arrived after Driver Final committed -> exact 1 pending row matching counts!
        st = await _assert_session_counts_match_pending(store._db, store, drv_batch.target.session_id)
        assert st["pending_count"] == 1
        if leaf2_subtree_cancel:
            assert st["wake_pending_count"] == 0 and st["execution_state"] == "idle"
        else:
            assert st["wake_pending_count"] == 1 and st["execution_state"] == "queued"
    finally:
        await peer.close()


async def test_b2_driver_write_message_false_and_failed_final_zero_max_seq(registry_store):
    """B2: Driver honors write_driver_message=False and never runs SELECT MAX(seq) on failure final."""
    from dataclasses import replace

    from power_loop.agent.runtime_store import ChildSpec, RuntimeInputStore

    from .session_native_fixture import ACTOR, DESCRIPTOR, target
    from .test_send_sql_profile import Profiled

    store, _ = registry_store
    port = RuntimeInputStore(store)
    main = await target(port, "b2-main")
    drv_desc = replace(DESCRIPTOR, kind="workflow_driver")

    # 1) Worker generic _final_failed parameter combo (status='failed', next_seq=None, write_driver_message=False)
    cm_fail = await port.create_many(
        parent=main, actor=ACTOR,
        specs=[ChildSpec(
            session_id="b2-drv-fail", message_id="b2-dm-fail", body={"wf": "fail"},
            descriptor=drv_desc, result_kind="workflow_result",
        )],
        node_id="n1", boot_id="b1", can_execute=True, free_slots=1,
    )
    fail_batch = cm_fail.preclaimed[0]
    original = store._db
    profiled = Profiled(original)
    store._db = profiled
    try:
        fb = await port.final(
            fail_batch, status="failed", final_text="", error="boom", error_code="failed",
        )
    finally:
        store._db = original
    assert fb.results[0].status == "failed" and fb.results[0].result_seq is None
    assert not any("max(seq)" in r["sql"].lower() for r in profiled.records), [r["sql"] for r in profiled.records]
    fail_msgs = await store._db.fetchall(
        f"SELECT seq,role FROM {store.t.messages} WHERE session_id=?", ("b2-drv-fail",),
    )
    assert fail_msgs == []
    fail_sess = await store._db.fetchone(
        f"SELECT next_seq FROM {store.t.sessions} WHERE session_id=?", ("b2-drv-fail",),
    )
    assert int(fail_sess["next_seq"]) == 1

    # 2) Late input settled with write_driver_message=False keeps assistant message count & next_seq unchanged
    cm_ok = await port.create_many(
        parent=main, actor=ACTOR,
        specs=[ChildSpec(
            session_id="b2-drv-late", message_id="b2-dm-ok", body={"wf": "ok"},
            descriptor=drv_desc, result_kind="workflow_result",
        )],
        node_id="n1", boot_id="b1", can_execute=True, free_slots=1,
    )
    ok_batch = cm_ok.preclaimed[0]
    await port.final(
        ok_batch, consumed=ok_batch.input_refs, status="completed", final_text="agg-done",
        next_seq=ok_batch.message_seq + 1, settle_start_input=True, write_driver_message=True,
    )
    asst_before = await store._db.fetchall(
        f"SELECT seq FROM {store.t.messages} WHERE session_id=? AND role='assistant'", ("b2-drv-late",),
    )
    seq_before = int((await store._db.fetchone(
        f"SELECT next_seq FROM {store.t.sessions} WHERE session_id=?", ("b2-drv-late",),
    ))["next_seq"])
    assert len(asst_before) == 1

    late_rcpt = await port.accept(ok_batch.target, "late-poke", message_id="b2-late-1", actor=ACTOR)
    late_batch = await port.claim(ok_batch.target, node_id="n1", boot_id="b1")
    assert late_batch is not None
    profiled2 = Profiled(original)
    store._db = profiled2
    try:
        late_fb = await port.final(
            late_batch,
            consumed=late_batch.input_refs,
            status="completed",
            final_text="",
            next_seq=max(1, late_batch.message_seq),
            settle_chain_ids=(late_rcpt.input_ref.message_id,),
            drain_driver_node_results=True,
            deliver_to_parent=False,
            write_driver_message=False,
        )
    finally:
        store._db = original
    assert not any("max(seq)" in r["sql"].lower() for r in profiled2.records), [r["sql"] for r in profiled2.records]
    assert late_fb.results[0].status == "completed" and late_fb.results[0].result_seq is None
    late_res = await port.get_result(late_rcpt.input_ref, actor=ACTOR)
    assert late_res is not None and late_res.status == "completed" and late_res.result_seq is None
    asst_after = await store._db.fetchall(
        f"SELECT seq FROM {store.t.messages} WHERE session_id=? AND role='assistant'", ("b2-drv-late",),
    )
    seq_after = int((await store._db.fetchone(
        f"SELECT next_seq FROM {store.t.sessions} WHERE session_id=?", ("b2-drv-late",),
    ))["next_seq"])
    assert len(asst_after) == 1 and seq_after == seq_before


@pytest.mark.parametrize("mode", ["normal", "subtree", "subtree_cancel_raced"])
async def test_b3_cancel_retry_idempotent_preserves_delivery_and_wake(registry_store, mode):
    """B3: Repeated cancel on settled normal/subtree leaf keeps 1 receipt, result_delivery_error=None, subtree wake=0."""
    from power_loop.agent.runtime_store import RuntimeInputStore

    from .session_native_fixture import ACTOR

    store, _ = registry_store
    port = RuntimeInputStore(store)
    _, drv_batch, (leaf_batch,) = await _setup_driver_and_leaf(port, f"b3-{mode}")
    drv_sid = drv_batch.target.session_id
    leaf_ref = leaf_batch.input_refs[0]

    # Finish driver first so we can observe whether Driver gets spuriously woken on subtree retries
    if mode.startswith("subtree"):
        await port.cancel(drv_batch.target, actor=ACTOR, scope="subtree", reason="stop tree")
        await port.final(
            drv_batch, status="cancelled", final_text="", error="stop tree", error_code="cancelled",
            drain_driver_node_results=True, settle_start_input=True, write_driver_message=True,
        )
        if mode == "subtree_cancel_raced":
            # Simulates leaf calling final(status='completed', final_text=...) while cancel_raced=True
            leaf_fin = await port.final(
                leaf_batch, status="completed", final_text="raced-output",
                usage={"total_tokens": 7},
            )
        else:
            leaf_fin = await port.final(
                leaf_batch, status="cancelled", final_text="", error="stop tree", error_code="cancelled",
            )
        expected_cby = "subtree"
        expected_wake = 0
    else:
        rcpt1 = await port.cancel(leaf_ref, actor=ACTOR, reason="stop one leaf")
        assert rcpt1.state == "cancel_requested"
        leaf_fin = await port.final(
            leaf_batch, status="cancelled", final_text="", error="stop one leaf", error_code="cancelled",
        )
        expected_cby = "user"
        expected_wake = 1

    assert leaf_fin.result_delivery_error is None
    assert leaf_fin.results[0].status == "cancelled"
    assert leaf_fin.results[0].cancel_by == expected_cby

    # Repeat cancel on BOTH original InputRef AND leaf TargetRef
    dup_input = await port.cancel(leaf_ref, actor=ACTOR, reason="retry-input")
    dup_target = await port.cancel(leaf_batch.target, actor=ACTOR, reason="retry-target")
    assert dup_input.state == "already_finished"
    assert dup_target.state == "already_finished"

    # Stored leaf result MUST keep result_delivery_error=None (never flipped to 'request_conflict'!)
    stored_leaf = await port.get_result(leaf_ref, actor=ACTOR)
    assert stored_leaf is not None
    assert stored_leaf.status == "cancelled" and stored_leaf.cancel_by == expected_cby
    assert stored_leaf.result_delivery_error is None

    # Driver MUST have exactly 1 workflow_node_result with original cancel_by and wake!
    receipts = await store._db.fetchall(
        f"SELECT item_id,wake,content_json FROM {store.t.follow_up_queue} "
        "WHERE session_id=? AND kind='workflow_node_result'",
        (drv_sid,),
    )
    assert len(receipts) == 1
    assert int(receipts[0]["wake"]) == expected_wake
    assert json.loads(receipts[0]["content_json"])["cancel_by"] == expected_cby
    if expected_wake == 0:
        drv_row = await store._db.fetchone(
            f"SELECT execution_state,wake_pending_count FROM {store.t.sessions} WHERE session_id=?",
            (drv_sid,),
        )
        assert dict(drv_row) == {"execution_state": "idle", "wake_pending_count": 0}

    # Redelivery check: delete receipt from Driver inbox (simulating missed delivery) and retry cancel ->
    # must rebuild the EXACT same receipt identity, cancel_by, and wake=0 for subtree!
    orig_item_id = receipts[0]["item_id"]
    await store._db.execute(
        f"DELETE FROM {store.t.follow_up_queue} WHERE session_id=? AND item_id=?",
        (drv_sid, orig_item_id),
    )
    await store._db.execute(
        f"UPDATE {store.t.sessions} SET wake_pending_count=0,pending_count=0,pending_bytes=0 "
        "WHERE session_id=?",
        (drv_sid,),
    )
    redeliv = await port.cancel(leaf_ref, actor=ACTOR, reason="redeliver")
    assert redeliv.state == "already_finished"
    rebuilt = await store._db.fetchall(
        f"SELECT item_id,wake,content_json FROM {store.t.follow_up_queue} "
        "WHERE session_id=? AND kind='workflow_node_result'",
        (drv_sid,),
    )
    assert len(rebuilt) == 1
    assert rebuilt[0]["item_id"] == orig_item_id
    assert int(rebuilt[0]["wake"]) == expected_wake
    assert json.loads(rebuilt[0]["content_json"])["cancel_by"] == expected_cby
    if expected_wake == 0:
        drv_row2 = await store._db.fetchone(
            f"SELECT execution_state,wake_pending_count FROM {store.t.sessions} WHERE session_id=?",
            (drv_sid,),
        )
        assert dict(drv_row2) == {"execution_state": "idle", "wake_pending_count": 0}

    # Also verify repeat cancel on a voided Driver continuation chain (orig -> wfc_1) emits ONLY 1 receipt on Main!
    if mode == "normal":
        main_c, drv_c, _ = await _setup_driver_and_leaf(port, "b3-cont-chain")
        orig_c = drv_c.input_refs[0]
        await store._db.execute(
            f"UPDATE {store.t.sessions} SET owner_lease_until_ms=0 WHERE session_id=?",
            (drv_c.target.session_id,),
        )
        await port.recover(drv_c.target, node_id="n2", boot_id="b2")
        first_void = await port.cancel(orig_c, actor=ACTOR, reason="cancel queued wfc")
        assert first_void.state == "voided"
        rep_orig = await port.cancel(orig_c, actor=ACTOR, reason="repeat orig")
        rep_drv = await port.cancel(drv_c.target, actor=ACTOR, reason="repeat drv")
        assert rep_orig.state == "already_finished" and rep_drv.state == "already_finished"
        main_wf_rows = await store._db.fetchall(
            f"SELECT item_id,wake,content_json FROM {store.t.follow_up_queue} "
            "WHERE session_id=? AND kind='workflow_result'",
            (main_c.session_id,),
        )
        assert len(main_wf_rows) == 1 and int(main_wf_rows[0]["wake"]) == 1
        assert json.loads(main_wf_rows[0]["content_json"])["cancel_by"] == "user"
        assert (await port.get_result(orig_c, actor=ACTOR)).result_delivery_error is None


async def test_b4_cancelled_driver_takeover_preserves_late_input_and_closes_chain(registry_store):
    """B4: Running Driver gets node receipt + new wake=1 input -> cancel -> lease expire -> takeover keeps new input schedulable."""
    from power_loop.agent.runtime_store import RuntimeInputStore

    from .session_native_fixture import ACTOR

    store, _ = registry_store
    port = RuntimeInputStore(store)

    # Case 1: open Driver with 1 continuation hop (orig -> wfc_1) + pending node receipt + new wake=1 input
    _, drv_b1, (leaf_b1,) = await _setup_driver_and_leaf(port, "b4-open")
    drv = drv_b1.target
    orig_ref = drv_b1.input_refs[0]

    # Trigger 1 uncancelled takeover so orig -> waiting_continuation and wfc_1 -> pending -> claimed (send_index=2)
    await store._db.execute(
        f"UPDATE {store.t.sessions} SET owner_lease_until_ms=0 WHERE session_id=?", (drv.session_id,),
    )
    first_rec = await port.recover(drv, node_id="n2", boot_id="b2")
    assert first_rec is not None and first_rec.execution_state == "queued"
    drv_b2 = await port.claim(drv, node_id="n2", boot_id="b2")
    assert drv_b2 is not None and drv_b2.send_index == 2
    wfc_ref = drv_b2.input_refs[0]

    # While running send_index=2: receive 1 leaf workflow_node_result AND accept 1 new wake=1 user input
    await port.final(leaf_b1, status="completed", final_text="leaf-done")
    late_rcpt = await port.accept(drv, "new-input-after-start", message_id="b4-late-user", actor=ACTOR)
    assert late_rcpt.state == "accepted"

    # Cancel running Driver, expire its lease, and take over!
    c_rcpt = await port.cancel(drv, actor=ACTOR, scope="self", reason="cancel active wf")
    assert c_rcpt.state == "cancel_requested"
    await store._db.execute(
        f"UPDATE {store.t.sessions} SET owner_lease_until_ms=0 WHERE session_id=?", (drv.session_id,),
    )
    recovered = await port.recover(drv, node_id="n3", boot_id="b3")
    assert recovered is not None and recovered.execution_state == "queued"

    # Entire cancelled chain (orig AND wfc_1) settled as delivered/cancelled; node receipt drained!
    for ref in (orig_ref, wfc_ref):
        r = await port.get_result(ref, actor=ACTOR)
        assert r is not None and r.status == "cancelled" and r.cancel_by == "user"
    pending_nodes = await store._db.fetchall(
        f"SELECT id FROM {store.t.follow_up_queue} "
        "WHERE session_id=? AND kind='workflow_node_result' AND status='pending'",
        (drv.session_id,),
    )
    assert pending_nodes == []

    # Late user input remains pending/wake=1, Driver is queued with exact counts, discoverable & claimable!
    st = await _assert_session_counts_match_pending(store._db, store, drv.session_id)
    assert st["execution_state"] == "queued" and st["wake_pending_count"] == 1 and st["pending_count"] == 1
    assert drv.session_id in await port.discover_queued(limit=32)
    claimed_late = await port.claim(drv, node_id="n3", boot_id="b3")
    assert claimed_late is not None and claimed_late.input_refs == (late_rcpt.input_ref,)

    # Case 2: suspended + cancelled Driver keeps counts for remaining wake=1 input while staying idle until resumed
    _, drv_s, (leaf_s,) = await _setup_driver_and_leaf(port, "b4-susp")
    await port.final(leaf_s, status="completed", final_text="leaf-s-done")
    late_s = await port.accept(drv_s.target, "queued-before-suspend", message_id="b4-late-s", actor=ACTOR)
    await port.set_lifecycle(drv_s.target, "suspended", actor=ACTOR, inflight="abort", subtree="none")
    await store._db.execute(
        f"UPDATE {store.t.sessions} SET owner_lease_until_ms=0 WHERE session_id=?",
        (drv_s.target.session_id,),
    )
    rec_s = await port.recover(drv_s.target, node_id="n4", boot_id="b4")
    assert rec_s is not None and rec_s.execution_state == "idle"
    st_s = await _assert_session_counts_match_pending(store._db, store, drv_s.target.session_id)
    assert st_s["execution_state"] == "idle" and st_s["wake_pending_count"] == 1 and st_s["pending_count"] == 1
    resumed = await port.set_lifecycle(drv_s.target, "open", actor=ACTOR)
    assert resumed.need_wake is True and resumed.execution_state == "queued"
    claimed_s = await port.claim(drv_s.target, node_id="n4", boot_id="b4")
    assert claimed_s is not None and claimed_s.input_refs == (late_s.input_ref,)


async def test_b5_inprocess_wake0_receipt_does_not_wake_driver_waiter_after_final_commit(registry_store):
    """B5: In-process wake=0 leaf receipt arriving after Driver Final commits (before waiter unregisters) triggers 0 signals."""
    from power_loop.agent.runtime_store import RuntimeInputStore

    from .session_native_fixture import ACTOR

    store, _ = registry_store
    in_undeliv = asyncio.Event()
    release_undeliv = asyncio.Event()

    async def pause_on_undeliverable(*_args, **_kwargs):
        in_undeliv.set()
        await asyncio.wait_for(release_undeliv.wait(), 5)

    port = RuntimeInputStore(store, on_result_undeliverable=pause_on_undeliverable)
    main, drv_batch, (leaf_batch,) = await _setup_driver_and_leaf(port, "b5-wake0")
    drv_sid = drv_batch.target.session_id

    # Close Main only so Driver's Final commits and pauses inside post-commit RESULT_UNDELIVERABLE hook
    await port.set_lifecycle(main, "closed", actor=ACTOR, inflight="abort", subtree="none")
    waiter_q, waiter_sig, _ = port.register_driver_waiter(drv_sid)
    try:
        drv_task = asyncio.create_task(
            port.final(
                drv_batch, status="completed", final_text="drv-finished",
                drain_driver_node_results=True, settle_start_input=True, write_driver_message=True,
            ),
        )
        await asyncio.wait_for(in_undeliv.wait(), 5)

        # Driver is already idle in DB while its in-process waiter is still registered
        drv_mid = await store._db.fetchone(
            f"SELECT execution_state,wake_pending_count FROM {store.t.sessions} WHERE session_id=?",
            (drv_sid,),
        )
        assert dict(drv_mid) == {"execution_state": "idle", "wake_pending_count": 0}

        # Now same-process leaf finishes with cancel_by='subtree' (wake=0)
        await port.cancel(
            leaf_batch.target, actor=ACTOR, scope="self", reason="subtree stop", cancel_by="subtree",
        )
        leaf_fin = await port.final(
            leaf_batch, status="cancelled", final_text="", error="subtree stop", error_code="cancelled",
        )
        assert leaf_fin.result_activation is None
        assert leaf_fin.result_delivery_error is None

        # Receipt MUST be persisted in DB with wake=0, yet NEITHER in-memory waiter NOR scheduling wakes!
        rec_rows = await store._db.fetchall(
            f"SELECT wake,status,content_json FROM {store.t.follow_up_queue} "
            "WHERE session_id=? AND kind='workflow_node_result'",
            (drv_sid,),
        )
        assert len(rec_rows) == 1
        assert int(rec_rows[0]["wake"]) == 0 and rec_rows[0]["status"] == "pending"
        assert json.loads(rec_rows[0]["content_json"])["cancel_by"] == "subtree"
        assert waiter_q.empty() is True
        assert waiter_sig.is_set() is False
        st = await _assert_session_counts_match_pending(store._db, store, drv_sid)
        assert st["execution_state"] == "idle" and st["wake_pending_count"] == 0 and st["pending_count"] == 1

        release_undeliv.set()
        drv_fin = await drv_task
        assert drv_fin.result_delivery_error == "session_closed"
    finally:
        release_undeliv.set()
        port.unregister_driver_waiter(drv_sid)


async def test_b6_independent_driver_cascade_only_for_cancel_subtree_not_lifecycle_linked(registry_store):
    """B6: Identical tree compares set_lifecycle(subtree='linked') vs cancel(scope='subtree')."""
    from dataclasses import replace

    from power_loop.agent.runtime_store import ChildSpec, RuntimeInputStore

    from .session_native_fixture import ACTOR, DESCRIPTOR, target

    store, _ = registry_store
    port = RuntimeInputStore(store)

    async def build_tree(prefix: str):
        main = await target(port, f"{prefix}-main")
        drv_desc = replace(DESCRIPTOR, kind="workflow_driver")
        bg_desc = replace(DESCRIPTOR, kind="background")
        cm_top = await port.create_many(
            parent=main, actor=ACTOR,
            specs=[
                ChildSpec(
                    session_id=f"{prefix}-drv", message_id="m-drv", body={"wf": 1},
                    descriptor=drv_desc, result_kind="workflow_result", cancel_propagation="independent",
                ),
                ChildSpec(
                    session_id=f"{prefix}-linked", message_id="m-linked", body="linked",
                    descriptor=DESCRIPTOR, result_kind="agent_result", cancel_propagation="linked",
                ),
                ChildSpec(
                    session_id=f"{prefix}-indep-agent", message_id="m-ia", body="indep-agent",
                    descriptor=DESCRIPTOR, result_kind="agent_result", cancel_propagation="independent",
                ),
                ChildSpec(
                    session_id=f"{prefix}-indep-bg", message_id="m-bg", body="indep-bg",
                    descriptor=bg_desc, result_kind="background_result", cancel_propagation="independent",
                ),
            ],
            node_id="n1", boot_id="b1", can_execute=True, free_slots=4,
        )
        drv_batch = cm_top.preclaimed[0]
        leaf_desc = replace(DESCRIPTOR, config={**dict(DESCRIPTOR.config), "workflow_node": {"id": "L1"}})
        cm_leaf = await port.create_many(
            parent=drv_batch.target, actor=ACTOR,
            specs=[ChildSpec(
                session_id=f"{prefix}-leaf", message_id="m-leaf", body="leaf",
                descriptor=leaf_desc, result_kind="workflow_node_result", cancel_propagation="linked",
            )],
            node_id="n1", boot_id="b1", can_execute=True, free_slots=1,
            parent_owner_epoch=drv_batch.owner_epoch, parent_cancel_seq=drv_batch.cancel_seq,
        )
        return {
            "main": main,
            "driver": drv_batch.target,
            "leaf": cm_leaf.preclaimed[0].target,
            "linked": cm_top.preclaimed[1].target,
            "indep_agent": cm_top.preclaimed[2].target,
            "indep_bg": cm_top.preclaimed[3].target,
        }

    async def read_cancel_state(sid: str):
        row = await store._db.fetchone(
            f"SELECT execution_state,cancel_by,cancel_seq FROM {store.t.sessions} WHERE session_id=?",
            (sid,),
        )
        return dict(row)

    # Tree A: set_lifecycle('closed', inflight='abort', subtree='linked') follows strict dependency edges ONLY!
    t_life = await build_tree("b6-life")
    st_life = await port.set_lifecycle(
        t_life["main"], "closed", actor=ACTOR, inflight="abort", subtree="linked", reason="close linked only",
    )
    assert st_life.stopped_descendant_ids == (t_life["linked"].session_id,)
    assert await read_cancel_state(t_life["linked"].session_id) == {
        "execution_state": "running", "cancel_by": "subtree", "cancel_seq": 1,
    }
    for untouched in ("driver", "leaf", "indep_agent", "indep_bg"):
        assert await read_cancel_state(t_life[untouched].session_id) == {
            "execution_state": "running", "cancel_by": None, "cancel_seq": 0,
        }, untouched

    # Tree B: cancel(Main, scope='subtree') cascades into independent workflow_driver + its linked leaf,
    # while STILL sparing ordinary independent agent & background children!
    t_can = await build_tree("b6-cancel")
    await port.cancel(t_can["main"], actor=ACTOR, scope="subtree", reason="cancel conversation subtree")
    for stopped in ("driver", "leaf", "linked"):
        assert await read_cancel_state(t_can[stopped].session_id) == {
            "execution_state": "running", "cancel_by": "subtree", "cancel_seq": 1,
        }, stopped
    for untouched in ("indep_agent", "indep_bg"):
        assert await read_cancel_state(t_can[untouched].session_id) == {
            "execution_state": "running", "cancel_by": None, "cancel_seq": 0,
        }, untouched


async def test_b7_direct_leaf_chat_rebind_and_lease_expiry_route_to_live_root(registry_store):
    """B7: Human direct chat on workflow leaf honors live root_session_id after rebind and on lease-expiry takeover."""
    from dataclasses import replace

    from power_loop.agent.runtime_store import ChildSpec, RuntimeInputStore

    from .session_native_fixture import ACTOR, DESCRIPTOR, target

    store, _ = registry_store
    port = RuntimeInputStore(store)
    main_a = await target(port, "b7-main-a")
    main_b = await target(port, "b7-main-b")
    drv_desc = replace(DESCRIPTOR, kind="workflow_driver")
    cm_drv = await port.create_many(
        parent=main_a, actor=ACTOR,
        specs=[ChildSpec(
            session_id="b7-drv", message_id="b7-dm1", body={"wf": "start"},
            descriptor=drv_desc, result_kind="workflow_result",
        )],
        node_id="n1", boot_id="b1", can_execute=True, free_slots=1,
    )
    drv_batch = cm_drv.preclaimed[0]
    drv_sid = drv_batch.target.session_id

    # Long leaf session_id (> 64 chars) to also verify <=64 deterministic sys_chat_ notice ID & retry idempotency
    long_leaf_sid = "b7-long-workflow-leaf-session-id-" + ("x" * 52)
    assert len(long_leaf_sid) > 64
    leaf_desc = replace(DESCRIPTOR, config={**dict(DESCRIPTOR.config), "workflow_node": {"id": "L1"}})
    cm_leaf = await port.create_many(
        parent=drv_batch.target, actor=ACTOR,
        specs=[ChildSpec(
            session_id=long_leaf_sid, message_id="b7-lm1", body={"text": "turn1"},
            descriptor=leaf_desc, result_kind="workflow_node_result",
        )],
        node_id="n1", boot_id="b1", can_execute=True, free_slots=1,
        parent_owner_epoch=drv_batch.owner_epoch, parent_cancel_seq=drv_batch.cancel_seq,
    )
    leaf_b1 = cm_leaf.preclaimed[0]
    leaf = leaf_b1.target
    await port.final(leaf_b1, status="completed", final_text="turn1-leaf")
    await port.final(
        drv_batch, status="completed", final_text="wf-done",
        drain_driver_node_results=True, settle_start_input=True, write_driver_message=True,
    )
    # Settle Main A's workflow_result so Main A is idle and has 0 system_notices
    main_a_b = await port.claim(main_a, node_id="n1", boot_id="b1")
    assert main_a_b is not None
    await port.final(main_a_b, status="completed", final_text="main-a-done")

    drv_node_cnt_0 = len(await store._db.fetchall(
        f"SELECT id FROM {store.t.follow_up_queue} WHERE session_id=? AND kind='workflow_node_result'",
        (drv_sid,),
    ))
    assert drv_node_cnt_0 == 1

    # Scenario 1: Human direct chat (reply_to_session_id=NULL) claimed under Main A -> rebind Driver tree to Main B -> Final!
    await port.accept(leaf, "human direct chat turn 2", message_id="b7-human-2", actor=ACTOR)
    leaf_b2 = await port.claim(leaf, node_id="n1", boot_id="b1")
    assert leaf_b2 is not None and leaf_b2.send_index == 2
    assert leaf_b2.root_session_id == main_a.session_id and leaf_b2.reply_to_session_id is None

    rebind_res = await port.rebind_children(main_a, main_b, actor=ACTOR)
    assert drv_sid in rebind_res.rebound_children and long_leaf_sid in rebind_res.updated_descendants

    fin2 = await port.final(leaf_b2, status="completed", final_text="reply to human 2")
    assert fin2.result_delivery_error is None

    # Main A gets 0 notices; Main B (live root) gets exactly 1 system_notice(wake=0, len<=64)!
    notices_a = await store._db.fetchall(
        f"SELECT item_id FROM {store.t.follow_up_queue} WHERE session_id=? AND kind='system_notice'",
        (main_a.session_id,),
    )
    notices_b = await store._db.fetchall(
        f"SELECT item_id,wake,status,content_json FROM {store.t.follow_up_queue} "
        "WHERE session_id=? AND kind='system_notice' ORDER BY id ASC",
        (main_b.session_id,),
    )
    assert notices_a == []
    assert len(notices_b) == 1
    nid2 = notices_b[0]["item_id"]
    assert nid2.startswith("sys_chat_") and len(nid2) <= 64
    assert int(notices_b[0]["wake"]) == 0 and notices_b[0]["status"] == "pending"
    # Idempotent retry of same notice ID MUST NOT duplicate or bump counts
    async with store._db.transaction() as tx:
        dup_ok = await port._deliver_system_notice_tx(
            tx,
            target_session_id=main_b.session_id,
            tenant_id=ACTOR.tenant_id,
            principal_id=ACTOR.principal_id,
            item_id=nid2,
            body=json.loads(notices_b[0]["content_json"]),
        )
    assert dup_ok is False
    st_b = await _assert_session_counts_match_pending(store._db, store, main_b.session_id)
    assert st_b["execution_state"] == "idle" and st_b["wake_pending_count"] == 0 and st_b["pending_count"] == 1
    drv_after_1 = await _assert_session_counts_match_pending(store._db, store, drv_sid)
    assert drv_after_1["execution_state"] == "idle" and drv_after_1["wake_pending_count"] == 0
    assert len(await store._db.fetchall(
        f"SELECT id FROM {store.t.follow_up_queue} WHERE session_id=? AND kind='workflow_node_result'",
        (drv_sid,),
    )) == drv_node_cnt_0

    # Scenario 2: Human direct chat -> lease expires -> recover() MUST route notice(wake=0) to live root (Main B),
    # NOT send interrupted workflow_node_result to finished Driver!
    await port.accept(leaf, "human direct chat turn 3", message_id="b7-human-3", actor=ACTOR)
    leaf_b3 = await port.claim(leaf, node_id="n1", boot_id="b1")
    assert leaf_b3 is not None and leaf_b3.send_index == 3 and leaf_b3.reply_to_session_id is None
    await store._db.execute(
        f"UPDATE {store.t.sessions} SET owner_lease_until_ms=0 WHERE session_id=?", (long_leaf_sid,),
    )
    rec3 = await port.recover(leaf, node_id="n2", boot_id="b2")
    assert rec3 is not None and rec3.results[0].status == "interrupted"

    notices_a2 = await store._db.fetchall(
        f"SELECT item_id FROM {store.t.follow_up_queue} WHERE session_id=? AND kind='system_notice'",
        (main_a.session_id,),
    )
    notices_b2 = await store._db.fetchall(
        f"SELECT item_id,wake,status,content_json FROM {store.t.follow_up_queue} "
        "WHERE session_id=? AND kind='system_notice' ORDER BY id ASC",
        (main_b.session_id,),
    )
    assert notices_a2 == []
    assert len(notices_b2) == 2
    nid3 = notices_b2[1]["item_id"]
    assert nid3.startswith("sys_chat_") and len(nid3) <= 64 and nid3 != nid2
    assert int(notices_b2[1]["wake"]) == 0 and notices_b2[1]["status"] == "pending"
    # Retry idempotency on send_index=3 notice
    async with store._db.transaction() as tx:
        dup_ok3 = await port._deliver_system_notice_tx(
            tx,
            target_session_id=main_b.session_id,
            tenant_id=ACTOR.tenant_id,
            principal_id=ACTOR.principal_id,
            item_id=nid3,
            body=json.loads(notices_b2[1]["content_json"]),
        )
    assert dup_ok3 is False
    st_b2 = await _assert_session_counts_match_pending(store._db, store, main_b.session_id)
    assert st_b2["execution_state"] == "idle" and st_b2["wake_pending_count"] == 0 and st_b2["pending_count"] == 2
    drv_after_2 = await _assert_session_counts_match_pending(store._db, store, drv_sid)
    assert drv_after_2["execution_state"] == "idle" and drv_after_2["wake_pending_count"] == 0
    assert len(await store._db.fetchall(
        f"SELECT id FROM {store.t.follow_up_queue} WHERE session_id=? AND kind='workflow_node_result'",
        (drv_sid,),
    )) == drv_node_cnt_0


async def test_m1a6_create_many_5_leaves_exact_1_tx_4_sql_multi_value_inserts(registry_store):
    """M1a-6 / B8 evidence: 5-leaf create_many executes in exact 1 tx, 4 SQL with 3 multi-value INSERTs."""
    from dataclasses import replace

    from power_loop.agent.runtime_store import ChildSpec, RuntimeInputStore

    from .session_native_fixture import ACTOR, DESCRIPTOR, target
    from .test_send_sql_profile import Profiled

    store, _ = registry_store
    port = RuntimeInputStore(store)
    main = await target(port, "m1a6-main")
    drv_desc = replace(DESCRIPTOR, kind="workflow_driver")
    cm_drv = await port.create_many(
        parent=main, actor=ACTOR,
        specs=[ChildSpec(
            session_id="m1a6-drv", message_id="m1a6-dm1", body={"wf": "start"},
            descriptor=drv_desc, result_kind="workflow_result",
        )],
        node_id="n1", boot_id="b1", can_execute=True, free_slots=1,
    )
    drv_batch = cm_drv.preclaimed[0]
    specs = [
        ChildSpec(
            session_id=f"m1a6-leaf-{i}",
            message_id=f"m1a6-lm-{i}",
            body={"text": f"leaf-{i}"},
            descriptor=replace(DESCRIPTOR, config={**dict(DESCRIPTOR.config), "workflow_node": {"id": f"L{i}"}}),
            result_kind="workflow_node_result",
        )
        for i in range(5)
    ]
    original = store._db
    profiled = Profiled(original)
    store._db = profiled
    try:
        res = await port.create_many(
            parent=drv_batch.target, actor=ACTOR, specs=specs,
            node_id="n1", boot_id="b1", can_execute=True, free_slots=2,
            parent_owner_epoch=drv_batch.owner_epoch, parent_cancel_seq=drv_batch.cancel_seq,
        )
    finally:
        store._db = original

    assert len(res.preclaimed) == 2 and len(res.queued_targets) == 3
    assert profiled.transactions == 1, profiled.tx_sites
    assert len(profiled.records) == 4, [r["sql"] for r in profiled.records]
    sqls = [r["sql"] for r in profiled.records]
    assert sqls[0].lstrip().upper().startswith("SELECT ")
    assert f"INSERT INTO {store.t.sessions}" in sqls[1] and sqls[1].count("),(") == 4
    assert f"INSERT INTO {store.t.follow_up_queue}" in sqls[2] and sqls[2].count("),(") == 4
    assert f"INSERT INTO {store.t.messages}" in sqls[3] and sqls[3].count("),(") == 4


@pytest.mark.parametrize("mode", ["normal_user", "subtree"])
async def test_b3_1_merged_inputs_cancel_retry_preserves_delivery_and_usage(registry_store, mode):
    """B3.1: Merged 2-input leaf cancelled & finalized preserves usage (7 tokens) and idempotent retry on 2nd input."""
    from dataclasses import replace

    from power_loop.agent.runtime_store import ChildSpec, RuntimeInputStore

    from .session_native_fixture import ACTOR, DESCRIPTOR, target

    store, _ = registry_store
    port = RuntimeInputStore(store)
    main = await target(port, f"b31-main-{mode}")
    drv_desc = replace(DESCRIPTOR, kind="workflow_driver")
    cm_drv = await port.create_many(
        parent=main, actor=ACTOR,
        specs=[ChildSpec(
            session_id=f"b31-drv-{mode}", message_id=f"b31-dm-{mode}", body={"wf": "go"},
            descriptor=drv_desc, result_kind="workflow_result",
        )],
        node_id="n1", boot_id="b1", can_execute=True, free_slots=1,
    )
    drv_batch = cm_drv.preclaimed[0]
    drv_sid = drv_batch.target.session_id
    leaf_desc = replace(DESCRIPTOR, config={**dict(DESCRIPTOR.config), "workflow_node": {"id": "L1"}})
    leaf_sid = f"b31-leaf-{mode}"
    first_id = f"b31-first-{mode}"
    second_id = f"b31-second-{mode}"
    # Create leaf queued (free_slots=0) so we can accept a second routed input and merge both in send_index=1
    cm_leaf = await port.create_many(
        parent=drv_batch.target, actor=ACTOR,
        specs=[ChildSpec(
            session_id=leaf_sid, message_id=first_id, body={"text": "first"},
            descriptor=leaf_desc, result_kind="workflow_node_result",
        )],
        node_id="n1", boot_id="b1", can_execute=False, free_slots=0,
        parent_owner_epoch=drv_batch.owner_epoch, parent_cancel_seq=drv_batch.cancel_seq,
    )
    leaf = cm_leaf.queued_targets[0]
    await port.accept(
        leaf, {"text": "second"}, message_id=second_id, actor=ACTOR,
        reply_route={"target": drv_batch.target, "result_kind": "workflow_node_result"},
    )
    leaf_batch = await port.claim(leaf, node_id="n1", boot_id="b1")
    assert leaf_batch is not None and [r.message_id for r in leaf_batch.input_refs] == [first_id, second_id]
    first_ref, second_ref = leaf_batch.input_refs[0], leaf_batch.input_refs[1]

    if mode == "normal_user":
        await port.cancel(first_ref, actor=ACTOR, reason="user stop")
        expected_cby, expected_wake = "user", 1
    else:
        await port.cancel(drv_batch.target, actor=ACTOR, scope="subtree", reason="subtree stop")
        expected_cby, expected_wake = "subtree", 0

    known_usage = {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}
    fin = await port.final(
        leaf_batch, status="cancelled", final_text="",
        usage=known_usage, error="stopped", error_code="cancelled",
    )
    assert fin.result_delivery_error is None
    assert fin.results[0].usage == known_usage

    # Verify initial River receipt carries canonical source=first_ref, known usage=7, and expected cancel_by/wake
    receipts_before = await store._db.fetchall(
        f"SELECT item_id,wake,content_json FROM {store.t.follow_up_queue} "
        "WHERE session_id=? AND kind='workflow_node_result' ORDER BY id ASC",
        (drv_sid,),
    )
    assert len(receipts_before) == 1
    body_before = json.loads(receipts_before[0]["content_json"])
    assert int(receipts_before[0]["wake"]) == expected_wake
    assert body_before["cancel_by"] == expected_cby
    assert body_before["input_ref"]["message_id"] == first_id
    assert body_before["usage"] == known_usage

    for mid in (first_id, second_id):
        qrow = await store._db.fetchone(
            f"SELECT terminal_status,result_delivery_error,cancel_by FROM {store.t.follow_up_queue} "
            "WHERE session_id=? AND item_id=?",
            (leaf_sid, mid),
        )
        assert dict(qrow) == {
            "terminal_status": "cancelled",
            "result_delivery_error": None,
            "cancel_by": expected_cby,
        }

    # Retry cancel on the SECOND input (merged in the same send_index=1) -> must stay idempotent with NO request_conflict
    rcpt_second = await port.cancel(second_ref, actor=ACTOR, reason="retry cancel second")
    assert rcpt_second.state == "already_finished"
    rcpt_first = await port.cancel(first_ref, actor=ACTOR, reason="retry cancel first")
    assert rcpt_first.state == "already_finished"

    receipts_after = await store._db.fetchall(
        f"SELECT item_id,wake,content_json FROM {store.t.follow_up_queue} "
        "WHERE session_id=? AND kind='workflow_node_result' ORDER BY id ASC",
        (drv_sid,),
    )
    assert len(receipts_after) == 1
    assert receipts_after[0]["item_id"] == receipts_before[0]["item_id"]
    assert int(receipts_after[0]["wake"]) == expected_wake
    assert json.loads(receipts_after[0]["content_json"]) == body_before

    for mid in (first_id, second_id):
        qrow = await store._db.fetchone(
            f"SELECT terminal_status,result_delivery_error,cancel_by FROM {store.t.follow_up_queue} "
            "WHERE session_id=? AND item_id=?",
            (leaf_sid, mid),
        )
        assert qrow["result_delivery_error"] is None, f"unexpected delivery error for {mid}: {dict(qrow)}"
        res_loaded = await port.get_result(InputRef(store.store_id, leaf_sid, mid), actor=ACTOR)
        assert res_loaded is not None and res_loaded.status == "cancelled"
        assert res_loaded.usage == known_usage and res_loaded.cancel_by == expected_cby

    # Also verify redelivery path via second_ref (if receipt had been missing) reconstructs canonical first_ref + usage=7
    await store._db.execute(
        f"DELETE FROM {store.t.follow_up_queue} WHERE session_id=? AND item_id=?",
        (drv_sid, receipts_before[0]["item_id"]),
    )
    await store._db.execute(
        f"UPDATE {store.t.sessions} SET pending_count=0, wake_pending_count=0 WHERE session_id=?",
        (drv_sid,),
    )
    rcpt_redeliver = await port.cancel(second_ref, actor=ACTOR, reason="redeliver via second")
    assert rcpt_redeliver.state == "already_finished"
    receipts_redel = await store._db.fetchall(
        f"SELECT item_id,wake,content_json FROM {store.t.follow_up_queue} "
        "WHERE session_id=? AND kind='workflow_node_result' ORDER BY id ASC",
        (drv_sid,),
    )
    assert len(receipts_redel) == 1
    body_redel = json.loads(receipts_redel[0]["content_json"])
    assert receipts_redel[0]["item_id"] == receipts_before[0]["item_id"]
    assert int(receipts_redel[0]["wake"]) == expected_wake
    assert body_redel["input_ref"]["message_id"] == first_id
    assert body_redel["cancel_by"] == expected_cby
    assert body_redel["usage"] == known_usage


async def test_b3_2_closed_inflight_leaf_repeat_cancel_keeps_driver_unchanged(registry_store):
    """B3.2: Running leaf closed (cancelled/removed) + Final -> repeat cancel(InputRef) keeps Driver receipts=0 & wake=0."""
    from dataclasses import replace

    from power_loop.agent.runtime_store import ChildSpec, RuntimeInputStore

    from .session_native_fixture import ACTOR, DESCRIPTOR, target

    store, _ = registry_store
    port = RuntimeInputStore(store)
    main = await target(port, "b32-main")
    drv_desc = replace(DESCRIPTOR, kind="workflow_driver")
    cm_drv = await port.create_many(
        parent=main, actor=ACTOR,
        specs=[ChildSpec(
            session_id="b32-drv", message_id="b32-dm1", body={"wf": "go"},
            descriptor=drv_desc, result_kind="workflow_result",
        )],
        node_id="n1", boot_id="b1", can_execute=True, free_slots=1,
    )
    drv_batch = cm_drv.preclaimed[0]
    drv_sid = drv_batch.target.session_id

    leaf_desc = replace(DESCRIPTOR, config={**dict(DESCRIPTOR.config), "workflow_node": {"id": "L1"}})
    cm_leaf = await port.create_many(
        parent=drv_batch.target, actor=ACTOR,
        specs=[ChildSpec(
            session_id="b32-leaf", message_id="b32-lm1", body={"text": "work"},
            descriptor=leaf_desc, result_kind="workflow_node_result",
        )],
        node_id="n1", boot_id="b1", can_execute=True, free_slots=1,
        parent_owner_epoch=drv_batch.owner_epoch, parent_cancel_seq=drv_batch.cancel_seq,
    )
    leaf_batch = cm_leaf.preclaimed[0]
    leaf = leaf_batch.target
    leaf_orig_ref = leaf_batch.input_refs[0]

    # Put Driver to idle so any spurious wake=1 receipt would visibly flip Driver to queued and bump wake_pending_count
    await port.final(
        drv_batch, status="completed", final_text="drv-idle",
        drain_driver_node_results=True, settle_start_input=True, write_driver_message=False,
    )
    drv_before = await _assert_session_counts_match_pending(store._db, store, drv_sid)
    assert drv_before["execution_state"] == "idle" and drv_before["wake_pending_count"] == 0

    # Close the running leaf with inflight='abort', subtree='none', then let original executor call Final
    await port.set_lifecycle(leaf, "closed", actor=ACTOR, inflight="abort", subtree="none")
    fin = await port.final(leaf_batch, status="completed", final_text="late leaf answer")
    assert fin.results[0].status == "cancelled"
    assert fin.results[0].cancel_by == "removed"
    assert fin.result_delivery_error is None

    # Verify Driver has 0 receipts and wake_pending_count == 0 after closed Final
    drv_rcpts_0 = await store._db.fetchall(
        f"SELECT id FROM {store.t.follow_up_queue} WHERE session_id=? AND kind='workflow_node_result'",
        (drv_sid,),
    )
    assert drv_rcpts_0 == []
    drv_mid = await _assert_session_counts_match_pending(store._db, store, drv_sid)
    assert drv_mid["execution_state"] == "idle" and drv_mid["wake_pending_count"] == 0

    # Repeat cancel on the original leaf InputRef (and on the closed leaf target)
    rcpt_ref = await port.cancel(leaf_orig_ref, actor=ACTOR, reason="repeat cancel closed leaf")
    assert rcpt_ref.state == "already_finished"
    rcpt_ref2 = await port.cancel(leaf_orig_ref, actor=ACTOR, reason="repeat cancel closed leaf 2")
    assert rcpt_ref2.state == "already_finished"
    rcpt_target = await port.cancel(leaf, actor=ACTOR, reason="repeat cancel target")
    assert rcpt_target.state in {"already_finished", "idle"}

    # Driver's receipt count, wake_pending_count, and execution_state MUST remain completely unchanged!
    drv_rcpts_after = await store._db.fetchall(
        f"SELECT id FROM {store.t.follow_up_queue} WHERE session_id=? AND kind='workflow_node_result'",
        (drv_sid,),
    )
    assert drv_rcpts_after == []
    drv_after = await _assert_session_counts_match_pending(store._db, store, drv_sid)
    assert drv_after["execution_state"] == "idle"
    assert drv_after["wake_pending_count"] == 0
    assert drv_after["pending_count"] == 0

    leaf_q = await store._db.fetchone(
        f"SELECT terminal_status,cancel_by,result_delivery_error FROM {store.t.follow_up_queue} "
        "WHERE session_id=? AND item_id=?",
        (leaf.session_id, leaf_orig_ref.message_id),
    )
    assert dict(leaf_q) == {
        "terminal_status": "cancelled",
        "cancel_by": "removed",
        "result_delivery_error": None,
    }


async def test_b4_1_cancelled_driver_takeover_late_input_handled_by_new_worker_without_restart(registry_store):
    """B4.1: Cancelled Driver lease-takeover closes run journal; real Worker 2 settles late input with 0 new leaf calls."""
    from power_loop.agent.runtime_client import RuntimeClient
    from power_loop.agent.runtime_store import RuntimeInputStore

    store, _ = registry_store
    port = RuntimeInputStore(store)
    leaf_calls = 0

    class B41Model(Model):
        async def complete(self, req, **kwargs):
            nonlocal leaf_calls
            if req.system_prompt.startswith("L1"):
                leaf_calls += 1
                return LLMResponse(raw_text=f"leaf-call-{leaf_calls}")
            return LLMResponse(raw_text="main-ack")

    # Submit a real workflow through RuntimeClient (no worker claiming yet) so we can deterministically
    # simulate Worker 1 owning Driver (running, send_index=1), receiving a late plain input, getting
    # subtree-cancelled, and crashing before Final (lease expiry).
    from power_loop.agent.runtime_transport import MemoryMessageTransport

    transport = MemoryMessageTransport()
    client = RuntimeClient(store=store, transport=transport)
    rt_helper = RunningAgent(
        store, B41Model(), config=AgentLoopConfig(context_budget_tokens=10**9), capacity=2,
        transport=transport,
    )
    rt2 = None
    try:
        main_target = await client.create_session(rt_helper.descriptor, actor=rt_helper.actor)
        handle = await client.workflow(
            main_target, {"name": "b41-wf", "root": leaf("L1")},
            message_id="b41-start", actor=rt_helper.actor,
        )
        drv_target = handle.target
        drv_sid = drv_target.session_id

        # Confirm client.workflow created the real open run journal
        init_journal = await store.get_runtime_state(drv_sid, journal.run_key(handle.input_ref.message_id))
        assert init_journal is not None and init_journal["aggregate_delivered"] is False

        # Simulate Worker 1 claiming Driver Turn 1
        drv_batch = await port.claim(drv_target, node_id="worker-1", boot_id="boot-1")
        assert drv_batch is not None and drv_batch.send_index == 1

        # While Driver is running, accept a plain wake=1 input, then cancel(driver, scope='subtree')
        late_rcpt = await port.accept(drv_target, "late plain input", message_id="b41-late", actor=rt_helper.actor)
        assert late_rcpt.state == "accepted"
        await port.cancel(drv_target, actor=rt_helper.actor, scope="subtree", reason="user cancelled workflow")

        # Expire Worker 1's lease and let a brand-new Worker 2 take over via periodic_tick + normal execution!
        await store._db.execute(
            f"UPDATE {store.t.sessions} SET owner_lease_until_ms=0 WHERE session_id=?", (drv_sid,),
        )
        assert leaf_calls == 0

        rt2 = await RunningAgent(
            store, B41Model(), config=AgentLoopConfig(context_budget_tokens=10**9), capacity=2,
            transport=transport,
        ).start(target=main_target)
        # Worker 2 sweeps expired lease -> recovers Driver (closes run journal + settles start input + delivers
        # cancelled workflow_result to Main + keeps Driver queued for b41-late), then executes b41-late!
        await rt2.runtime.worker.periodic_tick(limit=32)

        late_res = await rt2.runtime.client.wait(late_rcpt.input_ref, actor=rt_helper.actor, deadline_ms=now_ms() + 10000)
        assert late_res.status == "completed"
        await rt2.drain()

        # 1) Zero new leaf model invocations (workflow was NEVER restarted!)
        assert leaf_calls == 0

        # 2) Original workflow handle and stored start input remain cancelled
        wf_res = await handle.wait(deadline_ms=now_ms() + 10000)
        assert wf_res.status == "cancelled"
        assert wf_res.cancel_by == "user"
        orig_stored = await port.get_result(handle.input_ref, actor=rt_helper.actor)
        assert orig_stored is not None and orig_stored.status == "cancelled" and orig_stored.cancel_by == "user"

        # 3) Run journal stays closed & cancelled
        saved_journal = await store.get_runtime_state(drv_sid, journal.run_key(handle.input_ref.message_id))
        assert saved_journal is not None
        assert saved_journal["status"] == "cancelled"
        assert saved_journal["aggregate_delivered"] is True

        # 4) Main received ONLY ONE workflow_result receipt (cancelled, wake=1)
        main_wf_rcpts = await store._db.fetchall(
            f"SELECT item_id,wake,content_json FROM {store.t.follow_up_queue} "
            "WHERE session_id=? AND kind='workflow_result' ORDER BY id ASC",
            (main_target.session_id,),
        )
        assert len(main_wf_rcpts) == 1
        assert int(main_wf_rcpts[0]["wake"]) == 1
        assert json.loads(main_wf_rcpts[0]["content_json"])["status"] == "cancelled"

        # 5) Driver counts match pending (idle, 0 pending)
        drv_st = await _assert_session_counts_match_pending(store._db, store, drv_sid)
        assert drv_st["execution_state"] == "idle" and drv_st["wake_pending_count"] == 0
    finally:
        if rt2 is not None:
            await rt2.aclose()
        await client.aclose()
        await rt_helper.loop.aclose()
        await rt_helper.runtime.client.aclose()


async def test_b4_2_first_takeover_with_merged_plain_input_keeps_start_chain_and_wait_resolves(registry_store):
    """B4.2: Turn 1 merges [start_input, late_plain] -> uncancelled takeover attaches wfc_* to start_input & wait() completes."""
    from power_loop.agent.runtime_client import RuntimeClient
    from power_loop.agent.runtime_store import RuntimeInputStore

    store, _ = registry_store
    port = RuntimeInputStore(store)

    class B42Model(Model):
        async def complete(self, req, **kwargs):
            if req.system_prompt.startswith("L1"):
                return LLMResponse(raw_text="L1-completed-after-takeover")
            return LLMResponse(raw_text="main-ack")

    # Submit workflow through RuntimeClient (before Worker starts) so we can queue a plain input before Turn 1 claim
    from power_loop.agent.runtime_transport import MemoryMessageTransport

    transport = MemoryMessageTransport()
    client = RuntimeClient(store=store, transport=transport)
    rt_helper = RunningAgent(
        store, B42Model(), config=AgentLoopConfig(context_budget_tokens=10**9), capacity=2,
        transport=transport,
    )
    rt_worker = None
    try:
        main_target = await client.create_session(rt_helper.descriptor, actor=rt_helper.actor)
        handle = await client.workflow(
            main_target, {"name": "b42-wf", "root": leaf("L1")},
            message_id="b42-start", actor=rt_helper.actor,
        )
        drv_target = handle.target
        drv_sid = drv_target.session_id
        start_id = handle.input_ref.message_id

        # Queue a plain user input BEFORE Driver's first claim so Turn 1 merges [start_id, b42-late]
        await port.accept(drv_target, "plain input before first claim", message_id="b42-late", actor=rt_helper.actor)
        batch1 = await port.claim(drv_target, node_id="worker-dead", boot_id="boot-dead")
        assert batch1 is not None and batch1.send_index == 1
        assert [r.message_id for r in batch1.input_refs] == [start_id, "b42-late"]

        # Lease expires WITHOUT cancel -> recover() must attach continuation chain to start_id (NOT b42-late)!
        await store._db.execute(
            f"UPDATE {store.t.sessions} SET owner_lease_until_ms=0 WHERE session_id=?", (drv_sid,),
        )
        rec = await port.recover(drv_target, node_id="worker-2", boot_id="boot-2")
        assert rec is not None

        start_row = await store._db.fetchone(
            f"SELECT status,terminal_status FROM {store.t.follow_up_queue} WHERE session_id=? AND item_id=?",
            (drv_sid, start_id),
        )
        assert start_row["status"] == "waiting_continuation" and start_row["terminal_status"] is None
        wfc_rows = await store._db.fetchall(
            f"SELECT item_id,continuation_of,continuation_index,status FROM {store.t.follow_up_queue} "
            "WHERE session_id=? AND kind='workflow_continue' ORDER BY id ASC",
            (drv_sid,),
        )
        assert len(wfc_rows) == 1
        assert wfc_rows[0]["continuation_of"] == start_id
        assert int(wfc_rows[0]["continuation_index"]) == 1
        assert wfc_rows[0]["status"] == "pending"

        # Start a real Worker to execute the recovered workflow to completion and verify handle.wait() resolves!
        rt_worker = await RunningAgent(
            store, B42Model(), config=AgentLoopConfig(context_budget_tokens=10**9), capacity=2,
            transport=transport,
        ).start(target=main_target)
        await rt_worker.runtime.worker.periodic_tick(limit=32)

        wf_res = await handle.wait(deadline_ms=now_ms() + 10000)
        assert wf_res.status == "completed"
        assert wf_res.final is not None and wf_res.final.text == "L1-completed-after-takeover"

        stored_start = await port.get_result(handle.input_ref, actor=rt_helper.actor)
        assert stored_start is not None and stored_start.status == "completed"
        await rt_worker.drain()
    finally:
        if rt_worker is not None:
            await rt_worker.aclose()
        await client.aclose()
        await rt_helper.loop.aclose()
        await rt_helper.runtime.client.aclose()


@pytest.mark.parametrize("mode", ["normal_user", "subtree"])
async def test_b3_1_real_capacity_exceeded_redelivery_via_urgent_preserves_original_source_and_usage(
    registry_store, mode,
):
    """B3.1 (R3): Pre-staged original + urgent routed input -> real capacity_exceeded -> drain -> retry urgent keeps original ref."""
    from dataclasses import replace

    from power_loop.agent.runtime_store import ChildSpec, RuntimeInputStore

    from .session_native_fixture import ACTOR, DESCRIPTOR, target

    store, _ = registry_store
    # Real hard_pending_limit=1 so Leaf 2's receipt genuinely fills Driver inbox and causes capacity_exceeded
    port = RuntimeInputStore(store, hard_pending_limit=1)
    main = await target(port, f"b31r3-main-{mode}")
    drv_desc = replace(DESCRIPTOR, kind="workflow_driver")
    cm_drv = await port.create_many(
        parent=main, actor=ACTOR,
        specs=[ChildSpec(
            session_id=f"b31r3-drv-{mode}", message_id=f"b31r3-dm-{mode}", body={"wf": "go"},
            descriptor=drv_desc, result_kind="workflow_result",
        )],
        node_id="n1", boot_id="b1", can_execute=True, free_slots=1,
    )
    drv_batch = cm_drv.preclaimed[0]
    drv_sid = drv_batch.target.session_id

    leaf1_sid = f"b31r3-leaf1-{mode}"
    leaf2_sid = f"b31r3-leaf2-{mode}"
    orig_id = f"b31r3-orig-{mode}"
    urgent_id = f"b31r3-urgent-{mode}"
    leaf1_desc = replace(DESCRIPTOR, config={**dict(DESCRIPTOR.config), "workflow_node": {"id": "L1"}})
    leaf2_desc = replace(DESCRIPTOR, config={**dict(DESCRIPTOR.config), "workflow_node": {"id": "L2"}})

    # Pre-stage Leaf 1's original input (queued, free_slots=0) and preclaim Leaf 2
    cm_l1 = await port.create_many(
        parent=drv_batch.target, actor=ACTOR,
        specs=[ChildSpec(
            session_id=leaf1_sid, message_id=orig_id, body={"text": "original"},
            descriptor=leaf1_desc, result_kind="workflow_node_result",
        )],
        node_id="n1", boot_id="b1", can_execute=False, free_slots=0,
        parent_owner_epoch=drv_batch.owner_epoch, parent_cancel_seq=drv_batch.cancel_seq,
    )
    leaf1 = cm_l1.queued_targets[0]
    cm_l2 = await port.create_many(
        parent=drv_batch.target, actor=ACTOR,
        specs=[ChildSpec(
            session_id=leaf2_sid, message_id=f"b31r3-l2m-{mode}", body={"text": "filler"},
            descriptor=leaf2_desc, result_kind="workflow_node_result",
        )],
        node_id="n1", boot_id="b1", can_execute=True, free_slots=1,
        parent_owner_epoch=drv_batch.owner_epoch, parent_cancel_seq=drv_batch.cancel_seq,
    )
    leaf2_batch = cm_l2.preclaimed[0]

    # Before first claim of Leaf 1, accept an URGENT input ALSO routed to Driver
    await port.accept(
        leaf1, {"text": "urgent"}, message_id=urgent_id, actor=ACTOR, priority="urgent",
        reply_route={"target": drv_batch.target, "result_kind": "workflow_node_result"},
    )
    leaf1_batch = await port.claim(leaf1, node_id="n1", boot_id="b1")
    assert leaf1_batch is not None and leaf1_batch.send_index == 1
    assert [r.message_id for r in leaf1_batch.input_refs] == [orig_id, urgent_id]
    orig_ref, urgent_ref = leaf1_batch.input_refs[0], leaf1_batch.input_refs[1]

    # Fill Driver's inbox genuinely to hard_pending_limit (1) using Leaf 2's completion receipt
    fin_l2 = await port.final(leaf2_batch, status="completed", final_text="leaf2-fills-driver")
    assert fin_l2.result_delivery_error is None

    if mode == "normal_user":
        await port.cancel(orig_ref, actor=ACTOR, reason="user stop")
        expected_cby, expected_wake = "user", 1
    else:
        await port.cancel(drv_batch.target, actor=ACTOR, scope="subtree", reason="subtree stop")
        expected_cby, expected_wake = "subtree", 0

    known_usage = {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}
    fin_l1 = await port.final(
        leaf1_batch, status="cancelled", final_text="",
        usage=known_usage, error="stopped", error_code="cancelled",
    )
    # First delivery genuinely rejected due to real capacity_exceeded (no DB deletion/tampering!)
    assert fin_l1.result_delivery_error == "capacity_exceeded"
    expected_msg_id = result_message_id(leaf1, orig_ref, 1, "workflow_node_result")
    for mid in (orig_id, urgent_id):
        qrow = await store._db.fetchone(
            f"SELECT terminal_status,result_delivery_error,cancel_by FROM {store.t.follow_up_queue} "
            "WHERE session_id=? AND item_id=?",
            (leaf1_sid, mid),
        )
        assert dict(qrow) == {
            "terminal_status": "cancelled",
            "result_delivery_error": "capacity_exceeded",
            "cancel_by": expected_cby,
        }

    # Genuinely drain Driver's inbox so pending_count returns to 0
    await port.final(
        drv_batch, status="completed", final_text="drv-drained",
        drain_driver_node_results=True, settle_start_input=True, write_driver_message=False,
    )
    drv_after_drain = await _assert_session_counts_match_pending(store._db, store, drv_sid)
    assert drv_after_drain["pending_count"] == 0

    # Retry cancel via the URGENT input -> must redeliver with ORIGINAL source_ref, exact same receipt ID,
    # preserved wake, cancel_by, and 7-token usage, and clear result_delivery_error on both inputs!
    rcpt_retry = await port.cancel(urgent_ref, actor=ACTOR, reason="retry cancel via urgent")
    assert rcpt_retry.state == "already_finished"

    l1_rcpts = await store._db.fetchall(
        f"SELECT item_id,wake,content_json FROM {store.t.follow_up_queue} "
        "WHERE session_id=? AND kind='workflow_node_result' AND item_id=?",
        (drv_sid, expected_msg_id),
    )
    assert len(l1_rcpts) == 1
    body = json.loads(l1_rcpts[0]["content_json"])
    assert l1_rcpts[0]["item_id"] == expected_msg_id
    assert int(l1_rcpts[0]["wake"]) == expected_wake
    assert body["input_ref"]["message_id"] == orig_id
    assert body["cancel_by"] == expected_cby
    assert body["usage"] == known_usage

    for mid in (orig_id, urgent_id):
        qrow = await store._db.fetchone(
            f"SELECT result_delivery_error FROM {store.t.follow_up_queue} WHERE session_id=? AND item_id=?",
            (leaf1_sid, mid),
        )
        assert qrow["result_delivery_error"] is None

    # Repeat retry on both urgent and original remains completely idempotent
    assert (await port.cancel(urgent_ref, actor=ACTOR, reason="retry2")).state == "already_finished"
    assert (await port.cancel(orig_ref, actor=ACTOR, reason="retry3")).state == "already_finished"
    assert len(await store._db.fetchall(
        f"SELECT id FROM {store.t.follow_up_queue} WHERE session_id=? AND item_id=?",
        (drv_sid, expected_msg_id),
    )) == 1


async def test_b4_1a_handle_cancel_before_first_claim_then_late_input_zero_leaf_calls(registry_store):
    """B4.1a: handle.cancel() before first claim closes run journal; subsequent plain input settles with 0 leaf calls."""
    from power_loop.agent.runtime_client import RuntimeClient
    from power_loop.agent.runtime_store import RuntimeInputStore
    from power_loop.agent.runtime_transport import MemoryMessageTransport

    store, _ = registry_store
    port = RuntimeInputStore(store)
    leaf_calls = 0

    class B41aModel(Model):
        async def complete(self, req, **kwargs):
            nonlocal leaf_calls
            if req.system_prompt.startswith("L1"):
                leaf_calls += 1
                return LLMResponse(raw_text=f"leaf-unexpected-{leaf_calls}")
            return LLMResponse(raw_text="main-ack")

    transport = MemoryMessageTransport()
    client = RuntimeClient(store=store, transport=transport)
    rt_helper = RunningAgent(
        store, B41aModel(), config=AgentLoopConfig(context_budget_tokens=10**9), capacity=2,
        transport=transport,
    )
    rt_worker = None
    try:
        main_target = await client.create_session(rt_helper.descriptor, actor=rt_helper.actor)
        handle = await client.workflow(
            main_target, {"name": "b41a-wf", "root": leaf("L1")},
            message_id="b41a-start", actor=rt_helper.actor,
        )
        drv_sid = handle.target.session_id

        # Public handle.cancel() BEFORE any Worker has claimed Driver
        cancel_rcpt = await handle.cancel(reason="cancel before first claim")
        assert cancel_rcpt.state == "voided"

        wf_res_before = await handle.wait(deadline_ms=now_ms() + 5000)
        assert wf_res_before.status == "cancelled" and wf_res_before.cancel_by == "user"
        j_before = await store.get_runtime_state(drv_sid, journal.run_key(handle.input_ref.message_id))
        assert j_before is not None and j_before["status"] == "cancelled" and j_before["aggregate_delivered"] is True

        # Now send a plain user input to Driver and start a real Worker to process it
        late_rcpt = await port.accept(handle.target, "plain input after cancel", message_id="b41a-late", actor=rt_helper.actor)
        assert late_rcpt.state == "accepted"

        rt_worker = await RunningAgent(
            store, B41aModel(), config=AgentLoopConfig(context_budget_tokens=10**9), capacity=2,
            transport=transport,
        ).start(target=main_target)
        await rt_worker.runtime.worker.periodic_tick(limit=32)

        late_res = await rt_worker.runtime.client.wait(late_rcpt.input_ref, actor=rt_helper.actor, deadline_ms=now_ms() + 10000)
        assert late_res.status == "completed"
        await rt_worker.drain()

        # Zero leaf LLM calls; workflow & journal remain cancelled; Main has ONLY ONE cancelled workflow_result
        assert leaf_calls == 0
        wf_res_after = await handle.wait(deadline_ms=now_ms() + 5000)
        assert wf_res_after.status == "cancelled" and wf_res_after.cancel_by == "user"
        j_after = await store.get_runtime_state(drv_sid, journal.run_key(handle.input_ref.message_id))
        assert j_after is not None and j_after["status"] == "cancelled" and j_after["aggregate_delivered"] is True

        main_wf_rcpts = await store._db.fetchall(
            f"SELECT item_id,wake,content_json FROM {store.t.follow_up_queue} "
            "WHERE session_id=? AND kind='workflow_result' ORDER BY id ASC",
            (main_target.session_id,),
        )
        assert len(main_wf_rcpts) == 1
        assert int(main_wf_rcpts[0]["wake"]) == 1
        assert json.loads(main_wf_rcpts[0]["content_json"])["status"] == "cancelled"
        assert json.loads(main_wf_rcpts[0]["content_json"])["input_ref"]["message_id"] == handle.input_ref.message_id
    finally:
        if rt_worker is not None:
            await rt_worker.aclose()
        await client.aclose()
        await rt_helper.loop.aclose()
        await rt_helper.runtime.client.aclose()


async def test_b4_1b_cancel_driver_subtree_before_first_claim_with_plain_input_single_workflow_receipt(
    registry_store,
):
    """B4.1b: [start_input, late_plain] queued -> cancel(driver, scope='subtree') voids both & emits ONLY 1 workflow_result."""
    from power_loop.agent.runtime_client import RuntimeClient
    from power_loop.agent.runtime_store import RuntimeInputStore
    from power_loop.agent.runtime_transport import MemoryMessageTransport

    store, _ = registry_store
    port = RuntimeInputStore(store)
    transport = MemoryMessageTransport()
    client = RuntimeClient(store=store, transport=transport)
    rt_helper = RunningAgent(
        store, Model(), config=AgentLoopConfig(context_budget_tokens=10**9), capacity=2,
        transport=transport,
    )
    try:
        main_target = await client.create_session(rt_helper.descriptor, actor=rt_helper.actor)
        handle = await client.workflow(
            main_target, {"name": "b41b-wf", "root": leaf("L1")},
            message_id="b41b-start", actor=rt_helper.actor,
        )
        drv_sid = handle.target.session_id
        late_rcpt = await port.accept(handle.target, "queued plain input", message_id="b41b-late", actor=rt_helper.actor)
        assert late_rcpt.state == "accepted"

        # Both start_input and b41b-late are pending; cancel the whole Driver subtree
        await port.cancel(handle.target, actor=rt_helper.actor, scope="subtree", reason="stop queued driver")

        # Both inputs read cancelled terminal status
        start_res = await port.get_result(handle.input_ref, actor=rt_helper.actor)
        late_res = await port.get_result(late_rcpt.input_ref, actor=rt_helper.actor)
        assert start_res is not None and start_res.status == "cancelled" and start_res.cancel_by == "user"
        assert late_res is not None and late_res.status == "cancelled" and late_res.cancel_by == "user"

        wf_res = await handle.wait(deadline_ms=now_ms() + 5000)
        assert wf_res.status == "cancelled" and wf_res.cancel_by == "user"
        j_row = await store.get_runtime_state(drv_sid, journal.run_key(handle.input_ref.message_id))
        assert j_row is not None and j_row["status"] == "cancelled" and j_row["aggregate_delivered"] is True

        # Main receives EXACTLY ONE workflow_result(cancelled, wake=1) sourced from start_input (NOT b41b-late)
        main_wf_rcpts = await store._db.fetchall(
            f"SELECT item_id,wake,content_json FROM {store.t.follow_up_queue} "
            "WHERE session_id=? AND kind='workflow_result' ORDER BY id ASC",
            (main_target.session_id,),
        )
        assert len(main_wf_rcpts) == 1
        assert int(main_wf_rcpts[0]["wake"]) == 1
        body = json.loads(main_wf_rcpts[0]["content_json"])
        assert body["status"] == "cancelled"
        assert body["cancel_by"] == "user"
        assert body["input_ref"]["message_id"] == handle.input_ref.message_id

        drv_st = await _assert_session_counts_match_pending(store._db, store, drv_sid)
        assert drv_st["execution_state"] == "idle" and drv_st["pending_count"] == 0 and drv_st["wake_pending_count"] == 0
    finally:
        await client.aclose()
        await rt_helper.loop.aclose()
        await rt_helper.runtime.client.aclose()


async def test_b4_2_explicit_resume_takeover_before_prepare_executes_node_and_delivers_receipt(registry_store):
    """B4.2 (R3): Explicit handle.resume() claimed & lease expires before _prepare_explicit -> Worker 2 runs node & resolves."""
    from power_loop.agent.runtime_client import RuntimeClient
    from power_loop.agent.runtime_store import RuntimeInputStore
    from power_loop.agent.runtime_transport import MemoryMessageTransport

    store, _ = registry_store
    port = RuntimeInputStore(store)
    leaf_prompts: list[str] = []

    class B42ResumeModel(Model):
        async def complete(self, req, **kwargs):
            if req.system_prompt.startswith("L1"):
                latest_user = next(
                    (str(m.get("content")) for m in reversed(req.messages) if m.get("role") == "user"),
                    "",
                )
                leaf_prompts.append(latest_user)
                return LLMResponse(raw_text=f"L1-run-{len(leaf_prompts)}")
            return LLMResponse(raw_text="main-ack")

    transport = MemoryMessageTransport()
    rt1 = await RunningAgent(
        store, B42ResumeModel(), config=AgentLoopConfig(context_budget_tokens=10**9), capacity=2,
        transport=transport,
    ).start()
    client = RuntimeClient(store=store, transport=transport)
    rt2 = None
    try:
        main_target = rt1.target
        handle = await rt1.runtime.client.workflow(
            main_target, {"name": "b42-resume-wf", "root": leaf("L1")},
            message_id="b42r-start", actor=rt1.actor,
        )
        first = await handle.wait(deadline_ms=now_ms() + 10000)
        assert first.status == "completed" and first.results["L1"].text == "L1-run-1"
        await rt1.drain()
        assert len(leaf_prompts) == 1

        # Shut down Worker 1 cleanly before submitting resume so we deterministically inject crash before _prepare_explicit
        await rt1.runtime.worker.shutdown(timeout_s=2, cancel_running=True)

        # Submit explicit resume through public handle.resume() (attached to idle client)
        from power_loop.workflow.runner import WorkflowRunHandle

        handle_on_client = WorkflowRunHandle(client, rt1.actor, handle.target, handle.input_ref)
        resume_handle = await handle_on_client.resume(
            node_id="L1", message="please recompute L1", message_id="b42r-resume-1",
        )
        drv_sid = handle.target.session_id

        # Simulate old Worker claiming resume input and crashing BEFORE _prepare_explicit writes new definition_ref
        claim_res = await port.claim(handle.target, node_id="worker-dead", boot_id="boot-dead")
        assert claim_res is not None
        assert [r.message_id for r in claim_res.input_refs] == [resume_handle.input_ref.message_id]
        saved_pre = await store.get_runtime_state(drv_sid, journal.run_key(handle.input_ref.message_id))
        assert saved_pre is not None and saved_pre["aggregate_delivered"] is True
        assert int(saved_pre.get("execution_revision", 0)) == 0

        # Expire old Worker's lease and let real Worker 2 take over via periodic_tick + normal execution!
        await store._db.execute(
            f"UPDATE {store.t.sessions} SET owner_lease_until_ms=0 WHERE session_id=?", (drv_sid,),
        )
        rt2 = await RunningAgent(
            store, B42ResumeModel(), config=AgentLoopConfig(context_budget_tokens=10**9), capacity=2,
            transport=transport,
        ).start(target=main_target)
        await rt2.runtime.worker.periodic_tick(limit=32)

        resumed = await resume_handle.wait(deadline_ms=now_ms() + 10000)
        await rt2.drain()

        # 1) Specified node L1 actually executed with the resume message
        assert len(leaf_prompts) == 2
        assert "please recompute L1" in leaf_prompts[1]

        # 2) Resume handle returns the real new node result & execution_revision incremented to 1
        assert resumed.status == "completed"
        assert "L1" in resumed.results and resumed.results["L1"].text == "L1-run-2"
        assert resumed.final is not None and resumed.final.text == "L1-run-2"
        saved_post = await store.get_runtime_state(drv_sid, journal.run_key(handle.input_ref.message_id))
        assert saved_post is not None and int(saved_post.get("execution_revision", 0)) == 1

        # 3) Main received a second workflow_result receipt for this explicit resume
        main_wf_rcpts = await store._db.fetchall(
            f"SELECT item_id,wake,content_json FROM {store.t.follow_up_queue} "
            "WHERE session_id=? AND kind='workflow_result' ORDER BY id ASC",
            (main_target.session_id,),
        )
        assert len(main_wf_rcpts) == 2
        res_body = json.loads(main_wf_rcpts[1]["content_json"])
        assert res_body["status"] == "completed"
        assert res_body["input_ref"]["message_id"] == resume_handle.input_ref.message_id
        assert res_body["final_text"] == "L1-run-2"

        # 4) Original handle's stored result remains completely unchanged!
        orig_after = await handle_on_client.wait(deadline_ms=now_ms() + 5000)
        assert orig_after.to_dict() == first.to_dict()
    finally:
        if rt2 is not None:
            await rt2.aclose()
        await client.aclose()
        await rt1.aclose()


async def test_r4_1_cancel_completed_old_handle_does_not_stop_running_resume(registry_store):
    """R4-1 (M1a-1): Cancelling a completed old handle while a resume is inside model call must not stop the resume."""
    store, _ = registry_store
    resume_in_model = asyncio.Event()
    release_resume = asyncio.Event()
    leaf_calls = 0

    class R41Model(Model):
        async def complete(self, req, **kwargs):
            nonlocal leaf_calls
            if req.system_prompt.startswith("L1"):
                leaf_calls += 1
                if leaf_calls == 2:
                    resume_in_model.set()
                    await release_resume.wait()
                    return LLMResponse(raw_text="L1-resumed-ok")
                return LLMResponse(raw_text="L1-first-ok")
            return LLMResponse(raw_text="main-ack")

    runner = await RunningAgent(
        store, R41Model(), config=AgentLoopConfig(context_budget_tokens=10**9), capacity=2,
    ).start()
    try:
        handle = await runner.runtime.client.workflow(
            runner.target, {"name": "r41-wf", "root": leaf("L1")},
            message_id="r41-first", actor=runner.actor,
        )
        first = await handle.wait(deadline_ms=now_ms() + 10000)
        assert first.status == "completed" and first.results["L1"].text == "L1-first-ok"
        await runner.drain()

        # Start explicit resume and wait until new leaf actually enters model.complete()
        resume_handle = await handle.resume(
            node_id="L1", message="recompute L1", message_id="r41-resume",
        )
        await asyncio.wait_for(resume_in_model.wait(), 5)

        # Cancel the ALREADY-COMPLETED old handle while resume is running!
        old_cancel_rcpt = await handle.cancel(reason="stale cancel on completed handle")
        assert old_cancel_rcpt.state == "already_finished"

        drv_row = await store._db.fetchone(
            f"SELECT cancel_seq,cancel_by FROM {store.t.sessions} WHERE session_id=?",
            (handle.target.session_id,),
        )
        assert int(drv_row["cancel_seq"]) == 0 and drv_row["cancel_by"] is None

        # Release the running resume -> MUST complete normally without being cancelled!
        release_resume.set()
        resumed = await resume_handle.wait(deadline_ms=now_ms() + 10000)
        await runner.drain()

        assert resumed.status == "completed"
        assert resumed.results["L1"].text == "L1-resumed-ok"
        assert (await handle.wait(deadline_ms=now_ms() + 5000)).to_dict() == first.to_dict()
        drv_after = await store._db.fetchone(
            f"SELECT cancel_seq,cancel_by FROM {store.t.sessions} WHERE session_id=?",
            (handle.target.session_id,),
        )
        assert int(drv_after["cancel_seq"]) == 0 and drv_after["cancel_by"] is None
    finally:
        release_resume.set()
        await runner.aclose()


async def test_r4_2_finished_driver_late_plain_input_cancel_takeover_no_duplicate_main_receipt(registry_store):
    """R4-2 (M1a-2/3): Finished Driver claims only a plain late input -> cancel + takeover emits 0 extra workflow_results."""
    from power_loop.agent.runtime_store import RuntimeInputStore

    store, _ = registry_store
    port = RuntimeInputStore(store)
    runner = await RunningAgent(
        store, Model(), config=AgentLoopConfig(context_budget_tokens=10**9), capacity=2,
    ).start()
    try:
        handle = await runner.runtime.client.workflow(
            runner.target, {"name": "r42-wf", "root": leaf("L1")},
            message_id="r42-start", actor=runner.actor,
        )
        first = await handle.wait(deadline_ms=now_ms() + 10000)
        assert first.status == "completed"
        await runner.drain()
        # Shut down worker cleanly before simulating old Worker claiming only a plain late input
        await runner.runtime.worker.shutdown(timeout_s=2, cancel_running=True)

        drv_sid = handle.target.session_id
        main_rcpts_before = await store._db.fetchall(
            f"SELECT item_id,wake,content_json FROM {store.t.follow_up_queue} "
            "WHERE session_id=? AND kind='workflow_result' ORDER BY id ASC",
            (runner.target.session_id,),
        )
        assert len(main_rcpts_before) == 1

        # Accept and claim ONLY a plain late user input on the already-finished Driver
        late_rcpt = await port.accept(handle.target, "late plain input", message_id="r42-late", actor=runner.actor)
        batch2 = await port.claim(handle.target, node_id="worker-old", boot_id="boot-old")
        assert batch2 is not None and [r.message_id for r in batch2.input_refs] == ["r42-late"]

        # Stop Driver, expire lease, and take over via recover()
        await port.cancel(handle.target, actor=runner.actor, scope="subtree", reason="stop late turn")
        await store._db.execute(
            f"UPDATE {store.t.sessions} SET owner_lease_until_ms=0 WHERE session_id=?", (drv_sid,),
        )
        rec = await port.recover(handle.target, node_id="worker-new", boot_id="boot-new")
        assert rec is not None and rec.result_ref is None and rec.result_activation is None

        # 1) Late plain input reads cancelled/user
        late_res = await port.get_result(late_rcpt.input_ref, actor=runner.actor)
        assert late_res is not None and late_res.status == "cancelled" and late_res.cancel_by == "user"

        # 2) Original run journal and original start result remain completed
        saved = await store.get_runtime_state(drv_sid, journal.run_key(handle.input_ref.message_id))
        assert saved is not None and saved["status"] == "completed" and saved["aggregate_delivered"] is True
        orig_res = await port.get_result(handle.input_ref, actor=runner.actor)
        assert orig_res is not None and orig_res.status == "completed"

        # 3) Main still has ONLY the original workflow_result receipt (NO duplicate!)
        main_rcpts_after = await store._db.fetchall(
            f"SELECT item_id,wake,content_json FROM {store.t.follow_up_queue} "
            "WHERE session_id=? AND kind='workflow_result' ORDER BY id ASC",
            (runner.target.session_id,),
        )
        assert len(main_rcpts_after) == 1
        assert main_rcpts_after[0]["item_id"] == main_rcpts_before[0]["item_id"]
        assert json.loads(main_rcpts_after[0]["content_json"])["status"] == "completed"
    finally:
        await runner.aclose()


async def test_r4_3_explicit_resume_capacity_exceeded_redelivery_keeps_resume_input_ref(registry_store):
    """R4-3 (M1a-3): Real capacity_exceeded on explicit resume -> drain Main -> retry cancel redelivers with resume input_ref."""
    store, _ = registry_store
    undeliverable_events: list[dict] = []

    def on_undeliverable(child_sid, parent_sid, result_dict, reason):
        undeliverable_events.append({
            "child_sid": child_sid,
            "parent_sid": parent_sid,
            "input_ref": InputRef(**result_dict["input_ref"]),
            "send_index": result_dict["send_index"],
            "reason": reason,
        })

    runner = await RunningAgent(
        store, Model(), config=AgentLoopConfig(context_budget_tokens=10**9), capacity=2,
    ).start()
    try:
        # Enforce real hard_pending_limit=1 and observe post-commit RESULT_UNDELIVERABLE callback
        runner.runtime.client.inputs.hard_pending_limit = 1
        runner.runtime.worker.inputs.hard_pending_limit = 1
        runner.runtime.worker.inputs.on_result_undeliverable = on_undeliverable

        handle = await runner.runtime.client.workflow(
            runner.target, {"name": "r43-wf", "root": leaf("L1")},
            message_id="r43-start", actor=runner.actor,
        )
        first = await handle.wait(deadline_ms=now_ms() + 10000)
        assert first.status == "completed"
        await runner.drain()

        # Fill Main's inbox to hard_pending_limit=1 with a real wake=0 input (Main stays idle)
        fill_rcpt = await runner.runtime.client.send(
            runner.target, "fill-main-wake0", message_id="r43-fill-w0",
            actor=runner.actor, wake=False,
        )
        assert fill_rcpt.state == "accepted"

        # Execute explicit resume -> first River delivery to Main hits REAL capacity_exceeded
        resume_handle = await handle.resume(
            node_id="L1", message="resume-with-full-main", message_id="r43-route-resume",
        )
        resumed = await resume_handle.wait(deadline_ms=now_ms() + 10000)
        assert resumed.status == "completed"
        await runner.drain()

        assert len(undeliverable_events) == 1
        assert undeliverable_events[0]["input_ref"] == resume_handle.input_ref
        assert undeliverable_events[0]["send_index"] == 2
        assert undeliverable_events[0]["reason"] == "capacity_exceeded"

        q_before = await store._db.fetchone(
            f"SELECT result_delivery_error,result_send_index FROM {store.t.follow_up_queue} "
            "WHERE session_id=? AND item_id=?",
            (handle.target.session_id, resume_handle.input_ref.message_id),
        )
        assert q_before["result_delivery_error"] == "capacity_exceeded"
        assert int(q_before["result_send_index"]) == 2

        # Drain Main's inbox legitimately by consuming the wake=0 filler (normal send drains all pending)
        await runner.runtime.client.inputs.cancel(
            fill_rcpt.input_ref, actor=runner.actor, reason="clear filler",
        )
        main_st = await _assert_session_counts_match_pending(store._db, store, runner.target.session_id)
        assert main_st["pending_count"] == 0

        # Retry cancel on the completed resume handle to trigger redelivery
        retry_rcpt = await resume_handle.cancel(reason="redeliver resume result")
        assert retry_rcpt.state == "already_finished"
        await runner.drain()

        # Delivery error cleared
        q_after = await store._db.fetchone(
            f"SELECT result_delivery_error FROM {store.t.follow_up_queue} WHERE session_id=? AND item_id=?",
            (handle.target.session_id, resume_handle.input_ref.message_id),
        )
        assert q_after["result_delivery_error"] is None

        # Verify Main's redelivered workflow_result matches first delivery's item_id, input_ref, send_index, wake
        expected_mid = result_message_id(handle.target, resume_handle.input_ref, 2, "workflow_result")
        redel_rows = await store._db.fetchall(
            f"SELECT item_id,wake,content_json FROM {store.t.follow_up_queue} "
            "WHERE session_id=? AND kind='workflow_result' AND item_id=?",
            (runner.target.session_id, expected_mid),
        )
        assert len(redel_rows) == 1
        assert redel_rows[0]["item_id"] == expected_mid
        assert int(redel_rows[0]["wake"]) == 1
        body = json.loads(redel_rows[0]["content_json"])
        assert body["input_ref"]["message_id"] == resume_handle.input_ref.message_id
        assert int(body["send_index"]) == 2
        assert body["status"] == "completed"
    finally:
        await runner.aclose()


async def test_r4_4_stop_queued_subtree_preserves_driver_counts_and_spares_independent_siblings(registry_store):
    """R4-4 (M1a-4): stop(main, subtree) with queued Driver + pre-staged Leaf keeps Driver counts exact & spares independent siblings."""
    from dataclasses import replace

    from power_loop.agent.runtime_store import ChildSpec, RuntimeInputStore

    from .session_native_fixture import ACTOR, DESCRIPTOR, target

    store, _ = registry_store
    port = RuntimeInputStore(store)
    main = await target(port, "r44-main")

    # 1) Create independent workflow_driver (claimed Turn 1)
    drv_desc = replace(DESCRIPTOR, kind="workflow_driver")
    cm_drv = await port.create_many(
        parent=main, actor=ACTOR,
        specs=[ChildSpec(
            session_id="r44-drv", message_id="r44-dm1", body={"wf": "go"},
            descriptor=drv_desc, result_kind="workflow_result",
            cancel_propagation="independent",
        )],
        node_id="n1", boot_id="b1", can_execute=True, free_slots=1,
    )
    drv_b = cm_drv.preclaimed[0]
    drv = drv_b.target

    # 2) Pre-stage a queued, un-claimed Leaf under Driver
    leaf_desc = replace(DESCRIPTOR, config={**dict(DESCRIPTOR.config), "workflow_node": {"id": "L1"}})
    cm_leaf = await port.create_many(
        parent=drv, actor=ACTOR,
        specs=[ChildSpec(
            session_id="r44-leaf", message_id="r44-lm1", body={"text": "prestaged"},
            descriptor=leaf_desc, result_kind="workflow_node_result",
            cancel_propagation="linked",
        )],
        node_id="n1", boot_id="b1", can_execute=False, free_slots=0,
        parent_owner_epoch=drv_b.owner_epoch, parent_cancel_seq=drv_b.cancel_seq,
    )
    assert len(cm_leaf.queued_targets) == 1

    # 3) Create ordinary independent agent and independent background child under Main (both running)
    indep_agent = await port.create_session(
        DESCRIPTOR, actor=ACTOR, parent=main, cancel_propagation="independent",
    )
    await port.accept(indep_agent, "work", message_id="r44-ia1", actor=ACTOR)
    assert await port.claim(indep_agent, node_id="n1", boot_id="b1") is not None

    bg_desc = replace(DESCRIPTOR, kind="background")
    indep_bg = await port.create_session(
        bg_desc, actor=ACTOR, parent=main, cancel_propagation="independent",
    )
    await port.accept(indep_bg, "bg-work", message_id="r44-bg1", actor=ACTOR)
    assert await port.claim(indep_bg, node_id="n1", boot_id="b1") is not None

    # 4) Expire Driver's lease and recover() so Driver enters queued state with wfc_1 pending
    await store._db.execute(
        f"UPDATE {store.t.sessions} SET owner_lease_until_ms=0 WHERE session_id=?", (drv.session_id,),
    )
    rec_drv = await port.recover(drv, node_id="n2", boot_id="b2")
    assert rec_drv is not None and rec_drv.execution_state == "queued"

    # 5) Stop Main with scope='subtree'
    await port.cancel(main, actor=ACTOR, scope="subtree", reason="stop whole conversation")

    # 6) Both levels of receipts carry cancel_by='subtree' & wake=0
    drv_node_rcpts = await store._db.fetchall(
        f"SELECT item_id,wake,status,content_bytes,content_json FROM {store.t.follow_up_queue} "
        "WHERE session_id=? AND kind='workflow_node_result' AND status='pending'",
        (drv.session_id,),
    )
    assert len(drv_node_rcpts) == 1
    assert int(drv_node_rcpts[0]["wake"]) == 0
    assert json.loads(drv_node_rcpts[0]["content_json"])["cancel_by"] == "subtree"

    main_wf_rcpts = await store._db.fetchall(
        f"SELECT item_id,wake,status,content_json FROM {store.t.follow_up_queue} "
        "WHERE session_id=? AND kind='workflow_result' AND status='pending'",
        (main.session_id,),
    )
    assert len(main_wf_rcpts) == 1
    assert int(main_wf_rcpts[0]["wake"]) == 0
    assert json.loads(main_wf_rcpts[0]["content_json"])["cancel_by"] == "subtree"

    # 7) Driver & Main pending_count / pending_bytes / wake_pending_count match actual pending rows EXACTLY!
    drv_st = await _assert_session_counts_match_pending(store._db, store, drv.session_id)
    assert drv_st["execution_state"] == "idle"
    assert drv_st["pending_count"] == 1
    assert drv_st["pending_bytes"] == int(drv_node_rcpts[0]["content_bytes"])
    assert drv_st["wake_pending_count"] == 0
    await _assert_session_counts_match_pending(store._db, store, main.session_id)

    # 8) Ordinary independent agent and independent background child remain completely untouched!
    for sid in (indep_agent.session_id, indep_bg.session_id):
        row = await store._db.fetchone(
            f"SELECT execution_state,cancel_by,cancel_seq FROM {store.t.sessions} WHERE session_id=?",
            (sid,),
        )
        assert dict(row) == {"execution_state": "running", "cancel_by": None, "cancel_seq": 0}, sid


async def test_r4_5_finished_leaf_mixed_claim_partial_consume_human_only_does_not_wake_driver(registry_store):
    """R4-5 (M1a-5): Finished leaf claims [human_direct, routed_other] and Final consumes ONLY human_direct."""
    from dataclasses import replace

    from power_loop.agent.runtime_store import ChildSpec, RuntimeInputStore

    from .session_native_fixture import ACTOR, DESCRIPTOR, target

    store, _ = registry_store
    port = RuntimeInputStore(store)
    main = await target(port, "r45-main")
    drv_desc = replace(DESCRIPTOR, kind="workflow_driver")
    cm_drv = await port.create_many(
        parent=main, actor=ACTOR,
        specs=[ChildSpec(
            session_id="r45-drv", message_id="r45-dm1", body={"wf": "start"},
            descriptor=drv_desc, result_kind="workflow_result",
        )],
        node_id="n1", boot_id="b1", can_execute=True, free_slots=1,
    )
    drv_b = cm_drv.preclaimed[0]
    drv_sid = drv_b.target.session_id

    leaf_desc = replace(DESCRIPTOR, config={**dict(DESCRIPTOR.config), "workflow_node": {"id": "L1"}})
    cm_leaf = await port.create_many(
        parent=drv_b.target, actor=ACTOR,
        specs=[ChildSpec(
            session_id="r45-leaf", message_id="r45-lm1", body={"text": "turn1"},
            descriptor=leaf_desc, result_kind="workflow_node_result",
        )],
        node_id="n1", boot_id="b1", can_execute=True, free_slots=1,
        parent_owner_epoch=drv_b.owner_epoch, parent_cancel_seq=drv_b.cancel_seq,
    )
    leaf_b1 = cm_leaf.preclaimed[0]
    leaf_t = leaf_b1.target
    await port.final(leaf_b1, status="completed", final_text="turn1-done")
    await port.final(
        drv_b, status="completed", final_text="wf-done",
        drain_driver_node_results=True, settle_start_input=True, write_driver_message=True,
    )
    # Settle Main's workflow_result so Main starts clean
    main_b = await port.claim(main, node_id="n1", boot_id="b1")
    assert main_b is not None
    await port.final(main_b, status="completed", final_text="main-ack")

    drv_rcpt_cnt_0 = len(await store._db.fetchall(
        f"SELECT id FROM {store.t.follow_up_queue} WHERE session_id=? AND kind='workflow_node_result'",
        (drv_sid,),
    ))
    assert drv_rcpt_cnt_0 == 1

    # Queue BOTH a pure human direct-chat input (reply_to=None) AND a separately routed input
    human_rcpt = await port.accept(leaf_t, "pure human chat", message_id="r45-pure-human", actor=ACTOR)
    routed_rcpt = await port.accept(
        leaf_t, "routed input", message_id="r45-routed-other", actor=ACTOR,
        reply_route={"target": main, "result_kind": "agent_result"},
    )
    leaf_b2 = await port.claim(leaf_t, node_id="n1", boot_id="b1")
    assert leaf_b2 is not None and [r.message_id for r in leaf_b2.input_refs] == [
        "r45-pure-human", "r45-routed-other",
    ]

    # Final consumes ONLY pure-human; routed-other must return to pending with its route intact
    fin2 = await port.final(
        leaf_b2, consumed=(human_rcpt.input_ref,), status="completed", final_text="reply to human only",
    )
    assert fin2.result_delivery_error is None and fin2.result_activation is None

    # 1) Unconsumed routed input returned to pending with original reply_to_session_id preserved
    other_row = await store._db.fetchone(
        f"SELECT status,reply_to_session_id,reply_result_kind FROM {store.t.follow_up_queue} "
        "WHERE session_id=? AND item_id=?",
        (leaf_t.session_id, routed_rcpt.input_ref.message_id),
    )
    assert dict(other_row) == {
        "status": "pending",
        "reply_to_session_id": main.session_id,
        "reply_result_kind": "agent_result",
    }
    leaf_st = await _assert_session_counts_match_pending(store._db, store, leaf_t.session_id)
    assert leaf_st["execution_state"] == "queued" and leaf_st["pending_count"] == 1 and leaf_st["wake_pending_count"] == 1

    # 2) Live root (Main) receives unique system_notice(wake=0) and 0 agent_result
    notices = await store._db.fetchall(
        f"SELECT item_id,wake,status FROM {store.t.follow_up_queue} "
        "WHERE session_id=? AND kind='system_notice'",
        (main.session_id,),
    )
    assert len(notices) == 1 and int(notices[0]["wake"]) == 0 and notices[0]["status"] == "pending"
    agent_res_rows = await store._db.fetchall(
        f"SELECT id FROM {store.t.follow_up_queue} WHERE session_id=? AND kind='agent_result'",
        (main.session_id,),
    )
    assert agent_res_rows == []

    # 3) Finished Driver's workflow_node_result count, wake_pending_count, and idle state remain unchanged!
    drv_rcpt_cnt_after = len(await store._db.fetchall(
        f"SELECT id FROM {store.t.follow_up_queue} WHERE session_id=? AND kind='workflow_node_result'",
        (drv_sid,),
    ))
    assert drv_rcpt_cnt_after == drv_rcpt_cnt_0
    drv_st = await _assert_session_counts_match_pending(store._db, store, drv_sid)
    assert drv_st["execution_state"] == "idle" and drv_st["wake_pending_count"] == 0


async def test_r4_6_ordinary_agent_per_turn_reply_route_recover_matches_m1c_baseline(registry_store):
    """R4-6 (M1c 45bc524 baseline): Ordinary agent child with result_route_json=NULL & per-turn reply_route does not emit agent_result on recover."""
    from power_loop.agent.runtime_store import RuntimeInputStore

    from .session_native_fixture import ACTOR, DESCRIPTOR, target

    store, _ = registry_store
    port = RuntimeInputStore(store)
    parent = await target(port, "r46-parent")
    child = await port.create_session(DESCRIPTOR, actor=ACTOR, parent=parent, result_route=None)

    srow = await store._db.fetchone(
        f"SELECT session_kind,result_route_json FROM {store.t.sessions} WHERE session_id=?",
        (child.session_id,),
    )
    assert srow["session_kind"] == "agent" and srow["result_route_json"] is None

    rcpt = await port.accept(
        child, "per-turn routed task", message_id="r46-inp1", actor=ACTOR,
        reply_route={"target": parent, "result_kind": "agent_result"},
    )
    batch = await port.claim(child, node_id="n1", boot_id="b1")
    assert batch is not None and batch.result_route is None and batch.reply_to_session_id == parent.session_id

    # Lease expires -> recover() must follow M1c 45bc524 baseline: settle child input as interrupted,
    # emit 0 agent_result rows to parent, and do NOT wake parent!
    await store._db.execute(
        f"UPDATE {store.t.sessions} SET owner_lease_until_ms=0 WHERE session_id=?", (child.session_id,),
    )
    rec = await port.recover(child, node_id="n2", boot_id="b2")
    assert rec is not None
    assert len(rec.results) == 1 and rec.results[0].status == "interrupted"
    assert rec.result_ref is None and rec.result_activation is None

    stored = await port.get_result(rcpt.input_ref, actor=ACTOR)
    assert stored is not None and stored.status == "interrupted"

    parent_results = await store._db.fetchall(
        f"SELECT id FROM {store.t.follow_up_queue} WHERE session_id=? AND kind='agent_result'",
        (parent.session_id,),
    )
    assert parent_results == []
    parent_st = await _assert_session_counts_match_pending(store._db, store, parent.session_id)
    assert parent_st["execution_state"] == "idle"
    assert parent_st["pending_count"] == 0
    assert parent_st["wake_pending_count"] == 0


async def test_r5_1_finished_driver_late_plain_input_recover_then_cancel_queued_wfc_no_main_wake(registry_store):
    """R5-1 (M1a-1/2): Finished Driver claims plain late input -> recover (queued wfc_*) -> cancel(late_ref) never wakes Main."""
    from power_loop.agent.runtime_store import RuntimeInputStore

    store, _ = registry_store
    port = RuntimeInputStore(store)
    runner = await RunningAgent(
        store, Model(), config=AgentLoopConfig(context_budget_tokens=10**9), capacity=2,
    ).start()
    try:
        handle = await runner.runtime.client.workflow(
            runner.target, {"name": "r51-wf", "root": leaf("L1")},
            message_id="r51-start", actor=runner.actor,
        )
        first = await handle.wait(deadline_ms=now_ms() + 10000)
        assert first.status == "completed"
        await runner.drain()
        await runner.runtime.worker.shutdown(timeout_s=2, cancel_running=True)

        drv_sid = handle.target.session_id
        main_sid = runner.target.session_id
        main_rcpts_before = await store._db.fetchall(
            f"SELECT item_id,wake,content_json FROM {store.t.follow_up_queue} "
            "WHERE session_id=? AND kind='workflow_result' ORDER BY id ASC",
            (main_sid,),
        )
        assert len(main_rcpts_before) == 1
        main_st_before = await _assert_session_counts_match_pending(store._db, store, main_sid)
        assert main_st_before["wake_pending_count"] == 0 and main_st_before["execution_state"] == "idle"

        # Old Worker claims ONLY a plain late user input, lease expires WITHOUT cancel -> recover queues wfc_*
        late_rcpt = await port.accept(
            handle.target, "late plain chain", message_id="r51-late-plain", actor=runner.actor,
        )
        batch2 = await port.claim(handle.target, node_id="worker-old", boot_id="boot-old")
        assert batch2 is not None and [r.message_id for r in batch2.input_refs] == ["r51-late-plain"]

        await store._db.execute(
            f"UPDATE {store.t.sessions} SET owner_lease_until_ms=0 WHERE session_id=?", (drv_sid,),
        )
        rec = await port.recover(handle.target, node_id="worker-new", boot_id="boot-new")
        assert rec is not None and rec.execution_state == "queued"

        late_row_mid = await store._db.fetchone(
            f"SELECT status,terminal_status FROM {store.t.follow_up_queue} WHERE session_id=? AND item_id=?",
            (drv_sid, "r51-late-plain"),
        )
        assert dict(late_row_mid) == {"status": "waiting_continuation", "terminal_status": None}
        wfc_rows = await store._db.fetchall(
            f"SELECT item_id,status,continuation_of FROM {store.t.follow_up_queue} "
            "WHERE session_id=? AND kind='workflow_continue' ORDER BY id ASC",
            (drv_sid,),
        )
        assert len(wfc_rows) == 1
        assert wfc_rows[0]["status"] == "pending" and wfc_rows[0]["continuation_of"] == "r51-late-plain"
        wfc_ref = InputRef(store.store_id, drv_sid, str(wfc_rows[0]["item_id"]))

        # BEFORE new Worker claims wfc_*, cancel by original late input ref
        cancel_rcpt = await port.cancel(late_rcpt.input_ref, actor=runner.actor, reason="cancel late chain")
        assert cancel_rcpt.state == "voided"

        # 1) Queued wfc_* voided as cancelled/user; both original late item and wfc_* read cancelled/user
        wfc_after = await store._db.fetchone(
            f"SELECT status,terminal_status,cancel_by,result_delivery_error FROM {store.t.follow_up_queue} "
            "WHERE session_id=? AND item_id=?",
            (drv_sid, wfc_ref.message_id),
        )
        assert dict(wfc_after) == {
            "status": "void",
            "terminal_status": "cancelled",
            "cancel_by": "user",
            "result_delivery_error": None,
        }
        for ref_chk in (late_rcpt.input_ref, wfc_ref):
            res_chk = await port.get_result(ref_chk, actor=runner.actor)
            assert res_chk is not None and res_chk.status == "cancelled" and res_chk.cancel_by == "user"

        # 2) Original workflow result & run journal remain completed
        saved = await store.get_runtime_state(drv_sid, journal.run_key(handle.input_ref.message_id))
        assert saved is not None and saved["status"] == "completed" and saved["aggregate_delivered"] is True
        orig_res = await port.get_result(handle.input_ref, actor=runner.actor)
        assert orig_res is not None and orig_res.status == "completed"

        # 3) Main STILL has ONLY the original 1 workflow_result and unchanged wake_pending_count == 0
        main_rcpts_after = await store._db.fetchall(
            f"SELECT item_id,wake,content_json FROM {store.t.follow_up_queue} "
            "WHERE session_id=? AND kind='workflow_result' ORDER BY id ASC",
            (main_sid,),
        )
        assert len(main_rcpts_after) == 1
        assert main_rcpts_after[0]["item_id"] == main_rcpts_before[0]["item_id"]
        main_st_after = await _assert_session_counts_match_pending(store._db, store, main_sid)
        assert main_st_after["wake_pending_count"] == main_st_before["wake_pending_count"]
        assert main_st_after["execution_state"] == "idle"

        # 4) Repeat cancel on both refs remains idempotent and never wakes Main
        assert (await port.cancel(late_rcpt.input_ref, actor=runner.actor, reason="retry1")).state == "already_finished"
        assert (await port.cancel(wfc_ref, actor=runner.actor, reason="retry2")).state == "already_finished"
        assert len(await store._db.fetchall(
            f"SELECT id FROM {store.t.follow_up_queue} WHERE session_id=? AND kind='workflow_result'",
            (main_sid,),
        )) == 1
        assert (await _assert_session_counts_match_pending(store._db, store, main_sid))["wake_pending_count"] == 0
    finally:
        await runner.aclose()


async def test_r5_2_queued_continuation_cancel_capacity_exceeded_redelivery_clears_entire_chain(registry_store):
    """R5-2 (M1a-1/3): Queued wfc_* cancel hits real capacity_exceeded -> drain Main -> retry clears BOTH orig & wfc_*."""
    from power_loop.agent.runtime_client import RuntimeClient
    from power_loop.agent.runtime_store import RuntimeInputStore
    from power_loop.agent.runtime_transport import MemoryMessageTransport

    store, _ = registry_store
    port = RuntimeInputStore(store, hard_pending_limit=1)
    transport = MemoryMessageTransport()
    client = RuntimeClient(store=store, transport=transport)
    client.inputs.hard_pending_limit = 1
    rt_helper = RunningAgent(
        store, Model(), config=AgentLoopConfig(context_budget_tokens=10**9), capacity=2,
        transport=transport,
    )
    try:
        main_target = await client.create_session(rt_helper.descriptor, actor=rt_helper.actor)
        handle = await client.workflow(
            main_target, {"name": "r52-wf", "root": leaf("L1")},
            message_id="r52-start", actor=rt_helper.actor,
        )
        drv_sid = handle.target.session_id
        main_sid = main_target.session_id

        # Claim Driver Turn 1, expire lease, and recover() so orig_start is waiting_continuation & wfc_* is pending
        batch1 = await port.claim(handle.target, node_id="worker-dead", boot_id="boot-dead")
        assert batch1 is not None and batch1.send_index == 1
        await store._db.execute(
            f"UPDATE {store.t.sessions} SET owner_lease_until_ms=0 WHERE session_id=?", (drv_sid,),
        )
        rec = await port.recover(handle.target, node_id="worker-2", boot_id="boot-2")
        assert rec is not None and rec.execution_state == "queued"

        wfc_rows = await store._db.fetchall(
            f"SELECT item_id,status,continuation_of FROM {store.t.follow_up_queue} "
            "WHERE session_id=? AND kind='workflow_continue' ORDER BY id ASC",
            (drv_sid,),
        )
        assert len(wfc_rows) == 1
        assert wfc_rows[0]["status"] == "pending" and wfc_rows[0]["continuation_of"] == handle.input_ref.message_id
        wfc_ref = InputRef(store.store_id, drv_sid, str(wfc_rows[0]["item_id"]))

        # Fill Main's inbox to real hard_pending_limit=1 using a wake=0 input
        fill_rcpt = await client.send(
            main_target, "fill-main-wake0", message_id="r52-fill-w0",
            actor=rt_helper.actor, wake=False,
        )
        assert fill_rcpt.state == "accepted"

        # Cancel by original handle -> first River delivery genuinely hits capacity_exceeded!
        cancel_rcpt = await handle.cancel(reason="cancel queued continuation")
        assert cancel_rcpt.state == "voided"

        for ref_chk in (handle.input_ref, wfc_ref):
            res_err = await client.wait(ref_chk, actor=rt_helper.actor, deadline_ms=now_ms() + 5000)
            assert res_err.status == "cancelled"
            assert res_err.cancel_by == "user"
            assert res_err.result_delivery_error == "capacity_exceeded"
        assert await store._db.fetchall(
            f"SELECT id FROM {store.t.follow_up_queue} WHERE session_id=? AND kind='workflow_result'",
            (main_sid,),
        ) == []

        # Clear the wake=0 filler on Main, then retry cancel on the ORIGINAL handle to trigger redelivery
        await port.cancel(fill_rcpt.input_ref, actor=rt_helper.actor, reason="clear filler")
        assert (await _assert_session_counts_match_pending(store._db, store, main_sid))["pending_count"] == 0

        retry1 = await handle.cancel(reason="redeliver queued continuation cancel")
        assert retry1.state == "already_finished"

        # 1) Main received EXACTLY ONE workflow_result(cancelled, wake=1) with original receipt ID, orig input_ref, send_index=None
        expected_mid = result_message_id(handle.target, handle.input_ref, None, "workflow_result")
        main_rcpts = await store._db.fetchall(
            f"SELECT item_id,wake,content_json FROM {store.t.follow_up_queue} "
            "WHERE session_id=? AND kind='workflow_result' ORDER BY id ASC",
            (main_sid,),
        )
        assert len(main_rcpts) == 1
        assert main_rcpts[0]["item_id"] == expected_mid
        assert int(main_rcpts[0]["wake"]) == 1
        body = json.loads(main_rcpts[0]["content_json"])
        assert body["status"] == "cancelled"
        assert body["cancel_by"] == "user"
        assert body["input_ref"]["message_id"] == handle.input_ref.message_id
        assert body["send_index"] is None

        # 2) BOTH original handle AND wfc_* now have result_delivery_error=None and cancel_by='user'
        for ref_chk in (handle.input_ref, wfc_ref):
            res_ok = await client.wait(ref_chk, actor=rt_helper.actor, deadline_ms=now_ms() + 5000)
            assert res_ok.status == "cancelled"
            assert res_ok.cancel_by == "user"
            assert res_ok.result_delivery_error is None
            qrow = await store._db.fetchone(
                f"SELECT result_delivery_error,cancel_by FROM {store.t.follow_up_queue} "
                "WHERE session_id=? AND item_id=?",
                (drv_sid, ref_chk.message_id),
            )
            assert dict(qrow) == {"result_delivery_error": None, "cancel_by": "user"}

        # 3) Repeat retries keep receipt unique and BOTH handles clean
        assert (await handle.cancel(reason="retry-again-orig")).state == "already_finished"
        assert (await port.cancel(wfc_ref, actor=rt_helper.actor, reason="retry-again-wfc")).state == "already_finished"
        assert len(await store._db.fetchall(
            f"SELECT id FROM {store.t.follow_up_queue} WHERE session_id=? AND kind='workflow_result'",
            (main_sid,),
        )) == 1
        for ref_chk in (handle.input_ref, wfc_ref):
            res_ok2 = await client.wait(ref_chk, actor=rt_helper.actor, deadline_ms=now_ms() + 5000)
            assert res_ok2.result_delivery_error is None and res_ok2.cancel_by == "user"
    finally:
        await client.aclose()
        await rt_helper.loop.aclose()
        await rt_helper.runtime.client.aclose()


async def test_stage2_b1_mixed_claim_late_continuation_and_explicit_resume(registry_store):
    """Stage2-B1: Finished Driver's plain late input recovers to wfc_late, merged with handle.resume() in 1 claim."""
    from power_loop.agent.runtime_client import RuntimeClient
    from power_loop.agent.runtime_store import RuntimeInputStore
    from power_loop.agent.runtime_transport import MemoryMessageTransport
    from power_loop.workflow.runner import WorkflowRunHandle

    store, _ = registry_store
    port = RuntimeInputStore(store)
    leaf_calls = 0

    class B1MixedModel(Model):
        async def complete(self, req, **kwargs):
            nonlocal leaf_calls
            if req.system_prompt.startswith("A"):
                leaf_calls += 1
                return LLMResponse(raw_text=f"A-{leaf_calls}")
            return LLMResponse(raw_text="main-ack")

    transport = MemoryMessageTransport()
    rt1 = await RunningAgent(
        store, B1MixedModel(), config=AgentLoopConfig(context_budget_tokens=10**9), capacity=2,
        transport=transport,
    ).start()
    client = RuntimeClient(store=store, transport=transport)
    rt2 = None
    try:
        main_target = rt1.target
        handle = await rt1.runtime.client.workflow(
            main_target, {"name": "s2-b1-wf", "root": leaf("A")},
            message_id="s2-b1-start", actor=rt1.actor,
        )
        first = await handle.wait(deadline_ms=now_ms() + 10000)
        assert first.status == "completed" and first.results["A"].text == "A-1"
        await rt1.drain()
        assert leaf_calls == 1
        await rt1.runtime.worker.shutdown(timeout_s=2, cancel_running=True)

        drv_sid = handle.target.session_id
        # 1) Plain late input claimed by old Worker -> lease expires -> recover() queues wfc_late (BEFORE resume!)
        late_rcpt = await port.accept(handle.target, "plain late msg", message_id="s2-b1-late", actor=rt1.actor)
        batch2 = await port.claim(handle.target, node_id="worker-dead", boot_id="boot-dead")
        assert batch2 is not None and [r.message_id for r in batch2.input_refs] == ["s2-b1-late"]
        await store._db.execute(
            f"UPDATE {store.t.sessions} SET owner_lease_until_ms=0 WHERE session_id=?", (drv_sid,),
        )
        rec = await port.recover(handle.target, node_id="worker-2", boot_id="boot-2")
        assert rec is not None and rec.execution_state == "queued"

        wfc_row = await store._db.fetchone(
            f"SELECT item_id,status,continuation_of FROM {store.t.follow_up_queue} "
            "WHERE session_id=? AND kind='workflow_continue' ORDER BY id DESC LIMIT 1",
            (drv_sid,),
        )
        assert wfc_row is not None and wfc_row["status"] == "pending" and wfc_row["continuation_of"] == "s2-b1-late"
        wfc_late_ref = InputRef(store.store_id, drv_sid, str(wfc_row["item_id"]))

        # 2) Submit explicit handle.resume() AFTER wfc_late is already pending, BEFORE new Worker claims!
        handle_on_client = WorkflowRunHandle(client, rt1.actor, handle.target, handle.input_ref)
        resume_handle = await handle_on_client.resume(
            node_id="A", message="rerun A for B1", message_id="s2-b1-resume",
        )

        # 3) Start real Worker 2 -> claims BOTH [wfc_late, resume_input] in send_index=3!
        rt2 = await RunningAgent(
            store, B1MixedModel(), config=AgentLoopConfig(context_budget_tokens=10**9), capacity=2,
            transport=transport,
        ).start(target=main_target)
        await rt2.runtime.worker.periodic_tick(limit=32)

        resumed = await resume_handle.wait(deadline_ms=now_ms() + 10000)
        await rt2.drain()

        # Verify explicit resume ACTUALLY executed target leaf A and returned full result!
        assert leaf_calls == 2
        assert resumed.status == "completed"
        assert "A" in resumed.results and resumed.results["A"].text == "A-2"
        assert resumed.final is not None and resumed.final.text == "A-2"

        # Verify the late continuation chain (both s2-b1-late and wfc_late) settled cleanly as completed!
        for ref_chk in (late_rcpt.input_ref, wfc_late_ref):
            qrow = await store._db.fetchone(
                f"SELECT status,terminal_status FROM {store.t.follow_up_queue} "
                "WHERE session_id=? AND item_id=?",
                (drv_sid, ref_chk.message_id),
            )
            assert qrow["status"] == "delivered" and qrow["terminal_status"] == "completed"
            res_chk = await port.get_result(ref_chk, actor=rt1.actor)
            assert res_chk is not None and res_chk.status == "completed"

        # Verify Main received EXACTLY ONE new workflow_result pointing to resume_handle.input_ref (2 total)
        main_rcpts = await store._db.fetchall(
            f"SELECT item_id,wake,content_json FROM {store.t.follow_up_queue} "
            "WHERE session_id=? AND kind='workflow_result' ORDER BY id ASC",
            (main_target.session_id,),
        )
        assert len(main_rcpts) == 2
        body2 = json.loads(main_rcpts[1]["content_json"])
        assert body2["status"] == "completed"
        assert body2["input_ref"]["message_id"] == resume_handle.input_ref.message_id
        assert body2["final_text"] == "A-2"

        # Original handle result untouched
        assert (await handle_on_client.wait(deadline_ms=now_ms() + 5000)).to_dict() == first.to_dict()
    finally:
        if rt2 is not None:
            await rt2.aclose()
        await client.aclose()
        await rt1.aclose()


async def test_stage2_b2_inprocess_slow_leaf_zero_extra_db_reads_and_remote_delivery(registry_store):
    """Stage2-B2: Single-Worker slow leaf (>3x250ms) causes 0 poll/get_result DB reads; remote delivery still works."""
    from .test_send_sql_profile import Profiled

    store, connect = registry_store
    leaf_entered = asyncio.Event()
    release_leaf = asyncio.Event()

    class SlowLeafModel(Model):
        async def complete(self, req, **kwargs):
            if req.system_prompt.startswith("SLOW"):
                leaf_entered.set()
                await release_leaf.wait()
                return LLMResponse(raw_text="slow-leaf-done")
            return LLMResponse(raw_text="main-ack")

    # Part A: Single Worker, capacity=1, same Store — Driver & slow leaf in-process
    orig_db = store._db
    prof = Profiled(orig_db)
    store._db = prof
    runner = await RunningAgent(
        store, SlowLeafModel(), config=AgentLoopConfig(context_budget_tokens=10**9), capacity=1,
    ).start()
    try:
        handle = await runner.runtime.client.workflow(
            runner.target, {"name": "s2-b2-slow", "root": leaf("SLOW")},
            message_id="s2-b2-slow", actor=runner.actor,
        )
        await asyncio.wait_for(leaf_entered.wait(), 5)
        # Wait a short beat for node-journal write to settle, then clear profiler and sleep across 3x250ms!
        await asyncio.sleep(0.05)
        prof.records.clear()
        await asyncio.sleep(0.85)

        poll_reads = [
            r for r in prof.records
            if any("poll_pending_driver_results" in f or "poll_driver_results_for_send" in f for f in r["chain"])
        ]
        get_result_reads = [
            r for r in prof.records
            if any("get_result" in f or "_result_state" in f for f in r["chain"])
        ]
        assert poll_reads == [], f"Unexpected in-process Driver poll SQL during slow leaf: {poll_reads}"
        assert get_result_reads == [], f"Unexpected in-process leaf get_result SQL during slow leaf: {get_result_reads}"

        release_leaf.set()
        res = await handle.wait(deadline_ms=now_ms() + 10000)
        assert res.status == "completed" and res.final.text == "slow-leaf-done"
        await runner.drain()
    finally:
        release_leaf.set()
        await runner.aclose()
        store._db = orig_db

    # Part B: Remote leaf on a separate Store/Worker (shared_execution=True) still delivers cleanly!
    from .test_send_sql_profile import _cross_process_workflow_cluster

    remote_entered = asyncio.Event()
    release_remote = asyncio.Event()

    class RemoteMainModel(Model):
        async def complete(self, req, **kwargs):
            return LLMResponse(raw_text="main-remote-ack")

    class RemoteLeafModel(Model):
        async def complete(self, req, **kwargs):
            remote_entered.set()
            await release_remote.wait()
            return LLMResponse(raw_text="remote-slow-done")

    async with _cross_process_workflow_cluster(
        store, connect, RemoteMainModel(), RemoteLeafModel(), leaf_capacity=1,
    ) as (rt_a, rt_b, main_target, actor, desc, _master):
        h_rem = await rt_a.client.workflow(
            main_target,
            {
                "name": "s2-b2-remote",
                "root": {
                    "type": "agent", "id": "R1", "input": "work",
                    "spec": {"name": "R1", "system_prompt": "REMOTE", "tools": []},
                },
            },
            message_id="s2-b2-remote", actor=actor, descriptor=desc,
        )
        await asyncio.wait_for(remote_entered.wait(), 5)
        await asyncio.sleep(0.35)
        release_remote.set()
        r_res = await h_rem.wait(deadline_ms=now_ms() + 10000)
        assert r_res.status == "completed" and r_res.final.text == "remote-slow-done"
        async with asyncio.timeout(5):
            while rt_a.worker._active or rt_b.worker._active:
                await asyncio.sleep(0.01)


async def test_stage2_b3_concurrent_close_sessions_if_idle_vs_leaf_final_returns_busy_no_deadlock(registry_store):
    """Stage2-B3: Concurrent close_sessions_if_idle([driver]) while leaf Final holds leaf lock returns busy (0 deadlock)."""
    from dataclasses import replace

    from power_loop.agent.runtime_store import ChildSpec, RuntimeInputStore

    from .session_native_fixture import ACTOR, DESCRIPTOR, target

    store_a, connect = registry_store
    port_a = RuntimeInputStore(store_a)
    main = await target(port_a, "s2b3-main")
    drv_desc = replace(DESCRIPTOR, kind="workflow_driver")
    cm_drv = await port_a.create_many(
        parent=main, actor=ACTOR,
        specs=[ChildSpec(
            session_id="s2b3-drv", message_id="s2b3-dm1", body={"wf": "go"},
            descriptor=drv_desc, result_kind="workflow_result",
        )],
        node_id="n1", boot_id="b1", can_execute=True, free_slots=1,
    )
    drv_b = cm_drv.preclaimed[0]
    drv_sid = drv_b.target.session_id
    leaf_desc = replace(DESCRIPTOR, config={**dict(DESCRIPTOR.config), "workflow_node": {"id": "L1"}})
    cm_leaf = await port_a.create_many(
        parent=drv_b.target, actor=ACTOR,
        specs=[ChildSpec(
            session_id="s2b3-leaf", message_id="s2b3-lm1", body={"text": "work"},
            descriptor=leaf_desc, result_kind="workflow_node_result",
        )],
        node_id="n1", boot_id="b1", can_execute=True, free_slots=1,
        parent_owner_epoch=drv_b.owner_epoch, parent_cancel_seq=drv_b.cancel_seq,
    )
    leaf_b = cm_leaf.preclaimed[0]

    # Connection 1: pause leaf Final right AFTER locking/updating leaf row and BEFORE River UPDATE on Driver!
    leaf_locked_ev = asyncio.Event()
    release_river_ev = asyncio.Event()
    orig_river = port_a._river_deliver_tx

    async def paused_river(tx, *args, **kwargs):
        leaf_locked_ev.set()
        await release_river_ev.wait()
        return await orig_river(tx, *args, **kwargs)

    port_a._river_deliver_tx = paused_river

    store_b = await connect()
    try:
        leaf_task = asyncio.create_task(
            port_a.final(leaf_b, status="completed", final_text="leaf-done"),
        )
        await asyncio.wait_for(leaf_locked_ev.wait(), 5)

        # Connection 2: call close_sessions_if_idle([driver]) concurrently while Connection 1 holds leaf lock!
        close_task = asyncio.create_task(store_b.close_sessions_if_idle([drv_sid]))
        # Give close_task a brief beat to run concurrently before releasing leaf_task
        await asyncio.sleep(0.05)
        release_river_ev.set()

        close_status, leaf_fin = await asyncio.wait_for(
            asyncio.gather(close_task, leaf_task), timeout=10,
        )
        assert close_status == "busy"
        assert leaf_fin.result_delivery_error is None
    finally:
        release_river_ev.set()
        port_a._river_deliver_tx = orig_river
        await store_b.close()

    # Probe 9 closure & 2-page retention with REAL residual halt wake=0 workflow_node_result (NO manual delete!)
    cm_leaf2 = await port_a.create_many(
        parent=drv_b.target, actor=ACTOR,
        specs=[ChildSpec(
            session_id="s2b3-leaf-slow", message_id="s2b3-lm2", body={"text": "slow"},
            descriptor=replace(DESCRIPTOR, config={**dict(DESCRIPTOR.config), "workflow_node": {"id": "L2"}}),
            result_kind="workflow_node_result",
        )],
        node_id="n1", boot_id="b1", can_execute=True, free_slots=1,
        parent_owner_epoch=drv_b.owner_epoch, parent_cancel_seq=drv_b.cancel_seq,
    )
    leaf2_b = cm_leaf2.preclaimed[0]
    await port_a.cancel(leaf2_b.target, actor=ACTOR, scope="self", reason="halt", cancel_by="subtree")
    await port_a.final(
        drv_b, status="failed", final_text="", error="halt", error_code="failed",
        drain_driver_node_results=True, settle_start_input=True, write_driver_message=True,
    )
    # Slow sibling finishes AFTER Driver is idle -> on server DBs also verify concurrent close vs leaf2 Final!
    if store_a._db.dialect.name in ("postgres", "mysql"):
        leaf2_locked_ev = asyncio.Event()
        release_river2_ev = asyncio.Event()

        async def paused_river2(tx, *args, **kwargs):
            leaf2_locked_ev.set()
            await release_river2_ev.wait()
            return await orig_river(tx, *args, **kwargs)

        port_a._river_deliver_tx = paused_river2
        store_b2 = await connect()
        try:
            leaf2_task = asyncio.create_task(
                port_a.final(leaf2_b, status="cancelled", final_text="", error="halt", error_code="cancelled"),
            )
            await asyncio.wait_for(leaf2_locked_ev.wait(), 5)
            close2_task = asyncio.create_task(store_b2.close_sessions_if_idle([drv_sid]))
            await asyncio.sleep(0.05)
            release_river2_ev.set()
            close2_status, leaf2_fin = await asyncio.wait_for(
                asyncio.gather(close2_task, leaf2_task), timeout=10,
            )
            assert close2_status == "busy"
            assert leaf2_fin.result_delivery_error is None
        finally:
            release_river2_ev.set()
            port_a._river_deliver_tx = orig_river
            await store_b2.close()
    else:
        leaf2_fin = await port_a.final(
            leaf2_b, status="cancelled", final_text="", error="halt", error_code="cancelled",
        )
        assert leaf2_fin.result_delivery_error is None

    pending_wake0 = await store_a._db.fetchall(
        f"SELECT wake,status,content_json FROM {store_a.t.follow_up_queue} "
        "WHERE session_id=? AND kind='workflow_node_result' AND status='pending'",
        (drv_sid,),
    )
    assert len(pending_wake0) == 1 and int(pending_wake0[0]["wake"]) == 0
    assert json.loads(pending_wake0[0]["content_json"])["cancel_by"] == "subtree"

    # With the real pending wake=0 receipt STILL in Driver's inbox, close_sessions_if_idle MUST return 'closed'!
    assert await store_a.close_sessions_if_idle([drv_sid]) == "closed"
    future_ms = now_ms() + 31 * 86400 * 1000
    page1 = await store_a.purge_expired_page(future_ms, limit=50)
    page2 = await store_a.purge_expired_page(future_ms, limit=50)
    assert sum(page1.deleted.values()) > 0 or sum(page2.deleted.values()) > 0
    leftover = await store_a._db.fetchall(
        f"SELECT session_id FROM {store_a.t.sessions} WHERE session_id IN (?,?,?)",
        (drv_sid, "s2b3-leaf", "s2b3-leaf-slow"),
    )
    assert leftover == []


async def test_stage2_b4_real_two_process_5_leaf_sql_ledgers(registry_store, tmp_path):
    """Stage2-B4: Genuine 2-OS-process (separate PIDs, Stores, waiter tables, signal-only pipe) 5-leaf SQL ledgers."""
    import os
    import sys
    from pathlib import Path

    store, _ = registry_store
    backend = store._db.dialect.name
    repo_root = Path(__file__).resolve().parents[2]
    scenarios = ["concentrated", "dispersed", "mid_stop_1_leaf", "stop_driver_subtree", "real_halt"]
    xproc_tables = (
        "follow_up_queue", "messages", "project_messages",
        "session_runtime_state", "sessions", "schema_migrations",
    )

    async def _drop_prefix_tables(pfx: str) -> None:
        if backend in ("postgres", "mysql"):
            for tbl in xproc_tables:
                await store._db.execute(f"DROP TABLE IF EXISTS {pfx}{tbl}")

    for idx_s, scenario in enumerate(scenarios):
        prefix = f"b4xp{idx_s}_"
        await _drop_prefix_tables(prefix)
        if backend == "sqlite":
            dsn_or_path = str(tmp_path / f"b4_xproc_{idx_s}.db")
        elif backend == "postgres":
            dsn_or_path = os.environ["POWER_LOOP_TASK_PG_DSN"]
        else:
            dsn_or_path = os.environ["POWER_LOOP_TASK_MYSQL_DSN"]

        node_a_sids: set[str] = {"000_main"}
        events_a: asyncio.Queue[dict] = asyncio.Queue()
        events_b: asyncio.Queue[dict] = asyncio.Queue()
        signals_seen: list[dict] = []
        proc_a = None
        proc_b = None

        async def _send(proc, obj: dict) -> None:
            assert proc.stdin is not None
            proc.stdin.write((json.dumps(obj) + "\n").encode())
            await proc.stdin.drain()

        async def _relay(src_role: str, src_proc, evt_q: asyncio.Queue[dict]) -> None:
            assert src_proc.stdout is not None
            while True:
                line = await src_proc.stdout.readline()
                if not line:
                    err_txt = ""
                    if src_proc.stderr is not None:
                        try:
                            err_txt = (await src_proc.stderr.read()).decode("utf-8", errors="replace")[-4000:]
                        except Exception:
                            pass
                    await evt_q.put({
                        "type": "eof", "role": src_role, "returncode": src_proc.returncode, "stderr": err_txt,
                    })
                    return
                msg = json.loads(line)
                if msg.get("type") == "signal":
                    # Strictly enforce signal-only contract (ONLY session_id + kind + reason/cancel_by, 0 payload!)
                    assert set(msg.keys()) <= {"type", "kind", "session_id", "reason", "cancel_by"}, msg
                    signals_seen.append(msg)
                    sid = msg["session_id"]
                    if src_role == "a" and msg["kind"] == "wake" and msg.get("reason") == "accept" and len(node_a_sids) == 1:
                        node_a_sids.add(sid)
                    dst = proc_a if sid in node_a_sids else proc_b
                    assert dst is not None
                    await _send(dst, msg)
                else:
                    await evt_q.put(msg)

        async def _wait_evt(q: asyncio.Queue[dict], pred) -> dict:
            async with asyncio.timeout(25):
                while True:
                    m = await q.get()
                    if m.get("type") == "eof":
                        raise AssertionError(
                            f"xproc worker role={m['role']} exited early (rc={m.get('returncode')}):\n{m.get('stderr')}"
                        )
                    if pred(m):
                        return m

        # Start proc_a FIRST and wait for ready_a so DDL init runs once with zero concurrent CREATE TABLE race.
        ra = None
        rb = None
        try:
            proc_a = await asyncio.create_subprocess_exec(
                sys.executable, "-B", "-m", "tests.unit._m2_xproc_ledger_worker",
                backend, dsn_or_path, prefix, "a", scenario,
                cwd=str(repo_root),
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            ra = asyncio.create_task(_relay("a", proc_a, events_a))
            ready_a = await _wait_evt(events_a, lambda m: m.get("type") == "ready")
            proc_b = await asyncio.create_subprocess_exec(
                sys.executable, "-B", "-m", "tests.unit._m2_xproc_ledger_worker",
                backend, dsn_or_path, prefix, "b", scenario,
                cwd=str(repo_root),
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            assert proc_a.pid != proc_b.pid
            rb = asyncio.create_task(_relay("b", proc_b, events_b))
            ready_b = await _wait_evt(events_b, lambda m: m.get("type") == "ready")
            assert ready_a["pid"] != ready_b["pid"]

            await _send(proc_a, {"type": "start_workflow"})
            sub_ev = await _wait_evt(events_a, lambda m: m.get("name") == "workflow_submitted")
            node_a_sids.add(sub_ev["driver_sid"])

            if scenario == "mid_stop_1_leaf":
                # Wait until all 5 remote leaves have started in Process B
                seen_idxs = set()
                while len(seen_idxs) < 5:
                    ev = await _wait_evt(events_b, lambda m: m.get("name") == "leaf_started")
                    seen_idxs.add(ev["idx"])
                await _send(proc_a, {"type": "stop_l0"})
                st_ev = await _wait_evt(events_a, lambda m: m.get("name") == "l0_stopped")
                assert st_ev["state"] == "cancel_requested"
                await _send(proc_b, {"type": "release_l0"})
                await _send(proc_b, {"type": "release_all"})
                await _send(proc_a, {"type": "wait_workflow_and_main"})
                done_ev = await _wait_evt(events_a, lambda m: m.get("name") == "workflow_and_main_done")
                assert done_ev["wf_result_statuses"]["L0"] == "cancelled"
                assert done_ev["wf_result_cancel_bys"]["L0"] == "user"
                for i in range(1, 5):
                    assert done_ev["wf_result_statuses"][f"L{i}"] == "completed"
                assert done_ev["main_status"] == "completed"
            elif scenario == "stop_driver_subtree":
                seen_idxs = set()
                while len(seen_idxs) < 5:
                    ev = await _wait_evt(events_b, lambda m: m.get("name") == "leaf_started")
                    seen_idxs.add(ev["idx"])
                await _send(proc_a, {"type": "stop_driver_subtree"})
                st_ev = await _wait_evt(events_a, lambda m: m.get("name") == "driver_subtree_stopped")
                assert st_ev["state"] == "cancel_requested"
                # Wait for Driver + Main BEFORE releasing slow leaves (0 dead-wait!)
                await _send(proc_a, {"type": "wait_workflow_and_main"})
                done_ev = await _wait_evt(events_a, lambda m: m.get("name") == "workflow_and_main_done")
                assert done_ev["wf_status"] == "cancelled" and done_ev["wf_cancel_by"] == "user"
                assert done_ev["main_status"] == "completed"
                await _send(proc_b, {"type": "release_all"})
            elif scenario == "real_halt":
                # L0 fails once L1..L4 have started -> Driver halts slow siblings BEFORE release_all!
                await _send(proc_a, {"type": "wait_workflow_and_main"})
                done_ev = await _wait_evt(events_a, lambda m: m.get("name") == "workflow_and_main_done")
                assert done_ev["wf_status"] == "failed" and done_ev["main_status"] == "completed"
                await _send(proc_b, {"type": "release_all"})
            else:
                await _send(proc_a, {"type": "wait_workflow_and_main"})
                done_ev = await _wait_evt(events_a, lambda m: m.get("name") == "workflow_and_main_done")
                assert done_ev["wf_status"] == "completed" and done_ev["main_status"] == "completed"

            await _send(proc_a, {"type": "wait_idle"})
            await _send(proc_b, {"type": "wait_idle"})
            await _wait_evt(events_a, lambda m: m.get("name") == "idle")
            await _wait_evt(events_b, lambda m: m.get("name") == "idle")

            await _send(proc_a, {"type": "report"})
            await _send(proc_b, {"type": "report"})
            rep_a = await _wait_evt(events_a, lambda m: m.get("type") == "report")
            rep_b = await _wait_evt(events_b, lambda m: m.get("type") == "report")
            assert rep_a["legacy_job_control"] == 0 and rep_b["legacy_job_control"] == 0
            assert rep_a["billing"] == 0 and rep_b["billing"] == 0
            assert rep_a["main_llm_calls"] == 1 and rep_b["leaf_llm_calls"] == 5

            if out := os.environ.get("M2_REAL_XPROC_LEDGER_OUT"):
                with open(out, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps({
                        "scenario": scenario,
                        "db": backend,
                        "pid_a": rep_a["pid"],
                        "pid_b": rep_b["pid"],
                        "proc_a": {k: rep_a[k] for k in ("transactions", "session_control", "message", "total", "kv_sql", "by_func")},
                        "proc_b": {k: rep_b[k] for k in ("transactions", "session_control", "message", "total", "kv_sql", "by_func")},
                        "combined": {
                            "transactions": rep_a["transactions"] + rep_b["transactions"],
                            "session_control": rep_a["session_control"] + rep_b["session_control"],
                            "message": rep_a["message"] + rep_b["message"],
                            "total": rep_a["total"] + rep_b["total"],
                            "kv_sql": rep_a["kv_sql"] + rep_b["kv_sql"],
                        },
                        "signal_count": len(signals_seen),
                    }) + "\n")
        finally:
            for proc in (proc_a, proc_b):
                if proc is not None and proc.returncode is None:
                    try:
                        await _send(proc, {"type": "quit"})
                        await asyncio.wait_for(proc.wait(), 10)
                    except Exception:
                        proc.kill()
                        await proc.wait()
            for task in (ra, rb):
                if task is not None:
                    task.cancel()
            await asyncio.gather(*(t for t in (ra, rb) if t is not None), return_exceptions=True)
            # Drop ALL temporary prefix tables (including schema_migrations!) so repeat runs in the same DB succeed
            await _drop_prefix_tables(prefix)


async def test_stage2_driver_sigkill_takeover_completed_leaf_runs_once(registry_store):
    """Real Driver SIGKILL (-9) after leaf A persists completion: replacement Driver finishes B, A called once."""
    import os
    import signal
    import sys
    from pathlib import Path

    from power_loop import DatabasePollingTransport, RuntimeClient
    from power_loop.agent.runtime_store import RuntimeInputStore
    from power_loop.agent.runtime_types import RuntimeDescriptor

    store, _ = registry_store
    backend = store._db.dialect.name
    if backend == "sqlite":
        pytest.skip("real Driver SIGKILL takeover uses shared server DB (PG16/MySQL)")
    dsn = os.environ["POWER_LOOP_TASK_PG_DSN" if backend == "postgres" else "POWER_LOOP_TASK_MYSQL_DSN"]
    repo_root = Path(__file__).resolve().parents[2]

    desc = RuntimeDescriptor(
        2, "agent", "m2.sigkill@215", None, "m2.tools@215", "m2.hooks@215", "m2.res@215", {}, {},
    )
    actor = TrustedActor("t-sigkill", "u-sigkill", None)
    port = RuntimeInputStore(store)
    root = await port.create_session(desc, actor=actor, options={"session_id": "sigkill-main"})
    tp = DatabasePollingTransport(port, interval_s=0.03)
    client = RuntimeClient(store=store, transport=tp)

    async def _spawn(role: str):
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-B", "-m", "tests.unit._m2_xproc_ledger_worker",
            backend, dsn, store.table_prefix, role, "sigkill_ab",
            cwd=str(repo_root),
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        async with asyncio.timeout(20):
            line = await proc.stdout.readline()
            assert line, (await proc.stderr.read()).decode()[-4000:]
            msg = json.loads(line)
            assert msg.get("type") == "ready"
        return proc

    async def _wait_msg(proc, pred) -> dict:
        async with asyncio.timeout(25):
            while True:
                line = await proc.stdout.readline()
                assert line, (await proc.stderr.read()).decode()[-4000:]
                msg = json.loads(line)
                if pred(msg):
                    return msg

    async def _cmd(proc, obj: dict) -> None:
        assert proc.stdin is not None
        proc.stdin.write((json.dumps(obj) + "\n").encode())
        await proc.stdin.drain()

    procs = []
    try:
        spec = {
            "name": "sigkill-a-then-b",
            "root": {
                "type": "sequence",
                "steps": [
                    {
                        "type": "agent", "id": "A", "input": "step-A",
                        "spec": {"name": "A", "system_prompt": "LEAF-A", "tools": []},
                    },
                    {
                        "type": "agent", "id": "B", "input": "step-B", "inputs_from": ["A"],
                        "spec": {"name": "B", "system_prompt": "LEAF-B", "tools": []},
                    },
                ],
            },
        }
        handle = await client.workflow(root, spec, message_id="sigkill-ab-run", actor=actor)
        drv1 = await _spawn("sigkill_c1")
        procs.append(drv1)
        assert (await _wait_msg(drv1, lambda m: m.get("name") == "driver_claimed"))["send_index"] == 1
        leaf_w = await _spawn("sigkill_n")
        procs.append(leaf_w)

        # Wait until A completed & Driver persisted A + spawned B -> B actually started!
        await _wait_msg(leaf_w, lambda m: m.get("name") == "leaf_done" and m.get("node") == "A")
        await _wait_msg(leaf_w, lambda m: m.get("name") == "leaf_started" and m.get("node") == "B")

        # Real OS SIGKILL (-9) on Driver 1!
        os.kill(drv1.pid, signal.SIGKILL)
        rc = await drv1.wait()
        assert rc == -signal.SIGKILL

        # Start replacement Driver 2; wait for takeover at send_index=2, then release B.
        drv2 = await _spawn("sigkill_c2")
        procs.append(drv2)
        assert (await _wait_msg(drv2, lambda m: m.get("name") == "driver_claimed"))["send_index"] == 2
        await _cmd(leaf_w, {"type": "release_b"})

        wf_res = await handle.wait(deadline_ms=now_ms() + 20000)
        assert wf_res.status == "completed"
        assert wf_res.results["A"].text == "A-once-result"
        assert "A-once-result" in wf_res.results["B"].text

        await _cmd(leaf_w, {"type": "report_calls"})
        call_msg = await _wait_msg(leaf_w, lambda m: m.get("type") == "calls")
        assert call_msg["calls"] == {"A": 1, "B": 1}, call_msg
    finally:
        for proc in procs:
            if proc.returncode is None:
                try:
                    await _cmd(proc, {"type": "quit"})
                    await asyncio.wait_for(proc.wait(), 10)
                except Exception:
                    proc.kill()
                    await proc.wait()
        await client.aclose()
        await tp.aclose()


async def test_stage2_r2_b3_concurrent_rebind_children_preserves_rebound_child_on_close_and_delete(registry_store):
    """Stage2-R2-B3: Child rebound to same-depth new parent between discovery & lock stays open & intact."""
    from dataclasses import replace

    from power_loop.agent.runtime_client import RuntimeClient
    from power_loop.agent.runtime_store import ChildSpec, RuntimeInputStore
    from power_loop.agent.runtime_transport import MemoryMessageTransport
    from power_loop.runtime.store import closing as closing_mod

    from .session_native_fixture import ACTOR, DESCRIPTOR, target

    store_a, connect = registry_store
    if store_a._db.dialect.name == "sqlite":
        pytest.skip("concurrent two-connection rebind vs close/delete requires server DB (PG16/MySQL)")

    async def _setup_tree(port: RuntimeInputStore, tag: str):
        root = await target(port, f"{tag}-root")
        cm_parents = await port.create_many(
            parent=root, actor=ACTOR,
            specs=[
                ChildSpec(
                    session_id=f"{tag}-old-parent", message_id=f"{tag}-op-m1",
                    body={"text": "old parent"}, descriptor=DESCRIPTOR, result_kind="agent_result",
                ),
                ChildSpec(
                    session_id=f"{tag}-new-parent", message_id=f"{tag}-np-m1",
                    body={"text": "new parent"}, descriptor=DESCRIPTOR, result_kind="agent_result",
                ),
            ],
            node_id="n1", boot_id="b1", can_execute=True, free_slots=2,
        )
        old_b = next(b for b in cm_parents.preclaimed if b.target.session_id == f"{tag}-old-parent")
        new_b = next(b for b in cm_parents.preclaimed if b.target.session_id == f"{tag}-new-parent")
        cm_child = await port.create_many(
            parent=old_b.target, actor=ACTOR,
            specs=[
                ChildSpec(
                    session_id=f"{tag}-child", message_id=f"{tag}-ch-m1",
                    body={"text": "child work"},
                    descriptor=replace(DESCRIPTOR, config={**dict(DESCRIPTOR.config), "workflow_node": {"id": "C"}}),
                    result_kind="agent_result",
                ),
            ],
            node_id="n1", boot_id="b1", can_execute=True, free_slots=1,
            parent_owner_epoch=old_b.owner_epoch, parent_cancel_seq=old_b.cancel_seq,
        )
        child_b = cm_child.preclaimed[0]
        await port.final(child_b, status="completed", final_text=f"{tag}-child-result")
        await port.final(old_b, status="completed", final_text="old-done")
        await port.final(new_b, status="completed", final_text="new-done")
        # Consume old_parent's pending agent_result receipt from child so old_parent is completely idle!
        claim_old2 = await port.claim(old_b.target, node_id="n1", boot_id="b1")
        if claim_old2 is not None:
            await port.final(claim_old2, status="completed", final_text="old-drained")
        return old_b.target, new_b.target, child_b.target, child_b.input_refs[0]

    port_a = RuntimeInputStore(store_a)
    store_b = await connect()
    transport = MemoryMessageTransport()
    client_b = RuntimeClient(store=store_b, transport=transport)
    orig_discover = closing_mod._discover_linked_tree
    try:
        for tag, mode in (("s2r2b3-idle", "close_if_idle"), ("s2r2b3-tree", "close_session_tree")):
            old_p, new_p, child_t, child_ref = await _setup_tree(port_a, tag)

            discovered_ev = asyncio.Event()
            release_lock_ev = asyncio.Event()
            paused_once = False

            async def _paused_discover(st, tx, seed_ids, cascade):
                nonlocal paused_once
                res = await orig_discover(st, tx, seed_ids, cascade)
                if not paused_once and old_p.session_id in res and child_t.session_id in res:
                    paused_once = True
                    discovered_ev.set()
                    await release_lock_ev.wait()
                return res

            closing_mod._discover_linked_tree = _paused_discover
            try:
                if mode == "close_if_idle":
                    op_task = asyncio.create_task(store_a.close_sessions_if_idle([old_p.session_id]))
                else:
                    op_task = asyncio.create_task(store_a.close_session_tree(old_p.session_id))
                await asyncio.wait_for(discovered_ev.wait(), 5)

                # Connection 2: rebind child from old_parent to same-depth new_parent and commit!
                rb = await client_b.rebind_children(
                    old_p, new_p, child_targets=[child_t], actor=ACTOR,
                )
                assert rb.depth_delta == 0
                assert list(rb.rebound_children) == [child_t.session_id]

                release_lock_ev.set()
                op_result = await asyncio.wait_for(op_task, 10)
            finally:
                release_lock_ev.set()
                closing_mod._discover_linked_tree = orig_discover

            if mode == "close_if_idle":
                assert op_result == "closed"
                old_row = await store_b._db.fetchone(
                    f"SELECT lifecycle FROM {store_b.t.sessions} WHERE session_id=?",
                    (old_p.session_id,),
                )
                assert old_row is not None and old_row["lifecycle"] == "closing"
            else:
                assert op_result == [old_p.session_id], op_result
                old_row = await store_b._db.fetchone(
                    f"SELECT session_id FROM {store_b.t.sessions} WHERE session_id=?",
                    (old_p.session_id,),
                )
                assert old_row is None

            # Rebound child under new_parent MUST remain open with messages, input terminal status & result intact!
            ch_row = await store_b._db.fetchone(
                f"SELECT parent_session_id,lifecycle,execution_state FROM {store_b.t.sessions} WHERE session_id=?",
                (child_t.session_id,),
            )
            assert ch_row is not None
            assert ch_row["parent_session_id"] == new_p.session_id
            assert ch_row["lifecycle"] == "open"
            ch_msgs = await store_b._db.fetchall(
                f"SELECT role,content FROM {store_b.t.messages} WHERE session_id=? ORDER BY seq ASC",
                (child_t.session_id,),
            )
            assert any(m["role"] == "assistant" and m["content"] == f"{tag}-child-result" for m in ch_msgs)
            ch_insp = await client_b.inspect(child_ref, actor=ACTOR)
            assert ch_insp.terminal_status == "completed"
            ch_res = await client_b.inputs.get_result(child_ref, actor=ACTOR)
            assert ch_res is not None and ch_res.status == "completed" and ch_res.final_text == f"{tag}-child-result"
    finally:
        closing_mod._discover_linked_tree = orig_discover
        await client_b.aclose()
        await store_b.close()


async def test_stage2_r2_b1_two_explicit_resumes_claimed_together_both_execute_and_deliver(registry_store):
    """Stage2-R2-B1: Two accepted handle.resume() calls claimed in the same batch both execute and deliver."""
    from power_loop.agent.runtime_client import RuntimeClient
    from power_loop.agent.runtime_transport import MemoryMessageTransport
    from power_loop.workflow.runner import WorkflowRunHandle

    store, _ = registry_store
    seen_prompts: list[str] = []

    class TwoResumeModel(Model):
        async def complete(self, req, **kwargs):
            if req.system_prompt.startswith("A"):
                user_texts = [
                    str(m.get("content") if isinstance(m, dict) else getattr(m, "content", ""))
                    for m in req.messages
                    if (m.get("role") if isinstance(m, dict) else getattr(m, "role", None)) == "user"
                ]
                latest = user_texts[-1] if user_texts else ""
                seen_prompts.append(latest)
                return LLMResponse(raw_text=f"A-call-{len(seen_prompts)}:{latest}")
            return LLMResponse(raw_text="main-ack")

    transport = MemoryMessageTransport()
    rt1 = await RunningAgent(
        store, TwoResumeModel(), config=AgentLoopConfig(context_budget_tokens=10**9), capacity=2,
        transport=transport,
    ).start()
    client = RuntimeClient(store=store, transport=transport)
    rt2 = None
    try:
        main_target = rt1.target
        handle = await rt1.runtime.client.workflow(
            main_target, {"name": "s2-r2-b1-wf", "root": leaf("A")},
            message_id="s2-r2-b1-start", actor=rt1.actor,
        )
        first = await handle.wait(deadline_ms=now_ms() + 10000)
        assert first.status == "completed" and len(seen_prompts) == 1
        await rt1.drain()
        await rt1.runtime.worker.shutdown(timeout_s=2, cancel_running=True)

        # Submit TWO explicit handle.resume() calls with distinct message_ids BEFORE Worker 2 claims!
        handle_on_client = WorkflowRunHandle(client, rt1.actor, handle.target, handle.input_ref)
        res_h1 = await handle_on_client.resume(
            node_id="A", message="resume-prompt-one", message_id="s2-r2-b1-res-1",
        )
        res_h2 = await handle_on_client.resume(
            node_id="A", message="resume-prompt-two", message_id="s2-r2-b1-res-2",
        )

        # Confirm both are pending in Driver's inbox before Worker 2 starts
        pend = await store._db.fetchall(
            f"SELECT item_id,status FROM {store.t.follow_up_queue} "
            "WHERE session_id=? AND status='pending' ORDER BY id ASC",
            (handle.target.session_id,),
        )
        assert [r["item_id"] for r in pend] == ["s2-r2-b1-res-1", "s2-r2-b1-res-2"]

        # Start Worker 2 -> BOTH explicit resumes are claimed together in send_index=2!
        rt2 = await RunningAgent(
            store, TwoResumeModel(), config=AgentLoopConfig(context_budget_tokens=10**9), capacity=2,
            transport=transport,
        ).start(target=main_target)
        claimed_batches: list[tuple[int, list[str]]] = []
        orig_claim = rt2.runtime.worker.inputs.claim

        async def _spy_claim(target_arg, *args, **kwargs):
            b = await orig_claim(target_arg, *args, **kwargs)
            if b is not None and target_arg.session_id == handle.target.session_id:
                claimed_batches.append((int(b.send_index), [r.message_id for r in b.input_refs]))
            return b

        rt2.runtime.worker.inputs.claim = _spy_claim
        await rt2.runtime.worker.periodic_tick(limit=32)

        r1 = await res_h1.wait(deadline_ms=now_ms() + 10000)
        r2 = await res_h2.wait(deadline_ms=now_ms() + 10000)
        await rt2.drain()

        # Verify BOTH were claimed together in the same batch (send_index=2), and res-2 re-ran on send_index=3!
        assert claimed_batches[0] == (2, ["s2-r2-b1-res-1", "s2-r2-b1-res-2"]), claimed_batches
        assert claimed_batches[1] == (3, ["s2-r2-b1-res-2"]), claimed_batches
        claimed_rows = await store._db.fetchall(
            f"SELECT item_id,result_send_index,terminal_status,result_seq FROM {store.t.follow_up_queue} "
            "WHERE session_id=? AND item_id IN ('s2-r2-b1-res-1','s2-r2-b1-res-2') ORDER BY id ASC",
            (handle.target.session_id,),
        )
        assert len(claimed_rows) == 2
        assert [int(r["result_send_index"]) for r in claimed_rows] == [2, 3]
        assert all(r["terminal_status"] == "completed" for r in claimed_rows)
        assert claimed_rows[0]["result_seq"] != claimed_rows[1]["result_seq"]

        # Verify BOTH actually ran target leaf A with their respective prompts!
        assert len(seen_prompts) == 3
        assert "resume-prompt-one" in seen_prompts[1]
        assert "resume-prompt-two" in seen_prompts[2]
        assert r1.status == "completed" and "A" in r1.results and "resume-prompt-one" in r1.results["A"].text
        assert r2.status == "completed" and "A" in r2.results and "resume-prompt-two" in r2.results["A"].text

        # Verify Main received 3 distinct workflow_result receipts: start + res-1 + res-2!
        main_rcpts = await store._db.fetchall(
            f"SELECT item_id,content_json FROM {store.t.follow_up_queue} "
            "WHERE session_id=? AND kind='workflow_result' ORDER BY id ASC",
            (main_target.session_id,),
        )
        assert len(main_rcpts) == 3
        bodies = [json.loads(r["content_json"]) for r in main_rcpts]
        assert [b["input_ref"]["message_id"] for b in bodies] == [
            handle.input_ref.message_id, "s2-r2-b1-res-1", "s2-r2-b1-res-2",
        ]
        assert "resume-prompt-one" in bodies[1]["final_text"]
        assert "resume-prompt-two" in bodies[2]["final_text"]
        assert (await handle_on_client.wait(deadline_ms=now_ms() + 5000)).to_dict() == first.to_dict()
    finally:
        if rt2 is not None:
            await rt2.aclose()
        await client.aclose()
        await rt1.aclose()


async def test_stage2_r2_b2_inprocess_parallel_two_leaves_staggered_zero_extra_db_reads(registry_store):
    """Stage2-R2-B2: Same Worker/Store capacity=2 parallel leaves A & B staggered finish -> 0 poll & 0 get_result."""
    from .test_send_sql_profile import Profiled

    store, _ = registry_store
    a_entered = asyncio.Event()
    b_entered = asyncio.Event()
    release_a = asyncio.Event()
    release_b = asyncio.Event()

    class StaggeredTwoLeafModel(Model):
        async def complete(self, req, **kwargs):
            if req.system_prompt.startswith("A"):
                a_entered.set()
                await release_a.wait()
                return LLMResponse(raw_text="leaf-A-done")
            if req.system_prompt.startswith("B"):
                b_entered.set()
                await release_b.wait()
                return LLMResponse(raw_text="leaf-B-done")
            return LLMResponse(raw_text="main-ack")

    orig_db = store._db
    prof = Profiled(orig_db)
    store._db = prof
    runner = await RunningAgent(
        store, StaggeredTwoLeafModel(), config=AgentLoopConfig(context_budget_tokens=10**9), capacity=2,
    ).start()
    try:
        spec = {
            "name": "s2-r2-b2-parallel",
            "root": {"type": "parallel", "max_concurrency": 2, "branches": [leaf("A"), leaf("B")]},
        }
        handle = await runner.runtime.client.workflow(
            runner.target, spec, message_id="s2-r2-b2-par", actor=runner.actor,
        )
        await asyncio.wait_for(asyncio.gather(a_entered.wait(), b_entered.wait()), 5)
        await asyncio.sleep(0.05)

        # Find leaf A's session_id on orig_db (zero profiler impact)
        child_rows = await orig_db.fetchall(
            f"SELECT session_id,descriptor_json FROM {store.t.sessions} WHERE parent_session_id=?",
            (handle.target.session_id,),
        )
        a_sid = next(r["session_id"] for r in child_rows if '"node_id":"A"' in r["descriptor_json"] or '"node_id": "A"' in r["descriptor_json"])

        # Clear profiler BEFORE releasing A; release A while keeping B blocked!
        prof.records.clear()
        release_a.set()

        # Wait until A has committed Final (execution_state='idle'), then wait across >250ms boundary while B stays active!
        async with asyncio.timeout(5):
            while True:
                arow = await orig_db.fetchone(
                    f"SELECT execution_state FROM {store.t.sessions} WHERE session_id=?", (a_sid,),
                )
                if arow is not None and arow["execution_state"] == "idle":
                    break
                await asyncio.sleep(0.01)
        await asyncio.sleep(0.35)

        poll_reads = [
            r for r in prof.records
            if any("poll_pending_driver_results" in f or "poll_driver_results_for_send" in f for f in r["chain"])
        ]
        get_result_reads = [
            r for r in prof.records
            if any("get_result" in f or "_result_state" in f for f in r["chain"])
        ]
        assert poll_reads == [], f"Unexpected poll_pending_driver_results between A commit and B release: {poll_reads}"
        assert get_result_reads == [], f"Unexpected leaf get_result between A commit and B release: {get_result_reads}"

        release_b.set()
        res = await handle.wait(deadline_ms=now_ms() + 10000)
        assert res.status == "completed"
        assert res.results["A"].text == "leaf-A-done"
        assert res.results["B"].text == "leaf-B-done"
        await runner.drain()
    finally:
        release_a.set()
        release_b.set()
        await runner.aclose()
        store._db = orig_db
