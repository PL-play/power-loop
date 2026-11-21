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
