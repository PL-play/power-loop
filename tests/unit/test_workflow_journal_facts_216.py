"""S08 executor facts: real DB commits, parallel publication and bounded lifetime."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from power_loop import AgentLoopConfig, ExecutionEnvironment
from power_loop.agent.runtime_store import now_ms
from power_loop.agent.runtime_types import (
    CapacityError,
    RuntimeRequestError,
    TargetRef,
    TrustedActor,
    canonical,
    plain,
)
from power_loop.agent.runtime_worker import _SessionExecution
from power_loop.contracts.errors import SessionLeaseLost
from power_loop.core.execution_context import _bind_execution_context
from power_loop.workflow import NodeRequest, journal
from power_loop.workflow.runner import DurableExecutor
from tests.runtime_fixture import RunningAgent, ScriptedModel

from ._workflow_fixture_203 import leaf
from .session_native_fixture import UnknownCommit

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@asynccontextmanager
async def executor_fixture(store):
    agent = RunningAgent(store, ScriptedModel(), config=AgentLoopConfig(context_budget_tokens=10**9))
    agent.runtime.client._accepting = True
    agent.target = await agent.runtime.client.create_session(agent.descriptor, actor=agent.actor)
    handle = await agent.runtime.client.workflow(
        agent.target, {"name": "facts", "root": leaf("A")}, message_id="facts", actor=agent.actor)
    port = agent.runtime.worker.inputs
    batch = await port.claim(handle.target, node_id="manual", boot_id="manual", ttl_s=300)
    ctx = _SessionExecution(agent.runtime, handle.target.session_id, batch.descriptor, batch,
                            TrustedActor(batch.tenant_id, batch.principal_id, handle.target.session_id))
    ctx.ownership = SimpleNamespace(lost=False)
    env = ExecutionEnvironment(None, None, agent.config)
    saved = await journal.get(store, store._db, ctx.session_id, journal.run_key(handle.run_id))
    executor = DurableExecutor(agent.runtime.worker, ctx, env, handle.run_id, saved)
    try:
        with _bind_execution_context(ctx.snapshot):
            yield executor, agent
    finally:
        await agent.aclose()


def request(executor, name="A"):
    return NodeRequest(executor.run_id, name, [], 1,
                       {"name": name, "system_prompt": name, "tools": []}, name)


async def terminalize(executor, ref):
    target = TargetRef(ref.store_id, ref.session_id)
    batch = await executor.port.claim(target, node_id="leaf", boot_id="leaf", ttl_s=300)
    assert batch is not None
    final = await executor.port.final(batch, final_text="result:" + ref.session_id)
    return final.results[0]


async def start_ready(executor, agent, requests, monkeypatch):
    """Create/journal all nodes and stop their callers at wait, without a model."""
    waiting, release = asyncio.Event(), asyncio.Event()
    refs = []

    async def wait(ref, **kwargs):
        refs.append(ref)
        if len(refs) == len(requests):
            waiting.set()
        await release.wait()
        return await executor.port.get_result(ref, actor=executor.actor)

    monkeypatch.setattr(agent.runtime.client, "wait", wait)
    tasks = [asyncio.create_task(executor.run_agent(req)) for req in requests]
    await asyncio.wait_for(waiting.wait(), 10)
    return tasks, refs, release


async def test_parallel_results_publish_once_and_do_not_lock_wait(registry_store, monkeypatch):
    store, _ = registry_store
    async with executor_fixture(store) as (executor, agent):
        loads = 0
        original = journal.nodes

        async def counted(*args, **kwargs):
            nonlocal loads
            loads += 1
            return await original(*args, **kwargs)

        monkeypatch.setattr(journal, "nodes", counted)
        tasks, refs, release = await start_ready(
            executor, agent, [request(executor, "A"), request(executor, "B")], monkeypatch)
        try:
            assert not executor._journal_lock.locked() and len(executor._nodes) == 2
            for ref in refs:
                await terminalize(executor, ref)
            # Both source results commit before either waiting caller resumes.
            release.set()
            results = await asyncio.gather(*tasks)
            assert all(result["status"] == "completed" for result in results)
            assert loads == 1 and len(executor._by_input) == 2
            rows = await original(store, store._db, executor.ctx.session_id)
            assert len(rows) == 2 and all(row["result"] is not None for row in rows)
            assert all(not row["late_results"] for row in rows)
            inputs = await store._db.fetchall(
                f"SELECT terminal_status FROM {executor.port.q} WHERE session_id=? AND kind='workflow_node_result'",
                (executor.ctx.session_id,))
            assert len(inputs) == 2 and all(row["terminal_status"] == "completed" for row in inputs)
        finally:
            release.set()
            await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.parametrize("failure", ("rollback", "cancel", "unknown"))
@pytest.mark.parametrize("late", (False, True))
async def test_consumption_failure_does_not_publish_transaction_copies(
    registry_store, monkeypatch, failure, late,
):
    store, _ = registry_store
    async with executor_fixture(store) as (executor, agent):
        tasks, refs, release = await start_ready(executor, agent, [request(executor)], monkeypatch)
        try:
            await terminalize(executor, refs[0])
            if late:
                executor.saved = {**executor.saved, "aggregate_delivered": True, "status": "completed"}
                async with store._db.transaction() as tx:
                    await executor._check_owner(tx)
                    await journal.put(store, tx, executor.ctx.session_id,
                                      journal.run_key(executor.run_id), executor.saved)
            key = executor._by_input[refs[0]]
            old = executor._nodes[key]
            transaction = store._db.transaction
            armed = True

            @asynccontextmanager
            async def broken():
                nonlocal armed
                if not armed:
                    async with transaction() as tx:
                        yield tx
                    return
                armed = False
                async with transaction() as tx:
                    yield tx
                    if failure == "rollback":
                        raise RuntimeError("before commit")
                    if failure == "cancel":
                        raise asyncio.CancelledError("before commit")
                raise UnknownCommit("commit succeeded, acknowledgement lost")

            monkeypatch.setattr(store._db, "transaction", broken)
            error = {"rollback": RuntimeError, "cancel": asyncio.CancelledError, "unknown": UnknownCommit}[failure]
            with pytest.raises(error):
                await executor._consume_results()
            assert executor._nodes[key] is old and old["result"] is None and old["late_results"] == []
            assert executor._by_input[refs[0]] == key
            monkeypatch.setattr(store._db, "transaction", transaction)
            fresh = DurableExecutor(executor.worker, executor.ctx, executor.env, executor.run_id, executor.saved)
            updates = await fresh._consume_results()
            if failure == "unknown":
                assert updates == {}  # durable result was already consumed; no duplicate.
                async with fresh._journal_lock:
                    await fresh._load_nodes()
            entry = fresh._nodes[key]
            if late:
                assert entry["result"] is None and len(entry["late_results"]) == 1
                assert entry["late_results"][0]["input_ref"]["message_id"] == refs[0].message_id
            else:
                assert entry["result"] is not None and entry["late_results"] == []
                assert (await fresh.run_agent(request(fresh)))["status"] == "completed"
            stored = await journal.get(store, store._db, executor.ctx.session_id, key)
            assert stored == fresh._nodes[key]
            executor.ctx.shutdown_interrupted = True
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            release.set()
            await asyncio.gather(*tasks, return_exceptions=True)


async def test_takeover_rejects_old_map_and_new_executor_loads(registry_store, monkeypatch):
    store, _ = registry_store
    async with executor_fixture(store) as (old, agent):
        tasks, refs, release = await start_ready(old, agent, [request(old)], monkeypatch)
        try:
            await terminalize(old, refs[0])
            await store._db.execute(
                f"UPDATE {old.port.s} SET owner_epoch=owner_epoch+1 WHERE session_id=?",
                (old.ctx.session_id,))
            with pytest.raises(SessionLeaseLost):
                await old._consume_results()
            assert all(entry["result"] is None for entry in old._nodes.values())
            from dataclasses import replace
            new_batch = replace(old.ctx.batch, owner_epoch=old.ctx.batch.owner_epoch + 1)
            ctx = _SessionExecution(agent.runtime, old.ctx.session_id, new_batch.descriptor, new_batch, old.ctx.actor)
            fresh = DurableExecutor(old.worker, ctx, old.env, old.run_id, old.saved)
            assert fresh._nodes is None
            await fresh._consume_results()
            assert all(entry["result"] is not None for entry in fresh._nodes.values())
        finally:
            for task in tasks:
                task.cancel()
            old.ctx.shutdown_interrupted = True
            release.set()
            await asyncio.gather(*tasks, return_exceptions=True)


async def test_capacity_last_slot_leaves_loser_public_and_cancellable(registry_store, monkeypatch):
    store, _ = registry_store
    async with executor_fixture(store) as (executor, agent):
        monkeypatch.setattr(journal, "NODE_LIMIT", 1)
        create = executor.port.create_many
        created, release_create = asyncio.Event(), asyncio.Event()
        created_count = 0

        async def gated_create(*args, **kwargs):
            nonlocal created_count
            result = await create(*args, **kwargs)
            specs = kwargs.get("specs") if "specs" in kwargs else (args[2] if len(args) > 2 else ())
            created_count += len(specs)
            if created_count >= 2:
                created.set()
            await release_create.wait()
            return result

        monkeypatch.setattr(executor.port, "create_many", gated_create)
        waiting = asyncio.Event()

        async def wait(ref, **kwargs):
            waiting.set()
            return await terminalize(executor, ref)

        monkeypatch.setattr(agent.runtime.client, "wait", wait)
        tasks = [asyncio.create_task(executor.run_agent(request(executor, name))) for name in ("A", "B")]
        try:
            await asyncio.wait_for(created.wait(), 10)
            assert not executor._journal_lock.locked() and not executor._nodes
            release_create.set()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            assert sum(isinstance(r, CapacityError) for r in results) == 1, results
            assert sum(isinstance(r, dict) and r["status"] == "completed" for r in results) == 1
            assert len(executor._nodes) == len(executor._by_input) == 1
            children = await agent.runtime.client.list_children(executor.driver, actor=agent.actor)
            assert len(children.items) == 2
            known = {entry["session_id"] for entry in executor._nodes.values()}
            loser = next(child for child in children.items if child.session_id not in known)
            assert (await agent.runtime.client.inspect(loser, actor=agent.actor)).execution_state == "queued"
            receipt = await agent.runtime.client.cancel(loser, actor=agent.actor, scope="subtree")
            assert receipt.state == "voided"
            # Full map rejects a new key before creating another child.
            with pytest.raises(CapacityError):
                await executor.run_agent(request(executor, "C"))
            assert created_count == 2
            # Existing-key retry does not consume a slot or create/send again.
            name = next(entry["path"] for entry in executor._nodes.values())
            assert (await executor.run_agent(request(executor, name)))["status"] == "completed"
            assert created_count == 2
            fresh = DurableExecutor(executor.worker, executor.ctx, executor.env, executor.run_id, executor.saved)
            with pytest.raises(CapacityError):
                await fresh.run_agent(request(executor, "C"))
            assert created_count == 2
            # The same cap also rejects an oversized initial journal, rather than
            # only limiting subsequent additions.
            monkeypatch.setattr(journal, "NODE_LIMIT", 0)
            overflow = DurableExecutor(executor.worker, executor.ctx, executor.env, executor.run_id, executor.saved)
            with pytest.raises(CapacityError):
                await overflow.run_agent(request(executor, name))
        finally:
            release_create.set()
            await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.parametrize("failure", ("cancel", "unknown"))
async def test_create_commit_failure_keeps_map_and_input_index_unpublished(
    registry_store, monkeypatch, failure,
):
    store, _ = registry_store
    async with executor_fixture(store) as (executor, agent):
        transaction, put = store._db.transaction, journal.put
        node_write, armed = False, True

        async def tracked_put(*args):
            nonlocal node_write
            if args[3].startswith(journal.NODE_PREFIX):
                node_write = True
            return await put(*args)

        @asynccontextmanager
        async def broken():
            nonlocal node_write, armed
            node_write = False
            async with transaction() as tx:
                yield tx
                fail = node_write and armed
                if fail:
                    armed = False
                    if failure == "cancel":
                        raise asyncio.CancelledError("node journal before commit")
            if fail:
                raise UnknownCommit("node journal committed, acknowledgement lost")

        monkeypatch.setattr(journal, "put", tracked_put)
        monkeypatch.setattr(store._db, "transaction", broken)
        error = asyncio.CancelledError if failure == "cancel" else UnknownCommit
        with pytest.raises(error):
            await executor.run_agent(request(executor))
        assert executor._nodes == {} and executor._by_input == {}
        monkeypatch.setattr(store._db, "transaction", transaction)
        monkeypatch.setattr(journal, "put", put)
        children = await agent.runtime.client.list_children(executor.driver, actor=agent.actor)
        assert len(children.items) == 1

        async def wait(ref, **kwargs):
            return await terminalize(executor, ref)

        monkeypatch.setattr(agent.runtime.client, "wait", wait)
        fresh = DurableExecutor(executor.worker, executor.ctx, executor.env, executor.run_id, executor.saved)
        assert (await fresh.run_agent(request(fresh)))["status"] == "completed"
        assert len(fresh._nodes) == len(fresh._by_input) == 1
        assert len((await agent.runtime.client.list_children(executor.driver, actor=agent.actor)).items) == 1
        users = await store._db.fetchone(
            f"SELECT COUNT(*) AS n FROM {store.t.messages} WHERE session_id=? AND role='user'",
            (children.items[0].session_id,))
        assert users["n"] == 1


@pytest.mark.parametrize("kind", ("retry", "loop"))
async def test_real_graph_keeps_attempt_and_iteration_entries(registry_store, kind):
    store, _ = registry_store

    class Model(ScriptedModel):
        calls = 0

        async def complete(self, req, **kwargs):
            from power_loop import LLMResponse
            if req.system_prompt.startswith("A"):
                self.calls += 1
                if kind == "retry" and self.calls == 1:
                    raise RuntimeError("business failure")
                return LLMResponse(raw_text="pass" if self.calls == 2 else "not yet")
            return LLMResponse(raw_text="parent acknowledged")

    model = Model()
    runner = await RunningAgent(store, model, config=AgentLoopConfig(context_budget_tokens=10**9)).start()
    try:
        node = leaf("A")
        if kind == "retry":
            node["retry"] = {"max_attempts": 2, "on": ["failed"]}
            root = node
        else:
            root = {"type": "loop", "id": "cycle", "max_iterations": 2, "body": node,
                    "until": {"ref": "A", "op": "eq", "value": "pass"}}
        handle = await runner.runtime.client.workflow(
            runner.target, {"name": kind, "root": root}, message_id=kind, actor=runner.actor)
        result = await handle.wait(deadline_ms=now_ms() + 15000)
        assert result.status == "completed" and model.calls == 2
        await runner.drain()
        saved = await journal.read(store, runner.target.session_id, handle.run_id)
        assert len(saved["nodes"]) == 2 and all(entry["result"] is not None for entry in saved["nodes"])
        if kind == "retry":
            assert sorted(entry["attempt"] for entry in saved["nodes"]) == [1, 2]
            assert len({entry["session_id"] for entry in saved["nodes"]}) == 2
        else:
            assert len({str(entry["iteration"]) for entry in saved["nodes"]}) == 2
            assert len({entry["session_id"] for entry in saved["nodes"]}) == 2
    finally:
        await runner.aclose()


@pytest.mark.parametrize("entry", ("submit", "tool-create", "tool-status"))
async def test_public_workflow_boundaries_still_reject_non_main_actor(registry_store, entry):
    from power_loop.core.agent_context import (
        reset_current_loop,
        reset_session_id,
        set_current_loop,
        set_session_id,
    )
    from power_loop.workflow.facade import run_workflow_tool, submit_workflow

    store, _ = registry_store
    async with executor_fixture(store) as (executor, agent):
        async def ensure_store():
            return store

        loop = SimpleNamespace(config=AgentLoopConfig(delegation_policy="single_level"),
                               _ensure_store=ensure_store)
        loop_token = set_current_loop(loop)
        sid_token = set_session_id(executor.ctx.session_id)
        try:
            with pytest.raises(ValueError, match="execution_role_forbids_workflow"):
                if entry == "submit":
                    await submit_workflow({"spec": {"name": "forbidden", "root": leaf("A")}})
                elif entry == "tool-create":
                    await run_workflow_tool({"action": "create", "spec": {"name": "forbidden", "root": leaf("A")}})
                else:
                    await run_workflow_tool({"action": "status"})
            assert not (await agent.runtime.client.list_children(executor.driver, actor=agent.actor)).items
        finally:
            reset_session_id(sid_token)
            reset_current_loop(loop_token)


class SelectionRecorded(Exception):
    pass


@asynccontextmanager
async def capture_consumption(db, *, stop_after_select=False):
    transaction, selected = db.transaction, []

    class Tx:
        def __init__(self, tx):
            self.tx = tx

        def __getattr__(self, name):
            return getattr(self.tx, name)

        async def fetchall(self, sql, params=()):
            rows = await self.tx.fetchall(sql, params)
            selected.append(rows)
            if stop_after_select:
                raise SelectionRecorded()
            return rows

    @asynccontextmanager
    async def capture():
        async with transaction() as tx:
            yield Tx(tx)

    db.transaction = capture
    try:
        yield selected
    finally:
        db.transaction = transaction


async def seed_notifications(executor, body, items):
    """Range-test rows obey real inbox CHECKs; source consumption uses real starts."""
    await executor.port.db.execute(
        f"INSERT INTO {executor.port.q} (session_id,item_id,kind,priority,status,content_json,"
        "meta_json,request_hash,created_at,delivered_send_index,terminal_status,result_send_index,completed_at) VALUES "
        + ",".join("(?,?,?,'normal',?,?,'{}',?,1,?,?,?,?)" for _ in items),
        [value for sid, item, kind, status, send, terminal in items for value in (
            sid, item, kind, status, body, "0" * 64, send, terminal,
            send if terminal is not None else None, 1 if terminal is not None else None,
        )],
    )


async def test_consumer_selects_pending_and_current_delivered_only(registry_store, monkeypatch):
    store, _ = registry_store
    async with executor_fixture(store) as (executor, agent):
        tasks, refs, release = await start_ready(
            executor, agent, [request(executor, name) for name in ("A", "B", "C")], monkeypatch)
        try:
            for ref in refs[:2]:
                await terminalize(executor, ref)
            assert await executor.port.claim_follow_up(executor.ctx.batch) is not None
            await terminalize(executor, refs[2])
            live = await store._db.fetchall(
                f"SELECT id,content_json,status,completed_at FROM {executor.port.q} "
                "WHERE session_id=? AND kind='workflow_node_result' ORDER BY id", (executor.ctx.session_id,))
            assert [row["status"] for row in live] == ["delivered", "delivered", "pending"]
            assert all(row["completed_at"] is None for row in live)
            sid, send = executor.ctx.session_id, executor.ctx.send_index
            excluded = [(sid, "other-send", "workflow_node_result", "delivered", send + 1, None),
                        (sid, "other-kind", "user", "pending", None, None),
                        (sid, "closed", "workflow_node_result", "delivered", send, "completed"),
                        (agent.target.session_id, "other-session", "workflow_node_result", "pending", None, None)]
            await seed_notifications(executor, live[0]["content_json"], excluded)
            async with capture_consumption(store._db) as selected:
                updates = await executor._consume_results()
            assert [row["id"] for row in selected[0]] == [row["id"] for row in live]
            assert len(updates) == 3 and all(entry["result"]["send_index"] == 1 for entry in updates.values())
            assert all(not entry["late_results"] for entry in updates.values())
            for owner, item, _, status, _, terminal in excluded:
                row = await store._db.fetchone(
                    f"SELECT status,terminal_status FROM {executor.port.q} WHERE session_id=? AND item_id=?", (owner, item))
                assert row == {"status": status, "terminal_status": terminal}
            release.set()
            assert all(row["status"] == "completed" for row in await asyncio.gather(*tasks))
        finally:
            executor.ctx.shutdown_interrupted = True
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)


async def test_consumer_production_select_keeps_id_order_and_limit(registry_store, monkeypatch):
    store, _ = registry_store
    async with executor_fixture(store) as (executor, agent):
        tasks, refs, _release = await start_ready(executor, agent, [request(executor)], monkeypatch)
        try:
            await terminalize(executor, refs[0])
            source = await store._db.fetchone(
                f"SELECT content_json FROM {executor.port.q} WHERE session_id=? AND kind='workflow_node_result'",
                (executor.ctx.session_id,))
            sid, send = executor.ctx.session_id, executor.ctx.send_index
            await seed_notifications(executor, source["content_json"], [
                (sid, "skip-closed", "workflow_node_result", "delivered", send, "completed"),
                (sid, "skip-kind", "user", "pending", None, None),
                (sid, "skip-send", "workflow_node_result", "delivered", send + 1, None),
                (agent.target.session_id, "skip-session", "workflow_node_result", "pending", None, None),
            ])
            await seed_notifications(executor, source["content_json"], [
                (sid, f"ordered-{i}", "workflow_node_result", "delivered" if i % 2 else "pending",
                 send if i % 2 else None, None) for i in range(1002)])
            expected = await store._db.fetchall(
                f"SELECT id FROM {executor.port.q} WHERE session_id=? AND "
                "(item_id LIKE 'ordered-%' OR item_id NOT LIKE 'skip-%') "
                "AND kind='workflow_node_result' ORDER BY id", (sid,))
            assert len(expected) == 1003
            # Observe the real fenced SELECT, then roll back before 1000 repeated
            # notification updates. This is not a hand-written candidate query.
            async with capture_consumption(store._db, stop_after_select=True) as selected:
                with pytest.raises(SelectionRecorded):
                    await executor._consume_results()
            assert [row["id"] for row in selected[0]] == [row["id"] for row in expected[:1000]]
            assert all(entry["result"] is None for entry in executor._nodes.values())
        finally:
            executor.ctx.shutdown_interrupted = True
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)


async def test_consumer_still_rejects_unfinished_source(registry_store, monkeypatch):
    store, _ = registry_store
    async with executor_fixture(store) as (executor, agent):
        tasks, refs, _release = await start_ready(executor, agent, [request(executor)], monkeypatch)
        try:
            await seed_notifications(executor, canonical({"input_ref": plain(refs[0]), "status": "completed"}), [
                (executor.ctx.session_id, "unconfirmed", "workflow_node_result", "pending", None, None)])
            with pytest.raises(RuntimeRequestError) as error:
                await executor._consume_results()
            assert error.value.code == "workflow_result_source_not_terminal"
            assert all(entry["result"] is None for entry in executor._nodes.values())
            row = await store._db.fetchone(
                f"SELECT status,completed_at FROM {executor.port.q} WHERE session_id=? AND item_id='unconfirmed'",
                (executor.ctx.session_id,))
            assert row == {"status": "pending", "completed_at": None}
        finally:
            executor.ctx.shutdown_interrupted = True
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)


async def test_recovered_pending_result_keeps_null_completion_and_is_consumed(registry_store, monkeypatch):
    store, _ = registry_store
    async with executor_fixture(store) as (old, agent):
        tasks, refs, _release = await start_ready(old, agent, [request(old)], monkeypatch)
        try:
            await terminalize(old, refs[0])
            assert await old.port.claim_follow_up(old.ctx.batch) is not None
            await store._db.execute(
                f"UPDATE {old.port.s} SET owner_lease_until_ms=0 WHERE session_id=?", (old.ctx.session_id,))
            recovered = await old.port.recover(old.driver, node_id="new", boot_id="new", ttl_s=300)
            assert recovered.execution_state == "queued"
            row = await store._db.fetchone(
                f"SELECT status,completed_at FROM {old.port.q} WHERE session_id=? AND kind='workflow_node_result'",
                (old.ctx.session_id,))
            assert row == {"status": "pending", "completed_at": None}
            batch = await old.port.claim(old.driver, node_id="new", boot_id="new", ttl_s=300)
            assert batch.send_index == 2
            ctx = _SessionExecution(agent.runtime, old.ctx.session_id, batch.descriptor, batch, old.ctx.actor)
            fresh = DurableExecutor(old.worker, ctx, old.env, old.run_id, old.saved)
            updates = await fresh._consume_results()
            assert len(updates) == 1 and next(iter(updates.values()))["result"]["send_index"] == 1
            row = await store._db.fetchone(
                f"SELECT terminal_status,result_send_index,completed_at FROM {old.port.q} "
                "WHERE session_id=? AND kind='workflow_node_result'", (old.ctx.session_id,))
            assert row["terminal_status"] == "completed" and row["result_send_index"] == 2
            assert row["completed_at"] is not None
        finally:
            old.ctx.shutdown_interrupted = True
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
