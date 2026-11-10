"""A stale claim cannot become a transcript row or a paid model input."""

from __future__ import annotations

import asyncio
import contextvars
import json
from contextlib import asynccontextmanager

import pytest

from power_loop.contracts.errors import InboxClaimChanged

from .test_job_registry_192 import registry_store

__all__ = ["registry_store"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.mark.parametrize("changed", ["void", "release"])
async def test_stale_claim_rolls_back_message_counter_and_all_delivery_markers(registry_store, changed):
    store, _ = registry_store
    sid = await store.create_session()
    await store.inbox_put(
        sid,
        [
            {"item_id": "drop", "kind": "relay", "content": "withdrawn instruction"},
            {"item_id": "keep", "kind": "relay", "content": "valid instruction"},
        ],
    )
    claimed = await store.inbox_claim(sid)
    token = claimed[0]["claim_token"]
    before = await store.get_state(sid)
    if changed == "void":
        assert await store.inbox_void(sid, ["drop"]) == 1
    elif changed == "release":
        assert await store.inbox_release(sid, token) == 2
    with pytest.raises(InboxClaimChanged, match="claim"):
        await store.append_message(
            sid,
            role="user",
            content="withdrawn instruction\nvalid instruction",
            inbox={"claim_token": token, "ids": [r["id"] for r in claimed]},
        )
    assert await store.load_all_messages(sid) == []
    assert (await store.get_state(sid)).next_seq == before.next_seq
    assert all(r["status"] != "delivered" for r in await store.inbox_items(sid))


async def test_committed_delivery_wins_against_later_void_and_remains_deduplicated(registry_store):
    store, _ = registry_store
    sid = await store.create_session()
    await store.inbox_put(sid, [{"item_id": "one", "content": "delivered"}])
    [item] = await store.inbox_claim(sid)
    seq = await store.append_message(
        sid, role="user", content="delivered", inbox={"claim_token": item["claim_token"], "ids": [item["id"]]}
    )
    assert await store.inbox_void(sid, ["one"]) == 0
    [row] = await store.inbox_items(sid)
    assert row["status"] == "delivered" and row["delivered_seq"] == seq
    assert await store.inbox_put(sid, [{"item_id": "one", "content": "replacement"}]) == []


@pytest.mark.parametrize("bad", ["empty", "duplicate", "bool", "missing", "foreign", "token"])
async def test_invalid_batch_cannot_partially_deliver_or_write_audit(registry_store, bad):
    store, _ = registry_store
    sid, other = await store.create_session(), await store.create_session()
    await store.inbox_put(sid, [{"item_id": "one", "content": "one"}])
    await store.inbox_put(other, [{"item_id": "other", "content": "foreign"}])
    [row], [foreign] = await store.inbox_claim(sid), await store.inbox_claim(other)
    inbox = {"ids": [row["id"]], "claim_token": row["claim_token"]}
    if bad == "empty":
        inbox["ids"] = []
    elif bad == "duplicate":
        inbox["ids"] *= 2
    elif bad == "bool":
        inbox["ids"] = [True]
    elif bad == "missing":
        inbox["ids"].append(2**63 - 1)
    elif bad == "foreign":
        inbox["ids"].append(foreign["id"])
    else:
        inbox["claim_token"] = ""
    with pytest.raises(InboxClaimChanged):
        await store.append_message(
            sid,
            role="user",
            content="must not exist",
            inbox=inbox,
            hook_injected={"hook_point": "llm_before", "kind": "test", "payload": {"text": "no"}},
        )
    assert await store.load_all_messages(sid) == []
    assert await store.list_hook_events(sid) == []
    assert (await store.get_state(sid)).next_seq == 1
    assert [r["status"] for r in await store.inbox_items(sid)] == ["claimed"]
    assert [r["status"] for r in await store.inbox_items(other)] == ["claimed"]


async def test_delivery_transaction_serializes_concurrent_void_from_another_connection(registry_store, monkeypatch):
    store, connect = registry_store
    other = await connect()
    sid = await store.create_session()
    await store.inbox_put(sid, [{"item_id": "one", "content": "one"}])
    [row] = await store.inbox_claim(sid)
    held, release, void_started = asyncio.Event(), asyncio.Event(), asyncio.Event()
    original = type(store._db).transaction

    class HeldTransaction:
        def __init__(self, tx):
            self.tx = tx

        async def execute(self, sql, params=()):
            count = await self.tx.execute(sql, params)
            if "SET status='delivered'" in sql:
                held.set()
                await release.wait()
            return count

        def __getattr__(self, name):
            return getattr(self.tx, name)

    @asynccontextmanager
    async def transaction(db):
        if db is other._db:
            void_started.set()  # Retraction now locks session state before its queue UPDATE.
        async with original(db) as tx:
            yield HeldTransaction(tx) if db is store._db else tx

    monkeypatch.setattr(type(store._db), "transaction", transaction)
    async with asyncio.timeout(15):
        delivery = asyncio.create_task(
            store.append_message(
                sid, role="user", content="one", inbox={"ids": [row["id"]], "claim_token": row["claim_token"]}
            )
        )
        void = None
        try:
            await held.wait()
            void = asyncio.create_task(other.inbox_void(sid, ["one"]))
            await void_started.wait()
            assert not void.done()
            release.set()
            seq, count = await asyncio.gather(delivery, void)
            assert count == 0
            [saved] = await store.inbox_items(sid)
            assert saved["status"] == "delivered" and saved["delivered_seq"] == seq
        finally:
            release.set()
            for task in [delivery, void]:
                if task is not None and not task.done():
                    task.cancel()
            await asyncio.gather(*[t for t in [delivery, void] if t is not None], return_exceptions=True)




@pytest.mark.parametrize('winner', ['cancel', 'start'])
async def test_runtime_start_and_cancel_serialize_on_original_input(registry_store, winner):
    from power_loop import RuntimeClient

    from .test_runtime_messaging_203 import deadline, running
    store, connect = registry_store
    other = await connect()
    async with running(store, 'memory') as (runtime, model, target, actor):
        model.release.set()
        first = await runtime.client.send(target, 'first', message_id='first', actor=actor)
        original_result = await runtime.client.wait(first.input_ref, actor=actor, deadline_ms=deadline())
        while runtime.worker._tasks:
            await asyncio.wait(tuple(runtime.worker._tasks))
        model.release.clear()
        held, release = asyncio.Event(), asyncio.Event()
        if winner == 'cancel':
            build = runtime.factory._builders['test.agent@1']
            async def paused_build(*args):
                held.set()
                await release.wait()
                return await build(*args)
            runtime.factory._builders['test.agent@1'] = paused_build
        else:
            bind = runtime.worker.inputs.bind_input_job
            async def paused_bind(*args, **kwargs):
                await bind(*args, **kwargs)
                held.set()
                await release.wait()
            runtime.worker.inputs.bind_input_job = paused_bind
        client = RuntimeClient(store=other, transport=runtime.transport)
        cancel = None
        try:
            second = await client.send(target, 'second', message_id='second', actor=actor)
            await asyncio.wait_for(held.wait(), 5)
            cancel = asyncio.create_task(client.cancel(second.input_ref, actor=actor))
            if winner == 'cancel':
                assert (await asyncio.wait_for(cancel, 5)).job_ref is None
            else:
                await asyncio.sleep(.02)
                assert not cancel.done()
            release.set()
            await cancel
            result = await client.wait(second.input_ref, actor=actor, deadline_ms=deadline())
            assert result.status == 'cancelled'
            if winner == 'cancel':
                assert result.job_ref is None and len(model.requests) == 1
            else:
                assert result.job_ref is not None and result.job_ref != original_result.job_ref
            assert await client.wait(first.input_ref, actor=actor, deadline_ms=deadline()) == original_result
        finally:
            release.set()
            if cancel and not cancel.done():
                cancel.cancel()
                await asyncio.gather(cancel, return_exceptions=True)
            await client.aclose()


@pytest.mark.parametrize("keep", [False, True])
@pytest.mark.parametrize("image", [False, True])
async def test_runtime_append_retraction_keeps_original_send(registry_store, monkeypatch, keep, image):
    from power_loop import RequestOptions, RuntimeClient
    from power_loop.core.pipeline import AgentPipeline
    from tests.runtime_fixture import RunningAgent

    from .test_inbox import LLMResponse, _Gate, _registry, _tool_call, _wait_calls
    from .test_runtime_messaging_203 import deadline

    store, connect = registry_store
    other = await connect()
    model = _Gate([_tool_call("c1"), LLMResponse(raw_text="accepted")])
    first = model.gate(0)
    agent = await RunningAgent(store, model, _registry()).start()
    client = RuntimeClient(store=other, transport=agent.runtime.transport)
    held, release = asyncio.Event(), asyncio.Event()
    append = store.append_message
    observed, rolled_back = [], []
    pipeline_append = AgentPipeline._append_message

    async def observe_memory(pipeline, *args, **kwargs):
        before = (list(pipeline.history), pipeline._tok_len, pipeline._tok_total,
                  dict(pipeline._image_rounds), list(pipeline.sink._history_seqs), list(pipeline.sink._history_ord))
        accepted = await pipeline_append(pipeline, *args, **kwargs)
        if not accepted:
            assert (pipeline.history, pipeline._tok_len, pipeline._tok_total,
                    pipeline._image_rounds, pipeline.sink._history_seqs, pipeline.sink._history_ord) == before
            rolled_back.append(True)
        return accepted

    async def pause_append(sid, **kwargs):
        if kwargs.get("inbox") and not observed:
            observed.append(kwargs["inbox"])
            held.set()
            await release.wait()
        return await append(sid, **kwargs)

    monkeypatch.setattr(store, "append_message", pause_append)
    monkeypatch.setattr(AgentPipeline, "_append_message", observe_memory)
    run = asyncio.create_task(agent.send("start", options=RequestOptions(system_prompt="original config")))
    try:
        async with asyncio.timeout(15):
            await _wait_calls(model, 1)
            body = {"role": "user", "content": [{"type": "attachment", "name": "DROP.png", "url": "https://invalid.test/DROP.png"}]} if image else "DROP"
            drop = await client.send(agent.target, body, message_id="drop", actor=agent.actor)
            survivor = await client.send(agent.target, "KEEP", message_id="keep", actor=agent.actor) if keep else None
            ctx = agent.runtime.worker._active[agent.target.session_id]
            while len(ctx.append_refs) < 1 + int(keep):
                await asyncio.sleep(.005)
            original_job = ctx.job_ref
            first.set()
            await held.wait()
            assert set(observed[0]["item_ids"]) == ({"drop", "keep"} if keep else {"drop"})
            cancelled = await asyncio.create_task(client.cancel(drop.input_ref, actor=agent.actor), context=contextvars.Context())
            assert cancelled.job_ref is None
            release.set()
            result = await run
            assert result.status == "completed", result.error
            assert result.job_ref == original_job
            assert rolled_back == [True]
            assert len(model.calls) == 2
            assert "original config" in model.system_prompts[-1]
            assert "DROP" not in json.dumps(model.calls)
            assert "DROP" not in json.dumps([r.content for r in await store.load_all_messages(ctx.session_id)])
            if survivor:
                surviving_result = await client.wait(survivor.input_ref, actor=agent.actor, deadline_ms=deadline())
                assert surviving_result.job_ref == original_job
                rows = await store.load_all_messages(ctx.session_id)
                assert sum("KEEP" in (r.content or "") for r in rows) == 1
                assert "KEEP" in json.dumps(model.calls[-1])
            await agent.drain()
            assert (await store.inbox_counts(ctx.session_id)) == {"pending": 0, "steer": 0}
    finally:
        first.set()
        release.set()
        await client.aclose()
        await agent.aclose()
        await asyncio.gather(run, return_exceptions=True)
