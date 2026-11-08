"""Round accounting shares the first response append without losing originals."""
from __future__ import annotations

from contextlib import asynccontextmanager

import pytest

from power_loop import AgentLoopConfig, LLMResponse, LLMService, LLMTokenUsage
from power_loop.agent.sink import SQLiteSink

from .test_job_registry_192 import registry_store
from .test_send_sql_profile import Profiled

__all__ = ["registry_store"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def usage_rows(store, sid):
    return await store._db.fetchall(f"SELECT * FROM {store.t.usage_rounds} WHERE session_id=?", (sid,))


@pytest.mark.parametrize("fault", ["rollback", "ack_loss"])
async def test_failed_or_unknown_message_commit_has_the_same_usage_outcome(registry_store, monkeypatch, fault):
    store, connect = registry_store
    sid = await store.create_session()
    sink = SQLiteSink(store, sid)
    sink.send_index = 1
    transaction = store._db.transaction

    @asynccontextmanager
    async def fail():
        async with transaction() as tx:
            yield tx
            if fault == "rollback":
                raise OSError("before_commit")
        raise OSError("committed_ack_lost")

    with monkeypatch.context() as patch:
        patch.setattr(store._db, "transaction", fail)
        with pytest.raises(OSError, match="before_commit|committed_ack_lost"):
            await sink.on_message_appended({"role": "assistant", "content": "original", "send_index": 1,
                "_round_usage": {"prompt_tokens": 7, "completion_tokens": 5, "total_tokens": 12}}, round_index=0)
    cold = await connect(schema="verify")
    assert len(await cold.load_all_messages(sid)) == len(await usage_rows(cold, sid)) == int(fault == "ack_loss")


async def test_round_end_is_idempotent_after_usage_commits_with_message(registry_store):
    store, _ = registry_store
    sid = await store.create_session()
    sink = SQLiteSink(store, sid)
    sink.send_index = 1
    usage = {"prompt_tokens": 7, "completion_tokens": 5, "total_tokens": 12}
    original = store._db
    store._db = counted = Profiled(original)
    try:
        await sink.on_message_appended({"role": "assistant", "content": "original", "send_index": 1,
                                       "_round_usage": usage}, round_index=0)
        assert counted.transactions == 1
        before = len(counted.records), counted.transactions
        for _ in range(3):
            await sink.on_round_ended(0, usage=usage)
        assert (len(counted.records), counted.transactions) == before
        [row] = await usage_rows(store, sid)
        assert row["total_tokens"] == 12
    finally:
        store._db = original


async def test_empty_response_rounds_keep_usage_and_private_accounting_never_reaches_the_model(registry_store, runtime_agent):
    store, _ = registry_store

    class Model(LLMService):
        calls = 0
        async def complete(self, request, **kwargs):
            self.calls += 1
            assert all("_round_usage" not in row for row in request.messages)
            return LLMResponse(raw_text="" if self.calls == 1 else "done", token_usage=LLMTokenUsage(
                prompt_tokens=7, completion_tokens=5, total_tokens=12))
        def stream(self, request):
            raise AssertionError("nonstream fixture")
        async def close(self):
            pass

    model = Model()
    loop = await runtime_agent(model=model, store=store, config=AgentLoopConfig(max_rounds=3, compactor=None))
    try:
        sid = loop.target.session_id
        assert (await loop.send("respond")).status == "completed"
        rows = await usage_rows(store, sid)
        assert {(row["round_index"], row["total_tokens"]) for row in rows} == {(0, 12), (1, 12)}
        assert model.calls == 2
    finally:
        await loop.aclose()
