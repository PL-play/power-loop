"""Accepted wait decisions and their original model evidence commit together."""
from __future__ import annotations

import json
from contextlib import asynccontextmanager

import pytest

from power_loop.runtime import (
    wait_effects,
)
from power_loop.runtime.execution_state import CheckpointPolicy
from power_loop.runtime.jobs import JobManager

from .test_wait_effects_192 import decide
from .test_wait_handoff_192 import handoff, registry_store, runtime, waiting

__all__ = ["handoff", "registry_store", "runtime", "waiting"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def test_wait_decision_remains_usable_after_reopen(handoff):
    h, c = handoff, handoff.c
    h.loop.task_runtime.checkpoint_policy = CheckpointPolicy(interval_s=300)
    await decide(h)
    assert h.model.calls == 2
    reader = JobManager(await c.connect(), now_ms=c.manager.now)
    reply = await wait_effects.apply(reader, c.main, h.event_id)
    assert reply["status"] == "input_enqueued" and not reply["execution_authority"]
    assert h.model.calls == 2


@pytest.mark.parametrize("fault", ["rollback", "ack_loss"])
async def test_failed_or_unknown_decision_commit_preserves_original_response_and_intent(handoff, monkeypatch, fault):
    h, c = handoff, handoff.c
    h.loop.task_runtime.checkpoint_policy = CheckpointPolicy(interval_s=300)
    transaction = c.store._db.transaction
    injected = False

    @asynccontextmanager
    async def fail_once():
        nonlocal injected
        inject = False
        async with transaction() as tx:
            yield tx
            row = await tx.fetchone(f"SELECT event_id FROM {c.manager.events} WHERE event_type='wait_decision'")
            if row is not None and not injected:
                injected = inject = True
                if fault == "rollback":
                    raise OSError("decision checkpoint rolled back")
        if inject:
            raise OSError("decision checkpoint ACK lost")

    monkeypatch.setattr(c.store._db, "transaction", fail_once)
    await decide(h)
    if fault == "rollback":
        # The original failed operation was rolled back; its raw response stays
        # stored. A fresh model must never be used to fabricate the old intent.
        assert not await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE event_type='wait_decision'")
    else:
        assert len(await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE event_type='wait_decision'")) == 1
    assert injected and h.model.calls == 2
    messages = await c.store.load_all_messages(c.main.session_id)
    [tool] = [row for row in messages if row.role == "tool" and row.name == "wait_handoff"]
    reply = json.loads(tool.content)
    assert (reply.get("status") == "accepted") == (fault == "ack_loss")
    assert not await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE event_type='wait_effect'")
