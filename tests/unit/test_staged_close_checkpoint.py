"""A close must publish its suffix even after an earlier batch in the same TX."""
from __future__ import annotations

import json

import pytest

from .test_execution_checkpoints import case, complete
from .test_job_registry_192 import registry_store

__all__ = ["registry_store"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.mark.parametrize("outcome", ["commit", "rollback", "ack_lost"])
async def test_staged_open_batch_and_close_keep_every_record_until_transaction_outcome(registry_store, outcome):
    store, _, state, journal, _, _ = await case(registry_store)
    complete(state, 1)
    hooks, undo = [], []
    def defer(commit, rollback):
        hooks.append(commit)
        undo.append(rollback)
    try:
        async with store._db.transaction() as tx:
            await state.checkpoint(force=True, transaction=tx, defer=defer)
            assert state.cursor == 0
            await state.close(transaction=tx, defer=defer)
            head = await tx.fetchone(f"SELECT batch_no,summary_json FROM {journal.heads} WHERE job_id=? AND generation=?",
                                    (state.identity.job_id, state.identity.generation))
            assert json.loads(head["summary_json"])["closed"]
            assert head["batch_no"] == 2 and state.cursor == 0
            if outcome == "rollback":
                raise OSError("staged close rollback")
        if outcome == "ack_lost":
            raise OSError("staged close committed but ACK lost")
    except BaseException as exc:
        for hook in reversed(undo):
            hook()
        if not isinstance(exc, OSError):
            raise
        assert outcome != "commit" and state.cursor == 0
    else:
        for hook in hooks:
            hook()
        assert outcome == "commit" and state.cursor == 2
    if outcome == "ack_lost":
        from power_loop.runtime.jobs import JobError
        with pytest.raises(JobError, match="execution_checkpoint_conflict"):
            await state.close()
        saved = await journal.inspect(state.identity)
        assert saved['summary']['closed'] and saved['summary']['completed'] == 1
        return
    await state.close()
    saved = await journal.inspect(state.identity)
    assert saved["summary"]["closed"] and saved["summary"]["completed"] == 1
    assert saved["batch_no"] == state.cursor
    assert not state._dirty and state._pending is None


async def test_savepoint_rollback_retains_only_the_uncommitted_suffix_for_retry(registry_store):
    store, _, state, journal, _, _ = await case(registry_store)
    complete(state, 1)
    hooks, undo = [], []
    def defer(commit, rollback):
        hooks.append(commit)
        undo.append(rollback)
    async with store._db.transaction() as tx:
        await state.checkpoint(force=True, transaction=tx, defer=defer)
        mark = len(hooks)
        await tx.execute("SAVEPOINT test_staged_suffix")
        complete(state, 2)
        await state.checkpoint(force=True, transaction=tx, defer=defer)
        await tx.execute("ROLLBACK TO SAVEPOINT test_staged_suffix")
        for hook in reversed(undo[mark:]):
            hook()
        del hooks[mark:], undo[mark:]
    for hook in hooks:
        hook()
    assert state.cursor == 1 and state.checkpoint_failed
    saved = await journal.inspect(state.identity)
    assert saved["summary"]["completed"] == 1 and not saved["summary"]["closed"]
    await state.close()
    saved = await journal.inspect(state.identity)
    assert saved["summary"]["completed"] == 2 and saved["summary"]["closed"]
    assert state.cursor == 3 and not state._dirty and state._pending is None
