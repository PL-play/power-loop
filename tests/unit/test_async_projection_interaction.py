"""Async × projection interaction — the cross-cutting behavior the per-feature tests miss.

These pin the contract documented in docs/user-guide/async-orchestration.md:

- a ``follow_up`` on an IDLE session is a fresh send → its own ``send_index`` → projected as its own
  send (this is the path a timer/background-completion wake takes when the loop has gone idle);
- ``follow_up`` on idle returns a ``StatefulResult`` (ran), not ``FollowUpQueued`` (queued);
- background-task results re-enter via ``BackgroundRuntimeProjector``: unseen finished tasks are
  injected once as a ``<background_updates>`` message and then marked seen (no re-injection).

Reuses the scripted-LLM helpers from test_stateful_loop / test_history_projection_loop.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from power_loop._vendor.llm_client.interface import LLMResponse
from power_loop.agent.follow_up import FollowUpQueued
from power_loop.runtime.history_projector import DefaultDeterministicProjector
from power_loop.runtime.runtime_state import BackgroundRuntimeProjector
from power_loop.runtime.store.store import SessionStore
from tests.unit.test_history_projection_loop import _projector_loop
from tests.unit.test_stateful_loop import _Scripted, _tool_resp


@pytest.fixture
async def store() -> AsyncIterator[SessionStore]:
    s = await SessionStore.open(":memory:")
    yield s
    await s.close()


@pytest.mark.asyncio
async def test_follow_up_on_idle_opens_new_projected_send(store: SessionStore) -> None:
    """A wake (timer/background completion) on an idle session = a new send with its own
    send_index, projected independently — not appended to the prior send."""
    llm = _Scripted(responses=[
        _tool_resp("c1", "echo", '{"text": "a"}'),
        LLMResponse(raw_text="done1"),
        LLMResponse(raw_text="done2"),
    ])
    loop = _projector_loop(store, llm, DefaultDeterministicProjector())
    sid = await loop.new_session()

    await loop.send("first", session_id=sid)
    assert {(r.send_index, r.kind) for r in await store.load_project_messages(sid)} == {
        (1, "user"), (1, "project"),
    }

    # follow_up on an IDLE session falls through to send() and runs the loop.
    result = await loop.follow_up("second", session_id=sid)
    assert not isinstance(result, FollowUpQueued)  # ran, didn't queue

    # The wake became its OWN send (send_index 2), projected separately from send 1.
    assert {(r.send_index, r.kind) for r in await store.load_project_messages(sid)} == {
        (1, "user"), (1, "project"), (2, "user"), (2, "project"),
    }


@pytest.mark.asyncio
async def test_background_updates_reenter_once_and_are_marked_seen(store: SessionStore) -> None:
    """Background results re-enter the loop via BackgroundRuntimeProjector: a finished task is
    injected exactly once as <background_updates>, then marked seen so it never re-injects."""
    # new_session needs a loop; a bare store session is enough here.
    from power_loop.runtime.store.types import SessionKind

    sid = await store.create_session(kind=SessionKind.ROOT)
    await store.upsert_background_task(
        sid, task_id="bg-1", command="build", status="done", output_tail="BUILD OK",
    )

    projector = BackgroundRuntimeProjector()
    first = await projector.project(store=store, session_id=sid, round_index=0, context=None)
    assert len(first) == 1
    body = str(first[0]["content"])
    assert "<background_updates>" in body and "bg-1" in body and "BUILD OK" in body

    # Marked seen → the next round does not re-inject the same finished task.
    second = await projector.project(store=store, session_id=sid, round_index=1, context=None)
    assert second == []


@pytest.mark.asyncio
async def test_background_projector_can_keep_reinjecting_when_mark_seen_false(store: SessionStore) -> None:
    """mark_seen=False is the 'always show current state' mode — useful for a live status panel."""
    from power_loop.runtime.store.types import SessionKind

    sid = await store.create_session(kind=SessionKind.ROOT)
    await store.upsert_background_task(
        sid, task_id="bg-9", command="serve", status="running", output_tail="listening…",
    )
    projector = BackgroundRuntimeProjector(mark_seen=False)
    assert len(await projector.project(store=store, session_id=sid, round_index=0, context=None)) == 1
    # Still injected on the next round because it was never marked seen.
    assert len(await projector.project(store=store, session_id=sid, round_index=1, context=None)) == 1
