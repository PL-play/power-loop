"""Finite children cannot arm or deliver legacy timers outside their owner."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock

import pytest

from power_loop import AgentLoopConfig, StatefulAgentLoop, TimerRunner
from power_loop.core.agent_context import (
    reset_current_loop,
    reset_session_id,
    set_current_loop,
    set_session_id,
)
from power_loop.tools import default_tools

from .test_job_registry_192 import registry_store
from .test_timers import _Scripted

__all__ = ["registry_store"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.fixture
async def timer_role(registry_store):
    store, _ = registry_store
    model = _Scripted()
    model.complete = AsyncMock(wraps=model.complete)
    loop = StatefulAgentLoop(
        llm=model, store=store, config=AgentLoopConfig(delegation_policy="single_level")
    )
    parent = await loop.new_session()
    return store, loop, model, parent


async def create(loop, sid, entry, recurring=False):
    token = set_session_id(sid)
    current = set_current_loop(loop)
    try:
        if entry == "tool":
            return await default_tools.run_schedule_wakeup(
                action="schedule", delay_seconds=10, note="original note",
                every_seconds=60 if recurring else None,
            )
        return await loop.schedule_timer(
            sid, delay_s=10, note="original note", interval_s=60 if recurring else None
        )
    finally:
        reset_current_loop(current)
        reset_session_id(token)


@pytest.mark.parametrize("entry", ["tool", "sdk"])
@pytest.mark.parametrize("role", ["subagent", "workflow_node", "workflow"])
@pytest.mark.parametrize("recurring", [False, True])
async def test_child_creation_refused_before_any_timer_is_written(timer_role, entry, role, recurring):
    store, loop, _, parent = timer_role
    sid = await store.create_session(parent_session_id=parent, metadata={
        "execution_role": role, **({"kind": "wf_driver"} if role == "workflow" else {})
    })
    with pytest.raises(ValueError, match="execution_role_forbids_timer"):
        await create(loop, sid, entry, recurring)
    assert await store.list_timers(sid) == []
    assert await store.list_timers(parent) == []


@pytest.mark.parametrize("entry", ["tool", "sdk"])
async def test_finite_main_and_legacy_child_reminders_remain_available(timer_role, entry):
    store, loop, _, parent = timer_role
    await create(loop, parent, entry, recurring=True)
    [timer] = await store.list_timers(parent)
    assert timer.status == "armed" and timer.interval_s == 60
    loop.config.delegation_policy = "legacy"
    child = await store.create_session(parent_session_id=parent, metadata={"execution_role": "subagent"})
    await create(loop, child, entry)
    [timer] = await store.list_timers(child)
    assert timer.status == "armed" and timer.note == "original note"


async def test_child_wrapper_cannot_arm_a_main_session_instead(timer_role):
    store, loop, _, parent = timer_role
    child = await store.create_session(parent_session_id=parent, metadata={"execution_role": "subagent"})
    token = set_session_id(child)
    try:
        with pytest.raises(ValueError, match="execution_role_forbids_timer"):
            await loop.schedule_timer(parent, delay_s=10, note="wrong owner")
    finally:
        reset_session_id(token)
    assert await store.list_timers(parent) == []


async def test_orphaned_node_keeps_its_role_and_cannot_arm_a_timer(timer_role):
    store, loop, _, parent = timer_role
    child = await store.create_session(parent_session_id=parent, metadata={"execution_role": "workflow_node"})
    await store.close_session(parent, cascade=False)
    assert (await store.get_session(child)).parent_session_id is None
    with pytest.raises(ValueError, match="execution_role_forbids_timer"):
        await create(loop, child, "sdk")
    assert await store.list_timers(child) == []


@pytest.mark.parametrize("host_delivery", [False, True])
async def test_existing_child_timer_is_retained_without_hook_delivery_or_model_call(timer_role, host_delivery):
    store, loop, model, parent = timer_role
    child = await store.create_session(parent_session_id=parent, metadata={"execution_role": "subagent"})
    original = await store.create_timer(child, due_at=1, note="old child request", interval_s=60)
    deliveries = []

    async def delivery(ctx):
        deliveries.append(ctx)

    runner = TimerRunner(loop, delivery=delivery if host_delivery else None)
    assert await runner.scan_once() == 0
    saved = await store.get_timer(child, original.timer_id)
    assert saved.status == "armed" and saved.due_at > int(time.time() * 1000)
    assert saved.fire_count == 0 and saved.note == original.note
    assert model.complete.call_count == 0 and deliveries == []
    assert await store.load_all_messages(child) == []
    # Query/cancel remains possible to reconcile old rows without paid work.
    token = set_session_id(child)
    current = set_current_loop(loop)
    try:
        assert "old child request" in await default_tools.run_schedule_wakeup(action="list")
        assert "cancelled" in await default_tools.run_schedule_wakeup(
            action="cancel", timer_id=original.timer_id
        )
    finally:
        reset_current_loop(current)
        reset_session_id(token)
    assert (await store.get_timer(child, original.timer_id)).status == "cancelled"
