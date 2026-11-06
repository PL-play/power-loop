"""Actual later owner generations retain their original call and review budget."""

import asyncio
import json

import pytest

from power_loop import LLMResponse, LLMTokenUsage
from power_loop.runtime import delegation

from .test_job_registry_192 import registry_store
from .test_original_owner_review_192 import original_owner_case

__all__ = ["registry_store"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


def response(text="", calls=None):
    return LLMResponse(
        raw_text=text,
        tool_calls=calls,
        token_usage=LLMTokenUsage(prompt_tokens=5, completion_tokens=5, total_tokens=10),
    )




async def complete_initial(c, kind):
    s = c.state
    await c.loop.send("Use launch_owned once for initial setup. Then reply QUEUED.", s.root)
    if kind == "workflow_node":
        await asyncio.wait_for(s.handles[0].task, s.timeout)
        [node] = await c.manager.db.fetchall(
            f"SELECT * FROM {c.manager.jobs} WHERE kind='workflow_node' AND title='BUILD'"
        )
        s.owner_id = node["job_id"]
        s.child_sid = node["session_id"]
        # Node sessions are stored separately from their logical job row.
        rows = await c.manager.db.fetchall(
            f"SELECT payload_json FROM {c.manager.events} WHERE job_id=? AND event_type='node_session'", (s.owner_id,)
        )
        s.child_sid = json.loads(rows[0]["payload_json"])["session_id"]
    else:
        tasks = [r.task for (key, _), r in delegation._RUNNING.items() if key == id(c.store) and r.task]
        if tasks:
            await asyncio.wait_for(asyncio.gather(*tasks), s.timeout)
        s.owner_id = s.handles[0]["job_id"]
        s.child_sid = s.handles[0]["session_id"]
    s.phase = "drain"
    await c.loop.flush_follow_ups(s.root, tools=[], max_rounds=1)
    s.old_revision = await c.manager.get_result_revision(c.runtime.actor(s.root), s.owner_id, 1)
    s.old_members = await c.manager.get_execution_membership(c.runtime.actor(s.root), s.owner_id, 1)
    assert s.old_members["sealed"]
    s.baseline_notifications = len(s.notifications)
    s.baseline_after = s.counts["AFTER"]
    s.phase = "continue"
    s.entered.clear()
    s.release.clear()
    await c.loop.follow_up(
        "Use continue_owned exactly once for the original task. Then reply QUEUED.", session_id=s.root
    )
    if not hasattr(s, "continued"):
        rows = await c.store.load_all_messages(s.root)
        pytest.fail(str([(r.role, r.content) for r in rows if r.role == "tool"])[-3000:])
    if kind == "workflow_node" and s.continued.task.done() and not s.original.is_set():
        pytest.fail(str(await c.manager.get_job(c.runtime.actor(s.root), s.owner_id)))
    await asyncio.wait_for(s.original.wait(), s.timeout)


async def settle(c, kind):
    if kind == "workflow_node":
        await asyncio.wait_for(c.state.continued.task, c.state.timeout)
    else:
        tasks = [r.task for (key, _), r in delegation._RUNNING.items() if key == id(c.store) and r.task]
        if tasks:
            await asyncio.wait_for(asyncio.gather(*tasks), c.state.timeout)










async def start_collected(c, kind):
    await c.loop.send("Use launch_owned exactly once for initial setup. Then reply QUEUED.", c.state.root)
    if c.state.original.is_set():
        return
    if kind == "workflow_node":
        task = c.state.handles[0].task
    else:
        run = delegation._RUNNING.get((id(c.store), c.state.handles[0]["session_id"]))
        task = run.task if run is not None else None
    if task is None or task.done():
        raise AssertionError("Original owner stopped before the collected input model phase")
    waiting = asyncio.create_task(c.state.original.wait())
    try:
        done, _ = await asyncio.wait((waiting, task), timeout=c.state.timeout, return_when=asyncio.FIRST_COMPLETED)
        if c.state.original.is_set():
            return
        if task in done:
            raise AssertionError("Original owner stopped before the collected input model phase")
        raise TimeoutError("Original owner still active without processing collected input")
    finally:
        if not waiting.done():
            waiting.cancel()
        await asyncio.gather(waiting, return_exceptions=True)






@pytest.mark.parametrize("error", [RuntimeError("original-error"), asyncio.CancelledError("original-stop")])
async def test_shared_invocation_preserves_unrelated_exception_and_cancellation(error):
    from power_loop.runtime.original_owner_review import invoke

    calls = []

    async def operation():
        calls.append("once")
        raise error

    with pytest.raises(type(error)) as caught:
        await invoke(object(), "original-session", operation)
    assert caught.value is error and calls == ["once"]


async def test_fatal_sdk_exit_with_original_abort_ticket_does_not_dispatch_a_review(registry_store, monkeypatch):
    from contextlib import asynccontextmanager

    from . import test_original_owner_review_192 as original_fixture
    from .test_original_owner_review_192 import wait_owned

    class FatalStop(BaseException):
        pass

    from power_loop.runtime import original_owner_review

    reviews = []
    handle = original_owner_review.try_handle

    async def traced(*args, **kwargs):
        reviews.append(args[2])
        return await handle(*args, **kwargs)

    monkeypatch.setattr(original_owner_review, "try_handle", traced)
    inject = original_fixture.fail_plan_once

    def fatal_fault(manager, stage):
        inject(manager, stage)
        original = manager._write

        @asynccontextmanager
        async def fatal_write():
            try:
                async with original() as tx:
                    yield tx
            except OSError as exc:
                if str(exc) == "test_plan_seal":
                    raise FatalStop("Original controller must stop") from exc
                raise

        manager._write = fatal_write

    monkeypatch.setattr(original_fixture, "fail_plan_once", fatal_fault)
    c = await original_owner_case(registry_store, "subagent")
    initial = set(delegation._TASKS)
    c.state.release.set()
    try:
        await c.loop.send("Launch original task once.", c.state.root)
        await wait_owned(c, "subagent", initial)
        assert not reviews
        assert not c.state.entered.is_set() and c.state.counts["BUILD"] == 1
        source = await c.manager.get_job(c.runtime.actor(c.state.root), c.state.source)
        owner = await c.manager.get_job(c.runtime.actor(c.state.root), source.control_owner_job_id)
        assert source.status == "model_closed" and source.result_status == "none"
        assert owner.result_status == "none" and not c.state.effects
        assert (await c.store.get_state(source.session_id)).pending
    finally:
        c.state.release.set()
        await c.runtime.shutdown(timeout_s=3)
        await c.loop.aclose(drain_timeout_s=0.1)
