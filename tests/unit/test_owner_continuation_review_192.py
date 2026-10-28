"""Actual later owner generations retain their original call and review budget."""

import asyncio
import json

import pytest

from power_loop import LLMResponse, LLMTokenUsage, ToolDefinition
from power_loop.core.agent_context import get_current_loop, get_session_id
from power_loop.runtime import delegation
from power_loop.runtime.tool_contracts import FunctionRuntimeTool, ToolDispatchPolicy, ToolRuntimeContract

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


async def continuation_case(
    registry_store, kind, *, allowed_calls=3, stage="seal", provider=None, hide=False, fail_review=False
):
    c = await original_owner_case(
        registry_store,
        kind,
        allowed_calls=allowed_calls,
        stage=stage,
        provider=provider,
        hide=hide,
        fail_review=fail_review,
    )
    s = c.state
    s.phase, s.continue_calls, s.initial_calls = "initial", 0, 0
    s.timeout = 120 if provider else 30
    original = c.model.complete

    async def complete(request, **kwargs):
        sid = get_session_id()
        if sid != s.root and "ROLE-BUILD" in request.system_prompt and s.phase == "initial":
            s.initial_calls += 1
            return await provider.complete(request, **kwargs) if provider else response("INITIAL_READY")
        if sid == s.root and s.phase == "continue":
            s.continue_calls += 1
            calls = (
                [
                    {
                        "id": "actual-continuation",
                        "type": "function",
                        "function": {"name": "continue_owned", "arguments": "{}"},
                    }
                ]
                if s.continue_calls == 1
                else None
            )
            return (
                await provider.complete(request, **kwargs) if provider else response("" if calls else "QUEUED", calls)
            )
        return await original(request, **kwargs)

    c.model.complete = complete
    build_prompt = (
        "ROLE-BUILD\nInitial setup only returns INITIAL_READY. For later accepted instructions, "
        "use the actual native function-call mechanism when a tool is offered. "
        "Never write XML tool tags or claim an action happened without an actual tool result."
    )
    if kind == "workflow_node":
        from power_loop.workflow import create_workflow

        async def launch_native():
            workflow = create_workflow(
                {
                    "name": "later-owner",
                    "root": {
                        "type": "sequence",
                        "steps": [
                            {
                                "type": "agent",
                                "id": "BUILD",
                                "input": "Initial setup: return INITIAL_READY as plain text.",
                                "continue_on_error": True,
                                "spec": {
                                    "name": "build",
                                    "system_prompt": build_prompt,
                                    "tools": ["touch_once"],
                                    "max_rounds": 3,
                                },
                            },
                            {
                                "type": "agent",
                                "id": "AFTER",
                                "input": "Acknowledge the completed upstream step.",
                                "spec": {"name": "after", "system_prompt": "ROLE-AFTER", "tools": [], "max_rounds": 2},
                            },
                        ],
                    },
                },
                parent_loop=get_current_loop(),
                parent_session_id=get_session_id(),
            )
            handle = await workflow.start(detached=True)
            s.handles.append(handle)
            return json.dumps({"job_id": handle.job_id, "run_id": handle.run_id, "status": "accepted"})

        old = c.loop.tool_registry.get("launch_owned")
        c.loop.tool_registry.register(
            old.definition,
            FunctionRuntimeTool(
                launch_native,
                ToolRuntimeContract("example.launch-native-later", allowed_roles=("main",)),
                policy=lambda args: ToolDispatchPolicy(True),
            ),
            overwrite=True,
        )
    else:
        from power_loop import AgentSpec

        async def launch_native():
            result = await delegation.start(
                get_current_loop(),
                AgentSpec(
                    name="build", system_prompt=build_prompt, tools=["touch_once"], max_rounds=3, lifecycle="linked"
                ),
                "Initial setup only: reply with INITIAL_READY as plain text. Perform no actions.",
                on_complete=notified,
            )
            s.handles.append(result)
            return json.dumps(result)

        old = c.loop.tool_registry.get("launch_owned")
        c.loop.tool_registry.register(
            old.definition,
            FunctionRuntimeTool(
                launch_native,
                ToolRuntimeContract("example.launch-native-later", allowed_roles=("main",)),
                policy=lambda args: ToolDispatchPolicy(True),
            ),
            overwrite=True,
        )

    async def notified(*args):
        s.notifications.append(args)

    async def continue_owned():
        if kind == "subagent":
            s.continued = await delegation.nudge(
                get_current_loop(),
                s.child_sid,
                "Call touch_once exactly once. Process its original outcome without repeating it.",
                on_complete=notified,
            )
            return json.dumps(s.continued)
        from power_loop.workflow.resume import resume_detached

        handle = await resume_detached(
            get_current_loop(),
            s.root,
            s.handles[0].run_id,
            continuation=(
                "BUILD",
                "Call touch_once exactly once. Process its original outcome without repeating it.",
                3,
            ),
        )
        s.handles.append(handle)
        s.continued = handle
        return json.dumps(
            {
                "job_id": handle.job_id,
                "run_id": handle.run_id,
                "operation_id": handle.operation_id,
                "status": "accepted",
            }
        )

    c.loop.tool_registry.register(
        ToolDefinition("continue_owned", "Continue the original owner once", input_schema={"type": "object"}),
        FunctionRuntimeTool(
            continue_owned,
            ToolRuntimeContract("example.continue-owner", allowed_roles=("main",)),
            policy=lambda args: ToolDispatchPolicy(True),
        ),
    )
    if provider:
        c.loop.hooks.remove("test.owner.actual-phase-schema")
        from power_loop import HookPoint

        async def actual_schema(ctx):
            sid = get_session_id()
            role = "MAIN" if sid == s.root else "AFTER" if "ROLE-AFTER" in ctx.system_prompt else "BUILD"
            name = None
            if role == "MAIN" and s.phase == "initial" and not s.counts["MAIN"]:
                name = "launch_owned"
            elif role == "MAIN" and s.phase == "continue" and not s.continue_calls:
                name = "continue_owned"
            elif role == "BUILD" and s.phase in ("continue", "collect") and not s.source:
                name = "touch_once"
            ctx.tools = [c.loop.tool_registry.get(name).definition.to_openai_tool()] if name else []
            if role == "BUILD" and s.phase == "initial":
                ctx.system_prompt += "\nInitial setup only: reply INITIAL_READY as plain text. Use no tools."
            elif role == "BUILD" and name == "touch_once":
                ctx.system_prompt += (
                    "\nInitial setup is over. This turn must invoke the offered touch_once function "
                    "with {} through the actual native tool-call mechanism. Text or XML is not a call."
                )
            if role == "MAIN":
                ctx.system_prompt += "\nCall the single offered orchestration tool exactly once if present, then reply QUEUED as plain text. If no tool is offered, only acknowledge supplied results."

        c.loop.hooks.register(HookPoint.LLM_BEFORE, actual_schema, name="test.continuation.actual-schema")
    return c


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


@pytest.mark.parametrize("kind", ["workflow_node", "subagent"])
@pytest.mark.parametrize("stage", ["seal", "commit_ack"])
async def test_later_original_generation_reviews_before_output_without_mutating_old_revision(
    registry_store, kind, stage
):
    c = await continuation_case(registry_store, kind, stage=stage)
    try:
        await complete_initial(c, kind)
        # Review admission includes durable recovery and budget transactions.
        # Use the same bounded phase watchdog as continuation/restoration;
        # this test has no separate three-second latency requirement.
        await asyncio.wait_for(c.state.entered.wait(), c.state.timeout)
        assert c.state.counts["AFTER"] == c.state.baseline_after
        assert len(c.state.notifications) == c.state.baseline_notifications and not c.state.effects
        assert (await c.store.inbox_counts(c.state.root))["pending"] == 0
        source = await c.manager.get_job(c.runtime.actor(c.state.root), c.state.source)
        [scope] = await c.manager.db.fetchall(
            f"SELECT control_owner_generation FROM {c.manager.jobs} WHERE job_id=?", (source.job_id,)
        )
        assert scope["control_owner_generation"] == 2 and source.control_owner_job_id == c.state.owner_id
        assert source.status == "model_closed" and source.result_status == "pending_delivery"
        c.state.release.set()
        await settle(c, kind)
        source = await c.manager.get_job(c.runtime.actor(c.state.root), c.state.source)
        owner = await c.manager.get_job(c.runtime.actor(c.state.root), c.state.owner_id)
        assert source.result_status == "handled" and source.supervision_status == "closed"
        assert owner.generation == 2 and owner.outcome == "failed"
        assert owner.result["usage"]["total_tokens"] == 20
        assert (
            await c.manager.get_result_revision(c.runtime.actor(c.state.root), c.state.owner_id, 1)
            == c.state.old_revision
        )
        assert (
            await c.manager.get_execution_membership(c.runtime.actor(c.state.root), c.state.owner_id, 1)
            == c.state.old_members
        )
        assert not c.state.effects
    finally:
        c.state.release.set()
        await c.runtime.shutdown(timeout_s=3)
        await c.loop.aclose(drain_timeout_s=0.1)


async def test_slow_original_review_preserves_same_result_and_credit_invariants(registry_store, monkeypatch):
    original_factory = continuation_case
    delayed = []

    async def slow_factory(*args, **kwargs):
        c = await original_factory(*args, **kwargs)
        complete = c.model.complete

        async def slow_review(request, **options):
            if get_session_id() != c.state.root and c.state.source and not delayed:
                delayed.append(True)
                # Deliberately exceed the former phase timeout. No additional
                # model attempt, budget grant, or business action is introduced.
                await asyncio.sleep(3.2)
            return await complete(request, **options)

        c.model.complete = slow_review
        return c

    monkeypatch.setitem(globals(), "continuation_case", slow_factory)
    await test_later_original_generation_reviews_before_output_without_mutating_old_revision(
        registry_store, "workflow_node", "commit_ack")
    assert delayed == [True]


@pytest.mark.parametrize("kind", ["workflow_node", "subagent"])
@pytest.mark.parametrize("fault", ["quota", "hidden", "provider", "cancel"])
async def test_later_generation_cannot_release_unprocessed_results_or_borrow_old_credit(
    registry_store, monkeypatch, kind, fault
):
    c = await continuation_case(
        registry_store,
        kind,
        allowed_calls=2 if fault == "quota" else 3,
        hide=fault == "hidden",
        fail_review=fault == "provider",
    )
    if fault == "cancel":
        from power_loop.runtime import abort_outcomes

        reconcile = abort_outcomes._reconcile

        async def cancel_after_publication(loop, actor, source, **kwargs):
            value = await reconcile(loop, actor, source, **kwargs)
            original = await c.manager.get_job(c.runtime.actor(c.state.root), source)
            await c.manager.request_cancel(
                c.runtime.actor(c.state.root),
                original.control_owner_job_id,
                system_key="cancel-original-second-generation",
                reason="Stop original continuation before review",
            )
            return value

        monkeypatch.setattr(abort_outcomes, "_reconcile", cancel_after_publication)
    try:
        await complete_initial(c, kind)
        c.state.release.set()
        await settle(c, kind)
        source = await c.manager.get_job(c.runtime.actor(c.state.root), c.state.source)
        owner = await c.manager.get_job(c.runtime.actor(c.state.root), c.state.owner_id)
        assert source.result_status == "pending_delivery" and source.supervision_status == "open"
        assert owner.generation == 2 and owner.result_status == "none" and owner.outcome is None
        assert c.state.counts["AFTER"] == c.state.baseline_after and not c.state.effects
        assert c.state.counts["BUILD"] == (1 if fault in ("quota", "cancel") else 2)
        assert len(c.state.notifications) == c.state.baseline_notifications
        pending = (await c.store.inbox_counts(c.state.root))["pending"]
        if kind == "subagent":
            assert pending == 0
        else:
            # The restored controller may report a real failed/unknown outcome.
            # This is not a node result or proof that its pending work is handled.
            workflow = await c.manager.get_job(c.runtime.actor(c.state.root), c.state.continued.job_id)
            assert pending == 1 and workflow.outcome == "failed"
            assert workflow.resource_state == "unknown" and workflow.supervision_status == "open"
            assert workflow.result["status"] == "failed"
            assert any("settlement remains unresolved" in error for error in workflow.result["errors"])
            c.state.phase = "drain"
            await c.loop.flush_follow_ups(c.state.root, tools=[], max_rounds=1)
            workflow = await c.manager.get_job(c.runtime.actor(c.state.root), workflow.job_id)
            assert workflow.result_status == "handled" and workflow.supervision_status == "open"
            original = await c.manager.get_job(c.runtime.actor(c.state.root), source.job_id)
            assert original.result_status == "pending_delivery" and original.supervision_status == "open"
        assert (
            await c.manager.get_result_revision(c.runtime.actor(c.state.root), c.state.owner_id, 1)
            == c.state.old_revision
        )
        assert (
            await c.manager.get_execution_membership(c.runtime.actor(c.state.root), c.state.owner_id, 1)
            == c.state.old_members
        )
    finally:
        c.state.release.set()
        await c.runtime.shutdown(timeout_s=3)
        await c.loop.aclose(drain_timeout_s=0.1)


async def collected_case(registry_store, kind, monkeypatch, *, remaining=2, provider=None, start=True):
    c = await continuation_case(registry_store, kind, provider=provider)
    original = c.manager.close_model_execution
    c.state.queued = False

    async def enqueue_after_actual_exit(actor, lease, **kwargs):
        value = await original(actor, lease, **kwargs)
        row = await c.manager.get_job(c.runtime.actor(c.state.root), lease.job_id)
        if row.execution_role == kind and c.state.phase == "initial" and not c.state.queued:
            c.state.queued = True
            c.state.phase = "collect"
            from power_loop.runtime.completion import WINDOW

            c.state.window = WINDOW.get()
            if kind == "workflow_node":
                assert c.state.window is not None
                c.state.window.remaining = remaining
            c.state.child_sid = row.session_id
            c.state.owner_id = row.control_owner_job_id
            await c.store.inbox_put(
                row.session_id,
                [
                    {
                        "item_id": "accepted-original-collect-input",
                        "kind": "relay",
                        "mode": "queue",
                        "content": "Call touch_once exactly once. Process its original outcome without repeating it.",
                    }
                ],
            )
        return value

    monkeypatch.setattr(c.manager, "close_model_execution", enqueue_after_actual_exit)
    if start:
        try:
            await start_collected(c, kind)
        except BaseException:
            c.state.release.set()
            await c.runtime.shutdown(timeout_s=3)
            await c.loop.aclose(drain_timeout_s=0.1)
            raise
    return c


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


@pytest.mark.parametrize("kind", ["workflow_node", "subagent"])
async def test_same_generation_collector_reviews_without_double_counting_returned_rounds(
    registry_store, monkeypatch, kind
):
    c = await collected_case(registry_store, kind, monkeypatch)
    try:
        await asyncio.wait_for(c.state.entered.wait(), 5)
        assert not c.state.notifications and c.state.counts["AFTER"] == 0 and not c.state.effects
        c.state.release.set()
        if kind == "workflow_node":
            await asyncio.wait_for(c.state.handles[0].task, 30)
        else:
            await settle(c, kind)
        source = await c.manager.get_job(c.runtime.actor(c.state.root), c.state.source)
        owner = await c.manager.get_job(c.runtime.actor(c.state.root), c.state.owner_id)
        assert source.result_status == "handled" and source.supervision_status == "closed"
        assert owner.generation == 1 and owner.outcome == "failed"
        assert owner.result["usage"]["total_tokens"] == 30
        if kind == "workflow_node":
            assert c.state.window.remaining == 0
        assert (await c.manager.execution_usage(c.runtime.actor(c.state.root), owner.job_id, generation=1))[
            "calls"
        ] == 3
        assert not c.state.effects
    finally:
        c.state.release.set()
        await c.runtime.shutdown(timeout_s=3)
        await c.loop.aclose(drain_timeout_s=0.1)


async def test_collector_one_remaining_round_cannot_create_review_credit(registry_store, monkeypatch):
    c = await collected_case(registry_store, "workflow_node", monkeypatch, remaining=1)
    try:
        c.state.release.set()
        await asyncio.wait_for(c.state.handles[0].task, 30)
        source = await c.manager.get_job(c.runtime.actor(c.state.root), c.state.source)
        owner = await c.manager.get_job(c.runtime.actor(c.state.root), c.state.owner_id)
        assert not c.state.entered.is_set() and c.state.counts["BUILD"] == 1 and not c.state.effects
        assert source.result_status == "pending_delivery" and owner.result_status == "none"
        assert c.state.counts["AFTER"] == 0 and not c.state.notifications
    finally:
        c.state.release.set()
        await c.runtime.shutdown(timeout_s=3)
        await c.loop.aclose(drain_timeout_s=0.1)


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
