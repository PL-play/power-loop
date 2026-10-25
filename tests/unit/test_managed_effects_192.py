"""Trusted finite managed effects participate in original SDK job/receipt gates."""

import asyncio
import json
from contextlib import asynccontextmanager
from dataclasses import replace
from types import SimpleNamespace

import pytest

from power_loop import AgentLoopConfig, LLMResponse, LLMService, LLMTokenUsage, StatefulAgentLoop, ToolDefinition
from power_loop.runtime.jobs import JobError, JobManager, _hash, _json
from power_loop.runtime.managed_effects import (
    FunctionManagedTool,
    ManagedTool,
    ManagedToolContract,
    ManagedToolPreparation,
    ManagedToolResult,
    reconcile,
)
from power_loop.runtime.task_runtime import TaskRuntime
from power_loop.tools.registry import ToolRegistry

from .test_job_registry_192 import registry_store

__all__ = ["registry_store"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def scenario(
    registry_store,
    *,
    pending=False,
    budgeted=False,
    prepare_failure=False,
    fake_handler=False,
    capacity=None,
    class_based=False,
    tool_name="post_once",
    claim_failure=False,
    receipt_bind_failure=False,
):
    store, _ = registry_store
    sid = await store.create_session()
    manager = JobManager(store)
    from power_loop.runtime.budget_contracts import BudgetConfig, LocalBudgetScope

    from .test_model_budgets_192 import Host

    host = Host() if budgeted else None
    budget = (
        BudgetConfig("test-budget", (LocalBudgetScope("test", max_calls=2, max_tokens=200, max_inflight=1),), host)
        if host
        else None
    )
    runtime = TaskRuntime(
        manager,
        tenant_id="api-effect-test",
        principal_id="host",
        executor_id="executor",
        boot_id="boot",
        boot_contract_hash="b" * 64 if budgeted else None,
        budget_config=budget,
    )
    if receipt_bind_failure:
        bind_receipt = manager.reconcile_receipt
        attempts = 0

        async def fail_one_receipt(*args, **kwargs):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise OSError("test_original_receipt_store_unavailable")
            return await bind_receipt(*args, **kwargs)

        manager.reconcile_receipt = fail_one_receipt
    if claim_failure:
        claim = runtime.claim

        async def fail_before_claim(actor, registration, **kwargs):
            if registration.job.kind == "background_tool":
                if claim_failure == "after":
                    await claim(actor, registration, **kwargs)
                    raise JobError("test_original_claim_ack_lost")
                raise JobError("test_claim_not_started")
            return await claim(actor, registration, **kwargs)

        runtime.claim = fail_before_claim
    state = SimpleNamespace(prepared=0, dispatched=0, lookups=0, resolved=not pending)

    async def prepare(context, arguments):
        state.prepared += 1
        if prepare_failure:
            raise RuntimeError("prepare failed before any API dispatch")
        return ManagedToolPreparation({"original_call": context.tool_call_id, "text": arguments["text"]})

    async def dispatch(context, reference):
        state.dispatched += 1
        if pending:
            raise TimeoutError("original remote decision is unknown")
        return ManagedToolResult(
            "Original effect completed", {"original_call": reference["original_call"], "receipt_id": "original"}
        )

    async def lookup(context, reference):
        state.lookups += 1
        if not state.resolved:
            return None
        return ManagedToolResult(
            "Original effect completed", {"original_call": reference["original_call"], "receipt_id": "original"}
        )

    adapter = FunctionManagedTool(ManagedToolContract("test.original-effect", "a" * 64), prepare, dispatch, lookup)
    if class_based:

        class ExampleManagedTool(ManagedTool):
            @property
            def contract(self):
                return ManagedToolContract("test.original-effect", "a" * 64)

            async def prepare(self, context, arguments):
                return await prepare(context, arguments)

            async def execute(self, context, original_reference):
                return await dispatch(context, original_reference)

            async def recover(self, context, original_reference):
                return await lookup(context, original_reference)

        adapter = ExampleManagedTool()
    definition = ToolDefinition(
        name=tool_name,
        description="One original finite managed effect",
        input_schema={"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
    )
    registry = ToolRegistry()
    registry.register(definition, adapter.bind(runtime, definition))
    if fake_handler:

        async def impostor(**kwargs):
            state.dispatched += 1
            return "Unsafe lookalike"

        registry.register(definition, impostor, overwrite=True)

    class Model(LLMService):
        def __init__(self):
            self.calls = 0

        async def complete(self, request, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return LLMResponse(
                    raw_text="",
                    token_usage=LLMTokenUsage(prompt_tokens=5, completion_tokens=5, total_tokens=10),
                    tool_calls=[
                        {
                            "id": "original-call",
                            "type": "function",
                            "function": {"name": tool_name, "arguments": '{"text":"hello"}'},
                        }
                    ],
                )
            return LLMResponse(
                raw_text="DONE", token_usage=LLMTokenUsage(prompt_tokens=5, completion_tokens=5, total_tokens=10)
            )

        def stream(self, request):
            raise AssertionError("complete transport selected")

        async def close(self):
            pass

    model = Model()
    if host:
        host.service = model
    loop = StatefulAgentLoop(
        llm=model,
        store=store,
        task_runtime=runtime,
        tool_registry=registry,
        config=AgentLoopConfig(
            max_rounds=3,
            model="test-model",
            max_tokens=20,
            delegation_policy="single_level",
            **({"tool_result_max_chars": capacity} if capacity is not None else {}),
        ),
    )
    try:
        try:
            result = await asyncio.wait_for(loop.send("Post the original effect once", sid), 10)
        except OSError as exc:
            if not receipt_bind_failure or str(exc) != "test_original_receipt_store_unavailable":
                raise
            result = SimpleNamespace(status="interrupted", error=str(exc))
        return SimpleNamespace(
            store=store,
            runtime=runtime,
            manager=manager,
            adapter=adapter,
            state=state,
            model=model,
            result=result,
            sid=sid,
            actor=runtime.actor(sid),
            host=host,
        )
    finally:
        await loop.aclose(drain_timeout_s=0.1)


async def test_known_effect_has_required_child_and_exact_tool_receipt(registry_store):
    c = await scenario(registry_store)
    assert c.result.status == "completed" and c.model.calls == 2
    assert (c.state.prepared, c.state.dispatched, c.state.lookups) == (1, 1, 0)
    rows = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.jobs} ORDER BY kind")
    assert {r["kind"] for r in rows} == {"send", "background_tool"}
    child = next(r for r in rows if r["kind"] == "background_tool")
    assert child["blocks_completion"] == 1 and child["resource_state"] == "stopped"
    assert child["result_status"] == "handled" and child["supervision_status"] == "closed"
    original = await c.manager.db.fetchone(
        f"SELECT * FROM {c.manager.events} WHERE job_id=? AND action='create'", (child["job_id"],)
    )
    assert json.loads(original["call_ref_json"])["id"] and json.loads(original["receipt_ref_json"])["id"]


async def test_unknown_api_outcome_stops_another_model_and_can_be_queried(registry_store):
    c = await scenario(registry_store, pending=True)
    assert c.model.calls == 1 and c.state.dispatched == 1
    child = await c.manager.db.fetchone(f"SELECT * FROM {c.manager.jobs} WHERE kind='background_tool'")
    assert child["supervision_status"] == "open" and json.loads(child["result_json"])["managed_effect_pending"]
    assert await reconcile(c.runtime, c.actor, child["job_id"], c.adapter) is None
    c.state.resolved = True
    assert (await reconcile(c.runtime, c.actor, child["job_id"], c.adapter)).reference["receipt_id"] == "original"
    assert c.state.prepared == c.state.dispatched == 1 and c.model.calls == 1
    await c.manager.finalize_supervision(c.actor, child["job_id"], generation=1)
    assert (await c.manager.get_job(c.actor, child["job_id"])).supervision_status == "closed"


@pytest.mark.parametrize("pending", [False, True])
async def test_actual_budget_policy_admits_only_the_sdk_bound_api_handler(registry_store, pending):
    c = await scenario(registry_store, budgeted=True, pending=pending)
    assert c.state.prepared == c.state.dispatched == 1
    assert c.host.reserve_calls == c.model.calls == (1 if pending else 2)
    [child] = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.jobs} WHERE kind='background_tool'")
    assert json.loads(child["result_json"])["usage"] == {}


async def test_a_name_lookalike_does_not_receive_native_budget_permission(registry_store):
    c = await scenario(registry_store, budgeted=True, fake_handler=True)
    assert c.state.prepared == c.state.dispatched == 0
    assert not await c.manager.db.fetchall(f"SELECT job_id FROM {c.manager.jobs} WHERE kind='background_tool'")
    assert c.host.reserve_calls == c.model.calls == 2


async def test_failed_preparation_is_known_not_dispatched_and_does_not_stick_pending(registry_store):
    c = await scenario(registry_store, budgeted=True, prepare_failure=True)
    assert c.state.prepared == 1 and c.state.dispatched == 0
    assert c.model.calls == 2
    [child] = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.jobs} WHERE kind='background_tool'")
    result = json.loads(child["result_json"])
    assert result["managed_effect_pending"] is False
    assert result["managed_effect_result"]["resolution"] == "not_dispatched"
    assert child["supervision_status"] == "closed"


async def test_result_capacity_is_checked_before_creating_or_dispatching_effect(registry_store):
    c = await scenario(registry_store, capacity=16)
    assert c.state.prepared == c.state.dispatched == 0
    assert not await c.manager.db.fetchall(f"SELECT job_id FROM {c.manager.jobs} WHERE kind='background_tool'")


@pytest.mark.parametrize("tool_name", ["commit_record", "set_beacon"])
async def test_public_managed_tool_base_works_with_unrelated_names(registry_store, tool_name):
    c = await scenario(registry_store, budgeted=True, class_based=True, tool_name=tool_name)
    assert c.result.status == "completed" and c.state.prepared == c.state.dispatched == 1
    [child] = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.jobs} WHERE kind='background_tool'")
    assert child["title"] == tool_name and child["supervision_status"] == "closed"
    [creation] = await c.manager.db.fetchall(
        f"SELECT payload_json FROM {c.manager.events} WHERE job_id=? AND action='create'", (child["job_id"],)
    )
    contract = json.loads(creation["payload_json"])["request"]["managed_effect"]["contract"]
    assert contract["resource_model"] == "finite_effect" and contract["paid_requests"] == 0


async def test_base_requires_lifecycle_methods_and_supported_resource_contract():
    from power_loop.runtime.jobs import JobError

    with pytest.raises(TypeError):
        ManagedTool()
    for kwargs in [{"paid_requests": 1}, {"paid_requests": False}, {"resource_model": "untracked_process"}]:
        with pytest.raises(JobError, match="adapter_invalid"):
            ManagedToolContract("generic.tool", "a" * 64, **kwargs)


@pytest.mark.parametrize("old_protocol", [5, 6, 7])
async def test_old_protocol_requires_explicit_upgrade_and_keeps_original_identity(
    registry_store, monkeypatch, old_protocol
):
    import power_loop.runtime.jobs as jobs
    from power_loop.runtime.jobs import JobError
    from power_loop.runtime.store import task_schema
    from power_loop.runtime.store.schema import StoreSchemaError

    store, _ = registry_store
    manager = JobManager(store)
    sid = await store.create_session()
    seq = await store.append_message(sid, role="user", content="Retain original identity")
    ref = await store.message_ref(sid, seq)
    await manager.db.execute(f"UPDATE {manager.prefix}task_store SET protocol_version=? WHERE id=1", (old_protocol,))
    with pytest.raises(StoreSchemaError):
        await task_schema.verify_task_schema(manager.db, manager.prefix)
    with pytest.raises(StoreSchemaError, match="maintenance_required"):
        await task_schema.migrate_task_schema(manager.db, manager.prefix)
    await task_schema.migrate_task_schema(manager.db, manager.prefix, maintenance=True)
    assert await store.message_ref(sid, seq) == ref
    assert (
        await manager.db.fetchone(f"SELECT store_id,protocol_version FROM {manager.prefix}task_store WHERE id=1")
    ) == {"store_id": store.store_id, "protocol_version": 8}
    with monkeypatch.context() as patch:
        patch.setattr(jobs, "TASK_PROTOCOL_VERSION", old_protocol)
        with pytest.raises(JobError, match="store_identity_changed"):
            async with manager._write():
                pass


async def pending_case(registry_store):
    c = await scenario(registry_store, pending=True, budgeted=True)
    c.state.resolved = True
    [row] = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.jobs} WHERE kind='background_tool'")
    return c, row


async def rewrite_event(c, row, kind, change):
    from power_loop.runtime.managed_effects import _id

    event_id = _id(c.manager, kind, row["job_id"], row["generation"])
    event = await c.manager.db.fetchone(f"SELECT * FROM {c.manager.events} WHERE event_id=?", (event_id,))
    value = json.loads(event["payload_json"])
    change(value)
    # A recomputed projection hash must not bypass structural/original binding checks.
    await c.manager.db.execute(
        f"UPDATE {c.manager.events} SET payload_json=?,request_hash=? WHERE event_id=?",
        (_json(value), _hash(value), event_id),
    )
    return event_id


async def test_recovery_compares_the_entire_frozen_contract_before_query(registry_store):
    c, row = await pending_case(registry_store)
    changed = replace(c.adapter, descriptor=replace(c.adapter.contract, max_result_chars=2048))
    with pytest.raises(JobError, match="binding_changed"):
        await reconcile(c.runtime, c.actor, row["job_id"], changed)
    assert c.state.lookups == 0 and c.state.dispatched == 1 and c.model.calls == 1


async def test_recovery_rechecks_contract_after_callback_before_recording(registry_store):
    c, row = await pending_case(registry_store)

    async def query(context, reference):
        result = await c.adapter.recover(context, reference)
        object.__setattr__(adapter, "descriptor", replace(adapter.contract, max_result_chars=2048))
        return result

    adapter = replace(c.adapter, on_recover=query)
    with pytest.raises(JobError, match="binding_changed"):
        await reconcile(c.runtime, c.actor, row["job_id"], adapter)
    assert not await c.manager.db.fetchall(
        f"SELECT event_id FROM {c.manager.events} WHERE event_type='managed_effect_observation'"
    )
    assert c.state.lookups == c.state.dispatched == c.model.calls == 1


async def test_recovery_rechecks_preparation_after_query_before_recording(registry_store):
    c, row = await pending_case(registry_store)

    async def query(context, reference):
        result = await c.adapter.recover(context, reference)
        await rewrite_event(c, row, "prepared", lambda value: value["preparation"]["reference"].update(text="changed"))
        return result

    with pytest.raises(JobError, match="evidence_conflict"):
        await reconcile(c.runtime, c.actor, row["job_id"], replace(c.adapter, on_recover=query))
    assert not await c.manager.db.fetchall(
        f"SELECT event_id FROM {c.manager.events} WHERE event_type='managed_effect_observation'"
    )


@pytest.mark.parametrize("part", ["raw_call", "membership", "prepared_adapter", "prepared_version"])
async def test_recovery_refuses_corrupt_original_evidence_before_query(registry_store, part):
    from power_loop.runtime.execution_membership import scope_id

    c, row = await pending_case(registry_store)
    if part == "raw_call":
        [creation] = await c.manager.db.fetchall(
            f"SELECT call_ref_json FROM {c.manager.events} WHERE job_id=? AND action='create'", (row["job_id"],)
        )
        ref = json.loads(creation["call_ref_json"])
        await c.manager.db.execute(
            f"UPDATE {c.store.t.messages} SET tool_calls_json=? WHERE session_id=? AND id=?",
            (
                _json([{"id": "original-call", "function": {"name": "different", "arguments": '{"text":"hello"}'}}]),
                ref["session_id"],
                int(ref["id"]),
            ),
        )
    elif part == "membership":
        event_id = scope_id(c.manager, row["parent_job_id"], row["parent_generation"])
        event = await c.manager.db.fetchone(
            f"SELECT payload_json FROM {c.manager.events} WHERE event_id=?", (event_id,)
        )
        value = json.loads(event["payload_json"])
        value["lease"]["executor_id"] = "different-original-executor"
        await c.manager.db.execute(
            f"UPDATE {c.manager.events} SET payload_json=? WHERE event_id=?", (_json(value), event_id)
        )
    elif part == "prepared_adapter":
        await rewrite_event(c, row, "prepared", lambda value: value["adapter"].update(arguments_hash="c" * 64))
    else:
        await rewrite_event(c, row, "prepared", lambda value: value.update(version=True))
    with pytest.raises(JobError, match="managed_effect_"):
        await reconcile(c.runtime, c.actor, row["job_id"], c.adapter)
    assert c.state.lookups == 0 and c.state.dispatched == c.model.calls == 1


@pytest.mark.parametrize("part", ["version", "lease", "adapter", "extra", "result", "event_type", "job_id"])
async def test_corrupt_observation_cannot_release_paid_dispatch_or_owner_gate(registry_store, part):
    from power_loop.runtime.managed_effects import require_owned_resolved, require_source_resolved

    c, row = await pending_case(registry_store)
    assert await reconcile(c.runtime, c.actor, row["job_id"], c.adapter)

    def change(value):
        if part == "version":
            value["version"] = True
        elif part == "lease":
            value["lease"]["generation"] = True
        elif part == "adapter":
            value["adapter"]["contract"]["max_result_chars"] = 1
        elif part == "extra":
            value["trusted"] = True
        elif part == "result":
            value["result"]["resolution"] = "unknown"

    event_id = await rewrite_event(c, row, "observation", change)
    if part in ("event_type", "job_id"):
        await c.manager.db.execute(
            f"UPDATE {c.manager.events} SET {part}=? WHERE event_id=?",
            ("different_type" if part == "event_type" else row["parent_job_id"], event_id),
        )
    source = await c.manager.db.fetchone(f"SELECT * FROM {c.manager.jobs} WHERE job_id=?", (row["parent_job_id"],))
    for guard in (require_source_resolved, require_owned_resolved):
        async with c.manager._write() as tx:
            with pytest.raises(JobError, match="managed_effect_"):
                await guard(c.manager, tx, source)
    assert c.state.lookups == c.state.dispatched == c.model.calls == 1


async def test_observation_replay_is_idempotent_but_conflicting_result_is_rejected(registry_store):
    c, row = await pending_case(registry_store)
    first = await reconcile(c.runtime, c.actor, row["job_id"], c.adapter)
    assert await reconcile(c.runtime, c.actor, row["job_id"], c.adapter) == first

    async def different(context, reference):
        return ManagedToolResult("Different original result", {"receipt_id": "replacement"})

    with pytest.raises(JobError, match="evidence_conflict"):
        await reconcile(c.runtime, c.actor, row["job_id"], replace(c.adapter, on_recover=different))
    assert c.state.dispatched == c.model.calls == 1


async def test_non_finite_reference_documents_are_rejected():
    for value in (float("nan"), float("inf"), -float("inf")):
        with pytest.raises(JobError, match="document_invalid"):
            ManagedToolPreparation({"nested": [value]})


async def test_unclaimed_effect_cancels_and_closes_without_callbacks_or_fake_tool_receipt(registry_store):
    from power_loop.runtime.managed_effects import read_raw_receipt, terminate_unstarted

    c = await scenario(registry_store, budgeted=True, claim_failure=True)
    assert c.state.prepared == c.state.dispatched == c.state.lookups == 0
    assert c.model.calls == c.host.reserve_calls == 1
    [row] = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.jobs} WHERE kind='background_tool'")
    assert row["generation"] == 0 and row["resource_state"] == "not_started"
    observed = await read_raw_receipt(c.manager, c.actor, row["job_id"])
    assert not observed["acknowledges_result"] and observed["call_ref"]["id"] != observed["receipt_ref"]["id"]
    raw = await c.manager.db.fetchone(
        f"SELECT * FROM {c.store.t.messages} WHERE id=?", (int(observed["receipt_ref"]["id"]),)
    )
    assert "test_claim_not_started" in raw["content"]
    assert c.result.status == "degraded"
    result = await terminate_unstarted(c.manager, c.actor, row["job_id"])
    assert result["resolution"] == "not_dispatched" and not result["tool_receipt_written"]
    assert not result["execution_authority"] and not result["billing_settled"]
    job = await c.manager.get_job(c.actor, row["job_id"])
    assert job.supervision_status == "closed" and job.resource_state == "stopped" and job.outcome == "cancelled"
    assert not await c.manager.db.fetchall(
        f"SELECT event_id FROM {c.manager.events} WHERE job_id=? AND event_type IN ('managed_effect_dispatch','managed_effect_prepared','managed_effect_tool_receipt')",
        (row["job_id"],),
    )
    store = await registry_store[1](schema="compatible")
    assert await terminate_unstarted(JobManager(store), c.actor, row["job_id"]) == result
    assert await read_raw_receipt(JobManager(store), c.actor, row["job_id"]) == observed
    assert (
        await c.manager.db.fetchone(
            f"SELECT * FROM {c.store.t.messages} WHERE id=?", (int(observed["receipt_ref"]["id"]),)
        )
        == raw
    )
    assert c.state.dispatched == 0 and c.model.calls == c.host.reserve_calls == 1


async def test_unclaimed_cancel_and_closure_roll_back_as_one_transaction(registry_store, monkeypatch):
    from power_loop.runtime import finalization
    from power_loop.runtime.managed_effects import terminate_unstarted

    c = await scenario(registry_store, claim_failure=True)
    [before] = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.jobs} WHERE kind='background_tool'")

    async def fail(*args):
        raise RuntimeError("closure write failed")

    with monkeypatch.context() as patch:
        patch.setattr(finalization, "_publish", fail)
        with pytest.raises(RuntimeError, match="closure write failed"):
            await terminate_unstarted(c.manager, c.actor, before["job_id"])
    assert await c.manager.db.fetchone(f"SELECT * FROM {c.manager.jobs} WHERE job_id=?", (before["job_id"],)) == before
    assert not await c.manager.db.fetchall(
        f"SELECT event_id FROM {c.manager.events} WHERE job_id=? AND event_type IN ('managed_effect_unstarted','result','supervision_closed')",
        (before["job_id"],),
    )
    assert (await terminate_unstarted(c.manager, c.actor, before["job_id"]))["resolution"] == "not_dispatched"


async def test_claimed_effect_cannot_be_relabelled_never_started(registry_store):
    from power_loop.runtime.managed_effects import terminate_unstarted

    c = await scenario(registry_store, pending=True)
    [row] = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.jobs} WHERE kind='background_tool'")
    with pytest.raises(JobError, match="unstarted_conflict"):
        await terminate_unstarted(c.manager, c.actor, row["job_id"])
    assert c.state.dispatched == 1 and c.state.lookups == 0


@pytest.mark.parametrize("part", ["result", "negative_proof", "scope"])
async def test_closed_unclaimed_effect_revalidates_original_evidence(registry_store, part):
    from power_loop.runtime.managed_effects import _id, terminate_unstarted

    c = await scenario(registry_store, claim_failure=True)
    [row] = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.jobs} WHERE kind='background_tool'")
    await terminate_unstarted(c.manager, c.actor, row["job_id"])
    if part == "result":
        await c.manager.db.execute(
            f"UPDATE {c.manager.events} SET payload_json='{{}}' WHERE event_id=?", ("result_" + row["job_id"],)
        )
    elif part == "negative_proof":
        await c.manager.db.execute(
            f"DELETE FROM {c.manager.events} WHERE event_id=?", (_id(c.manager, "unstarted", row["job_id"], 0),)
        )
    else:
        await c.manager.db.execute(
            f"UPDATE {c.manager.jobs} SET executor_id='unexpected' WHERE job_id=?", (row["job_id"],)
        )
    with pytest.raises(JobError):
        await terminate_unstarted(c.manager, c.actor, row["job_id"])
    assert c.state.dispatched == c.state.prepared == c.state.lookups == 0


@pytest.mark.parametrize("unclaimed", [True, False])
async def test_retained_original_row_is_rebound_after_storage_failure_without_replay(registry_store, unclaimed):
    from power_loop.runtime.managed_effects import read_raw_receipt, reconcile_raw_receipt

    c = await scenario(registry_store, budgeted=True, claim_failure=unclaimed, receipt_bind_failure=True)
    [row] = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.jobs} WHERE kind='background_tool'")
    assert c.model.calls == c.host.reserve_calls == 1
    with pytest.raises(JobError, match="evidence_missing"):
        await read_raw_receipt(c.manager, c.actor, row["job_id"])
    before = await c.manager.db.fetchall(f"SELECT * FROM {c.store.t.messages} ORDER BY seq")
    restored = JobManager(await registry_store[1](schema="compatible"))
    observed = await reconcile_raw_receipt(restored, c.actor, row["job_id"])
    assert observed["call_ref"]["id"] != observed["receipt_ref"]["id"]
    assert not observed["acknowledges_result"] and not observed["execution_authority"]
    assert await reconcile_raw_receipt(restored, c.actor, row["job_id"]) == observed
    assert await c.manager.db.fetchall(f"SELECT * FROM {c.store.t.messages} ORDER BY seq") == before
    assert c.state.dispatched == (0 if unclaimed else 1) and c.state.lookups == 0
    assert c.model.calls == c.host.reserve_calls == 1
    current = await c.manager.get_job(c.actor, row["job_id"])
    assert current.result_status == ("none" if unclaimed else "handled")


@pytest.mark.parametrize("part", ["raw_content", "raw_send", "observation", "pending_projection"])
async def test_raw_receipt_or_projection_tampering_cannot_release_another_model(registry_store, part):
    from power_loop.runtime.managed_effects import read_raw_receipt, require_source_resolved

    c = await scenario(registry_store, budgeted=True, pending=True)
    [row] = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.jobs} WHERE kind='background_tool'")
    observed = await read_raw_receipt(c.manager, c.actor, row["job_id"])
    if part == "pending_projection":
        result = json.loads(row["result_json"])
        result["managed_effect_pending"] = False
        await c.manager.db.execute(
            f"UPDATE {c.manager.jobs} SET result_json=? WHERE job_id=?", (_json(result), row["job_id"])
        )
    elif part == "observation":
        await c.manager.db.execute(
            f"UPDATE {c.manager.events} SET payload_json='{{}}' WHERE event_id=?", (observed["observation_id"],)
        )
    else:
        column, value = ("content", "changed") if part == "raw_content" else ("send_index", 99)
        await c.manager.db.execute(
            f"UPDATE {c.store.t.messages} SET {column}=? WHERE id=?", (value, int(observed["receipt_ref"]["id"]))
        )
    c.state.resolved = True
    await reconcile(c.runtime, c.actor, row["job_id"], c.adapter)
    source = await c.manager.db.fetchone(f"SELECT * FROM {c.manager.jobs} WHERE job_id=?", (row["parent_job_id"],))
    async with c.manager._write() as tx:
        with pytest.raises(JobError):
            await require_source_resolved(c.manager, tx, source)
    assert c.state.dispatched == c.model.calls == 1


@pytest.mark.parametrize("missing", [True, False])
async def test_missing_or_ambiguous_original_tool_rows_are_not_invented_or_guessed(registry_store, missing):
    from power_loop.runtime.managed_effects import reconcile_raw_receipt

    c = await scenario(registry_store, claim_failure=True, receipt_bind_failure=True)
    [job] = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.jobs} WHERE kind='background_tool'")
    [raw] = await c.manager.db.fetchall(f"SELECT * FROM {c.store.t.messages} WHERE role='tool'")
    if missing:
        await c.manager.db.execute(f"DELETE FROM {c.store.t.messages} WHERE id=?", (raw["id"],))
    else:
        await c.store.append_message(
            c.sid,
            role="tool",
            name=raw["name"],
            content=raw["content"],
            tool_call_id=raw["tool_call_id"],
            send_index=raw["send_index"],
            round_index=raw["round_index"],
        )
    before = await c.manager.db.fetchall(f"SELECT * FROM {c.store.t.messages} ORDER BY seq")
    if missing:
        assert await reconcile_raw_receipt(c.manager, c.actor, job["job_id"]) is None
    else:
        with pytest.raises(JobError, match="receipt_ambiguous"):
            await reconcile_raw_receipt(c.manager, c.actor, job["job_id"])
    assert await c.manager.db.fetchall(f"SELECT * FROM {c.store.t.messages} ORDER BY seq") == before
    assert c.state.dispatched == c.state.prepared == 0 and c.model.calls == 1


@pytest.mark.parametrize("after_commit", [False, True])
async def test_raw_index_and_observation_are_atomic_and_lost_ack_is_queryable(
    registry_store, monkeypatch, after_commit
):
    from power_loop.runtime.managed_effects import reconcile_raw_receipt

    c = await scenario(registry_store, claim_failure=True, receipt_bind_failure=True)
    [job] = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.jobs} WHERE kind='background_tool'")
    manager = JobManager(await registry_store[1](schema="compatible"))
    write = manager._write

    class Proxy:
        def __init__(self, tx):
            self.tx = tx

        def __getattr__(self, name):
            return getattr(self.tx, name)

        async def execute(self, sql, params=()):
            if sql.startswith(f"UPDATE {manager.events} SET receipt_ref_json="):
                raise OSError("test_index_write_failed")
            return await self.tx.execute(sql, params)

    @asynccontextmanager
    async def fail_write():
        async with write() as tx:
            yield tx if after_commit else Proxy(tx)
        if after_commit:
            raise OSError("test_receipt_ack_lost")

    with monkeypatch.context() as patch:
        patch.setattr(manager, "_write", fail_write)
        with pytest.raises(OSError):
            await reconcile_raw_receipt(manager, c.actor, job["job_id"])
    events = await manager.db.fetchall(
        f"SELECT * FROM {manager.events} WHERE job_id=? AND event_type='managed_effect_raw_receipt'", (job["job_id"],)
    )
    [operation] = await manager.db.fetchall(
        f"SELECT * FROM {manager.events} WHERE job_id=? AND action='create'", (job["job_id"],)
    )
    assert bool(events) == bool(operation["receipt_ref_json"]) == after_commit
    observed = await reconcile_raw_receipt(manager, c.actor, job["job_id"])
    assert observed and not observed["acknowledges_result"]
    assert c.model.calls == 1 and c.state.dispatched == c.state.lookups == 0


async def test_protocol_seven_original_pending_receipt_can_be_upgraded_and_reindexed_without_execution(registry_store):
    from power_loop.runtime import jobs
    from power_loop.runtime.managed_effects import _id, read_raw_receipt, reconcile_raw_receipt
    from power_loop.runtime.store import task_schema
    from power_loop.runtime.store.schema import StoreSchemaError

    c = await scenario(registry_store, budgeted=True, pending=True)
    [job] = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.jobs} WHERE kind='background_tool'")
    before = await read_raw_receipt(c.manager, c.actor, job["job_id"])
    # Protocol7 retained the canonical row/index but had no independent raw
    # observation. This pending original has no child/parent closure to rewrite.
    await c.manager.db.execute(
        f"DELETE FROM {c.manager.events} WHERE event_id=?", (_id(c.manager, "raw_receipt", job["job_id"], 0),)
    )
    await c.manager.db.execute(f"UPDATE {c.manager.prefix}task_store SET protocol_version=7 WHERE id=1")
    with pytest.raises(StoreSchemaError):
        await task_schema.verify_task_schema(c.manager.db, c.manager.prefix)
    await task_schema.migrate_task_schema(c.manager.db, c.manager.prefix, maintenance=True)
    after = await reconcile_raw_receipt(c.manager, c.actor, job["job_id"])
    assert after["call_ref"] == before["call_ref"] and after["receipt_ref"] == before["receipt_ref"]
    assert not after["acknowledges_result"]
    assert c.state.dispatched == c.model.calls == c.host.reserve_calls == 1 and c.state.lookups == 0
    c.state.resolved = True
    await reconcile(c.runtime, c.actor, job["job_id"], c.adapter)
    await c.manager.finalize_supervision(c.actor, job["job_id"], generation=1)
    assert (await c.manager.get_job(c.actor, job["job_id"])).supervision_status == "closed"
    assert jobs.TASK_PROTOCOL_VERSION == 8


async def test_claim_membership_survives_lost_ack_and_cannot_be_relabelled_unclaimed(registry_store):
    from power_loop.runtime.execution_membership import scope_id
    from power_loop.runtime.managed_effects import terminate_unstarted

    c = await scenario(registry_store, claim_failure="after")
    [job] = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.jobs} WHERE kind='background_tool'")
    assert job["generation"] == 1 and c.state.prepared == c.state.dispatched == 0
    await c.manager.db.execute(
        f"DELETE FROM {c.manager.events} WHERE event_id=?", (scope_id(c.manager, job["job_id"], 1),)
    )
    await c.manager.db.execute(
        f"UPDATE {c.manager.jobs} SET generation=0,executor_id=NULL,boot_id=NULL,lease_until_ms=NULL,status='queued',intake='open',resource_state='not_started' WHERE job_id=?",
        (job["job_id"],),
    )
    with pytest.raises(JobError, match="unstarted_conflict"):
        await terminate_unstarted(c.manager, c.actor, job["job_id"])
    assert c.model.calls == 1 and c.state.dispatched == c.state.lookups == 0
