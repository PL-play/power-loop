"""Managed effects deliver the original call's result to the parent model."""
from __future__ import annotations

import asyncio
import json
import os
import re
from contextlib import asynccontextmanager, nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from power_loop import AgentLoopConfig, LLMResponse, LLMService, TrustedActor
from power_loop.agent.runtime_client import RuntimeClient
from power_loop.agent.runtime_store import RuntimeInputStore
from power_loop.agent.runtime_types import (
    InputRef,
    RuntimeRequestError,
    TargetRef,
    plain,
    result_message_id,
)
from power_loop.contracts.tools import ToolDefinition
from power_loop.core.execution_context import get_execution_context
from power_loop.runtime import managed_effects
from power_loop.runtime.managed_effects import (
    FunctionManagedTool,
    ManagedToolContract,
    ManagedToolPreparation,
    ManagedToolResult,
    reconcile,
)
from power_loop.runtime.message_ref import MessageRef
from power_loop.runtime.pricing import Free
from power_loop.tools.registry import ToolRegistry
from tests.runtime_fixture import RunningAgent

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]

_CALL_ID = "original-call"
_KIND = "managed_effect_result"
_DISPLAY = "Managed post receipt: R18-731"
_ORIGIN = re.compile(r"\[managed origin (original-call) ([a-z0-9_]+) ([a-z0-9_]+)\]")
_HEADER = re.compile(r"^\[managed_effect_result · (res_[a-f0-9]+) · normal\]\n\n", re.MULTILINE)


class ManagedParent(LLMService):
    def __init__(self):
        self.calls = []
        self.notices = []

    async def complete(self, request, **kwargs):
        execution = get_execution_context()
        observed = {"input_ref": plain(execution.input_ref), "send_index": execution.send_index,
                    "messages": request.messages}
        self.calls.append(observed)
        if len(self.calls) == 1:
            return LLMResponse(raw_text="", tool_calls=[{
                "id": _CALL_ID, "type": "function", "function": {
                    "name": "post_once", "arguments": json.dumps({"text": "hello"}),
                },
            }])

        # The current projector truncates historical tool JSON. Carry the identity forward
        # in the initial assistant reply, derived from the actual call and receipt it saw.
        original_call = next((call for message in request.messages if message.get("role") == "assistant"
                              for call in message.get("tool_calls", ()) or () if call["id"] == _CALL_ID), None)
        tool_receipt = next((json.loads(message["content"]) for message in request.messages
                             if message.get("role") == "tool" and message.get("tool_call_id") == _CALL_ID), None)
        if original_call is not None and tool_receipt is not None:
            start = InputRef(**tool_receipt["input_ref"])
            assert tool_receipt["target_ref"] == plain(TargetRef(start.store_id, start.session_id))
            identity = f"[managed origin {original_call['id']} {start.session_id} {start.message_id}]"
            return LLMResponse(raw_text="waiting for managed result " + identity)

        origin = next((match.groups() for message in request.messages if message.get("role") == "assistant"
                       and (match := _ORIGIN.search(message.get("content") or ""))), None)
        reply = "waiting for managed result"
        for message in request.messages:
            if message.get("role") != "user" or not isinstance(message.get("content"), str):
                continue
            for header in _HEADER.finditer(message["content"]):
                if header[1] != execution.input_ref.message_id:
                    continue
                payload, _ = json.JSONDecoder().raw_decode(message["content"][header.end():])
                output = payload.get("structured_output") or {}
                reference = output.get("reference") or {}
                call_ref = reference.get("call_ref") or {}
                child_input = InputRef(**payload["input_ref"])
                child_target = TargetRef(**payload["child_target"])
                same_target = (origin is not None and child_input.store_id == execution.store_id
                               and child_target == TargetRef(execution.store_id, origin[1])
                               and child_input.session_id == origin[1])
                if (same_target and payload["status"] == "interrupted"
                        and child_input.message_id == origin[2]
                        and payload["error_code"] == "interrupted"):
                    reply = "parent processed unknown outcome"
                elif (same_target and payload["status"] == "completed" and output.get("resolution") == "committed"
                      and reference.get("original_call") == origin[0]
                      and call_ref.get("store_id") == execution.store_id
                      and call_ref.get("session_id") == execution.session_id
                      and call_ref.get("role") == "assistant"
                      and payload.get("final_text") == output.get("display") == _DISPLAY):
                    reply = "parent processed: " + payload["final_text"]
                else:
                    reply = "managed result not processed"
                self.notices.append({"input_ref": plain(execution.input_ref), "send_index": execution.send_index,
                                     "payload": payload, "reply": reply, "projected_origin": origin})
        return LLMResponse(raw_text=reply)


def _adapter(state, *, pending, missing_body):
    async def prepare(context, arguments):
        state.prepared += 1
        assert context.tool_call_id == _CALL_ID
        return ManagedToolPreparation({
            "original_call": context.tool_call_id,
            "call_ref": context.call_ref.to_dict(),
            "text": arguments["text"],
        })

    async def execute(context, reference):
        state.actions.append(reference)
        display = "Managed post receipt body missing" if missing_body else _DISPLAY
        state.external_result = ManagedToolResult(display, reference)
        state.action_finished.set()
        if pending:
            raise TimeoutError("remote decision unknown")
        await state.release.wait()
        return state.external_result

    async def recover(context, reference):
        state.probes.append(reference)
        assert context.call_ref.to_dict() == reference["call_ref"]
        assert context.tool_call_id == reference["original_call"]
        assert state.actions == [reference]
        return state.external_result if state.resolved else None

    return FunctionManagedTool(
        ManagedToolContract("test.original-effect", "a" * 64),
        prepare,
        execute,
        recover,
    )


def _registry(adapter):
    registry = ToolRegistry()
    definition = ToolDefinition(
        name="post_once",
        description="one managed effect",
        input_schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        pricing=Free("test"),
    )
    registry.register(definition, adapter.bind(None, definition))
    return registry


async def _exercise_managed_notice(store, raw_db, *, pending, measure=nullcontext,
                                   connect=None, interleave=None):
    state = SimpleNamespace(prepared=0, actions=[], probes=[], resolved=False, external_result=None,
                            action_finished=asyncio.Event(), release=asyncio.Event())
    missing_body = os.environ.get("R18_CONTROL_EMPTY_BODY") == "1"
    adapter = _adapter(state, pending=pending, missing_body=missing_body)
    model = ManagedParent()
    runner = await RunningAgent(
        store, model, _registry(adapter), AgentLoopConfig(context_budget_tokens=10**9),
    ).start()
    client = runner.runtime.client
    operation = "managed_notice_resolved" if pending else "managed_notice_known"
    evidence = {"db": raw_db.dialect.name, "operation": operation, "missing_body_control": missing_body,
                "parent": plain(runner.target), "checks_passed": False, "stage": "parent_tool_request"}
    try:
        parent = await runner.send("post once")
        assert parent.status == "completed"
        rows = await store.load_all_messages(runner.target.session_id)
        [tool] = [row for row in rows if row.role == "tool" and row.tool_call_id == _CALL_ID]
        [assistant] = [row for row in rows if row.role == "assistant"
                       and any(call["id"] == _CALL_ID for call in row.tool_calls or ())]
        call_ref = MessageRef(store.store_id, assistant.session_id, assistant.id, assistant.seq,
                              assistant.role, assistant.tool_call_id,
                              assistant.send_index, assistant.round_index).to_dict()
        receipt = json.loads(tool.content)
        start = InputRef(**receipt["input_ref"])
        assert parent.final_text == f"waiting for managed result [managed origin {_CALL_ID} {start.session_id} {start.message_id}]"
        target = TargetRef(**receipt["target_ref"])
        assert target == TargetRef(start.store_id, start.session_id)
        evidence.update(original_parent=plain(parent), tool_call_id=_CALL_ID, call_ref=call_ref,
                        tool_receipt=receipt, stage="external_action")
        async with asyncio.timeout(10):
            await state.action_finished.wait()
        assert state.prepared == 1 and state.actions == [
            {"original_call": _CALL_ID, "call_ref": call_ref, "text": "hello"}]

        def parent_ref(send_index):
            return InputRef(store.store_id, runner.target.session_id,
                            result_message_id(target, start, send_index, _KIND))

        initial_ref = parent_ref(1)
        original = None
        if pending:
            evidence["stage"] = "initial_unknown_notice"
            async with asyncio.timeout(10):
                await runner.drain(after_input=initial_ref)
            original = await client.inputs.get_result(start, actor=runner.actor)
            first_parent = await client.inputs.get_result(initial_ref, actor=runner.actor)
            evidence.update(original_before_recovery=plain(original), initial_parent=plain(first_parent))
            assert original.input_ref == start and original.status == "interrupted"
            assert first_parent.input_ref == initial_ref and first_parent.status == "completed"
            assert first_parent.final_text == "parent processed unknown outcome"
            assert not state.probes
            assert await reconcile(store, runner.actor, start, adapter) is None
            assert state.probes == state.actions

        delivered = parent_ref(2) if pending else initial_ref
        evidence.update(stage="measure_delivery", measured_parent_input_ref=plain(delivered))
        if interleave is not None:
            await interleave(runner, start, delivered)
        async with measure():
            if pending:
                state.resolved = True
                recovered = await reconcile(store, runner.actor, start, adapter, client=client)
                evidence["reconcile_result"] = recovered.to_dict() if recovered else None
                assert recovered == state.external_result
            else:
                state.release.set()
            async with asyncio.timeout(10):
                await runner.drain(after_input=delivered)

        evidence["stage"] = "verify_persisted_identity_and_body"
        if connect is not None:
            client = RuntimeClient(store=await connect(), transport=runner.runtime.transport)
        child_result = await client.inputs.get_result(start, actor=runner.actor)
        child = await client.inspect(start, actor=runner.actor)
        assert child_result.input_ref == start and child.result_delivery_error is None
        parent_result = await client.inputs.get_result(delivered, actor=runner.actor)
        assert parent_result.input_ref == delivered and parent_result.status == "completed"
        parent_status = await client.inspect(delivered, actor=runner.actor)
        assert parent_status.terminal_status == "completed" and parent_status.result_seq == parent_result.result_seq
        queue = await raw_db.fetchall(
            f"SELECT item_id,kind,status,terminal_status,delivered_send_index,result_seq,content_json "
            f"FROM {store.t.follow_up_queue} WHERE session_id=? AND kind=? ORDER BY id",
            (runner.target.session_id, _KIND),
        )
        routed = [{**dict(row), "content": json.loads(row["content_json"])} for row in queue]
        evidence.update(original_tool_result=plain(child_result), measured_parent_result=plain(parent_result),
                        parent_queue=routed)
        expected = [initial_ref, delivered] if pending else [delivered]
        assert [item["input_ref"] for item in model.calls] == [
            plain(parent.input_ref), plain(parent.input_ref), *(plain(ref) for ref in expected)]
        assert [item["input_ref"] for item in model.notices] == [plain(ref) for ref in expected]
        assert [row["item_id"] for row in routed] == [ref.message_id for ref in expected]
        for notice, row in zip(model.notices, routed, strict=True):
            assert notice["projected_origin"] == (_CALL_ID, start.session_id, start.message_id)
            assert notice["payload"] == row["content"]
            assert row["status"] == "delivered"
            inspected = await client.inputs.inspect(
                InputRef(store.store_id, runner.target.session_id, row["item_id"]), actor=runner.actor,
            )
            assert inspected.terminal_status == "completed"
            assert row["delivered_send_index"] == notice["send_index"]
        actual = model.notices[-1]["payload"]
        assert actual["child_target"] == plain(target) and actual["status"] == "completed"
        assert actual["structured_output"]["reference"] == state.actions[0]
        assert parent_result.send_index == model.notices[-1]["send_index"]
        if pending:
            assert child_result == original and child.terminal_status == "interrupted"
            recovery_ref = InputRef(**actual["input_ref"])
            assert recovery_ref != start and recovery_ref.session_id == start.session_id
            recovered = await client.inputs.get_result(recovery_ref, actor=runner.actor)
            evidence["recovered_tool_result"] = plain(recovered)
            assert routed[0]["content"]["input_ref"] == plain(start)
            assert routed[0]["content"]["status"] == "interrupted"
            child_recovery = await client.inspect(recovery_ref, actor=runner.actor)
            assert recovery_ref == managed_effects._recovery_ref(start, 1)
            assert recovered.status == "completed" and recovered.send_index == 2
            assert child_recovery.terminal_status == "completed" and child_recovery.result_seq == recovered.result_seq
            page = await client.messages(target, actor=runner.actor)
            user, assistant = [row for row in page.items if row.send_index == 2]
            assert user.seq + 1 == assistant.seq == recovered.result_seq
            assert json.loads(user.content)["items"] == [{
                "message_id": recovery_ref.message_id, "kind": "user", "priority": "normal",
                "content": {"managed_effect": "recovered", "result": plain(recovered.structured_output)},
            }]
            assert assistant.content == recovered.final_text
            assert state.probes == state.actions * 2
        else:
            recovered = child_result
            assert actual["input_ref"] == plain(start)
            assert child.terminal_status == "completed" and not state.probes
        assert actual["result_seq"] == recovered.result_seq
        assert actual["final_text"] == recovered.final_text
        assert actual["structured_output"] == plain(recovered.structured_output)
        rows = await raw_db.fetchall(
            f"SELECT state_key,value_json FROM {store.t.session_runtime_state} WHERE session_id=?",
            (start.session_id,),
        )
        [saved] = [json.loads(row["value_json"]) for row in rows]
        assert saved["phase"] == "completed" and saved["result"] == actual["structured_output"]
        assert len(state.actions) == 1
        evidence["stage"] = "verify_parent_consumption"
        assert parent_result.final_text == "parent processed: " + _DISPLAY, (
            "parent did not process the expected managed result body", actual["final_text"], parent_result.final_text)
        if pending and interleave is not None:
            repeat = await reconcile(store, runner.actor, start, adapter, client=client)
            assert repeat == state.external_result and state.probes == state.actions * 2
            await runner.drain()
            assert len(model.notices) == 2 and state.prepared == len(state.actions) == 1
        if not pending and measure is nullcontext:
            repeat = await reconcile(store, runner.actor, start, adapter, client=client)
            assert repeat == state.external_result
            with pytest.raises(RuntimeRequestError, match="not_found"):
                await client.inputs.get_result(managed_effects._recovery_ref(start, 1), actor=runner.actor)
            notices = await raw_db.fetchall(
                f"SELECT item_id FROM {store.t.follow_up_queue} WHERE session_id=? AND kind=? ORDER BY id",
                (runner.target.session_id, _KIND),
            )
            assert [row["item_id"] for row in notices] == [delivered.message_id]
            assert len(model.notices) == 1 and not state.probes and state.prepared == len(state.actions) == 1
        evidence.update(checks_passed=True, stage="complete", fresh_client=connect is not None,
                        original_status=plain(child), parent_status=plain(parent_status))
        return evidence
    finally:
        evidence.update(parent_model_calls=model.calls, model_notices=model.notices, prepared=state.prepared,
                        external_actions=state.actions, read_only_probes=state.probes)
        if output := os.environ.get("R18_MANAGED_OUT"):
            with Path(output).open("a") as stream:
                stream.write(json.dumps(evidence, ensure_ascii=False) + "\n")
        state.release.set()
        if client is not runner.runtime.client:
            await client.aclose()
        await runner.aclose()


async def test_prepare_execute_and_original_call_state(registry_store):
    store, _ = registry_store
    await _exercise_managed_notice(store, store._db, pending=False)


async def test_unknown_recover_is_read_only_and_late_result_is_new_input(registry_store):
    store, _ = registry_store
    await _exercise_managed_notice(store, store._db, pending=True)


async def test_verified_recovery_is_never_claimable(registry_store, monkeypatch):
    store, connect = registry_store
    observed = []
    contender = None

    async def interleave(runner, start, parent_ref):
        nonlocal contender
        contender = RunningAgent(await connect(), runner.model, runner.registry,
                                 coordinator=runner.runtime.coordinator, transport=runner.runtime.transport)
        # A separately connected node, with the same real factory and model observation.
        contender.runtime.factory = runner.runtime.factory
        contender.runtime.node_id = "recovery-contender"
        await contender.start(target=runner.target)
        commit = managed_effects._commit_recovery
        write_state = managed_effects._write_state
        recover_ref = managed_effects._recovery_ref(start, 1)
        port = contender.runtime.client.inputs
        claim = RuntimeInputStore.claim

        async def record_claim(self, target, **kwargs):
            batch = await claim(self, target, **kwargs)
            if kwargs.get("node_id") == "recovery-contender" and batch is not None:
                observed.append(("claim", target.session_id, [ref.message_id for ref in batch.input_refs]))
                assert recover_ref not in batch.input_refs
            return batch

        async def discover(stage, exists):
            ref = await port.input(port.db, recover_ref)
            assert (ref is not None) == exists
            if ref is not None:
                assert ref.row["terminal_status"] == "completed"
            await contender.runtime.worker.recover()
            observed.append((stage, exists))
            await contender.drain()

        async def gated_commit(*args):
            if args[-1] is None:
                return await commit(*args)  # idempotent re-report after the first publication
            assert args[-1]["display"] == _DISPLAY  # real probe returned before any write
            await discover("probe_returned", False)
            value = await commit(*args)
            # No notification has been published; the second node discovers the parent itself.
            await discover("committed_before_hint", True)
            async with asyncio.timeout(10), runner.execution_started:
                await runner.execution_started.wait_for(lambda: parent_ref in runner.executions)
            await runner.executions[parent_ref]
            return value

        async def gated_write(tx, port, sid, key, value, timestamp):
            await write_state(tx, port, sid, key, value, timestamp)
            if sid == start.session_id and value.get("recovered_send_index") is not None:
                await discover("before_commit", False)

        monkeypatch.setattr(RuntimeInputStore, "claim", record_claim)
        monkeypatch.setattr(managed_effects, "_write_state", gated_write)
        monkeypatch.setattr(managed_effects, "_commit_recovery", gated_commit)

    try:
        # The gate is installed after the authentic original unknown notice.
        evidence = await _exercise_managed_notice(store, store._db, pending=True,
                                                   connect=connect, interleave=interleave)
        seen_stages = [row[0] for row in observed]
        assert seen_stages[:2] == ["probe_returned", "before_commit"]
        assert "committed_before_hint" in seen_stages and "claim" in seen_stages
        evidence.update(contender=observed, external_execute_count=1)
        if output := os.environ.get("R20_CASES"):
            with Path(output).open("a") as stream:
                stream.write(json.dumps(evidence, ensure_ascii=False) + "\n")
    finally:
        if contender is not None:
            await contender.aclose()


@asynccontextmanager
async def _original_unknown(store):
    state = SimpleNamespace(prepared=0, actions=[], probes=[], resolved=True, external_result=None,
                            action_finished=asyncio.Event(), release=asyncio.Event())
    adapter = _adapter(state, pending=True, missing_body=False)
    runner = await RunningAgent(store, ManagedParent(), _registry(adapter)).start()
    try:
        await runner.send("post once")
        async with asyncio.timeout(10):
            await state.action_finished.wait()
        tool = next(row for row in await store.load_all_messages(runner.target.session_id)
                    if row.role == "tool" and row.tool_call_id == _CALL_ID)
        start = InputRef(**json.loads(tool.content)["input_ref"])
        target = TargetRef(start.store_id, start.session_id)
        first_parent = InputRef(start.store_id, runner.target.session_id,
                               result_message_id(target, start, 1, _KIND))
        await runner.drain(after_input=first_parent)
        assert runner.model.notices[0]["payload"]["status"] == "interrupted"
        assert state.prepared == len(state.actions) == 1
        yield runner, start, target, adapter, state
    finally:
        await runner.aclose()


def _record_case(store, case, **data):
    if output := os.environ.get("R20_CASES"):
        with Path(output).open("a") as stream:
            stream.write(json.dumps(plain({"db": store._db.dialect.name, "case": case, **data})) + "\n")


async def test_two_probes_same_and_conflicting_result(registry_store):
    store, connect = registry_store
    async with _original_unknown(store) as (runner, start, target, adapter, state):
        reached = asyncio.Event()
        release = asyncio.Event()
        late, same = await connect(), await connect()
        def delayed(result):
            async def probe(context, reference):
                await adapter.recover(context, reference)
                if len(state.probes) == 2:
                    reached.set()
                await release.wait()
                return result
            return replace(adapter, on_recover=probe)

        other = replace(state.external_result, display="different confirmed body")
        async with asyncio.TaskGroup() as group:
            async def probe_conflict():
                with pytest.raises(RuntimeRequestError, match="request_conflict") as error:
                    await reconcile(late, runner.actor, start, delayed(other))
                return error.value.code

            disagree = group.create_task(probe_conflict())
            agree = group.create_task(reconcile(same, runner.actor, start, delayed(state.external_result)))
            async with asyncio.timeout(10):
                await reached.wait()
            try:
                win = await reconcile(store, runner.actor, start, adapter, client=runner.runtime.client)
                assert win == state.external_result
            finally:
                release.set()
        assert disagree.result() == "request_conflict" and agree.result() == win
        parent_ref = InputRef(start.store_id, runner.target.session_id, result_message_id(target, start, 2, _KIND))
        await runner.drain(after_input=parent_ref)
        fresh = RuntimeInputStore(await connect())
        recovery_ref = managed_effects._recovery_ref(start, 1)
        published = await fresh.get_result(recovery_ref, actor=runner.actor)
        assert published.final_text == win.display and len(state.probes) == 3
        assert state.actions == state.probes[:1] and len(runner.model.notices) == 2
        _record_case(store, "same_and_conflict", start=start, publication=published,
                     probes=state.probes, actions=state.actions, notices=runner.model.notices,
                     conflict=disagree.result())


@pytest.mark.parametrize("busy", ["pending", "active_full", "active_closed"])
async def test_recovery_does_not_consume_unrelated_child_or_lose_observation(registry_store, busy):
    store, connect = registry_store
    async with _original_unknown(store) as (runner, start, target, adapter, state):
        independent = await connect()
        port = RuntimeInputStore(independent)
        original = await port.get_result(start, actor=runner.actor)
        fake = {"managed_effect": "recovered", "result": state.external_result.to_dict()}
        staged = None
        owner = None
        parent_input = None

        async def recover(context, reference):
            nonlocal staged, owner, parent_input
            result = await adapter.recover(context, reference)
            staged = (await port.accept(target, fake, message_id="unrelated", actor=runner.actor)).input_ref
            if busy != "pending":
                owner = await port.claim(target, node_id="other-owner", boot_id="boot")
                parent_port = RuntimeInputStore(independent, inbox_item_limit=1)
                if busy == "active_full":
                    parent_input = (await parent_port.accept(
                        runner.target, "real backlog", actor=runner.actor, message_id="undelivered",
                    )).input_ref
                else:
                    # Hold the real lifecycle lock and close the parent. The public close
                    # also enforces linked-child policy; inject only the lifecycle fact.
                    async with independent._db.transaction() as tx:
                        await independent._db.dialect.lock_state(tx, port.s, runner.target.session_id)
                        await tx.execute(f"UPDATE {store.t.sessions} SET lifecycle='closed' WHERE session_id=?",
                                         (runner.target.session_id,))
            return result

        result = await reconcile(store, runner.actor, start, replace(adapter, on_recover=recover),
                                 client=runner.runtime.client)
        recovery_ref = managed_effects._recovery_ref(start, 1)
        row = await port.input(port.db, staged)
        if busy == "pending":
            assert result == state.external_result and row.row["status"] == "pending"
            assert row.row["terminal_status"] is None
            send_index = 2
        else:
            assert result is None and owner.input_refs == (staged,)
            observed = await port.input(port.db, start)
            assert observed.meta[managed_effects._OBSERVATION]["result"] == state.external_result.to_dict()
            assert await port.input(port.db, recovery_ref) is None
            lock = await port.boundary(target)
            assert lock["send_index"] == owner.send_index and lock["owner_epoch"] == owner.owner_epoch
            assert row.row["terminal_status"] is None and row.row["delivered_send_index"] == owner.send_index
            assert await port.get_result(start, actor=runner.actor) == original
            assert await reconcile(store, runner.actor, start, adapter) is None
            await port.final(replace(owner, result_route=None), status="failed", error="unrelated input")
            # Set the same stricter acceptance limit on the recovery port in this test.
            if parent_input is not None:
                # Fill the actual configured byte cap (1 MiB) with distinct pending items so even
                # internal River delivery (hard_pending_limit=1024) hits capacity_exceeded.
                for index in range(10):
                    await port.accept(
                        runner.target, "x" * 104_800, actor=runner.actor, message_id=f"backlog-{index}",
                    )
            result = await reconcile(store, runner.actor, start, adapter, client=runner.runtime.client)
            assert result == state.external_result
            send_index = owner.send_index + 1
        child = await port.get_result(recovery_ref, actor=runner.actor)
        assert child.status == "completed" and child.send_index == send_index
        assert len(state.probes) == len(state.actions) == 1 and state.prepared == 1
        assert await port.get_result(start, actor=runner.actor) == original
        parent_ref = InputRef(start.store_id, runner.target.session_id,
                             result_message_id(target, start, send_index, _KIND))
        if busy != "pending":
            reason = "capacity_exceeded" if parent_input else "session_closed"
            assert child.result_delivery_error == reason
            assert await port.input(port.db, parent_ref) is None
            if parent_input:
                # Cancel backlog through its native port without starting the parent.
                for item in [parent_input.message_id, *(f"backlog-{i}" for i in range(10))]:
                    await port.cancel(InputRef(start.store_id, runner.target.session_id, item), actor=runner.actor)
            else:
                async with independent._db.transaction() as tx:
                    await independent._db.dialect.lock_state(tx, port.s, runner.target.session_id)
                    await tx.execute(f"UPDATE {store.t.sessions} SET lifecycle='open' WHERE session_id=?",
                                     (runner.target.session_id,))
            assert await reconcile(store, runner.actor, start, adapter) == state.external_result
            assert (await port.get_result(recovery_ref, actor=runner.actor)).result_delivery_error is None
            assert await port.input(port.db, parent_ref) is not None
            await runner.runtime.worker.recover()
        await runner.drain(after_input=parent_ref)
        assert (await port.get_result(parent_ref, actor=runner.actor)).final_text == "parent processed: " + _DISPLAY
        assert len(runner.model.notices) == 2 and len(state.probes) == 1
        _record_case(store, "busy_and_parent_admission", variant=busy, original=original,
                     child=child, parent=parent_ref, unrelated=row.row, actions=state.actions,
                     probes=state.probes, notices=runner.model.notices)


async def test_reconcile_identity_probe_none_and_exact_retry(registry_store):
    store, connect = registry_store
    async with _original_unknown(store) as (runner, start, target, adapter, state):
        ancestor = replace(runner.actor, session_id=runner.target.session_id)
        bad = [(replace(start, store_id="elsewhere"), ancestor, adapter, "wrong_store"),
               (replace(start, message_id="not-the-start"), ancestor, adapter, "forbidden"),
               (start, TrustedActor("other", "host", None), adapter, "forbidden"),
               (start, replace(ancestor, session_id="not-an-ancestor"), adapter, "forbidden"),
               (start, ancestor, replace(adapter, descriptor=replace(adapter.contract, implementation_hash="b" * 64)),
                "forbidden")]
        state.resolved = False
        assert await reconcile(store, ancestor, start, adapter) is None
        fresh = RuntimeInputStore(await connect())
        assert await fresh.input(fresh.db, managed_effects._recovery_ref(start, 1)) is None
        state.resolved = True
        for ref, actor, supplied, code in bad:
            with pytest.raises(RuntimeRequestError) as refused:
                await reconcile(store, actor, ref, supplied)
            assert refused.value.code == code
        subscription = await runner.runtime.client.subscribe_output(
            managed_effects._recovery_ref(start, 1), actor=ancestor, before_send=True,
        )
        try:
            assert await reconcile(store, ancestor, start, adapter, client=runner.runtime.client) == state.external_result
            async with asyncio.timeout(10):
                observed = await subscription.wait()
            assert observed.status == "completed" and observed.structured_output == state.external_result.to_dict()
        finally:
            await subscription.aclose()
        for ref, actor, supplied, code in bad:
            with pytest.raises(RuntimeRequestError) as refused:
                await reconcile(store, actor, ref, supplied)
            assert refused.value.code == code
        assert await reconcile(store, ancestor, start, adapter) == state.external_result
        assert len(state.probes) == 2 and state.prepared == len(state.actions) == 1
        _record_case(store, "identity_and_subscription", rejection_codes=[x[-1] for x in bad],
                     observed=observed, actions=state.actions, probes=state.probes)


async def test_public_forgery_and_original_rerun_cannot_execute(registry_store):
    store, connect = registry_store
    async with _original_unknown(store) as (runner, start, target, adapter, state):
        independent = await connect()
        port = RuntimeInputStore(independent)
        forged = (await runner.runtime.client.send(target, {"managed_effect": "recovered",
            "result": state.external_result.to_dict()}, message_id="user-crafted", actor=runner.actor)).input_ref
        await runner.drain(after_input=forged)
        assert (await port.get_result(forged, actor=runner.actor)).error_code == "managed_effect_original_send_required"
        assert len(runner.model.notices) == 1
        assert (await port.get_result(start, actor=runner.actor)).status == "interrupted"
        # Explicitly replay the original through the real claim and worker under a new
        # send. The public send is idempotent, so reset only its queue state to inject this.
        async with independent._db.transaction() as tx:
            await independent._db.dialect.lock_state(tx, port.s, start.session_id)
            await tx.execute(f"UPDATE {port.q} SET status='pending',delivered_at=NULL,delivered_seq=NULL,"
                             "delivered_send_index=NULL,terminal_status=NULL,result_seq=NULL,"
                             "result_send_index=NULL,completed_at=NULL,error_code=NULL,error_detail=NULL "
                             "WHERE session_id=? AND item_id=?", (start.session_id, start.message_id))
            await tx.execute(
                f"UPDATE {port.s} SET execution_state='queued',wake_pending_count=1,pending_count=1 WHERE session_id=?",
                (start.session_id,),
            )
        await runner.runtime.worker.recover()
        await runner.drain(after_input=start)
        rerun = await port.get_result(start, actor=runner.actor)
        assert rerun.status == "interrupted" and rerun.send_index == 3
        assert state.prepared == len(state.actions) == 1 and not state.probes
        _record_case(store, "forgery_and_rerun", forged=forged, rerun=rerun, actions=state.actions)


@pytest.mark.parametrize("stage", ["prepare", "execute", "cancel_before_dispatch", "dispatch_ack_lost",
                                   "local_result_late", "failed_prepare_late"])
async def test_managed_checkpoint_cannot_write_after_ownership_loss(registry_store, monkeypatch, stage):
    from .session_native_fixture import fail_next_commit

    store, connect = registry_store
    state = SimpleNamespace(prepared=0, actions=[], probes=[], resolved=True, external_result=None,
                            action_finished=asyncio.Event(), release=asyncio.Event())
    source = _adapter(state, pending=False, missing_body=False)
    blocked = asyncio.Event()
    release = asyncio.Event()
    original_write = managed_effects._write_owned_state

    async def prepare(context, arguments):
        result = await source.prepare(context, arguments)
        if stage in {"prepare", "cancel_before_dispatch", "local_result_late", "failed_prepare_late"}:
            blocked.set()
            await release.wait()
        if stage == "failed_prepare_late":
            raise ValueError("adapter preparation refused after ownership changed")
        if stage == "local_result_late":
            return replace(result, local_result=ManagedToolResult(_DISPLAY, result.reference, "not_dispatched"))
        return result

    async def execute(context, reference):
        state.release.set()
        result = await source.execute(context, reference)
        if stage == "execute":
            blocked.set()
            await release.wait()
        return result

    async def lose_ack(port, batch, key, value):
        if value["phase"] == "dispatched":
            fail_next_commit(monkeypatch, store)
        return await original_write(port, batch, key, value)

    async def recover_never_dispatched(context, reference):
        assert context.call_ref.to_dict() == reference["call_ref"]
        assert context.tool_call_id == reference["original_call"] and not state.actions
        state.probes.append(reference)
        return None

    if stage == "dispatch_ack_lost":
        monkeypatch.setattr(managed_effects, "_write_owned_state", lose_ack)
    adapter = replace(source, on_prepare=prepare, on_execute=execute)
    if stage == "dispatch_ack_lost":
        adapter = replace(adapter, on_recover=recover_never_dispatched)
    runner = await RunningAgent(store, ManagedParent(), _registry(adapter)).start()
    try:
        await runner.send("post once")
        receipt = next(json.loads(row.content) for row in await store.load_all_messages(runner.target.session_id)
                       if row.role == "tool" and row.tool_call_id == _CALL_ID)
        start = InputRef(**receipt["input_ref"])
        target = TargetRef(**receipt["target_ref"])
        independent = await connect()
        port = RuntimeInputStore(independent)
        if stage == "dispatch_ack_lost":
            await runner.drain(after_input=start)
            assert not state.actions
            assert await port.get_result(start, actor=runner.actor) is None
        else:
            async with asyncio.timeout(10):
                await blocked.wait()
        if stage == "cancel_before_dispatch":
            await port.cancel(start, actor=runner.actor)
        else:
            await independent._db.execute(
                f"UPDATE {port.s} SET owner_lease_until_ms=0 WHERE session_id=?", (start.session_id,),
            )
            recovered = await port.recover(target, node_id="replacement", boot_id="new-boot")
            assert recovered is not None and recovered.results[0].status == "interrupted"
        async def persisted():
            return await independent._db.fetchall(
                f"SELECT state_key,value_json FROM {store.t.session_runtime_state} WHERE session_id=?",
                (start.session_id,),
            )
        before = await persisted()
        release.set()
        await runner.drain()
        after = await persisted()
        assert before == after
        if stage in {"execute", "dispatch_ack_lost"}:
            [checkpoint] = after
            assert json.loads(checkpoint["value_json"])["phase"] == "dispatched"
        else:
            assert after == []
        result = await port.get_result(start, actor=runner.actor)
        assert result.status == ("cancelled" if stage == "cancel_before_dispatch" else "interrupted")
        assert len(state.actions) == (1 if stage == "execute" else 0)
        if stage == "dispatch_ack_lost":
            preparation = json.loads(after[0]["value_json"])["preparation"]
            assert await reconcile(store, runner.actor, start, adapter, client=runner.runtime.client) is None
            assert state.probes == [preparation["reference"]] and not state.actions and state.prepared == 1
            assert await persisted() == after
            assert await port.get_result(start, actor=runner.actor) == result
            assert await port.input(port.db, managed_effects._recovery_ref(start, 1)) is None
            parent_ref = InputRef(start.store_id, runner.target.session_id, result_message_id(target, start, 2, _KIND))
            assert await port.input(port.db, parent_ref) is None
        _record_case(store, stage, result=result, before=[dict(row) for row in before],
                     after=[dict(row) for row in after], actions=state.actions, probes=state.probes)
    finally:
        release.set()
        await runner.aclose()


async def test_recovery_commit_ack_lost_reads_same_result(registry_store, monkeypatch):
    from .session_native_fixture import fail_next_commit

    store, connect = registry_store
    async with _original_unknown(store) as (runner, start, target, adapter, state):
        fail_next_commit(monkeypatch, store)
        outcome = await reconcile(store, runner.actor, start, adapter, client=runner.runtime.client)
        assert outcome == state.external_result
        await runner.runtime.worker.recover()  # even without a hint, the normal parent queue is durable
        parent_ref = InputRef(start.store_id, runner.target.session_id, result_message_id(target, start, 2, _KIND))
        await runner.drain(after_input=parent_ref)
        fresh = RuntimeInputStore(await connect())
        published = await fresh.get_result(managed_effects._recovery_ref(start, 1), actor=runner.actor)
        assert published.structured_output == outcome.to_dict()
        assert (await fresh.get_result(parent_ref, actor=runner.actor)).final_text == "parent processed: " + _DISPLAY
        assert (await fresh.get_result(start, actor=runner.actor)).status == "interrupted"
        assert len(state.probes) == len(state.actions) == 1
        _record_case(store, "recovery_ack_lost", recovery=published, parent=parent_ref,
                     actions=state.actions, probes=state.probes)


@pytest.mark.parametrize("resolution", ["rejected", "not_dispatched"])
async def test_recovery_keeps_adapter_resolution_and_reference(registry_store, resolution):
    store, connect = registry_store
    async with _original_unknown(store) as (runner, start, target, adapter, state):
        async def resolved(context, reference):
            source = await adapter.recover(context, reference)
            # A provider can use a receipt ID rather than echo the request ID.
            return replace(source, resolution=resolution, reference={"provider_receipt": "different-id"})
        outcome = await reconcile(store, runner.actor, start, replace(adapter, on_recover=resolved),
                                  client=runner.runtime.client)
        child = await RuntimeInputStore(await connect()).get_result(
            managed_effects._recovery_ref(start, 1), actor=runner.actor,
        )
        assert child.structured_output == outcome.to_dict()
        assert child.structured_output["reference"] == {"provider_receipt": "different-id"}
        assert child.structured_output["resolution"] == resolution
        parent_ref = InputRef(start.store_id, runner.target.session_id,
                             result_message_id(target, start, child.send_index, _KIND))
        await runner.drain(after_input=parent_ref)
        assert runner.model.notices[-1]["payload"]["structured_output"] == outcome.to_dict()
        assert state.prepared == len(state.actions) == len(state.probes) == 1
        _record_case(store, "provider_resolution", resolution=resolution, child=child,
                     actions=state.actions, probes=state.probes)


async def test_child_close_during_probe_prevents_publication(registry_store):
    store, connect = registry_store
    async with _original_unknown(store) as (runner, start, target, adapter, state):
        other = RuntimeInputStore(await connect())
        async def probe(context, reference):
            result = await adapter.recover(context, reference)
            await other.close(target, actor=runner.actor)
            return result

        with pytest.raises(RuntimeRequestError) as error:
            await reconcile(store, runner.actor, start, replace(adapter, on_recover=probe),
                            client=runner.runtime.client)
        assert error.value.code == "session_closed"
        assert await other.input(other.db, managed_effects._recovery_ref(start, 1)) is None
        assert (await other.get_result(start, actor=runner.actor)).status == "interrupted"
        assert len(state.actions) == len(state.probes) == 1 and len(runner.model.notices) == 1
        _record_case(store, "child_closed", start=start, code=error.value.code,
                     actions=state.actions, probes=state.probes)


@pytest.mark.parametrize("checkpoint", ["dispatched", "completed", "failed_prepare"])
async def test_explicit_reconcile_after_native_takeover(registry_store, monkeypatch, checkpoint):
    from tests.write_meter import WriteMeter

    from .test_host_ports_cost_profile_208 import PortsProfile, _category

    store, connect = registry_store
    state = SimpleNamespace(prepared=0, actions=[], probes=[], resolved=False, external_result=None,
                            action_finished=asyncio.Event(), release=asyncio.Event())
    source = _adapter(state, pending=False, missing_body=False)
    blocked, release = asyncio.Event(), asyncio.Event()
    original_db = store._db
    save = managed_effects._write_owned_state
    notify = RuntimeClient._notify
    current_task = asyncio.current_task()
    held_parent_hints = []
    measure_parent = None

    async def hold_parent_hint(self, target, message_id, ref, channels, payload=None):
        if asyncio.current_task() is current_task and ref == measure_parent and channels == ["work", "activation"]:
            held_parent_hints.append((self, target, message_id, ref, channels, payload))
            return True
        return await notify(self, target, message_id, ref, channels, payload)

    async def pause_after_save(port, batch, key, value):
        await save(port, batch, key, value)
        if checkpoint != "dispatched" and value["phase"] == "completed":
            blocked.set()
            await release.wait()

    async def prepare(context, arguments):
        prepared = await source.prepare(context, arguments)
        if checkpoint == "failed_prepare":
            raise ValueError("local preparation refused")
        return prepared

    async def execute(context, reference):
        state.release.set()
        result = await source.execute(context, reference)
        if checkpoint == "dispatched":
            blocked.set()
            await release.wait()
        return result

    async def recover(context, reference):
        assert checkpoint == "dispatched", "durable results must not query the adapter"
        return await source.recover(context, reference)

    class Parent(ManagedParent):
        async def complete(self, request, **kwargs):
            count = len(self.notices)
            reply = await super().complete(request, **kwargs)
            if len(self.notices) == count or checkpoint != "failed_prepare":
                return reply
            notice = self.notices[-1]
            output = notice["payload"].get("structured_output") or {}
            if (output.get("resolution") == "not_dispatched"
                    and output["reference"] == {"decision": "not_dispatched", "error": "local preparation refused"}
                    and notice["payload"]["final_text"] == output["display"]
                    and notice["projected_origin"] is not None
                    and notice["payload"]["child_target"]["session_id"] == notice["projected_origin"][1]):
                notice["reply"] = "parent processed: " + output["display"]
                return LLMResponse(raw_text=notice["reply"])
            return reply

    monkeypatch.setattr(managed_effects, "_write_owned_state", pause_after_save)
    monkeypatch.setattr(RuntimeClient, "_notify", hold_parent_hint)
    adapter = replace(source, on_prepare=prepare, on_execute=execute, on_recover=recover)
    model = Parent()
    meter = await WriteMeter.install(original_db, store.table_prefix)
    profile = PortsProfile(original_db, meter)
    store._db = profile
    runner = await RunningAgent(store, model, _registry(adapter)).start()
    client = runner.runtime.client
    data = {"checkpoint_type": checkpoint}
    fresh = None
    try:
        await runner.send("post once")
        async with asyncio.timeout(10):
            await blocked.wait()
        [receipt] = [json.loads(row.content) for row in await store.load_all_messages(runner.target.session_id)
                     if row.role == "tool" and row.tool_call_id == _CALL_ID]
        start = InputRef(**receipt["input_ref"])
        target = TargetRef(**receipt["target_ref"])
        independent = await connect()
        port = RuntimeInputStore(independent)

        async def saved_checkpoint():
            [row] = await port.db.fetchall(
                f"SELECT value_json FROM {store.t.session_runtime_state} WHERE session_id=?", (start.session_id,),
            )
            return json.loads(row["value_json"])

        # Before native takeover no external query or replacement publication is permitted.
        pending = await reconcile(store, runner.actor, start, adapter, client=client)
        assert not state.probes
        if checkpoint == "dispatched":
            assert pending is None
        data["running_reconcile_returned"] = plain(pending)
        await port.db.execute(f"UPDATE {port.s} SET owner_lease_until_ms=0 WHERE session_id=?",
                              (start.session_id,))
        takeover = await port.recover(target, node_id="replacement", boot_id="replacement")
        assert len(takeover.results) == 1
        assert takeover.results[0].input_ref == start and takeover.results[0].status == "interrupted"
        release.set()
        await runner.runtime.worker.recover()
        await runner.drain(after_input=takeover.result_ref)
        original = await port.get_result(start, actor=runner.actor)
        original_parent = await port.get_result(takeover.result_ref, actor=runner.actor)
        assert original.status == "interrupted" and original_parent.final_text == "parent processed unknown outcome"
        saved = await saved_checkpoint()
        assert saved["phase"] == ("dispatched" if checkpoint == "dispatched" else "completed")
        assert "recovered_send_index" not in saved
        if checkpoint == "failed_prepare":
            assert saved["preparation"] is None and not state.actions
            expected = ManagedToolResult.from_dict(saved["result"])
        else:
            assert state.prepared == len(state.actions) == 1
            expected = state.external_result
        recovered_ref = managed_effects._recovery_ref(start, original.send_index)
        parent_ref = InputRef(start.store_id, runner.target.session_id,
                              result_message_id(target, start, 2, _KIND))
        data.update(original_before=original, initial_parent=original_parent, saved_before=saved,
                    expected=expected.to_dict(), recovery_ref=recovered_ref, parent_ref=parent_ref)
        if checkpoint == "dispatched":
            assert await reconcile(store, runner.actor, start, adapter, client=client) is None
            # The baseline misses this probe; continue to capture its publication-window cost.
            data["miss_probes"] = len(state.probes)
            assert await saved_checkpoint() == saved
            assert await port.input(port.db, recovered_ref) is None
            state.resolved = True

        async def snapshot(part):
            records = [{**row, "category": _category(row["chain"], row["sql"])} for row in profile.records]
            assert len(profile.tx_sites) == len(profile.tx_categories) == profile.transactions
            assert None not in profile.tx_categories
            evidence = dict(operation=f"takeover_{checkpoint}", part=part, records=records,
                            transactions=profile.transactions, tx_sites=profile.tx_sites,
                            tx_categories=profile.tx_categories, writes=await meter.report())
            if output := os.environ.get("P7_PORTS_OUT"):
                with Path(output).open("a") as stream:
                    stream.write(json.dumps(evidence) + "\n")

        await meter.reset()
        profile.records.clear()
        profile.tx_sites.clear()
        profile.tx_categories.clear()
        profile.transactions = 0
        measure_parent = parent_ref
        result = await reconcile(store, runner.actor, start, adapter, client=client)
        await snapshot("child")
        had_hint = bool(held_parent_hints)
        if had_hint:
            [args] = held_parent_hints
            await notify(*args)
            await runner.drain(after_input=parent_ref)
        await snapshot("total")
        data.update(outcome=plain(result), parent_hints=len(held_parent_hints))
        assert result == expected and had_hint, data
        assert data.get("miss_probes", 0) == (1 if checkpoint == "dispatched" else 0)

        fresh = RuntimeClient(store=await connect(), transport=runner.runtime.transport)
        results = {label: await fresh.inputs.get_result(ref, actor=runner.actor)
                   for label, ref in (("original", start), ("child", recovered_ref), ("parent", parent_ref))}
        statuses = {label: await fresh.inspect(result.input_ref, actor=runner.actor)
                    for label, result in results.items()}
        data.update(results=results, statuses=statuses)
        assert results["original"] == original
        recovered, parent = results["child"], results["parent"]
        assert recovered.status == parent.status == "completed"
        assert recovered.input_ref == recovered_ref and parent.input_ref == parent_ref
        assert recovered.structured_output == expected.to_dict() and recovered.final_text == expected.display
        assert recovered.send_index == 2
        for label, observed in statuses.items():
            assert (observed.terminal_status, observed.result_seq) == (results[label].status, results[label].result_seq)

        [child_user, child_answer] = [row for row in (await fresh.messages(target, actor=runner.actor)).items
                                      if row.send_index == recovered.send_index]
        assert child_user.seq + 1 == child_answer.seq == recovered.result_seq
        assert child_answer.content == expected.display
        assert json.loads(child_user.content)["items"][0]["content"]["result"] == expected.to_dict()
        [before, after] = model.notices
        assert before["input_ref"] == plain(takeover.result_ref)
        assert after["input_ref"] == plain(parent_ref) and after["send_index"] == parent.send_index
        assert after["reply"] == parent.final_text == f"parent processed: {expected.display}"
        queued = await port.input(port.db, parent_ref)
        assert after["payload"] == json.loads(queued.row["content_json"])
        assert after["payload"]["input_ref"] == plain(recovered_ref)
        assert after["payload"]["structured_output"] == expected.to_dict()
        assert await fresh.inputs.get_result(takeover.result_ref, actor=runner.actor) == original_parent
        assert await reconcile(store, runner.actor, start, adapter, client=client) == expected
        await runner.drain()
        assert len(model.notices) == 2 and len(held_parent_hints) == 1
        assert len(state.probes) == (2 if checkpoint == "dispatched" else 0)
        assert state.prepared == 1 and len(state.actions) == (0 if checkpoint == "failed_prepare" else 1)
        data["repeat"] = "same result, one recovery notice"
    finally:
        _record_case(store, f"takeover_{checkpoint}", **data, prepared=state.prepared,
                     actions=state.actions, probes=state.probes, model_notices=model.notices)
        release.set()
        if fresh is not None:
            await fresh.aclose()
        await runner.aclose()
        store._db = original_db
        await meter.close()
