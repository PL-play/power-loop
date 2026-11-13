"""Actual model uses the shared workflow tool to send a correction to its original live node."""
from __future__ import annotations

import asyncio
import json

import pytest

from power_loop import AgentLoopConfig, InputRef, SessionStore
from power_loop.tools.registry import ToolRegistry
from power_loop.workflow.facade import WorkflowToolPorts, register_workflow_tool

from ._bounded import BoundedProvider

pytestmark = [pytest.mark.real_llm, pytest.mark.asyncio]


@pytest.mark.parametrize("urgent", [True, False])
@pytest.mark.parametrize("invalid_first", [False, True])
async def test_real_shared_workflow_send_preserves_original_node_and_call_refs(
    tmp_path, record_property, urgent, invalid_first, runtime_agent,
):
    store = await SessionStore.open(str(tmp_path / "node-facade.db"), schema="tasks")
    provider = BoundedProvider(limit=18)
    accepted = []

    async def on_accepted(admission):
        accepted.append(admission)

    registry = ToolRegistry()
    register_workflow_tool(registry, ports=WorkflowToolPorts(on_accepted=on_accepted))
    loop = await runtime_agent(
        model=provider, store=store, registry=registry, capacity=2,
        config=AgentLoopConfig(
            delegation_policy="single_level", max_rounds=5, max_tokens=2048,
            system_prompt="Follow the requested workflow tool action exactly once in each turn. "
            "An accepted receipt is not a result. Never create replacement work, extend rounds or poll. "
            "After an accepted receipt reply WAITING with no more calls. "
            "When the workflow result is delivered, report its computed number in assistant text without tools.",
        ),
    )
    parent = loop.target.session_id
    provider.root_sid, provider.hold_children = parent, True
    spec = {
        "name": "real-node-control",
        "root": {
            "type": "agent", "id": "calculate", "input": "Compute 19+23.",
            "spec": {
                "name": "calculate",
                "system_prompt": "Compute the latest supplied arithmetic correction. Return only the number.",
                "tools": [], "max_rounds": 3,
            },
        },
    }
    node_sid = None
    try:
        async with asyncio.timeout(240):
            await loop.send(
                "Call workflow action=create exactly once with spec " + json.dumps(spec)
                + ". After acceptance reply WAITING and stop. Do not call any other workflow action yet."
            )
            await asyncio.wait_for(provider.child_entered.wait(), 10)
            [admission] = accepted
            handle = admission.handle
            children = await loop.runtime.client.list_children(handle.target, actor=loop.actor)
            assert len(children.items) == 1
            node_sid = children.items[0].session_id
            if invalid_first:
                invalid = {
                    "action": "send", "run_id": handle.run_id,
                    "node_session_id": "ses_" + "f" * 56,
                    "message": "Do not dispatch a substitute node.",
                }
                await loop.send(
                    "Call workflow exactly once with these arguments, deliberately selecting an unknown saved "
                    "session: " + json.dumps(invalid)
                    + ". The expected result is an error. After this error reply WAITING and stop; "
                    "do not correct, retry or replace anything yet."
                )
                errors = [
                    m for m in await store.load_all_messages(parent)
                    if m.role == "tool" and m.name == "workflow"
                ]
                assert len(errors) == 2
                assert json.loads(errors[-1].content)["error"]["code"] == "workflow_node_not_found", errors
                assert not (await loop.runtime.client.inspect(handle.input_ref, actor=loop.actor)).result_available
            message = "Correction: compute 20+23 instead of 19+23. Return only the resulting number."
            options = {
                "action": "send", "run_id": handle.run_id,
                "node_session_id": node_sid, "message": message, "urgent": urgent,
            }
            await loop.send(
                "Call workflow exactly once with these arguments: " + json.dumps(options)
                + ". Then reply WAITING. Do not create, resume, poll or extend anything."
            )
            messages = [
                m for m in await store.load_all_messages(parent)
                if m.role == "tool" and m.name == "workflow"
            ]
            assert len(messages) == (3 if invalid_first else 2), messages
            receipt = json.loads(messages[-1].content)
            assert receipt["input_ref"]["session_id"] == node_sid
            assert receipt["target"]["session_id"] == node_sid
            queued = await loop.runtime.client.inputs.input(
                loop.runtime.client.inputs.db, InputRef(**receipt["input_ref"]),
            )
            assert json.loads(queued.row["content_json"]) == message
            assert not (await loop.runtime.client.inspect(handle.input_ref, actor=loop.actor)).result_available
            provider.release_children.set()
            from power_loop.agent.runtime_store import now_ms
            finished = await handle.wait(deadline_ms=now_ms() + 100000)
            await loop.drain()
            assert finished.results["calculate"].text.strip() == "43"
            assert (await store.inbox_counts(parent))["pending"] == 0
            assert len(accepted) == 1
            drivers = await store._db.fetchall(
                f"SELECT session_id FROM {store.t.sessions} WHERE session_kind='workflow_driver'",
            )
            assert len(drivers) == 1
            nodes = await store._db.fetchall(
                f"SELECT session_id FROM {store.t.sessions} WHERE parent_session_id=?",
                (handle.target.session_id,),
            )
            assert len(nodes) == 1 and nodes[0]["session_id"] == node_sid
    finally:
        provider.release_children.set()
        try:
            await loop.aclose()
        finally:
            for name, sid in (("main", parent), ("node", node_sid)):
                if sid:
                    record_property(name + "_transcript", json.dumps([
                        dict(id=str(m.id), seq=str(m.seq), role=m.role, name=m.name, content=m.content,
                             tool_calls=m.tool_calls, tool_call_id=m.tool_call_id)
                        for m in await store.load_all_messages(sid)
                    ], ensure_ascii=False))
            provider.report(record_property)
            await provider.close()
            await store.close()
