"""Real model reads registered legacy node history before/after driver cleanup."""
from __future__ import annotations

import asyncio
import json

import pytest

from power_loop import AgentLoopConfig, SessionStore, StatefulAgentLoop
from power_loop.runtime.env import RuntimeEnv, runtime_env_context
from power_loop.runtime.representation import ProjectedRepresentation
from power_loop.tools.registry import ToolRegistry
from power_loop.workflow import journal
from power_loop.workflow.facade import WorkflowToolPorts, register_workflow_tool

from .test_real_async_boundaries_192 import BoundedProvider

pytestmark = [pytest.mark.real_llm, pytest.mark.asyncio]


@pytest.mark.parametrize("layout", ["single", "foreach"])
async def test_real_legacy_leaves_logs_projection_and_export_never_reexecute_nodes(tmp_path, record_property, layout):
    store = await SessionStore.open(str(tmp_path/"legacy.db"))
    provider, accepted = BoundedProvider(limit=18), []
    async def admission(value):
        accepted.append(value)
    async def completed(_):
        pass
    registry = ToolRegistry()
    register_workflow_tool(registry, ports=WorkflowToolPorts(on_accepted=admission, on_complete=completed))
    loop = StatefulAgentLoop(llm=provider, store=store, tool_registry=registry,
        config=AgentLoopConfig(delegation_policy="single_level", max_rounds=5, max_tokens=2048,
            representation=ProjectedRepresentation(), system_prompt="Follow the exact requested workflow call once per request, "
                "then give an actual assistant answer. Reading history never resumes or executes nodes. "
                "Do not poll or use pass_turn to replace the assistant answer. Legacy views do not prove physical stop or billing settlement."))
    sid = await loop.new_session()
    provider.root_sid, provider.hold_children = sid, True
    leaf = {"type": "agent", "id": "calculate", "input": "Compute 19+23 and give the number in assistant text.",
        "spec": {"name": "calculate", "system_prompt": "Compute the supplied arithmetic and return its number.", "tools": [], "max_rounds": 2}}
    spec = {"name": "legacy-audit", "root": leaf if layout == "single" else {
        "type": "foreach", "id": "each", "as": "item", "items": ["a", "b"], "max_concurrency": 2, "body": leaf}}
    try:
        with runtime_env_context(RuntimeEnv(workspace_dir=tmp_path)):
            async with asyncio.timeout(180):
                await loop.send("Call workflow action=create exactly once with spec="+json.dumps(spec)+
                    ". After acceptance finish with WAITING and no more tools.", sid)
                assert len(accepted) == 1
                handle = accepted[0].handle
                await asyncio.wait_for(provider.child_entered.wait(), 10)
                async def registered():
                    while True:
                        state = await store.get_runtime_state(sid, "workflow:history:"+handle.run_id)
                        if state and len(state["sessions"]) == (1 if layout == "single" else 2):
                            return
                        await asyncio.sleep(.02)
                await asyncio.wait_for(registered(), 10)
                await loop.send("Read only the registered sessions. Call workflow action=leaves exactly once with run_id='"+
                    handle.run_id+"'. Then report the session IDs in assistant text; no other tools.", sid)
                tools = [json.loads(m.content) for m in await store.load_all_messages(sid) if m.role == "tool" and m.name == "workflow"]
                listing = tools[-1]
                assert listing["backend"] == "legacy" and listing["history_complete"]
                assert len(listing["nodes"]) == (1 if layout == "single" else 2)
                assert all(n["available"] and n["binding_source"] == "owned_history_registry" for n in listing["nodes"])
                provider.release_children.set()
                await asyncio.wait_for(handle.task, 45)
                before = {n["session_id"]: await store.load_all_messages(n["session_id"]) for n in listing["nodes"]}
                for physical in before:
                    assert (await store.get_session(physical)).parent_session_id is None
                snapshot = await journal.read(store, sid, handle.run_id)
                physical = listing["nodes"][0]["session_id"]
                arguments = {"action": "logs", "run_id": handle.run_id, "sid": physical}
                answer = await loop.send("Read only saved work. Call workflow exactly once with "+json.dumps(arguments)+
                    ". Then report the saved arithmetic answer in assistant text, no other tools.", sid)
                assert answer.status == "completed" and "42" in answer.final_text
                tools = [json.loads(m.content) for m in await store.load_all_messages(sid) if m.role == "tool" and m.name == "workflow"]
                assert tools[-1]["format"] == "round_preview" and "42" in str(tools[-1]["rounds"])
                assert tools[-1]["raw_message_id_status"] == "unavailable" and not tools[-1]["execution_authority"]
                await loop.send("Read saved projections only. Call workflow exactly once with "+json.dumps({
                    **arguments, "source": "projection", "limit": 20})+". Then report the saved arithmetic answer in assistant text.", sid)
                tools = [json.loads(m.content) for m in await store.load_all_messages(sid) if m.role == "tool" and m.name == "workflow"]
                assert tools[-1]["status"] == "ok" and tools[-1]["source"] == "projection"
                output = "audit/legacy.jsonl"
                await loop.send("Export only the existing raw records. Call workflow exactly once with "+json.dumps({
                    **arguments, "output_file": output})+". Then report its returned path/count in assistant text; no other tools.", sid)
                tools = [json.loads(m.content) for m in await store.load_all_messages(sid) if m.role == "tool" and m.name == "workflow"]
                receipt = tools[-1]
                assert receipt["status"] == "exported" and receipt["path"] == output and receipt["records"] > 0
                rows = [json.loads(line) for line in (tmp_path/output).read_text().splitlines()]
                assert len(rows) == receipt["records"] and all(row["session_id"] == physical and row["id"] is None for row in rows)
                for physical, records in before.items():
                    assert await store.load_all_messages(physical) == records
                assert snapshot == await journal.read(store, sid, handle.run_id) and len(accepted) == 1
    finally:
        provider.release_children.set()
        try:
            await asyncio.wait_for(asyncio.gather(*(a.handle.task for a in accepted), return_exceptions=True), 15)
        finally:
            record_property("main_transcript", json.dumps([dict(id=str(m.id), seq=str(m.seq), role=m.role,
                name=m.name, content=m.content, tool_calls=m.tool_calls, tool_call_id=m.tool_call_id)
                for m in await store.load_all_messages(sid)], ensure_ascii=False))
            provider.report(record_property)
            await provider.close()
            await store.close()
