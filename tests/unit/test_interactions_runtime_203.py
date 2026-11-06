"""Answers resolve a persisted interaction after the original worker exits."""
import asyncio
import json

import pytest

from power_loop import AgentLoopConfig, LLMResponse, LLMService
from power_loop.agent.runtime_store import now_ms
from power_loop.agent.runtime_types import RuntimeRequestError
from power_loop.tools import create_default_tool_registry
from tests.runtime_fixture import RunningAgent

from .test_job_registry_192 import registry_store

__all__ = ["registry_store"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


class AskingModel(LLMService):
    def __init__(self):
        self.calls = []

    async def complete(self, request, **kwargs):
        self.calls.append(request)
        answers = [m for m in request.messages if m.get("tool_call_id") == "question"]
        if not answers:
            return LLMResponse(raw_text="", tool_calls=[{"id": "question", "type": "function",
                "function": {"name": "request_user_input", "arguments": json.dumps({"prompt": "Which value?"})}}])
        return LLMResponse(raw_text="answer=" + answers[-1]["content"])

    async def close(self):
        pass


@pytest.mark.parametrize("mode", ["main", "child", "batched"])
async def test_answer_after_worker_change_is_exactly_one_tool_response(registry_store, tmp_path, mode):
    store, _ = registry_store
    model = AskingModel()
    registry = create_default_tool_registry(workspace_dir=str(tmp_path))
    if mode == "child":
        from power_loop.tools.spawn_agent import register_spawn_agent

        from .test_subagent_runtime_203 import ParentAndChild
        parent = ParentAndChild()
        complete = model.complete
        async def child_question(request, **kwargs):
            if request.system_prompt.startswith("CHILD"):
                return await complete(request, **kwargs)
            response = await parent.complete(request, **kwargs)
            if response.tool_calls:
                args = json.loads(response.tool_calls[0]["function"]["arguments"])
                args["tools"] = ["request_user_input"]
                response.tool_calls[0]["function"]["arguments"] = json.dumps(args)
            return response
        model.complete = child_question
        register_spawn_agent(registry)
    elif mode == "batched":
        complete = model.complete
        async def batched(request, **kwargs):
            response = await complete(request, **kwargs)
            if response.tool_calls:
                response.tool_calls.append({"id":"later","type":"function","function":{"name":"pass_turn","arguments":"{}"}})
            return response
        model.complete = batched
    first = await RunningAgent(store, model, registry, AgentLoopConfig(compactor=None)).start()
    second = None
    try:
        receipt = await first.runtime.client.send(first.target, "ask", message_id="ask", actor=first.actor)
        target = first.target
        if mode == "child":
            from power_loop.agent.runtime_types import InputRef, TargetRef
            await first.runtime.client.wait(receipt.input_ref, actor=first.actor, deadline_ms=now_ms()+10000)
            rows = await store.load_all_messages(first.target.session_id)
            created = json.loads(next(r.content for r in rows if r.tool_call_id == "spawn"))
            receipt = type("Receipt", (), {"input_ref":InputRef(**created["input_ref"])})()
            target = TargetRef(store.store_id, created["session_id"])
        for _ in range(300):
            state = await store.get_state(target.session_id)
            if state.pending and state.pending.get("runtime_wait"):
                break
            await asyncio.sleep(0.01)
        assert state.pending.get("pending_interactions")
        waiting = state.pending
        interaction = waiting["pending_interactions"][0]
        status = await first.runtime.client.inspect(receipt.input_ref, actor=first.actor)
        assert status.state == "waiting_input" and not status.result_available
        with pytest.raises(RuntimeRequestError, match="waiting_input"):
            await first.runtime.client.send(target, "ordinary message", message_id="not-answer", actor=first.actor)
        await first.runtime.worker.shutdown(timeout_s=2)
        second = await RunningAgent(store, model, registry, AgentLoopConfig(compactor=None)).start()
        client = second.runtime.client
        kwargs = dict(interaction_id=interaction["interaction_id"], wait_round=waiting["round_index"],
                      value="blue", message_id="answer", actor=first.actor)
        answered = await client.answer(status.job_ref, **kwargs)
        repeated = await client.answer(status.job_ref, **kwargs)
        assert repeated.input_ref == answered.input_ref
        with pytest.raises(RuntimeRequestError, match="interaction_conflict"):
            await client.answer(status.job_ref, **{**kwargs, "value": "red"})
        result = await client.wait(receipt.input_ref, actor=first.actor, deadline_ms=now_ms() + 10000)
        assert result.job_ref == status.job_ref
        assert result.status == "completed" and result.final_text == "answer=blue"
        rows = await store.load_all_messages(target.session_id)
        assert len([r for r in rows if r.tool_call_id == "question"]) == 1
        final = next(r for r in rows if r.role == "assistant" and r.content == "answer=blue")
        assert final.send_index == 1 and final.round_index == waiting["round_index"] + 1
        assert result.usage["calls"] == 2
    finally:
        if second:
            await second.aclose()
        await first.aclose()


@pytest.mark.parametrize("reject", ["expired", "cancelled", "wrong_round"])
async def test_invalid_answer_preserves_original_question(registry_store, tmp_path, reject):
    model = AskingModel()
    store, _ = registry_store
    registry = create_default_tool_registry(workspace_dir=str(tmp_path))
    runner = await RunningAgent(store, model, registry, AgentLoopConfig(compactor=None)).start()
    try:
        receipt = await runner.runtime.client.send(runner.target, "ask", message_id="ask", actor=runner.actor)
        for _ in range(300):
            state = await store.get_state(runner.target.session_id)
            if state.pending and state.pending.get("runtime_wait"):
                break
            await asyncio.sleep(.01)
        pending = state.pending
        question = pending["pending_interactions"][0]
        status = await runner.runtime.client.inspect(receipt.input_ref,actor=runner.actor)
        if reject == "expired":
            question["metadata"]["expires_at_ms"] = now_ms()-1
            await store.set_pending(runner.target.session_id,pending)
        elif reject == "cancelled":
            await runner.runtime.client.cancel(status.job_ref,actor=runner.actor)
            result = await runner.runtime.client.wait(receipt.input_ref,actor=runner.actor,deadline_ms=now_ms()+1000)
            assert result.status=="cancelled"
        with pytest.raises(RuntimeRequestError, match="interaction_expired|target_cancelled"):
            await runner.runtime.client.answer(status.job_ref,interaction_id=question["interaction_id"],
                wait_round=pending["round_index"]+(1 if reject=="wrong_round" else 0),value="invalid",
                message_id="invalid",actor=runner.actor)
        rows = await store.load_all_messages(runner.target.session_id)
        assert all(row.content!="invalid" for row in rows)
        assert len(model.calls)==1
    finally:
        await runner.aclose()
