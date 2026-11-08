"""Real-LLM end-to-end test for power_loop.workflow.

Exercises the full middle-tier path against the configured provider:
WorkflowSpec validation → engine → output_schema enforced as a real
response_format → foreach fan-out → inputs_from → usage recorded (B2).

Skipped automatically when no POWER_LOOP_* / OPENAI_COMPAT_* provider env is set
(or with --no-real). See tests/conftest.py.
"""

from __future__ import annotations

import pytest

from ._workflow_runtime import run_workflow

pytestmark = pytest.mark.real_llm


SPEC = {
    "name": "research_and_summarize",
    "input": "the Japanese tea ceremony",
    "root": {
        "type": "sequence",
        "steps": [
            {
                "type": "agent", "id": "plan",
                "spec": {"name": "planner", "system_prompt": (
                    "Break the topic into exactly 2 short subtopics. "
                    'Respond ONLY with JSON: {"subtopics": ["...", "..."]}.')},
                "input": "Topic: {{input}}",
                "output_schema": {"name": "Plan", "schema": {
                    "type": "object", "required": ["subtopics"],
                    "properties": {"subtopics": {"type": "array", "items": {"type": "string"}}}}},
            },
            {
                "type": "foreach", "id": "research",
                "items_from": "plan.subtopics", "as": "subtopic",
                "parallel": True, "max_concurrency": 2,
                "body": {"type": "agent", "id": "researcher", "spec": {
                    "name": "researcher",
                    "system_prompt": "Write one factual sentence about the given subtopic."},
                    "input": "Subtopic: {{subtopic}}"},
            },
            {
                "type": "agent", "id": "synthesize",
                "spec": {"name": "writer", "system_prompt": "Synthesize the notes into one sentence."},
                "inputs_from": ["research"],
            },
        ],
    },
}


async def test_real_workflow_sequence_foreach_structured(runtime_agent):

    result = await run_workflow(runtime_agent, SPEC)

    assert result.status == "completed", f"errors: {result.errors}"
    # output_schema was enforced at the provider → structured subtopics parsed
    plan = result.results["plan"].payload
    assert plan and isinstance(plan.get("subtopics"), list) and plan["subtopics"]
    # foreach fanned out one researcher per subtopic
    assert len(result.results["research"].payload["items"]) == len(plan["subtopics"])
    # synthesis ran with the research as context
    assert result.results["synthesize"].ok and result.results["synthesize"].text.strip()
    # B2: token usage was recorded and rolled up
    assert result.usage.get("total_tokens", 0) > 0


async def test_real_detached_workflow_has_durable_result_without_timer(runtime_agent):
    from power_loop import RuntimeClient, create_llm_service_from_env
    from power_loop.agent.runtime_store import now_ms
    agent = await runtime_agent(model=create_llm_service_from_env())
    spec = {'name':'one','root':{'type':'agent','id':'a',
        'spec':{'name':'a','system_prompt':'Reply with the single word: ok.'}}}
    handle = await agent.runtime.client.workflow(agent.target, spec, message_id='detached', actor=agent.actor)
    reader = RuntimeClient(store=agent.runtime.store, transport=agent.runtime.transport,
                           namespace=agent.runtime.namespace)
    try:
        result = await handle.wait(deadline_ms=now_ms()+120000)
        assert result.status == 'completed'
        persisted = await reader.wait(handle.input_ref, actor=agent.actor, deadline_ms=now_ms()+1000)
        assert persisted.status == 'completed' and persisted.structured_output['run_id'] == handle.run_id
        assert not await agent.runtime.store.list_timers(agent.target.session_id)
    finally:
        await reader.aclose()
   # in-process wake, no durable timer
