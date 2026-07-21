"""Real-LLM end-to-end test for power_loop.workflow.

Exercises the full middle-tier path against the configured provider:
WorkflowSpec validation → engine → output_schema enforced as a real
response_format → foreach fan-out → inputs_from → usage recorded (B2).

Skipped automatically when no POWER_LOOP_* / OPENAI_COMPAT_* provider env is set
(or with --no-real). See tests/conftest.py.
"""

from __future__ import annotations

import tempfile

import pytest

from power_loop import AgentLoopConfig, StatefulAgentLoop, create_llm_service_from_env
from power_loop.workflow import create_workflow

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


async def test_real_workflow_sequence_foreach_structured():
    loop = StatefulAgentLoop(
        llm=create_llm_service_from_env(),
        db_path=tempfile.mktemp(suffix=".db"),
        config=AgentLoopConfig(system_prompt="orchestrator", max_rounds=4),
    )
    result = await create_workflow(SPEC, parent_loop=loop).run()

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


async def test_real_detached_on_complete_seam():
    """3.21 host wake seam, end-to-end against a real provider: a detached run fires
    ``on_complete`` with a ``completed`` WorkflowCompletion and arms NO durable timer."""
    from power_loop.workflow import WorkflowCompletion

    loop = StatefulAgentLoop(
        llm=create_llm_service_from_env(),
        db_path=tempfile.mktemp(suffix=".db"),
        config=AgentLoopConfig(system_prompt="orchestrator", max_rounds=3),
    )
    psid = await loop.new_session()
    seen: list[WorkflowCompletion] = []

    async def on_complete(c: WorkflowCompletion) -> None:
        seen.append(c)

    spec = {"name": "one", "root": {"type": "agent", "id": "a",
            "spec": {"name": "a", "system_prompt": "Reply with the single word: ok."}}}
    wf = create_workflow(spec, parent_loop=loop, parent_session_id=psid)
    handle = await wf.start(detached=True, on_complete=on_complete)
    await handle.task

    assert len(seen) == 1
    assert seen[0].status == "completed"
    assert seen[0].run_id == handle.run_id
    assert not await loop.list_timers(psid)   # in-process wake, no durable timer
