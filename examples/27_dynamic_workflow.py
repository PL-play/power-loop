"""Declarative workflow through the common runtime, with an offline example model.

The driver and leaves use independent sessions. This example uses the default
single-process adapters; shared workers use host-supplied shared adapters.
"""
import asyncio
import json
from uuid import uuid4

from power_loop import (
    AgentLoopConfig,
    ExecutionEnvironment,
    LLMResponse,
    LLMService,
    RegistryRuntimeFactory,
    Runtime,
    RuntimeDescriptor,
    SessionStore,
    TrustedActor,
)

SPEC = {
    "name": "research_and_summarize",
    "input": "the Japanese tea ceremony",
    "budget": {"max_tokens": 200_000, "stop_at_remaining_pct": 5},
    "root": {
        "type": "sequence",
        "steps": [
            {
                "type": "agent",
                "id": "plan",
                "spec": {
                    "name": "planner",
                    "system_prompt": (
                        "Break the topic into exactly 3 short subtopics. "
                        'Respond ONLY with JSON: {"subtopics": ["...", "...", "..."]}.'
                    ),
                },
                "input": "Topic: {{input}}",
                # Structured output so downstream foreach can read plan.subtopics.
                "output_schema": {
                    "name": "Plan",
                    "schema": {
                        "type": "object",
                        "required": ["subtopics"],
                        "properties": {
                            "subtopics": {"type": "array", "items": {"type": "string"}}
                        },
                    },
                },
            },
            {
                "type": "foreach",
                "id": "research",
                "items_from": "plan.subtopics",
                "as": "subtopic",
                "parallel": True,
                "max_concurrency": 3,
                "body": {
                    "type": "agent",
                    "id": "researcher",
                    "spec": {
                        "name": "researcher",
                        "system_prompt": "Write 2-3 factual sentences on the given subtopic.",
                    },
                    "input": "Subtopic: {{subtopic}}",
                },
            },
            {
                "type": "agent",
                "id": "synthesize",
                "spec": {
                    "name": "writer",
                    "system_prompt": "Synthesize the research notes into one tight paragraph.",
                },
                "inputs_from": ["research"],
            },
        ],
    },
}

class WorkflowModel(LLMService):
    async def complete(self, request, **kwargs):
        if 'Break the topic' in request.system_prompt:
            return LLMResponse(raw_text=json.dumps({'subtopics':['setting', 'utensils', 'sequence']}))
        return LLMResponse(raw_text='Prepared tea, shared attentively.')

    async def close(self):
        pass

async def main():
    factory = RegistryRuntimeFactory()
    async def build(descriptor, context):
        return ExecutionEnvironment(WorkflowModel(), None, AgentLoopConfig(compactor=None))
    factory.register("example.workflow@1", build)
    store = await SessionStore.open(":memory:")
    runtime = Runtime(store=store, factory=factory, node_id="example", capacity=1)
    actor = TrustedActor("example", "host", None)
    descriptor = RuntimeDescriptor(1, "agent", "example.workflow@1", None,
        "example.tools@1", "example.hooks@1", "example.resources@1", {}, {})
    try:
        await runtime.worker.start()
        target = await runtime.open_session(descriptor, actor=actor)
        handle = await runtime.client.workflow(target, SPEC, message_id=uuid4().hex, actor=actor)
        print("Accepted", handle.job_ref)
        result = await handle.wait()
        assert result.status == 'completed', result
        assert 'synthesize' in result.results
        print(result.summary())
    finally:
        await runtime.aclose()


if __name__ == "__main__":
    asyncio.run(main())
