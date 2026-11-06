"""Public workflow submission uses durable client references."""

from .runner import submit
from .spec import WorkflowSpec


class Workflow:
    def __init__(self, spec, *, client, target, actor, message_id, descriptor=None, options=None):
        self.spec = WorkflowSpec.from_json(spec.to_dict() if isinstance(spec, WorkflowSpec) else spec)
        self.client, self.target, self.actor = client, target, actor
        self.message_id, self.descriptor, self.options = message_id, descriptor, options

    async def start(self):
        return await submit(
            self.client,
            self.target,
            self.spec,
            message_id=self.message_id,
            actor=self.actor,
            descriptor=self.descriptor,
            options=self.options,
        )

    async def run(self, *, deadline_ms=None):
        return await (await self.start()).wait(deadline_ms=deadline_ms)


def create_workflow(spec_json, **kwargs):
    return Workflow(spec_json, **kwargs)
