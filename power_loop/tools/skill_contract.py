"""Presentation of the SDK skill reader, independent of its registered name."""
from dataclasses import dataclass

from power_loop.runtime.tool_contracts import RuntimeTool, ToolContextPresentation, ToolRuntimeContract


@dataclass(frozen=True)
class SkillTool(RuntimeTool):
    handler: object

    @property
    def runtime_contract(self):
        return ToolRuntimeContract('sdk.skill-reader')

    def __call__(self, **arguments):
        return self.handler(**arguments)

    def context_presentation(self, source):
        label = str(source.arguments.get('name') or source.arguments.get('skill') or source.arguments.get('path') or '')
        if source.arguments.get('doc'):
            label += '/' + str(source.arguments['doc'])
        return ToolContextPresentation(archive_label=label or None)
