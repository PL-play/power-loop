"""SDK background implementation's presentation and original-result reader.

The loop consumes RuntimeTool methods. Aliases and host replacements do not
acquire these behaviors merely by sharing a built-in tool's name.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from power_loop.runtime.tool_contracts import RuntimeTool, ToolRuntimeContract, result_container


@dataclass(frozen=True)
class BackgroundTool(RuntimeTool):
    handler: object

    @property
    def runtime_contract(self):
        return ToolRuntimeContract(
            "sdk.background", capabilities=("tool.background-submit.v1",),
            background_allowed=False, result_container=True,
        )

    async def __call__(self, **arguments):
        return await self.handler(**arguments)

    def context_spec(self, spec):
        fn = spec['function']
        params = fn.setdefault('parameters', {})
        props = params.setdefault('properties', {})
        action = props.get('action', {})
        if 'enum' in action:
            action['enum'] = [x for x in action['enum'] if x != 'run']
        props.pop('command', None)
        fn['description'] = fn.get('description', '') + (
            '\nStart registered tools with action=tool. '
            'For shell commands use a registered shell tool and its required arguments. '
            'Do not use action=run.')
        return spec

    def background_hint(self, definition, *, scheduler_name):
        return (
            f'{scheduler_name}(action="tool", tool="{definition.name}", '
            'args={…}) 立即返回 task_id 不阻塞；结果完成后自动送到你面前，不用 check。'
            '不需要立刻用结果时优先异步。'
        )

    async def result_sources(self, source, *, store, registry):
        from power_loop.runtime.background import RESULT_PREFIX, load_record
        from power_loop.runtime.tool_context import ToolSource

        if RESULT_PREFIX not in source.result:
            return []
        try:
            payload = json.loads(source.result.split(RESULT_PREFIX, 1)[1])
            sources = []
            for task in payload['tasks']:
                record = await load_record(store, source.session_id, str(task['task_id']))
                if not record or record.get('status') not in ('completed', 'failed', 'done'):
                    return []
                name, args = record.get('tool'), record.get('args')
                registered = registry.get(name) if registry is not None else None
                if (not name or not isinstance(args, dict) or registered is None
                        or result_container(registered)):
                    return []
                sources.append(ToolSource(source.session_id, source.send_index, source.seq,
                    str(task['task_id']), name, args, str(record.get('result') or '')))
            return sources
        except (ValueError, KeyError, TypeError):
            return []
