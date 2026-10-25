"""The SDK shell's model-facing schema; independent of its registered name."""
import json
from dataclasses import dataclass

from power_loop.runtime.tool_contracts import RuntimeTool, ToolContextPresentation, ToolRuntimeContract


@dataclass(frozen=True)
class ShellTool(RuntimeTool):
    handler: object

    @property
    def runtime_contract(self):
        return ToolRuntimeContract("sdk.shell")

    def __call__(self, **arguments):
        # Keep the built-in shell's synchronous registry API.
        return self.handler(**arguments)

    def context_spec(self, spec):
        fn = spec['function']
        params = fn.setdefault('parameters', {})
        props = params.setdefault('properties', {})
        props.setdefault('description', {'type': 'string', 'minLength': 1, 'maxLength': 1000,
            'description': 'One sentence purpose; preserved in archives.'})
        params['required'] = list(dict.fromkeys([*params.get('required', []), 'description']))
        fn['description'] = fn.get('description', '').replace('Output is truncated, ', '') + (
            '\nContext policy: description is required. Default retire; keep only for persistent '
            'instructions/skills, not logs, tests or listings. keep holds the output unchanged for the '
            'rest of this task; later tasks see a short record. Recall originals when details are needed.')
        return spec


    def context_presentation(self, source):
        from power_loop.runtime.tool_truncation import bash_exit
        code = bash_exit(source.result).removeprefix('exit_code=')
        description = str(source.arguments.get('description') or '')
        return ToolContextPresentation(
            keep_in_send=source.arguments.get('context_policy') == 'keep',
            archive_arguments=json.dumps({'description': description}, ensure_ascii=False) if description else '{}',
            archive_label=f'{description}（exit {code}）' if code else description or '(shell)',
        )

    def truncate_context(self, source, policy, *, raw_arguments):
        from power_loop.runtime.tool_truncation import truncate_bash
        return truncate_bash(dict(source.arguments), source.result, policy, source.recall)
