"""20 · File, shell, skill and background tools with explicit workspace grants."""
import asyncio
import json
import time
from pathlib import Path
from tempfile import TemporaryDirectory

from _runtime import demo_runtime

from power_loop import AgentLoopConfig, JobRef, LLMResponse, LLMService, RuntimeEnv, create_default_tool_registry


class ToolsModel(LLMService):
    def __init__(self):
        self.calls = iter([
            ('write_file', {'path':'notes.txt','content':'alpha\nbeta\n'}),
            ('read_file', {'path':'notes.txt'}),
            ('edit_file', {'path':'notes.txt','old_text':'beta','new_text':'BETA'}),
            ('apply_patch', {'path':'notes.txt','patch':'--- notes.txt\n+++ notes.txt\n@@ -1,2 +1,2 @@\n-alpha\n+ALPHA\n BETA\n'}),
            ('todo', {'items':[{'id':'demo','text':'Verify tools','status':'completed'}]}),
            ('glob', {'pattern':'*.txt'}),
            ('grep', {'pattern':'BETA','path':'.'}),
            ('bash', {'command':'cat notes.txt'}),
            ('load_skill', {'name':'demo'}),
            ('background_run', {'action':'run','command':'printf background-finished'})])
        self.index = 0

    async def complete(self, request, **kwargs):
        call = next(self.calls, None)
        if call is None:
            return LLMResponse(raw_text='Tools finished.')
        self.index += 1
        name, args = call
        return LLMResponse(raw_text='', tool_calls=[{'id':f'tool-{self.index}','type':'function',
            'function':{'name':name,'arguments':json.dumps(args)}}])

    async def close(self):
        pass


async def main():
    with TemporaryDirectory(prefix='power_loop_tools_test_') as directory:
        workspace = Path(directory)
        skills = workspace/'skills'
        (skills/'demo').mkdir(parents=True)
        (skills/'demo'/'SKILL.md').write_text('---\nname: demo\ndescription: example skill\n---\nSay skill-loaded.\n')
        registry = create_default_tool_registry(workspace_dir=workspace, skills_dir=skills)
        async with demo_runtime(ToolsModel(), registry, runtime_env=RuntimeEnv(workspace_dir=workspace, skills_dir=skills),
                config=AgentLoopConfig(max_rounds=12, compactor=None)) as (runtime, loop, target, actor):
            result = await loop.send(target, 'Exercise the configured tools', message_id='tools', actor=actor,
                                     deadline_ms=int(time.time()*1000)+15000)
            assert result.status == 'completed'
            rows = await runtime.store.load_all_messages(target.session_id)
            outputs = [row for row in rows if row.role == 'tool']
            assert len(outputs) == 10 and all(row.name in registry.names() for row in outputs) and not any(row.content.startswith('Error:') for row in outputs), [(row.name,row.content) for row in outputs]
            receipt = json.loads(next(row.content for row in outputs if row.name=='background_run'))
            child = await runtime.client.wait(JobRef(**receipt['job_ref']), actor=actor, deadline_ms=int(time.time()*1000)+10000)
            assert child.status == 'completed' and 'background-finished' in child.final_text
            assert (workspace/'notes.txt').read_text() == 'ALPHA\nBETA\n'
            print([row.name for row in outputs], child.final_text)
            return result.final_text


if __name__ == '__main__':
    asyncio.run(main())
