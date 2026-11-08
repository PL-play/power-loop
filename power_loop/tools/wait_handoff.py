"""Request finite child review; only the original main decides its continuation."""
from power_loop.agent.runtime_types import JobRef, RuntimeRequestError, canonical
from power_loop.contracts.tools import ToolDefinition
from power_loop.core.agent_context import get_current_loop
from power_loop.core.execution_context import get_execution_context
from power_loop.runtime import wait_review
from power_loop.runtime.pricing import Free
from power_loop.runtime.timer_backend import timer_context

WAIT_HANDOFF_DEFINITION = ToolDefinition(name='wait_handoff', pricing=Free('local'),
    description='Finite children request a timed review using request with due_at_ms, note and continuation. '
    'Only the original main may inspect/decide using the original job_id, generation and event_id. '
    'continue answers the original waiting tool call; defer requires next_review_at_ms before the original '
    'deadline; cancel stops the original task. Never create replacement work. No decision is a completed result.',
    input_schema={'type':'object','properties':{
        'action':{'type':'string','enum':['request','inspect','decide']},
        'due_at_ms':{'type':'integer'},'note':{'type':'string','maxLength':4000},
        'continuation':{'type':'string','maxLength':64000},'job_id':{'type':'string'},
        'generation':{'type':'integer','minimum':1},'event_id':{'type':'string'},
        'decision':{'type':'string','enum':['continue','defer','cancel']},
        'next_review_at_ms':{'type':'integer'},'field':{'type':'string','enum':['summary','note','continuation']},
        'char_offset':{'type':'integer','minimum':0},'max_chars':{'type':'integer','minimum':256,'maximum':16000}},
        'required':['action'],'additionalProperties':False},required_params=('action',))


def register_wait_handoff_tool(registry, *, policy=None, overwrite=False):
    from power_loop.runtime.tool_contracts import FunctionRuntimeTool, ToolDispatchPolicy, ToolRuntimeContract
    async def run(action, **options):
        loop=get_current_loop()
        try:
            if action=='request':
                if set(options)!={'due_at_ms','note','continuation'}:
                    raise RuntimeRequestError('invalid_options')
                selected=policy(get_execution_context()) if callable(policy) else policy
                return await wait_review.request(loop,policy=selected,**options)
            ctx=await timer_context()  # Original persisted main tool call supplies authority.
            if ctx.actor.session_id is None:
                raise RuntimeRequestError('forbidden')
            ref=JobRef(ctx.store_id,options.pop('job_id'),options.pop('generation'))
            expected=options.pop('event_id')
            worker=loop._runtime_worker
            if action=='inspect':
                return canonical(await wait_review.inspect(worker.runtime.client,ref,ctx.actor,expected,**options))
            if action=='decide':
                return canonical(await wait_review.decide(worker,ref,ctx.actor,expected,**options))
            raise RuntimeRequestError('invalid_options')
        except RuntimeRequestError as exc:
            return canonical({'error':{'code':exc.code}})
        except (KeyError,TypeError):
            return canonical({'error':{'code':'invalid_options'}})
    registry.register(WAIT_HANDOFF_DEFINITION,FunctionRuntimeTool(run,
        ToolRuntimeContract('sdk.wait-review',allowed_roles=('main','subagent','workflow_node'),
            capabilities=('task.wait-review.v1',)),
        policy=lambda arguments:ToolDispatchPolicy(native_admitted=True,control_only=arguments.get('action')=='inspect')),
        overwrite=overwrite)
