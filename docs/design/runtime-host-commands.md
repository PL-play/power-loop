# Runtime host commands (design 203 P5a)

This describes the unpublished unified runtime branch. It is not a package release or deployment instruction.

A trusted `RuntimeFactory.build(descriptor, context)` can return an `ExecutionEnvironment` with `command=(tool_name, arguments)`. The factory resolves the command from a frozen host descriptor; it must not treat arbitrary user text as a command grant. The named tool must be registered, included in the input's allowed tools, and explicitly priced `Free`. The worker invokes it in the ordinary input/job execution context, including cancellation, start/end hooks, original root attribution and normal model/tool charging for inner calls. It does not invent an assistant `tool_call` to make an HTTP request executable.

This supports host operations such as model probes, structured questions and platform review. Model calls inside the command use `ModelCalls`; paid supplier calls retain their actual receipts. An inline child uses the standard child runtime and releases the parent's execution capacity while waiting. The consumed child result is recorded by its original `result_id`; its parent inbox does not create another normal model turn.

`RuntimeClient.resume(ref, message_id=..., actor=..., body=..., options=...)` explicitly continues an original task generation. Optional `options` can only narrow `tools`, `max_rounds`, `max_tokens` and `max_tokens_per_run`. It cannot expand the saved grant or replace the frozen definition. Completed or failed subagents may be explicitly continued; prior generation result lookups keep returning the original result. A continuation is new requested work, not a replay of an unknown external action.

Host root provenance now follows the original create request's `InputRef` and `input:<message_id>` key. Supervision reminders use `RuntimeInputStore` and copy that original input's descriptor/options. Hosts notify this persisted input; they must not separately create another parent input for the same alert or result.

Verification: `tests/unit/test_host_commands_203.py`, `test_runtime_root_context_203.py`, the affected inspection/messaging cases and the three-database cost profiles. Host command tests cover capacity one, actual inner model charging, child consumption, explicit repair continuation, and rejection of expanded tools or limits.

A trusted ingress can call `subscribe_output(InputRef(...), actor=host_actor, before_send=True)` before submitting that exact request. The SDK first authorizes the target session; tool actors and foreign tenants cannot pre-subscribe. Consume only after send admission, and always close the subscription. Each output retains its original InputRef/job generation and sequence; `gap` explicitly means intermediate text is not replayed.
