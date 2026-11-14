# Workflow sessions

SDK 12.0.0 / schema 17 / protocol 43 / descriptor 2.

A workflow driver is a direct independent child session of its caller. Each leaf executes in an agent session. The driver journal stores graph progress, node TargetRef/InputRef, iteration and attempt coordinates. Results arrive as ordinary driver inputs; an unfinished node continues with ordinary send. Graph definition, sequence/parallel/branch/loop interpretation and authorization narrowing remain in the existing workflow modules.

`runtime.client.workflow(target, spec, message_id=..., actor=...)` returns a handle with `target` and `input_ref`. Wait for that original input, or inspect messages/children through the current client. Model-visible workflow tools use the same admission path. Admin queries caller children with session_kind='workflow_driver' and reads workflow:run:* / workflow:node:* from their session_runtime_state.

Unknown external work is queried or reported unknown. A replacement driver reads persisted node progress; it does not replay arbitrary external actions. The graph budget remains a dispatch constraint, and host billing uses design 214 events.

See [Runtime 12](../../runtime-12.md), [workflow YAML](../../workflow-yaml.md), and [the runnable workflow example](../../../examples/27_dynamic_workflow.py).
