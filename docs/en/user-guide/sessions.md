# Sessions and input references

SDK 11.0.0 · task schema 15 · protocol 42 · descriptor schema 1.

[Runtime contract](../../runtime-11.md) · [API](../api/index.md)

Runtime.open_session(descriptor, actor=actor) returns TargetRef, which includes store identity. The trusted actor comes from host authentication. Send a new turn to this target, retain its Receipt.input_ref, then wait or inspect by that reference. JobRef identifies an execution generation; it is not a session ID.

RuntimeClient.send is the only input call. On an idle session it starts a job; while a job runs, the message joins that job at its next round, so serial sends never split into separate jobs. `urgent=True` additionally skips the current batch's tool calls that have not started; a running tool or model call is never interrupted. RequestOptions are frozen when a message starts a job. Address a running job with its JobRef to message it, cancel(ref) to stop it (an undelivered input is voided; a delivered one stops the job it joined), and resume on the original execution. RuntimeClient.messages(job_ref) lists a job's latest messages (default 20, each shortened to 500 characters). A closed workflow node retains late descendant output for explicit resume; receipt delivery does not mean model consumption.

Persist the target/reference and reconstruct the same versioned factory on restart. Unknown external results are queried or reported unknown, never replayed automatically. Store history is available through load_all_messages and the public query ports. Closing a facade releases that facade; Runtime.aclose drains workers and closes owned resources. Session deletion and retention are explicit store operations, not a synonym for closing a client.

## Runnable examples

Run from the SDK checkout with its installed environment. These complete programs include resource cleanup; numbered examples share examples/_runtime.py. Host-dependent examples state their required services. For a standalone composition without that helper, use design203_single_process.py.

| Example | Behavior |
|---|---|
| [01_multi_turn_chat.py](../../../examples/01_multi_turn_chat.py) | History |
| [11_cross_process_resume.py](../../../examples/11_cross_process_resume.py) | Reopen |
| [22_follow_up_steering.py](../../../examples/22_follow_up_steering.py) | Message a running job |
| [23_per_send_overrides.py](../../../examples/23_per_send_overrides.py) | Frozen options |
| [12_retry_and_cancel.py](../../../examples/12_retry_and_cancel.py) | Retry and cancel |

```bash
.venv/bin/python examples/01_multi_turn_chat.py
```
