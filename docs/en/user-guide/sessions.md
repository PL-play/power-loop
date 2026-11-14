# Sessions and input references

SDK 12.0.0 / schema 17 / protocol 43 / descriptor 2.

Build models, tools, hooks and resources in the Runtime factory. Create a session, then send with a stable message ID. A receipt confirms acceptance; wait on its InputRef for completion. Children, background work and workflows use sessions and ordinary result inputs. Continue with send and stop with cancel. The host supplies trusted identity.

[Runtime 12](../../runtime-12.md)

## Runnable examples

Run from the SDK checkout with its installed environment. These complete programs include resource cleanup; numbered examples share examples/_runtime.py. Host-dependent examples state their required services. For a standalone composition without that helper, use design215_single_process.py.

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
