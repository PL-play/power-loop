# Sub-agents and task ownership

SDK 12.0.0 / schema 17 / protocol 43 / descriptor 2.

Build models, tools, hooks and resources in the Runtime factory. Create a session, then send with a stable message ID. A receipt confirms acceptance; wait on its InputRef for completion. Children, background work and workflows use sessions and ordinary result inputs. Continue with send and stop with cancel. The host supplies trusted identity.

[Runtime 12](../../runtime-12.md)

## Runnable examples

Run from the SDK checkout with its installed environment. These complete programs include resource cleanup; numbered examples share examples/_runtime.py. Host-dependent examples state their required services. For a standalone composition without that helper, use design215_single_process.py.

| Example | Behavior |
|---|---|
| [03_subagent_delegation.py](../../../examples/03_subagent_delegation.py) | Model delegation |
| [06_declarative_subagent.py](../../../examples/06_declarative_subagent.py) | AgentSpec |
| [14_structured_card.py](../../../examples/14_structured_card.py) | Structured results |
| [27_dynamic_workflow.py](../../../examples/27_dynamic_workflow.py) | Workflow ownership |

```bash
.venv/bin/python examples/03_subagent_delegation.py
```
