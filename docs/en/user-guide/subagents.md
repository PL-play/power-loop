# Sub-agents and task ownership

SDK 11.0.0 · task schema 15 · protocol 42 · descriptor schema 1.

[Runtime contract](../../runtime-11.md) · [API](../api/index.md)

Register spawn_agent in the factory's ToolRegistry. A model-visible create returns an accepted child reference without waiting for model execution. The child has its own session and Runtime job; completion is delivered to the parent inbox. Capacity one remains usable because waiting does not hold the only execution slot.

AgentSpec selects name, system_prompt, tool allowlist, model options, round limits and optional output_schema. Validate those values before admission. Tool allowlists and depth limits apply on the child node. Model changes require the host's allowed_models declaration. Structured output is parsed and validated as a child result, not trusted solely because it looks like JSON.

Use the original JobRef to inspect, wait, nudge, cancel or resume. Parent interruption does not silently cancel a healthy child; the result remains queryable. Workflow-node budgets and original owners continue across explicit resumes and do not refresh into a new root. Late child output retained in a finished node's history is not marked consumed until the node actually processes it.

Factories reconstruct tools, workspace and hooks from stable profiles; neither a closure around a parent loop nor an old session handle is a deployment resource.

## Runnable examples

Run from the SDK checkout with its installed environment. These complete programs include resource cleanup; numbered examples share examples/_runtime.py. Host-dependent examples state their required services. For a standalone composition without that helper, use design203_single_process.py.

| Example | Behavior |
|---|---|
| [03_subagent_delegation.py](../../../examples/03_subagent_delegation.py) | Model delegation |
| [06_declarative_subagent.py](../../../examples/06_declarative_subagent.py) | AgentSpec |
| [14_structured_card.py](../../../examples/14_structured_card.py) | Structured results |
| [27_dynamic_workflow.py](../../../examples/27_dynamic_workflow.py) | Workflow ownership |

```bash
.venv/bin/python examples/03_subagent_delegation.py
```
