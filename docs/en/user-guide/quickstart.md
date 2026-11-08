# Quickstart

SDK 11.0.0 · task schema 15 · protocol 42 · descriptor schema 1.

[Runtime contract](../../runtime-11.md) · [API](../api/index.md)

Start a Store, register a factory, then start Runtime.worker before submitting input. The factory resolves the model, tools, hooks and resources on the execution node. The descriptor stores versioned references and serializable configuration; it does not serialize live Python objects or secrets.

The standalone example below uses a fixed model and SQLite. No provider credentials, Redis or broker are needed. For a real provider, replace FixedModel in build_demo. Runtime.aclose owns shutdown. Reuse the TargetRef across turns and choose a new message_id for each new user input; retry the original input with its original ID.

RuntimeClient.send returns Receipt when accepted. RuntimeClient.wait returns the persisted Result. StatefulAgentLoop(runtime=runtime) is a convenience facade that submits and waits. Cancellation of a waiter does not cancel the task.

## Runnable examples

Run from the SDK checkout with its installed environment. These complete programs include resource cleanup; numbered examples share examples/_runtime.py. Host-dependent examples state their required services. For a standalone composition without that helper, use design203_single_process.py.

| Example | Behavior |
|---|---|
| [design203_single_process.py](../../../examples/design203_single_process.py) | Complete local composition |
| [01_multi_turn_chat.py](../../../examples/01_multi_turn_chat.py) | Multi-turn history |
| [02_tool_calling.py](../../../examples/02_tool_calling.py) | Tool calls |
| [08_streaming.py](../../../examples/08_streaming.py) | Streaming output |
| [19_full_chatbot.py](../../../examples/19_full_chatbot.py) | Combined chatbot |

```bash
.venv/bin/python examples/design203_single_process.py
```
