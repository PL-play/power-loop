# Extending tools

SDK 11.0.0 · task schema 15 · protocol 42 · descriptor schema 1.

[Runtime contract](../../runtime-11.md) · [API](../api/index.md)

Define ToolDefinition with a JSON input schema, register a synchronous or asynchronous handler in ToolRegistry, and return that registry from the execution factory. Tool names in RequestOptions.tools are an allowlist; missing required tools fail explicitly. Handlers receive validated arguments; keep error output useful without inventing successful results.

Ordinary stateless tools can perform local computations. Tools that own external work, fees or managed resources need an explicit ToolRuntimeContract and the relevant host backend. Do not capture an old loop or use its private methods. Resolve execution identity through trusted runtime context, not model-supplied tenant/session values.

For MCP, install the mcp extra, configure the actual transport, register discovered definitions and close the connection with execution resources. The included MCP example starts a local stdio server and exercises actual calls. An unavailable dependency is an error, not a silent fallback.

Host business permissions and budget checks remain at their authoritative API. An unknown external action is queried by original identity or returned unknown; it is never automatically replayed just because a local request failed.

## Runnable examples

Run from the SDK checkout with its installed environment. These complete programs include resource cleanup; numbered examples share examples/_runtime.py. Host-dependent examples state their required services. For a standalone composition without that helper, use design203_single_process.py.

| Example | Behavior |
|---|---|
| [02_tool_calling.py](../../../examples/02_tool_calling.py) | Definition and handler |
| [37_custom_retrieval_tool.py](../../../examples/37_custom_retrieval_tool.py) | Retrieval |
| [38_mcp_tools.py](../../../examples/38_mcp_tools.py) | Local MCP |
| [41_custom_async_tool.py](../../../examples/41_custom_async_tool.py) | Async handler |
| [42_build_your_own_tools.py](../../../examples/42_build_your_own_tools.py) | Managed tool contracts |

```bash
.venv/bin/python examples/02_tool_calling.py
```
