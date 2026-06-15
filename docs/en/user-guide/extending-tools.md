# Extending power-loop with your own tools

[中文](../../zh/user-guide/extending-tools.md) | [User Guide](../index.md)

power-loop is a **kernel**, not a batteries-included platform: it ships no connectors, no
vector stores, no SaaS integrations. That is deliberate — those carry dependencies,
opinions, and churn that don't belong in a ~5k-line core. Instead you bring your own
tools through the `ToolRegistry`, and reach external systems either by writing a thin tool
or via [MCP](#external-systems-mcp). This page is the recipe.

## The recipe: definition + handler + register

A tool is two things bound by one name — a `ToolDefinition` (the JSON-Schema the model
sees) and a handler (your code):

```python
from power_loop import ToolDefinition, ToolRegistry

def search_docs(query: str, top_k: int = 1) -> str:
    ...                      # your domain logic — DB, HTTP, vector store, anything
    return "results…"

registry = ToolRegistry()
registry.register(
    ToolDefinition(
        name="search_docs",
        description="Search the knowledge base; returns the most relevant note(s).",
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}, "top_k": {"type": "integer"}},
            "required": ["query"],
        },
        required_params=("query",),    # missing-arg validation before the handler runs
    ),
    search_docs,
)

loop = StatefulAgentLoop(llm=llm, tool_registry=registry)
```

Full runnable example: [`examples/37_custom_retrieval_tool.py`](../../../examples/37_custom_retrieval_tool.py).

## Sync vs async handlers

- A **sync** handler is run on a worker thread (`asyncio.to_thread`) by `invoke_async`, so
  a blocking call (DB, subprocess, HTTP) never stalls the event loop. contextvars (runtime
  env, session identity) propagate into the thread.
- An **`async def`** handler runs on the loop — use it for already-async I/O (`httpx`,
  async DB drivers).

The model only ever sees the JSON Schema; the return value (a string) becomes the tool
message. Validation (`required_params`) runs before your handler, so you can assume the
declared inputs are present.

## Per-call allowlisting

Restrict which tools a given run may use without building a new registry:

```python
await loop.send("…", session_id=sid, tools=["search_docs"])   # the model sees only these
```

`tools=` accepts a name list (allowlisted from the loop registry via `registry.subset`) or
a `ToolRegistry` to use directly.

## External systems: MCP

For tools that live in another process or service, the connector path is the
**[Model Context Protocol](https://modelcontextprotocol.io)** rather than a bundled
integration. `power_loop.contrib.mcp` surfaces an MCP server's tools as power-loop tools:

```python
from power_loop.contrib.mcp import StdioMCPClient, register_mcp_tools   # power-loop[mcp]

client = await StdioMCPClient("npx", ["-y", "@modelcontextprotocol/server-filesystem", "/data"]).connect()
await register_mcp_tools(registry, client, prefix="fs.")   # MCP tools → ToolDefinitions
...
await client.aclose()
```

The integration point is the small `MCPToolSource` Protocol (`list_tools` / `call_tool`),
so the `mcp` SDK is optional — implement the Protocol over any client (HTTP/SSE, a fake in
tests). `register_mcp_tools` maps each tool's `inputSchema` straight to a `ToolDefinition`.

## Why not bundle connectors?

A connector you didn't write is a dependency you didn't choose, a schema you can't change,
and a breakage you can't fix. The kernel gives you the *seam* (`ToolRegistry`) and a
standard remote path (MCP); the 30-second recipe above is usually less code than wiring a
generic connector — and it's yours.
