"""37 · 自定义检索工具 / Custom retrieval tool (ECO-6)

What you learn / 你将学到
--------------------------
- power-loop **不**捆绑连接器/向量库——内核理念是:你用 `ToolRegistry` 注册自己的工具(领域逻辑
  在你手里)。这就是「扩展」的全部:`ToolDefinition`(JSON Schema)+ 一个 handler + `register()`。
  / power-loop bundles NO connectors/vector stores — you register your own tools via the
  ToolRegistry. Extending = ToolDefinition (JSON Schema) + a handler + register().
- 这里用一个**进程内、确定性**的朴素检索(词项重叠打分)当工具,Agent 调它来据库作答。换成真实
  向量库只需在 handler 里替换实现。外部服务(MCP server)走 `contrib.mcp`(见扩展工具文档)。
  / An in-process deterministic retrieval tool; swap the handler body for a real vector DB.
  External services (MCP servers) go through ``contrib.mcp``.

Run / 运行
----------
    python examples/37_custom_retrieval_tool.py
"""

from __future__ import annotations

import asyncio

from _helpers import make_llm

from power_loop import AgentLoopConfig, StatefulAgentLoop, ToolDefinition, ToolRegistry

# A tiny in-memory "knowledge base". In a real app this is your vector store / DB.
_DOCS = {
    "founding": "power-loop's first commit was on 2026-03-23.",
    "deps": "The power-loop core has zero runtime dependencies; transports are extras.",
    "license": "power-loop is MIT licensed.",
}


def _search_docs(query: str, top_k: int = 1) -> str:
    """Naive deterministic retrieval: score docs by query-term overlap."""
    terms = {w.lower().strip(".,?") for w in query.split()}
    scored = sorted(
        _DOCS.items(),
        key=lambda kv: len(terms & {w.lower().strip(".,?") for w in kv[1].split()}),
        reverse=True,
    )
    return "\n".join(f"[{k}] {v}" for k, v in scored[:top_k])


def _registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(
        ToolDefinition(
            name="search_docs",
            description="Search the project knowledge base; returns the most relevant note(s).",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "what to look up"},
                    "top_k": {"type": "integer", "description": "how many notes", "default": 1},
                },
                "required": ["query"],
            },
            required_params=("query",),
        ),
        _search_docs,  # a plain sync function — the registry runs it off the loop
    )
    return reg


async def main() -> str:
    loop = StatefulAgentLoop(
        llm=make_llm(max_tokens=128),
        tool_registry=_registry(),
        config=AgentLoopConfig(
            system_prompt=(
                "Answer ONLY from the knowledge base. Call search_docs first, then answer "
                "concisely in English."
            ),
            max_rounds=4,
        ),
    )
    sid = await loop.new_session()
    result = await loop.send("How many runtime dependencies does the power-loop core have?",
                             session_id=sid, tools=["search_docs"])
    print(result.final_text)
    return result.final_text or ""


if __name__ == "__main__":
    asyncio.run(main())
