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
import time

from _runtime import demo_runtime

from power_loop import LLMResponse, LLMService, ToolDefinition, ToolRegistry
from power_loop.runtime.pricing import Free

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
            pricing=Free("local example tool"),
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


class SearchModel(LLMService):
    async def complete(self, request, **kwargs):
        results = [m['content'] for m in request.messages if m.get('role') == 'tool']
        if results:
            return LLMResponse(raw_text=results[-1])
        return LLMResponse(raw_text='', tool_calls=[{'id':'search','type':'function','function':{
            'name':'search_docs','arguments':'{"query":"MIT licensed"}'}}])
    async def close(self):
        pass

async def main():
    async with demo_runtime(SearchModel(), _registry()) as (runtime, loop, target, actor):
        result = await loop.send(target, 'Find the license note', message_id='search', actor=actor,
            deadline_ms=int(time.time()*1000)+10000)
        assert result.status == 'completed' and '[license]' in result.final_text
        print(result.final_text)
        return result.final_text

if __name__ == '__main__':
    asyncio.run(main())
