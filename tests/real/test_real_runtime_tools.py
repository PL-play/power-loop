from __future__ import annotations

import pytest

from power_loop import (
    AgentLoopConfig,
    RuntimeProjector,
    SessionStore,
    ToolDefinition,
    ToolRegistry,
    create_default_tool_registry,
    get_tool_runtime_context,
)
from power_loop.runtime.pricing import Free

from ._llm import make_llm


class _RealCustomProjector(RuntimeProjector):
    async def project(self, *, store, session_id, round_index, context):
        state = await store.get_runtime_state(session_id, "real_custom", default={}) or {}
        marker = state.get("marker")
        if not marker:
            return []
        return [{"role": "user", "name": "real_custom_state", "content": f"<real_custom>{marker}</real_custom>"}]




@pytest.mark.asyncio
async def test_real_llm_todo_tool_persists_to_sqlite(tmp_path, runtime_agent) -> None:
    store = await SessionStore.open(tmp_path / "todo.db")
    try:
        loop = await runtime_agent(model=make_llm(max_tokens=512, temperature=0.0), store=store, registry=create_default_tool_registry(include=["todo"]), config=AgentLoopConfig(
                system_prompt=(
                    "You must use the todo tool when the user asks you to create a todo. "
                    "Create exactly one todo item with id 'real' and text 'REAL_LLM_SQLITE_TODO'. "
                    "After the tool result is returned, answer briefly."
                ),
                max_rounds=4,
                context_budget_tokens=10**9,
            ))
        sid = loop.target.session_id
        result = await loop.send("Create the todo now using the todo tool.")
        todo_state = await store.get_runtime_state(sid, "todo", default={}) or {}
        rendered = str(todo_state.get("rendered", ""))
        assert "REAL_LLM_SQLITE_TODO" in rendered, result.final_text
    finally:
        await loop.aclose()
        await store.close()


@pytest.mark.asyncio
async def test_real_llm_custom_tool_uses_public_runtime_context(tmp_path, runtime_agent) -> None:
    store = await SessionStore.open(tmp_path / "custom-tool.db")
    try:
        registry = ToolRegistry()

        async def save_runtime_marker(marker: str) -> str:
            ctx = get_tool_runtime_context(required=True)
            await ctx.store.set_runtime_state(ctx.session_id, "real_custom", {"marker": marker})
            return f"saved marker {marker}"

        registry.register(
            ToolDefinition(
                name="save_runtime_marker",
                description="Persist a runtime marker in the current session.",
                input_schema={
                    "type": "object",
                    "properties": {"marker": {"type": "string"}},
                    "required": ["marker"],
                },
                required_params=("marker",),
            pricing=Free("test_fixture")),
            save_runtime_marker,
        )
        loop = await runtime_agent(model=make_llm(max_tokens=512, temperature=0.0), store=store, registry=registry, config=AgentLoopConfig(
                system_prompt=(
                    "You must call save_runtime_marker with marker='REAL_CUSTOM_RUNTIME_42'. "
                    "After the tool result is returned, answer briefly."
                ),
                max_rounds=4,
                context_budget_tokens=10**9,
                runtime_projectors=(_RealCustomProjector(),),
            ))
        sid = loop.target.session_id
        result = await loop.send("Save the runtime marker using the tool now.")

        state = await store.get_runtime_state(sid, "real_custom", default={}) or {}
        assert state.get("marker") == "REAL_CUSTOM_RUNTIME_42", result.final_text
    finally:
        await loop.aclose()
        await store.close()


@pytest.mark.asyncio
async def test_real_llm_sees_custom_projector_after_tool_write(tmp_path, runtime_agent) -> None:
    store = await SessionStore.open(tmp_path / "custom-projector.db")
    try:
        registry = ToolRegistry()

        async def save_runtime_marker(marker: str) -> str:
            ctx = get_tool_runtime_context(required=True)
            await ctx.store.set_runtime_state(ctx.session_id, "real_custom", {"marker": marker})
            return f"saved marker {marker}"

        registry.register(
            ToolDefinition(
                name="save_runtime_marker",
                description="Persist a runtime marker in the current session.",
                input_schema={
                    "type": "object",
                    "properties": {"marker": {"type": "string"}},
                    "required": ["marker"],
                },
                required_params=("marker",),
            pricing=Free("test_fixture")),
            save_runtime_marker,
        )
        loop = await runtime_agent(model=make_llm(max_tokens=512, temperature=0.0), store=store, registry=registry, config=AgentLoopConfig(
                system_prompt=(
                    "First call save_runtime_marker with marker='REAL_PROJECTOR_VISIBLE_42'. "
                    "On the next round, if you see REAL_PROJECTOR_VISIBLE_42 in runtime state, "
                    "reply with exactly: CUSTOM_PROJECTOR_VISIBLE"
                ),
                max_rounds=4,
                context_budget_tokens=10**9,
                runtime_projectors=(_RealCustomProjector(),),
            ))
        result = await loop.send("Run the custom runtime flow.")

        if "CUSTOM_PROJECTOR_VISIBLE" not in result.final_text:
            assert "REAL_PROJECTOR_VISIBLE_42" in result.final_text
    finally:
        await loop.aclose()
        await store.close()
