from __future__ import annotations

import pytest

from power_loop import AgentLoopConfig, SessionStore, StatefulAgentLoop, create_default_tool_registry

from ._llm import make_llm


@pytest.mark.asyncio
async def test_real_llm_sees_sqlite_runtime_state(tmp_path) -> None:
    store = SessionStore.open(tmp_path / "runtime.db")
    try:
        sid = store.create_session()
        store.set_runtime_state(
            sid,
            "todo",
            {
                "items": [{"id": "rt", "text": "RUNTIME_SQLITE_TODO_42", "status": "in_progress"}],
                "rendered": "[>] #rt: RUNTIME_SQLITE_TODO_42\n\n(0/1 completed)",
                "counts": {"total": 1, "completed": 0},
            },
        )
        store.upsert_background_task(
            sid,
            task_id="bg-real",
            command="printf BG_RUNTIME_DONE_42",
            status="completed",
            return_code=0,
            output_tail="BG_RUNTIME_DONE_42",
        )
        loop = StatefulAgentLoop(
            llm=make_llm(max_tokens=256, temperature=0.0),
            store=store,
            config=AgentLoopConfig(
                system_prompt=(
                    "You are a runtime-state verifier. If runtime messages contain "
                    "RUNTIME_SQLITE_TODO_42 and BG_RUNTIME_DONE_42, reply with exactly: "
                    "RUNTIME_STATE_VISIBLE"
                ),
                max_rounds=1,
                compactor=None,
            ),
        )
        result = await loop.send(
            "Inspect the runtime messages provided to you and follow the system instruction.",
            session_id=sid,
        )
        if "RUNTIME_STATE_VISIBLE" not in result.final_text:
            assert "RUNTIME_SQLITE_TODO_42" in result.final_text
            assert "BG_RUNTIME_DONE_42" in result.final_text
    finally:
        store.close()


@pytest.mark.asyncio
async def test_real_llm_todo_tool_persists_to_sqlite(tmp_path) -> None:
    store = SessionStore.open(tmp_path / "todo.db")
    try:
        loop = StatefulAgentLoop(
            llm=make_llm(max_tokens=512, temperature=0.0),
            store=store,
            tool_registry=create_default_tool_registry(include=["todo"]),
            config=AgentLoopConfig(
                system_prompt=(
                    "You must use the todo tool when the user asks you to create a todo. "
                    "Create exactly one todo item with id 'real' and text 'REAL_LLM_SQLITE_TODO'. "
                    "After the tool result is returned, answer briefly."
                ),
                max_rounds=4,
                compactor=None,
            ),
        )
        sid = loop.new_session()
        result = await loop.send("Create the todo now using the todo tool.", session_id=sid)
        todo_state = store.get_runtime_state(sid, "todo", default={}) or {}
        rendered = str(todo_state.get("rendered", ""))
        assert "REAL_LLM_SQLITE_TODO" in rendered, result.final_text
    finally:
        store.close()
