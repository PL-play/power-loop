"""Real-LLM companion for per-send override unit tests (test_per_send_overrides.py).

``send(system_prompt=…)`` overrides the persona for that send only, and ``send(tools=[…])`` restricts
which tools the live model can see/use for that send.
"""

from __future__ import annotations

import pytest

from power_loop import AgentLoopConfig, SessionStore, StatefulAgentLoop
from power_loop.contracts.tools import ToolDefinition
from power_loop.tools.registry import ToolRegistry

from ._llm import make_llm
from .judge import assert_passes


@pytest.mark.asyncio
async def test_per_send_system_prompt_override_changes_behavior() -> None:
    store = await SessionStore.open(":memory:")
    try:
        loop = StatefulAgentLoop(
            llm=make_llm(max_tokens=256, temperature=0),
            store=store,
            config=AgentLoopConfig(system_prompt="You always answer in English.",
                                   max_rounds=1, max_tokens=256, temperature=0, compactor=None),
        )
        sid = await loop.new_session()
        r = await loop.send(
            "Say hello.", session_id=sid,
            system_prompt="You always answer ONLY in French. Never use English.",
        )
        assert r.status == "completed"
        await assert_passes(question="Say hello (per-send override: answer in French).",
                            answer=r.final_text,
                            rubric="The reply is written in French, not English.")
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_per_send_tools_allowlist_restricts_what_model_can_call() -> None:
    # The loop has two tools, but the send allowlists only one. The excluded tool must not be
    # invoked even though the prompt invites it.
    store = await SessionStore.open(":memory:")
    try:
        calls: list[str] = []
        reg = ToolRegistry()
        for name in ("alpha", "beta"):
            reg.register(
                ToolDefinition(name=name, description=f"The {name} tool.",
                               input_schema={"type": "object", "properties": {}}),
                (lambda n: (lambda **kw: (calls.append(n), "ok")[1]))(name),
            )
        loop = StatefulAgentLoop(
            llm=make_llm(max_tokens=384, temperature=0), store=store, tool_registry=reg,
            config=AgentLoopConfig(system_prompt="Use whatever tools are available to comply.",
                                   max_rounds=3, max_tokens=384, temperature=0, compactor=None),
        )
        sid = await loop.new_session()
        # Only `alpha` is exposed for this send.
        r = await loop.send("Please call the beta tool, and also the alpha tool.",
                            session_id=sid, tools=["alpha"])
        assert r.status == "completed"
        assert "beta" not in calls  # the excluded tool was never callable this send
    finally:
        await store.close()
