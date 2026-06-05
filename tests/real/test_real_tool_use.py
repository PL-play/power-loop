"""Real-LLM end-to-end check that tool calling round-trips through the
StatefulAgentLoop / sink / store pipeline.

The model is given a single ``lookup`` tool whose canned data is the only
way to answer the question correctly. If the loop fails to route through
the tool the judge rejects the answer.
"""

from __future__ import annotations

import pytest

from power_loop import AgentLoopConfig, SessionStore, StatefulAgentLoop
from power_loop.contracts.tools import ToolDefinition
from power_loop.tools.registry import ToolRegistry

from ._llm import make_llm
from .judge import assert_passes


def _build_registry(call_log: list[str]) -> ToolRegistry:
    reg = ToolRegistry()

    def lookup(**kwargs):
        city = str(kwargs.get("city") or "").strip().lower()
        call_log.append(city)
        return {
            "tokyo": "Tokyo's signature dish is sushi (in particular Edomae nigiri).",
            "rome": "Rome's signature dish is cacio e pepe.",
            "lima": "Lima's signature dish is ceviche.",
        }.get(city, f"No record for {city}")

    reg.register(
        ToolDefinition(
            name="lookup",
            description="Return the signature local dish for a given city.",
            input_schema={
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
            required_params=("city",),
        ),
        lookup,
    )
    return reg


@pytest.mark.asyncio
async def test_tool_round_routes_through_registry_and_judges_answer() -> None:
    call_log: list[str] = []
    store = SessionStore.open(":memory:")
    try:
        loop = StatefulAgentLoop(
            llm=make_llm(max_tokens=512, temperature=0),
            store=store,
            config=AgentLoopConfig(
                system_prompt=(
                    "You answer questions about local cuisine. Use the "
                    "`lookup` tool whenever the user names a city — it is "
                    "the only authoritative source."
                ),
                max_rounds=4, max_tokens=512, temperature=0,
                compactor=None,
            ),
            tool_registry=_build_registry(call_log),
        )
        q = "What is Lima's signature dish? Be brief."
        r = await loop.send(q)
        assert r.status == "completed"
        assert "lima" in call_log, (
            f"the model did not invoke the lookup tool; call_log={call_log}"
        )

        # Verify pending is cleared after every tool resolves.
        assert loop.get_pending(r.session_id) is None

        # Verify persistence: the store records the assistant(tool_calls)
        # and the matching tool message.
        msgs = loop.get_messages(r.session_id)
        roles = [m["role"] for m in msgs]
        assert "tool" in roles, f"no tool message persisted; roles={roles}"

        await assert_passes(
            question=q,
            answer=r.final_text,
            rubric=(
                "Answer identifies ceviche as Lima's signature dish "
                "(case-insensitive, may include extra words)."
            ),
        )
    finally:
        store.close()
