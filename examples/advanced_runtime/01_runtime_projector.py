"""Advanced runtime 01: project incident state into the next LLM round.

Scenario
--------
An incident commander dashboard keeps authoritative state in SQLite. The LLM
should see the latest incident state every round, but that projection should
not be persisted as chat history.

Run:
    python examples/advanced_runtime/01_runtime_projector.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _runtime_helpers import ScriptedLLM

from llm_client.interface import LLMResponse
from power_loop import AgentLoopConfig, RuntimeProjector, SessionStore, StatefulAgentLoop


class IncidentProjector(RuntimeProjector):
    def project(self, *, store: Any, session_id: str, round_index: int, context: Any) -> list[dict[str, Any]]:
        incident = store.get_runtime_state(session_id, "incident", default={}) or {}
        if not incident:
            return []
        return [
            {
                "role": "user",
                "name": "incident_runtime_state",
                "content": (
                    "<incident_state>\n"
                    f"severity: {incident['severity']}\n"
                    f"summary: {incident['summary']}\n"
                    "</incident_state>"
                ),
            }
        ]


async def main() -> str:
    store = SessionStore.open(":memory:")
    sid = store.create_session()
    store.set_runtime_state(
        sid,
        "incident",
        {"severity": "SEV2", "summary": "checkout latency above threshold"},
    )

    llm = ScriptedLLM(responses=[LLMResponse(raw_text="Incident state visible: SEV2 checkout latency.")])
    loop = StatefulAgentLoop(
        llm=llm,
        store=store,
        config=AgentLoopConfig(
            system_prompt="You assist an incident commander.",
            max_rounds=1,
            compactor=None,
            runtime_projectors=(IncidentProjector(),),
        ),
    )

    result = await loop.send("What is the latest incident state?", session_id=sid)
    print(result.final_text)
    print("projected message:", llm.calls[0][-1]["content"])
    return result.final_text


if __name__ == "__main__":
    asyncio.run(main())
