"""21 · Request user input / 可恢复的人类输入

This example uses a real LLM. It starts two independent sessions, lets each
session pause through the built-in ``request_user_input`` tool, prints the
returned ``StatefulResult``, asks you to fill the answers in a tiny terminal
REPL, then resumes both sessions with ``submit_input``.

Run:

    python examples/21_request_user_input.py
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from collections.abc import Sequence
from dataclasses import asdict, is_dataclass
from typing import Any

from _helpers import make_llm

from power_loop import AgentLoopConfig, SessionStore, StatefulAgentLoop, create_default_tool_registry

SYSTEM_PROMPT = """You are demonstrating resumable human input.

Rules:
- For each new user request, first call request_user_input exactly once.
- Use kind="choice".
- Provide two concise options with ids.
- After the tool result arrives, produce one final sentence that mentions the chosen id.
- Do not call request_user_input again after receiving its tool result.
"""

REQUESTS = {
    "summary": (
        "Draft a one-sentence relationship check-in. Before finalizing, ask "
        "whether the tone should be 'gentle' or 'direct'."
    ),
    "send": (
        "Prepare a one-sentence note to send. Before finalizing, ask whether "
        "the caller wants to 'send' or 'revise'."
    ),
}


async def main(answers: Sequence[str] | None = None) -> list[dict[str, Any]]:
    llm = make_llm(max_tokens=700, temperature=0.0)
    registry = create_default_tool_registry(include=["request_user_input"])
    provided_answers = list(answers or [])

    with tempfile.TemporaryDirectory(prefix="power-loop-input-") as tmp:
        store = SessionStore.open(f"{tmp}/sessions.sqlite3")
        loop = StatefulAgentLoop(
            llm=llm,
            store=store,
            tool_registry=registry,
            config=AgentLoopConfig(system_prompt=SYSTEM_PROMPT, max_rounds=4, compactor=None),
        )

        try:
            waiting: list[dict[str, Any]] = []
            for label, request in REQUESTS.items():
                sid = loop.new_session(metadata={"label": label})
                result = await loop.send(request, session_id=sid)
                if result.status != "waiting_for_input" or not result.pending_interactions:
                    raise RuntimeError(f"session {label} did not pause for input: {result!r}")

                print(f"\n[{label}] first StatefulResult")
                print(_json(result))
                waiting.append({"label": label, "sid": sid, "result": result})

            completed: list[dict[str, Any]] = []
            for item in waiting:
                label = item["label"]
                result = item["result"]
                interaction = result.pending_interactions[0]
                answer = _next_answer(provided_answers, interaction)
                resumed = await loop.submit_input(
                    item["sid"],
                    interaction["interaction_id"],
                    {"choice": answer},
                )

                print(f"\n[{label}] resumed StatefulResult")
                print(_json(resumed))
                completed.append(
                    {
                        "label": label,
                        "session_id": item["sid"],
                        "answer": answer,
                        "status": resumed.status,
                        "final_text": resumed.final_text,
                    }
                )

            return completed
        finally:
            await llm.close()
            store.close()


def _next_answer(provided_answers: list[str], interaction: dict[str, Any]) -> str:
    options = [str(option.get("id") or option.get("label") or "") for option in interaction.get("options", [])]
    options = [option for option in options if option]
    if provided_answers:
        answer = provided_answers.pop(0)
        print(f"\n[auto-answer] {interaction['prompt']} -> {answer}")
        return answer

    print("\n[input required]")
    print(interaction["prompt"])
    if options:
        print("Options: " + ", ".join(options))
    while True:
        answer = input("> ").strip()
        if answer and (not options or answer in options):
            return answer
        if options:
            print("Please enter one of: " + ", ".join(options))
        else:
            print("Please enter a non-empty answer.")


def _json(value: Any) -> str:
    def default(obj: Any) -> Any:
        if is_dataclass(obj):
            return asdict(obj)
        return repr(obj)

    return json.dumps(value, ensure_ascii=False, indent=2, default=default)


if __name__ == "__main__":
    asyncio.run(main())
