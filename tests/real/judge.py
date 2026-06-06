"""LLM-as-judge helper for real-LLM tests.

Real-LLM responses are non-deterministic. Asserting exact string equality
makes tests brittle, but asserting "looks vaguely English" is too loose. The
sweet spot is to spawn a second power-loop (a "judge") that reads
``{question, answer, rubric}`` and returns a structured ``{passed, reason}``
verdict — power-loop evaluating itself.

The judge runs with a separate :class:`SessionStore` (in-memory) and a
strict JSON-output system prompt. Failures dump the rubric and the answer
into the assertion message so debugging is one-glance.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from power_loop import AgentLoopConfig, SessionStore, StatefulAgentLoop, StructuredOutputError, parse_structured

from ._llm import make_llm

_JUDGE_SYSTEM = """You are a strict test judge. You will be given:
- QUESTION: the user request the agent received
- ANSWER: the agent's reply text
- RUBRIC: a short list of pass conditions (all must hold)

Decide whether the ANSWER satisfies the RUBRIC. Output ONLY a JSON object on
a single line, no prose, no markdown:

  {"passed": true|false, "reason": "<one short sentence>"}

If unsure, lean toward false and explain in `reason`."""


@dataclass(frozen=True)
class Verdict:
    passed: bool
    reason: str


async def judge_async(
    *,
    question: str,
    answer: str,
    rubric: str,
    model: str | None = None,
) -> Verdict:
    """Spawn an evaluator power-loop and return its verdict.

    Uses a fresh in-memory ``SessionStore`` per call so judges never share
    state with the system under test.
    """
    llm = make_llm(max_tokens=256, temperature=0, model=model)
    store = SessionStore.open(":memory:")
    try:
        loop = StatefulAgentLoop(
            llm=llm,
            store=store,
            config=AgentLoopConfig(
                system_prompt=_JUDGE_SYSTEM,
                max_rounds=1,
                temperature=0,
                max_tokens=256,
                compactor=None,
            ),
        )
        user_msg = (
            f"QUESTION:\n{question}\n\n"
            f"ANSWER:\n{answer}\n\n"
            f"RUBRIC (all must hold):\n{rubric}"
        )
        sid = loop.new_session()
        result = await loop.send(user_msg, session_id=sid)
        return _parse_verdict(result.final_text)
    finally:
        store.close()


def judge_sync(**kwargs) -> Verdict:
    return asyncio.run(judge_async(**kwargs))


async def assert_passes(
    *,
    question: str,
    answer: str,
    rubric: str,
    model: str | None = None,
) -> None:
    """Async pytest helper: await the judge, raise AssertionError on fail."""
    v = await judge_async(question=question, answer=answer, rubric=rubric, model=model)
    assert v.passed, (
        f"Judge rejected the answer.\n"
        f"  reason : {v.reason}\n"
        f"  rubric : {rubric}\n"
        f"  answer : {answer[:600]}{'…' if len(answer) > 600 else ''}"
    )


_VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "passed": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["passed", "reason"],
}


def _parse_verdict(text: str) -> Verdict:
    text = (text or "").strip()
    if not text:
        return Verdict(passed=False, reason="judge returned empty text")

    candidates = [text]
    if text.count("{") > text.count("}"):
        candidates.append(text + ("}" * (text.count("{") - text.count("}"))))

    for candidate in candidates:
        try:
            obj = parse_structured(candidate, schema=_VERDICT_SCHEMA)
        except StructuredOutputError:
            continue
        passed = bool(obj.get("passed"))
        reason = str(obj.get("reason") or "")
        return Verdict(passed=passed, reason=reason)

    return Verdict(
        passed=False,
        reason=f"judge produced unparseable output: {text[:200]}",
    )
