"""Real-LLM check that prompt cache-read tokens (schema v5) are captured from the provider's usage
and persisted to ``session_stats.cached_tokens``.

Several sends share a large, stable system prompt + the growing history, so a caching provider
(OpenAI / DeepSeek-style prompt caching) is likely to report cache reads on later sends. The robust
assertion is the PLUMBING: whatever the real LLM reports as ``cache_read_tokens``, the store's
cumulative ``cached_tokens`` equals the sum across sends — the feature works regardless of whether
caching actually fired this run.
"""

from __future__ import annotations

import pytest

from power_loop import AgentLoopConfig, StatefulAgentLoop

from ._llm import make_llm

# A large, stable system prompt maximizes the chance the provider's prompt cache fires on later sends.
_BIG_SP = "You are a terse assistant; reply with a single lowercase word.\n" + ("context note. " * 600)


@pytest.mark.asyncio
async def test_real_cached_tokens_persisted(tmp_path) -> None:
    db = str(tmp_path / "cached.db")
    loop = StatefulAgentLoop(
        llm=make_llm(max_tokens=16, temperature=0),
        db_path=db,
        config=AgentLoopConfig(system_prompt=_BIG_SP, max_rounds=1, max_tokens=16, temperature=0),
    )
    sid = await loop.new_session()
    expected_cached = 0
    try:
        for prompt in ("reply: one", "reply: two", "reply: three"):
            r = await loop.send(prompt, session_id=sid)
            assert r.status == "completed", r.status
            expected_cached += int(r.usage.get("cache_read_tokens") or 0)

        stats = await loop.store.get_session_stats(sid)
        assert stats is not None
        # The feature: the store's cumulative cached_tokens == the sum the real provider reported.
        assert stats.cached_tokens == expected_cached
        assert stats.cached_tokens >= 0
        # Sanity that usage flows end to end (prompt/completion are populated by the same path).
        assert stats.prompt_tokens > 0 and stats.completion_tokens > 0
        print(
            f"\n[cached-tokens] cached={stats.cached_tokens} "
            f"prompt={stats.prompt_tokens} completion={stats.completion_tokens}"
        )
    finally:
        await loop.aclose()
