"""Common completion must not refresh an owner's time, rounds, or answer attempt."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from power_loop.runtime import completion, delegation
from power_loop.runtime.cancellation import CancellationToken

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


def response(status="completed", *, rounds=1, text="finished"):
    return SimpleNamespace(status=status, rounds=rounds, final_text=text,
        usage={"total_tokens": 5}, new_input_after_pass=False)


def initial(**updates):
    return {"status": "completed", "session_id": "leaf", "rounds": 1,
        "final_text": "finished", "usage": {"total_tokens": 10}, **updates}


class Loop:
    def __init__(self):
        self.pending = 0
        self.unseen = []
        self.background = []
        self.calls = []
        self.responses = []
        self.keep_pending = False

    async def ensure_store(self):
        return self

    async def inbox_pending(self, sid):
        return {"pending": self.pending}

    async def list_background_tasks(self, sid):
        return self.background

    async def list_unseen_background_updates(self, sid):
        return self.unseen

    async def get_background_task(self, sid, tid):
        return next((row for row in self.background if row.task_id == tid), None)

    async def send(self, *args, **kwargs):
        self.calls.append(kwargs)
        self.pending = 1 if self.keep_pending else 0
        self.unseen = []
        value = self.responses.pop(0)
        if isinstance(value, BaseException):
            raise value
        if isinstance(value, asyncio.Event):
            await value.wait()
        return value

    follow_up = send
    flush_follow_ups = send


@pytest.fixture(autouse=True)
def local_resolver(monkeypatch):
    from power_loop.runtime import spec
    async def resolve(loop, sid):
        return loop
    monkeypatch.setattr(spec, "workflow_continuation_loop", resolve)


async def test_reentry_keeps_spent_rounds_and_original_deadline():
    loop, window = Loop(), completion.CompletionWindow(5, 2)
    loop.pending, loop.responses = 1, [response(), response()]
    first = await completion.collect_required(loop, initial(), window=window)
    original_deadline = window.deadline
    assert window.remaining == 1 and first["usage"]["total_tokens"] == 15
    loop.pending = 1
    second = await completion.collect_required(loop, first, window=window)
    assert second["rounds"] == 3 and window.remaining == 0 and window.deadline == original_deadline
    loop.pending = 1
    third = await completion.collect_required(loop, second, window=window)
    assert third["status"] == "hit_round_limit" and len(loop.calls) == 2
    assert [c["max_rounds"] for c in loop.calls] == [2, 1]


async def test_reentry_cannot_refresh_time_before_another_paid_round():
    loop, window = Loop(), completion.CompletionWindow(0.02)
    await completion.collect_required(loop, initial(), window=window)
    await asyncio.sleep(0.03)
    loop.pending, loop.responses = 1, [response()]
    result = await completion.collect_required(loop, initial(), window=window)
    assert result["status"] == "failed" and "timed out" in result["error"] and not loop.calls


async def test_original_round_ceiling_blocks_an_unbounded_settlement_allowance():
    loop = Loop()
    loop.pending, loop.responses = 1, [response()]
    result = await completion.collect_required(loop, initial(rounds=3),
        window=completion.CompletionWindow(5, 30, round_ceiling=3))
    assert result["status"] == "hit_round_limit" and not loop.calls


async def test_zero_reported_rounds_cannot_evade_original_contract_ceiling():
    loop = Loop()
    loop.pending, loop.keep_pending = 1, True
    loop.responses = [response(rounds=0), response(rounds=0)]
    window = completion.CompletionWindow(5, 30, round_ceiling=3)
    result = await completion.collect_required(loop, initial(), window=window)
    assert result["status"] == "hit_round_limit" and len(loop.calls) == 2
    assert window.remaining == 28 and result["rounds"] == 1
    assert result["usage"]["total_tokens"] == 20


@pytest.mark.parametrize("final_answer", [False, True])
async def test_excessive_reported_work_preserves_usage_but_never_proves_success(final_answer):
    loop = Loop()
    loop.pending = 0 if final_answer else 1
    loop.responses = [response(rounds=3)]
    window = completion.CompletionWindow(5, 1)
    result = (await completion.finish_text(loop, initial(final_text=""), window=window) if final_answer else
        await completion.collect_required(loop, initial(), window=window))
    assert result["status"] == "failed" and "allowance exceeded" in result["error"]
    assert result["usage"]["total_tokens"] == 15 and result["rounds"] == 4


async def test_invalid_round_evidence_does_not_discard_reported_token_usage():
    loop = Loop()
    loop.pending, loop.responses = 1, [response(rounds=-1)]
    result = await completion.collect_required(loop, initial())
    assert result["status"] == "failed" and "round_evidence" in result["error"]
    assert result["usage"]["total_tokens"] == 15


async def test_final_answer_uses_same_remaining_budget_and_is_attempted_once():
    loop, window = Loop(), completion.CompletionWindow(5, 2)
    loop.pending, loop.responses = 1, [response(text=""), response(text="answer")]
    result = await completion.finish_text(loop, initial(final_text=""), window=window)
    assert result["final_text"] == "answer" and result["rounds"] == 3
    assert loop.calls[-1]["tools"] == [] and loop.calls[-1]["max_rounds"] == 1
    assert window.remaining == 0 and window.answer_attempted
    result = await completion.finish_text(loop, initial(final_text=""), window=window)
    assert result["status"] == "failed" and len(loop.calls) == 2


async def test_concurrent_collection_never_creates_a_second_model_processing_round(monkeypatch):
    loop, window = Loop(), completion.CompletionWindow(5)
    entered, release = asyncio.Event(), asyncio.Event()
    async def wait(*args):
        entered.set()
        await release.wait()
    monkeypatch.setattr(delegation, "await_children", wait)
    first = asyncio.create_task(completion.collect_required(loop, initial(), window=window))
    try:
        await entered.wait()
        second = await completion.collect_required(loop, initial(), window=window)
        assert second["status"] == "failed" and "already running" in second["error"]
    finally:
        release.set()
        await first
    assert not window._collecting and not loop.calls


async def test_cancellation_during_collection_releases_window_and_does_not_call_model(monkeypatch):
    loop, window, entered = Loop(), completion.CompletionWindow(5), asyncio.Event()
    async def wait(*args):
        entered.set()
        await asyncio.Event().wait()
    monkeypatch.setattr(delegation, "await_children", wait)
    task = asyncio.create_task(completion.collect_required(loop, initial(), window=window))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not window._collecting and not loop.calls


@pytest.mark.parametrize("status", ["failed", "cancelled", "hit_round_limit", "context_checkpoint", "degraded"])
async def test_unsuccessful_processing_never_triggers_more_work_or_answer_repair(status):
    loop = Loop()
    loop.pending, loop.keep_pending, loop.responses = 1, True, [response(status)]
    result = await completion.finish_text(loop, initial())
    assert result["status"] == status and len(loop.calls) == 1


@pytest.mark.parametrize("rounds", [0, 1])
async def test_zero_reported_rounds_are_still_bounded_and_prior_usage_is_preserved(rounds):
    loop = Loop()
    loop.pending, loop.keep_pending = 1, True
    loop.responses = [response(rounds=rounds), response(rounds=rounds)]
    result = await completion.collect_required(loop, initial(), window=completion.CompletionWindow(5, 2))
    assert result["status"] == "hit_round_limit" and result["usage"]["total_tokens"] == 20 and len(loop.calls) == 2


async def test_terminal_background_update_is_processed_even_after_it_stopped_running():
    loop = Loop()
    loop.unseen, loop.responses = [SimpleNamespace(status="completed")], [response()]
    result = await completion.collect_required(loop, initial())
    assert result["status"] == "completed" and len(loop.calls) == 1 and result["usage"]["total_tokens"] == 15


async def test_disappearing_background_record_is_a_failure():
    loop = Loop()
    loop.background = [SimpleNamespace(task_id="missing", status="queued")]
    async def vanished(*args):
        return None
    loop.get_background_task = vanished
    result = await completion.collect_required(loop, initial())
    assert result["status"] == "failed" and "disappeared" in result["error"] and not loop.calls


async def test_cancelled_owner_starts_no_paid_work():
    loop, token = Loop(), CancellationToken()
    loop.pending = 1
    token.cancel("stop")
    result = await completion.finish_text(loop, initial(), token)
    assert result["status"] == "cancelled" and not loop.calls


@pytest.mark.parametrize("options", [{"timeout_s": 0}, {"timeout_s": float("inf")}, {"max_rounds": -1}, {"max_rounds": True}, {"round_ceiling": True}, {"multi_session": "yes"}])
async def test_invalid_completion_budget_is_rejected(options):
    with pytest.raises(ValueError, match="invalid_completion"):
        completion.CompletionWindow(**options)


async def test_scope_rebinding_requires_the_owning_engine_policy():
    loop, window = Loop(), completion.CompletionWindow()
    await completion.collect_required(loop, initial(), window=window)
    with pytest.raises(ValueError, match="scope_changed"):
        await completion.collect_required(loop, initial(session_id="other"), window=window)
    window = completion.CompletionWindow(multi_session=True)
    await completion.collect_required(loop, initial(), window=window)
    deadline = window.deadline
    await completion.collect_required(loop, initial(session_id="retry"), window=window)
    assert window.deadline == deadline
