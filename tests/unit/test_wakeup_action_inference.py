"""6.13.0：schedule_wakeup 的 action 可省略，按参数形状推断。"""

from __future__ import annotations

import pytest

from power_loop.contracts.tools import DEFAULT_REQUIRED_PARAMS, validate_tool_args
from power_loop.tools import default_tools as dt


def test_action_is_no_longer_a_required_param():
    assert "schedule_wakeup" not in DEFAULT_REQUIRED_PARAMS
    assert validate_tool_args("schedule_wakeup", {"delay_seconds": 300, "note": "查进度"}) is None


@pytest.mark.asyncio
async def test_infers_schedule_list_and_cancel(monkeypatch):
    calls = []

    async def _sched(delay, note, every=None):
        calls.append(("schedule", delay, note, every))
        return "Wake-up #1"

    async def _list():
        calls.append(("list",))
        return "no wake-ups"

    async def _cancel(tid):
        calls.append(("cancel", tid))
        return "cancelled"

    monkeypatch.setattr(dt, "_schedule_wakeup", _sched)
    monkeypatch.setattr(dt, "_list_wakeups", _list)
    monkeypatch.setattr(dt, "_cancel_wakeup", _cancel)

    await dt.run_schedule_wakeup(action=None, delay_seconds=300, note="查进度")   # 模型的高频写法
    await dt.run_schedule_wakeup(action="")                                       # 空 → list
    await dt.run_schedule_wakeup(action=None, timer_id=7)                         # 只有 id → cancel
    await dt.run_schedule_wakeup(action="schedule", delay_seconds=60, note="x")   # 显式仍然有效
    assert [c[0] for c in calls] == ["schedule", "list", "cancel", "schedule"]
    assert calls[0][1] == 300 and calls[0][2] == "查进度" and calls[2][1] == 7


@pytest.mark.asyncio
async def test_bad_action_still_errors():
    with pytest.raises(ValueError):
        await dt.run_schedule_wakeup(action="snooze")
