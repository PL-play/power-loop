"""``async_capable`` 的 action 粒度（6.15.0）。

为什么需要它：很多多义工具在同一个入口下既有只读 action 也有写 action
（``design_reference`` 的 get/list 只读、freeze 要写）。工具级的一个布尔值只能二选一——
整体不标，纯读的那几个 action 也没法并发；整体标上，写 action 会被并发或被后台重跑。

判据不对称是这组测试的主题：**漏判只是慢，误判是数据损坏**。所以拿不到 args 时一律判否。
"""
from __future__ import annotations

import pytest

from power_loop.contracts.tools import ToolDefinition
from power_loop.tools.registry import ToolRegistry, async_capable_actions, async_capable_for

RO = frozenset({"get", "list", "download"})


def _d(flag):
    return ToolDefinition(name="design_reference", description="d", async_capable=flag)


def test_bool_true_is_still_tool_wide():
    """旧写法完全不受影响。"""
    d = _d(True)
    assert async_capable_for(d, {}) is True
    assert async_capable_for(d, {"action": "freeze"}) is True
    assert async_capable_for(d, None) is True
    assert async_capable_actions(d) == ()


def test_bool_false_is_never_async():
    d = _d(False)
    assert async_capable_for(d, {"action": "get"}) is False


def test_only_the_listed_actions_pass():
    d = _d(RO)
    assert async_capable_for(d, {"action": "get"}) is True
    assert async_capable_for(d, {"action": "list"}) is True
    assert async_capable_for(d, {"action": "freeze"}) is False   # 写 action 必须挡住
    assert async_capable_actions(d) == ("download", "get", "list")


@pytest.mark.parametrize("args", [None, {}, {"action": ""}, {"action": "   "}, {"other": "get"}])
def test_missing_action_is_refused_not_guessed(args):
    """拿不到 action 就判否——同轮并发要在发起**前**决定，宁可少并发一次，
    也不能把一个写 action 当成只读的并发出去。"""
    assert async_capable_for(_d(RO), args) is False


def test_empty_frozenset_is_not_async():
    """空集合 = 一个 action 都没放行，不能被当成「真值缺省放行」。"""
    assert async_capable_for(_d(frozenset()), {"action": "get"}) is False


def test_openai_description_names_the_allowed_actions():
    """模型看得到范围才不会拿 freeze 去后台跑（报错即出路的前一步：先别让它撞）。"""
    reg = ToolRegistry()
    reg.register(_d(RO), lambda **kw: "")
    reg.register(ToolDefinition(name="background_run", description="bg"), lambda **kw: "")
    desc = next(t["function"]["description"] for t in reg.to_openai_tools()
                if t["function"]["name"] == "design_reference")
    assert "可异步（仅 action=download/get/list）" in desc
    assert "freeze" not in desc.split("可异步")[1]
