"""后台任务的回执措辞：**不许把 check 说成取结果的正常路径**（conv-238 复盘，2026-09-09）。

真事：229 把两个 platform 提问扔进后台，然后连着四轮只做一件事——check「好了没」，
每次拿回「还在跑」。四轮白烧约 12 万 prompt token、三分半钟墙上时间，还逼出两条给用户的
「还在等」消息（其中一条在道歉）。根因不在模型：启动回执的后半句把 check 摆出来当取结果的
正规路径，而完成通知本来就会把结果送到。这里把措辞钉住，别再退回去。
"""

from power_loop.tools.default_manifest import DEFAULT_TOOL_DEFINITIONS


def _bg_description() -> str:
    for d in DEFAULT_TOOL_DEFINITIONS:
        if d.name == "background_run":
            return d.description
    raise AssertionError("background_run 不在默认工具清单里")


def test_tool_description_forbids_polling():
    desc = _bg_description()
    assert "do not poll" in desc.lower(), "工具描述必须明说别轮询"
    assert "pass_turn" in desc, "要给出「没别的可做就停轮」这条出路，否则模型只剩 check 可调"
    assert "fallback" in desc.lower(), "check 必须被降格成兜底，不能读成取结果的正常路径"


def test_launch_receipts_send_the_model_away_not_to_check():
    """两条启动回执（tool 与 shell）都要说清：结果会自动送到，别去问。"""
    import inspect

    from power_loop.tools import default_tools

    src = inspect.getsource(default_tools.BackgroundManager)
    assert "完成时结果会自动送到你面前，不用去取" in src
    assert "只会空烧回合" in src
    assert "do NOT poll with check" in src
    # 兜底出口仍要留着：卡住时得有办法查。
    assert 'action=\\"check\\"' in src or "action=check" in src
