"""待办的 owner 维度（6.16.0）：派出去的活也算「进行中」。

旧模型只有一个 in_progress 名额——那是「一个人一次做一件事」的假设，对自己动手完全正确，
但一个 agent 可以同时把三件活分别丢给后台、子 agent、workflow。旧模型逼它二选一：
要么谎报（把派出去的标 pending，清单就看不出有活在飞），要么撞
「Only one task can be in_progress」的硬错。
"""
from __future__ import annotations

import pytest

from power_loop.runtime.todos import render_todos, todo_counts, validate_todos


def _i(id_, text, status="pending", **kw):
    return {"id": id_, "text": text, "status": status, **kw}


def test_many_delegated_items_can_run_at_once():
    items = validate_todos([
        _i("1", "画三张封面", "in_progress", owner="background", ref="task_9f2a"),
        _i("2", "查竞品", "in_progress", owner="subagent", ref="sess_ab12"),
        _i("3", "六屏施工", "in_progress", owner="workflow", ref="wf_77"),
        _i("4", "我自己写产品定义", "in_progress"),
    ])
    assert len(items) == 4
    assert todo_counts(items)["delegated"] == 3


def test_still_only_one_self_in_progress():
    """单例规则收窄成它本来的意思，而不是取消。"""
    with pytest.raises(ValueError, match="我自己动手"):
        validate_todos([_i("1", "a", "in_progress"), _i("2", "b", "in_progress")])


def test_delegated_without_ref_is_refused():
    """「派出去了」说不出派到哪，就是不可核查的——那一行会永远挂着没人收。"""
    with pytest.raises(ValueError, match="必须带 ref"):
        validate_todos([_i("1", "画封面", "in_progress", owner="background")])


def test_owner_defaults_to_self_so_old_lists_still_work():
    items = validate_todos([_i("1", "a", "in_progress"), _i("2", "b")])
    assert [x["owner"] for x in items] == ["self", "self"]
    assert "ref" not in items[0]          # 没给 ref 就不要凭空塞一个


def test_bad_owner_names_the_allowed_ones():
    with pytest.raises(ValueError, match="self/background/subagent/workflow"):
        validate_todos([_i("1", "a", owner="someone_else", ref="x")])


def test_render_shows_where_the_work_went():
    out = render_todos(validate_todos([
        _i("1", "画三张封面", "in_progress", owner="background", ref="task_9f2a"),
        _i("2", "我自己写产品定义", "in_progress"),
    ]))
    assert "(后台 task_9f2a)" in out          # 看得见派到哪，才回得来收
    assert "1 件在外面跑" in out
    assert "#2: 我自己写产品定义" in out and "(" not in out.split("#2")[1].split("\n")[0]
