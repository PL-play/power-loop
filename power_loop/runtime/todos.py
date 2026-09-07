"""待办清单的单一校验/渲染口径（6.16.0）。

**为什么要有 owner**：原来的模型是「同一时刻只能有一件 in_progress」——那是「一个人一次
只做一件事」的假设，对**自己动手**的活完全正确。但一个 agent 可以把活派出去：后台命令、
子 agent、workflow。派出去之后那件事**确实在进行中**，而且可以同时有好几件。旧模型逼着
它要么谎报（把派出去的标成 pending，于是清单看不出有活在飞），要么撞上
「Only one task can be in_progress」的硬错。

所以拆成两个维度：

- ``status`` 只说**进展**：pending / in_progress / completed。
- ``owner`` 说**谁在做**：self / background / subagent / workflow。

单例规则随之收窄成它本来的意思：**同时只能有一件「我自己动手」的活**；派出去的不限。

``ref``（task_id / run id）在 owner != self 时**必填**——「派出去了」如果说不出派到哪，
就是不可核查的：清单上那一行会永远挂着，没人回来收。这一条把 design/95 的判据
（「我怎么判断它做完了」）变成结构约束，而不是一句劝告。
"""
from __future__ import annotations

from typing import Any

MAX_TODOS = 20
STATUSES = ("pending", "in_progress", "completed")
#: 谁在做这件事。self = 我自己动手（受单例约束）；其余三个都是派出去的。
OWNERS = ("self", "background", "subagent", "workflow")
_OWNER_LABEL = {"background": "后台", "subagent": "子 agent", "workflow": "workflow"}


def validate_todos(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """校验并归一化一份清单。抛 ValueError 即模型看到的报错，所以每条都要写出**出路**。"""
    if len(items) > MAX_TODOS:
        raise ValueError(f"Max {MAX_TODOS} todos allowed")
    validated: list[dict[str, Any]] = []
    self_running = 0
    for i, item in enumerate(items):
        text = str(item.get("text", "")).strip()
        status = str(item.get("status", "pending")).lower()
        item_id = str(item.get("id", str(i + 1)))
        owner = str(item.get("owner") or "self").lower()
        ref = str(item.get("ref") or "").strip()
        if not text:
            raise ValueError(f"Item {item_id}: text required")
        if status not in STATUSES:
            raise ValueError(f"Item {item_id}: invalid status '{status}'")
        if owner not in OWNERS:
            raise ValueError(
                f"Item {item_id}: invalid owner '{owner}' — 只能是 "
                + "/".join(OWNERS)
                + "（self=你自己动手，其余是派出去的）"
            )
        if owner != "self" and not ref:
            raise ValueError(
                f"Item {item_id}: owner={owner} 必须带 ref（task_id / run id / 子会话 id）"
                "——说不出派到哪的活没法回来收，清单上会永远挂着。"
                "先真的派出去、拿到句柄，再把这一条标成 owner={owner}。".format(owner=owner)
            )
        if status == "in_progress" and owner == "self":
            self_running += 1
        entry: dict[str, Any] = {"id": item_id, "text": text, "status": status, "owner": owner}
        if ref:
            entry["ref"] = ref
        validated.append(entry)
    if self_running > 1:
        raise ValueError(
            "同时只能有一件「我自己动手」的活（owner=self 且 in_progress）。"
            "真正在并行跑的，把 owner 标成 background/subagent/workflow 并带上 ref；"
            "还没开始的收回 pending。"
        )
    return validated


def render_todos(items: list[dict[str, Any]]) -> str:
    if not items:
        return "No todos."
    lines: list[str] = []
    for item in items:
        marker = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}[item["status"]]
        tail = ""
        owner = str(item.get("owner") or "self")
        if owner != "self":
            label = _OWNER_LABEL.get(owner, owner)
            ref = str(item.get("ref") or "")
            tail = f"  ({label}{' ' + ref if ref else ''})"
        lines.append(f"{marker} #{item['id']}: {item['text']}{tail}")
    done = sum(1 for item in items if item["status"] == "completed")
    flying = sum(1 for item in items
                 if item["status"] == "in_progress" and str(item.get("owner") or "self") != "self")
    summary = f"\n({done}/{len(items)} completed"
    summary += f"; {flying} 件在外面跑)" if flying else ")"
    lines.append(summary)
    return "\n".join(lines)


def todo_counts(items: list[dict[str, Any]]) -> dict[str, int]:
    """给回执与 follow_up_rules 谓词用的计数。``delegated`` = 派出去且仍在跑的件数。"""
    return {
        "total": len(items),
        "completed": sum(1 for t in items if t["status"] == "completed"),
        "delegated": sum(1 for t in items if t["status"] == "in_progress"
                         and str(t.get("owner") or "self") != "self"),
    }
