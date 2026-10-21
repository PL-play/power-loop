"""pass_turn —— 内置的「这一步做完了，结束这一轮」(design/188)。

一个 send 有两种正常的结束：这一轮只写字、不调工具（自然结束）；或者调 ``pass_turn``（明确结束）。
不强制以 pass_turn 结尾。

- 工具**永远在**：pipeline 建立时 ``ensure_pass_turn`` 把它补进工具表（宿主去不掉）。
- 宿主可以用同名工具覆盖说明和处理（例如先核对再决定停不停）；处理里调 ``request_pass(理由)`` 才算数。
- 硬停：标记之后，pipeline 在**下一轮开始前**停（不再调模型），结果上 ``passed=True``、``pass_reason``；
  停的时候若有刚折进来的新输入（转向消息），记 ``new_input_after_pass``，由宿主决定要不要另起一轮。
- 状态按 send 隔离（ContextVar，每个 pipeline.run 开始时装一份、结束时还原）：子 agent 在同一个任务里
  内联跑，它的 pass 不会翻父 send 的标记。
"""
from __future__ import annotations

from contextvars import ContextVar, Token
from typing import Any

from power_loop.contracts.tools import ToolDefinition

NAME = "pass_turn"
REASON_MAX = 300

DESCRIPTION = (
    "这一步的事做完了，结束这一轮（不再继续调用模型）。"
    "什么时候用：要做的已经做完、结果已经交出去（写出来了或用工具交了），没有别的要做；"
    "或者在等后台的结果（它好了会叫醒你）。"
    "只写字、不调工具也会结束这一轮——用它是明确地说「做完了」。reason 写一句为什么结束。"
)

PASS_TURN_TOOL = ToolDefinition(
    name=NAME,
    description=DESCRIPTION,
    input_schema={
        "type": "object",
        "properties": {"reason": {"type": "string", "description": "一句话：为什么结束这一轮（记日志，不展示）。"}},
        "required": ["reason"],
    },
    required_params=("reason",),
)

_state: ContextVar[dict[str, Any] | None] = ContextVar("power_loop_pass_turn", default=None)


def begin() -> Token:
    """一个 send 开始：装一份新的状态。返回还原用的 token。"""
    return _state.set({"passed": False, "reason": None, "new_input_after_pass": False})


def end(token: Token) -> None:
    _state.reset(token)


def current() -> dict[str, Any] | None:
    return _state.get()


def request_pass(reason: str = "") -> bool:
    """标记「这一轮结束」：下一轮开始前停。不在 send 里（没有状态）返回 False。"""
    st = _state.get()
    if st is None:
        return False
    st["passed"] = True
    st["reason"] = (reason or "").strip()[:REASON_MAX]
    return True


def pass_requested() -> bool:
    st = _state.get()
    return bool(st and st.get("passed"))


def default_handler(reason: str = "") -> str:
    if not request_pass(reason):
        return "pass_turn 记下了（当前没有在跑的一轮）。"
    return "这一轮结束了。不用再写别的。"


def ensure_pass_turn(registry: Any) -> Any:
    """工具表里没有 pass_turn 就补上（永远在）。"""
    if not registry.has(NAME):
        registry.register(PASS_TURN_TOOL, default_handler)
    return registry
