"""How long each kind of work gets to wind down once it is told to stop (design/124 §8.4).

Stopping is mechanical — no model call is ever made to "wrap up" (a stop must work when the model
is unreachable, and must not start new work). Every wait below is bounded; past it the work is
forced to end. All values are configuration, not constants: the defaults are the design's
recommendations; hosts expose them (DeepTalk: admin 系统配置).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StopPolicy:
    #: an ``interrupt="abort"`` tool: after cancelling it, how long to let its cleanup run
    abort_tool_s: float = 5.0
    #: an ``interrupt="finish"`` tool (quick, or can't be stopped — a thread): how long to wait
    #: for its real result before giving up on it
    finish_tool_s: float = 30.0
    #: a sub-task tool (``interrupt="background"``: sub-agent, platform Q&A, awaited workflow):
    #: its child stop token is already flipped — how long it gets to stop at its own checkpoints
    subtask_s: float = 30.0
    #: a workflow (sync or detached): how long its leaves get before they're cancelled
    workflow_s: float = 60.0
    #: a background tool task stopped by id
    background_task_s: float = 10.0
    #: a background shell command: SIGTERM → this many seconds → SIGKILL
    shell_term_s: float = 5.0
    #: the whole process shutting down (hosts: keep below the orchestrator's kill grace)
    shutdown_s: float = 45.0

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if not isinstance(value, (int, float)) or value < 0:
                raise ValueError(f"StopPolicy.{name} must be a number >= 0, got {value!r}")


__all__ = ["StopPolicy"]
