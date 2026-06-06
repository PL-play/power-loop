"""07 · 中断 / 用户确认 / Human approval: confirm before executing bash

What you learn / 你将学到
--------------------------
- ``TOOL_BEFORE`` hook 是 async 的：handler 里可以 ``await`` 任意 UI / WebSocket /
  CLI 输入，模型那边就**真的在等**——没有定时器
  / ``TOOL_BEFORE`` hook is async: handler can ``await`` any UI / WebSocket /
  CLI input — the model **actually waits**, no timers
- 同意 → 默认 CONTINUE，工具正常跑
  / Approve → default CONTINUE, tool runs normally
- 拒绝 → ``ctx.output = "[denied]"`` + ``ctx.directive = HookDirective.SKIP``
  会被 pipeline 当成这个工具的返回结果，写一条 ``role=tool`` 消息回灌给 LLM；
  pending 状态自动清零，协议合法
  / Deny → ``ctx.output = "[denied]"`` + ``ctx.directive = HookDirective.SKIP``
  becomes the tool's return value, written as ``role=tool`` message back to LLM;
  pending state auto-clears, protocol stays valid
- LLM 看到 ``[denied]`` 会自然改方向
  / LLM naturally changes direction upon seeing ``[denied]``

设计要点 / Design notes
------------------------
- ``confirm_fn`` 是注入式回调，CLI 跑时用 ``input()``，测试时塞固定决策
  / ``confirm_fn`` is an injectable callback: ``input()`` for CLI, fixed decision for tests
- 白名单：``ls`` / ``pwd`` / ``echo`` / ``cat`` 这类只读命令自动放行，不打扰用户
  / whitelist: read-only commands like ``ls`` / ``pwd`` auto-approved, no user interruption
- 危险关键字（``rm`` / ``sudo`` / ``mv`` / ``dd`` 等）一律要求 confirm
  / dangerous keywords (``rm`` / ``sudo`` / ``mv`` / ``dd`` etc.) always require confirm

⚠️ 工具参数命名注意 / Tool parameter naming note
--------------------------------------------------
power-loop 内置 ``DEFAULT_REQUIRED_PARAMS`` 给名为 ``bash`` 的工具预留了参数名
``command``。我们用相同名字以避开校验冲突——如果换成 ``cmd``，registry 的
默认校验会把所有 bash 调用挡掉。
/ power-loop's built-in ``DEFAULT_REQUIRED_PARAMS`` reserves the parameter name
``command`` for tools named ``bash``. We use the same name to avoid validation
conflicts — using ``cmd`` would block all bash calls.

Run / 运行
----------
    python examples/07_human_approval.py
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Awaitable, Callable

from _helpers import make_llm

from power_loop import (
    AgentHooks,
    AgentLoopConfig,
    HookDirective,
    HookPoint,
    StatefulAgentLoop,
    ToolDefinition,
    ToolRegistry,
)
from power_loop.contracts.hook_contexts import ToolBeforeCtx

# ── 1. 一个简单的 bash 工具 / A simple bash tool ─────────────────────────

EXECUTED: list[str] = []


def fake_bash(**kwargs) -> str:
    """Fake 实现避免真的跑命令。真实场景下接 subprocess。
    / Fake implementation to avoid running real commands. Use subprocess in production."""
    cmd = str(kwargs.get("command") or "")
    EXECUTED.append(cmd)
    if cmd.startswith("ls"):
        return "file1.txt\nfile2.txt\nREADME.md"
    if cmd.startswith("pwd"):
        return "/Users/demo/project"
    return f"(simulated) ran: {cmd}"


BASH_TOOL = ToolDefinition(
    name="bash",
    description="Run a shell command and return its stdout.",
    input_schema={
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
    },
    required_params=("command",),
)


# ── 2. 确认策略 / Confirmation policy ─────────────────────────────────────

DANGEROUS_TOKENS = ("rm ", "rm\t", " rm", "sudo", "mv ", "dd ", "chmod ", "chown ", " > /")
SAFE_PREFIXES = ("ls", "pwd", "echo", "cat ", "head ", "tail ", "wc ")


def is_safe(cmd: str) -> bool:
    stripped = cmd.strip()
    return any(stripped.startswith(p.strip()) for p in SAFE_PREFIXES) and \
        not any(t in cmd for t in DANGEROUS_TOKENS)


ConfirmFn = Callable[[str], Awaitable[bool]]


async def cli_confirm(cmd: str) -> bool:
    """默认实现：CLI input / Default: CLI input."""
    print(f"\n[CONFIRM] About to run:\n    {cmd}\nproceed? [y/N]: ", end="", flush=True)
    line = await asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline)
    return line.strip().lower() in ("y", "yes")


def build_hook(confirm_fn: ConfirmFn) -> AgentHooks:
    hooks = AgentHooks()

    async def ask_before_bash(ctx: ToolBeforeCtx) -> None:
        if ctx.tool_name != "bash":
            return
        cmd = str(ctx.tool_args.get("command") or "")

        if is_safe(cmd):
            print(f"[auto-approve] {cmd}")
            return                                # 直接放行 / auto-approve

        approved = await confirm_fn(cmd)
        if not approved:
            ctx.output = f"[denied by user — command was not executed: {cmd!r}]"
            ctx.directive = HookDirective.SKIP

    hooks.register(HookPoint.TOOL_BEFORE, ask_before_bash)
    return hooks


# ── 3. 主程序 / Main program ──────────────────────────────────────────────


async def run_with_policy(confirm_fn: ConfirmFn, prompt: str) -> str:
    registry = ToolRegistry()
    registry.register(BASH_TOOL, fake_bash)

    loop = StatefulAgentLoop(
        llm=make_llm(max_tokens=512, temperature=0),
        db_path=":memory:",
        tool_registry=registry,
        hooks=build_hook(confirm_fn),
        config=AgentLoopConfig(
            system_prompt=(
                "You have a `bash` tool that takes a `command` string. "
                "Call it for every shell action. If a tool call is denied, "
                "do NOT retry the same command — explain to the user that "
                "it was denied and stop."
            ),
            max_rounds=5,
            compactor=None,
        ),
    )
    sid = loop.new_session()
    r = await loop.send(prompt, session_id=sid)
    print(f"\n[reply] status={r.status}, rounds={r.rounds}")
    print(f"[reply] {r.final_text}")
    print(f"[stats] commands actually executed: {EXECUTED}")
    return r.final_text


async def main() -> str:
    # 演示：用户**拒绝**任何危险命令
    # Demo: user **denies** all dangerous commands
    async def always_deny(cmd: str) -> bool:
        print(f"[CONFIRM] {cmd!r} → simulating user input N (deny)")
        return False

    EXECUTED.clear()
    return await run_with_policy(
        always_deny,
        "Please list the files in the project, then delete README.md.",
    )


if __name__ == "__main__":
    asyncio.run(main())
