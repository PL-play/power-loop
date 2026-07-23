"""``board`` tool — agents read/write the shared blackboard (the pull side).

One action-dispatched tool (``action=read|post|update|remove``; the four
separate ``board_*`` tools were merged in 5.0.0). It resolves the live
:class:`~power_loop.runtime.blackboard.Blackboard` and the agent's ``board_id``
from the current :class:`~power_loop.runtime.env.RuntimeEnv` at call time (so a
host injects the implementation + id per run/send, exactly like the sandbox
``ShellBackend``). The author is taken from the session metadata.
``kind``/``status`` vocabularies are the host's policy, set when the tool is
registered.

Opt-in: a board-less agent simply doesn't register this (or, if it somehow
calls it with no board configured, gets a clear error rather than a crash).
"""

from __future__ import annotations

from typing import Any

from power_loop.contracts.tools import ToolDefinition
from power_loop.runtime.blackboard import Blackboard, BlackboardError, render_entries
from power_loop.runtime.env import get_runtime_env
from power_loop.runtime.runtime_state import get_tool_runtime_context

__all__ = ["register_blackboard_tools"]

_NO_BOARD = "Error: no shared board is configured for this agent."


def _resolve_board() -> tuple[Blackboard, str] | None:
    """The live board + this agent's board_id, or ``None`` when none is configured.
    Returning a single sentinel (not ``(None, None)``) lets callers narrow BOTH the
    board and the id to non-None with one ``is None`` check."""
    env = get_runtime_env()
    if env.blackboard is None or not env.blackboard_id:
        return None
    return env.blackboard, env.blackboard_id


async def _author() -> str | None:
    ctx = get_tool_runtime_context()
    if ctx.store is not None and ctx.session_id:
        try:
            row = await ctx.store.get_session(ctx.session_id)
            md = (getattr(row, "metadata", None) or {}) if row is not None else {}
            return md.get("spec_name") or md.get("name") or ctx.session_id
        except Exception:  # noqa: BLE001 — author is best-effort
            return ctx.session_id
    return ctx.session_id


async def _snapshot(board: Blackboard, board_id: str, *, header: str) -> str:
    entries = await board.read(board_id)
    return render_entries(entries, header=header, empty="(the board is empty)")


async def _do_read(board: Blackboard, board_id: str, _kw: dict[str, Any]) -> str:
    return await _snapshot(board, board_id, header="Shared board:")


def _make_post(kinds: tuple[str, ...], statuses: tuple[str, ...], default_kind: str):
    async def _do_post(board: Blackboard, board_id: str, kw: dict[str, Any]) -> str:
        text = str(kw.get("text") or "").strip()
        if not text:
            return "Error: action=post requires 'text'."
        kind = kw.get("kind") or default_kind
        if kind not in kinds:
            return f"Error: kind must be one of {list(kinds)}."
        status = kw.get("status")
        if status is not None and status not in statuses:
            return f"Error: status must be one of {list(statuses)}."
        try:
            await board.post(board_id, text=text, kind=kind, status=status, author=await _author())
        except BlackboardError as exc:
            return f"Error: {exc}"
        return await _snapshot(board, board_id, header="Posted. Shared board:")

    return _do_post


def _make_update(statuses: tuple[str, ...]):
    async def _do_update(board: Blackboard, board_id: str, kw: dict[str, Any]) -> str:
        try:
            entry_id = int(kw["entry_id"])
        except (KeyError, TypeError, ValueError):
            return "Error: action=update requires 'entry_id' (integer)."
        status = kw.get("status")
        if status is not None and status not in statuses:
            return f"Error: status must be one of {list(statuses)}."
        try:
            await board.update(board_id, entry_id, text=kw.get("text"), status=status)
        except BlackboardError as exc:
            return f"Error: {exc}"
        return await _snapshot(board, board_id, header="Updated. Shared board:")

    return _do_update


async def _do_remove(board: Blackboard, board_id: str, kw: dict[str, Any]) -> str:
    try:
        entry_id = int(kw["entry_id"])
    except (KeyError, TypeError, ValueError):
        return "Error: action=remove requires 'entry_id' (integer)."
    try:
        await board.remove(board_id, entry_id)
    except BlackboardError as exc:
        return f"Error: {exc}"
    return await _snapshot(board, board_id, header="Removed. Shared board:")


def register_blackboard_tools(
    registry: Any,
    *,
    kinds: tuple[str, ...] = ("note", "task"),
    statuses: tuple[str, ...] = ("open", "doing", "done"),
    default_kind: str = "note",
    overwrite: bool = False,
) -> None:
    """Register the single action-dispatched ``board`` tool.

    ``kinds`` / ``statuses`` set the validated vocabularies (and the tool schema
    the model sees). The host must inject a ``Blackboard`` + ``blackboard_id`` on
    the per-send ``RuntimeEnv`` for the tool to operate.
    """
    board_def = ToolDefinition(
        name="board",
        description=(
            "The shared agent board — a PRIVATE coordination space for the agents sharing "
            "this board (never user-facing), managed with one action: post, read, update, or "
            "remove. action=post adds an entry (claim a task, leave a note, flag an open "
            "question). action=read re-checks the board — it's usually shown at the start of "
            "each turn, so call this only after doing other work. action=update edits an "
            "entry's text and/or status by entry_id. action=remove deletes an entry by "
            "entry_id (clear done/stale items)."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["post", "read", "update", "remove"],
                    "description": "Operation to perform.",
                },
                "text": {
                    "type": "string",
                    "description": "Required for action=post: the entry content. Optional new text for action=update.",
                },
                "kind": {
                    "type": "string",
                    "enum": list(kinds),
                    "description": f"For action=post: entry kind. Default '{default_kind}'.",
                },
                "status": {
                    "type": "string",
                    "enum": list(statuses),
                    "description": "Optional status (e.g. for tasks) — action=post/update.",
                },
                "entry_id": {
                    "type": "integer",
                    "description": "Required for action=update/remove: the entry id (#n).",
                },
            },
            "required": ["action"],
        },
        required_params=("action",),
    )

    actions = {
        "read": _do_read,
        "post": _make_post(kinds, statuses, default_kind),
        "update": _make_update(statuses),
        "remove": _do_remove,
    }

    async def _board_handler(**kw: Any) -> str:
        operation = str(kw.get("action") or "").strip().lower()
        do = actions.get(operation)
        if do is None:
            return "Error: board action must be one of: post, read, update, remove."
        resolved = _resolve_board()
        if resolved is None:
            return _NO_BOARD
        board, board_id = resolved
        return await do(board, board_id, kw)

    registry.register(board_def, _board_handler, overwrite=overwrite)
