"""``board_*`` tools — agents read/write the shared blackboard (the pull side).

These resolve the live :class:`~power_loop.runtime.blackboard.Blackboard` and the
agent's ``board_id`` from the current :class:`~power_loop.runtime.env.RuntimeEnv`
at call time (so a host injects the implementation + id per run/send, exactly
like the sandbox ``ShellBackend``). The author is taken from the session
metadata. ``kind``/``status`` vocabularies are the host's policy, set when the
tools are registered.

Opt-in: a board-less agent simply doesn't register these (or, if it somehow
calls one with no board configured, gets a clear error rather than a crash).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from power_loop.contracts.tools import ToolDefinition
from power_loop.runtime.blackboard import Blackboard, BlackboardError, render_entries
from power_loop.runtime.env import get_runtime_env
from power_loop.runtime.runtime_state import get_tool_runtime_context

__all__ = ["register_blackboard_tools"]

_NO_BOARD = "Error: no shared board is configured for this agent."


def _resolve_board() -> tuple[Blackboard | None, str | None]:
    env = get_runtime_env()
    if env.blackboard is None or not env.blackboard_id:
        return None, None
    return env.blackboard, env.blackboard_id


def _author() -> str | None:
    ctx = get_tool_runtime_context()
    if ctx.store is not None and ctx.session_id:
        try:
            row = ctx.store.get_session(ctx.session_id)
            md = (getattr(row, "metadata", None) or {}) if row is not None else {}
            return md.get("spec_name") or md.get("name") or ctx.session_id
        except Exception:  # noqa: BLE001 — author is best-effort
            return ctx.session_id
    return ctx.session_id


async def _snapshot(board: Blackboard, board_id: str, *, header: str) -> str:
    entries = await board.read(board_id)
    return render_entries(entries, header=header, empty="(the board is empty)")


async def _board_read(**_kw: Any) -> str:
    board, board_id = _resolve_board()
    if board is None:
        return _NO_BOARD
    return await _snapshot(board, board_id, header="Shared board:")


def _make_post(kinds: tuple[str, ...], statuses: tuple[str, ...], default_kind: str):
    async def _board_post(**kw: Any) -> str:
        board, board_id = _resolve_board()
        if board is None:
            return _NO_BOARD
        text = str(kw.get("text") or "").strip()
        if not text:
            return "Error: 'text' is required."
        kind = kw.get("kind") or default_kind
        if kind not in kinds:
            return f"Error: kind must be one of {list(kinds)}."
        status = kw.get("status")
        if status is not None and status not in statuses:
            return f"Error: status must be one of {list(statuses)}."
        try:
            await board.post(board_id, text=text, kind=kind, status=status, author=_author())
        except BlackboardError as exc:
            return f"Error: {exc}"
        return await _snapshot(board, board_id, header="Posted. Shared board:")

    return _board_post


def _make_update(statuses: tuple[str, ...]):
    async def _board_update(**kw: Any) -> str:
        board, board_id = _resolve_board()
        if board is None:
            return _NO_BOARD
        try:
            entry_id = int(kw["entry_id"])
        except (KeyError, TypeError, ValueError):
            return "Error: 'entry_id' (integer) is required."
        status = kw.get("status")
        if status is not None and status not in statuses:
            return f"Error: status must be one of {list(statuses)}."
        try:
            await board.update(board_id, entry_id, text=kw.get("text"), status=status)
        except BlackboardError as exc:
            return f"Error: {exc}"
        return await _snapshot(board, board_id, header="Updated. Shared board:")

    return _board_update


async def _board_remove(**kw: Any) -> str:
    board, board_id = _resolve_board()
    if board is None:
        return _NO_BOARD
    try:
        entry_id = int(kw["entry_id"])
    except (KeyError, TypeError, ValueError):
        return "Error: 'entry_id' (integer) is required."
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
    """Register ``board_read`` / ``board_post`` / ``board_update`` / ``board_remove``.

    ``kinds`` / ``statuses`` set the validated vocabularies (and the tool schemas
    the model sees). The host must inject a ``Blackboard`` + ``blackboard_id`` on
    the per-send ``RuntimeEnv`` for the tools to operate.
    """
    read_def = ToolDefinition(
        name="board_read",
        description=(
            "Read the shared agent board — the private space where agents sharing "
            "this board coordinate. It's usually shown at the start of each turn; "
            "call this only to re-check after doing other work."
        ),
        input_schema={"type": "object", "properties": {}},
        required_params=(),
    )
    post_def = ToolDefinition(
        name="board_post",
        description=(
            "Add an entry to the shared agent board to coordinate with peer agents "
            "(claim a task, leave a note, flag an open question). Not user-facing."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "The entry content."},
                "kind": {"type": "string", "enum": list(kinds),
                         "description": f"Entry kind. Default '{default_kind}'."},
                "status": {"type": "string", "enum": list(statuses),
                           "description": "Optional status (e.g. for tasks)."},
            },
            "required": ["text"],
        },
        required_params=("text",),
    )
    update_def = ToolDefinition(
        name="board_update",
        description="Edit a board entry's text and/or status by its id.",
        input_schema={
            "type": "object",
            "properties": {
                "entry_id": {"type": "integer", "description": "The entry id (#n)."},
                "text": {"type": "string", "description": "New text (optional)."},
                "status": {"type": "string", "enum": list(statuses),
                           "description": "New status (optional)."},
            },
            "required": ["entry_id"],
        },
        required_params=("entry_id",),
    )
    remove_def = ToolDefinition(
        name="board_remove",
        description="Delete a board entry by its id (clear done/stale items).",
        input_schema={
            "type": "object",
            "properties": {"entry_id": {"type": "integer", "description": "The entry id (#n)."}},
            "required": ["entry_id"],
        },
        required_params=("entry_id",),
    )
    handlers: list[tuple[ToolDefinition, Callable[..., Awaitable[str]]]] = [
        (read_def, _board_read),
        (post_def, _make_post(kinds, statuses, default_kind)),
        (update_def, _make_update(statuses)),
        (remove_def, _board_remove),
    ]
    for definition, handler in handlers:
        registry.register(definition, handler, overwrite=overwrite)
