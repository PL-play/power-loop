"""Assembled-history robustness backstop.

Providers reject a structurally-invalid message list with a hard error: an assistant turn
that declares ``tool_calls`` whose ids are not each answered by a following ``tool`` message,
or a ``tool`` message with no matching call. If a corrupt row ever lands in ``pl_messages`` —
a crash between writing the assistant row and its tool row, a bad import, a manual edit, a
projection-table mismatch — that error would repeat on every load and **brick the session
forever** (one bad message ruins the whole conversation).

:func:`align_tool_calls` repairs the *assembled prompt* (not the audit log) so it is always
structurally valid, regardless of what is in the DB:

* an **orphan tool result** (a ``tool`` message whose id no open assistant call matches) is
  **dropped** — its real ``seq`` is reported so the caller can optionally deactivate the row;
* a **mid-history assistant call left unanswered** before the next non-tool message gets a
  **synthesized placeholder** ``tool`` result, so the pairing is valid.

It operates on the parallel ``(messages, seqs, ords)`` lists the loop already maintains and
keeps them aligned (synthesized rows carry ``seq``/``ord`` = ``None``). It is **pure**, never
raises, and is a **no-op on a healthy history** (so it changes nothing in the common case).

Deliberately conservative: it does NOT touch the TRAILING unanswered call (the in-flight
send's pending tools, which the loop is about to execute on a resume) — only calls followed
by another message are repaired.
"""

from __future__ import annotations

import json
from typing import Any

LoopMessage = dict[str, Any]

_PLACEHOLDER_RESULT = "[执行中断，结果未知 / tool result missing — repaired]"


def _tool_call_ids(message: LoopMessage) -> list[str]:
    """The ids an assistant message declares (robust to a malformed/non-dict tool_call)."""
    out: list[str] = []
    for tc in message.get("tool_calls") or []:
        if isinstance(tc, dict) and tc.get("id"):
            out.append(str(tc["id"]))
    return out


def align_tool_calls(
    messages: list[LoopMessage],
    seqs: list[int | None],
    ords: list[int | None],
    *,
    close_trailing: bool = False,
) -> tuple[list[LoopMessage], list[int | None], list[int | None], list[int], int]:
    """Repair tool-call/result pairing in an assembled history. Returns
    ``(messages, seqs, ords, dropped_seqs, synthesized)`` where ``dropped_seqs`` are the real
    ``seq``\\ s of orphan ``tool`` rows removed (for optional DB repair) and ``synthesized`` is
    the count of placeholder results inserted. The three parallel lists stay length-aligned.
    Pure; never raises; a no-op (identical lists, empty drops, 0 synthesized) on a valid history."""
    out_m: list[LoopMessage] = []
    out_s: list[int | None] = []
    out_o: list[int | None] = []
    dropped_seqs: list[int] = []
    synthesized = 0
    # Ids declared by the most recent assistant turn that are still awaiting a result, in order.
    open_ids: list[str] = []

    def _flush_placeholders() -> None:
        nonlocal synthesized
        for cid in open_ids:
            out_m.append({"role": "tool", "tool_call_id": cid, "content": _PLACEHOLDER_RESULT})
            out_s.append(None)
            out_o.append(None)
            synthesized += 1
        open_ids.clear()

    n = len(messages)
    for idx in range(n):
        m = messages[idx]
        s = seqs[idx] if idx < len(seqs) else None
        o = ords[idx] if idx < len(ords) else None
        role = m.get("role")
        if role == "tool":
            cid = m.get("tool_call_id")
            cid = str(cid) if cid is not None else None
            if cid is not None and cid in open_ids:
                out_m.append(m)
                out_s.append(s)
                out_o.append(o)
                open_ids.remove(cid)
            else:
                # Orphan tool result — no open call matches. Drop it (it would 400 the provider);
                # record the real seq so the caller can optionally deactivate the row.
                if s is not None:
                    dropped_seqs.append(int(s))
            continue
        # A non-tool message follows: any still-open calls are unanswered mid-history → close
        # them with placeholders BEFORE this message so the pairing is valid.
        if open_ids:
            _flush_placeholders()
        out_m.append(m)
        out_s.append(s)
        out_o.append(o)
        if role == "assistant":
            open_ids = _tool_call_ids(m)
    # Loading history may precede pending execution. At the model boundary there
    # are no in-flight tools, so unanswered trailing calls must be closed too.
    if close_trailing:
        _flush_placeholders()
    return out_m, out_s, out_o, dropped_seqs, synthesized


def _join_content(left: Any, right: Any) -> Any:
    if not left:
        return right
    if not right:
        return left
    if isinstance(left, str) and isinstance(right, str):
        return left + "\n\n" + right
    def blocks(value: Any) -> list:
        return value if isinstance(value, list) else [{"type": "text", "text": value}]
    return [*blocks(left), *blocks(right)]


def sanitize_request_messages(messages: list[LoopMessage], *, strict: bool = False) -> list[LoopMessage]:
    """Repair a model request in memory, without changing history or pending work.

    Call only after pending tools have stopped/executed. Tool results remain
    separate messages even when adjacent: each answers its own call id.
    """
    used_ids = {str(tc["id"]) for m in messages for tc in m.get("tool_calls") or []
                if isinstance(tc, dict) and tc.get("id")}
    normalized = []
    for index, original in enumerate(messages):
        msg = dict(original)
        if msg.get("role") == "system" and normalized and normalized[-1]["role"] != "system":
            msg["role"] = "user"
            msg["content"] = _join_content("[历史系统说明]", msg.get("content"))
        if msg.get("role") == "assistant":
            calls, seen = [], set()
            for pos, raw in enumerate(msg.get("tool_calls") or []):
                # These are damaged stored/provider messages, not trusted internal state.
                if not isinstance(raw, dict) or not isinstance(raw.get("function"), dict):
                    continue
                function = dict(raw["function"])
                if not function.get("name"):
                    continue
                cid = str(raw.get("id") or "")
                if not cid:
                    cid = f"repaired_{index}_{pos}"
                    while cid in used_ids:
                        cid += "_"
                    used_ids.add(cid)
                if cid in seen:
                    continue
                seen.add(cid)
                args = function.get("arguments", "{}")
                if isinstance(args, str):
                    try:
                        json.loads(args)
                    except ValueError:
                        args = json.dumps({"_invalid_arguments": args}, ensure_ascii=False)
                else:
                    args = json.dumps(args, ensure_ascii=False)
                function["arguments"] = args
                calls.append({**raw, "id": cid, "type": "function", "function": function})
            if calls:
                msg["tool_calls"] = calls
            else:
                msg.pop("tool_calls", None)
                if not msg.get("content") or (isinstance(msg["content"], str) and not msg["content"].strip()):
                    continue
        normalized.append(msg)

    aligned, _, _, _, _ = align_tool_calls(normalized, [], [], close_trailing=True)
    if strict:
        # Keep the latest tool round after the latest user input. Older tool
        # exchanges become ordinary descriptions, retaining their actual results.
        boundary = max((i for i, msg in enumerate(aligned)
                        if msg["role"] == "user" or msg.get("tool_calls")), default=0)
        for msg in aligned[:boundary]:
            if msg.get("tool_calls"):
                descriptions = [f"[tool call {tc['id']}] {tc['function']['name']}({tc['function']['arguments']})"
                                for tc in msg.pop("tool_calls")]
                msg["content"] = _join_content(msg.get("content"), "\n".join(descriptions))
            elif msg["role"] == "tool":
                msg["role"] = "assistant"
                msg["content"] = _join_content(f"[tool result {msg.pop('tool_call_id')}]", msg.get("content"))
    out: list[LoopMessage] = []
    for msg in aligned:
        if out and msg["role"] == out[-1]["role"] and msg["role"] != "tool":
            previous = out[-1]
            previous["content"] = _join_content(previous.get("content"), msg.get("content"))
            if msg.get("tool_calls"):
                previous["tool_calls"] = msg["tool_calls"]
            if msg.get("reasoning_content"):
                previous["reasoning_content"] = _join_content(
                    previous.get("reasoning_content"), msg["reasoning_content"])
        else:
            out.append(msg)
    return out
