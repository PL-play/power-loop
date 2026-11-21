"""Zero-hot-path-SQL Three-Line LLM Format Self-Healing & History Sanitizer (Design 217 §7.4).

Line 1 (Takeover persistent stitching):
  - ``find_dangling_tool_calls`` + ``build_interrupted_turn_repairs``
  - Calls optional ``on_repair_interrupted_turn`` hook (seam for M1c ``HookPoint.TURN_REPAIR``)
  - Synthesizes closed ``tool`` rows carrying ``meta._assistant_seq`` (S09-R5) and
    ``round_index`` (S09-R6).

Line 2 (Pre-LLM pure in-memory sanitizer, 0 SQL):
  - ``align_tool_calls`` (preserves parallel ``(messages, seqs, ords)`` alignment)
  - ``sanitize_request_messages`` / ``sanitize_llm_messages``: normalizes mid-history
    ``system``/``system_notice``, drops empty assistant turns, backfills/deduplicates
    ``tool_call_id``, repairs malformed JSON arguments, pairs/closes dangling ``tool_calls``,
    merges consecutive same-role messages, strips internal ``_source_send_index``.

Line 3 (HTTP 400 fallback rewrite & single retry):
  - ``fold_tool_rounds_by_send_index`` / ``fold_older_tool_rounds_for_retry``: folds structured
    tool rounds attributed to each turn's ORIGINAL ``send_index`` (Probe 10), never squashing
    multiple historical turns into ``send_index - 1``.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Awaitable, Callable
from typing import Any

LoopMessage = dict[str, Any]
RepairInterruptedHook = Callable[[str, list[dict[str, Any]]], Awaitable[dict[str, str]]]

_PLACEHOLDER_RESULT = "[执行中断，结果未知 / tool result missing — repaired]"


# ── Line 1: Takeover persistent stitching (S09-R5 / S09-R6 / TURN_REPAIR seam) ──


def find_dangling_tool_calls(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Find assistant ``tool_calls`` with no matching subsequent ``tool`` result message.

    Preserves originating ``_assistant_seq`` and ``_assistant_round_index`` on returned dicts
    when available on the assistant message (for S09-R5 / S09-R6).
    """
    pending: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for msg in messages:
        role = msg.get("role")
        if role == "assistant":
            tcs = msg.get("tool_calls") or []
            if isinstance(tcs, str):
                try:
                    tcs = json.loads(tcs)
                except ValueError:
                    tcs = []
            asst_seq = msg.get("seq")
            asst_round = msg.get("round_index", 0)
            for tc in tcs:
                if isinstance(tc, dict) and tc.get("id"):
                    cid = str(tc["id"])
                    if cid not in pending:
                        tc_copy = dict(tc)
                        if asst_seq is not None:
                            tc_copy["_assistant_seq"] = int(asst_seq)
                        tc_copy["_assistant_round_index"] = int(asst_round or 0)
                        pending[cid] = tc_copy
                        order.append(cid)
        elif role == "tool":
            cid = msg.get("tool_call_id")
            if cid and str(cid) in pending:
                del pending[str(cid)]
    return [pending[cid] for cid in order if cid in pending]


async def build_interrupted_turn_repairs(
    session_id: str,
    messages: list[dict[str, Any]],
    *,
    on_repair_interrupted_turn: RepairInterruptedHook | None = None,
) -> list[dict[str, Any]]:
    """Line 1: Takeover recovery hook seam (M1c ``TURN_REPAIR``) + closure for dangling calls.

    1. Calls optional host hook ``on_repair_interrupted_turn(session_id, dangling_calls)``.
    2. For any remaining unresolved call, synthesizes a closed ``tool`` message carrying
       ``meta._assistant_seq`` (S09-R5) and ``round_index`` (S09-R6).
    """
    dangling = find_dangling_tool_calls(messages)
    if not dangling:
        return []
    recovered: dict[str, str] = {}
    if on_repair_interrupted_turn is not None:
        result = await on_repair_interrupted_turn(session_id, dangling)
        if result:
            recovered = {str(k): str(v) for k, v in result.items()}
    repairs: list[dict[str, Any]] = []
    for tc in dangling:
        cid = str(tc["id"])
        fn = (tc.get("function") or {}).get("name") or tc.get("name") or "unknown_tool"
        if cid in recovered:
            content = recovered[cid]
            source = "host_recovered"
        else:
            content = f"<interrupted: 执行中断，结果未知> tool '{fn}' interrupted before result was persisted"
            source = "sdk_auto_closed"
        meta: dict[str, Any] = {"interrupted": True, "repaired": True, "repair_source": source}
        if "_assistant_seq" in tc:
            meta["_assistant_seq"] = int(tc["_assistant_seq"])
        repairs.append({
            "role": "tool",
            "tool_call_id": cid,
            "name": fn,
            "content": content,
            "round_index": int(tc.get("_assistant_round_index") or 0),
            "meta": meta,
        })
    return repairs


# ── Line 2: Pure in-memory pre-LLM sanitizer (0 SQL) ────────────────────────────


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
                if s is not None:
                    dropped_seqs.append(int(s))
            continue
        if open_ids:
            _flush_placeholders()
        out_m.append(m)
        out_s.append(s)
        out_o.append(o)
        if role == "assistant":
            open_ids = _tool_call_ids(m)
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


def _repair_json_args(raw: Any) -> str:
    if isinstance(raw, dict):
        return json.dumps(raw, ensure_ascii=False, sort_keys=True)
    if not isinstance(raw, str):
        return json.dumps({"_invalid_arguments": str(raw)}, ensure_ascii=False)
    s = raw.strip()
    if not s:
        return "{}"
    try:
        parsed = json.loads(s)
        if isinstance(parsed, dict):
            return raw
        return json.dumps({"_value": parsed}, ensure_ascii=False)
    except ValueError:
        return json.dumps({"_invalid_arguments": raw}, ensure_ascii=False)


def sanitize_request_messages(messages: list[LoopMessage], *, strict: bool = False) -> list[LoopMessage]:
    """Line 2: Repair a model request purely in memory (0 SQL), without changing DB history.

    Guarantees a strictly protocol-valid sequence:
      1. Mid-history ``system`` or ``system_notice`` messages become safe ``user`` text.
      2. Drops empty ``assistant`` messages (no content & no ``tool_calls``).
      3. Backfills missing ``tool_call_id``, deduplicates duplicate IDs across the request,
         repairs malformed JSON arguments.
      4. Pairs every ``assistant(tool_calls)`` with matching ``tool`` results; drops
         orphan/duplicate ``tool`` rows; auto-closes dangling calls (including at the tail).
      5. Merges consecutive same-role messages (never merging ``tool`` messages, and never
         merging an ``assistant(tool_calls)`` with an adjacent plain ``assistant`` improperly).
      6. Strips internal ``_source_send_index`` tagging before sending to provider.
    """
    used_ids: set[str] = set()
    normalized: list[LoopMessage] = []
    auto_seq = 0
    for index, original in enumerate(messages):
        msg = copy.deepcopy(original)
        msg.pop("_source_send_index", None)
        role = msg.get("role")
        meta = msg.get("meta") if isinstance(msg.get("meta"), dict) else {}

        is_notice = role == "system_notice" or meta.get("kind") == "system_notice"
        if is_notice:
            raw_content = msg.get("content")
            if isinstance(raw_content, str):
                text = raw_content.strip()
                if not text:
                    continue
                if not text.startswith("[system_notice"):
                    text = f"[system_notice · {text}]"
                msg = {"role": "user", "content": text}
            else:
                if not raw_content:
                    continue
                msg = {"role": "user", "content": _join_content("[system_notice]", raw_content)}
            role = "user"
        elif role == "system" and normalized and normalized[-1]["role"] != "system":
            msg["role"] = "user"
            msg["content"] = _join_content("[历史系统说明]", msg.get("content"))
            role = "user"

        if role == "assistant":
            tcs_raw = msg.get("tool_calls")
            if isinstance(tcs_raw, str):
                try:
                    tcs_raw = json.loads(tcs_raw)
                except ValueError:
                    tcs_raw = []
            calls: list[dict[str, Any]] = []
            for pos, raw in enumerate(tcs_raw or []):
                if not isinstance(raw, dict):
                    continue
                fn_raw = raw.get("function")
                if isinstance(fn_raw, dict):
                    function = dict(fn_raw)
                elif raw.get("name"):
                    function = {"name": raw["name"], "arguments": raw.get("arguments", "{}")}
                else:
                    continue
                if not function.get("name"):
                    continue
                cid = str(raw.get("id") or "").strip()
                if not cid:
                    auto_seq += 1
                    cid = f"repaired_{index}_{pos}"
                    while cid in used_ids:
                        cid += "_"
                elif cid in used_ids:
                    auto_seq += 1
                    cid = f"{cid}_dedup_{auto_seq}"
                used_ids.add(cid)
                function["arguments"] = _repair_json_args(function.get("arguments", "{}"))
                calls.append({**raw, "id": cid, "type": "function", "function": function})
            if calls:
                msg["tool_calls"] = calls
            else:
                msg.pop("tool_calls", None)
                content = msg.get("content")
                if not content or (isinstance(content, str) and not content.strip()):
                    continue
        normalized.append(msg)

    aligned, _, _, _, _ = align_tool_calls(normalized, [], [], close_trailing=True)
    if strict:
        # Keep the latest tool round after the latest user input. Older tool
        # exchanges become ordinary descriptions, retaining their actual results.
        boundary = max(
            (i for i, msg in enumerate(aligned) if msg["role"] == "user" or msg.get("tool_calls")),
            default=0,
        )
        for msg in aligned[:boundary]:
            if msg.get("tool_calls"):
                descriptions = [
                    f"[tool call {tc['id']}] {tc['function']['name']}({tc['function']['arguments']})"
                    for tc in msg.pop("tool_calls")
                ]
                msg["content"] = _join_content(msg.get("content"), "\n".join(descriptions))
            elif msg["role"] == "tool":
                msg["role"] = "assistant"
                msg["content"] = _join_content(f"[tool result {msg.pop('tool_call_id')}]", msg.get("content"))

    out: list[LoopMessage] = []
    for msg in aligned:
        if (
            out
            and msg["role"] == out[-1]["role"]
            and msg["role"] != "tool"
            and not (msg["role"] == "assistant" and out[-1].get("tool_calls"))
        ):
            previous = out[-1]
            previous["content"] = _join_content(previous.get("content"), msg.get("content"))
            if msg.get("tool_calls"):
                previous["tool_calls"] = msg["tool_calls"]
            if msg.get("reasoning_content"):
                previous["reasoning_content"] = _join_content(
                    previous.get("reasoning_content"), msg["reasoning_content"]
                )
        else:
            out.append(msg)
    return out


def sanitize_llm_messages(
    raw_messages: list[LoopMessage],
    *,
    strip_reasoning: bool = True,
    keep_last_turn_reasoning: bool = False,
) -> list[LoopMessage]:
    """Alias/wrapper over :func:`sanitize_request_messages` with optional reasoning stripping."""
    last_user_idx = -1
    if strip_reasoning and keep_last_turn_reasoning:
        for i in range(len(raw_messages) - 1, -1, -1):
            r = raw_messages[i].get("role")
            m_meta = raw_messages[i].get("meta") if isinstance(raw_messages[i].get("meta"), dict) else {}
            if r in ("user", "system_notice") or m_meta.get("kind") == "system_notice" or (r == "system" and i > 0):
                last_user_idx = i
                break
    prepared: list[LoopMessage] = []
    for idx, orig in enumerate(raw_messages):
        msg = copy.deepcopy(orig)
        if strip_reasoning and not (keep_last_turn_reasoning and last_user_idx != -1 and idx > last_user_idx):
            msg.pop("reasoning_content", None)
        prepared.append(msg)
    return sanitize_request_messages(prepared)


# ── Line 3: HTTP 400 per-send_index folding & retry (Probe 10) ──────────────────


def fold_older_tool_rounds_for_retry(
    messages: list[LoopMessage],
    *,
    historical_end_idx: int | None = None,
) -> tuple[list[LoopMessage], str, str]:
    """Defense 3 fallback: fold tool rounds into plain text safely.

    Returns ``(repaired_messages, historical_folded_text, current_turn_folded_text)``.
    Folds historical completed turns first; only if none exist does it fold active-turn
    tool rounds into ``current_turn_folded_text`` (never overwriting ``send_index - 1``).
    """
    sanitized = sanitize_llm_messages(messages, strip_reasoning=True)
    last_user_idx = -1
    for i in range(len(sanitized) - 1, -1, -1):
        if sanitized[i]["role"] == "user":
            last_user_idx = i
            break
    hist_boundary = (
        min(historical_end_idx, len(sanitized))
        if historical_end_idx is not None
        else (last_user_idx if last_user_idx != -1 else 0)
    )

    def _fold(fold_active_turn: bool) -> tuple[list[LoopMessage], list[str]]:
        result: list[LoopMessage] = []
        folded_lines: list[str] = []
        i = 0
        while i < len(sanitized):
            m = sanitized[i]
            is_leading_sys = i == 0 and m["role"] == "system"
            in_active_turn = i >= hist_boundary
            keep_unfolded = is_leading_sys or (not fold_active_turn and in_active_turn)
            if keep_unfolded:
                result.append(m)
                i += 1
                continue
            if m["role"] == "assistant" and m.get("tool_calls"):
                call_names = {
                    tc["id"]: tc["function"]["name"]
                    for tc in m["tool_calls"]
                    if isinstance(tc, dict) and tc.get("id")
                }
                summary_parts: list[str] = []
                if m.get("content"):
                    summary_parts.append(str(m["content"]).strip())
                j = i + 1
                while j < len(sanitized) and sanitized[j]["role"] == "tool":
                    tm = sanitized[j]
                    tname = tm.get("name") or call_names.get(tm.get("tool_call_id", ""), "tool")
                    snippet = str(tm.get("content") or "").replace("\n", " ")[:120]
                    summary_parts.append(f"[folded_tool:{tname}] {snippet}")
                    j += 1
                folded_text = "\n".join(summary_parts) or "[folded_tool_round]"
                folded_lines.append(folded_text)
                result.append({"role": "assistant", "content": folded_text})
                i = j
                continue
            result.append(m)
            i += 1
        return result, folded_lines

    result, hist_folded = _fold(fold_active_turn=False)
    cur_folded: list[str] = []
    if not hist_folded:
        result, cur_folded = _fold(fold_active_turn=True)
    cleaned = sanitize_llm_messages(result, strip_reasoning=True)
    return cleaned, "\n".join(hist_folded), "\n".join(cur_folded)


def fold_tool_rounds_by_send_index(
    messages: list[LoopMessage],
    *,
    current_send_index: int,
    historical_end_idx: int | None = None,
) -> tuple[list[LoopMessage], dict[int, str], str]:
    """Line 3 (per-send_index): Fold historical tool rounds attributed to their ORIGINAL ``send_index``.

    Never squashes multiple historical turns into ``send_index - 1`` (Probe 10)!
    Messages before ``historical_end_idx`` with structured ``tool_calls`` are folded into
    ``folded_by_send[orig_send_index]`` using each message's ``_source_send_index``.
    If no historical structured tool rounds exist, falls back to folding active-turn tool rounds
    into ``cur_folded_text`` (for ``current_send_index``).
    """
    hist_boundary = historical_end_idx if historical_end_idx is not None else 0
    folded_by_send: dict[int, list[str]] = {}
    out: list[LoopMessage] = []
    i = 0
    while i < len(messages):
        m = messages[i]
        in_hist = i < hist_boundary
        if in_hist and m.get("role") == "assistant" and m.get("tool_calls"):
            src_si = int(m.get("_source_send_index") or max(1, current_send_index - 1))
            tcs = m.get("tool_calls") or []
            call_names = {
                str(tc.get("id")): (tc.get("function") or {}).get("name") or tc.get("name") or "tool"
                for tc in tcs
                if isinstance(tc, dict) and tc.get("id")
            }
            parts: list[str] = []
            if m.get("content"):
                parts.append(str(m["content"]).strip())
            j = i + 1
            while j < len(messages) and messages[j].get("role") == "tool":
                tm = messages[j]
                tname = tm.get("name") or call_names.get(str(tm.get("tool_call_id") or ""), "tool")
                snippet = str(tm.get("content") or "").replace("\n", " ")[:120]
                parts.append(f"[folded_tool:{tname}] {snippet}")
                j += 1
            ftxt = "\n".join(parts) or "[folded_tool_round]"
            folded_by_send.setdefault(src_si, []).append(ftxt)
            out.append({"role": "assistant", "content": ftxt, "_source_send_index": src_si})
            i = j
            continue
        out.append(m)
        i += 1

    if folded_by_send:
        cleaned = sanitize_llm_messages(out, strip_reasoning=True)
        return cleaned, {si: "\n".join(lines) for si, lines in folded_by_send.items()}, ""

    cleaned, _hist_txt, cur_txt = fold_older_tool_rounds_for_retry(
        messages, historical_end_idx=historical_end_idx
    )
    return cleaned, {}, cur_txt
