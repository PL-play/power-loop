"""Token budgeting utilities — heuristic, vendor-neutral.

These are deliberately approximate: we want a single number that's cheap to
compute, monotonic with content size, and works without a tokenizer
dependency. Used by the compactor to decide *when* to trigger; not used for
billing.

Rule of thumb: ~4 chars per token for English-heavy LLM transcripts. Adjust
via :data:`CHARS_PER_TOKEN` if needed.
"""

from __future__ import annotations

import json
from typing import Any

CHARS_PER_TOKEN = 4
"""Approximate chars-per-token used by :func:`estimate_tokens`."""


def estimate_text_tokens(text: str | None) -> int:
    if not text:
        return 0
    return max(1, len(text) // CHARS_PER_TOKEN)


def estimate_message_tokens(message: dict[str, Any]) -> int:
    """Heuristic token count for a single message dict.

    Counts content (string or JSON-serialized non-string), tool_calls
    arguments, name fields, and a small per-message overhead for the role
    framing.
    """
    overhead = 4  # role tag + delimiters
    content = message.get("content")
    if isinstance(content, str):
        body = content
    elif content is None:
        body = ""
    else:
        body = json.dumps(content, ensure_ascii=False)
    body += str(message.get("reasoning_content") or "")
    tool_calls = message.get("tool_calls") or []
    if tool_calls:
        body += json.dumps(tool_calls, ensure_ascii=False)
    name = message.get("name") or ""
    return overhead + estimate_text_tokens(body) + estimate_text_tokens(name)


def estimate_tokens(messages: list[dict[str, Any]]) -> int:
    return sum(estimate_message_tokens(m) for m in messages)


def _assistant_tool_call_ids(message: dict[str, Any]) -> list[str]:
    """Return declared non-empty tool_call IDs for an assistant message."""
    if message.get("role") != "assistant":
        return []
    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list) or not tool_calls:
        return []
    ids: list[str] = []
    for tc in tool_calls:
        if isinstance(tc, dict):
            tid = tc.get("id")
            if isinstance(tid, str) and tid:
                ids.append(tid)
    return ids


def _partition_tool_groups(
    messages: list[dict[str, Any]],
) -> list[tuple[list[dict[str, Any]], bool]]:
    """Split a message slice into atomic trim units.

    Each element is ``(unit_messages, is_valid_unit)``:
    - Any message without valid ``tool_calls`` is a single valid unit.
    - An ``assistant`` message with valid ``tool_calls`` is grouped with all
      immediately following ``role == "tool"`` messages whose ``tool_call_id``
      matches one of its declared call IDs. ``is_valid_unit`` is True iff every
      declared call ID is answered within this contiguous group.
    """
    units: list[tuple[list[dict[str, Any]], bool]] = []
    i = 0
    n = len(messages)
    while i < n:
        call_ids = _assistant_tool_call_ids(messages[i])
        if not call_ids:
            units.append(([messages[i]], True))
            i += 1
            continue

        expected = set(call_ids)
        answered: set[str] = set()
        j = i + 1
        while j < n:
            m = messages[j]
            if m.get("role") != "tool":
                break
            tid = m.get("tool_call_id")
            if not isinstance(tid, str) or tid not in expected:
                break
            answered.add(tid)
            j += 1
        units.append((messages[i:j], expected.issubset(answered)))
        i = j
    return units


def trim_history(
    messages: list[dict[str, Any]],
    max_tokens: int,
    *,
    keep_system: bool = True,
    keep_last_n: int = 2,
) -> list[dict[str, Any]]:
    """Drop messages from the middle until the token budget fits.

    Used by callers that want to cap context before calling the LLM
    directly (without the full compactor pipeline). This is a **pure
    trim** — no summarisation, no LLM calls. For LLM-backed summary
    compaction, use :class:`power_loop.runtime.compact.DefaultCompactor`.

    Preserves:
    * All ``role=system`` messages at the front (when ``keep_system=True``).
    * The last ``keep_last_n`` exchanges (user-bounded). An exchange is
      a ``user`` message followed by everything up to (but not including)
      the next ``user`` — so ``assistant(tool_calls) ↔ tool`` pairs stay
      together.
    * The ``assistant(tool_calls)`` ↔ ``tool`` atomic pair near the cut
      boundary: if the first removed message is a ``tool``, we walk back
      to include the preceding ``assistant(tool_calls)`` so the protocol
      isn't broken.

    Returns a new list (does not mutate the input). If the budget
    already fits, the original list is returned unchanged.
    """
    if not messages or max_tokens <= 0:
        return []

    if estimate_tokens(messages) <= max_tokens:
        return list(messages)

    # Partition: leading system block, body, tail.
    n = len(messages)
    sys_end = 0
    if keep_system:
        while sys_end < n and messages[sys_end].get("role") == "system":
            sys_end += 1

    # Find tail exchanges (keep_last_n) by walking backwards from the end.
    exchanges = 0
    tail_start = n
    for i in range(n - 1, sys_end - 1, -1):
        if messages[i].get("role") == "user":
            exchanges += 1
            if exchanges >= keep_last_n:
                tail_start = i
                break
    else:
        tail_start = sys_end  # not enough exchanges — keep everything after system

    system_tokens = estimate_tokens(messages[:sys_end])
    tail_tokens = estimate_tokens(messages[tail_start:])

    # Walk the cut boundary back to preserve a complete assistant(tool_calls) ↔ tool
    # group immediately preceding tail_start, without letting an older group displace
    # the leading system block or exceed max_tokens on its own.
    while tail_start > sys_end and messages[tail_start - 1].get("role") == "tool" \
            and isinstance(messages[tail_start - 1].get("tool_call_id"), str) \
            and messages[tail_start - 1].get("tool_call_id"):
        tid = messages[tail_start - 1]["tool_call_id"]
        extended = tail_start - 1
        matched_ids: list[str] = []
        while extended > sys_end:
            extended -= 1
            role = messages[extended].get("role")
            if role == "tool":
                continue
            call_ids = _assistant_tool_call_ids(messages[extended])
            if call_ids and tid in call_ids:
                matched_ids = call_ids
            break
        if not matched_ids:
            break  # safety: cannot find contiguous pair, stop extending
        expected = set(matched_ids)
        group_slice = messages[extended:tail_start]
        answered = {
            m.get("tool_call_id")
            for m in group_slice[1:]
            if m.get("role") == "tool" and isinstance(m.get("tool_call_id"), str)
        }
        if not expected.issubset(answered) or any(
            m.get("role") != "tool" or m.get("tool_call_id") not in expected
            for m in group_slice[1:]
        ):
            break
        group_tokens = estimate_tokens(group_slice)
        if system_tokens + tail_tokens <= max_tokens:
            if system_tokens + group_tokens + tail_tokens > max_tokens:
                break
        elif group_tokens + tail_tokens > max_tokens:
            break
        tail_start = extended
        tail_tokens += group_tokens

    # Build result: system + trimmed body + tail.
    budget_tail = tail_tokens
    remaining = max_tokens - budget_tail
    body_budget = remaining - system_tokens

    if body_budget < 0:
        tail = messages[tail_start:]
        if not tail:
            # No tail at all (e.g. an all-system history that exceeds budget).
            # Never return an empty list for a non-empty input: keep the system
            # block (contract: system is preserved when keep_system=True; dropping
            # it would lose the prompt entirely and break the next LLM call).
            return list(messages[:sys_end]) if sys_end > 0 else list(messages)
        # Can't even fit system + tail. Degrade: drop system, keep only tail.
        if budget_tail > max_tokens:
            # Tail alone exceeds budget — last resort: trim from tail end too,
            # keeping each assistant(tool_calls) + full tool-result group atomic.
            kept_units: list[list[dict[str, Any]]] = []
            spent = 0
            for unit, is_valid in reversed(_partition_tool_groups(tail)):
                if not is_valid:
                    break
                cost = estimate_tokens(unit)
                if spent + cost > max_tokens:
                    break
                kept_units.append(unit)
                spent += cost
            kept_units.reverse()
            return [m for unit in kept_units for m in unit]
        return list(tail)

    # Take body messages from index sys_end to tail_start, keeping as many
    # complete units from the front as fit so an assistant(tool_calls) message
    # is never separated from its full set of tool results.
    body: list[dict[str, Any]] = []
    spent = system_tokens
    limit = max_tokens - budget_tail
    for unit, is_valid in _partition_tool_groups(messages[sys_end:tail_start]):
        if not is_valid:
            break
        cost = estimate_tokens(unit)
        if spent + cost > limit:
            break
        body.extend(unit)
        spent += cost

    return list(messages[:sys_end]) + body + list(messages[tail_start:])
