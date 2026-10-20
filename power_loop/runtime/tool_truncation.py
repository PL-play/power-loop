"""Opt-in independent text previews; originals remain in the audit store."""
from __future__ import annotations

import json

from power_loop.contracts.context_policy import ToolContextPolicy


def truncate_pair(raw: str, result: str, policy: ToolContextPolicy, recall: str) -> tuple[str, str] | None:
    args = json.loads(raw)
    if not isinstance(args, dict) or '_distilled' in args:
        return None
    description = args.get('description', '')
    output = result
    # Bash archives can expose a bounded preview; only use a complete capture.
    try:
        receipt = json.loads(result)
    except (ValueError, TypeError):
        receipt = None
    if isinstance(receipt, dict) and 'capture_complete' in receipt:
        if not receipt.get('capture_complete') or receipt.get('truncated') or not isinstance(receipt.get('output'), str):
            return None
        output = receipt['output']
        if receipt.get('output_chars', len(output)) != len(output):
            return None
        description = description or receipt.get('description', '')
    a, r = policy.argument_max_chars, policy.result_max_chars
    clip_a = policy.argument_truncation_enabled and a is not None and len(raw) > a
    clip_r = policy.result_truncation_enabled and r is not None and len(output) > r
    if not clip_a and not clip_r:
        return None

    def preview(text, limit, label):
        return json.dumps({'description': description, 'preview': text[:limit],
            'notice': f'{label}已截断，仅保留前{limit}字符（原{len(text)}字符）；不是完整原文。',
            '_distilled': recall}, ensure_ascii=False, separators=(',', ':'))

    return (preview(raw, a, '参数') if clip_a else raw,
            preview(output, r, '输出') if clip_r else result)
