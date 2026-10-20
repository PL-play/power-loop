"""Version-2 deterministic projection. Raw rows are never modified.

The session runner serializes writes. Derived patches and exact group membership are
committed in the existing atomic runtime-state row BEFORE changing the in-memory view.
"""
from __future__ import annotations

import copy
import json
from dataclasses import replace

from power_loop.runtime.tool_context import ToolSource, _hash, _json


def archive_text(source: ToolSource) -> str:
    if source.name == 'bash':
        label = str(source.arguments.get('description') or '旧调用未提供描述')
    elif source.name == 'load_skill':
        label = str(source.arguments.get('name') or source.arguments.get('skill') or source.arguments.get('path') or '技能身份见原文')
        if source.arguments.get('doc'):
            label += '/' + str(source.arguments['doc'])
    else:
        label = ''
    return f'历史 {source.name} 已归档；正文未展示。{label}\n[{source.recall}]'


async def _derive(engine, source, trigger, level, raw_arguments=None):
    if not getattr(engine, '_implementation_matches', True):
        return None
    policy = engine.policy(source)
    if policy.hard_preserve:
        return None
    if source.name == 'background_run':
        targets = await engine._background_sources(source)
        if not targets:
            return None
        values = [await _derive(engine, t, trigger, level) for t in targets]
        if any(v is None for v in values):
            if trigger == 'between_sends':
                return _json(dict(source.arguments)), _json({
                    'receipt': source.result, 'targets': [
                        {'tool': t.name, 'task_id': t.call_id, 'arguments': dict(t.arguments), 'result': t.result}
                        for t in targets]})
            return None
        return _json({'_distilled': source.recall}), '\n'.join(
            f'{t.name} task={t.call_id}: {v[1]}' for t, v in zip(targets, values, strict=True))
    strategy = policy.general if level == 2 or trigger == 'between_sends' else policy.strategy(trigger)
    if (level == 1 and trigger != 'between_sends' and source.name == 'bash'
            and source.arguments.get('context_policy') == 'keep'):
        return None
    if strategy == 'preserve':
        return None
    if strategy == 'archive':
        return _json({'_distilled': source.recall}), archive_text(source)
    if strategy == 'truncate':
        from power_loop.runtime.tool_truncation import truncate_pair
        return truncate_pair(raw_arguments or _json(dict(source.arguments)), source.result, policy, source.recall)
    # Preserve extension points (rule / precomputed llm_summary), without recursively
    # applying a truncation to its own preview. The original source is always used.
    call_trigger = 'between_sends' if level == 2 else trigger
    effective = replace(policy, **{call_trigger: strategy})
    text = await engine.distill(source, call_trigger, policy_override=effective)
    return (_json({'_distilled': source.recall}), text) if text is not None else None


async def _commit(engine, patch):
    previous = copy.deepcopy(engine.state)
    engine.state.setdefault('archive_v2', {}).update(patch)
    try:
        await engine._save()
    except BaseException:
        engine.state = previous
        raise


def restore_history(engine, pipeline):
    """Reapply committed paired patches, idempotently, including a resumed send."""
    patches = engine.state.get('archive_v2', {}).get('patches', {})
    seqs = getattr(pipeline.sink, '_history_seqs', ()) or ()
    positions = {s: i for i, s in enumerate(seqs) if s is not None}
    grouped = {seq for g in engine.state.get('archive_v2', {}).get('view_groups', [])
               if g['send_index'] == pipeline.send_index for seq in g['seqs']}
    for patch in patches.values():
        if patch['call_seq'] in grouped:
            continue
        if patch['send_index'] != pipeline.send_index:
            continue
        ai, ri = positions.get(patch['call_seq']), positions.get(patch['result_seq'])
        if ai is None or ri is None:
            continue
        row = pipeline.history[ai]
        calls = list(row.get('tool_calls') or [])
        ci = next((i for i, c in enumerate(calls) if c.get('id') == patch['call_id']), None)
        result = pipeline.history[ri]
        if ci is None or result.get('tool_call_id') != patch['call_id']:
            continue
        call = calls[ci]
        if call['function'].get('arguments') == patch['arguments'] and result.get('content') == patch['result']:
            continue
        calls[ci] = {**call, 'function': {**call['function'], 'arguments': patch['arguments']}}
        pipeline.history[ai] = {**row, 'tool_calls': calls}
        pipeline.history[ri] = {**result, 'content': patch['result']}
        pipeline._tok_len = -1
    _restore_groups(engine, pipeline)


def _restore_groups(engine, pipeline):
    seqs = pipeline.sink._history_seqs
    for group in engine.state.get('archive_v2', {}).get('view_groups', []):
        if group['send_index'] != pipeline.send_index:
            continue
        consumed = group['seqs']
        if consumed[0] not in seqs or consumed[1] not in seqs:
            continue
        start = seqs.index(consumed[0])
        # First pair remains a real, complete call/result pair. Never fabricate a tool.
        row = pipeline.history[start]
        calls = row.get('tool_calls') or []
        if len(calls) != 1 or calls[0].get('id') != group['call_id']:
            continue
        end = start + len(consumed)
        if seqs[start:end] == consumed:
            del pipeline.history[start + 2:end]
            del seqs[start + 2:end]
            ords = getattr(pipeline.sink, '_history_ord', None)
            if ords is not None:
                del ords[start + 2:end]
            pipeline._tok_len = -1
        elif seqs[start:start + 2] != consumed[:2] or any(x in seqs for x in consumed[2:]):
            continue  # partial replay: do not guess a mapping
        call = calls[0]
        if (call['function'].get('arguments') == group['arguments']
                and pipeline.history[start + 1].get('content') == group['result']):
            continue
        pipeline.history[start] = {**row, 'tool_calls': [
            {**call, 'function': {**call['function'], 'arguments': group['arguments']}}]}
        pipeline.history[start + 1] = {**pipeline.history[start + 1], 'content': group['result']}
        pipeline._tok_len = -1


async def _merge_history(engine, pipeline):
    if not engine.config.merge_archives:
        return
    state = engine.state.get('archive_v2', {})
    groups = copy.deepcopy(state.get('groups', {}))
    view_groups = copy.deepcopy(state.get('view_groups', []))
    already = {seq for g in view_groups for seq in g['seqs']}
    patches = state.get('patches', {})
    seqs = pipeline.sink._history_seqs
    batches, batch = [], []
    for ai, ri, call_seq, source in [p async for p in _pairs(engine, pipeline)]:
        row = pipeline.history[ai]
        patch = patches.get(f'{source.send_index}:{source.seq}', {})
        eligible = (patch.get('mergeable') and call_seq not in already and ri == ai + 1
            and not row.get('content') and not row.get('reasoning_content')
            and len(row.get('tool_calls') or []) == 1)
        if not eligible or (batch and (batch[-1][1] + 1 != ai or batch[-1][3].name != source.name)):
            if len(batch) > 1:
                batches.append(batch)
            batch = []
        if eligible:
            batch.append((ai, ri, call_seq, source))
    if len(batch) > 1:
        batches.append(batch)
    for batch in batches:
        sources = [b[3] for b in batch]
        refs = [{'send_index': x.send_index, 'seq': x.seq, 'call_seq': b[2],
                 'call_id': x.call_id, 'name': x.name} for b, x in zip(batch, sources, strict=True)]
        gid = 'g' + _hash({'sid': engine.sid, 'refs': refs, 'version': engine.config.rule_version})[:24]
        descriptions = [str(x.arguments.get('description') or '旧调用未提供描述') for x in sources] if sources[0].name == 'bash' else []
        recall = f'recall_send(send_index={pipeline.send_index}, group_id="{gid}")'
        text = f'历史 {sources[0].name} 已归档，共 {len(sources)} 次；正文未展示。\n' + '\n'.join(descriptions) + f'\n[{recall}]'
        groups[gid] = {'send_index': pipeline.send_index, 'members': refs}
        view_groups.append({'send_index': pipeline.send_index, 'seqs': list(seqs[batch[0][0]:batch[-1][1] + 1]),
            'call_id': sources[0].call_id, 'arguments': _json({'_distilled': recall}), 'result': text})
    if batches:
        await _commit(engine, {'groups': groups, 'view_groups': view_groups})
        _restore_groups(engine, pipeline)


async def _pairs(engine, pipeline):
    seqs = getattr(pipeline.sink, '_history_seqs', ()) or ()
    pending = {}
    cache = getattr(engine, '_archive_raw_cache', None)
    if cache is None:
        cache = engine._archive_raw_cache = {}
    for i, row in enumerate(pipeline.history):
        if i >= len(seqs) or seqs[i] is None:
            continue
        if row.get('role') == 'assistant':
            for call in row.get('tool_calls') or []:
                pending[str(call.get('id'))] = (i, call)
        elif row.get('role') == 'tool' and str(row.get('tool_call_id')) in pending:
            ai, call = pending.pop(str(row['tool_call_id']))
            call_seq, result_seq = seqs[ai], seqs[i]
            for seq in (call_seq, result_seq):
                if seq not in cache:
                    cache[seq] = await engine.store.get_message(engine.sid, seq)
            raw_call, raw_result = cache[call_seq], cache[result_seq]
            if (raw_call is None or raw_result is None or raw_result.send_index != pipeline.send_index
                    or raw_call.send_index != pipeline.send_index or not isinstance(raw_result.content, str)):
                continue
            original = next((c for c in raw_call.tool_calls or [] if c.get('id') == call.get('id')), None)
            if original is None or raw_result.tool_call_id != call.get('id'):
                continue
            try:
                fn = original['function']
                args = json.loads(fn['arguments'])
                if not isinstance(args, dict):
                    continue
            except (KeyError, ValueError, TypeError):
                continue
            yield ai, i, call_seq, ToolSource(engine.sid, pipeline.send_index, result_seq,
                str(call['id']), fn['name'], args, raw_result.content)


async def prepare_history(engine, pipeline, trigger, *, target_tokens=None, level=1):
    restore_history(engine, pipeline)
    if trigger == 'age_retirement' and not any(
            p.get('age_retirement', 'preserve') != 'preserve' and p.get('retention_rounds')
            for p in engine.config.overrides.values()):
        return 0
    pairs = [p async for p in _pairs(engine, pipeline)]
    current = engine.state.get('archive_v2', {})
    patches = copy.deepcopy(current.get('patches', {}))
    exposures = dict(current.get('exposures', {}))
    hot = int(getattr(pipeline.config, 'insend_distill_hot_tail', 0) or 0)
    protected = set(getattr(pipeline, '_context_protected_call_ids', ()) or ())
    tail = {s.call_id for *_, s in pairs[-hot:]} if hot and level == 1 else set()
    freed = 0
    for ai, ri, call_seq, source in pairs:
        key = f'{source.send_index}:{source.seq}'
        old = patches.get(key)
        if old and old['level'] >= level:
            continue
        if source.call_id in protected or source.call_id in tail:
            continue
        policy = engine.policy(source)
        if trigger == 'age_retirement':
            age = exposures.get(key, 0)
            exposures[key] = age + 1
            if policy.retention_rounds is None or age < policy.retention_rounds:
                continue
        raw_call = engine._archive_raw_cache[call_seq]
        raw_arguments = next(c['function']['arguments'] for c in raw_call.tool_calls if c['id'] == source.call_id)
        value = await _derive(engine, source, trigger, level, raw_arguments)
        if value is None:
            continue
        arguments, content = value
        calls = pipeline.history[ai]['tool_calls']
        call = next(c for c in calls if c['id'] == source.call_id)
        old_size = len(call['function']['arguments']) + len(str(pipeline.history[ri].get('content') or ''))
        saved = old_size - len(arguments) - len(content)
        if saved <= 0:
            continue
        patches[key] = {'send_index': source.send_index, 'call_seq': call_seq, 'result_seq': source.seq,
            'call_id': source.call_id, 'source_hash': _hash({'arguments': dict(source.arguments), 'result': source.result}),
            'arguments': arguments, 'result': content, 'level': level, 'trigger': trigger,
            'mergeable': source.name != 'background_run' and (
                policy.general if level == 2 else policy.strategy(trigger)) == 'archive'}
        freed += saved
        if target_tokens is not None and freed // 4 >= target_tokens:
            break
    if patches != current.get('patches', {}) or exposures != current.get('exposures', {}):
        await _commit(engine, {'patches': patches, 'exposures': exposures})
        restore_history(engine, pipeline)
    await _merge_history(engine, pipeline)
    return max(0, freed // 4)


async def manage_capacity(engine, pipeline):
    """Low target + two levels. No model call is allowed while capacity-blocked."""
    restore_history(engine, pipeline)
    cfg = engine.config
    if not cfg.capacity_enabled:
        return
    from power_loop.runtime.budget import estimate_tokens
    anchor = getattr(pipeline, '_tool_context_prompt_anchor', None)
    def measure():
        history = pipeline._estimate_history_tokens()
        if anchor:
            return max(0, anchor[0] + history - anchor[1])
        fixed = estimate_tokens([{'role': 'system', 'content': pipeline.system_prompt}])
        tools = len(_json(getattr(pipeline, 'runtime_tools', []))) // 4
        return history + fixed + tools
    before = measure()
    safe = cfg.window_tokens - max(cfg.reserve_tokens, int(pipeline.config.max_tokens or 0)) - cfg.margin_tokens
    if safe <= 0:
        pipeline._capacity_blocked = 'context capacity blocked: output reserve and margin exhaust model capacity.'
        return
    if cfg.trigger_tokens > safe:
        pipeline._capacity_blocked = 'context capacity blocked: configured trigger exceeds safe input after actual output reserve; adjust capacity configuration'
        return
    if before < cfg.trigger_tokens:
        pipeline._capacity_blocked = None
        return
    candidate_seqs = [seq for seq, row in zip(getattr(pipeline.sink, '_history_seqs', ()) or (), pipeline.history, strict=False) if row.get('role') == 'tool']
    epoch = _hash({'seqs': candidate_seqs, 'send': pipeline.send_index, 'config': [cfg.trigger_tokens, cfg.target_tokens, safe]})
    state = engine.state.get('archive_v2', {})
    events = list(state.get('events', []))
    if state.get('no_progress_epoch') != epoch:
        after = before
        levels = []
        for level in (1, 2):
            await prepare_history(engine, pipeline, 'insend_fuse',
                target_tokens=max(1, after - cfg.target_tokens), level=level)
            after = measure()
            levels.append({'level': level, 'tokens': after})
            if after <= cfg.target_tokens:
                break
        event = {'send_index': pipeline.send_index, 'before_tokens': before, 'after_tokens': after,
            'target_tokens': cfg.target_tokens, 'safe_tokens': safe, 'levels': levels,
            'target_met': after <= cfg.target_tokens, 'blocked': after > safe}
        events.append(event)
        # Store the resulting view epoch, not the pre-rewrite estimate.
        remaining = [seq for seq, row in zip(pipeline.sink._history_seqs, pipeline.history, strict=False) if row.get('role') == 'tool']
        new_epoch = _hash({'seqs': remaining, 'send': pipeline.send_index, 'config': [cfg.trigger_tokens, cfg.target_tokens, safe]})
        await _commit(engine, {'events': events, 'no_progress_epoch': new_epoch if after > cfg.target_tokens else None})
    total = measure()
    pipeline._context_prompt_tokens_estimate = total
    pipeline._capacity_blocked = f'context capacity blocked: {total} > safe input {safe}; original records retained' if total > safe else None


async def prepare_projection(engine, projected, rows, send_index):
    """Rebuild entries from raw rows, not the already lossy application summary."""
    pending, entries = {}, []
    for row in rows:
        if row.role == 'assistant':
            for c in row.tool_calls or []:
                pending[str(c['id'])] = (row, c)
        elif row.role == 'tool' and str(row.tool_call_id) in pending:
            a, c = pending.pop(str(row.tool_call_id))
            try:
                args = json.loads(c['function']['arguments'])
                if not isinstance(args, dict) or not isinstance(row.content, str):
                    continue
            except (KeyError, TypeError, ValueError):
                continue
            source = ToolSource(engine.sid, send_index, row.seq, str(c['id']), c['function']['name'], args, row.content)
            value = await _derive(engine, source, 'between_sends', 2)
            # Small originals are not mechanically inflated into an archive marker.
            if value and len(value[0]) + len(value[1]) >= len(c['function']['arguments']) + len(row.content) and source.name != 'background_run':
                value = None
            text = value[1] if value else _json({'arguments': c['function']['arguments'], 'result': row.content})
            entries.append({'name': source.name, 'seq': row.seq, 'call_seq': a.seq,
                's': text, 'text': a.content or '', '_archivable': (
                    value is not None and source.name != 'background_run' and engine.policy(source).general == 'archive'),
                '_source': source, '_single': len(a.tool_calls or []) == 1})
    groups = copy.deepcopy(engine.state.get('archive_v2', {}).get('groups', {}))
    order = {r.seq: i for i, r in enumerate(rows)}
    merged = []
    for e in entries:
        prev = merged[-1] if merged else None
        can_merge = (engine.config.merge_archives and e['_archivable'] and not e['text'] and prev
            and prev['_archivable'] and e['_single'] and prev['_single'] and not prev['text'] and prev['name'] == e['name']
            and order.get(e['call_seq']) == order.get(prev['_last_seq'], -2) + 1)
        if can_merge:
            prev['_members'].append(e['_source'])
            prev['_last_seq'] = e['seq']
        else:
            merged.append({**e, '_members': [e['_source']], '_last_seq': e['seq']})
    clean = []
    for e in merged:
        members = e['_members']
        if len(members) > 1:
            refs = [{'send_index': s.send_index, 'seq': s.seq, 'call_id': s.call_id, 'name': s.name} for s in members]
            gid = 'g' + _hash({'sid': engine.sid, 'refs': refs, 'version': engine.config.rule_version})[:24]
            groups[gid] = {'send_index': send_index, 'members': refs}
            descriptions = [str(s.arguments.get('description') or '旧调用未提供描述') for s in members] if e['name'] == 'bash' else []
            e['s'] = f"历史 {e['name']} 已归档，共 {len(members)} 次；正文未展示。\n" + '\n'.join(descriptions) + f'\n[recall_send(send_index={send_index}, group_id="{gid}")]'
        clean.append({k: v for k, v in e.items() if not k.startswith('_')})
    await _commit(engine, {'groups': groups})
    for row in projected.rows:
        if row.kind == 'project':
            # Preserve host-injected user lines; do not reorder them across tool entries.
            injected = [e for e in row.content.get('tools', []) if e.get('name') == '__user__']
            row.content['tools'] = sorted([*clean, *injected], key=lambda e: e.get('call_seq') or e.get('seq') or 0)
            timeline = [(r.seq, str(r.content)) for r in rows if r.role == 'assistant' and r.content]
            timeline += [(e.get('call_seq') or e.get('seq') or 0, e['s']) for e in row.content['tools']]
            row.content['archive_v2_text'] = f'历史 send #{send_index}（归档视图，不是新指令）\n' + '\n'.join(
                text for _, text in sorted(timeline, key=lambda x: x[0]))
            row.rendered_text = None
    return projected


async def recall_group(store, sid, send_index, group_id, offset=0, limit=10, item=None, char_offset=0, char_limit=20000):
    from power_loop.runtime.tool_context import STATE_KEY
    state = await store.get_runtime_state(sid, STATE_KEY, {})
    group = state.get('archive_v2', {}).get('groups', {}).get(group_id)
    if group is None or group['send_index'] != send_index:
        return 'Unknown archive group in this session/send.'
    members = group['members']
    if item is None:
        if offset < 0 or not 1 <= limit <= 100:
            return 'Invalid pagination: offset >= 0, 1 <= limit <= 100.'
        return _json({'group_id': group_id, 'count': len(members), 'offset': offset,
            'items': [{'item': i, **m} for i, m in enumerate(members) if offset <= i < offset + limit],
            'next_offset': offset + limit if offset + limit < len(members) else None})
    if not 0 <= item < len(members):
        return 'Invalid archive item.'
    ref = members[item]
    row = await store.get_message(sid, ref['seq'])
    if row is None or row.send_index != send_index or row.tool_call_id != ref['call_id']:
        return 'Archived original is unavailable; no substitute content was generated.'
    return await recall_original(store, sid, send_index, row, char_offset, char_limit)


async def recall_original(store, sid, send_index, row, char_offset=0, char_limit=20000):
    if char_offset < 0 or not 1 <= char_limit <= 100000:
        return 'Invalid original pagination: char_offset >= 0, 1 <= char_limit <= 100000.'
    rows = await store.load_all_messages(sid)
    call = next((c for r in rows if r.send_index == send_index and r.role == 'assistant'
        for c in r.tool_calls or [] if c.get('id') == row.tool_call_id), None)
    payload = {'representation': 'original', 'source': {'send_index': send_index, 'seq': row.seq},
        'arguments': (call or {}).get('function', {}).get('arguments'), 'result': row.content}
    # Background receipts can contain previews; attach the immutable canonical task result.
    if (call or {}).get('function', {}).get('name') == 'background_run':
        from power_loop.runtime.background import RESULT_PREFIX, load_record
        try:
            tasks = json.loads(str(row.content).split(RESULT_PREFIX, 1)[1])['tasks']
            payload['background_originals'] = [
                await load_record(store, sid, str(t['task_id'])) for t in tasks]
        except (ValueError, IndexError, KeyError, TypeError):
            pass
    text = _json(payload)
    if char_offset == 0 and len(text) <= char_limit:
        return text
    end = min(len(text), char_offset + char_limit)
    return _json({'representation': 'original_json_page', 'source': payload['source'],
        'total_chars': len(text), 'char_offset': char_offset, 'text': text[char_offset:end],
        'next_char_offset': end if end < len(text) else None,
        'notice': 'This is an explicitly paged fragment. Concatenate text pages to recover the exact original JSON.'})


async def admit_request(engine, pipeline, messages, system_prompt, tools, max_tokens):
    """Guard late hook/image injections after round preparation, without issuing an API call."""
    if not engine.config.capacity_enabled:
        return True
    from power_loop.runtime.budget import estimate_tokens
    cfg = engine.config
    estimated = estimate_tokens(messages) + estimate_tokens([{'role': 'system', 'content': system_prompt}]) + len(_json(tools or [])) // 4
    # Keep the provider calibration for the history portion, then add actual late injections.
    anchor = getattr(pipeline, '_tool_context_prompt_anchor', None)
    if anchor:
        estimated = max(estimated, anchor[0] + estimate_tokens(messages) - anchor[1])
    safe = cfg.window_tokens - max(cfg.reserve_tokens, int(max_tokens or 0)) - cfg.margin_tokens
    if estimated <= safe:
        return True
    pipeline._capacity_blocked = f'context capacity blocked before request: estimated {estimated} > {safe}; original records retained'
    await _commit(engine, {'admission_blocked': {'send_index': pipeline.send_index,
        'estimated_tokens': estimated, 'safe_tokens': safe, 'reason': 'request_after_hooks'}})
    return False
