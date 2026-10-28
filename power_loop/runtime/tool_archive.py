"""Deterministic tool-context rewrites: the in-send fuse and the between-sends entries.

Raw rows are never modified. The session runner serializes writes. Derived patches and exact
group membership are committed in the existing atomic runtime-state row BEFORE changing the
in-memory view.
"""
from __future__ import annotations

import copy
import json
from dataclasses import replace

from power_loop.runtime.tool_context import TOOL_CONTEXT_STATE_KEY, ToolSource, _hash, _json, recall_note

#: Sub-key of the runtime state that holds patches, groups and fuse events.
ARCHIVE_KEY = 'archive'
_LEGACY_ARCHIVE_KEY = 'archive_v2'


def adopt_legacy_state(state: dict) -> None:
    """Move state written under the earlier sub-key; issued group ids stay recallable."""
    legacy = state.pop(_LEGACY_ARCHIVE_KEY, None)
    if legacy is not None and ARCHIVE_KEY not in state:
        state[ARCHIVE_KEY] = legacy


def _archive(state: dict) -> dict:
    return state.get(ARCHIVE_KEY) or state.get(_LEGACY_ARCHIVE_KEY) or {}


def slim_arguments(source: ToolSource, registered=None) -> str:
    """Default is empty arguments; an implemented view may retain a useful label."""
    from .tool_contracts import context_presentation
    return context_presentation(registered, source).archive_arguments or '{}'


def archive_label(source: ToolSource, registered=None) -> str:
    """Use the implemented label, otherwise bounded generic arguments."""
    from .tool_contracts import context_presentation
    label = context_presentation(registered, source).archive_label
    if label is not None:
        return label
    args = {k: v for k, v in source.arguments.items() if k not in ('context_policy', '_distilled')}
    text = _json(args)
    return text if len(text) <= 160 else text[:160] + '…'


def archive_text(source: ToolSource, registered=None) -> str:
    return f'{archive_label(source, registered)}\n{recall_note(source.recall)}'


async def _derive(engine, source, trigger, level, raw_arguments=None):
    if not getattr(engine, '_implementation_matches', True):
        return None
    policy = engine.policy(source)
    registered = engine.registry.get(source.name) if engine.registry else None
    if policy.hard_preserve:
        return None
    if engine.is_result_container(source):
        targets = await engine._background_sources(source)
        if not targets:
            return None
        values = [await _derive(engine, t, trigger, level) for t in targets]
        if any(v is None for v in values):
            if trigger == 'general':
                return _json(dict(source.arguments)), _json({
                    'receipt': source.result, 'targets': [
                        {'tool': t.name, 'task_id': t.call_id, 'arguments': dict(t.arguments), 'result': t.result}
                        for t in targets]})
            return None
        return slim_arguments(source, registered), '\n'.join(
            f'{t.name} task={t.call_id}: {v[1]}' for t, v in zip(targets, values, strict=True))
    strategy = policy.general if level == 2 else policy.strategy(trigger)
    if engine.preserves_context(source, trigger):
        return None
    from .tool_contracts import format_result
    view = format_result(registered, source.result, policy.result_max_chars or 600)
    if view is not None:
        return _json(dict(source.arguments)), view
    if strategy == 'preserve':
        return None
    if strategy == 'archive':
        return slim_arguments(source, registered), archive_text(source, registered)
    if strategy == 'truncate':
        from .tool_contracts import truncate_context
        return truncate_context(registered, source, policy,
                                raw_arguments=raw_arguments or _json(dict(source.arguments)))
    # Preserve extension points (rule / precomputed llm_summary), without recursively
    # applying a truncation to its own preview. The original source is always used.
    call_trigger = 'general' if level == 2 else trigger
    effective = replace(policy, **{call_trigger: strategy})
    text = await engine.distill(source, call_trigger, policy_override=effective)
    return (slim_arguments(source, registered), text) if text is not None else None


async def _commit(engine, patch):
    previous = copy.deepcopy(engine.state)
    adopt_legacy_state(engine.state)
    engine.state.setdefault(ARCHIVE_KEY, {}).update(patch)
    try:
        await engine._save()
    except BaseException:
        engine.state = previous
        raise


def restore_history(engine, pipeline):
    """Reapply committed paired patches, idempotently, including a resumed send."""
    patches = _archive(engine.state).get('patches', {})
    seqs = getattr(pipeline.sink, '_history_seqs', ()) or ()
    positions = {s: i for i, s in enumerate(seqs) if s is not None}
    grouped = {seq for g in _archive(engine.state).get('view_groups', [])
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
    for group in _archive(engine.state).get('view_groups', []):
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
    if not engine.config.merge_consecutive:
        return
    state = _archive(engine.state)
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
        text = group_text([archive_label(x, engine.registry.get(x.name) if engine.registry else None) for x in sources], pipeline.send_index, gid)
        groups[gid] = {'send_index': pipeline.send_index, 'members': refs}
        view_groups.append({'send_index': pipeline.send_index, 'seqs': list(seqs[batch[0][0]:batch[-1][1] + 1]),
            'call_id': sources[0].call_id, 'arguments': slim_arguments(sources[0], engine.registry.get(sources[0].name) if engine.registry else None), 'result': text})
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
    current = _archive(engine.state)
    patches = copy.deepcopy(current.get('patches', {}))
    exposures = dict(current.get('exposures', {}))
    hot = int(engine.config.hot_tail or 0)
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
            'mergeable': not engine.is_result_container(source) and (
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
    state = _archive(engine.state)
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


def _sources(engine, rows, send_index):
    """Paired original calls of one send, keyed by result seq."""
    pending, sources = {}, {}
    for row in rows:
        if row.role == 'assistant':
            for c in row.tool_calls or []:
                pending[str(c.get('id'))] = c
        elif row.role == 'tool' and str(row.tool_call_id) in pending:
            c = pending.pop(str(row.tool_call_id))
            try:
                args = json.loads(c['function']['arguments'])
            except (KeyError, TypeError, ValueError):
                continue
            if isinstance(args, dict) and isinstance(row.content, str):
                sources[row.seq] = ToolSource(engine.sid, send_index, row.seq, str(c['id']),
                                              c['function']['name'], args, row.content)
    return sources


def group_text(lines: list[str], send_index: int, group_id: str) -> str:
    """One entry for consecutive calls of one tool: each member's distilled line, then one recall
    note for the whole group (members' own notes are dropped)."""
    return '\n'.join(lines) + '\n' + recall_note(f'recall_send(send_index={send_index}, group_id="{group_id}")')


def _without_note(text: str) -> str:
    head, sep, tail = text.rpartition('\n（原文 recall_send(')
    return head if sep and tail.endswith('）') and '\n' not in tail else text


async def prepare_projection(engine, projected, rows, send_index):
    """Between sends every tool uses its ``general`` rule.

    ``rule`` keeps the projector's own distilled entry (it already carries row coordinates);
    other strategies rewrite the entry from the original paired rows. Then consecutive entries
    of one tool merge into a group entry. Breaks: another tool, or an injected user line. The
    assistant's text and reasoning are not entries, so they never break a run; parallel calls
    of one tool in one round are consecutive entries.
    """
    sources = _sources(engine, rows, send_index)
    groups = copy.deepcopy(_archive(engine.state).get('groups', {}))
    added = False
    for prow in projected.rows:
        if prow.kind != 'project' or not isinstance(prow.content, dict):
            continue
        entries = [e for e in prow.content.get('tools') or [] if isinstance(e, dict)]
        for e in entries:
            source = sources.get(e.get('seq'))
            if source is None or e.get('name') == '__user__':
                continue
            policy = engine.policy(source)
            from .tool_contracts import format_result
            registered = engine.registry.get(source.name) if engine.registry else None
            view = format_result(registered, source.result, policy.result_max_chars or 600)
            if view is not None:
                e['s'] = view
                e.pop('result', None)
                continue
            if policy.hard_preserve or policy.general == 'rule':
                continue
            if policy.general == 'preserve':
                text = _json({'arguments': dict(source.arguments), 'result': source.result})
            else:
                value = await _derive(engine, source, 'general', 2)
                if value is None:
                    continue
                text = value[1]
            coords = {k: e[k] for k in ('name', 'seq', 'call_seq') if k in e}
            e.clear()
            e.update(coords, s=text)
        # Split into runs: consecutive joinable entries of one tool; anything else stands alone.
        runs: list[list[dict]] = []
        for e in entries:
            joinable = e.get('name') != '__user__' and e.get('seq') in sources
            if (joinable and runs and runs[-1][-1].get('_join')
                    and runs[-1][-1].get('name') == e.get('name')):
                runs[-1].append({**e, '_join': True})
            else:
                runs.append([{**e, '_join': joinable}])
        merged: list[dict] = []
        for run in runs:
            run = [{k: v for k, v in e.items() if k != '_join'} for e in run]
            if len(run) < 2 or not engine.config.merge_consecutive:
                merged.extend(run)
                continue
            members = [sources[e['seq']] for e in run]
            refs = [{'send_index': m.send_index, 'seq': m.seq, 'call_id': m.call_id, 'name': m.name}
                    for m in members]
            gid = 'g' + _hash({'sid': engine.sid, 'refs': refs, 'version': engine.config.rule_version})[:24]
            groups[gid] = {'send_index': send_index, 'members': refs}
            added = True
            merged.append({'name': run[0]['name'], 'seq': run[0]['seq'], 'n': len(run), 'group_id': gid,
                           'seqs': [e['seq'] for e in run],
                           's': group_text([_without_note(str(e.get('s') or '')) for e in run],
                                           send_index, gid)})
        prow.content['tools'] = merged
        prow.rendered_text = None
    if added:
        await _commit(engine, {'groups': groups})
    return projected


async def recall_group(store, sid, send_index, group_id, offset=0, limit=10, item=None, char_offset=0, char_limit=20000):
    state = await store.get_runtime_state(sid, TOOL_CONTEXT_STATE_KEY, {})
    group = _archive(state).get('groups', {}).get(group_id)
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
    from power_loop.runtime.background import RESULT_PREFIX, load_record
    # Historical data reader only: aliases retain access to the same stored
    # receipt format. This does not select or invoke any current implementation.
    if RESULT_PREFIX in str(row.content):
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
