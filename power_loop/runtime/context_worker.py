"""Background summarization. Neither generation nor delivery is awaited by a send."""
from __future__ import annotations

import asyncio
import hashlib
import logging
import time
import uuid
from typing import Any

from power_loop._vendor.llm_client.attempt_scope import single_attempt
from power_loop._vendor.llm_client.interface import LLMRequest
from power_loop.runtime.store.context_cache import ContextCache

logger = logging.getLogger(__name__)


class SummaryWorker:
    def __init__(self, store: Any):
        self.cache = ContextCache(store)
        self.bindings: dict[str, Any] = {}
        self.tasks: list[asyncio.Task] = []
        self.wakeup = asyncio.Event()
        self.closing = False

    def bind(self, engine: Any) -> None:
        self.bindings[engine.executor_key] = engine
        if not self.tasks:
            self.tasks = [asyncio.create_task(self._run()) for _ in range(2)]
        self.wakeup.set()

    async def close(self) -> None:
        self.closing = True
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        self.tasks.clear()

    async def _run(self) -> None:
        while not self.closing:
            worked = False
            try:
                for executor, engine in list(self.bindings.items()):
                    job = await self.cache.claim(executor)
                    if job is not None:
                        worked = True
                        await self.execute(job, engine)
                    await self.cache.reconcile(engine.config.deliver_usage,
                                               delivery_timeout=engine.config.delivery_budget_s)
                if self.bindings:
                    cfg = next(iter(self.bindings.values())).config
                    await self.cache.prune(max_entries=cfg.cache_max_entries, max_bytes=cfg.cache_max_bytes)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception('summary worker iteration failed; durable jobs retained')
            if not worked:
                self.wakeup.clear()
                try:
                    await asyncio.wait_for(self.wakeup.wait(), 2)
                except TimeoutError:
                    pass

    async def execute(self, job: dict, engine: Any) -> None:
        payload = job['payload']
        attempt_id = uuid.uuid4().hex
        record = {**payload['attribution'], 'attempt_id': attempt_id, 'call_id': attempt_id,
                  'operation_id': job['cache_key'], 'purpose': 'tool_distill', 'trigger': 'prefetch',
                  'owner_session_id': payload['owner_session_id'], 'owner_send_index': payload['owner_send_index'],
                  'source_session_id': payload['owner_session_id'], 'source_send_index': payload['source_send_index'],
                  'source_seq': payload['source_seq'], 'source_call_id': payload['source_call_id'],
                  'target_tool': payload['name'], 'model': payload['model'],
                  'model_identity': engine.model_identity, 'outcome': 'started', 'started_at': time.time(),
                  'usage': None, 'usage_unknown': True, 'success': False}
        await self.cache.begin_attempt(job, record)
        token = single_attempt.set(True)
        summary = None
        try:
            response = await asyncio.wait_for((engine.config.summary_llm or engine.llm).complete(LLMRequest(
                messages=[{'role': 'user', 'content': payload['input']}], system_prompt=payload['prompt'],
                model=payload['model'], max_tokens=payload['max_tokens'], temperature=0)), payload['timeout_s'])
            usage = getattr(response, 'token_usage', None)
            record['usage'] = usage.as_dict() if hasattr(usage, 'as_dict') else usage
            record['usage_unknown'] = not isinstance(record['usage'], dict) or not any(
                value is not None for value in record['usage'].values())
            record.update(outcome='success', success=True)
            text = getattr(response, 'content_text', '') or getattr(response, 'raw_text', '')
            if (not getattr(response, 'tool_calls', None) and isinstance(text, str) and text.strip()
                    and text.strip() != '<preserve>' and len(text) + 128 < len(payload['input'])):
                summary = text
        except asyncio.CancelledError:
            record['outcome'] = 'cancelled'
            raise
        except Exception as exc:
            status = getattr(exc, 'status_code', None)
            record.update(outcome='failed' if status else 'interrupted',
                          error_type=type(exc).__name__, error_http_status=status)
            usage = getattr(exc, 'token_usage', None)
            if usage is not None:
                record['usage'] = usage.as_dict() if hasattr(usage, 'as_dict') else usage
                record['usage_unknown'] = not bool(record['usage'])
        finally:
            single_attempt.reset(token)
            usage = record.get('usage')
            if isinstance(usage, dict):
                if usage.get('total_tokens') is None and usage.get('prompt_tokens') is not None and usage.get('completion_tokens') is not None:
                    usage['total_tokens'] = usage['prompt_tokens'] + usage['completion_tokens']
                if usage.get('prompt_cached_tokens') is None:
                    for alias in ('cached_tokens', 'cache_hit_tokens', 'cache_read_tokens'):
                        if usage.get(alias) is not None:
                            usage['prompt_cached_tokens'] = usage[alias]
                            break
            record['finished_at'] = time.time()
            record['duration_ms'] = int((record['finished_at'] - record['started_at']) * 1000)
            digest = hashlib.sha256(summary.encode()).hexdigest() if summary is not None else None
            await self.cache.finish(job, record, summary, digest)
        # The ledger owns billing; result.usage is only a snapshot at send return.
        # No mutation of a finished result object and no session runtime-state writes here.
