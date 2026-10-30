"""
LLM factory utilities for zrag.

This module provides a small, explicit way to build an OpenAI-compatible `LLMService`
from environment / `src.config.settings`, similar in spirit to agent-psychology's
model utilities, but adapted to zrag's `LLMService` Protocol.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import random
from collections.abc import AsyncIterator, Callable, Sequence
from typing import TYPE_CHECKING, Any, cast

from openai import AsyncOpenAI, DefaultAsyncHttpxClient
from openai._streaming import SSEDecoder

from .capabilities import ModelCapabilities, coerce_capabilities
from .exact_json import configure_http_client, exact_event
from .interface import LLMRequest, LLMResponse, LLMService, LLMStreamChunk, LLMTokenUsage, OpenAICompatibleChatConfig
from .llm_utils import parse_json_from_model_output_detailed
from .proxy_fallback import env_proxy_http_client

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionChunk
else:
    ChatCompletionChunk = Any

logger = logging.getLogger(__name__)


class ExactSSEDecoder(SSEDecoder):
    async def aiter_bytes(self, iterator):
        async for event in super().aiter_bytes(iterator):
            yield exact_event(event)

    def iter_bytes(self, iterator):
        for event in super().iter_bytes(iterator):
            yield exact_event(event)


class ExactAsyncOpenAI(AsyncOpenAI):
    def __init__(self, *args, **options):
        client = options.get('http_client') or DefaultAsyncHttpxClient()
        options['http_client'] = configure_http_client(client)
        super().__init__(*args, **options)

    def _make_sse_decoder(self):
        return ExactSSEDecoder()


_PROXY_ENV_KEYS = (
    "ALL_PROXY",
    "all_proxy",
    "HTTPS_PROXY",
    "https_proxy",
    "HTTP_PROXY",
    "http_proxy",
)


def _normalize_proxy_url_scheme(url: str) -> str:
    text = (url or "").strip()
    if text.lower().startswith("socks://"):
        # httpx/openai expects socks5://, not socks://.
        return f"socks5://{text[len('socks://'):]}"
    return text


def _normalize_proxy_env_inplace() -> None:
    for key in _PROXY_ENV_KEYS:
        value = os.environ.get(key)
        if not value:
            continue
        normalized = _normalize_proxy_url_scheme(value)
        if normalized != value:
            os.environ[key] = normalized
            logger.info("Normalized proxy env %s from socks:// to socks5://", key)


def _sanitize_debug_payload(value: Any) -> Any:
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            if key == "url" and isinstance(item, str) and item.startswith("data:"):
                prefix, _, payload = item.partition(",")
                sanitized[key] = f"{prefix},<base64:{len(payload)} chars>"
            else:
                sanitized[key] = _sanitize_debug_payload(item)
        return sanitized
    if isinstance(value, list):
        return [_sanitize_debug_payload(item) for item in value]
    return value


def _format_messages_for_debug(messages: Sequence[dict[str, Any]]) -> str:
    sanitized = _sanitize_debug_payload(list(messages))
    return json.dumps(sanitized, ensure_ascii=False, indent=2)




def merge_request_extra(base: dict[str, Any] | None, override: dict[str, Any] | None) -> dict[str, Any]:
    """Config-level ``request_extra`` under a request's ``extra`` (request wins; ``extra_body`` merged)."""
    out: dict[str, Any] = dict(base or {})
    for k, v in dict(override or {}).items():
        if k == "extra_body" and isinstance(v, dict) and isinstance(out.get("extra_body"), dict):
            out["extra_body"] = {**out["extra_body"], **v}
        else:
            out[k] = v
    if isinstance(out.get("extra_body"), dict):
        out["extra_body"] = dict(out["extra_body"])
    return out


def _is_permanent_status(exc: BaseException) -> bool:
    """4xx other than 408/409/429 — mirrors power_loop.runtime.retry.is_permanent_llm_error (kept
    local: the vendored client must not import the runtime package)."""
    status = getattr(exc, "status_code", None)
    if not isinstance(status, int):
        status = getattr(getattr(exc, "response", None), "status_code", None)
    return isinstance(status, int) and 400 <= status < 500 and status not in (408, 409, 429)

class OpenAICompatibleChatLLMService(LLMService):
    """
    Minimal OpenAI-compatible chat completion client.

    - Uses `openai.AsyncOpenAI`
    - Records token usage from response.usage (if present)
    """

    def __init__(self, cfg: OpenAICompatibleChatConfig):
        self._cfg = cfg
        self._client: Any = None
        self._last_usage: dict[str, Any] = {}
        # Read straight from config — no inference from the model name (that guessing table
        # is gone; see capabilities.py). Declared nothing => capable of nothing but text.
        # A declaration that names ANOTHER model (e.g. handed down from a parent process whose
        # model differs from this client's) says nothing about this one → undeclared.
        self._capabilities: ModelCapabilities = coerce_capabilities(
            cfg.capabilities, model=cfg.model
        ).for_model(cfg.model)

        logger.info(
            "GraphExtractor LLM: base_url=%s model=%s timeout_s=%s max_tokens=%s temperature=%s api_key=%s",
            cfg.base_url,
            cfg.model,
            cfg.timeout_s,
            cfg.max_tokens,
            cfg.temperature,
            "set" if bool(cfg.api_key) else "missing",
        )

    def get_last_token_usage(self) -> dict[str, Any]:
        return dict(self._last_usage or {})

    @property
    def capabilities(self) -> ModelCapabilities:
        return self._capabilities

    def _ensure_client(self) -> Any:
        from .attempt_scope import single_attempt
        if self._client is not None:
            return self._client.with_options(max_retries=0) if single_attempt.get() else self._client
        _normalize_proxy_env_inplace()

        self._client = ExactAsyncOpenAI(
            base_url=self._cfg.base_url,
            api_key=self._cfg.api_key,
            timeout=self._cfg.timeout_s,
            http_client=env_proxy_http_client(DefaultAsyncHttpxClient),
        )
        return self._client.with_options(max_retries=0) if single_attempt.get() else self._client

    async def close(self) -> None:
        """
        Close underlying HTTP resources (httpx) to avoid ResourceWarning in tests.
        """
        if self._client is None:
            return
        try:
            # openai.AsyncOpenAI exposes `close()` (async).
            await self._client.close()
        finally:
            self._client = None

    def _record_usage(self, usage: Any, *, method: str) -> None:
        self._last_usage = self._usage_dict_from_any(usage)

        total_tokens = self._last_usage.get("total_tokens")
        if isinstance(total_tokens, int) and total_tokens > 0:
            logger.info(
                "[llm:%s] token_usage prompt=%s completion=%s total=%s",
                method,
                self._last_usage.get("prompt_tokens"),
                self._last_usage.get("completion_tokens"),
                self._last_usage.get("total_tokens"),
            )

    def _emit_debug_payload(self, *, method: str, messages: Any, kwargs: Any) -> None:
        """
        Emit request debug payload.

        This is a best-effort helper used by stream/complete paths. It must exist
        because some code paths call it unconditionally.
        """
        # Avoid spamming logs unless explicitly requested.
        env_flag = (os.getenv("POWER_LOOP_LLM_DEBUG") or "").strip().lower()
        debug_enabled = env_flag in {"1", "true", "yes", "y"} or logger.isEnabledFor(logging.DEBUG)
        if not debug_enabled:
            return

        try:
            rendered_messages = (
                _format_messages_for_debug(messages)
                if isinstance(messages, (list, tuple))
                else _sanitize_debug_payload(messages)
            )
        except Exception:
            rendered_messages = "<unrenderable messages>"

        try:
            sanitized_kwargs = _sanitize_debug_payload(kwargs)
        except Exception:
            sanitized_kwargs = "<unrenderable kwargs>"

        logger.debug("[llm:%s] request messages=%s kwargs=%s", method, rendered_messages, sanitized_kwargs)

    def _usage_dict_from_any(self, usage: Any) -> dict[str, Any]:
        from .usage_normalization import normalize_usage
        return normalize_usage(usage)

    def _usage_obj(self, usage: Any = None):
        from power_loop.llm_metrics import DefaultTokenUsageExtractor
        return DefaultTokenUsageExtractor().extract({"usage": self._last_usage if usage is None else usage})

    async def _with_retries(self, fn, *, method: str) -> Any:
        """
        Lightweight retry wrapper inspired by LangChain's ergonomics.
        """
        last_err: Exception | None = None
        from .attempt_scope import single_attempt
        attempts = 1 if single_attempt.get() else max(1, int(self._cfg.max_retries) + 1)
        for i in range(attempts):
            try:
                return await fn()
            except Exception as e:
                last_err = e
                if i >= attempts - 1 or _is_permanent_status(e):
                    # A 4xx other than 408/409/429 (bad key, insufficient balance, unknown model…)
                    # comes back identical on a retry — don't spend the budget (design/124 Z11).
                    break
                # exponential backoff + jitter
                base = float(self._cfg.retry_base_delay_s)
                delay = base * (2 ** i) * (0.75 + 0.5 * random.random())
                logger.warning("[llm:%s] call failed (attempt %s/%s), retrying in %.2fs: %s", method, i + 1, attempts,
                               delay, e)
                await asyncio.sleep(delay)
        raise cast(Exception, last_err)

    def _request_kwargs(self, request: LLMRequest) -> dict[str, Any]:  # noqa: C901
        """
        Map `LLMRequest` into kwargs for OpenAI-compatible chat.completions.create.
        Falls back to self._cfg for default settings.
        """
        out: dict[str, Any] = merge_request_extra(getattr(self._cfg, "request_extra", None), request.extra)
        out["model"] = request.model if request.model else self._cfg.model
        
        req_temp = request.temperature if request.temperature is not None else self._cfg.temperature
        if req_temp is not None:
            out["temperature"] = float(req_temp)

        req_max_tokens = request.max_tokens if request.max_tokens is not None else self._cfg.max_tokens
        if req_max_tokens is not None:
            try:
                max_tokens = int(req_max_tokens)
            except Exception:
                max_tokens = None
            if max_tokens is not None and max_tokens > 0:
                out["max_tokens"] = max_tokens

        reason_value: bool | None = None
        if "reason" in out:
            reason_value = bool(out.pop("reason"))
        elif request.reason is not None:
            reason_value = bool(request.reason)

        if reason_value is not None:
            extra_body = out.get("extra_body")
            if not isinstance(extra_body, dict):
                extra_body = {}
            if "reason" not in extra_body:
                extra_body["reason"] = reason_value
            out["extra_body"] = extra_body

        if request.tools is not None:
            out["tools"] = request.tools
        if request.tool_choice is not None:
            out["tool_choice"] = request.tool_choice
        if request.response_format is not None:
            out["response_format"] = request.response_format
        return out

    def _build_resume_request(self, request: LLMRequest, partial_text: str) -> LLMRequest:
        resumed_messages = list(request.messages or [])
        trimmed = (partial_text or "").strip()
        if trimmed:
            resumed_messages.append({"role": "assistant", "content": trimmed})
        resumed_messages.append({"role": "user", "content": self._cfg.stream_resume_instruction})
        return LLMRequest(
            messages=resumed_messages,
            system_prompt=request.system_prompt,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            parse_json=request.parse_json,
            reason=request.reason,
            tools=request.tools,
            tool_choice=request.tool_choice,
            response_format=request.response_format,
            extra=dict(request.extra or {}),
        )

    async def complete(
        self,
        request: LLMRequest,
        *,
        on_chunk_delta_text: Callable[[str], Any] | None = None,
        on_chunk_think: Callable[[str], Any] | None = None,
        on_stream_end: Callable[[LLMResponse], Any] | None = None,
    ) -> LLMResponse:
        """
        Canonical non-streaming API (preferred).
        """
        text_parts: list[str] = []
        think_parts: list[str] = []
        chunks: list[LLMStreamChunk] = []
        last_tool_calls: list[dict[str, Any]] = []
        final_usage: LLMTokenUsage | None = None
        last_raw_event: Any = None
        final_cost = None
        final_metrics = {}

        async for chunk in self.stream(request):
            chunks.append(chunk)
            if chunk.delta_text:
                text_parts.append(chunk.delta_text)
                if on_chunk_delta_text:
                    res_val = on_chunk_delta_text(chunk.delta_text)
                    if inspect.isawaitable(res_val):
                        await res_val
            if chunk.think:
                think_parts.append(chunk.think)
                if on_chunk_think:
                    res_val = on_chunk_think(chunk.think)
                    if inspect.isawaitable(res_val):
                        await res_val
            if chunk.tool_calls:
                last_tool_calls = list(chunk.tool_calls)
            if chunk.token_usage is not None:
                final_usage = chunk.token_usage
            if chunk.provider_cost is not None:
                final_cost = chunk.provider_cost
            if chunk.response_metrics:
                final_metrics = chunk.response_metrics
            if chunk.raw_event is not None:
                last_raw_event = chunk.raw_event

        text = "".join(text_parts)

        if request.parse_json:
            res = parse_json_from_model_output_detailed(text)
        else:
            res = LLMResponse(raw_text=text, content_text=text)

        # No fallback to the instance's LAST usage: that belonged to some other call (possibly
        # another session sharing this service) — reporting it would double-count or misattribute
        # (design/124 U1). No usage in the stream = unknown; the pipeline estimates and flags it.
        res.token_usage = final_usage
        res.tool_calls = last_tool_calls
        res.stream_chunks = chunks
        res.raw_completion = last_raw_event
        res.think = "".join(think_parts)
        res.provider_cost = final_cost
        res.debug['response_extraction'] = final_metrics

        if on_stream_end:
            res_val = on_stream_end(res)
            if inspect.isawaitable(res_val):
                await res_val

        return res

    async def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamChunk]:
        """
        Streaming API.

        Best-effort real streaming for OpenAI-compatible servers.
        If provider does not support streaming, callers can switch to `complete()`.
        """
        client: AsyncOpenAI = self._ensure_client()
        # json_schema only to a model that declares it; otherwise the schema rides in the system
        # prompt. Done once here so kwargs, rendering and a stream resume all see the same request.
        request = request.with_structured_fallback(self._capabilities.for_model(request.model))
        kwargs = self._request_kwargs(request)

        stream_kwargs = dict(kwargs)
        stream_kwargs["stream"] = True
        stream_kwargs.setdefault("stream_options", {"include_usage": True})

        last_event: ChatCompletionChunk | None = None
        final_usage: LLMTokenUsage | None = None
        from power_loop.llm_metrics import ProviderCost, ResponseMetricAccumulator
        response_metrics = ResponseMetricAccumulator(self._cfg.response_extraction)
        tool_call_store: dict[str, dict[str, Any]] = {}
        tool_call_order: list[str] = []
        tool_call_index_to_key: dict[int, str] = {}
        aggregated_text: str = ""
        current_request = request
        restart_count = 0
        max_restarts = max(0, int(self._cfg.stream_max_restarts or 0))
        from .attempt_scope import single_attempt
        resume_enabled = bool(self._cfg.stream_resume_on_error) and not single_attempt.get()

        def _as_dict(obj: Any) -> Any:
            if obj is None:
                return None
            if isinstance(obj, dict):
                return obj
            if hasattr(obj, "model_dump"):
                try:
                    return obj.model_dump()
                except Exception:
                    return obj
            # best-effort: fallback to __dict__
            try:
                return dict(obj.__dict__)
            except Exception:
                return obj

        def _merge_tool_calls_delta(delta_tool_calls_obj: Any) -> list[dict[str, Any]]:
            """
            Merge OpenAI-compatible streaming `delta.tool_calls` into an aggregated list.

            Delta items commonly look like:
            {"index":0,"id":"...","type":"function","function":{"name":"x","arguments":"{...partial..."}}
            """
            if not delta_tool_calls_obj:
                # return current snapshot
                return [tool_call_store[k] for k in tool_call_order]

            items = delta_tool_calls_obj
            # sometimes it's a single object
            if not isinstance(items, list):
                items = [items]

            for it in items:
                d = _as_dict(it)
                if not isinstance(d, dict):
                    continue
                explicit_id = d.get("id")
                delta_index = d.get("index")

                call_key: str
                if explicit_id and f"id_{explicit_id}" in tool_call_store:
                    call_key = f"id_{explicit_id}"
                elif isinstance(delta_index, int) and delta_index in tool_call_index_to_key:
                    call_key = tool_call_index_to_key[delta_index]
                elif explicit_id:
                    call_key = f"id_{explicit_id}"
                elif delta_index is not None:
                    call_key = f"index_{delta_index}"
                else:
                    # Ambiguous: neither id nor index (non-standard provider). Keying purely by this
                    # event's position collided sequential DISTINCT calls (both arriving at pos 0 in
                    # separate events) into one entry, merging their arguments (llm-transport-5).
                    # OpenAI-style streaming starts a call with a function NAME and continues it with
                    # arguments-only deltas: a name-bearing (or first-ever) ambiguous delta opens a
                    # NEW slot; an arguments-only one continues the most-recent call.
                    fn_obj = d.get("function")
                    fn_obj = fn_obj if isinstance(fn_obj, dict) else (_as_dict(fn_obj) or {})
                    if fn_obj.get("name") or not tool_call_order:
                        call_key = f"ambiguous_{len(tool_call_order)}"
                    else:
                        call_key = tool_call_order[-1]

                if call_key not in tool_call_store:
                    tool_call_store[call_key] = {
                        "id": explicit_id or call_key,
                        "type": d.get("type") or "function",
                        "function": {"name": "", "arguments": ""},
                    }
                    tool_call_order.append(call_key)

                entry = tool_call_store[call_key]
                if explicit_id and entry.get("id") != explicit_id:
                    entry["id"] = explicit_id
                if isinstance(delta_index, int):
                    tool_call_index_to_key[delta_index] = call_key
                fn = d.get("function") or {}
                if not isinstance(fn, dict):
                    fn = _as_dict(fn) or {}
                if isinstance(fn, dict):
                    name = fn.get("name")
                    if name:
                        entry["function"]["name"] = name
                    args = fn.get("arguments")
                    if args:
                        entry["function"]["arguments"] = (entry["function"].get("arguments") or "") + str(args)

                # allow other keys to be updated if present
                if d.get("type"):
                    entry["type"] = d.get("type")

            return [tool_call_store[k] for k in tool_call_order]

        def _extract_text_from_value(v: Any) -> str:
            if isinstance(v, str):
                return v
            if isinstance(v, list):
                parts: list[str] = []
                for item in v:
                    if isinstance(item, str):
                        parts.append(item)
                    elif isinstance(item, dict):
                        t = item.get("text") or item.get("content")
                        if isinstance(t, str):
                            parts.append(t)
                    else:
                        d = _as_dict(item)
                        if isinstance(d, dict):
                            t2 = d.get("text") or d.get("content")
                            if isinstance(t2, str):
                                parts.append(t2)
                return "".join(parts)
            return ""

        def _extract_text_and_think_from_value(v: Any) -> tuple[str, str]:
            """Split provider content blocks into user-visible text vs reasoning text.

            Some OpenAI-compatible providers stream `delta.content` as a list of
            typed blocks (for example type=reasoning). Those reasoning blocks
            should feed the think panel, not the main chat text.
            """
            if isinstance(v, str):
                return v, ""

            if isinstance(v, list):
                text_parts: list[str] = []
                think_parts: list[str] = []
                for item in v:
                    if isinstance(item, str):
                        text_parts.append(item)
                        continue

                    data = item if isinstance(item, dict) else _as_dict(item)
                    if not isinstance(data, dict):
                        continue

                    part = _extract_text_from_value(data.get("text") or data.get("content"))
                    if not part:
                        continue

                    item_type = str(data.get("type") or data.get("role") or "").lower()
                    if any(tag in item_type for tag in ("reason", "think", "thought")):
                        think_parts.append(part)
                    else:
                        text_parts.append(part)

                return "".join(text_parts), "".join(think_parts)

            data = _as_dict(v)
            if isinstance(data, dict):
                part = _extract_text_from_value(data.get("text") or data.get("content"))
                if not part:
                    return "", ""
                item_type = str(data.get("type") or data.get("role") or "").lower()
                if any(tag in item_type for tag in ("reason", "think", "thought")):
                    return "", part
                return part, ""

            return "", ""

        def _extract_delta_text_and_think(delta: Any) -> tuple[str, str]:
            if delta is None:
                return "", ""

            delta_dict = _as_dict(delta)
            if isinstance(delta_dict, dict):
                text, think = _extract_text_and_think_from_value(delta_dict.get("content"))
                if not text:
                    text = _extract_text_from_value(delta_dict.get("text"))
                for key in [
                    "reasoning_content",
                    "reasoning",
                    "thinking",
                    "thinking_content",
                    "reasoning_text",
                    "thought",
                ]:
                    extra_think = _extract_text_from_value(delta_dict.get(key))
                    if extra_think:
                        think += extra_think
                        break
                return text, think

            text, think = _extract_text_and_think_from_value(getattr(delta, "content", None))
            if not text:
                text = _extract_text_from_value(getattr(delta, "text", None))
            for key in [
                "reasoning_content",
                "reasoning",
                "thinking",
                "thinking_content",
                "reasoning_text",
                "thought",
            ]:
                extra_think = _extract_text_from_value(getattr(delta, key, None))
                if extra_think:
                    think += extra_think
                    break
            return text, think

        def _extract_delta_tool_calls(delta: Any) -> Any:
            if delta is None:
                return None

            delta_dict = _as_dict(delta)
            if isinstance(delta_dict, dict):
                tc = delta_dict.get("tool_calls")
                if tc:
                    return tc
                fc = delta_dict.get("function_call")
                if fc:
                    return [{"index": 0, "type": "function", "function": fc}]
                return None

            tc = getattr(delta, "tool_calls", None)
            if tc:
                return tc
            fc = getattr(delta, "function_call", None)
            if fc:
                return [{"index": 0, "type": "function", "function": fc}]
            return None

        rendered_messages = current_request.to_messages(
            self._capabilities.for_model(current_request.model))
        self._emit_debug_payload(method="stream", messages=rendered_messages, kwargs=stream_kwargs)

        while True:
            # Bind loop variables explicitly to avoid late-binding in the closure (B023).
            async def _open_stream(_msgs=rendered_messages, _kw=stream_kwargs) -> Any:
                return await client.chat.completions.create(messages=_msgs, **_kw)

            stream: Any = await self._with_retries(
                _open_stream,
                method="stream_open",
            )

            try:
                async for event in stream:
                    delta_text = ""
                    delta_think = ""
                    last_event = event
                    delta_tool_calls: Any = None
                    aggregated_tool_calls: list[dict[str, Any]] = []
                    delta = None
                    choices = getattr(event, "choices", None)
                    if isinstance(choices, list) and choices:
                        choice0 = choices[0]
                        delta = getattr(choice0, "delta", None)
                    elif isinstance(choices, tuple) and choices:
                        choice0 = choices[0]
                        delta = getattr(choice0, "delta", None)
                    else:
                        # Some providers emit non-content events with empty choices.
                        delta = getattr(event, "delta", None)
                        if delta is None and hasattr(event, "model_dump"):
                            try:
                                dumped = event.model_dump()
                                if isinstance(dumped, dict):
                                    dumped_choices = dumped.get("choices")
                                    if isinstance(dumped_choices, list) and dumped_choices:
                                        choice0 = dumped_choices[0]
                                        if isinstance(choice0, dict):
                                            delta = choice0.get("delta")
                            except Exception as exc:
                                logger.error("[llm:stream] failed to extract delta from model_dump: %s", exc)

                    try:
                        delta_text, delta_think = _extract_delta_text_and_think(delta)
                        delta_tool_calls = _extract_delta_tool_calls(delta)
                        aggregated_tool_calls = _merge_tool_calls_delta(delta_tool_calls)
                    except Exception as e:
                        delta_text = ""
                        delta_think = ""
                        delta_tool_calls = None
                        aggregated_tool_calls = [tool_call_store[k] for k in tool_call_order]
                        logger.error("[llm:stream] failed to extract delta: %s", e)
                    response_metrics.add(event)
                    extracted = response_metrics.snapshot()
                    if extracted['usage'] is not None:
                        final_usage = LLMTokenUsage(**extracted['usage'])
                        self._last_usage = dict(extracted['usage'])

                    if delta_text or delta_think or delta_tool_calls:
                        if delta_text:
                            aggregated_text += str(delta_text)
                        yield LLMStreamChunk(
                            delta_text=delta_text,
                            think=delta_think,
                            delta_tool_calls=delta_tool_calls,
                            tool_calls=aggregated_tool_calls,
                            token_usage=None,
                            raw_event=event,
                            is_final=False,
                        )
                break
            except Exception as e:
                if not resume_enabled or restart_count >= max_restarts:
                    raise
                restart_count += 1
                logger.warning(
                    "[llm:stream] stream interrupted, attempting resume (%s/%s): %s",
                    restart_count,
                    max_restarts,
                    e,
                )
                current_request = self._build_resume_request(request, aggregated_text)
                rendered_messages = current_request.to_messages(
                    self._capabilities.for_model(current_request.model))
                self._emit_debug_payload(method="stream-resume", messages=rendered_messages, kwargs=stream_kwargs)
                # Reset the tool-call accumulators before re-opening the stream: a tool
                # call only partially streamed on the interrupted attempt has truncated
                # (invalid-JSON) arguments and would otherwise be emitted ALONGSIDE the
                # resume attempt's call — a duplicate, and an empty-args execution of the
                # broken one. The resume request carries only the partial assistant TEXT
                # plus a continue instruction, which cannot validly resume a half-streamed
                # tool-call JSON, so the resume attempt must produce a fresh, complete set.
                # (aggregated_text is intentionally kept — it is the resume's basis.) (U3)
                tool_call_store.clear()
                tool_call_order.clear()
                tool_call_index_to_key.clear()
                continue

        extracted = response_metrics.snapshot()
        # Final chunk carries best-effort usage + last raw event so callers can inspect finish_reason, tool_calls, etc.
        yield LLMStreamChunk(
            delta_text="",
            think="",
            delta_tool_calls=None,
            tool_calls=[tool_call_store[k] for k in tool_call_order],
            token_usage=final_usage,
            raw_event=last_event,
            provider_cost=ProviderCost(**extracted['cost']) if extracted['cost'] is not None else None,
            response_metrics=response_metrics.snapshot(),
            is_final=True,
        )

    async def chat_completion(
            self,
            *,
            messages: Sequence[dict[str, str]],
            **kwargs: Any,
    ) -> Any:
        """
        Return the raw ChatCompletion object.
        """
        client = self._ensure_client()
        model = str(kwargs.get("model") or self._cfg.model)
        temperature = float(kwargs.get("temperature", self._cfg.temperature))
        max_tokens = int(kwargs.get("max_tokens", self._cfg.max_tokens))
        message_list = [dict(message) for message in messages]
        debug_kwargs = dict(kwargs)
        debug_kwargs.update({"model": model, "temperature": temperature, "max_tokens": max_tokens})
        self._emit_debug_payload(method="chat_completion", messages=message_list, kwargs=debug_kwargs)

        async def _call() -> Any:
            return await client.chat.completions.create(
                model=model,
                messages=message_list,
                temperature=temperature,
                max_tokens=max_tokens,
                **{k: v for k, v in kwargs.items() if k not in {"model", "temperature", "max_tokens"}},
            )

        resp: Any = await self._with_retries(_call, method="chat_completion")
        usage_obj = getattr(resp, "usage", None)
        if usage_obj is None and hasattr(resp, "model_dump"):
            try:
                dumped = resp.model_dump()
                if isinstance(dumped, dict):
                    usage_obj = dumped.get("usage") or dumped.get("x_openai_usage") or dumped.get("x_usage")
            except Exception:
                pass
        self._record_usage(usage_obj, method="chat_completion")
        return resp

    def _message_text(self, msg: Any) -> str:
        """
        Convert a ChatCompletionMessage content to plain text.
        Some providers may return list content; we best-effort stringify.
        """
        content = getattr(msg, "content", None)
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        # List/parts fallback
        try:
            return "".join(str(part) for part in content).strip()
        except Exception:
            return str(content).strip()

    # NOTE: `predict/chat/predict_stream` wrappers are provided as default methods on the Protocol.
