"""Anthropic Messages API transport for the shared ``LLMService`` interface."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Callable
from typing import Any

from anthropic import AsyncAnthropic

from .interface import AnthropicChatConfig, LLMRequest, LLMResponse, LLMService, LLMStreamChunk, LLMTokenUsage
from .llm_utils import parse_json_from_model_output_detailed

logger = logging.getLogger(__name__)


def _as_dict(obj: Any) -> dict[str, Any]:
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        try:
            dumped = obj.model_dump()
            if isinstance(dumped, dict):
                return dumped
        except Exception:
            return {}
    try:
        return dict(obj.__dict__)
    except Exception:
        return {}


class AnthropicMessagesLLMService(LLMService):
    """Native Anthropic-compatible Messages API client.

    The public ``LLMRequest`` shape stays OpenAI-like because the rest of
    power-loop speaks that canonical format. This transport converts at the
    edge: OpenAI tool schemas become Anthropic tools, assistant ``tool_calls``
    become ``tool_use`` blocks, and ``tool`` messages become ``tool_result``
    blocks.
    """

    def __init__(self, cfg: AnthropicChatConfig):
        self._cfg = cfg
        self._client: AsyncAnthropic | None = None
        self._last_usage: dict[str, Any] = {}
        logger.info(
            "Anthropic LLM: base_url=%s model=%s timeout_s=%s max_tokens=%s temperature=%s api_key=%s",
            cfg.base_url,
            cfg.model,
            cfg.timeout_s,
            cfg.max_tokens,
            cfg.temperature,
            "set" if bool(cfg.api_key) else "missing",
        )

    def get_last_token_usage(self) -> dict[str, Any]:
        return dict(self._last_usage or {})

    def _ensure_client(self) -> AsyncAnthropic:
        if self._client is None:
            self._client = AsyncAnthropic(
                api_key=self._cfg.api_key,
                base_url=self._cfg.base_url,
                timeout=self._cfg.timeout_s,
                max_retries=self._cfg.max_retries,
            )
        return self._client

    async def close(self) -> None:
        if self._client is None:
            return
        try:
            await self._client.close()
        finally:
            self._client = None

    def _usage_dict_from_any(self, usage: Any) -> dict[str, Any]:
        raw = _as_dict(usage)

        def _int_or_none(value: Any) -> int | None:
            try:
                return None if value is None else int(value)
            except Exception:
                return None

        prompt = _int_or_none(raw.get("input_tokens") or raw.get("prompt_tokens"))
        completion = _int_or_none(raw.get("output_tokens") or raw.get("completion_tokens"))
        total = _int_or_none(raw.get("total_tokens"))
        if total is None and prompt is not None and completion is not None:
            total = prompt + completion
        raw["prompt_tokens"] = prompt
        raw["completion_tokens"] = completion
        raw["total_tokens"] = total
        return raw

    def _usage_obj(self, usage: Any = None) -> LLMTokenUsage:
        d = self._last_usage if usage is None else self._usage_dict_from_any(usage)

        def _int_or_none(value: Any) -> int | None:
            try:
                return None if value is None else int(value)
            except Exception:
                return None

        return LLMTokenUsage(
            prompt_tokens=_int_or_none(d.get("prompt_tokens")),
            completion_tokens=_int_or_none(d.get("completion_tokens")),
            total_tokens=_int_or_none(d.get("total_tokens")),
        )

    def _record_usage(self, usage: Any) -> None:
        self._last_usage = self._usage_dict_from_any(usage)

    def _system_and_messages(self, request: LLMRequest) -> tuple[str | None, list[dict[str, Any]]]:
        system_parts: list[str] = []
        if request.system_prompt:
            system_parts.append(request.system_prompt)

        messages: list[dict[str, Any]] = []
        pending_tool_results: list[dict[str, Any]] = []

        def flush_tool_results() -> None:
            nonlocal pending_tool_results
            if pending_tool_results:
                messages.append({"role": "user", "content": pending_tool_results})
                pending_tool_results = []

        for raw in request.messages or []:
            msg = dict(raw)
            role = str(msg.get("role") or "user")
            if role == "system":
                content = self._text_from_content(msg.get("content"))
                if content:
                    system_parts.append(content)
                continue
            if role == "tool":
                pending_tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": str(msg.get("tool_call_id") or ""),
                    "content": self._text_from_content(msg.get("content")),
                })
                continue

            flush_tool_results()
            anthropic_role = "assistant" if role == "assistant" else "user"
            blocks = self._anthropic_content_blocks(msg)
            self._append_or_merge(messages, {"role": anthropic_role, "content": blocks})

        flush_tool_results()

        if request.response_format is not None:
            instruction = self._json_instruction(request.response_format)
            if instruction:
                system_parts.append(instruction)

        system = "\n\n".join(part for part in system_parts if part).strip() or None
        return system, messages

    def _anthropic_content_blocks(self, msg: dict[str, Any]) -> list[dict[str, Any]]:
        blocks: list[dict[str, Any]] = []
        text = self._text_from_content(msg.get("content"))
        if text:
            blocks.append({"type": "text", "text": text})

        for call in self._normalize_openai_tool_calls(msg.get("tool_calls") or msg.get("function_call")):
            fn = call.get("function") or {}
            name = str(fn.get("name") or "")
            if not name:
                continue
            blocks.append({
                "type": "tool_use",
                "id": str(call.get("id") or f"toolu_{len(blocks)}"),
                "name": name,
                "input": self._json_object(fn.get("arguments")),
            })

        if not blocks:
            blocks.append({"type": "text", "text": ""})
        return blocks

    def _append_or_merge(self, messages: list[dict[str, Any]], item: dict[str, Any]) -> None:
        if messages and messages[-1].get("role") == item.get("role"):
            prev = messages[-1].setdefault("content", [])
            if isinstance(prev, list) and isinstance(item.get("content"), list):
                prev.extend(item["content"])
                return
        messages.append(item)

    def _text_from_content(self, content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text") or item.get("content")
                    if isinstance(text, str):
                        parts.append(text)
            return "".join(parts)
        return str(content)

    def _normalize_openai_tool_calls(self, value: Any) -> list[dict[str, Any]]:
        if not value:
            return []
        if isinstance(value, dict) and value.get("name"):
            return [{"type": "function", "function": value}]
        items = value if isinstance(value, list) else [value]
        out: list[dict[str, Any]] = []
        for item in items:
            data = item if isinstance(item, dict) else _as_dict(item)
            if isinstance(data, dict):
                out.append(data)
        return out

    def _json_object(self, value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return value
        if isinstance(value, str) and value.strip():
            try:
                parsed = json.loads(value)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                return {"arguments": value}
        return {}

    def _json_instruction(self, response_format: dict[str, Any]) -> str:
        schema = response_format.get("json_schema")
        if not isinstance(schema, dict):
            return "Return only valid JSON. Do not wrap it in Markdown."
        payload = schema.get("schema")
        try:
            rendered = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        except Exception:
            rendered = "{}"
        return f"Return only valid JSON matching this JSON Schema. Do not wrap it in Markdown.\nSchema: {rendered}"

    def _anthropic_tools(self, tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
        if not tools:
            return None
        out: list[dict[str, Any]] = []
        for tool in tools:
            fn = tool.get("function") if isinstance(tool, dict) else None
            if not isinstance(fn, dict):
                continue
            name = fn.get("name")
            if not name:
                continue
            out.append({
                "name": str(name),
                "description": str(fn.get("description") or ""),
                "input_schema": fn.get("parameters") if isinstance(fn.get("parameters"), dict) else {"type": "object"},
            })
        return out or None

    def _tool_choice(self, choice: Any) -> Any:
        if choice in (None, "auto"):
            return None
        if choice == "none":
            return {"type": "none"}
        if isinstance(choice, dict):
            if "dashscope.aliyuncs.com" in self._cfg.base_url:
                logger.warning(
                    "Skipping forced Anthropic tool_choice for DashScope endpoint; "
                    "the endpoint rejects object tool_choice in thinking mode."
                )
                return None
            fn = choice.get("function")
            if isinstance(fn, dict) and fn.get("name"):
                return {"type": "tool", "name": str(fn["name"])}
            if choice.get("name"):
                return {"type": "tool", "name": str(choice["name"])}
        return None

    def _request_kwargs(self, request: LLMRequest) -> dict[str, Any]:
        system, messages = self._system_and_messages(request)
        kwargs: dict[str, Any] = dict(request.extra or {})
        kwargs["model"] = request.model or self._cfg.model
        kwargs["messages"] = messages
        kwargs["max_tokens"] = int(request.max_tokens if request.max_tokens is not None else self._cfg.max_tokens)

        temperature = request.temperature if request.temperature is not None else self._cfg.temperature
        if temperature is not None:
            kwargs["temperature"] = float(temperature)
        if system:
            kwargs["system"] = system
        tools = self._anthropic_tools(request.tools)
        if tools:
            kwargs["tools"] = tools
        tool_choice = self._tool_choice(request.tool_choice)
        if tool_choice:
            kwargs["tool_choice"] = tool_choice
        return kwargs

    async def complete(
        self,
        request: LLMRequest,
        *,
        on_chunk_delta_text: Callable[[str], Any] | None = None,
        on_chunk_think: Callable[[str], Any] | None = None,
        on_stream_end: Callable[[LLMResponse], Any] | None = None,
    ) -> LLMResponse:
        client = self._ensure_client()
        kwargs = self._request_kwargs(request)
        response = await client.messages.create(**kwargs)
        text, think, tool_calls = self._extract_response(response)
        self._record_usage(getattr(response, "usage", None))

        if request.parse_json:
            result = parse_json_from_model_output_detailed(text)
        else:
            result = LLMResponse(raw_text=text, content_text=text)
        result.raw_completion = response
        result.raw_message = response
        result.think = think
        result.tool_calls = tool_calls
        result.token_usage = self._usage_obj()

        if on_chunk_delta_text and text:
            maybe = on_chunk_delta_text(text)
            if hasattr(maybe, "__await__"):
                await maybe
        if on_chunk_think and think:
            maybe = on_chunk_think(think)
            if hasattr(maybe, "__await__"):
                await maybe
        if on_stream_end:
            maybe = on_stream_end(result)
            if hasattr(maybe, "__await__"):
                await maybe
        return result

    async def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamChunk]:
        response = await self.complete(request)
        if response.raw_text or response.think or response.tool_calls:
            yield LLMStreamChunk(
                delta_text=response.raw_text,
                think=response.think,
                tool_calls=response.tool_calls,
                token_usage=None,
                raw_event=response.raw_completion,
                is_final=False,
            )
        yield LLMStreamChunk(
            delta_text="",
            think="",
            tool_calls=response.tool_calls,
            token_usage=response.token_usage,
            raw_event=response.raw_completion,
            is_final=True,
        )

    def _extract_response(self, response: Any) -> tuple[str, str, list[dict[str, Any]]]:
        text_parts: list[str] = []
        think_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        for block in getattr(response, "content", []) or []:
            data = _as_dict(block)
            block_type = str(data.get("type") or "").lower()
            if block_type == "text":
                text = data.get("text")
                if isinstance(text, str):
                    text_parts.append(text)
            elif block_type in {"thinking", "reasoning"}:
                text = data.get("thinking") or data.get("text")
                if isinstance(text, str):
                    think_parts.append(text)
            elif block_type == "tool_use":
                name = data.get("name")
                if not name:
                    continue
                tool_calls.append({
                    "id": data.get("id") or f"toolu_{len(tool_calls)}",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(data.get("input") or {}, ensure_ascii=False),
                    },
                })
        return "".join(text_parts), "".join(think_parts), tool_calls


__all__ = ["AnthropicMessagesLLMService"]
