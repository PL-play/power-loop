from typing import TypedDict, Literal, Any, Dict, List, Optional, AsyncIterator, Protocol, Callable

from dataclasses import dataclass, field

LLMRole = Literal["system", "user", "assistant", "tool"]
LLMContent = str | List[Dict[str, Any]]


class LLMMessage(TypedDict, total=False):
    role: LLMRole
    content: LLMContent
    name: str
    tool_call_id: str
    # OpenAI-compatible: assistant may include tool_calls / function_call when requesting tools.
    tool_calls: Any
    function_call: Any


class LLMToolFunction(TypedDict, total=False):
    name: str
    description: str
    parameters: Dict[str, Any]


class LLMTool(TypedDict, total=False):
    # OpenAI-compatible tool schema: {"type":"function","function":{...}}
    type: Literal["function"]
    function: LLMToolFunction


@dataclass
class LLMRequest:
    """
    Canonical request object for LLM calls.

    - Put conversation turns in `messages` (no system message here by default)
    - Put system instruction in `system_prompt` (we will prepend it during execution)
    """

    # Message history. Rule: do NOT include a system message here; use `system_prompt`.
    messages: List[LLMMessage | Dict]
    system_prompt: Optional[str] = None

    # Common generation params (optional; implementation may ignore some)
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None

    # Structured output helpers
    parse_json: bool = False

    # Reasoning switch for providers supporting `reason` flag.
    # Default enabled as requested; set to False to disable.
    reason: Optional[bool] = True

    # Tool calling (OpenAI-compatible). Keep optional to avoid forcing every impl to support it.
    tools: Optional[List[LLMTool | Dict]] = None
    tool_choice: Optional[Any] = None

    # Provider-specific passthrough
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_messages(self, capabilities: Any = None) -> List[Dict[str, Any]]:
        """
        Normalize messages for OpenAI-compatible APIs.
        """
        msgs: List[Dict[str, Any]] = []
        if self.system_prompt:
            msgs.append({"role": "system", "content": self.system_prompt})

        for raw_msg in self.messages or []:
            msg = dict(raw_msg)
            if capabilities is not None and "content" in msg:
                from .multimodal import render_message_content

                msg["content"] = render_message_content(
                    msg.get("content"),
                    role=str(msg.get("role") or "user"),
                    capabilities=capabilities,
                )
            msgs.append(msg)
        return msgs

    @classmethod
    def from_prompt(
            cls,
            *,
            prompt: str,
            system_prompt: Optional[str] = None,
            model: Optional[str] = None,
            temperature: Optional[float] = None,
            max_tokens: Optional[int] = None,
            parse_json: bool = False,
            reason: Optional[bool] = True,
            **kwargs: Any,
    ) -> "LLMRequest":
        return cls(
            messages=[{"role": "user", "content": prompt}],
            system_prompt=system_prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            parse_json=parse_json,
            reason=reason,
            extra=dict(kwargs or {}),
        )


@dataclass
class LLMTokenUsage:
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None

    prompt_audio_tokens: Optional[int] = None
    prompt_cached_tokens: Optional[int] = None
    prompt_cache_miss_tokens: Optional[int] = None
    prompt_text_tokens: Optional[int] = None
    prompt_image_tokens: Optional[int] = None

    completion_reasoning_tokens: Optional[int] = None
    completion_audio_tokens: Optional[int] = None
    completion_text_tokens: Optional[int] = None
    completion_image_tokens: Optional[int] = None

    accepted_prediction_tokens: Optional[int] = None
    rejected_prediction_tokens: Optional[int] = None

    cached_tokens: Optional[int] = None
    cache_hit_tokens: Optional[int] = None
    cache_miss_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None
    accepted_tokens: Optional[int] = None
    rejected_tokens: Optional[int] = None

    def as_dict(self) -> Dict[str, Optional[int]]:
        return dict(self.__dict__)

    def to_log_str(self) -> str:
        parts = [f"{k}={v}" for k, v in self.__dict__.items() if v is not None]
        return f"LLMTokenUsage({', '.join(parts)})"

    def __str__(self) -> str:
        return self.to_log_str()

    def __repr__(self) -> str:
        return self.to_log_str()


@dataclass
class LLMStreamChunk:
    """
    Streaming chunk event.
    """

    delta_text: str = ""
    # Best-effort streamed thinking/reasoning text.
    think: str = ""
    # Optional tool/function-call delta payload (provider-specific shape).
    # For OpenAI-compatible streaming this is typically `choices[0].delta.tool_calls`.
    delta_tool_calls: Any = None
    # Aggregated tool calls (best-effort normalized), built by the LLMService streaming implementation.
    # Shape (OpenAI-compatible):
    # [{"id": "...", "type": "function", "function": {"name": "...", "arguments": "..."}}]
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    # Best-effort usage. Some providers only send usage in the final event.
    token_usage: Optional[LLMTokenUsage] = None
    raw_event: Any = None
    is_final: bool = False


@dataclass
class LLMResponse:
    """
    Unified response wrapper for LLM calls, with optional JSON parsing.

    Conceptually we keep 4 layers (some may be empty depending on output):
    - raw_text: extracted message content text
    - raw_json_data: if raw_text can be parsed as a JSON object directly
    - content_text: extracted "business payload" text (typically a JSON substring)
    - json_data: parsed JSON object from content_text (after fallback/repairs)
    """

    raw_text: str
    # Raw message/completion object references (provider-specific)
    raw_message: Any = None
    raw_completion: Any = None

    raw_json_data: Dict[str, Any] = field(default_factory=dict)
    raw_json_error: Optional[str] = None

    content_text: str = ""
    think: str = ""
    json_data: Dict[str, Any] = field(default_factory=dict)
    token_usage: LLMTokenUsage = field(default_factory=LLMTokenUsage)
    parse_error: Optional[str] = None
    debug: Dict[str, Any] = field(default_factory=dict)
    # Best-effort extracted tool calls (if any). Providers may expose this on raw_message/raw_completion.
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    # Raw streamed chunks captured when complete() is implemented via stream aggregation.
    stream_chunks: List[LLMStreamChunk] = field(default_factory=list)

    def get_tool_calls(self) -> List[Dict[str, Any]]:
        """
        Best-effort accessor for tool calls.

        Preference order:
        - explicit `self.tool_calls` (if caller filled it)
        - `raw_message.tool_calls` / `raw_message.function_call`
        - `raw_completion.choices[0].message.tool_calls` / `.function_call`
        - `model_dump()` dict fallbacks
        """

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
            try:
                return dict(obj.__dict__)
            except Exception:
                return obj

        def _normalize_tool_calls(obj: Any) -> List[Dict[str, Any]]:
            if not obj:
                return []
            if isinstance(obj, list):
                out: List[Dict[str, Any]] = []
                for it in obj:
                    d = _as_dict(it)
                    if isinstance(d, dict):
                        out.append(d)
                return out
            d = _as_dict(obj)
            return [d] if isinstance(d, dict) else []

        def _normalize_function_call(obj: Any) -> List[Dict[str, Any]]:
            if not obj:
                return []
            d = _as_dict(obj)
            if not isinstance(d, dict):
                return []
            # legacy shape: {"name": "...", "arguments": "..."}
            return [{"type": "function", "function": {"name": d.get("name"), "arguments": d.get("arguments")}}]

        if self.tool_calls:
            return list(self.tool_calls)

        # raw_message direct
        msg = self.raw_message
        tcs = getattr(msg, "tool_calls", None) if msg is not None else None
        if tcs:
            return _normalize_tool_calls(tcs)
        fc = getattr(msg, "function_call", None) if msg is not None else None
        if fc:
            return _normalize_function_call(fc)

        # raw_completion -> first message
        comp = self.raw_completion
        try:
            msg2 = comp.choices[0].message if comp is not None else None
            tcs2 = getattr(msg2, "tool_calls", None) if msg2 is not None else None
            if tcs2:
                return _normalize_tool_calls(tcs2)
            fc2 = getattr(msg2, "function_call", None) if msg2 is not None else None
            if fc2:
                return _normalize_function_call(fc2)
        except Exception:
            pass

        # model_dump fallback (raw_message)
        if msg is not None and hasattr(msg, "model_dump"):
            try:
                dumped = msg.model_dump()
                if isinstance(dumped, dict):
                    if dumped.get("tool_calls"):
                        return _normalize_tool_calls(dumped.get("tool_calls"))
                    if dumped.get("function_call"):
                        return _normalize_function_call(dumped.get("function_call"))
            except Exception:
                pass

        # model_dump fallback (raw_completion)
        if comp is not None and hasattr(comp, "model_dump"):
            try:
                dumped = comp.model_dump()
                if isinstance(dumped, dict):
                    choices = dumped.get("choices") or []
                    if choices and isinstance(choices, list) and isinstance(choices[0], dict):
                        msgd = choices[0].get("message") or {}
                        if isinstance(msgd, dict):
                            if msgd.get("tool_calls"):
                                return _normalize_tool_calls(msgd.get("tool_calls"))
                            if msgd.get("function_call"):
                                return _normalize_function_call(msgd.get("function_call"))
            except Exception:
                pass

        return []

    @property
    def ok(self) -> bool:
        return bool(self.json_data) and self.parse_error is None

    def to_log_str(
            self,
            *,
            max_raw_chars: int = 600,
            max_content_chars: int = 600,
            include_debug: bool = False,
    ) -> str:
        import json as _json

        def _trunc(s: str, n: int) -> str:
            s = s or ""
            if n <= 0:
                return ""
            if len(s) <= n:
                return s
            return s[: n - 3] + "..."

        parts = [
            f"ok={self.ok}",
            f"token_usage={self.token_usage.as_dict()}",
            f"has_raw_message={self.raw_message is not None}",
            f"has_raw_completion={self.raw_completion is not None}",
            f"tool_calls={len(self.tool_calls) or len(self.get_tool_calls())}",
            f"raw_json_ok={bool(self.raw_json_data) and self.raw_json_error is None}",
            f"parse_error={self.parse_error!r}",
            f"strategy={self.debug.get('strategy')!r}",
        ]
        raw_preview = _trunc(self.raw_text, max_raw_chars)
        content_preview = _trunc(self.content_text, max_content_chars)
        parts.append(f"raw_text={raw_preview!r}")
        if self.content_text:
            parts.append(f"content_text={content_preview!r}")
        if self.json_data:
            try:
                parts.append(
                    "json_data="
                    + _json.dumps(self.json_data, ensure_ascii=False, sort_keys=True)[: max_content_chars]
                )
            except Exception:
                parts.append("json_data=<unserializable>")
        if include_debug and self.debug:
            try:
                parts.append("debug=" + _json.dumps(self.debug, ensure_ascii=False, sort_keys=True))
            except Exception:
                parts.append(f"debug={self.debug!r}")
        return "LLMResponse(" + ", ".join(parts) + ")"

    def __str__(self) -> str:
        return self.to_log_str()

    def __repr__(self) -> str:
        return self.to_log_str()


class LLMService(Protocol):
    """
    Unified LLM interface (no LangChain dependency).

    Why this design:
    - There is ONE canonical request shape: `LLMRequest` (messages + optional system_prompt)
    - There is ONE canonical non-streaming call: `complete(request) -> LLMResponse`
    - Streaming uses the same request: `stream(request) -> AsyncIterator[LLMStreamChunk]`
    - Convenience wrappers (`predict`, `chat`, `predict_stream`) are thin sugar over the above.
    """

    async def complete(
            self,
            request: "LLMRequest",
            *,
            on_chunk_delta_text: Optional[Callable[[str], Any]] = None,
            on_chunk_think: Optional[Callable[[str], Any]] = None,
            on_stream_end: Optional[Callable[["LLMResponse"], Any]] = None,
    ) -> LLMResponse:
        """Preferred API (non-streaming). Can optional receive chunk hooks."""
        ...

    async def stream(self, request: "LLMRequest") -> AsyncIterator["LLMStreamChunk"]:
        """Preferred API (streaming)."""
        ...

    async def close(self) -> None:
        """Close underlying HTTP resources, if any."""
        ...

    # -----------------------
    # Convenience wrappers
    # -----------------------

    async def predict(
            self,
            prompt: str,
            system_prompt: Optional[str] = None,
            **kwargs: Any,
    ) -> str:
        """
        Convenience: single-turn text completion.

        Note: this simply builds an `LLMRequest` with one user message.
        """
        req = LLMRequest.from_prompt(prompt=prompt, system_prompt=system_prompt, **kwargs)
        resp = await self.complete(req)
        return (resp.raw_text or "").strip()

    async def chat(
            self,
            messages: List[Dict[str, str]],
            system_prompt: Optional[str] = None,
            **kwargs: Any,
    ) -> str:
        """
        Convenience: multi-turn chat returning plain text.

        Rule: do NOT include a system message in `messages`. If you need one, pass `system_prompt`.
        """
        req = LLMRequest(messages=messages, system_prompt=system_prompt, **kwargs)
        resp = await self.complete(req)
        return (resp.raw_text or "").strip()

    async def predict_stream(
            self,
            prompt: str,
            system_prompt: Optional[str] = None,
            **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        Convenience: single-turn streaming text.
        """
        req = LLMRequest.from_prompt(prompt=prompt, system_prompt=system_prompt, **kwargs)
        async for chunk in self.stream(req):
            if chunk.delta_text:
                yield chunk.delta_text


@dataclass
class OpenAICompatibleChatConfig:
    base_url: str
    api_key: str
    model: str
    timeout_s: float = 180.0
    max_tokens: int = 8000
    temperature: float = 0.0
    max_retries: int = 3
    retry_base_delay_s: float = 0.5
    stream_resume_on_error: bool = False
    stream_max_restarts: int = 0
    stream_resume_instruction: str = "继续，从你上次中断的位置继续输出。不要重复已经输出的内容。"
    capability_overrides: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_ready(self) -> bool:
        return bool(self.base_url and self.api_key and self.model)
