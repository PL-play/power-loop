"""「验证一次」模型上下文长度：只在运营点按钮时跑，结果经 api 记进能力表的 ``context_window`` 一项。

做法：先发一个小请求，用供应商返回的 prompt_tokens 算出填充文本每段多少 token；再按声明窗口的
95% 发一次请求（max_tokens 很小），看供应商收不收、实际计了多少 token。

- 收下且计费接近目标 → ``yes``：这个端点确实接受接近声明长度的输入。
- 4xx 且报错说的是长度 / 上下文 → ``no``：声明偏大，该改小。
- 收下但计费远低于目标（可能被悄悄截断）、没返回用量、限流 / 超时 / 余额等 → ``inconclusive``，原因写在证据里。

一次验证大约花掉一个满窗口的输入 token，所以不自动跑、不重试；结果只给人看，不自动改模型库。
"""

from __future__ import annotations

import dataclasses
import logging
import time
from collections.abc import Callable
from typing import Any

FILL_SHARE = 0.95
#: 计费低于目标的这个比例就当作没真收下这么长（可能被截断）。
ACCEPTED_SHARE = 0.9
CALIBRATION_UNITS = 400
UNIT = "上下文长度验证 context length check 0123456789. "
MAX_TOKENS = 64
TIMEOUT_S = 600.0
#: 4xx 报错里出现这些词 → 判定为长度超限。
LENGTH_WORDS = ("context", "too long", "too many tokens", "maximum", "length", "exceed", "上下文", "超出", "过长")
CHECK = "context_window"

log = logging.getLogger("admin.context_window_check")


def _request(units: int) -> Any:
    from power_loop._vendor.llm_client.interface import LLMRequest

    text = UNIT * units + "\n\n以上是填充文本，只回复 OK。"
    return LLMRequest(messages=[{"role": "user", "content": text}], max_tokens=MAX_TOKENS, temperature=0.0)


async def _complete(make: Callable[[], Any], request: Any) -> Any:
    service = make()
    try:
        return await service.complete(request)
    finally:
        close = getattr(service, "close", None)
        if close is not None:
            try:
                await close()
            except Exception:
                log.debug("context_window_check: closing the client failed", exc_info=True)


def _prompt_tokens(resp: Any) -> int | None:
    value = getattr(getattr(resp, "token_usage", None), "prompt_tokens", None)
    return value if isinstance(value, int) and value > 0 else None


async def verify_context_window(config: Any, declared: int, *,
                                service_factory: Callable[[Any], Any] | None = None) -> Any:
    """按 ``declared``（模型库填的窗口）验证一次，返回 power-loop 的 ``CheckResult``。"""
    from power_loop.runtime.capability_probe import (
        INCONCLUSIVE,
        NO,
        YES,
        CheckResult,
        _classify_error,
        _usage,
    )
    from power_loop.runtime.provider import create_llm_service_from_config

    factory = service_factory or create_llm_service_from_config
    cfg = dataclasses.replace(config, max_retries=0, max_tokens=MAX_TOKENS, temperature=0.0, timeout_s=TIMEOUT_S)
    secret = str(getattr(config, "api_key", "") or "")
    model = str(getattr(config, "model", "") or "")
    usage: dict[str, int] = {}

    def add(resp: Any) -> None:
        for key, value in _usage(resp).items():
            usage[key] = usage.get(key, 0) + value

    target = int(declared * FILL_SHARE)
    t0 = time.monotonic()

    def done(status: str, evidence: str, kind: str | None = None) -> Any:
        return CheckResult(CHECK, status, evidence[:300], int((time.monotonic() - t0) * 1000), dict(usage),
                           error_kind=kind)

    stage = "校准请求"
    try:
        calibration = await _complete(lambda: factory(cfg), _request(CALIBRATION_UNITS))
        add(calibration)
        measured = _prompt_tokens(calibration)
        if measured is None:
            return done(INCONCLUSIVE, "供应商没有返回 prompt_tokens，无法按 token 计量，没有发长请求", "empty")
        units = max(CALIBRATION_UNITS, int(CALIBRATION_UNITS * target / measured))
        stage = f"按声明 {declared:,} 的 {FILL_SHARE:.0%} 发送"
        resp = await _complete(lambda: factory(cfg), _request(units))
        add(resp)
    except Exception as exc:  # noqa: BLE001 — every failure becomes a recorded result
        status, evidence, kind = _classify_error(exc, about=LENGTH_WORDS, secret=secret, model=model)
        if status == NO and stage == "校准请求":
            status, kind = INCONCLUSIVE, "unknown"
        prefix = f"声明 {declared:,} 偏大：" if status == NO else f"{stage}失败："
        return done(status, prefix + evidence, kind)
    sent = _prompt_tokens(resp)
    if sent is None:
        return done(INCONCLUSIVE, "长请求被收下，但供应商没有返回 prompt_tokens，无法确认实际长度", "empty")
    if sent < target * ACCEPTED_SHARE:
        return done(INCONCLUSIVE, f"目标约 {target:,} token，供应商只计了 {sent:,}，可能被截断或计量不准", "unknown")
    return done(YES, f"声明 {declared:,}：发送 {sent:,} token（{sent / declared:.0%}）被接受")
