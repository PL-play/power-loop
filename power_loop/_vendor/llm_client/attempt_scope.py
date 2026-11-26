"""Disable hidden SDK retries and redirects when the caller maintains a per-attempt ledger."""
from contextvars import ContextVar
from typing import Any

single_attempt: ContextVar[bool] = ContextVar('llm_single_attempt', default=False)


def _bind_single_attempt_send(http_client: Any) -> None:
    if http_client is None or getattr(http_client, "_power_loop_single_attempt_send", False):
        return
    orig_send = getattr(http_client, "send", None)
    if not callable(orig_send):
        return

    async def _send(request: Any, *args: Any, **kwargs: Any) -> Any:
        if single_attempt.get():
            kwargs["follow_redirects"] = False
        return await orig_send(request, *args, **kwargs)

    http_client.send = _send
    http_client._power_loop_single_attempt_send = True


def single_attempt_client(client: Any) -> Any:
    """Return a client view that issues at most one HTTP request per attempt.

    Disables SDK-level retries (``max_retries=0``) and per-request HTTP
    redirects (``follow_redirects=False`` at ``http_client.send`` while
    ``single_attempt`` is active) without altering unmanaged callers or
    creating a new connection pool.
    """
    if not single_attempt.get():
        return client
    scoped = client.with_options(max_retries=0)
    _bind_single_attempt_send(getattr(scoped, "_client", None))
    return scoped
