"""Design 218 H7 regression tests: P1-20 ~ P1-23 and Phase 2 version 14.1.0."""

from __future__ import annotations

import ast
import asyncio
import importlib.metadata
import io
import os
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import httpx
import pytest

import power_loop
from power_loop import LLMProviderConfig, create_default_tool_registry, create_llm_service_from_config
from power_loop._vendor.llm_client.interface import (
    LLMRequest,
    LLMResponse,
    LLMTokenUsage,
    OpenAICompatibleChatConfig,
)
from power_loop._vendor.llm_client.llm_factory import ExactAsyncOpenAI, OpenAICompatibleChatLLMService
from power_loop.core.agent_context import (
    reset_current_cancel_token,
    reset_current_loop,
    set_current_cancel_token,
    set_current_loop,
)
from power_loop.llm_metrics import ResponseMetricAccumulator, ResponseMetrics, compile_extractor
from power_loop.runtime.bash_capture import BASH_CAPTURE_MAX_BYTES, BashOutputCapture
from power_loop.runtime.cancellation import CancellationToken
from power_loop.runtime.env import RuntimeEnv, runtime_env_context
from power_loop.runtime.tool_context import ToolContextConfig, active_engine
from power_loop.tools import background_shell
from power_loop.tools import default_tools as dt

pytestmark = pytest.mark.unit


# ── P1-20: LLMResponse.token_usage None safety ──────────────────────────────


@pytest.mark.asyncio
async def test_h7_p1_20_llm_response_token_usage_none_log_str_and_repr() -> None:
    resp = LLMResponse(raw_text="hi", token_usage=None)
    log_str = resp.to_log_str()
    assert "token_usage=None" in log_str
    assert "raw_text='hi'" in log_str
    assert str(resp) == log_str
    assert repr(resp) == log_str

    # Also verify real stream completion without usage produces token_usage=None safely.
    class _NoUsageCompletions:
        async def create(self, **kwargs: Any) -> Any:
            async def _gen() -> Any:
                yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="hello"))])

            return _gen()

    svc = OpenAICompatibleChatLLMService(
        OpenAICompatibleChatConfig(base_url="https://example.invalid/v1", api_key="sk-test", model="m1")
    )
    svc._client = cast(Any, SimpleNamespace(chat=SimpleNamespace(completions=_NoUsageCompletions())))
    completed = await svc.complete(LLMRequest(messages=[{"role": "user", "content": "hi"}]))
    assert completed.token_usage is None
    assert "token_usage=None" in repr(completed)
    assert "raw_text='hello'" in str(completed)


# ── P1-21: BashOutputCapture / background_shell / BashSession bounds & cancel ─


def test_h7_p1_21_bash_capture_hard_byte_cap_on_large_output(tmp_path: Path) -> None:
    # Direct BashOutputCapture cap enforcement
    cap_path = tmp_path / "direct.txt"
    cap = BashOutputCapture(cap_path, max_bytes=1024)
    with cap:
        for _ in range(500):
            cap.write_line("X" * 100)
    assert cap.truncated is True
    assert cap.bytes_written <= 1024
    assert cap_path.stat().st_size <= 1024
    assert cap.dropped_bytes > 0
    text = cap.read_text()
    assert "bash capture truncated at 1024 bytes" in text
    assert len(text.encode("utf-8")) < 1280

    # End-to-end run_bash with active_engine emitting > 3 MiB
    token = active_engine.set(SimpleNamespace(config=ToolContextConfig()))
    try:
        with runtime_env_context(RuntimeEnv(workspace_dir=tmp_path)):
            out = dt.run_bash(
                "python3 -c 'import sys; line=\"A\"*1023+\"\\n\"; sys.stdout.write(line*3500)'",
                description="generate >3 MiB output",
                timeout=30,
            )
        assert out.startswith("exit_code=0\n")
        assert "bash capture truncated" in out
        assert len(out.encode("utf-8")) <= BASH_CAPTURE_MAX_BYTES + 512
    finally:
        active_engine.reset(token)
        dt.close_all_bash_sessions()


def test_h7_p1_21_bash_session_bounded_queue_and_no_newline_buf(tmp_path: Path) -> None:
    sess = dt.BashSession(tmp_path)
    try:
        assert sess._q.maxsize == dt._BASH_QUEUE_MAXSIZE
        assert sess._q.maxsize > 0
        with runtime_env_context(RuntimeEnv(workspace_dir=tmp_path)):
            # Emit > 3 MiB with NO newlines at all; reader buf must chunk & stay bounded.
            out = sess.execute(
                "python3 -c 'import sys; sys.stdout.write(\"Z\" * (3 * 1024 * 1024))'",
                timeout=30,
            )
        assert out.startswith("exit_code=0\n")
        assert len(out) <= dt.RESULT_MAX_CHARS + 256
    finally:
        sess.close()


def test_h7_p1_21_background_shell_bounded_memory_on_large_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Direct _BoundedStreamReader test (> 4 MiB input capped at 64 KiB)
    payload = "B" * (4 * 1024 * 1024)
    reader = background_shell._BoundedStreamReader(io.StringIO(payload), max_bytes=64 * 1024)
    reader.start()
    assert reader.join(timeout=5)
    assert reader.truncated is True
    assert reader.omitted_bytes > 3 * 1024 * 1024
    rendered = reader.text()
    assert len(rendered.encode("utf-8")) <= 64 * 1024 + 256

    # End-to-end background_shell.execute() with > 3 MiB output
    monkeypatch.setattr(background_shell, "BACKGROUND_SHELL_MAX_OUTPUT_BYTES", 64 * 1024)
    fake_loop = SimpleNamespace(_runtime_worker_context=SimpleNamespace(session_id="s_bg_cap", send_index=1))
    loop_tok = set_current_loop(cast(Any, fake_loop))
    cancel_tok = set_current_cancel_token(CancellationToken())
    try:
        with runtime_env_context(RuntimeEnv(workspace_dir=tmp_path)):
            out = background_shell.execute(
                "python3 -c 'import sys; sys.stdout.write(\"M\" * (3 * 1024 * 1024))'",
                timeout=30,
            )
        assert out.startswith("exit_code=0\n")
        assert len(out) <= dt.RESULT_MAX_CHARS + 256
    finally:
        reset_current_cancel_token(cancel_tok)
        reset_current_loop(loop_tok)


def test_h7_p1_21_background_shell_cancel_stops_process_and_closes_streams(tmp_path: Path) -> None:
    pidfile = tmp_path / "bg_sleep.pid"
    cancel_token = CancellationToken()
    fake_loop = SimpleNamespace(_runtime_worker_context=SimpleNamespace(session_id="s_bg_cancel", send_index=1))
    exc_box: list[BaseException] = []

    def _run() -> None:
        loop_tok = set_current_loop(cast(Any, fake_loop))
        cancel_tok = set_current_cancel_token(cancel_token)
        try:
            with runtime_env_context(RuntimeEnv(workspace_dir=tmp_path)):
                background_shell.execute(
                    f"setsid sh -c 'echo $$ > {pidfile}; exec sleep 30' & wait",
                    timeout=30,
                )
        except BaseException as exc:
            exc_box.append(exc)
        finally:
            reset_current_cancel_token(cancel_tok)
            reset_current_loop(loop_tok)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    wait_deadline = time.monotonic() + 5.0
    while time.monotonic() < wait_deadline and not (pidfile.exists() and pidfile.read_text().strip()):
        time.sleep(0.02)
    assert pidfile.exists() and pidfile.read_text().strip()
    child_pid = int(pidfile.read_text().strip())

    t0 = time.monotonic()
    cancel_token.cancel("stop_bg")
    t.join(timeout=3.0)
    assert not t.is_alive()
    assert time.monotonic() - t0 < 3.0
    assert len(exc_box) == 1 and isinstance(exc_box[0], power_loop.CancellationRequested)
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and _proc_state(child_pid) is not None:
        time.sleep(0.05)
    assert _proc_state(child_pid) is None


def _proc_state(pid: int) -> str | None:
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8") as fh:
            return fh.read().split(") ", 1)[1].split()[0]
    except OSError:
        return None


def test_h7_p1_21_bash_foreground_cancel_stops_sleep_within_2s_and_no_zombie(tmp_path: Path) -> None:
    pidfile = tmp_path / "sleep.pid"
    sess = dt.BashSession(tmp_path)
    old_shell_pid = sess._proc.pid if sess._proc is not None else None
    cancel_token = CancellationToken()
    result_box: dict[str, Any] = {}

    def _run() -> None:
        tok = set_current_cancel_token(cancel_token)
        try:
            with runtime_env_context(RuntimeEnv(workspace_dir=tmp_path)):
                result_box["out"] = sess.execute(
                    f"setsid sh -c 'echo $$ > {pidfile}; exec sleep 30' & wait",
                    timeout=30,
                )
        finally:
            reset_current_cancel_token(tok)

    try:
        t = threading.Thread(target=_run, daemon=True)
        t.start()
        wait_deadline = time.monotonic() + 5.0
        while time.monotonic() < wait_deadline and not (pidfile.exists() and pidfile.read_text().strip()):
            time.sleep(0.02)
        assert pidfile.exists() and pidfile.read_text().strip(), "sleep command did not start"
        child_pid = int(pidfile.read_text().strip())

        t0 = time.monotonic()
        cancel_token.cancel("user_cancelled")
        t.join(timeout=2.0)
        elapsed = time.monotonic() - t0
        assert not t.is_alive(), "BashSession.execute did not abort within 2s after cancel_token.cancel()"
        assert elapsed < 2.0
        assert "cancelled" in result_box["out"].lower()

        # Verify child process was killed and old shell process is NOT a zombie.
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and _proc_state(child_pid) is not None:
            time.sleep(0.05)
        assert _proc_state(child_pid) is None
        if old_shell_pid is not None:
            assert _proc_state(old_shell_pid) != "Z"

        # Session was restarted cleanly and still works.
        with runtime_env_context(RuntimeEnv(workspace_dir=tmp_path)):
            assert "after-cancel" in sess.execute("echo after-cancel", timeout=5)
    finally:
        sess.close()


def test_h7_p1_21_restart_and_close_wait_after_kill_and_close_stdin(tmp_path: Path) -> None:
    events: list[str] = []

    class _FakeStdin:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True
            events.append("stdin.close")

    class _StubbornProc:
        def __init__(self) -> None:
            self.stdin = _FakeStdin()
            self.wait_timeouts: list[float | None] = []

        def poll(self) -> int | None:
            return None

        def terminate(self) -> None:
            events.append("terminate")

        def kill(self) -> None:
            events.append("kill")

        def wait(self, timeout: float | None = None) -> int:
            self.wait_timeouts.append(timeout)
            if len(self.wait_timeouts) == 1:
                raise subprocess.TimeoutExpired("bash", timeout or 0)
            events.append(f"wait_after_kill:{timeout}")
            return -9

    sess = dt.BashSession.__new__(dt.BashSession)
    sess._lock = threading.Lock()
    sess._reader_stop = threading.Event()
    sess._master_fd = None
    fake_proc = _StubbornProc()
    sess._proc = cast(Any, fake_proc)

    sess.close()
    assert fake_proc.stdin.closed is True
    assert events == ["stdin.close", "terminate", "kill", "wait_after_kill:2"]
    assert sess._proc is None

    # Also verify a real BashSession closes stdin and leaves no zombie on close().
    real = dt.BashSession(tmp_path)
    proc = real._proc
    assert proc is not None and proc.stdin is not None
    pid = proc.pid
    real.close()
    assert proc.stdin.closed is True
    assert _proc_state(pid) != "Z"


# ── P1-22: ExactAsyncOpenAI max_retries=0, stream closing, metrics & cache ──


@pytest.mark.asyncio
async def test_h7_p1_22_exact_async_openai_max_retries_zero() -> None:
    direct = ExactAsyncOpenAI(base_url="https://example.invalid/v1", api_key="sk-test")
    try:
        assert direct.max_retries == 0
    finally:
        await direct.close()

    svc = OpenAICompatibleChatLLMService(
        OpenAICompatibleChatConfig(
            base_url="https://example.invalid/v1",
            api_key="sk-test",
            model="m1",
            max_retries=3,
        )
    )
    try:
        client = svc._ensure_client()
        assert isinstance(client, ExactAsyncOpenAI)
        assert client.max_retries == 0
    finally:
        await svc.close()

    # With cfg.max_retries=0 (as capability_probe uses), a 503 hits the server exactly ONCE.
    calls = 0

    def _transport(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(503, json={"error": {"message": "overloaded"}})

    probe_cfg = LLMProviderConfig(
        base_url="https://example.invalid/v1",
        api_key="sk-test",
        model="m1",
        max_retries=0,
    )
    probe_svc = create_llm_service_from_config(probe_cfg)
    probe_svc._client = ExactAsyncOpenAI(
        base_url=probe_cfg.base_url,
        api_key=probe_cfg.api_key,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(_transport)),
    )
    try:
        assert probe_svc._client.max_retries == 0
        with pytest.raises(Exception):
            await probe_svc.complete(LLMRequest(messages=[{"role": "user", "content": "ping"}]))
        assert calls == 1
    finally:
        await probe_svc.close()


@pytest.mark.asyncio
async def test_h7_p1_22_stream_closed_on_error_and_cancellation() -> None:
    class _TrackedAsyncStream:
        def __init__(self, events: list[Any], *, use_ctx: bool = True) -> None:
            self._events = list(events)
            self._use_ctx = use_ctx
            self.entered = False
            self.closed = False

        async def __aenter__(self) -> _TrackedAsyncStream:
            if not self._use_ctx:
                raise AttributeError("__aenter__")
            self.entered = True
            return self

        async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
            await self.close()

        async def close(self) -> None:
            self.closed = True

        def __aiter__(self) -> _TrackedAsyncStream:
            return self

        async def __anext__(self) -> Any:
            if not self._events:
                raise StopAsyncIteration
            nxt = self._events.pop(0)
            if isinstance(nxt, BaseException):
                raise nxt
            return nxt

    class _CloseOnlyStream:
        def __init__(self, events: list[Any]) -> None:
            self._events = list(events)
            self.closed = False

        async def close(self) -> None:
            self.closed = True

        def __aiter__(self) -> _CloseOnlyStream:
            return self

        async def __anext__(self) -> Any:
            if not self._events:
                raise StopAsyncIteration
            nxt = self._events.pop(0)
            if isinstance(nxt, BaseException):
                raise nxt
            return nxt

    def _delta(text: str) -> Any:
        return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text))])

    # 1) Mid-stream exception closes AsyncStream via __aexit__ -> close()
    err_stream = _TrackedAsyncStream([_delta("a"), RuntimeError("boom")])
    svc = OpenAICompatibleChatLLMService(
        OpenAICompatibleChatConfig(base_url="https://example.invalid/v1", api_key="sk-test", model="m1")
    )
    svc._client = cast(
        Any,
        SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_kw: _ret(err_stream)))),
    )
    with pytest.raises(RuntimeError, match="boom"):
        async for _ in svc.stream(LLMRequest(messages=[{"role": "user", "content": "hi"}])):
            pass
    assert err_stream.entered is True
    assert err_stream.closed is True

    # 2) Mid-stream generator close AND task cancellation both close AsyncStream
    cancel_stream = _TrackedAsyncStream([_delta("a"), _delta("b"), _delta("c")])
    svc._client = cast(
        Any,
        SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_kw: _ret(cancel_stream)))),
    )
    agen = svc.stream(LLMRequest(messages=[{"role": "user", "content": "hi"}]))
    first = await agen.__anext__()
    assert first.delta_text == "a"
    await agen.aclose()
    assert cancel_stream.entered is True
    assert cancel_stream.closed is True

    # 2b) Real asyncio.Task cancellation while blocked inside stream iteration
    gate = asyncio.Event()
    blocked_closed = False

    class _BlockingStream:
        def __init__(self) -> None:
            self.entered = False

        async def __aenter__(self) -> _BlockingStream:
            self.entered = True
            return self

        async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
            await self.close()

        async def close(self) -> None:
            nonlocal blocked_closed
            blocked_closed = True

        def __aiter__(self) -> _BlockingStream:
            return self

        async def __anext__(self) -> Any:
            gate.set()
            await asyncio.sleep(30)
            return _delta("never")

    blocking_stream = _BlockingStream()
    svc._client = cast(
        Any,
        SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_kw: _ret(blocking_stream)))),
    )

    async def _consume() -> None:
        async for _ in svc.stream(LLMRequest(messages=[{"role": "user", "content": "hi"}])):
            pass

    task = asyncio.create_task(_consume())
    await gate.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert blocking_stream.entered is True
    assert blocked_closed is True

    # 3) Stream implementing only close() (no __aenter__) is also closed on cancellation
    close_only = _CloseOnlyStream([_delta("x"), _delta("y")])
    svc._client = cast(
        Any,
        SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_kw: _ret(close_only)))),
    )
    agen2 = svc.stream(LLMRequest(messages=[{"role": "user", "content": "hi"}]))
    assert (await agen2.__anext__()).delta_text == "x"
    await agen2.aclose()
    assert close_only.closed is True


async def _ret(val: Any) -> Any:
    return val


@pytest.mark.asyncio
async def test_h7_p1_22_thousand_plain_text_deltas_skip_redundant_extraction_and_cache_compile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = [
        SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="a"))], usage=None)
        for _ in range(1000)
    ]
    events.append(
        SimpleNamespace(
            choices=[],
            usage={"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
        )
    )

    class _Completions:
        async def create(self, **kwargs: Any) -> Any:
            async def _gen() -> Any:
                for ev in events:
                    yield ev

            return _gen()

    add_calls = 0
    snapshot_calls = 0
    inspect_calls = 0
    orig_add = ResponseMetricAccumulator.add
    orig_snapshot = ResponseMetricAccumulator.snapshot
    orig_inspect = ResponseMetrics.inspect

    def _spy_add(self: ResponseMetricAccumulator, response: Any) -> Any:
        nonlocal add_calls
        add_calls += 1
        return orig_add(self, response)

    def _spy_snapshot(self: ResponseMetricAccumulator) -> Any:
        nonlocal snapshot_calls
        snapshot_calls += 1
        return orig_snapshot(self)

    def _spy_inspect(self: ResponseMetrics, response: Any) -> Any:
        nonlocal inspect_calls
        inspect_calls += 1
        return orig_inspect(self, response)

    monkeypatch.setattr(ResponseMetricAccumulator, "add", _spy_add)
    monkeypatch.setattr(ResponseMetricAccumulator, "snapshot", _spy_snapshot)
    monkeypatch.setattr(ResponseMetrics, "inspect", _spy_inspect)

    svc = OpenAICompatibleChatLLMService(
        OpenAICompatibleChatConfig(base_url="https://example.invalid/v1", api_key="sk-test", model="m1")
    )
    svc._client = cast(Any, SimpleNamespace(chat=SimpleNamespace(completions=_Completions())))
    result = await svc.complete(LLMRequest(messages=[{"role": "user", "content": "hi"}]))

    assert result.raw_text == "a" * 1000
    assert result.token_usage is not None
    assert result.token_usage.total_tokens == 120
    # Across 1000 pure-text deltas + 1 usage event: only the usage event is inspected, snapshot once at end.
    assert add_calls == 1
    assert inspect_calls == 1
    assert snapshot_calls == 1

    # Verify compile_extractor caches compiled functions across repeated calls.
    compile_extractor.cache_clear()
    parse_calls = 0
    orig_parse = ast.parse

    def _spy_parse(*args: Any, **kwargs: Any) -> Any:
        nonlocal parse_calls
        parse_calls += 1
        return orig_parse(*args, **kwargs)

    monkeypatch.setattr(ast, "parse", _spy_parse)
    snippet = "def extract(response):\n    return {'prompt_tokens': response.get('p', 1), 'completion_tokens': 2}\n"
    fn1 = compile_extractor(snippet)
    fn2 = compile_extractor(snippet)
    assert fn1 is fn2
    parser = ResponseMetrics({"usage": {"code": snippet}})
    for idx in range(25):
        usage = parser.extract_usage({"p": idx + 1})
        assert usage is not None and usage.prompt_tokens == idx + 1
    assert parse_calls == 1
    info = compile_extractor.cache_info()
    assert info.misses == 1
    assert info.hits == 26


# ── P1-23: _grep_with_python symlink escape & oversized file caps ───────────


def test_h7_p1_23_grep_with_python_blocks_symlink_escape_and_skips_oversized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws = tmp_path / "workspace"
    ws.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    secret = outside / "secret.txt"
    secret.write_text("CANARY_SECRET_VALUE\n", encoding="utf-8")

    # Symlink inside workspace pointing outside workspace
    escape_link = ws / "escape.txt"
    os.symlink(secret, escape_link)

    # Legitimate in-workspace file and in-workspace symlink
    ok_file = ws / "inside.txt"
    ok_file.write_text("CANARY_SECRET_VALUE\n", encoding="utf-8")

    # Oversized file inside workspace (> TEXT_FILE_MAX_BYTES)
    huge_file = ws / "huge.txt"
    huge_file.write_bytes(b"CANARY_SECRET_VALUE\n" + b"x" * dt.TEXT_FILE_MAX_BYTES)

    def _no_rg(*args: Any, **kwargs: Any) -> Any:
        raise FileNotFoundError("rg unavailable")

    monkeypatch.setattr(dt.subprocess, "run", _no_rg)
    registry = create_default_tool_registry(workspace_dir=ws)
    out = str(registry.invoke("grep", {"path": ".", "pattern": "CANARY_SECRET_VALUE", "max_results": 50}))

    assert "inside.txt:1:CANARY_SECRET_VALUE" in out
    assert "escape.txt" not in out
    assert "secret.txt" not in out
    assert "huge.txt" not in out

    # Direct _grep_with_python call on the escaping symlink must also refuse to read it.
    with runtime_env_context(RuntimeEnv(workspace_dir=ws)):
        direct_out = dt._grep_with_python("CANARY_SECRET_VALUE", escape_link, None, 50, True, True, False)
        assert direct_out == ""


def test_h7_p1_23_edit_apply_patch_and_guard_excerpt_reject_oversized_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws = tmp_path / "workspace"
    ws.mkdir()
    huge = ws / "huge.txt"
    huge.write_bytes(b"anchor_line\n" + b"y" * dt.TEXT_FILE_MAX_BYTES)

    huge_reads = 0
    orig_read_bytes = Path.read_bytes

    def _spy_read_bytes(self: Path) -> bytes:
        nonlocal huge_reads
        if self.resolve() == huge.resolve():
            huge_reads += 1
        return orig_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", _spy_read_bytes)

    with runtime_env_context(RuntimeEnv(workspace_dir=ws)):
        # 1) Unread edit triggers _guard_excerpt — must NOT read huge.txt into memory.
        unread_res = dt.run_edit("huge.txt", "anchor_line", "new_line")
        assert "has not been read" in unread_res
        assert "anchor_line" not in unread_res
        assert huge_reads == 0

        # 2) Even after marking file read, run_edit and run_apply_patch must reject oversized files before reading.
        dt._remember_read(huge)
        edit_res = dt.run_edit("huge.txt", "anchor_line", "new_line")
        assert "too large" in edit_res.lower()
        assert huge_reads == 0

        patch_res = dt.run_apply_patch("huge.txt", "@@ -1,1 +1,1 @@\n-anchor_line\n+new_line\n")
        assert "too large" in patch_res.lower()
        assert huge_reads == 0


# ── Phase 2 version closeout ────────────────────────────────────────────────


def test_h7_phase2_version_is_14_1_0() -> None:
    assert power_loop.__version__ == "14.1.0"
    assert importlib.metadata.version("power-loop") == "14.1.0"
    assert isinstance(LLMTokenUsage(), LLMTokenUsage)
