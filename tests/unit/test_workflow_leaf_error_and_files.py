"""叶子级错误语义（retry / fallback / continue_on_error）与文件产出端口。

design/64 §1-§2。这三件都发生在 `_exec_agent` 的派发路径上，用一个可编程的假 executor 驱动，
不碰真 LLM。
"""

from __future__ import annotations

import tempfile
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

import pytest

from power_loop import AgentLoopConfig, StatefulAgentLoop
from power_loop._vendor.llm_client.interface import LLMRequest, LLMResponse, LLMService, LLMStreamChunk
from power_loop.workflow import WorkflowSpec
from power_loop.workflow.engine import WorkflowEngine

pytestmark = pytest.mark.unit


@dataclass
class _FakeLLM(LLMService):
    async def complete(self, request: LLMRequest, *, on_chunk_delta_text=None,
                       on_chunk_think=None, on_stream_end=None) -> LLMResponse:
        return LLMResponse(raw_text="ok", content_text="ok")

    def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamChunk]:
        async def _e() -> AsyncIterator[LLMStreamChunk]:
            if False:
                yield LLMStreamChunk()
        return _e()

    async def close(self) -> None:
        return None


def _loop() -> StatefulAgentLoop:
    return StatefulAgentLoop(
        llm=_FakeLLM(), db_path=tempfile.mktemp(suffix=".db"),
        config=AgentLoopConfig(system_prompt="o", max_rounds=3, compactor=None),
    )


@dataclass
class _ScriptedExecutor:
    """按 node_id 给一串预设结果；每次调用取下一个（用尽后重复最后一个）。"""

    script: dict = field(default_factory=dict)
    calls: list = field(default_factory=list)

    async def run_agent(self, spec, user_input, *, parent_loop, driver_sid, stop_event=None):
        nid = spec.metadata.get("idempotency_key", "").split(":")[-1] or spec.name
        self.calls.append({
            "node": nid,
            "attempt": spec.metadata.get("attempt"),
            "idem": spec.metadata.get("idempotency_key"),
            "input": user_input,
            "output_file": spec.metadata.get("output_file"),
        })
        seq = self.script.get(nid) or [{"status": "completed", "final_text": f"{nid} ok"}]
        idx = min(sum(1 for c in self.calls if c["node"] == nid) - 1, len(seq) - 1)
        return {"session_id": f"s-{nid}", "usage": {}, **seq[idx]}


def _leaf(node_id: str, **extra) -> dict:
    return {"type": "agent", "id": node_id,
            "spec": {"name": node_id, "system_prompt": "p"}, **extra}


async def _run(root: dict, executor, *, file_io=None, run_id="r1"):
    loop = _loop()
    engine = WorkflowEngine(loop, executor=executor, run_id=run_id, file_io=file_io)
    return await engine.run(WorkflowSpec.from_json({"name": "wf", "input": "GOAL", "root": root}))


# ── retry ────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_retry_reruns_until_it_completes():
    ex = _ScriptedExecutor(script={"a": [
        {"status": "failed", "error": "boom"},
        {"status": "failed", "error": "boom"},
        {"status": "completed", "final_text": "third time"},
    ]})
    res = await _run(_leaf("a", retry={"max_attempts": 3}), ex)
    assert res.status == "completed"
    assert res.results["a"].text == "third time"
    assert [c["attempt"] for c in ex.calls] == [1, 2, 3]


@pytest.mark.asyncio
async def test_idempotency_key_is_stable_across_attempts_but_attempt_number_is_not():
    """🔴 key 变了工具就没法去重;attempt 号是给工具区分「第几次」用的。"""
    ex = _ScriptedExecutor(script={"a": [
        {"status": "failed", "error": "boom"},
        {"status": "completed", "final_text": "ok"},
    ]})
    await _run(_leaf("a", retry={"max_attempts": 2}), ex)
    assert {c["idem"] for c in ex.calls} == {"r1:a"}          # 恒定
    assert [c["attempt"] for c in ex.calls] == [1, 2]          # 递增


@pytest.mark.asyncio
async def test_empty_trigger_retries_a_completed_but_silent_leaf():
    ex = _ScriptedExecutor(script={"a": [
        {"status": "completed", "final_text": "   "},
        {"status": "completed", "final_text": "real answer"},
    ]})
    res = await _run(_leaf("a", retry={"max_attempts": 2, "on": ["empty"]}), ex)
    assert res.results["a"].text == "real answer"
    assert len(ex.calls) == 2


@pytest.mark.asyncio
async def test_failed_trigger_not_selected_means_no_retry():
    ex = _ScriptedExecutor(script={"a": [{"status": "failed", "error": "boom"}]})
    res = await _run(_leaf("a", retry={"max_attempts": 3, "on": ["empty"]}), ex)
    assert len(ex.calls) == 1 and res.status == "failed"


@pytest.mark.asyncio
async def test_cancelled_is_never_retried():
    ex = _ScriptedExecutor(script={"a": [{"status": "cancelled"}, {"status": "completed"}]})
    await _run(_leaf("a", retry={"max_attempts": 3}), ex)
    assert len(ex.calls) == 1          # 取消不是「没跑好」


# ── continue_on_error / run 终态 ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_failed_leaf_now_fails_the_run():
    """以前只要没抛异常就报 completed——哪怕某个节点整个失败了。"""
    ex = _ScriptedExecutor(script={"a": [{"status": "failed", "error": "boom"}]})
    res = await _run(_leaf("a"), ex)
    assert res.status == "failed"
    assert any("failed node(s): a" in e for e in res.errors)


@pytest.mark.asyncio
async def test_continue_on_error_keeps_the_run_completed():
    ex = _ScriptedExecutor(script={"a": [{"status": "failed", "error": "boom"}]})
    res = await _run(_leaf("a", continue_on_error=True), ex)
    assert res.status == "completed"
    assert res.results["a"].status == "failed"        # 节点自己仍如实记为失败


@pytest.mark.asyncio
async def test_downstream_gets_an_explicit_failure_note_not_an_empty_string():
    """🔴 空字符串会让下游模型把「没说话」读成「没意见」。"""
    ex = _ScriptedExecutor(script={"up": [{"status": "failed", "error": "boom"}]})
    root = {"type": "sequence", "steps": [
        _leaf("up", continue_on_error=True),
        _leaf("down", inputs_from=["up"]),
    ]}
    await _run(root, ex)
    down_input = next(c["input"] for c in ex.calls if c["node"] == "down")
    assert "上游节点 'up' 未产出结果" in down_input and "boom" in down_input
    assert "不要把「没有发现」当成「没有问题」" in down_input


# ── fallback ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_fallback_substitutes_the_primary_result_for_downstream():
    ex = _ScriptedExecutor(script={
        "a": [{"status": "failed", "error": "boom"}],
        "a_fb": [{"status": "completed", "final_text": "fallback answer"}],
    })
    root = {"type": "sequence", "steps": [
        _leaf("a", fallback=_leaf("a_fb")),
        _leaf("down", inputs_from=["a"]),
    ]}
    res = await _run(root, ex)
    assert res.status == "completed"
    # 下游引用的是主节点 id —— 结果要顶替上去
    assert res.results["a"].text == "fallback answer"
    assert "recovered by fallback 'a_fb'" in (res.results["a"].error or "")
    assert "fallback answer" in next(c["input"] for c in ex.calls if c["node"] == "down")
    assert res.results["a_fb"].status == "completed"       # 兜底自己的记录也留着


@pytest.mark.asyncio
async def test_fallback_runs_only_after_retries_are_exhausted():
    ex = _ScriptedExecutor(script={
        "a": [{"status": "failed"}, {"status": "completed", "final_text": "recovered"}],
        "a_fb": [{"status": "completed", "final_text": "should not run"}],
    })
    res = await _run(_leaf("a", retry={"max_attempts": 2}, fallback=_leaf("a_fb")), ex)
    assert res.results["a"].text == "recovered"
    assert not any(c["node"] == "a_fb" for c in ex.calls)


@pytest.mark.asyncio
async def test_failing_fallback_leaves_the_run_failed():
    ex = _ScriptedExecutor(script={
        "a": [{"status": "failed", "error": "boom"}],
        "a_fb": [{"status": "failed", "error": "also boom"}],
    })
    res = await _run(_leaf("a", fallback=_leaf("a_fb")), ex)
    assert res.status == "failed"


# ── 文件产出端口 ─────────────────────────────────────────────────────────────


@dataclass
class _FakeFileIO:
    files: dict = field(default_factory=dict)
    archived: list = field(default_factory=list)

    def output_path(self, node_id: str, *, iteration=None) -> str:
        return f"outputs/{node_id}.md" if iteration is None else f"outputs/{node_id}.{iteration}.md"

    def render_ref(self, path: str, slice_expr: str) -> str:
        body = self.files.get(path)
        if body is None:
            return f"（上游产物 {path} 还不存在）"
        return f"（{path}{slice_expr} 共{len(body)}行）" + "|".join(body)

    def before_attempt(self, node_id: str, path: str, attempt: int) -> None:
        self.archived.append((node_id, path, attempt))


@pytest.mark.asyncio
async def test_output_file_is_threaded_into_leaf_metadata():
    ex, fio = _ScriptedExecutor(), _FakeFileIO()
    await _run(_leaf("a"), ex, file_io=fio)
    assert ex.calls[0]["output_file"] == "outputs/a.md"


@pytest.mark.asyncio
async def test_explicit_output_file_wins_over_the_default_naming():
    ex, fio = _ScriptedExecutor(), _FakeFileIO()
    await _run(_leaf("a", output_file="reports/custom.md"), ex, file_io=fio)
    assert ex.calls[0]["output_file"] == "reports/custom.md"


@pytest.mark.asyncio
async def test_filerefs_in_input_are_rendered_by_the_host_port():
    ex = _ScriptedExecutor()
    fio = _FakeFileIO(files={"outputs/up.md": ["l1", "l2", "l3"]})
    root = _leaf("a", input="看这个：@@FILEREF:outputs/up.md[-2:]@@ 完")
    await _run(root, ex, file_io=fio)
    got = ex.calls[0]["input"]
    assert "outputs/up.md[-2:] 共3行" in got and "l1|l2|l3" in got
    assert "@@FILEREF" not in got          # 占位符必须被吃掉


@pytest.mark.asyncio
async def test_missing_fileio_leaves_input_untouched_and_metadata_clean():
    ex = _ScriptedExecutor()
    await _run(_leaf("a", input="ref @@FILEREF:x.md@@"), ex)   # file_io=None → 老行为
    assert ex.calls[0]["output_file"] is None
    assert "@@FILEREF:x.md@@" in ex.calls[0]["input"]


@pytest.mark.asyncio
async def test_retry_archives_the_previous_output_before_rerunning():
    """重跑会把产出追加到旧证据后面——host 要有机会先归档。"""
    ex = _ScriptedExecutor(script={"a": [{"status": "failed"}, {"status": "completed"}]})
    fio = _FakeFileIO()
    await _run(_leaf("a", retry={"max_attempts": 2}), ex, file_io=fio)
    assert fio.archived == [("a", "outputs/a.md", 2)]      # 第 1 次之前不归档


@pytest.mark.asyncio
async def test_a_broken_fileio_never_takes_down_the_leaf():
    class _Broken:
        def output_path(self, node_id, *, iteration=None):
            raise RuntimeError("disk gone")

        def render_ref(self, path, slice_expr):
            raise RuntimeError("disk gone")

        def before_attempt(self, node_id, path, attempt):
            raise RuntimeError("disk gone")

    ex = _ScriptedExecutor()
    res = await _run(_leaf("a", input="ref @@FILEREF:x.md@@"), ex, file_io=_Broken())
    assert res.status == "completed"
    assert "无法读取上游产物 x.md" in ex.calls[0]["input"]


# ── 指数退避 + foreach 迭代序号（5.2.1） ──────────────────────────────────────


@pytest.mark.asyncio
async def test_backoff_is_exponential_when_a_factor_is_given(monkeypatch):
    slept: list[float] = []

    async def _fake_sleep(d):
        slept.append(d)

    monkeypatch.setattr("power_loop.workflow.engine.asyncio.sleep", _fake_sleep)
    ex = _ScriptedExecutor(script={"a": [{"status": "failed"}] * 3 + [{"status": "completed"}]})
    await _run(_leaf("a", retry={"max_attempts": 4, "backoff_s": 2, "backoff_factor": 2}), ex)
    assert slept == [2.0, 4.0, 8.0]          # 第 N 次重试等 backoff_s * factor**(N-1)


@pytest.mark.asyncio
async def test_default_backoff_factor_is_a_fixed_delay(monkeypatch):
    slept: list[float] = []
    monkeypatch.setattr(
        "power_loop.workflow.engine.asyncio.sleep",
        lambda d: slept.append(d) or _noop(),
    )

    async def _noop():
        return None

    ex = _ScriptedExecutor(script={"a": [{"status": "failed"}] * 2 + [{"status": "completed"}]})
    await _run(_leaf("a", retry={"max_attempts": 3, "backoff_s": 1.5}), ex)
    assert slept == [1.5, 1.5]


@pytest.mark.asyncio
async def test_a_single_backoff_wait_is_capped():
    from power_loop.workflow.spec import MAX_BACKOFF_S

    assert MAX_BACKOFF_S == 60.0     # 等太久等于把整个 run 挂在那儿


@pytest.mark.asyncio
async def test_foreach_iterations_get_distinct_output_files():
    """body 的所有迭代共享一个 node_id——不区分的话 N 个并发迭代会追加进同一个文件。"""
    ex, fio = _ScriptedExecutor(), _FakeFileIO()
    root = {
        "type": "foreach", "id": "fan", "as": "item", "items": ["x", "y", "z"],
        "body": _leaf("worker", input="做 {{item}}"),
    }
    await _run(root, ex, file_io=fio)
    files = sorted(c["output_file"] for c in ex.calls)
    assert files == ["outputs/worker.0.md", "outputs/worker.1.md", "outputs/worker.2.md"]
    assert sorted(c["input"].split("做 ")[1].split("\n")[0] for c in ex.calls) == ["x", "y", "z"]


# ── continuation（6.7.0 耗尽续跑）────────────────────────────────────────────

@dataclass
class _ContinuingExecutor(_ScriptedExecutor):
    """run_agent 走脚本；continue_agent 也走脚本（key = f"{sid}#cont"）。"""

    cont_script: dict = field(default_factory=dict)
    cont_calls: list = field(default_factory=list)

    async def continue_agent(self, session_id, user_input, *, parent_loop,
                             extra_rounds, stop_event=None):
        self.cont_calls.append({"sid": session_id, "input": user_input,
                                "extra_rounds": extra_rounds})
        seq = self.cont_script.get(session_id) or [{"status": "completed", "final_text": "done"}]
        idx = min(sum(1 for c in self.cont_calls if c["sid"] == session_id) - 1, len(seq) - 1)
        return {"session_id": session_id, "rounds": 2, "usage": {"total_tokens": 7}, **seq[idx]}


@pytest.mark.asyncio
async def test_continuation_resumes_same_session_on_hit_round_limit():
    ex = _ContinuingExecutor(
        script={"a": [{"status": "hit_round_limit",
                       "final_text": "[hit_round_limit]\n做了 3 屏，剩 2 屏", "rounds": 25}]},
        cont_script={"s-a": [{"status": "completed", "final_text": "全部完成"}]},
    )
    res = await _run(_leaf("a", continuation={"gate": "always"}), ex)
    assert res.status == "completed"
    assert res.results["a"].text == "全部完成"
    assert len(ex.cont_calls) == 1 and ex.cont_calls[0]["sid"] == "s-a"
    assert "接着做剩余的部分" in ex.cont_calls[0]["input"]
    assert ex.cont_calls[0]["extra_rounds"] == 8
    # run_agent 只跑了一次——续跑不是新会话重跑
    assert len(ex.calls) == 1


@pytest.mark.asyncio
async def test_continuation_todo_gate_blocks_without_a_todo_ledger():
    """默认门 todo_remaining：叶子会话没有（读不到）未完成 todo → 不续，落地为耗尽。"""
    ex = _ContinuingExecutor(
        script={"a": [{"status": "hit_round_limit", "final_text": "[hit_round_limit]\nx"}]},
    )
    res = await _run(_leaf("a", continuation={}), ex)
    assert ex.cont_calls == []
    assert res.results["a"].status == "hit_round_limit"
    assert res.status == "failed"          # 非 completed 叶子照旧拖垮 run（语义不变）


@pytest.mark.asyncio
async def test_continuation_stops_at_max_and_result_stays_hit_round_limit():
    ex = _ContinuingExecutor(
        script={"a": [{"status": "hit_round_limit", "final_text": "[hit_round_limit]\n1"}]},
        cont_script={"s-a": [
            {"status": "hit_round_limit", "final_text": "[hit_round_limit]\n2"},
            {"status": "hit_round_limit", "final_text": "[hit_round_limit]\n3"},
        ]},
    )
    res = await _run(_leaf("a", continuation={"gate": "always", "max_continuations": 2}), ex)
    assert len(ex.cont_calls) == 2
    assert res.results["a"].status == "hit_round_limit"


@pytest.mark.asyncio
async def test_retry_failed_trigger_no_longer_catches_round_exhaustion():
    """语义拆分：failed ≠ hit_round_limit。耗尽不再被 on:[failed] 从头重跑。"""
    ex = _ScriptedExecutor(script={"a": [
        {"status": "hit_round_limit", "final_text": "[hit_round_limit]\nx"},
        {"status": "completed", "final_text": "would be retry"},
    ]})
    res = await _run(_leaf("a", retry={"max_attempts": 2, "on": ["failed"]}), ex)
    assert len(ex.calls) == 1, "耗尽被 failed 触发器重跑了——语义拆分失效"
    assert res.results["a"].status == "hit_round_limit"


@pytest.mark.asyncio
async def test_retry_hit_round_limit_trigger_restores_old_behaviour_explicitly():
    ex = _ScriptedExecutor(script={"a": [
        {"status": "hit_round_limit", "final_text": "[hit_round_limit]\nx"},
        {"status": "completed", "final_text": "fresh rerun"},
    ]})
    res = await _run(_leaf("a", retry={"max_attempts": 2, "on": ["hit_round_limit"]}), ex)
    assert len(ex.calls) == 2
    assert res.results["a"].text == "fresh rerun"


@pytest.mark.asyncio
async def test_continuation_usage_and_rounds_fold_into_the_node_result():
    ex = _ContinuingExecutor(
        script={"a": [{"status": "hit_round_limit", "final_text": "[hit_round_limit]\nx",
                       "rounds": 25, "usage": {"total_tokens": 100}}]},
        cont_script={"s-a": [{"status": "completed", "final_text": "ok"}]},
    )
    res = await _run(_leaf("a", continuation={"gate": "always"}), ex)
    assert res.results["a"].usage.get("total_tokens") == 107
