"""Design/191: real loop/store, deterministic model; no production database."""
import asyncio
import json
from dataclasses import dataclass, field

import pytest

from power_loop import AgentLoopConfig, AgentSpec, LLMResponse, LLMService, SessionStore, StatefulAgentLoop
from power_loop.core.agent_context import reset_session_id, set_session_id
from power_loop.runtime import delegation
from power_loop.runtime.env import RuntimeEnv, runtime_env_context
from power_loop.runtime.recall import recall
from power_loop.runtime.representation import ProjectedRepresentation

pytestmark = pytest.mark.unit


@dataclass
class Model(LLMService):
    answers: list[str] = field(default_factory=lambda: ["first", "second"])
    requests: list = field(default_factory=list)
    hold: asyncio.Event | None = None
    entered: asyncio.Event = field(default_factory=asyncio.Event)

    async def complete(self, request, **kwargs):
        self.requests.append(request)
        self.entered.set()
        if self.hold is not None:
            await self.hold.wait()
        return LLMResponse(raw_text=self.answers.pop(0) if self.answers else "done")

    def stream(self, request):
        async def chunks():
            if False:
                yield None
        return chunks()

    async def close(self):
        pass


@pytest.fixture
async def setup():
    store = await SessionStore.open(":memory:")
    model = Model()
    loop = StatefulAgentLoop(llm=model, store=store, config=AgentLoopConfig(
        max_rounds=4, representation=ProjectedRepresentation()))
    sid = await loop.new_session()
    token = set_session_id(sid)
    yield store, loop, model, sid
    reset_session_id(token)
    tasks = list(delegation._TASKS)
    if tasks:
        await asyncio.gather(*tasks)
    await store.close()


async def drain():
    while delegation._TASKS:
        await asyncio.gather(*list(delegation._TASKS))


async def test_create_then_continue_have_disjoint_ranges_and_history(setup):
    store, loop, model, parent = setup
    results = []

    async def complete(sid, result):
        results.append(result)

    first = await delegation.start(loop, AgentSpec(name="review", system_prompt="Review", lifecycle="linked"),
                                   "review A", on_complete=complete)
    await drain()
    second = await delegation.nudge(loop, first["session_id"], "review repaired A", on_complete=complete)
    await drain()
    assert first["session_id"] == second["session_id"]
    assert first["run_id"] != second["run_id"]
    assert len(results) == 2
    assert all(r["projection_status"] == "ready" for r in results)
    assert results[0]["messages"][1] < results[1]["messages"][0]
    assert results[0]["projection"][1] < results[1]["projection"][0]
    text = str(model.requests[1].messages)
    assert "first" in text and "review repaired A" in text
    record = await store.get_runtime_state(parent, "subagent:run:" + first["run_id"])
    assert record["result"]["final_text"] == "first"


async def test_async_receipt_and_running_steering(setup):
    store, loop, model, parent = setup
    model.hold = asyncio.Event()
    receipt = await asyncio.wait_for(delegation.start(
        loop, AgentSpec(name="review", system_prompt="Review", lifecycle="linked"), "review"), 1)
    await model.entered.wait()
    follow = await delegation.nudge(loop, receipt["session_id"], "Also check B")
    assert follow["mode"] == "steer" and follow["run_id"] == receipt["run_id"]
    model.hold.set()
    await drain()
    result = await store.get_runtime_state(parent, "subagent:run:" + receipt["run_id"])
    assert result["phase"] == "completed"
    messages = await store.load_all_messages(receipt["session_id"])
    assert any("Also check B" in str(r.content) for r in messages)


async def test_ephemeral_tombstone_does_not_grant_other_parent_access(setup):
    store, loop, _, parent = setup
    receipt = await delegation.start(loop, AgentSpec(name="x", system_prompt="x"), "task")
    await drain()
    sid = receipt["session_id"]
    assert await store.get_session(sid) is None
    out = json.loads(await recall(store, parent, sub_session=sid, seq=1))
    assert out["reason"] == "session_deleted"
    stranger = await loop.new_session()
    denied = json.loads(await recall(store, stranger, sub_session=sid, seq=1))
    assert denied["status"] == "error" and "access_denied" in denied["error"]
    nudged = await delegation.nudge(loop, sid, "do more")
    assert nudged["reason"] == "session_deleted"


async def test_projection_id_stable_on_upsert_and_never_reused(setup):
    store, loop, _, parent = setup
    await store.upsert_project_message(parent, send_index=1, kind="project", content={"a": 1})
    one = (await store.load_project_messages(parent))[0]
    await store.upsert_project_message(parent, send_index=1, kind="project", content={"a": 2})
    two = (await store.load_project_messages(parent))[0]
    assert one.id == two.id and two.content == {"a": 2}
    other = await loop.new_session()
    await store.upsert_project_message(other, send_index=1, kind="project", content={})
    assert (await store.load_project_messages(other))[0].id > one.id


async def test_recall_pages_long_record_without_send_and_exports_exclusively(setup, tmp_path):
    store, loop, _, parent = setup
    child = await store.create_session(parent_session_id=parent)
    for i in range(5):
        await store.append_message(child, role="tool", content="x" * 9000 + str(i),
                                   tool_calls=None, send_index=1)
    first = json.loads(await recall(store, parent, sub_session=child, ids=[1, 5], limit=2))
    assert len(first["records"]) == 2 and first["next_cursor"]
    page = json.loads(await recall(store, parent, sub_session=child, ids=[1, 5], limit=2,
                                   cursor=first["next_cursor"]))
    assert [r["seq"] for r in page["records"]] == [3, 4]
    bad = json.loads(await recall(store, parent, ids=[1, 5], cursor=first["next_cursor"]))
    assert bad["status"] == "error"
    single = json.loads(await recall(store, parent, sub_session=child, seq=1))
    assert len(json.dumps(single, ensure_ascii=False, separators=(",", ":"))) <= 8000
    assert single["next_char_offset"] == len(single["text"]) < 8000
    rest = json.loads(await recall(store, parent, sub_session=child, seq=1, char_offset=single["next_char_offset"]))
    assert json.loads(single["text"] + rest["text"])["content"].endswith("0")
    with runtime_env_context(RuntimeEnv(workspace_dir=tmp_path)):
        saved = json.loads(await recall(store, parent, sub_session=child, ids=[1, 5],
                                        output_file="records/data.jsonl"))
        assert saved["records"] == 5
        path = tmp_path / "records/data.jsonl"
        original = path.read_bytes()
        assert all(len(json.loads(line)["content"]) == 9001 for line in path.read_text().splitlines())
        refused = json.loads(await recall(store, parent, sub_session=child, ids=[1, 5],
                                          output_file="records/data.jsonl"))
        assert refused["status"] == "error" and path.read_bytes() == original
        conflict = json.loads(await recall(store, parent, sub_session=child, seq=1,
                                           output_file="other.jsonl", char_limit=8000))
        assert conflict["status"] == "error" and not (tmp_path / "other.jsonl").exists()


async def test_wake_failure_keeps_result_in_durable_parent_inbox(setup):
    store, loop, _, parent = setup
    async def fail(*args):
        raise RuntimeError("host unavailable")
    receipt = await delegation.start(loop, AgentSpec(name="x", system_prompt="x", lifecycle="linked"),
                                      "task", on_complete=fail)
    await drain()
    assert (await store.inbox_counts(parent))["pending"] == 1
    record = await store.get_runtime_state(parent, "subagent:run:" + receipt["run_id"])
    assert record["phase"] == "completed" and not record["notified"]


async def test_queued_child_accepts_nudge_before_model_admission(setup):
    store, loop, _, _ = setup
    semaphore = asyncio.Semaphore(0)
    receipt = await asyncio.wait_for(delegation.start(
        loop, AgentSpec(name="queued", system_prompt="queued", lifecycle="linked"),
        "original", admission=semaphore), 1)
    steering = await delegation.nudge(loop, receipt["session_id"], "additional evidence")
    assert steering["mode"] == "steer"
    semaphore.release()
    await drain()
    rows = await store.load_all_messages(receipt["session_id"])
    assert any("additional evidence" in str(r.content) for r in rows)


async def test_settling_nudge_is_consumed_without_concurrent_send(setup):
    store, loop, _, parent = setup
    entered, release = asyncio.Event(), asyncio.Event()
    once = False
    async def finalize(sid, result):
        nonlocal once
        if not once:
            once = True
            entered.set()
            await release.wait()
        return result
    receipt = await delegation.start(loop, AgentSpec(name="x", system_prompt="x", lifecycle="linked"),
                                      "task", finalize=finalize)
    await entered.wait()
    steering = await delegation.nudge(loop, receipt["session_id"], "arrived during settling")
    assert steering["run_id"] == receipt["run_id"]
    release.set()
    await drain()
    result = await store.get_runtime_state(parent, "subagent:run:" + receipt["run_id"])
    assert result["result"]["rounds"] == 2
    assert (await store.inbox_counts(receipt["session_id"]))["pending"] == 0


async def test_foreign_projection_ids_do_not_escape_session_filter(setup):
    store, loop, _, parent = setup
    child = await store.create_session(parent_session_id=parent)
    other = await loop.new_session()
    for sid in [child, other, child]:
        count = len(await store.load_project_messages(sid)) + 1
        await store.upsert_project_message(sid, send_index=count, kind="project", content={"sid": sid},
                                           source_seq_lo=count, source_seq_hi=count)
    bounds = await store.history_bounds(child, source="projection")
    data = json.loads(await recall(store, parent, sub_session=child, source="projection", ids=bounds))
    assert len(data["records"]) == 2 and other not in json.dumps(data)


async def test_missing_projection_ids_in_legacy_schema_is_explicit(setup):
    store, _, _, parent = setup
    await store.upsert_project_message(parent, send_index=1, kind="project", content={})
    await store._db.execute(f"ALTER TABLE {store.t.project_messages} RENAME TO old_projection")
    await store._db.execute(f"CREATE TABLE {store.t.project_messages} AS SELECT "
                            "session_id,send_index,kind,content_json,rendered_text,source_seq_lo,source_seq_hi,"
                            "compact_from_send,compact_to_send,projector_version,token_estimate,created_at "
                            "FROM old_projection")
    assert (await store.load_project_messages(parent))[0].id is None
    result = json.loads(await recall(store, parent, source="projection", id=1))
    assert result["status"] == "error" and "legacy schema" in result["error"]


async def test_nested_async_child_is_collected_before_outer_completion(setup):
    store, loop, _, parent = setup
    from power_loop.tools.registry import ToolRegistry
    from power_loop.tools.spawn_agent import register_spawn_agent
    class Nested(Model):
        outer_calls = 0
        async def complete(self, request, **kwargs):
            if "OUTER-ROLE" not in (request.system_prompt or ""):
                return LLMResponse(raw_text="INNER-PROOF")
            self.outer_calls += 1
            if self.outer_calls == 1:
                return LLMResponse(raw_text="", tool_calls=[{
                    "id": "nested", "type": "function", "function": {
                        "name": "spawn_agent", "arguments": json.dumps({"task": "inner task", "tools": []})}}])
            if "INNER-PROOF" in str(request.messages):
                return LLMResponse(raw_text="OUTER-COLLECTED-INNER-PROOF")
            return LLMResponse(raw_text="Waiting for nested result")
    registry = ToolRegistry()
    register_spawn_agent(registry)
    loop.tool_registry = registry
    loop.llm = Nested()
    receipt = await delegation.start(loop, AgentSpec(name="outer", system_prompt="OUTER-ROLE",
                                                     lifecycle="linked"), "outer task")
    await asyncio.wait_for(drain(), 5)
    record = await store.get_runtime_state(parent, "subagent:run:" + receipt["run_id"])
    assert record["result"]["final_text"] == "OUTER-COLLECTED-INNER-PROOF"


async def test_projection_failure_is_not_business_failure(setup, monkeypatch):
    store, loop, _, parent = setup
    async def fail(*args, **kwargs):
        raise RuntimeError("projection storage failed")
    monkeypatch.setattr(StatefulAgentLoop, "_write_send_projection", fail)
    receipt = await delegation.start(loop, AgentSpec(name="x", system_prompt="x", lifecycle="linked"), "task")
    await drain()
    record = await store.get_runtime_state(parent, "subagent:run:" + receipt["run_id"])
    result = record["result"]
    assert result["status"] == "completed" and result["messages"]
    assert result["projection"] is None and result["projection_status"] == "failed"


async def test_stop_during_finalization_keeps_owner_result_without_external_wake(setup):
    store, loop, _, parent = setup
    entered = asyncio.Event()
    wakes = []

    async def notified(*args):
        wakes.append(args)

    async def blocked(sid, result):
        entered.set()
        await asyncio.Event().wait()
    receipt = await delegation.start(loop, AgentSpec(name="x", system_prompt="x", lifecycle="linked"),
                                      "task", finalize=blocked, on_complete=notified)
    await entered.wait()
    assert any(j["id"] == "sub:" + receipt["run_id"] for j in await loop.list_jobs(parent))
    status = await delegation.stop(store, parent, receipt["run_id"], 0.01)
    assert status == "forced"
    await drain()
    assert not delegation.running_jobs(store, parent)
    record = await store.get_runtime_state(parent, "subagent:run:" + receipt["run_id"])
    assert record["result"]["status"] == "cancelled"
    assert (await store.inbox_counts(parent))["pending"] == 1
    [item] = await store.inbox_items(parent)
    payload = json.loads(item["content"].removeprefix(delegation.RESULT_PREFIX))
    assert payload["status"] == "cancelled" and payload["run_id"] == receipt["run_id"]
    assert not wakes
