"""Damaged history must still produce valid requests, without rewriting stored rows."""
import json
from copy import deepcopy

import pytest

from power_loop import AgentLoopConfig, JobRef, LLMResponse, LLMService
from power_loop.agent.runtime_store import now_ms
from power_loop.runtime.history_sanitize import sanitize_request_messages
from tests.unit.test_stateful_loop import _echo_registry


def call(cid="c", arguments="{}"):
    return {"id": cid, "type": "function", "function": {"name": "echo", "arguments": arguments}}


def assistant(*calls, content=None):
    return {"role": "assistant", "content": content, "tool_calls": list(calls)}


def assert_valid_request(messages):
    """Independent provider-style checks; parallel tool results are separate rows."""
    pending = set()
    previous = None
    for index, msg in enumerate(messages):
        role = msg["role"]
        assert role != "system" or index == 0
        assert role != previous or role == "tool"
        if role == "tool":
            assert msg["tool_call_id"] in pending
            pending.remove(msg["tool_call_id"])
        else:
            assert not pending
        if role == "assistant":
            calls = msg.get("tool_calls", [])
            assert msg.get("content") or calls
            for tc in calls:
                assert isinstance(tc["id"], str) and tc["id"]
                assert tc["id"] not in pending
                assert tc["type"] == "function" and tc["function"]["name"]
                assert isinstance(tc["function"]["arguments"], str)
                json.loads(tc["function"]["arguments"])
                pending.add(tc["id"])
        previous = role
    assert not pending


CASES = [
    pytest.param([assistant(call())], id="trailing-call-without-pending"),
    pytest.param([assistant(call(None))], id="missing-id"),
    pytest.param([assistant(call(arguments="{broken"))], id="invalid-json"),
    pytest.param([assistant(call(arguments={"text": "kept"}))], id="non-string-arguments"),
    pytest.param([assistant(call(), call())], id="duplicate-id"),
    pytest.param([assistant(content=" "), assistant(content="kept")], id="empty-assistant"),
    pytest.param([{"role": "user", "content": "one"}, {"role": "user", "content": "two"},
                  assistant(content="three"), assistant(content="four")], id="consecutive-roles"),
    pytest.param([{"role": "user", "content": "hi"}, {"role": "system", "content": "old note"}],
                 id="middle-system"),
    pytest.param([{"role": "system", "content": "one"}, {"role": "system", "content": "two"}],
                 id="leading-systems"),
    pytest.param([assistant(call("a"), call("b")),
                  {"role": "tool", "tool_call_id": "a", "content": "known"},
                  {"role": "tool", "tool_call_id": "a", "content": "duplicate"},
                  {"role": "tool", "tool_call_id": "ghost", "content": "orphan"}],
                 id="partial-and-orphan-results"),
]


@pytest.mark.parametrize("messages", CASES)
def test_request_repairs_are_pure_and_idempotent(messages):
    before = deepcopy(messages)
    repaired = sanitize_request_messages(messages)
    assert_valid_request(repaired)
    assert messages == before
    assert sanitize_request_messages(repaired) == repaired
    if messages == [assistant(call())]:
        assert "结果未知" in repaired[-1]["content"]


class StrictModel(LLMService):
    def __init__(self, responses=()):
        self.responses = list(responses)
        self.requests = []

    async def complete(self, request, **kwargs):
        assert_valid_request(request.to_messages())
        self.requests.append(deepcopy(request.messages))
        return self.responses.pop(0) if self.responses else LLMResponse(raw_text="done")

    def stream(self, request):
        raise AssertionError("complete is the tested boundary")

    async def close(self):
        return None


@pytest.mark.asyncio
@pytest.mark.parametrize("messages", CASES)
@pytest.mark.parametrize("action", ["send", "child_resume"])
async def test_bad_stored_history_can_continue(tmp_path, messages, action, runtime_agent):
    registry = _echo_registry()
    if action == 'child_resume':
        from power_loop.tools.spawn_agent import register_spawn_agent
        register_spawn_agent(registry)
        class Model(StrictModel):
            spawned = False
            async def complete(self, request, **kwargs):
                if request.system_prompt.startswith('CHILD'):
                    return await super().complete(request, **kwargs)
                if not self.spawned:
                    self.spawned = True
                    return LLMResponse(raw_text='', tool_calls=[{'id':'spawn','type':'function','function':{
                        'name':'spawn_agent','arguments':'{"task":"initial","system_prompt":"CHILD","tools":["echo"]}'}}])
                return LLMResponse(raw_text='parent done')
        model = Model([LLMResponse(raw_text='initial child result')])
    else:
        model = StrictModel()
    loop = await runtime_agent(model=model, registry=registry,
        config=AgentLoopConfig(system_prompt='system', compactor=None))
    store = loop.runtime.store
    if action == 'child_resume':
        await loop.send('delegate')
        rows = await store.load_all_messages(loop.target.session_id)
        created = json.loads(next(r.content for r in rows if r.tool_call_id == 'spawn'))
        ref = JobRef(**created['job_ref'])
        original = await loop.runtime.client.wait(ref, actor=loop.actor, deadline_ms=now_ms()+10000)
        assert original.status == 'completed' and original.final_text == 'initial child result'
        ref = original.job_ref
        await loop.drain()
        sid = created['session_id']
    else:
        sid = loop.target.session_id
    model.requests.clear()
    model.responses[:] = [LLMResponse(raw_text='', tool_calls=[call('new', '{"text":"ok"}')])]
    for message in messages:
        await store.append_message(sid, **message)
    before = await store.load_active_messages(sid)
    if action == 'send':
        result = await loop.send('continue')
    else:
        receipt = await loop.runtime.client.resume(ref, message_id='resume', actor=loop.actor, body='continue')
        result = await loop.runtime.client.wait(receipt.input_ref, actor=loop.actor, deadline_ms=now_ms()+10000)
        await loop.drain()
    assert result.status == 'completed' and len(model.requests) == 2
    after = await store.load_active_messages(sid)
    assert after[:len(before)] == before


def test_merging_preserves_multimodal_content_and_known_results():
    image = {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}}
    messages = [{"role": "user", "content": "look"}, {"role": "user", "content": [image]},
                assistant(call("a"), call("b")),
                {"role": "tool", "tool_call_id": "a", "content": "known"}]
    out = sanitize_request_messages(messages)
    assert_valid_request(out)
    assert out[0]["content"] == [{"type": "text", "text": "look"}, image]
    assert out[-2]["content"] == "known"
    assert "结果未知" in out[-1]["content"]
