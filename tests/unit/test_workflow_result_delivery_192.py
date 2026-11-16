"""Frozen workflow admission and ordinary session result identity."""
import json
from dataclasses import replace

import pytest

from power_loop import AgentLoopConfig, RequestOptions
from power_loop.agent.runtime_store import now_ms
from power_loop.agent.runtime_types import RuntimeRequestError, plain
from power_loop.workflow import journal
from tests.runtime_fixture import RunningAgent

from ._workflow_fixture_203 import Model, leaf

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def test_admission_retry_keeps_one_driver_and_original_result(registry_store):
    store, _ = registry_store
    runner = await RunningAgent(store, Model(), config=AgentLoopConfig(context_budget_tokens=10**9)).start()
    spec = {"name": "once", "root": leaf("A")}
    try:
        first = await runner.runtime.client.workflow(runner.target, spec, message_id="run", actor=runner.actor)
        result = await first.wait(deadline_ms=now_ms() + 10000)
        again = await runner.runtime.client.workflow(runner.target, spec, message_id="run", actor=runner.actor)
        assert first.input_ref == again.input_ref and first.target == again.target
        assert (await again.wait(deadline_ms=now_ms() + 10000)).to_dict() == result.to_dict()
        with pytest.raises(RuntimeRequestError, match="creation_conflict"):
            await runner.runtime.client.workflow(
                runner.target, {**spec, "root": leaf("B")}, message_id="run", actor=runner.actor,
            )
        with pytest.raises(RuntimeRequestError, match="forbidden"):
            await runner.runtime.client.wait(
                first.input_ref, actor=replace(runner.actor, tenant_id="foreign"), deadline_ms=None,
            )
        saved = await journal.read(store, runner.target.session_id, first.run_id)
        assert len(saved["nodes"]) == 1
        assert saved["nodes"][0]["result"]["status"] == "completed"
        assert saved["nodes"][0]["result_input_ref"]["message_id"].startswith("res_")
        assert saved["nodes"][0]["input_ref"]["message_id"].startswith("inp_")
        assert not {"job_id", "generation", "result_event"} & set(saved["nodes"][0])
        aggregate = await store._db.fetchall(
            f"SELECT seq FROM {store.t.messages} WHERE session_id=? AND role='assistant'",
            (first.target.session_id,),
        )
        assert len(aggregate) == 1
    finally:
        await runner.aclose()


@pytest.mark.parametrize("options", [None, RequestOptions(max_tokens=256, tools=[])])
async def test_retry_uses_original_frozen_driver_descriptor(registry_store, options):
    store, _ = registry_store
    runner = await RunningAgent(
        store, Model(), config=AgentLoopConfig(context_budget_tokens=10**9, temperature=0.25),
    ).start()
    client = runner.runtime.client
    spec = {"name": "frozen", "root": leaf("A")}
    try:
        first = await client.workflow(
            runner.target, spec, message_id="same", actor=runner.actor, options=options,
        )
        result = await first.wait(deadline_ms=now_ms() + 10000)
        again = await client.workflow(runner.target, spec, message_id="same", actor=runner.actor)
        assert again.input_ref == first.input_ref and again.target == first.target
        assert (await again.wait(deadline_ms=now_ms() + 10000)).to_dict() == result.to_dict()
        row = await store._db.fetchone(
            f"SELECT descriptor_json FROM {store.t.sessions} WHERE session_id=?",
            (first.target.session_id,),
        )
        descriptor = json.loads(row["descriptor_json"])
        assert descriptor["kind"] == "workflow_driver"
        assert descriptor["config"]["temperature"] == 0.25
        assert descriptor["config"]["workflow"]["spec"]["name"] == spec["name"]
        assert descriptor["config"]["workflow"]["spec"]["root"]["id"] == spec["root"]["id"]
        expected_tools = None if options is None else list(options.tools)
        assert descriptor["config"]["workflow"]["allowed_tools"] == expected_tools
        for change in (
            {"spec": {**spec, "root": leaf("B")}},
            {"descriptor": replace(runner.descriptor, config={**plain(runner.descriptor.config), "temperature": 0.75})},
            {"options": RequestOptions(temperature=0.75)},
            {"options": RequestOptions(tools=["new"])},
        ):
            with pytest.raises(RuntimeRequestError, match="creation_conflict"):
                await client.workflow(
                    runner.target, change.get("spec", spec), message_id="same", actor=runner.actor,
                    **{key: value for key, value in change.items() if key != "spec"},
                )
        fresh = await client.workflow(runner.target, spec, message_id="new", actor=runner.actor)
        assert fresh.target != first.target
        assert (await fresh.wait(deadline_ms=now_ms() + 10000)).status == "completed"
        nodes = await store._db.fetchall(
            f"SELECT session_id,parent_session_id FROM {store.t.sessions} "
            "WHERE descriptor_json LIKE '%workflow_node%'",
        )
        assert len(nodes) == 2
        assert {row["parent_session_id"] for row in nodes} == {first.target.session_id, fresh.target.session_id}
    finally:
        await runner.aclose()
