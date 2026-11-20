"""Container-level acceptance of design 215. No in-process replacement nodes."""
import asyncio
import json

import httpx
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


async def test_three_identities_fixed_owner_next_send_and_busy_batch(cluster):
    c = cluster
    health = await asyncio.gather(*(c.health(n) for n in "abc"))
    for field in ("node_id", "boot_id", "hostname", "workspace", "local_namespace"):
        assert len({h[field] for h in health}) == 3, field
    assert len({h["store_id"] for h in health}) == 1
    await c.configure("a", enabled=True)
    sid = await c.create(mode="loop", rounds=4, gates=[1])
    await c.send(sid)
    await c.event("model", session=sid, node="a", call=1)
    before = len((await c.health("b"))["published"])
    receipts = [await c.send(sid, f"busy-{i}", body=f"input {i}", priority="urgent" if i == 4 else "normal")
                for i in range(5)]
    assert all(r["notification"] == "not_needed" for r in receipts)
    assert len((await c.health("b"))["published"]) == before
    await c.release(sid, "model:1")
    assert (await c.result(sid))["status"] == "completed"
    results = [await c.result(sid, f"busy-{i}") for i in range(5)]
    assert {r["send_index"] for r in results} == {1}
    rows = await c.messages(sid)
    merged = [r for r in rows if r["role"] == "user" and "busy-0" in r["content"]]
    assert len(merged) == 1 and all(f"busy-{i}" in merged[0]["content"] for i in range(5))
    calls = await c.events("model", session=sid)
    assert len(calls) == 4 and {r["node"] for r in calls} == {"a"}
    await c.configure("a", enabled=False)
    await c.configure("b", enabled=True)
    await c.send(sid, "next")
    result = await c.result(sid, "next")
    assert result["status"] == "completed" and result["send_index"] == 2
    assert {r["node"] for r in await c.events("model", session=sid, send=2)} == {"b"}


async def test_expired_paused_owner_rejected_and_new_send_on_c(cluster):
    c = cluster
    await c.configure("a", enabled=True)
    sid = await c.create(gates=[1])
    await c.send(sid)
    await c.event("model", session=sid, node="a")
    await c.docker("pause", "a")
    await c.configure("c", enabled=True)
    assert (await c.result(sid))["status"] == "interrupted"
    await c.release(sid, "model:1")
    await c.send(sid, "new")
    assert (await c.result(sid, "new"))["final_text"] == "done c"
    before = await c.messages(sid)
    await c.docker("unpause", "a")
    await asyncio.sleep(.5)
    assert await c.messages(sid) == before
    assert not any(r["role"] == "assistant" and r["content"] == "done a" for r in before)


async def test_incapable_notification_loss_and_duplicate_activation(cluster):
    c = cluster
    await c.configure("abc", drop_notifications=True)
    await c.configure("a", enabled=True, roles=["background"])
    sid = await c.create()
    await c.send(sid)
    await c.call("a", "recover")
    status = await c.call("b", "inspect", session=sid, message_id="start")
    assert status["input_state"] == "pending" and status["terminal_status"] is None
    assert await c.messages(sid) == []
    await c.configure("bc", enabled=True)
    await asyncio.gather(*(c.call(n, "recover") for n in "abcabc"))
    assert (await c.result(sid))["status"] == "completed"
    assert len(await c.events("build", session=sid)) == 1


@pytest.mark.parametrize("mode", ["model", "tool"])
async def test_container_kill_repairs_unknown_without_replay(cluster, mode):
    c = cluster
    await c.configure("a", enabled=True)
    sid = await c.create(**({"gates": [1]} if mode == "model" else
                            {"mode": "loop", "rounds": 2, "tool_gate": True}))
    await c.send(sid)
    await c.event(mode, session=sid, node="a")
    await c.docker("kill", "a")
    await c.configure("c", enabled=True)
    assert (await c.result(sid))["status"] == "interrupted"
    if mode == "tool":
        rows = await c.messages(sid)
        results = [r for r in rows if r["role"] == "tool"]
        assert len(results) == 1 and results[0]["send_index"] == 1
        assert len(await c.events("tool", session=sid)) == 1
    await c.release(sid, "model:1")
    await c.release(sid, "tool")
    await c.send(sid, "continue")
    assert (await c.result(sid, "continue"))["status"] == "completed"


async def _workflow_leaves(c, driver):
    facts = await c.call("b", "facts", events=["claim_return"])
    rows = [row for row in facts if row.get("claimed") and row["kind"] == "agent"
            and row.get("parent") == driver and row["send"] == 1]
    leaves = {row["session"]: row["inputs"][0] for row in rows}
    return leaves if len(leaves) == 2 else None


async def _leaf_gates(c, leaves):
    rows = await c.events("gate", node="b", phase="enter")
    keys = {row["key"] for row in rows}
    return rows if all(f"{leaf}:child" in keys for leaf in leaves) else None


async def _pending_leaf_reads(c, driver, leaves):
    reads = await c.call("b", "facts", events=["result_read_return"])
    pending = {(row["session"], row["input"]) for row in reads
               if row["actor"] == driver and not row["terminal"]}
    return reads if all((leaf, input_id) in pending for leaf, input_id in leaves.items()) else None


async def _results_delivered(c, expected, *, send=1):
    facts = await c.call("b", "result_facts")
    for target, input_id in expected.items():
        if not any(
            fact["session"] == target and fact["result_ref"]
            and any(result["input"] == input_id and (send is None or result["send"] == send)
                    and result["status"] == "completed" and result["delivery_error"] is None
                    for result in fact["results"])
            for fact in facts
        ):
            return None
    return facts


def _assert_workflow_result(result, driver, input_id, leaves):
    assert result["input_ref"]["session_id"] == driver
    assert result["input_ref"]["message_id"] == input_id
    assert result["status"] == "completed"
    nodes = result["structured_output"]["results"]
    assert set(nodes) == {"N0", "N1"}
    assert {node["session_id"] for node in nodes.values()} == set(leaves)
    assert all(node["status"] == "completed" for node in nodes.values())


@pytest.mark.parametrize("mode,kind", [("spawn", "agent"), ("workflow", "workflow_driver"),
                                       ("background", "background")])
async def test_real_child_workflow_background_result_routes(cluster, mode, kind):
    c = cluster
    await c.configure("abc", drop_notifications=mode != "workflow")
    await c.configure("a", enabled=True, roles=["main"])
    await c.configure("b", enabled=True, roles=["child", "workflow_driver", "background"],
                      delay_second_spawn=mode == "spawn")
    sid = await c.create(mode=mode, gates=[2])
    await c.send(sid)
    [parent_call] = await c.event("model", session=sid, node="a", call=2)
    tool_name = {"spawn": "spawn_agent", "workflow": "workflow", "background": "background_run"}[mode]
    receipts = [json.loads(row["content"]) for row in parent_call["messages"]
                if row["role"] == "tool" and row["name"] == tool_name]
    expected = {row["target_ref"]["session_id"]: row["input_ref"]["message_id"] for row in receipts}
    assert len(expected) == (2 if mode == "spawn" else 1)
    if mode == "spawn":
        await c.event("build", node="b", kind=kind)
        legacy_builds = [r for r in await c.events("build") if r["session"] in expected]
        delayed = set(expected) - {r["session"] for r in legacy_builds}
        c.note("legacy_snapshot", expected=sorted(expected),
               builds=[r["session"] for r in legacy_builds], omitted=sorted(delayed))
        assert len(delayed) == 1
        for target in delayed:
            await c.release(target, "build")

    async def expected_builds():
        rows = [r for r in await c.events("build", node="b", kind=kind) if r["session"] in expected]
        return rows if {r["session"] for r in rows} == set(expected) else None

    builds = await c.until(expected_builds)
    leaves = set()
    if mode == "workflow":
        driver, = expected
        leaves = await c.until(lambda: _workflow_leaves(c, driver))
        await c.until(lambda: _leaf_gates(c, leaves))
        await c.until(lambda: _pending_leaf_reads(c, driver, leaves))
    else:
        suffix = "tool" if mode == "background" else "child"

        async def target_gates():
            rows = await c.events("gate", node="b", phase="enter")
            keys = {r["key"] for r in rows}
            return rows if all(f"{target}:{suffix}" in keys for target in expected) else None
        await c.until(target_gates)
    for child in set(expected) | set(leaves):
        await c.release(child, "child")
        await c.release(child, "tool")

    await c.until(lambda: _results_delivered(c, expected))
    for target, input_id in expected.items():
        result = await c.result(target, input_id)
        assert result["status"] == "completed"
        if mode == "workflow":
            assert result["send_index"] == 1
            _assert_workflow_result(result, target, input_id, leaves)
            assert len(await c.events("build", session=target)) == 1
    c.note("parent_release", session=sid, expected=expected, leaves=sorted(leaves), builds=[{
        "session": row["session"], "send": row["send"], "epoch": row["epoch"], "kind": row["kind"],
    } for row in builds])
    await c.release(sid, "model:2")
    assert (await c.result(sid))["status"] == "completed"
    async def parent_consumed():
        rows = await c.messages(sid)
        result_kind = {"spawn": "agent_result", "workflow": "workflow_result", "background": "background_result"}[mode]
        return [r for r in rows if r["role"] == "user" and result_kind in r["content"]]
    merged = await c.until(parent_consumed)
    assert len(merged) == 1
    if mode == "spawn":
        assert merged[0]["content"].count('"kind":"agent_result"') == 2


async def test_workflow_notification_loss_explicit_recovery(cluster):
    c = cluster
    await c.configure("abc", drop_notifications=True)
    await c.configure("a", enabled=True, roles=["main"])
    await c.configure("b", enabled=True, roles=["child", "workflow_driver", "background"])
    sid = await c.create(mode="workflow", gates=[2])
    await c.send(sid)
    [parent_call] = await c.event("model", session=sid, node="a", call=2)
    [receipt] = [json.loads(row["content"]) for row in parent_call["messages"]
                 if row["role"] == "tool" and row["name"] == "workflow"]
    driver = receipt["target_ref"]["session_id"]
    driver_input = receipt["input_ref"]["message_id"]
    await c.event("build", session=driver, node="b", kind="workflow_driver")
    leaves = await c.until(lambda: _workflow_leaves(c, driver))
    await c.until(lambda: _leaf_gates(c, leaves))
    await c.until(lambda: _pending_leaf_reads(c, driver, leaves))
    for leaf in leaves:
        await c.release(leaf, "child")
        await c.release(leaf, "tool")
    await c.until(lambda: _results_delivered(c, leaves))

    def driver_reads(rows):
        return [row for row in rows if row["actor"] == driver
                and (row["session"], row["input"]) in leaves.items()]

    async def read_facts():
        return await c.call("b", "facts", events=["result_read_return"])

    initial_reads = driver_reads(await read_facts())
    assert len(initial_reads) == 2 and all(not row["terminal"] for row in initial_reads)

    async def successful_renews():
        rows = await c.call("b", "facts", events=["renew_return"])
        return len([row for row in rows if row["session"] == driver
                    and row["epoch"] == 1 and row["renewed"]])

    renews_before = await successful_renews()

    async def two_more_renews():
        return await successful_renews() >= renews_before + 2

    await c.until(two_more_renews)
    renews_after = await successful_renews()
    assert driver_reads(await read_facts()) == initial_reads
    assert not [row for row in await c.call("b", "result_facts") if row["session"] == driver]
    status = await c.call("b", "inspect", session=driver, message_id=driver_input)
    assert status["execution_state"] == "running" and status["terminal_status"] is None
    assert not await c.events("gate", node="a", phase="exit", key=f"{sid}:model:2")

    a_before = await c.health("a")
    b_before = await c.health("b")
    before_trace = await c.call("b", "trace")
    c.note("before_explicit_recovery", parent=sid, driver=driver, driver_input=driver_input,
           leaves=leaves, parent_boot=a_before["boot_id"], driver_boot=b_before["boot_id"],
           successful_renews=renews_after)
    c.save("before-explicit-recovery", {
        "parent": sid, "driver": driver, "driver_input": driver_input, "leaves": leaves,
        "parent_boot": a_before["boot_id"], "driver_boot": b_before["boot_id"],
        "initial_driver_reads": initial_reads, "successful_renews": renews_after,
        "driver_status": status, "trace": before_trace, "commands": c.commands,
    })
    c.note("explicit_recovery_now", action="restart_b", parent=sid, driver=driver,
           parent_boot=a_before["boot_id"], old_driver_boot=b_before["boot_id"])
    await c.docker("restart", "b")
    await c.ready()
    assert (await c.health("a"))["boot_id"] == a_before["boot_id"]
    b_after = await c.health("b")
    assert b_after["boot_id"] != b_before["boot_id"]
    await c.configure("b", enabled=True, roles=["child", "workflow_driver", "background"],
                      drop_notifications=True, delay_second_spawn=False)
    c.note("explicit_recovery_configured", driver=driver, new_driver_boot=b_after["boot_id"])

    await c.until(lambda: _results_delivered(c, {driver: driver_input}, send=None))
    result = await c.result(driver, driver_input)
    assert result["send_index"] > 1
    _assert_workflow_result(result, driver, driver_input, leaves)
    for leaf, input_id in leaves.items():
        leaf_result = await c.result(leaf, input_id)
        assert leaf_result["input_ref"]["message_id"] == input_id
        assert leaf_result["status"] == "completed" and leaf_result["send_index"] == 1
        models = await c.events("model", session=leaf)
        builds = await c.events("build", session=leaf)
        assert len(models) == 1 and models[0]["boot"] == b_before["boot_id"]
        assert len(builds) == 1 and builds[0]["boot"] == b_before["boot_id"]
        assert not await c.events("tool", session=leaf)

    c.note("parent_release_after_explicit_recovery", parent=sid, driver=driver,
           driver_input=driver_input, leaves=leaves)
    await c.release(sid, "model:2")
    parent_result = await c.result(sid)
    assert parent_result["status"] == "completed"

    async def parent_consumed():
        rows = await c.messages(sid)
        return [row for row in rows if row["role"] == "user" and "workflow_result" in row["content"]]

    merged = await c.until(parent_consumed)
    assert len(merged) == 1
    c.save("after-explicit-recovery", {
        "parent": sid, "driver": driver, "driver_input": driver_input, "leaves": leaves,
        "parent_boot": (await c.health("a"))["boot_id"], "driver_boot": b_after["boot_id"],
        "driver_result": result, "parent_result": parent_result, "merged_count": len(merged),
        "trace": await c.call("b", "trace"), "commands": c.commands,
    })


@pytest.mark.parametrize("stage", ["accept", "claim", "final"])
async def test_commit_ack_unknown_keeps_one_logical_input(cluster, stage):
    c = cluster
    await c.configure("a", enabled=True)
    sid = await c.create()
    node = "b" if stage == "accept" else "a"
    await c.configure(node, fault={"stage": stage, "action": "unknown"})
    if stage == "accept":
        with pytest.raises(AssertionError, match="StoreError"):
            await c.send(sid)
        receipt = await c.send(sid)
        assert not receipt["newly_accepted"]
    else:
        await c.send(sid)
    await c.event("commit_fault", stage=stage)
    result = await c.result(sid)
    assert result["status"] in {"completed", "interrupted"}
    assert len([r for r in await c.messages(sid) if r["role"] == "user"]) == 1
    await c.send(sid, "after")
    assert (await c.result(sid, "after"))["status"] == "completed"


@pytest.mark.parametrize("stage", ["accept", "claim", "final"])
async def test_container_kill_at_committed_boundaries(cluster, stage):
    c = cluster
    await c.configure("a", enabled=True)
    sid = await c.create()
    await c.configure("a", fault={"stage": stage, "action": "block"})
    sending = asyncio.create_task(c.send(sid, node="a"))
    await c.event("commit_fault", stage=stage, node="a")
    await c.docker("kill", "a")
    try:
        await sending
    except httpx.HTTPError:
        assert stage == "accept"
    await c.configure("c", enabled=True)
    result = await c.result(sid)
    assert result["status"] == ("interrupted" if stage == "claim" else "completed")
    assert len([r for r in await c.messages(sid) if r["role"] == "user"]) == 1
    await c.send(sid, "after")
    assert (await c.result(sid, "after"))["status"] == "completed"


async def test_driver_container_dies_healthy_leaf_continues(cluster):
    c = cluster
    await c.configure("a", enabled=True, roles=["main"])
    await c.configure("b", enabled=True, roles=["workflow_driver"])
    await c.configure("c", enabled=True, roles=["child"])
    sid = await c.create(mode="workflow", gates=[2])
    await c.send(sid)
    await c.event("model", session=sid, node="a", call=2)
    async def leaf_pair():
        rows = await c.events("model", node="c")
        return rows if len(rows) == 2 else None
    leaves = await c.until(leaf_pair)
    rows = await c.messages(sid)
    receipt = json.loads(next(r["content"] for r in rows if r["role"] == "tool"))
    await c.docker("kill", "b")
    c.controller = "c"
    await c.configure("a", roles=["main", "workflow_driver"])
    for leaf in leaves:
        await c.release(leaf["session"], "child")
    result = await c.result(receipt["input_ref"]["session_id"], receipt["input_ref"]["message_id"])
    assert result["status"] == "completed" and len(result["structured_output"]["results"]) == 2
    for leaf in leaves:
        assert len(await c.events("build", session=leaf["session"])) == 1
    await c.release(sid, "model:2")
    assert (await c.result(sid))["status"] == "completed"


async def test_linked_and_independent_cancel_from_other_container(cluster):
    c = cluster
    await c.configure("a", enabled=True)
    parent = await c.create(gates=[1])
    children = {}
    for propagation in ("linked", "independent"):
        child = await c.call("b", "create", parent=parent, cancel_propagation=propagation,
                             config={"gates": [1]})
        children[propagation] = child["session_id"]
    for sid in [parent, *children.values()]:
        await c.send(sid)
        await c.event("model", session=sid, node="a")
    await c.call("c", "cancel", session=parent, scope="subtree")
    for sid in [parent, *children.values()]:
        await c.release(sid, "model:1")
    assert (await c.result(parent))["status"] == "cancelled"
    assert (await c.result(children["linked"]))["status"] == "cancelled"
    assert (await c.result(children["independent"]))["status"] == "completed"


async def test_container_network_loss_has_no_coordinator_fallback(cluster):
    c = cluster
    await c.configure("a", enabled=True)
    sid = await c.create()
    await c.docker("exec", "-T", "a", "iptables", "-A", "OUTPUT", "-p", "tcp", "--dport", "35432", "-j", "REJECT")
    try:
        await c.send(sid)
        await asyncio.sleep(1)
        assert not await c.events("build", session=sid)
        assert (await c.call("b", "inspect", session=sid, message_id="start"))["input_state"] == "pending"
    finally:
        await c.docker("exec", "-T", "a", "iptables", "-D", "OUTPUT", "-p", "tcp", "--dport", "35432", "-j", "REJECT")
    await c.call("a", "recover")
    assert (await c.result(sid))["status"] == "completed"


async def test_projection_loss_reads_originals_on_next_node(cluster):
    c = cluster
    await c.configure("a", enabled=True)
    sid = await c.create(projected=True)
    await c.send(sid, body="original fact one")
    assert (await c.result(sid))["status"] == "completed"
    original = await c.messages(sid)
    await c.configure("a", enabled=False)
    await c.call("b", "drop_projection", session=sid)
    await c.configure("b", enabled=True)
    await c.send(sid, "next", body="continue")
    assert (await c.result(sid, "next"))["status"] == "completed"
    [call] = await c.events("model", session=sid, node="b")
    assert "original fact one" in json.dumps(call["messages"])
    assert "done a" in json.dumps(call["messages"])
    assert (await c.messages(sid))[:len(original)] == original
