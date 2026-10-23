"""Root and actual owner scope limits apply atomically to original model calls."""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from types import SimpleNamespace

import pytest

from power_loop import AgentLoopConfig, LLMResponse, LLMService, LLMTokenUsage, StatefulAgentLoop
from power_loop.core.agent_context import get_session_id
from power_loop.runtime import budget_scopes, delegation
from power_loop.runtime import model_attempts as attempts
from power_loop.runtime import model_budgets as budgets
from power_loop.runtime.budget_contracts import BudgetConfig, LocalBudgetScope
from power_loop.runtime.jobs import JobError, JobManager, _hash, _json
from power_loop.runtime.task_runtime import TaskRuntime
from power_loop.tools.registry import ToolRegistry
from power_loop.tools.spawn_agent import register_spawn_agent
from power_loop.workflow.facade import WorkflowToolPorts, register_workflow_tool

from .test_job_registry_192 import registry_store
from .test_model_budgets_192 import Host, new_send, observed, start

__all__ = ["registry_store"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.fixture
async def tree_case(registry_store, request):
    store, connect = registry_store
    params = getattr(request, "param", {})
    host, manager = Host(), JobManager(store)
    scopes = params.get(
        "scopes", (LocalBudgetScope("authority", max_calls=3), LocalBudgetScope("root", max_calls=1, binding="root"))
    )
    config = BudgetConfig("tree-host-v1", scopes, host, max_pending=64)
    runtime = TaskRuntime(
        manager,
        tenant_id="budget-tenant",
        principal_id="host",
        executor_id="executor",
        boot_id="boot",
        lease_ms=300000,
        boot_contract_hash="a" * 64,
        budget_config=config,
    )
    c = SimpleNamespace(store=store, connect=connect, host=host, manager=manager, config=config, runtime=runtime)
    try:
        yield c
    finally:
        await runtime.shutdown(timeout_s=2)


async def rows(c):
    result = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.prefix}task_budget_scopes")
    return [(row, json.loads(row["limits_json"]), json.loads(row["counters_json"])) for row in result]


async def test_roots_are_independent_but_authority_cap_is_shared_across_sends(tree_case):
    c = tree_case
    await new_send(c)
    root1, actor1 = c.binding, c.actor
    await observed(c, await start(c))
    with pytest.raises(JobError, match="budget_max_calls_exhausted"):
        await start(c)
    await new_send(c)
    root2 = c.binding
    await observed(c, await start(c))
    values = await rows(c)
    roots = [
        (limits["subject_job_id"], counter["used_calls"])
        for _, limits, counter in values
        if limits.get("binding") == "root"
    ]
    assert set(roots) == {(root1.lease.job_id, 1), (root2.lease.job_id, 1)}
    assert next(counter["used_calls"] for _, limits, counter in values if limits["key"] == "authority") == 2
    c.binding, c.actor, c.sid = root1, actor1, actor1.session_id
    # The rejected preparation now has durable non-dispatch closure. Its
    # original root allowance is still spent, so admission fails on that cap.
    with pytest.raises(JobError, match="budget_max_calls_exhausted"):
        await start(c)
    assert c.host.reserve_calls == 2


@pytest.mark.parametrize(
    "tree_case",
    [{"scopes": (LocalBudgetScope("authority", max_calls=1), LocalBudgetScope("root", max_calls=1, binding="root"))}],
    indirect=True,
)
async def test_independent_root_cannot_escape_authority_limit_by_getting_a_new_id(tree_case):
    c = tree_case
    await new_send(c)
    await observed(c, await start(c))
    await new_send(c)
    with pytest.raises(JobError, match="budget_max_calls_exhausted"):
        await start(c)
    assert c.host.reserve_calls == 1
    assert all(counter["held_calls"] == 0 for _, _, counter in await rows(c))


@pytest.mark.parametrize("fault", ["row", "marker", "limits", "counters"])
async def test_missing_or_corrupt_scope_instance_never_creates_a_new_allowance(tree_case, fault):
    c = tree_case
    await new_send(c)
    await observed(c, await start(c))
    [item] = [item for item in await rows(c) if item[1].get("binding") == "root"]
    row = item[0]
    if fault == "row":
        await c.manager.db.execute(
            f"DELETE FROM {c.manager.prefix}task_budget_scopes WHERE scope_key=?", (row["scope_key"],)
        )
    elif fault == "marker":
        await c.manager.db.execute(
            f"DELETE FROM {c.manager.events} WHERE event_id=?", ("mb_scope_" + row["scope_key"][3:],)
        )
    elif fault == "limits":
        await c.manager.db.execute(
            f"UPDATE {c.manager.prefix}task_budget_scopes SET limits_json='{{}}' WHERE scope_key=?", (row["scope_key"],)
        )
    else:
        await c.manager.db.execute(
            f"UPDATE {c.manager.prefix}task_budget_scopes SET counters_json='{{}}' WHERE scope_key=?",
            (row["scope_key"],),
        )
    code = {
        "row": "budget_scope_missing_or_changed",
        "marker": "budget_scope_evidence_corrupt",
        "limits": "budget_evidence_corrupt",
        "counters": "budget_evidence_corrupt",
    }[fault]
    with pytest.raises(JobError, match=code):
        await attempts.enable(c.binding)
    assert c.host.reserve_calls == 1


async def test_scope_limits_survive_reconstruction_of_the_original_runtime(tree_case):
    c = tree_case
    await new_send(c)
    await observed(c, await start(c))
    replacement = TaskRuntime(
        c.manager,
        tenant_id="budget-tenant",
        principal_id="host",
        executor_id="executor",
        boot_id="boot",
        lease_ms=300000,
        boot_contract_hash="a" * 64,
        budget_config=c.config,
    )
    try:
        old_binding, actor = c.binding, c.actor
        from power_loop.runtime.task_runtime import Execution

        c.binding = Execution(replacement, actor, old_binding.lease)
        with pytest.raises(JobError, match="budget_max_calls_exhausted"):
            await start(c)
        assert c.host.reserve_calls == 1
    finally:
        await replacement.shutdown(timeout_s=1)


@pytest.mark.parametrize(
    "tree_case",
    [{"scopes": (LocalBudgetScope("node", max_calls=1, binding="owner", owner_kinds=("workflow_node",)),)}],
    indirect=True,
)
async def test_nonapplicable_owner_template_is_not_an_unlimited_main_send(tree_case):
    c = tree_case
    with pytest.raises(JobError, match="budget_scope_not_applicable"):
        await new_send(c)
    assert c.host.reserve_calls == 0


async def test_cancelled_unstarted_attempt_refunds_exact_original_root_and_not_its_sibling(tree_case):
    c = tree_case
    await new_send(c)
    original = c.host.reserve

    async def stop_after_grant(plan):
        grant = await original(plan)
        await c.manager.request_cancel(c.actor, c.binding.lease.job_id, system_key="stop-original", reason="test")
        return grant

    c.host.reserve = stop_after_grant
    with pytest.raises(JobError, match="cancel_requested"):
        await start(c)
    [event] = await c.manager.db.fetchall(
        f"SELECT payload_json FROM {c.manager.events} WHERE event_type='model_attempt_prepared'"
    )
    prepared = json.loads(event["payload_json"])
    old = attempts.Attempt(c.binding, prepared["attempt_id"], prepared)
    old_actor, old_root = c.actor, c.binding.lease.job_id
    c.host.reserve = original
    await new_send(c)
    new_root = c.binding.lease.job_id
    await observed(c, await start(c))
    await budgets.close_unstarted(old)
    root_values = {
        limits["subject_job_id"]: count for _, limits, count in await rows(c) if limits.get("binding") == "root"
    }
    assert root_values[old_root]["used_calls"] == root_values[old_root]["held_calls"] == 0
    assert root_values[new_root]["used_calls"] == 1 and root_values[new_root]["held_calls"] == 0
    assert (await budgets.get_budget(c.manager, old_actor, old.attempt_id))["phase"] == "released"


@pytest.mark.parametrize("fault", ["row", "marker", "held_calls", "held_tokens", "inflight"])
async def test_final_dispatch_rechecks_original_scope_after_host_grants(tree_case, fault):
    c = tree_case
    await new_send(c)
    reserve = c.host.reserve

    async def damage_after_grant(plan):
        grant = await reserve(plan)
        [item] = [item for item in await rows(c) if item[1].get("binding") == "root"]
        row, _, counter = item
        if fault == "row":
            await c.manager.db.execute(
                f"DELETE FROM {c.manager.prefix}task_budget_scopes WHERE scope_key=?", (row["scope_key"],)
            )
        elif fault == "marker":
            await c.manager.db.execute(
                f"DELETE FROM {c.manager.events} WHERE event_id=?", ("mb_scope_" + row["scope_key"][3:],)
            )
        else:
            counter[fault] = 0
            await c.manager.db.execute(
                f"UPDATE {c.manager.prefix}task_budget_scopes SET counters_json=?,counters_hash=? WHERE scope_key=?",
                (_json(counter), _hash(counter), row["scope_key"]),
            )
        return grant

    c.host.reserve = damage_after_grant
    code = {"row": "budget_scope_missing_or_changed", "marker": "budget_scope_evidence_corrupt"}.get(
        fault, "budget_reservation_hold_missing"
    )
    with pytest.raises(JobError, match=code):
        await start(c)
    assert c.host.reserve_calls == 1
    assert not await c.manager.db.fetchall(
        f"SELECT event_id FROM {c.manager.events} WHERE event_type='model_attempt_dispatched'"
    )
    assert not c.host.applied and not c.host.released


async def test_lost_usage_ack_replays_original_frozen_tree_after_another_root_finishes(tree_case):
    c = tree_case
    await new_send(c)
    old_root = c.binding.lease.job_id
    attempt = await start(c)
    c.host.apply_lost_ack = True
    saved = await observed(c, attempt)
    assert saved["pending"] and saved["ack"] is None
    frozen_scopes, submission = saved["scopes"], saved["submission"]
    c.host.apply_lost_ack = False
    await new_send(c)
    new_root = c.binding.lease.job_id
    await observed(c, await start(c))
    before = {row["scope_key"]: counter for row, _, counter in await rows(c)}
    await budgets.deliver(attempt)
    await budgets.deliver(attempt)
    assert {row["scope_key"]: counter for row, _, counter in await rows(c)} == before
    final = await budgets.get_budget(c.manager, attempt.binding.actor, attempt.attempt_id)
    assert final["scopes"] == frozen_scopes and final["submission"] == submission and not final["pending"]
    assert c.host.reserve_calls == 2 and c.host.apply_calls == 3 and len(c.host.applied) == 2
    root_counters = {
        limits["subject_job_id"]: counter for _, limits, counter in await rows(c) if limits["key"] == "root"
    }
    assert set(root_counters) == {old_root, new_root}
    assert all(counter["used_calls"] == 1 and counter["held_calls"] == 0 for counter in root_counters.values())


@pytest.mark.parametrize(
    "fault", ["missing_authority", "missing_root", "duplicate", "foreign_root", "template", "kind", "key", "empty"]
)
async def test_frozen_scope_set_rejects_missing_or_changed_original_bindings(tree_case, fault):
    import copy

    c = tree_case
    await new_send(c)
    attempt = await start(c)
    state = copy.deepcopy(await budgets.get_budget(c.manager, c.actor, attempt.attempt_id))
    policy = {"policy_key": state["policy_key"], "manifest": c.config.manifest()}
    assert budget_scopes.frozen(policy, state) == state["scopes"]
    root = next(s for s in state["scopes"] if s["key"] == "root")
    if fault.startswith("missing_"):
        state["scopes"] = [s for s in state["scopes"] if s["key"] != fault.removeprefix("missing_")]
    elif fault == "duplicate":
        state["scopes"].append(copy.deepcopy(root))
    elif fault == "foreign_root":
        root["subject_job_id"] = "other-root"
    elif fault == "template":
        root["max_calls"] = 999
    elif fault == "empty":
        state["scopes"] = []
    else:
        root["subject_kind" if fault == "kind" else "key"] = []
    with pytest.raises(JobError, match="budget_scope_evidence_corrupt"):
        budget_scopes.frozen(policy, state)
    await observed(c, attempt)


@pytest.mark.parametrize("fault", ["cycle", "outside_tree", "generation", "executor", "boot"])
async def test_owner_lineage_requires_original_ancestor_execution_identity(fault):
    owner = {
        "job_id": "node",
        "generation": 1,
        "executor_id": "executor",
        "boot_id": "boot",
        "control_owner_job_id": None,
    }
    send = {
        "job_id": "send",
        "kind": "send",
        "control_owner_job_id": "node",
        "control_owner_generation": 1,
        "control_owner_executor_id": "executor",
        "control_owner_boot_id": "boot",
    }
    assert budget_scopes._owners(send, {"node": owner}) == [owner]
    if fault == "cycle":
        send["control_owner_job_id"] = "send"
    elif fault == "outside_tree":
        send["control_owner_job_id"] = "foreign-node"
    else:
        owner[{"generation": "generation", "executor": "executor_id", "boot": "boot_id"}[fault]] = (
            2 if fault == "generation" else "new"
        )
    with pytest.raises(JobError, match="budget_scope_lineage_changed"):
        budget_scopes._owners(send, {"node": owner})


class TreeModel(LLMService):
    def __init__(self, spec):
        self.spec, self.root, self.calls = spec, None, Counter()
        self.entered, self.release = asyncio.Event(), asyncio.Event()
        self.block = False
        self.spawn_from_build = False
        self.audit_entered = asyncio.Event()
        self.hold = set()
        self.next_main = []

    async def complete(self, request, **kwargs):
        sid = get_session_id()
        role = (
            "MAIN"
            if sid == self.root
            else next(
                (r for r in ("BUILD", "REVIEW", "AUDIT") if "TREE-" + r in (request.system_prompt or "")), "OTHER"
            )
        )
        self.calls[role] += 1
        text, calls = "42", None
        if role == "MAIN" and self.next_main:
            text, calls = (
                "",
                [
                    {
                        "id": "tree-control-" + str(self.calls[role]),
                        "type": "function",
                        "function": {"name": "workflow", "arguments": json.dumps(self.next_main.pop(0))},
                    }
                ],
            )
        elif role == "MAIN" and self.calls[role] == 1:
            text, calls = (
                "",
                [
                    {
                        "id": "create-tree",
                        "type": "function",
                        "function": {
                            "name": "workflow",
                            "arguments": json.dumps({"action": "create", "spec": self.spec}),
                        },
                    }
                ],
            )
        elif role == "BUILD" and self.spawn_from_build and self.calls[role] == 1:
            text, calls = (
                "",
                [
                    {
                        "id": "create-audit",
                        "type": "function",
                        "function": {
                            "name": "spawn_agent",
                            "arguments": json.dumps(
                                {
                                    "name": "audit",
                                    "system_prompt": "TREE-AUDIT",
                                    "tools": [],
                                    "task": "Return 42",
                                    "max_rounds": 2,
                                    "max_tokens": 20,
                                }
                            ),
                        },
                    }
                ],
            )
        elif role == "AUDIT":
            self.audit_entered.set()
        if role != "MAIN" and (self.block or role in self.hold):
            self.entered.set()
            await self.release.wait()
        return LLMResponse(
            raw_text=text,
            tool_calls=calls,
            token_usage=LLMTokenUsage(prompt_tokens=7, completion_tokens=3, total_tokens=10),
        )

    def stream(self, request):
        raise AssertionError("test uses original complete transport")

    async def close(self):
        pass


def leaf(name):
    return {
        "type": "agent",
        "id": name,
        "spec": {"name": name, "system_prompt": "TREE-" + name, "tools": [], "max_rounds": 2, "max_tokens": 20},
    }


@pytest.mark.parametrize(
    "tree_case",
    [
        {
            "scopes": (
                LocalBudgetScope("authority", max_calls=20),
                LocalBudgetScope("root", max_calls=20, binding="root"),
                LocalBudgetScope("workflow", max_calls=1, binding="owner", owner_kinds=("workflow",)),
                LocalBudgetScope("node", max_calls=2, binding="owner", owner_kinds=("workflow_node",)),
            )
        }
    ],
    indirect=True,
)
async def test_original_workflow_owner_cap_stops_second_node_before_any_provider_call(tree_case):
    c = tree_case
    spec = {"name": "tree-budget", "root": {"type": "sequence", "steps": [leaf("BUILD"), leaf("REVIEW")]}}
    model, accepted = TreeModel(spec), []
    c.host.service = model
    registry = ToolRegistry()

    async def on_accepted(admission):
        accepted.append(admission)

    register_workflow_tool(registry, ports=WorkflowToolPorts(on_accepted=on_accepted))
    loop = StatefulAgentLoop(
        llm=model,
        store=c.store,
        task_runtime=c.runtime,
        tool_registry=registry,
        config=AgentLoopConfig(delegation_policy="single_level", max_rounds=3, max_tokens=20),
    )
    sid = await loop.new_session()
    model.root = sid
    try:
        first = await asyncio.wait_for(loop.send("Create the workflow", sid), 20)
        assert first.status == "completed" and len(accepted) == 1
        handle = accepted[0].handle
        await asyncio.wait_for(handle.task, 20)
        workflow = await c.manager.get_job(c.runtime.actor(sid), handle.job_id)
        assert workflow.outcome == "failed" and model.calls["BUILD"] == 1 and model.calls["REVIEW"] == 0
        assert c.host.reserve_calls == sum(model.calls.values()) == 3
        [owner] = [counter for _, limits, counter in await rows(c) if limits["key"] == "workflow"]
        assert owner["used_calls"] == 1 and owner["held_calls"] == 0
        [root] = [counter for _, limits, counter in await rows(c) if limits["key"] == "root"]
        assert root["used_calls"] == 3
        assert "budget_max_calls_exhausted" in str(workflow.result)
    finally:
        model.release.set()
        if accepted:
            await asyncio.wait_for(asyncio.gather(*(a.handle.task for a in accepted), return_exceptions=True), 20)


@pytest.fixture
async def tree_workflow(tree_case):
    c = tree_case
    c.spec = {"name": "tree-budget", "root": {"type": "sequence", "steps": [leaf("BUILD"), leaf("REVIEW")]}}
    c.model, c.accepted = TreeModel(c.spec), []
    c.host.service = c.model
    registry = ToolRegistry()
    register_spawn_agent(registry)

    async def on_accepted(admission):
        c.accepted.append(admission)

    register_workflow_tool(registry, ports=WorkflowToolPorts(on_accepted=on_accepted))
    c.loop = StatefulAgentLoop(
        llm=c.model,
        store=c.store,
        task_runtime=c.runtime,
        tool_registry=registry,
        config=AgentLoopConfig(delegation_policy="single_level", max_rounds=3, max_tokens=20),
    )
    c.sid = await c.loop.new_session()
    c.model.root = c.sid
    before = set(delegation._TASKS)
    try:
        yield c
    finally:
        c.model.release.set()
        tasks = [a.handle.task for a in c.accepted] + list(set(delegation._TASKS) - before)
        if tasks:
            await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 20)


@pytest.mark.parametrize(
    "tree_case",
    [
        {
            "scopes": (
                LocalBudgetScope("authority", max_calls=20),
                LocalBudgetScope("root", max_calls=20, binding="root"),
                LocalBudgetScope("workflow", binding="owner", owner_kinds=("workflow",), **cap),
                LocalBudgetScope("node", max_calls=2, binding="owner", owner_kinds=("workflow_node",)),
            )
        }
        for cap in ({"max_calls": 1}, {"max_tokens": 150}, {"max_inflight": 1})
    ],
    indirect=True,
)
async def test_parallel_nodes_compete_for_one_workflow_call_before_the_first_response_returns(tree_workflow):
    c = tree_workflow
    c.spec["root"] = {
        "type": "parallel",
        "branches": [leaf("BUILD"), leaf("REVIEW")],
        "max_concurrency": 2,
        "on_error": "continue",
    }
    c.model.block = True
    first = await asyncio.wait_for(c.loop.send("Create parallel workflow", c.sid), 20)
    assert first.status == "completed" and len(c.accepted) == 1
    await asyncio.wait_for(c.model.entered.wait(), 20)
    for _ in range(500):
        failed = await c.manager.db.fetchone(
            f"SELECT job_id FROM {c.manager.jobs} WHERE kind='workflow_node' AND outcome='failed'"
        )
        if failed:
            break
        await asyncio.sleep(0.01)
    else:
        pytest.fail("The competing node was not rejected before dispatch")
    rejected = await c.manager.get_job(c.runtime.actor(c.sid), failed["job_id"])
    assert "budget_" in str(rejected.result) and "_exhausted" in str(rejected.result)
    [counter] = [counter for _, limits, counter in await rows(c) if limits["key"] == "workflow"]
    assert counter["held_calls"] == 1 and counter["used_calls"] == 0
    assert c.model.calls["BUILD"] + c.model.calls["REVIEW"] == 1
    assert c.host.reserve_calls == 3
    c.model.release.set()
    await asyncio.wait_for(c.accepted[0].handle.task, 20)
    [counter] = [counter for _, limits, counter in await rows(c) if limits["key"] == "workflow"]
    assert counter["held_calls"] == 0 and counter["used_calls"] == 1
    assert sum(c.model.calls.values()) == 3


@pytest.mark.parametrize(
    "tree_case",
    [
        {
            "scopes": (
                LocalBudgetScope("authority", max_calls=40),
                LocalBudgetScope("root", max_calls=40, binding="root"),
                LocalBudgetScope("workflow", max_calls=10, binding="owner", owner_kinds=("workflow",)),
                LocalBudgetScope("node", max_calls=2, binding="owner", owner_kinds=("workflow_node",)),
                LocalBudgetScope("agent", max_calls=1, binding="owner", owner_kinds=("subagent",)),
            )
        }
    ],
    indirect=True,
)
async def test_subagent_charges_its_original_node_and_workflow_without_double_billing_the_host(tree_workflow):
    c = tree_workflow
    c.model.spawn_from_build = True
    c.spec["root"]["steps"][0]["spec"]["tools"] = ["spawn_agent"]
    prepare = c.host.prepare

    async def wait_for_original_child(context, logical_request, *, service):
        if "TREE-BUILD" in (logical_request.system_prompt or "") and c.model.calls["BUILD"] == 1:
            # The host gates reservation, before the parent can take the last
            # node slot. The child's actual call wins it; no result is faked.
            await asyncio.wait_for(c.model.audit_entered.wait(), 10)
        return await prepare(context, logical_request, service=service)

    c.host.prepare = wait_for_original_child
    first = await asyncio.wait_for(c.loop.send("Create workflow with one audit", c.sid), 20)
    assert first.status == "completed" and len(c.accepted) == 1
    await asyncio.wait_for(c.accepted[0].handle.task, 20)
    await asyncio.wait_for(asyncio.gather(*list(delegation._TASKS), return_exceptions=True), 20)
    assert c.model.calls == {"MAIN": 2, "BUILD": 1, "AUDIT": 1}
    counters_by_template = {
        name: [counter for _, limits, counter in await rows(c) if limits["key"] == name]
        for name in ("authority", "root", "workflow", "node", "agent")
    }
    assert counters_by_template["authority"][0]["used_calls"] == counters_by_template["root"][0]["used_calls"] == 4
    assert counters_by_template["workflow"][0]["used_calls"] == 2
    assert sum(c["used_calls"] for c in counters_by_template["node"]) == 2
    assert counters_by_template["agent"][0]["used_calls"] == 1
    assert c.host.reserve_calls == 4 and len(c.host.applied) == 4
    assert all(counter["held_calls"] == 0 for _, _, counter in await rows(c))


@pytest.mark.parametrize(
    "tree_case",
    [
        {
            "scopes": (
                LocalBudgetScope("authority", max_calls=40),
                LocalBudgetScope("root", max_calls=40, binding="root"),
                LocalBudgetScope("workflow", max_calls=10, binding="owner", owner_kinds=("workflow",)),
                LocalBudgetScope("node", max_calls=1, binding="owner", owner_kinds=("workflow_node",)),
            )
        }
    ],
    indirect=True,
)
async def test_original_node_resume_does_not_get_a_fresh_owner_allowance_or_move_to_the_new_main_root(tree_workflow):
    from power_loop.workflow import runner

    c = tree_workflow
    c.spec["root"] = {
        "type": "sequence",
        "steps": [
            {"type": "parallel", "branches": [leaf("BUILD"), leaf("AUDIT")], "max_concurrency": 2},
            leaf("REVIEW"),
        ],
    }
    c.model.hold = {"AUDIT"}
    await asyncio.wait_for(c.loop.send("Create workflow", c.sid), 20)
    await asyncio.wait_for(c.model.audit_entered.wait(), 20)
    [accepted] = c.accepted
    engine = runner._ENGINES[accepted.handle.run_id]
    build = None
    for _ in range(500):
        [row] = await c.manager.db.fetchall(
            f"SELECT job_id FROM {c.manager.jobs} WHERE kind='workflow_node' AND title='BUILD'"
        )
        build = await c.manager.get_job(c.runtime.actor(c.sid), row["job_id"])
        if build.outcome == "succeeded" and any(
            not f["active"] and f["result"] and f["result"].node_id == "BUILD" for f in engine._frames.values()
        ):
            break
        await asyncio.sleep(0.01)
    else:
        pytest.fail("Original BUILD did not settle")
    previous = build.result
    from power_loop.runtime.task_sessions import list_sessions

    saved = await list_sessions(c.manager, c.runtime.actor(c.sid), build.job_id)
    [physical] = saved["bindings"]
    original_sid = physical["session_id"]
    assert physical["generation"] == 1 and original_sid == previous["session_id"]
    c.model.next_main.append(
        {
            "action": "nudge",
            "run_id": accepted.handle.run_id,
            "node": "BUILD",
            "sid": original_sid,
            "message": "Check the original result again",
        }
    )
    control = await asyncio.wait_for(c.loop.send("Correct BUILD", c.sid), 20)
    assert control.status == "completed"
    messages = await c.store.load_all_messages(c.sid)
    responses = [json.loads(m.content) for m in messages if m.role == "tool" and m.name == "workflow"]
    assert responses[-1].get("generation") == 2 and responses[-1].get("session_id") == original_sid, responses
    c.model.release.set()
    await asyncio.wait_for(accepted.handle.task, 20)
    final = await c.manager.get_job(c.runtime.actor(c.sid), build.job_id)
    assert final.generation == 2 and final.outcome == "failed" and c.model.calls["BUILD"] == 1
    assert (await c.manager.get_result_revision(c.runtime.actor(c.sid), build.job_id, 1))["payload"][
        "result"
    ] == previous
    [node_counter] = [counter for _, limits, counter in await rows(c) if limits.get("subject_job_id") == build.job_id]
    assert node_counter["used_calls"] == 1 and node_counter["held_calls"] == 0
    assert c.model.calls["REVIEW"] == 0 and c.host.reserve_calls == sum(c.model.calls.values())
    roots = [counter for _, limits, counter in await rows(c) if limits["key"] == "root"]
    assert sorted(counter["used_calls"] for counter in roots) == [2, 4]
