"""Configurable MAX_SPAWN_DEPTH.

The depth ceiling is no longer a hard module constant: it lives on the
``SessionStore`` instance (default :data:`MAX_SPAWN_DEPTH`) and can be set via
``SessionStore.open(max_spawn_depth=...)`` or the ``StatefulAgentLoop``
``max_spawn_depth`` argument. Both enforcement points honor it:

* ``SessionStore.create_session`` raises past the limit;
* ``run_agent_spec`` pre-checks and returns ``status="rejected"``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from power_loop import (
    MAX_SPAWN_DEPTH,
    RuntimeRequestError,
    SessionStore,
)
from tests.unit.session_native_fixture import ACTOR as STORE_ACTOR
from tests.unit.session_native_fixture import DESCRIPTOR as STORE_DESCRIPTOR

pytestmark = pytest.mark.unit


@pytest.fixture
async def store() -> AsyncIterator[SessionStore]:
    s = await SessionStore.open(":memory:")
    yield s
    await s.close()


# ── store-level config ────────────────────────────────────────────────────


async def test_default_is_module_constant() -> None:
    s = await SessionStore.open(":memory:")
    try:
        assert s.max_spawn_depth == MAX_SPAWN_DEPTH
    finally:
        await s.close()


async def test_open_accepts_custom_limit(registry_store) -> None:
    s, _ = registry_store
    s.max_spawn_depth = 1
    assert s.max_spawn_depth == 1
    root = await s.create_session(STORE_DESCRIPTOR, actor=STORE_ACTOR)
    child = await s.create_session(STORE_DESCRIPTOR, actor=STORE_ACTOR, parent_session_id=root)  # depth 1 == limit: ok
    with pytest.raises(RuntimeRequestError, match="exceeds max 1"):
        await s.create_session(STORE_DESCRIPTOR, actor=STORE_ACTOR, parent_session_id=child)  # depth 2 > 1: rejected


async def test_higher_limit_allows_deeper_chain() -> None:
    s = await SessionStore.open(":memory:", max_spawn_depth=5)
    try:
        cur = await s.create_session(STORE_DESCRIPTOR, actor=STORE_ACTOR)
        for _ in range(5):  # would blow the default of 3
            cur = await s.create_session(STORE_DESCRIPTOR, actor=STORE_ACTOR, parent_session_id=cur)
        assert s.max_spawn_depth == 5
    finally:
        await s.close()


@pytest.mark.parametrize("bad", [0, -1, "x", None])
async def test_invalid_limit_rejected(bad: object) -> None:
    with pytest.raises(ValueError, match="max_spawn_depth"):
        await SessionStore.open(":memory:", max_spawn_depth=bad)  # type: ignore[arg-type]


def test_property_setter_validates(store: SessionStore) -> None:
    store.max_spawn_depth = 7
    assert store.max_spawn_depth == 7
    with pytest.raises(ValueError, match="max_spawn_depth"):
        store.max_spawn_depth = 0


@pytest.mark.parametrize("depth", [1, 2, 9])
async def test_runtime_uses_explicit_store_depth_without_overriding_it(store, runtime_agent, depth):
    import json

    from power_loop import AgentLoopConfig
    from power_loop.tools.registry import ToolRegistry
    from power_loop.tools.spawn_agent import register_spawn_agent
    from tests.unit.test_subagent_sessions_215 import ParentAndChild

    store.max_spawn_depth = depth
    registry = ToolRegistry()
    register_spawn_agent(registry)
    agent = await runtime_agent(model=ParentAndChild(), store=store, registry=registry,
                                config=AgentLoopConfig(context_budget_tokens=10**9), capacity=2)
    result = await agent.send("delegate")
    assert result.status == "completed"
    await agent.drain()
    rows = await store.load_all_messages(agent.target.session_id)
    child = json.loads(next(row.content for row in rows if row.tool_call_id == "spawn"))
    assert store.max_spawn_depth == depth
    row = await store.get_session(child["session_id"])
    assert row.parent_session_id == agent.target.session_id
