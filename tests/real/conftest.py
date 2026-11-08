"""Auto-apply the `real_llm` marker to every test under tests/real/.

This relies on the parent ``tests/conftest.py`` to handle the actual
skip-when-missing-env logic.
"""
from __future__ import annotations

import pytest


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    real_marker = pytest.mark.real_llm
    for item in items:
        if "tests/real/" in str(item.fspath).replace("\\", "/"):
            item.add_marker(real_marker)


@pytest.fixture
async def runtime_agent(runtime_agent):
    """Tests own live providers across inputs; workers close their borrowed views."""
    from ._live_fixture import BorrowedProvider
    providers, agents = {}, []
    async def create(*, model, **options):
        providers[id(model)] = model
        agent = await runtime_agent(model=BorrowedProvider(model), **options)
        agents.append(agent)
        return agent
    yield create
    for agent in reversed(agents):
        await agent.aclose()
    for provider in providers.values():
        await provider.close()
