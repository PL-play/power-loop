"""Workflow admission and graph execution through real client/worker paths."""

import pytest

from power_loop import LLMResponse, LLMService

from .test_job_registry_192 import registry_store

__all__ = ["registry_store"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


class Model(LLMService):
    async def complete(self, request, **kwargs):
        return LLMResponse(raw_text="result:" + request.system_prompt)

    async def close(self):
        pass


def leaf(name):
    return {
        "type": "agent",
        "id": name,
        "input": name,
        "spec": {"name": name, "system_prompt": name, "tools": [], "max_rounds": 2},
    }
