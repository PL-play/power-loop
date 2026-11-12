"""Active workflow node steering uses exact durable routes and original sessions."""
from __future__ import annotations

import json

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def target(c):
    rows = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.events} WHERE event_type='node_session'")
    return [(r, json.loads(r["payload_json"])) for r in rows]
