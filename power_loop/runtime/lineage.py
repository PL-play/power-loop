"""Durable workflow ancestry independent of ephemeral driver/session parents."""
from __future__ import annotations

from copy import deepcopy
from typing import Any


def workflow_ancestry(parent: Any) -> list[dict[str, Any]]:
    if parent is None:
        return []
    metadata = parent.metadata or {}
    ancestry = deepcopy(metadata.get('workflow_ancestry') or [])
    if metadata.get('workflow_run_id'):
        ancestry.append({'session_id': parent.session_id, 'workflow_run_id': metadata['workflow_run_id'],
                         'node_id': metadata.get('workflow_node_id'), 'node_attempt': metadata.get('attempt')})
    return ancestry
