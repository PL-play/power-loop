"""Actual model delivery/consumption followed by a non-executing recovery plan."""
from __future__ import annotations

import pytest

pytestmark = [pytest.mark.real_llm, pytest.mark.asyncio]
