"""Actual background receipt -> wait -> same-session evidence consumption."""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.real_llm, pytest.mark.asyncio]
