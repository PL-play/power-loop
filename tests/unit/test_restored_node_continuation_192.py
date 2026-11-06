"""The original leaf adapter cannot silently adopt another workflow execution."""
from __future__ import annotations

import pytest

from .test_job_registry_192 import registry_store

__all__ = ['registry_store']
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]
