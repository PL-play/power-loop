"""Original node continuation must not inspect another tree under its writer."""
from __future__ import annotations

import pytest

from .test_job_registry_192 import registry_store

__all__ = ['registry_store']
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]
