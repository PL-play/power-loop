"""The timing helper must wait for completed work, not submission alone."""
import asyncio

import pytest

from tests.design203_benchmark import batch, percentile


@pytest.mark.asyncio
async def test_batch_includes_completion_and_exact_success_count():
    completed = []

    async def operation():
        await asyncio.sleep(0.01)
        completed.append(True)

    result = await batch([operation] * 8, 34)
    assert len(completed) == result['completed'] == 34
    assert result['errors'] == result['backlog'] == 0
    assert result['p50_ms'] >= 10
    assert result['seconds'] >= .05  # five waves, including final completion
    assert result['throughput_rps'] == 34 / result['seconds']


def test_percentile_does_not_report_the_fastest_sample():
    assert percentile(list(range(1, 101)), .95) == 95
