"""Monitoring tools for ExecutionSnapshot health and performance.

Provides tools to monitor:
- Snapshot size distribution
- Snapshot vs journal consistency
- Snapshot availability rate
- Performance metrics
"""
import json
from typing import Dict, List, Any
from dataclasses import dataclass, asdict


@dataclass
class SnapshotMetrics:
    """Metrics for snapshot health monitoring."""
    job_id: str
    generation: int
    snapshot_exists: bool
    snapshot_valid: bool
    snapshot_size_bytes: int
    pending_tool_calls_count: int
    round_index: int
    total_tokens: int
    captured_at_ms: int
    source: str  # "snapshot" or "events_reconstructed"


@dataclass
class PerformanceMetrics:
    """Performance comparison metrics."""
    snapshot_inspect_ms: float
    journal_inspect_ms: float
    speedup_percent: float
    checkpoint_count: int


async def collect_snapshot_metrics(manager, job_id: str, generation: int) -> SnapshotMetrics:
    """Collect metrics for a single snapshot."""
    from power_loop.runtime.execution_snapshot import load_snapshot

    snapshot = await load_snapshot(manager, job_id, generation)

    if snapshot is None:
        return SnapshotMetrics(
            job_id=job_id,
            generation=generation,
            snapshot_exists=False,
            snapshot_valid=False,
            snapshot_size_bytes=0,
            pending_tool_calls_count=0,
            round_index=0,
            total_tokens=0,
            captured_at_ms=0,
            source="none"
        )

    snapshot_json = json.dumps(snapshot.to_dict())
    size_bytes = len(snapshot_json.encode('utf-8'))

    return SnapshotMetrics(
        job_id=job_id,
        generation=generation,
        snapshot_exists=True,
        snapshot_valid=snapshot.is_valid(),
        snapshot_size_bytes=size_bytes,
        pending_tool_calls_count=len(snapshot.pending_tool_calls),
        round_index=snapshot.round_index,
        total_tokens=snapshot.total_tokens,
        captured_at_ms=snapshot.captured_at_ms,
        source="snapshot"
    )


async def check_snapshot_availability(manager, job_ids: List[str]) -> Dict[str, Any]:
    """Check snapshot availability across multiple jobs.

    Returns:
        {
            "total_jobs": int,
            "snapshots_available": int,
            "snapshots_valid": int,
            "availability_rate": float,
            "validity_rate": float,
            "avg_size_bytes": float
        }
    """
    metrics_list = []

    for job_id in job_ids:
        # Assume generation=1 for most jobs
        metrics = await collect_snapshot_metrics(manager, job_id, 1)
        metrics_list.append(metrics)

    available = sum(1 for m in metrics_list if m.snapshot_exists)
    valid = sum(1 for m in metrics_list if m.snapshot_valid)
    sizes = [m.snapshot_size_bytes for m in metrics_list if m.snapshot_exists]

    return {
        "total_jobs": len(job_ids),
        "snapshots_available": available,
        "snapshots_valid": valid,
        "availability_rate": available / len(job_ids) if job_ids else 0,
        "validity_rate": valid / available if available else 0,
        "avg_size_bytes": sum(sizes) / len(sizes) if sizes else 0,
        "min_size_bytes": min(sizes) if sizes else 0,
        "max_size_bytes": max(sizes) if sizes else 0
    }


async def analyze_snapshot_size_distribution(manager, job_ids: List[str]) -> Dict[str, Any]:
    """Analyze snapshot size distribution.

    Returns:
        {
            "buckets": {
                "0-1KB": int,
                "1-5KB": int,
                "5-10KB": int,
                "10-50KB": int,
                "50KB+": int
            },
            "avg_size": float,
            "median_size": float,
            "p95_size": float
        }
    """
    sizes = []

    for job_id in job_ids:
        metrics = await collect_snapshot_metrics(manager, job_id, 1)
        if metrics.snapshot_exists:
            sizes.append(metrics.snapshot_size_bytes)

    if not sizes:
        return {"error": "No snapshots found"}

    sizes.sort()

    buckets = {
        "0-1KB": sum(1 for s in sizes if s < 1024),
        "1-5KB": sum(1 for s in sizes if 1024 <= s < 5120),
        "5-10KB": sum(1 for s in sizes if 5120 <= s < 10240),
        "10-50KB": sum(1 for s in sizes if 10240 <= s < 51200),
        "50KB+": sum(1 for s in sizes if s >= 51200)
    }

    median_idx = len(sizes) // 2
    p95_idx = int(len(sizes) * 0.95)

    return {
        "buckets": buckets,
        "avg_size": sum(sizes) / len(sizes),
        "median_size": sizes[median_idx],
        "p95_size": sizes[p95_idx] if p95_idx < len(sizes) else sizes[-1],
        "count": len(sizes)
    }


async def compare_snapshot_vs_journal_health(manager, actor, job_id: str, generation: int) -> Dict[str, Any]:
    """Compare snapshot and journal health for a job.

    Returns:
        {
            "snapshot_available": bool,
            "journal_available": bool,
            "consistent": bool,
            "differences": List[str]
        }
    """
    from power_loop.runtime.execution_snapshot_inspect import compare_snapshot_with_journal

    comparison = await compare_snapshot_with_journal(manager, actor, job_id, generation)

    return {
        "snapshot_available": comparison["snapshot_view"] is not None,
        "journal_available": comparison["journal_view"] is not None,
        "consistent": comparison["match"],
        "differences": comparison["differences"]
    }


def format_metrics_report(metrics: SnapshotMetrics) -> str:
    """Format metrics for human-readable report."""
    report = f"""
Snapshot Metrics Report
=======================
Job ID:           {metrics.job_id}
Generation:       {metrics.generation}
Snapshot Exists:  {metrics.snapshot_exists}
Snapshot Valid:   {metrics.snapshot_valid}
Size:             {metrics.snapshot_size_bytes:,} bytes ({metrics.snapshot_size_bytes/1024:.2f} KB)
Pending Tools:    {metrics.pending_tool_calls_count}
Round Index:      {metrics.round_index}
Total Tokens:     {metrics.total_tokens:,}
Captured At:      {metrics.captured_at_ms}
Source:           {metrics.source}
"""
    return report


def format_availability_report(availability: Dict[str, Any]) -> str:
    """Format availability metrics for human-readable report."""
    report = f"""
Snapshot Availability Report
============================
Total Jobs:           {availability['total_jobs']}
Snapshots Available:  {availability['snapshots_available']} ({availability['availability_rate']*100:.1f}%)
Snapshots Valid:      {availability['snapshots_valid']} ({availability['validity_rate']*100:.1f}%)
Average Size:         {availability['avg_size_bytes']:.0f} bytes ({availability['avg_size_bytes']/1024:.2f} KB)
Size Range:           {availability['min_size_bytes']:.0f} - {availability['max_size_bytes']:.0f} bytes
"""
    return report
