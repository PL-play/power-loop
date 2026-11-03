"""Tests for execution_snapshot module."""
import json
import pytest
from power_loop.runtime.execution_snapshot import ExecutionSnapshot, load_snapshot, update_snapshot, rebuild_from_events
from power_loop.runtime.jobs import JobError


def test_execution_snapshot_creation():
    """Test basic snapshot creation."""
    snapshot = ExecutionSnapshot(
        captured_at_ms=1699000000000,
        round_index=5,
        send_index=3,
        total_rounds=5,
        total_tokens=50000,
        generation=1
    )

    assert snapshot.snapshot_version == 1
    assert snapshot.snapshot_type == "intermediate"
    assert snapshot.round_index == 5
    assert snapshot.send_index == 3
    assert snapshot.is_valid()


def test_execution_snapshot_to_dict():
    """Test snapshot serialization to dict."""
    snapshot = ExecutionSnapshot(
        captured_at_ms=1699000000000,
        round_index=5,
        send_index=3,
        pending_tool_calls=[
            {"tool_call_id": "call_123", "tool_name": "web_search", "status": "running"}
        ]
    )

    data = snapshot.to_dict()
    assert isinstance(data, dict)
    assert data["round_index"] == 5
    assert data["send_index"] == 3
    assert len(data["pending_tool_calls"]) == 1


def test_execution_snapshot_from_dict():
    """Test snapshot deserialization from dict."""
    data = {
        "snapshot_version": 1,
        "snapshot_type": "intermediate",
        "captured_at_ms": 1699000000000,
        "round_index": 5,
        "send_index": 3,
        "total_rounds": 5,
        "total_tokens": 50000,
        "pending_tool_calls": [],
        "user_input_expected": False,
        "pause_reason": None,
        "budget_policy_key": "default",
        "budget_used_calls": 10,
        "budget_used_tokens": 50000,
        "budget_remaining_calls": 90,
        "budget_remaining_tokens": 450000,
        "budget_last_checkpoint_at": 1699000000000,
        "cancel_requested": False,
        "needs_reconciliation": False,
        "resource_state": "running",
        "executor_id": "exec_001",
        "boot_id": "boot_abc",
        "generation": 1,
        "checkpoint_reason": "periodic",
        "last_model_call_at": None
    }

    snapshot = ExecutionSnapshot.from_dict(data)
    assert snapshot.round_index == 5
    assert snapshot.send_index == 3
    assert snapshot.budget_used_calls == 10
    assert snapshot.is_valid()


def test_execution_snapshot_from_nested_dict():
    """Test snapshot deserialization from old nested format."""
    data = {
        "snapshot_version": 1,
        "snapshot_type": "intermediate",
        "captured_at_ms": 1699000000000,
        "execution": {
            "round_index": 5,
            "send_index": 3,
            "total_rounds": 5,
            "total_tokens": 50000
        },
        "pending": {
            "tool_calls": [{"tool_call_id": "call_123"}],
            "user_input_expected": False
        },
        "budget": {
            "policy_key": "default",
            "used_calls": 10,
            "used_tokens": 50000
        },
        "flags": {
            "cancel_requested": False,
            "needs_reconciliation": False,
            "resource_state": "running"
        },
        "metadata": {
            "executor_id": "exec_001",
            "generation": 1
        }
    }

    snapshot = ExecutionSnapshot.from_dict(data)
    assert snapshot.round_index == 5
    assert snapshot.send_index == 3
    assert snapshot.budget_used_calls == 10
    assert len(snapshot.pending_tool_calls) == 1


def test_execution_snapshot_validation():
    """Test snapshot validation."""
    # Valid snapshot
    valid = ExecutionSnapshot(
        captured_at_ms=1699000000000,
        generation=1
    )
    assert valid.is_valid()

    # Invalid: no capture time
    invalid1 = ExecutionSnapshot(
        captured_at_ms=0,
        generation=1
    )
    assert not invalid1.is_valid()

    # Invalid: bad generation
    invalid2 = ExecutionSnapshot(
        captured_at_ms=1699000000000,
        generation=0
    )
    assert not invalid2.is_valid()


def test_execution_snapshot_needs_repair():
    """Test snapshot repair detection."""
    # No repair needed
    snapshot1 = ExecutionSnapshot(
        captured_at_ms=1699000000000,
        generation=1
    )
    assert not snapshot1.needs_repair()

    # Needs reconciliation
    snapshot2 = ExecutionSnapshot(
        captured_at_ms=1699000000000,
        generation=1,
        needs_reconciliation=True
    )
    assert snapshot2.needs_repair()

    # Unknown resource state
    snapshot3 = ExecutionSnapshot(
        captured_at_ms=1699000000000,
        generation=1,
        resource_state="unknown"
    )
    assert snapshot3.needs_repair()


def test_execution_snapshot_invalid_type():
    """Test snapshot with invalid type."""
    with pytest.raises(JobError) as exc_info:
        ExecutionSnapshot(snapshot_type="invalid")
    assert exc_info.value.code == "invalid_snapshot_type"


def test_execution_snapshot_invalid_reason():
    """Test snapshot with invalid checkpoint reason."""
    with pytest.raises(JobError) as exc_info:
        ExecutionSnapshot(checkpoint_reason="invalid")
    assert exc_info.value.code == "invalid_checkpoint_reason"


def test_execution_snapshot_round_trip():
    """Test full serialization round trip."""
    original = ExecutionSnapshot(
        captured_at_ms=1699000000000,
        round_index=10,
        send_index=5,
        total_rounds=10,
        total_tokens=100000,
        pending_tool_calls=[
            {"tool_call_id": "call_1", "status": "running"},
            {"tool_call_id": "call_2", "status": "completed"}
        ],
        budget_used_calls=20,
        budget_used_tokens=100000,
        cancel_requested=False,
        generation=2
    )

    # Serialize
    data = original.to_dict()
    json_str = json.dumps(data)

    # Deserialize
    parsed_data = json.loads(json_str)
    restored = ExecutionSnapshot.from_dict(parsed_data)

    # Verify
    assert restored.round_index == original.round_index
    assert restored.send_index == original.send_index
    assert restored.total_tokens == original.total_tokens
    assert len(restored.pending_tool_calls) == len(original.pending_tool_calls)
    assert restored.budget_used_calls == original.budget_used_calls
    assert restored.generation == original.generation
    assert restored.is_valid()


# Mock fixtures
@pytest.fixture
def mock_manager():
    """Create a mock JobManager for testing."""
    class MockDB:
        def __init__(self):
            self._return_value = None

        async def fetchone(self, *args):
            return self._return_value

        async def fetchall(self, *args):
            return []

        def set_return(self, value):
            """Set return value for next fetchone call."""
            self._return_value = value

    class MockManager:
        def __init__(self):
            self.db = MockDB()
            self.jobs = "jobs"
            self.events = "job_events"

        def now(self):
            return 1699000000000

    return MockManager()


@pytest.mark.asyncio
async def test_load_snapshot_not_found(mock_manager):
    """Test loading snapshot when job doesn't exist."""
    mock_manager.db.set_return(None)

    snapshot = await load_snapshot(mock_manager, "job_123", 1)
    assert snapshot is None


@pytest.mark.asyncio
async def test_load_snapshot_no_result(mock_manager):
    """Test loading snapshot when result_json is NULL."""
    mock_manager.db.set_return({"result_json": None})

    snapshot = await load_snapshot(mock_manager, "job_123", 1)
    assert snapshot is None


@pytest.mark.asyncio
async def test_load_snapshot_invalid_json(mock_manager):
    """Test loading snapshot with invalid JSON."""
    mock_manager.db.set_return({"result_json": "invalid json{"})

    snapshot = await load_snapshot(mock_manager, "job_123", 1)
    assert snapshot is None


@pytest.mark.asyncio
async def test_load_snapshot_success(mock_manager):
    """Test successful snapshot loading."""
    snapshot_data = ExecutionSnapshot(
        captured_at_ms=1699000000000,
        round_index=5,
        generation=1
    ).to_dict()

    mock_manager.db.set_return({"result_json": json.dumps(snapshot_data)})

    snapshot = await load_snapshot(mock_manager, "job_123", 1)
    assert snapshot is not None
    assert snapshot.round_index == 5
    assert snapshot.is_valid()
