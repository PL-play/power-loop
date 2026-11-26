"""Controls for DB notification labels and transaction membership in the test profiler."""

import asyncio
from types import SimpleNamespace

import pytest

from power_loop.agent.runtime_store import RuntimeInputStore
from power_loop.agent.runtime_transport import SubscriptionSpec
from power_loop.agent.runtime_types import InputRef

from .session_native_fixture import ACTOR, DESCRIPTOR
from .test_composition_sql_profile_215 import ReadWork, freeze_request
from .test_projection_stream_sql_216 import StreamModel, _reset
from .test_send_sql_profile import _CATEGORIES, Profiled, _category, _counts

pytestmark = [pytest.mark.unit]

POLL = "agent.runtime_store:poll_notifications"
DISCOVER = "agent.runtime_store:discover_unfinished"
OUTPUT_PAGE = (
    "SELECT id,session_id,item_id FROM r13_follow_up_queue WHERE terminal_status IS NOT NULL "
    "ORDER BY completed_at,id LIMIT ?"
)
OUTPUT_POINT = (
    "SELECT id,session_id,item_id FROM r13_follow_up_queue WHERE session_id=? AND item_id=? "
    "AND terminal_status IS NOT NULL"
)
DISCOVERY = "SELECT session_id FROM (SELECT session_id FROM r13_sessions LIMIT ?) AS candidates"


@pytest.mark.parametrize(("chain", "sql", "expected"), [
    ([POLL, "agent.runtime_transport:_poll"], OUTPUT_PAGE, "periodic"),
    ([POLL, "agent.runtime_transport:_poll"], OUTPUT_POINT, "periodic"),
    ([DISCOVER, POLL], DISCOVERY, "periodic"),
    ([DISCOVER, "agent.runtime_worker:recover"], DISCOVERY, "session_control"),
    (["agent.runtime_store:_result_state"], OUTPUT_POINT, "session_control"),
    ([POLL], "UPDATE r13_follow_up_queue SET status=? WHERE item_id=?", "session_control"),
    ([POLL], "SELECT q.id FROM r13_follow_up_queue q JOIN r13_jobs j ON q.id=j.id", "legacy_job_control"),
    ([POLL], "SELECT claim_token FROM r13_follow_up_queue", "legacy_job_control"),
    (["runtime.jobs:ready", POLL], OUTPUT_POINT, "legacy_job_control"),
    (["runtime.execution_budget:charge", POLL], OUTPUT_POINT, "billing"),
])
def test_notification_classification_does_not_hide_other_sql(chain, sql, expected):
    assert _category(chain, sql) == expected


def test_unknown_sql_through_notification_entry_still_fails_classification():
    with pytest.raises(AssertionError, match="unclassified SQL"):
        _category([DISCOVER, POLL], "SELECT id FROM unexpected_table")


@pytest.mark.parametrize(("chain", "sql", "expected"), [
    (["runtime.store.process_guard:acquire"], "SELECT pg_try_advisory_lock($1)", "session_control"),
    (["runtime.store.process_guard:acquire"], "SELECT GET_LOCK(%s,0)", "session_control"),
    (["runtime.store.process_guard:release"], "SELECT RELEASE_LOCK(%s)", "session_control"),
    (["runtime.store.process_guard:check"], "SELECT 1", "session_control"),
    (["runtime.store.process_guard:check", "agent.runtime_worker:_watch_guard"], "SELECT 1", "periodic"),
    (["runtime.store.process_guard:check", "runtime.execution_budget:charge"], "SELECT 1", "billing"),
    (["runtime.store.process_guard:check", "runtime.jobs:ready"], "SELECT 1", "legacy_job_control"),
])
def test_process_guard_classifies_real_lock_and_check_sql(chain, sql, expected):
    assert _category(chain, sql) == expected


@pytest.mark.parametrize(("chain", "sql"), [
    ([], "SELECT 1"),
    (["agent.runtime_worker:_watch_guard"], "SELECT 1"),
    (["runtime.store.process_guard:acquire"], "SELECT 1"),
    (["runtime.store.process_guard:acquire"], "SELECT something_else()"),
    (["runtime.store.process_guard:release"], "SELECT 1"),
    (["runtime.store.process_guard:check"], "SELECT 2"),
    ([], "SELECT GET_LOCK(%s,0)"),
    ([], "SELECT RELEASE_LOCK(%s)"),
])
def test_process_guard_does_not_claim_unmatched_sql(chain, sql):
    with pytest.raises(AssertionError, match="unclassified SQL"):
        _category(chain, sql)


async def _paused_message_snapshot(profile, store, target, entered, release):
    async with profile.read_snapshot() as reader:
        await reader.fetchone(
            f"SELECT role FROM {store.t.messages} WHERE session_id=? LIMIT 1", (target.session_id,),
        )
        entered.set()
        await release.wait()


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["accept_poll", "snapshot_poll", "snapshot_accept"])
async def test_transaction_classifies_only_its_own_statements(registry_store, monkeypatch, case):
    from contextlib import asynccontextmanager

    store, _ = registry_store
    raw = store._db
    port = RuntimeInputStore(store)
    target = await port.create_session(DESCRIPTOR, actor=ACTOR)
    profile = Profiled(raw)
    entered, release = asyncio.Event(), asyncio.Event()
    exit_records = []
    original_tx = profile.transaction

    def attributed(records, site):
        exit_records.append((list(records), site))
        return Profiled._transaction_category(records, site)

    @asynccontextmanager
    async def held_tx():
        async with original_tx() as tx:
            yield tx
            entered.set()
            await release.wait()

    monkeypatch.setattr(profile, "_transaction_category", attributed)
    monkeypatch.setattr(store, "_db", profile)
    if case == "accept_poll":
        monkeypatch.setattr(profile, "transaction", held_tx)
        outer = asyncio.create_task(port.accept(target, "first", message_id="one", actor=ACTOR))
    else:
        outer = asyncio.create_task(_paused_message_snapshot(profile, store, target, entered, release))
    try:
        async with asyncio.timeout(10):
            await entered.wait()
            if raw.dialect.name == "sqlite":
                # This store serializes access on a single connection.
                release.set()
                await outer
            if case == "snapshot_accept":
                await port.accept(target, "second", message_id="two", actor=ACTOR)
            else:
                await port.poll_notifications(SubscriptionSpec(
                    "probe", store.store_id, "output", "observer", target.session_id,
                    discovery_message_id="watch",
                ))
            if raw.dialect.name != "sqlite":
                assert not outer.done()
                assert len(exit_records) == int(case == "snapshot_accept")
            release.set()
            await outer
    finally:
        release.set()
        await asyncio.gather(outer, return_exceptions=True)

    accept_sql = 2
    expected = {
        "accept_poll": (accept_sql + 1, [list(range(accept_sql))], ["session_control"]),
        "snapshot_poll": (2, [[0]], ["message"]),
        "snapshot_accept": (1 + accept_sql, [[0], list(range(1, 1 + accept_sql))], ["message", "session_control"]),
    }[case]
    sql, memberships, categories = expected
    assert len(profile.records) == sql
    assert profile.transactions == len(memberships) == len(profile.tx_sites)
    assert profile.tx_categories == categories
    indices = {id(record): index for index, record in enumerate(profile.records)}
    entry_sites = {id(site): index for index, site in enumerate(profile.tx_sites)}
    actual = {entry_sites[id(site)]: [indices[id(record)] for record in records]
              for records, site in exit_records}
    assert [actual[i] for i in range(profile.transactions)] == memberships
    exit_order = [entry_sites[id(site)] for _, site in exit_records]
    assert exit_order == ([1, 0] if case == "snapshot_accept" and raw.dialect.name != "sqlite"
                          else list(range(profile.transactions)))
    expected_sites = [["agent.runtime_store:accept"] if cat == "session_control" else [] for cat in categories]
    assert [site[:1] for site in profile.tx_sites] == expected_sites
    assert sum(POLL in record["chain"] for record in profile.records) == int(case != "snapshot_accept")


@pytest.mark.asyncio
async def test_failed_snapshot_and_empty_transaction_keep_window_accounting(registry_store):
    store, _ = registry_store
    profile = Profiled(store._db)
    async with profile.transaction():
        pass
    assert profile.tx_categories == [_category(profile.tx_sites[0])]
    profile.records.clear()
    profile.tx_sites.clear()  # Existing tx_sites-only callers start a new measurement window this way.
    with pytest.raises(RuntimeError, match="leave snapshot"):
        async with profile.read_snapshot() as reader:
            await reader.fetchone(f"SELECT id FROM {store.t.messages} LIMIT 1")
            raise RuntimeError("leave snapshot")
    assert len(profile.records) == len(profile.tx_sites) == 1
    assert profile.tx_categories == ["message"]
    assert profile.transactions == 2
    assert _counts(profile.records, profile.tx_sites)["legacy_job_control_tx"] == 1
    assert _counts(profile.records, profile.tx_sites, profile.tx_categories)["message_tx"] == 1
    assert _counts([], profile.tx_sites, [])["legacy_job_control_tx"] == 0


async def _stored_inputs(raw, port, target):
    return await raw.fetchall(
        f"SELECT item_id,status FROM {port.q} WHERE session_id=? ORDER BY item_id", (target.session_id,),
    )


@pytest.mark.asyncio
async def test_clear_inflight_accept_returns_the_committed_receipt(registry_store, monkeypatch):
    from contextlib import asynccontextmanager

    store, _ = registry_store
    raw = store._db
    port = RuntimeInputStore(store)
    target = await port.create_session(DESCRIPTOR, actor=ACTOR)
    profile = Profiled(ReadWork(raw))
    entered, release = asyncio.Event(), asyncio.Event()
    original_tx = profile.transaction

    @asynccontextmanager
    async def held_tx():
        async with original_tx() as tx:
            yield tx
            entered.set()
            await release.wait()

    monkeypatch.setattr(store, "_db", profile)
    monkeypatch.setattr(profile, "transaction", held_tx)
    accept_sql = 2
    task = asyncio.create_task(port.accept(target, "A", message_id="A", actor=ACTOR))
    try:
        async with asyncio.timeout(10):
            await entered.wait()
            assert len(profile.records) == accept_sql and profile.tx_categories == [None]
            _reset(profile, StreamModel(text="test"))
            release.set()
            receipt = await task
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)

    assert receipt.input_ref == InputRef(store.store_id, target.session_id, "A")
    assert receipt.target == target and receipt.message_id == "A" and receipt.newly_accepted
    assert await _stored_inputs(raw, port, target) == [{"item_id": "A", "status": "pending"}]
    assert profile.transactions == 0
    assert not profile.records and not profile.tx_sites and not profile.tx_categories


@pytest.mark.asyncio
@pytest.mark.parametrize("replacement", ["accept", "same_site"])
async def test_old_snapshot_does_not_write_the_new_window(registry_store, monkeypatch, replacement):
    store, _ = registry_store
    raw = store._db
    port = RuntimeInputStore(store)
    target = await port.create_session(DESCRIPTOR, actor=ACTOR)
    profile = Profiled(ReadWork(raw))
    entered, release = asyncio.Event(), asyncio.Event()
    monkeypatch.setattr(store, "_db", profile)
    old = asyncio.create_task(_paused_message_snapshot(profile, store, target, entered, release))
    try:
        async with asyncio.timeout(10):
            await entered.wait()
            [old_site] = profile.tx_sites
            _reset(profile, StreamModel(text="test"))
            if raw.dialect.name == "sqlite":
                # A second SQL operation on this connection must wait until the snapshot exits.
                release.set()
                await old
                assert profile.transactions == 0 and not profile.tx_categories
            if replacement == "accept":
                receipt = await port.accept(target, "B", message_id="B", actor=ACTOR)
                assert receipt.input_ref == InputRef(store.store_id, target.session_id, "B")
            else:
                async with profile.read_snapshot() as reader:
                    row = await reader.fetchone(
                        f"SELECT execution_state FROM {store.t.sessions} WHERE session_id=?",
                        (target.session_id,),
                    )
                    assert row["execution_state"] == "idle"
                [new_site] = profile.tx_sites
                assert new_site == old_site and new_site is not old_site
            assert profile.tx_categories == ["session_control"]
            if raw.dialect.name != "sqlite":
                assert not old.done()
            release.set()
            await old
    finally:
        release.set()
        await asyncio.gather(old, return_exceptions=True)

    accept_sql = 2
    assert len(profile.records) == (accept_sql if replacement == "accept" else 1)
    assert profile.transactions == len(profile.tx_sites) == 1
    assert profile.tx_categories == ["session_control"]
    if replacement == "accept":
        assert await _stored_inputs(raw, port, target) == [{"item_id": "B", "status": "pending"}]


@pytest.mark.asyncio
async def test_freeze_counts_open_transactions_without_rewriting_the_snapshot(registry_store, monkeypatch):
    store, _ = registry_store
    raw = store._db
    port = RuntimeInputStore(store)
    target = await port.create_session(DESCRIPTOR, actor=ACTOR)
    profile = Profiled(ReadWork(raw))
    entered, release = asyncio.Event(), asyncio.Event()
    monkeypatch.setattr(store, "_db", profile)
    old = asyncio.create_task(_paused_message_snapshot(profile, store, target, entered, release))
    pooled = raw.dialect.name != "sqlite"
    try:
        async with asyncio.timeout(10):
            await entered.wait()
            if pooled:
                await port.accept(target, "B", message_id="B", actor=ACTOR)
            assert not old.done()
            sample = freeze_request(SimpleNamespace(_composition_notifications={}), profile)
            sample_labels = [None, "session_control"] if pooled else [None]
            assert sample["tx_categories"] == sample_labels
            counts = _counts(sample["records"], sample["tx_sites"], sample["tx_categories"])
            assert counts["session_control_tx"] == int(pooled)
            assert counts["unfinished_transactions"] == 1
            finished = sum(counts[k + "_tx"] for k in _CATEGORIES)
            assert finished + counts["unfinished_transactions"] == len(sample["tx_sites"]) == sample["transactions"]
            release.set()
            await old
            if not pooled:
                await port.accept(target, "B", message_id="B", actor=ACTOR)
    finally:
        release.set()
        await asyncio.gather(old, return_exceptions=True)

    assert profile.tx_categories == ["message", "session_control"]
    assert sample["tx_categories"] == sample_labels
    assert _counts(sample["records"], sample["tx_sites"], sample["tx_categories"]) == counts
    complete = _counts(profile.records, profile.tx_sites, profile.tx_categories)
    assert "unfinished_transactions" not in complete
    assert complete["message_tx"] == complete["session_control_tx"] == 1
    assert sum(complete[k + "_tx"] for k in _CATEGORIES) == profile.transactions == 2
