"""Real workflow leaf execution uses the current worker factory across a process boundary."""
import pytest

from power_loop import SessionStore

from .test_real_worker import isolated

pytestmark=[pytest.mark.real_llm,pytest.mark.asyncio]


async def test_leaf_runs_real_model_in_subprocess(tmp_path,record_property):
    path,saved=await isolated(tmp_path,'workflow')
    result=saved['result']
    assert result['status']=='completed',result
    store=await SessionStore.open(str(path),schema='verify')
    try:
        leaves=await store._db.fetchall(f"SELECT session_id FROM {store.table_prefix}jobs WHERE kind='workflow_node'")
        assert len(leaves)==1
        rows=await store.load_all_messages(leaves[0]['session_id'])
        assert any(row.role=='assistant' and 'paris' in str(row.content).lower() for row in rows)
        record_property('reported_tokens',result['usage'].get('total_tokens'))
    finally:
        await store.close()
