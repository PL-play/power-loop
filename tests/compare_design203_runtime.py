"""Compare measured P1 costs with the immutable G0 B; never rewrites B."""
import argparse
import hashlib
import json
import statistics
from pathlib import Path

from tests.unit.test_send_sql_profile import _category, _counts


def change(before, after):
    return {'B': before, 'N': after, 'delta': after-before,
            'percent': (after/before-1)*100 if before else (0 if after == 0 else None)}


def median(rows, field):
    return statistics.median(r[field] for r in rows if r[field] is not None)


def compare(directory):
    baseline = Path(__file__).parent/'baselines/design203_b.json'
    base = json.loads(baseline.read_text())
    result = {'B_sha256': hashlib.sha256(baseline.read_bytes()).hexdigest(),
              'sql': [], 'rows': [], 'sql_writes': [], 'history': [], 'timing': []}
    for entry in base['sql']:
        path = directory/'sql'/f"send-{entry['db']}-m{entry['models']}-b{int(entry['budgeted'])}{'-paid' if entry['paid_tools'] else ''}.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        after = _counts(data['records'], data['tx_sites'])
        after.update(total_sql=len(data['records']), transactions=data['transactions'],
                     billing_tx=sum(_category(c) == 'billing' for c in data['tx_sites']),
                     ordinary_tx=sum(_category(c) == 'ordinary' for c in data['tx_sites']))
        result['sql'].append({**{k:entry[k] for k in ('db','models','budgeted','paid_tools')},
                              'metrics': {k:change(v,after[k]) for k,v in entry['B'].items()}})
    measured = [json.loads(line) for line in (directory/'rows.jsonl').read_text().splitlines()]
    for entry in base['rows']:
        data = next(r for r in measured if all(entry[k] == r[k] for k in ('db','R','K','budgeted')))
        after = {'net_rows': sum(v[0] for v in data['tables'].values()),
                 'net_bytes': sum(v[1] for v in data['tables'].values())}
        metrics = {k:change(entry['B'][k],v) for k,v in after.items()}
        if data['writes']:
            metrics.update({k:change(v,data['writes'][k]) for k,v in entry['B']['writes'].items() if k != 'tables'})
        tables = {}
        for table in set(entry['B']['tables'])|set(data['tables']):
            b = entry['B']['tables'].get(table,(0,0))
            n = data['tables'].get(table,(0,0))
            tables[table] = {'net_rows':change(b[0],n[0]),'net_bytes':change(b[1],n[1])}
        result['rows'].append({**{k:entry[k] for k in ('db','R','K','budgeted')}, 'metrics':metrics, 'tables':tables})
    measured = [json.loads(line) for line in (directory/'sql-writes.jsonl').read_text().splitlines()]
    for entry in base['sql_writes']:
        keys = ('db', 'models', 'budgeted', 'paid_tools', 'history')
        data = next(r for r in measured if all(entry[k] == r[k] for k in keys))
        result['sql_writes'].append({**{k:entry[k] for k in keys},
            'metrics': {k:change(v,data['writes'][k]) for k,v in entry['B'].items() if k != 'tables'}})
    measured = [json.loads(line) for line in (directory/'history.jsonl').read_text().splitlines()]
    for entry in base['history']:
        keys = ('db', 'models', 'budgeted', 'history')
        data = next(r for r in measured if all(entry[k] == r[k] for k in keys))
        before = {**entry['sql'], 'total_sql':entry['total_sql'], 'transactions':entry['transactions']}
        after = {**data['sql'], 'total_sql':data['total_sql'], 'transactions':data['transactions']}
        result['history'].append({**{k:entry[k] for k in keys},
                                  'metrics': {k:change(v,after[k]) for k,v in before.items()}})
    for path in sorted((directory/'timing').glob('*.json')):
        data = json.loads(path.read_text())
        entry = next(r for r in base['timing'] if all(r[k] == data[k] for k in ('db','scenario','concurrency')))
        before = [r for r in entry['pairs'] if r['arm'] == 'B']
        fields = ['p50_ms','p95_ms','throughput_rps','cpu_seconds','rss_before','rss_peak','rss_after_gc','rss_increase']
        if data['scenario'] == 'cancel':
            fields += ['control_p50_ms','control_p95_ms']
        result['timing'].append({**{k:data[k] for k in ('db','scenario','concurrency')},
            'completed':sum(r['completed'] for r in data['pairs']),
            'errors':sum(r['errors'] for r in data['pairs']),
            'backlog':sum(r['backlog'] for r in data['pairs']),
            'metrics':{k:change(median(before,k),median(data['pairs'],k)) for k in fields}})
    (directory/'cost-comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--directory',type=Path,required=True)
    compare(parser.parse_args().directory)
