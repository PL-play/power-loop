"""Freeze the P0 measurements; all stages report against this B, without numeric performance gates.

Usage: python -m tests.freeze_design203_baseline --source OUTPUT_DIR --out FILE
Input files are produced by the SQL/row profiles and P0 benchmark commands.
"""
import argparse
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path

from tests.unit.test_send_sql_profile import _category, _counts


def json_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def paired_summary(data, metrics):
    before = sorted((row for row in data['pairs'] if row['arm'] == 'B'), key=lambda row: row['pair'])
    repeat = sorted((row for row in data['pairs'] if row['arm'] == 'repeat'), key=lambda row: row['pair'])
    assert len(before) == len(repeat) == 3
    assert sum(row['completed'] for row in before) >= 100
    assert sum(row['completed'] for row in repeat) >= 100
    assert all(row['errors'] == 0 for row in before + repeat)
    baseline = {key: statistics.median(row[key] for row in before)
                for key in metrics if before[0].get(key) is not None}
    ratios = {key: statistics.median(repeat[i][key] / before[i][key] for i in range(3))
              for key in baseline if all(row[key] for row in before)}
    result = {**data, 'B_medians': baseline,
              'repeat_to_B_median_ratios': ratios, 'N': 'not implemented; repeat is B again'}
    return result


def freeze(source):
    sql = []
    for path in sorted((source / 'sql').glob('*.json')):
        data = json.loads(path.read_text())
        baseline = {**_counts(data['records'], data['tx_sites']),
                    'total_sql': len(data['records']), 'transactions': data['transactions'],
                    'billing_tx': sum(_category(site) == 'billing' for site in data['tx_sites']),
                    'ordinary_tx': sum(_category(site) == 'ordinary' for site in data['tx_sites'])}
        statements = Counter((row['sql'], tuple(row['chain'])) for row in data['records'])
        sql.append({'db': path.name.split('-')[1], 'models': data['models'],
                    'paid_tools': data['paid_tools'], 'budgeted': data['budgeted'],
                    'B': baseline, 'N': 'production execution unchanged',
                    'activity_updates': sum('sessions SET updated_at' in row['sql'] for row in data['records']),
                    'statement_counts': [{'sql': statement, 'chain': chain, 'count': count}
                                         for (statement, chain), count in statements.items()],
                    'transaction_sites': data['tx_sites'],
                    'seconds_observed_with_sql_recorder': data['elapsed_s']})
    rows = []
    writes = json_lines(source / 'writes-final.jsonl')
    for data in json_lines(source / 'rows.jsonl'):
        measured = next(row for row in writes if all(row[key] == data[key] for key in ('db', 'R', 'K', 'budgeted')))
        baseline = {'net_rows': sum(row[0] for row in data['tables'].values()),
                    'net_bytes': sum(row[1] for row in data['tables'].values()),
                    'tables': data['tables'], 'largest': data['largest'], 'writes': measured['writes']}
        rows.append({**{key: data[key] for key in ('db', 'R', 'K', 'budgeted')},
                     'B': baseline, 'sql_without_meter': data['sql'],
                     'transactions': data['transactions'], 'sql_with_meter': measured['sql'],
                     'seconds_with_meter': measured['elapsed_s'],
                     'seconds_without_meter': data['elapsed_s'],
                     'N': 'production execution unchanged'})
    sql_writes = [{**{key: value for key, value in row.items() if key != 'writes'},
                   'B': row['writes']}
                  for row in json_lines(source / 'sql-writes.jsonl')]
    metrics = ('p50_ms', 'p95_ms', 'throughput_rps', 'cpu_seconds', 'rss_peak', 'rss_increase',
               'rss_after_gc', 'control_p50_ms', 'control_p95_ms')
    timing = [paired_summary(json.loads(path.read_text()), metrics)
              for path in sorted((source / 'timing').glob('*.json'))]
    maintenance = []
    for path in sorted((source / 'maintenance').glob('*.json')):
        data = paired_summary(json.loads(path.read_text()), metrics)
        maintenance.append(data)
    history = [row for row in json_lines(source / 'history.jsonl') if row['models'] == 3]
    assert (len(sql), len(rows), len(sql_writes), len(timing), len(maintenance), len(history)) == (54, 24, 66, 45, 6, 12)
    hashes = {str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()
              for pattern in ('sql/*.json', '*.jsonl', 'timing/*.json', 'maintenance/*.json', 'environment.json')
              for path in sorted(source.glob(pattern))}
    return {'sdk_base': '83ae157498d540b053e89132d00181eaf6de80ec',
            'contract': 'Design203 P0; fixed B, actual costs and differences; no numeric performance gates',
            'zero_rule': 'zero billing/retired tables/final temporary residues remain zero',
            'table_rule': 'per-table figures attribute actual cost; structural zero and cleanup rules still apply',
            'sql': sql, 'rows': rows, 'sql_writes': sql_writes, 'timing': timing,
            'maintenance': maintenance, 'history': history, 'raw_input_sha256': hashes,
            'environment': json.loads((source / 'environment.json').read_text())}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(freeze(args.source), ensure_ascii=False, indent=2) + '\n')
