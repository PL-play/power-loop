"""Compare final SQL/row measurements with fixed B.

Run: python -m tests.compare_design203_costs OUTPUT_DIRECTORY
"""

import hashlib
import json
import sys
from pathlib import Path

from tests.unit.test_send_sql_profile import _category, _counts

out = Path(sys.argv[1])
base = Path("tests/baselines/design203_b.json")
b = json.loads(base.read_text())


def read_lines(name):
    return [json.loads(s) for s in (out / name).read_text().splitlines()]


def flat(value, prefix=""):
    result = {}
    if isinstance(value, dict):
        for k, v in value.items():
            result.update(flat(v, prefix + "." + k if prefix else k))
    elif isinstance(value, (tuple, list)):
        for i, v in enumerate(value):
            result.update(flat(v, prefix + "." + str(i)))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        result[prefix] = value
    return result


def compare(old, new):
    a, c = flat(old), flat(new)
    return {
        key: {
            "B": a.get(key, 0),
            "N": c.get(key, 0),
            "delta": c.get(key, 0) - a.get(key, 0),
            "percent": round(100 * (c.get(key, 0) - a[key]) / a[key], 4) if a.get(key) else None,
        }
        for key in sorted(a.keys() | c.keys())
    }


def entry(keys, old, new):
    return {**keys, "B": old, "N": new, "comparison": compare(old, new)}


sql = []
for p in sorted((out / "sql").glob("*.json")):
    d = json.loads(p.read_text())
    keys = {"db": p.name.split("-")[1], **{k: d[k] for k in ("models", "budgeted", "paid_tools")}}
    old = next(x for x in b["sql"] if all(x[k] == v for k, v in keys.items()))["B"]
    new = {
        **_counts(d["records"], d["tx_sites"]),
        "total_sql": len(d["records"]),
        "transactions": d["transactions"],
        "billing_tx": sum(_category(s) == "billing" for s in d["tx_sites"]),
        "ordinary_tx": sum(_category(s) == "ordinary" for s in d["tx_sites"]),
    }
    sql.append(entry(keys, old, new))
rows = []
for d in read_lines("rows.jsonl"):
    keys = {k: d[k] for k in ("db", "R", "K", "budgeted")}
    old = next(x for x in b["rows"] if all(x[k] == v for k, v in keys.items()))["B"]
    new = {
        "net_rows": sum(v[0] for v in d["tables"].values()),
        "net_bytes": sum(v[1] for v in d["tables"].values()),
        "tables": d["tables"],
        "largest": d["largest"],
        "writes": d["writes"],
    }
    rows.append(entry(keys, old, new))
writes = []
for d in read_lines("sql-writes.jsonl"):
    keys = {k: d[k] for k in ("db", "models", "budgeted", "paid_tools", "history")}
    old = next(x for x in b["sql_writes"] if all(x[k] == v for k, v in keys.items()))["B"]
    writes.append(entry(keys, old, d["writes"]))
history = []
for d in read_lines("history.jsonl"):
    if d["models"] != 3:
        continue
    keys = {k: d[k] for k in ("db", "models", "budgeted", "history")}
    old = next(x for x in b["history"] if all(x[k] == v for k, v in keys.items()))
    history.append(
        entry(keys, {k: old[k] for k in ("sql", "transactions")}, {k: d[k] for k in ("sql", "transactions")})
    )
assert (len(sql), len(rows), len(writes), len(history)) == (54, 24, 66, 12)
result = {
    "sdk_start": "7dc39e1beb54434dc7263a16e7f74fcb516c1e20",
    "fixed_B": b["sdk_base"],
    "baseline_sha256": hashlib.sha256(base.read_bytes()).hexdigest(),
    "scope": "Full request including acceptance, execution, result query, owner/resource release and parent handling. No performance gates.",
    "sql": sql,
    "rows": rows,
    "sql_writes": writes,
    "history": history,
    "p3_flows": read_lines("flows.jsonl"),
}
(out / "cost-comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(
    json.dumps(
        {
            "sql": len(sql),
            "rows": len(rows),
            "writes": len(writes),
            "history": len(history),
            "flows": len(result["p3_flows"]),
        }
    )
)
for x in sql:
    if x["models"] == 1 and not x["paid_tools"]:
        print({k: x[k] for k in ("db", "budgeted")}, x["comparison"]["total_sql"], x["comparison"]["transactions"])
for x in rows:
    if (x["R"], x["K"]) in [(1, 0), (32, 4)]:
        print(
            {k: x[k] for k in ("db", "R", "K", "budgeted")},
            {
                k: x["comparison"][k]
                for k in (
                    "net_rows",
                    "net_bytes",
                    "writes.affected_rows",
                    "writes.write_bytes",
                    "writes.max_tx_rows",
                    "writes.max_tx_bytes",
                )
            },
        )
for x in result["p3_flows"]:
    print(
        {k: x[k] for k in ("db", "flow", "sql", "elapsed_s")},
        {
            "net_rows": sum(v[0] for v in x["tables"].values()),
            "net_bytes": sum(v[1] for v in x["tables"].values()),
            "writes": {k: x["writes"][k] for k in ("affected_rows", "write_bytes", "max_tx_rows")},
        },
    )
