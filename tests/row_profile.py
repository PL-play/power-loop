"""Row and byte measurements across supported SQL backends."""
async def _snapshot(db, prefix, *, largest=None):
    d = db.dialect.name
    if d == "sqlite":
        tables = [r["name"] for r in await db.fetchall("SELECT name FROM sqlite_master WHERE type='table'")]
    elif d == "postgres" or d == "postgresql":
        tables = [r["table_name"] for r in await db.fetchall(
            "SELECT table_name FROM information_schema.tables WHERE table_schema=current_schema() AND table_type='BASE TABLE'")]
    else:
        tables = [r["TABLE_NAME"] if "TABLE_NAME" in r else r["table_name"] for r in await db.fetchall(
            "SELECT table_name AS table_name FROM information_schema.tables WHERE table_schema=DATABASE() AND table_type='BASE TABLE'")]
    tables = [t for t in tables if t.startswith(prefix) and t != prefix + "p0_meter"]
    out = {}
    for t in tables:
        if d == "sqlite":
            cols = [r["name"] for r in await db.fetchall(f"PRAGMA table_info({t})")]
            size = "+".join(f"COALESCE(LENGTH({c}),0)" for c in cols) or "0"
        elif d == "mysql":
            cols = [r["column_name"] for r in await db.fetchall(
                f"SELECT column_name AS column_name FROM information_schema.columns WHERE table_schema=DATABASE() AND table_name='{t}'")]
            size = "+".join(f"COALESCE(LENGTH(`{c}`),0)" for c in cols) or "0"
        else:
            cols = [r["column_name"] for r in await db.fetchall(
                f"SELECT column_name FROM information_schema.columns WHERE table_schema=current_schema() AND table_name='{t}'")]
            size = "+".join(f"COALESCE(OCTET_LENGTH(CAST(\"{c}\" AS TEXT)),0)" for c in cols) or "0"
        row = await db.fetchone(f"SELECT COUNT(*) AS n, COALESCE(SUM({size}),0) AS b, COALESCE(MAX({size}),0) AS m FROM {t}")
        out[t[len(prefix):]] = (int(row["n"]), int(row["b"]))
        if largest is not None:
            largest[t[len(prefix):]] = int(row['m'])
    return out
