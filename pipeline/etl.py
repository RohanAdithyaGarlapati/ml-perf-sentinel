"""ETL: raw benchmark JSON -> DuckDB warehouse.

Validates each raw result document, flattens nested metrics into long-format
rows, and idempotently loads them into a `benchmarks` table (re-running the
ETL never duplicates rows, so distributed workers and retries are safe).
"""
from __future__ import annotations

import argparse
import logging

import duckdb

from pipeline.storage import LocalResultStore, ResultStore

log = logging.getLogger("etl")

SCHEMA = """
CREATE TABLE IF NOT EXISTS benchmarks (
    run_id     VARCHAR NOT NULL,
    ts         TIMESTAMP NOT NULL,
    git_sha    VARCHAR,
    worker_id  VARCHAR NOT NULL,
    backend    VARCHAR,
    workload   VARCHAR NOT NULL,
    mode       VARCHAR NOT NULL,
    batch_size INTEGER NOT NULL,
    metric     VARCHAR NOT NULL,
    value      DOUBLE NOT NULL
);
"""

REQUIRED_TOP = {"run_id", "worker_id", "timestamp", "records"}
REQUIRED_RECORD = {"workload", "mode", "batch_size", "metrics"}


def validate(doc: dict) -> list[str]:
    """Return a list of validation errors (empty list means valid)."""
    errors = [f"missing field: {f}" for f in REQUIRED_TOP - doc.keys()]
    for i, rec in enumerate(doc.get("records", [])):
        errors += [f"record {i}: missing {f}" for f in REQUIRED_RECORD - rec.keys()]
        for metric, value in rec.get("metrics", {}).items():
            if not isinstance(value, (int, float)) or value < 0:
                errors.append(f"record {i}: bad value for {metric}: {value!r}")
    return errors


def flatten(doc: dict) -> list[tuple]:
    rows = []
    for rec in doc["records"]:
        for metric, value in rec["metrics"].items():
            rows.append((
                doc["run_id"], doc["timestamp"], doc.get("git_sha"),
                doc["worker_id"], rec.get("backend"), rec["workload"],
                rec["mode"], rec["batch_size"], metric, float(value),
            ))
    return rows


def load(store: ResultStore, db_path: str = "results/perf.duckdb") -> dict:
    """Ingest every raw result document into DuckDB. Returns load stats."""
    con = duckdb.connect(db_path)
    con.execute(SCHEMA)
    stats = {"files": 0, "rows_seen": 0, "rows_inserted": 0, "files_rejected": 0}

    for key in store.list_results():
        doc = store.read_result(key)
        errors = validate(doc)
        if errors:
            stats["files_rejected"] += 1
            log.warning("rejected %s: %s", key, "; ".join(errors[:3]))
            continue
        rows = flatten(doc)
        stats["files"] += 1
        stats["rows_seen"] += len(rows)

        con.executemany(
            """
            INSERT INTO benchmarks
            SELECT ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            WHERE NOT EXISTS (
                SELECT 1 FROM benchmarks b
                WHERE b.run_id = ? AND b.worker_id = ? AND b.workload = ?
                  AND b.mode = ? AND b.batch_size = ? AND b.metric = ?
            )
            """,
            [r + (r[0], r[3], r[5], r[6], r[7], r[8]) for r in rows],
        )

    stats["rows_inserted"] = con.execute("SELECT count(*) FROM benchmarks").fetchone()[0]
    con.close()
    return stats


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
    parser = argparse.ArgumentParser(description="Load raw results into DuckDB")
    parser.add_argument("--results-dir", default="results/raw")
    parser.add_argument("--db", default="results/perf.duckdb")
    args = parser.parse_args()

    stats = load(LocalResultStore(args.results_dir), args.db)
    print(f"[etl] {stats['files']} files ok, {stats['files_rejected']} rejected, "
          f"{stats['rows_seen']} rows seen, table now holds {stats['rows_inserted']} rows")


if __name__ == "__main__":
    main()
