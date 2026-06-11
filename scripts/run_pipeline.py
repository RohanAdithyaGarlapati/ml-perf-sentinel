"""End-to-end pipeline orchestrator.

    bench (optional) -> ETL -> regression detection -> changepoint triage -> alerts

Exits non-zero with --fail-on-regression so it can act as a CI release gate.

Examples:
    python scripts/run_pipeline.py --bench                     # measure + analyze
    python scripts/run_pipeline.py --bench --inject mlp        # demo a regression
    python scripts/run_pipeline.py --fail-on-regression        # CI quality gate
"""
from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.harness import run_suite                   # noqa: E402
from detect.alerts import emit_alerts                 # noqa: E402
from detect.changepoint import find_changepoint       # noqa: E402
from detect.regression import detect_regressions, series_for  # noqa: E402
from pipeline.etl import load                         # noqa: E402
from pipeline.storage import LocalResultStore         # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the perf pipeline end to end")
    parser.add_argument("--bench", action="store_true", help="run benchmarks first")
    parser.add_argument("--inject", default=None, help="workload to slow down (demo)")
    parser.add_argument("--slowdown", type=float, default=1.3)
    parser.add_argument("--results-dir", default="results/raw")
    parser.add_argument("--db", default="results/perf.duckdb")
    parser.add_argument("--fail-on-regression", action="store_true")
    parser.add_argument("--store", choices=["local", "s3"], default="local")
    parser.add_argument("--bucket", default=None)
    args = parser.parse_args()

    if args.store == "s3":
        from pipeline.storage import S3ResultStore
        store = S3ResultStore(args.bucket, prefix="results/raw")
        print(f"[pipeline] using S3 store: s3://{args.bucket}/results/raw")
    else:
        store = LocalResultStore(args.results_dir)

    if args.bench:
        result = run_suite(f"run-{uuid.uuid4().hex[:8]}", "w0",
                           inject=args.inject, slowdown=args.slowdown)
        store.write_result(result)
        print(f"[bench] completed run {result['run_id']} "
              f"({len(result['records'])} records)")

    stats = load(store, args.db)
    print(f"[etl]   {stats['files']} files loaded, {stats['files_rejected']} rejected, "
          f"warehouse rows: {stats['rows_inserted']}")

    findings = detect_regressions(args.db)
    if not findings:
        print("[detect] no regressions — release gate: GREEN")
        return 0

    # Localize the first offending run for each finding (triage).
    import duckdb
    con = duckdb.connect(args.db, read_only=True)
    changepoints = {}
    for f in findings:
        rows = series_for(con, f["workload"], f["mode"], f["batch_size"], f["metric"])
        cp = find_changepoint([r[2] for r in rows])
        if cp.get("index") is not None:
            cp["run_id"] = rows[cp["index"]][0]
        changepoints[f"{f['workload']}-{f['mode']}-bs{f['batch_size']}"] = cp
    con.close()

    paths = emit_alerts(findings, changepoints)
    print(f"[detect] {len(findings)} regression(s) — release gate: RED")
    for f in findings:
        key = f"{f['workload']}-{f['mode']}-bs{f['batch_size']}"
        cp = changepoints.get(key, {})
        print(f"  - {key}: {f['pct_change']:+.1f}% (z={f['z']:.1f}), "
              f"likely introduced in {cp.get('run_id', '?')}")
    print(f"[alerts] wrote {len(paths)} alert payload(s) to alerts/")

    return 1 if args.fail_on_regression else 0


if __name__ == "__main__":
    raise SystemExit(main())
