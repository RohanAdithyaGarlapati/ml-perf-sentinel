"""Seed synthetic benchmark history.

Generates N historical runs (with realistic noise) through the same raw-JSON
path the real harness uses, so the full ETL -> detection -> dashboard flow is
exercised. With --inject, the last few runs of one workload are degraded to
demonstrate detection, triage, and alerting.

In production this history would accumulate naturally in object storage; the
seeder exists so the demo and CI have day-one data.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.workloads import BATCH_SIZES, WORKLOADS  # noqa: E402
from pipeline.storage import LocalResultStore       # noqa: E402

BASE_LATENCY = {  # ms, per (workload, mode) at batch_size=1
    ("mlp", "eager"): 2.4, ("mlp", "compiled"): 1.6,
    ("tiny_cnn", "eager"): 5.1, ("tiny_cnn", "compiled"): 3.8,
    ("mini_transformer", "eager"): 3.3, ("mini_transformer", "compiled"): 2.2,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic run history")
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--inject", default=None, help="workload to degrade")
    parser.add_argument("--inject-last", type=int, default=3)
    parser.add_argument("--slowdown", type=float, default=1.25)
    parser.add_argument("--results-dir", default="results/raw")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    store = LocalResultStore(args.results_dir)
    start = datetime.now(timezone.utc) - timedelta(hours=args.runs)

    for i in range(args.runs):
        run_id = f"run-{i + 1:04d}"
        ts = start + timedelta(hours=i)
        degraded = args.inject and i >= args.runs - args.inject_last
        for w in range(args.workers):
            records = []
            for name in WORKLOADS:
                for mode in ("eager", "compiled"):
                    for bs in BATCH_SIZES:
                        base = BASE_LATENCY[(name, mode)] * (1 + 0.55 * np.log2(max(bs, 1)))
                        lat = base * float(rng.normal(1.0, 0.03))
                        if degraded and name == args.inject:
                            lat *= args.slowdown
                        p50 = round(lat, 4)
                        records.append({
                            "workload": name, "mode": mode, "batch_size": bs,
                            "backend": "synthetic",
                            "metrics": {
                                "latency_ms_p50": p50,
                                "latency_ms_p95": round(lat * float(rng.uniform(1.15, 1.3)), 4),
                                "latency_ms_mean": round(lat * float(rng.uniform(1.0, 1.05)), 4),
                                "throughput_sps": round(bs / (p50 / 1000.0), 2),
                            },
                        })
            store.write_result({
                "schema_version": 1, "run_id": run_id, "git_sha": f"sha{i:04d}",
                "worker_id": f"w{w}", "timestamp": ts.isoformat(), "records": records,
            })

    print(f"[seed] wrote {args.runs} runs x {args.workers} workers to {args.results_dir}"
          + (f" (injected {args.slowdown}x slowdown on '{args.inject}', "
             f"last {args.inject_last} runs)" if args.inject else ""))


if __name__ == "__main__":
    main()
