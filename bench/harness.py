"""Benchmark harness.

Runs each workload across batch sizes and execution modes (eager vs
torch.compile when torch is available), collects latency samples, and emits a
structured JSON result file via the configured ResultStore.

Usage:
    python -m bench.harness --run-id run-0042 --worker-id w1
    python -m bench.harness --inject-regression mlp --slowdown 1.3   # demo
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
import uuid
from datetime import datetime, timezone

import numpy as np

from bench import workloads as wl
from pipeline.storage import LocalResultStore

WARMUP_ITERS = 5
TIMED_ITERS = 30


def _percentile(samples: list[float], pct: float) -> float:
    return float(np.percentile(np.asarray(samples), pct))


def _time_callable(fn, iters: int = TIMED_ITERS, warmup: int = WARMUP_ITERS) -> list[float]:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000.0)  # ms
    return samples


def benchmark_workload(name: str, batch_size: int, mode: str, slowdown: float = 1.0) -> dict:
    """Benchmark one (workload, batch_size, mode) combination.

    `slowdown` simulates a performance regression for demos/tests by scaling
    recorded latencies. It is a test fixture, not a measurement feature.
    """
    if wl.TORCH_AVAILABLE:
        import torch

        model = wl.build_torch_model(name).eval()
        x = wl.torch_input(name, batch_size)
        if mode == "compiled":
            try:
                model = torch.compile(model)
            except Exception:
                mode = "eager-fallback"
        with torch.inference_mode():
            samples = _time_callable(lambda: model(x))
        backend = "torch"
    else:
        if mode == "compiled":
            return {}  # compiled mode only exists on the torch backend
        rng = np.random.default_rng(0)
        samples = _time_callable(lambda: wl.numpy_forward(name, batch_size, rng))
        backend = "numpy"

    samples = [s * slowdown for s in samples]
    p50 = _percentile(samples, 50)
    return {
        "workload": name,
        "mode": mode,
        "batch_size": batch_size,
        "backend": backend,
        "metrics": {
            "latency_ms_p50": p50,
            "latency_ms_p95": _percentile(samples, 95),
            "latency_ms_mean": statistics.fmean(samples),
            "throughput_sps": batch_size / (p50 / 1000.0) if p50 > 0 else 0.0,
        },
    }


def run_suite(run_id: str, worker_id: str, inject: str | None = None, slowdown: float = 1.0) -> dict:
    records = []
    modes = ["eager", "compiled"] if wl.TORCH_AVAILABLE else ["eager"]
    for name in wl.WORKLOADS:
        for batch_size in wl.BATCH_SIZES:
            for mode in modes:
                factor = slowdown if inject == name else 1.0
                rec = benchmark_workload(name, batch_size, mode, slowdown=factor)
                if rec:
                    records.append(rec)
    return {
        "schema_version": 1,
        "run_id": run_id,
        "git_sha": os.environ.get("GITHUB_SHA", "local"),
        "worker_id": worker_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the benchmark suite")
    parser.add_argument("--run-id", default=f"run-{uuid.uuid4().hex[:8]}")
    parser.add_argument("--worker-id", default=os.environ.get("WORKER_ID", "w0"))
    parser.add_argument("--results-dir", default="results/raw")
    parser.add_argument("--inject-regression", dest="inject", default=None,
                        help="Workload name to artificially slow down (demo only)")
    parser.add_argument("--slowdown", type=float, default=1.3)
    args = parser.parse_args()

    result = run_suite(args.run_id, args.worker_id, inject=args.inject, slowdown=args.slowdown)
    store = LocalResultStore(args.results_dir)
    path = store.write_result(result)
    print(f"[harness] wrote {len(result['records'])} records -> {path}")
    print(json.dumps({r["workload"] + "/" + r["mode"] + f"/bs{r['batch_size']}":
                      round(r["metrics"]["latency_ms_p50"], 3) for r in result["records"]}, indent=2))


if __name__ == "__main__":
    main()
