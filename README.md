![perf-ci](https://github.com/RohanAdithyaGarlapati/ml-perf-sentinel/actions/workflows/perf-ci.yml/badge.svg)
# ml-perf-sentinel

A miniature of the testing and performance-characterization infrastructure used to
validate ML accelerator software: an automated benchmark harness, a results ETL
pipeline, statistical regression detection with changepoint triage, owner-routed
alerting, a release-readiness dashboard, and a CI quality gate.

Built to mirror how teams shipping ML compilers/runtimes (e.g. for custom training
silicon) detect performance regressions before a release reaches customers.

```
                                ┌──────────────────────────────────────────────┐
  ┌───────────────┐  raw JSON   │                 ResultStore                  │
  │ bench harness │────────────▶│  LocalResultStore  /  S3ResultStore (boto3)  │
  │  (N workers,  │             └──────────────────────┬───────────────────────┘
  │ eager + torch.│                                    │ ETL (validate, flatten,
  │   compile)    │                                    ▼  idempotent load)
  └───────────────┘                          ┌──────────────────┐
                                             │ DuckDB warehouse │
                                             └────────┬─────────┘
                       ┌──────────────────────────────┼──────────────────────────┐
                       ▼                              ▼                          ▼
            ┌─────────────────────┐      ┌─────────────────────┐    ┌────────────────────┐
            │ regression detector │      │ changepoint triage  │    │ Streamlit dashboard│
            │ (z-score + Mann-    │─────▶│ ("which run broke   │    │ trends, gate, eager│
            │  Whitney U)         │      │   it?")             │    │ vs compiled speedup│
            └──────────┬──────────┘      └──────────┬──────────┘    └────────────────────┘
                       └──────────────┬─────────────┘
                                      ▼
                       ┌──────────────────────────────┐
                       │ owner-routed alerts (YAML    │
                       │ routing) + CI release gate   │
                       └──────────────────────────────┘
```

## Quickstart (2 minutes, no GPU/AWS needed)

```bash
pip install -r requirements.txt

# 1. Seed 30 runs of history with a 25% slowdown injected into `mlp`
python scripts/seed_history.py --inject mlp

# 2. ETL -> detect -> triage -> alert (exits 1, i.e. gate is RED)
python scripts/run_pipeline.py --fail-on-regression

# 3. Explore the dashboard
streamlit run dashboard/app.py

# 4. Run a real benchmark sweep on your machine (separate warehouse — real
#    measurements shouldn't share a baseline with synthetic demo data)
python scripts/run_pipeline.py --bench --results-dir results/real --db results/real.duckdb
```

With PyTorch installed (`pip install torch --index-url https://download.pytorch.org/whl/cpu`),
the harness benchmarks each workload in both **eager** and **`torch.compile`** modes, and
the dashboard reports per-workload compilation speedups. Without torch it falls back to a
NumPy backend so the full pipeline still runs on lightweight CI machines.

## What's inside

| Component | Path | What it does |
|---|---|---|
| Benchmark harness | `bench/` | Times workloads across batch sizes and execution modes; emits structured JSON per (run, worker) |
| Storage abstraction | `pipeline/storage.py` | `ResultStore` interface; local backend now, S3 backend included and ready for credentials |
| ETL | `pipeline/etl.py` | Validates raw documents, flattens to long format, idempotently loads DuckDB (safe under retries / distributed workers) |
| Regression detection | `detect/regression.py` | Rolling-baseline z-score with a percent-change noise floor; Mann-Whitney U confirmation (non-parametric) |
| Changepoint triage | `detect/changepoint.py` | Localizes the first offending run via a standardized mean-shift scan |
| Alerting | `detect/alerts.py` | Slack-shaped payloads routed to owners via fnmatch rules in `owners.yaml` |
| Dashboard | `dashboard/app.py` | Trend lines, flagged regressions, release-readiness gate, eager-vs-compiled speedup table |
| CI | `.github/workflows/perf-ci.yml` | 3-worker benchmark matrix → merged ETL → tests → regression gate that fails the build |

## Design decisions

- **Statistical detection over fixed thresholds.** A static "fail if >X ms" threshold
  either pages constantly or misses slow drift. A rolling baseline z-score adapts to each
  series' own noise; the percent-change floor prevents paging on statistically-significant-
  but-microscopic shifts; Mann-Whitney U adds a distribution-free confirmation on sustained
  regressions.
- **Changepoint localization for triage.** Detection answers "is it slower?"; triage needs
  "since when?". A mean-shift scan pins the first offending run so the alert lands with the
  likely culprit commit attached.
- **Idempotent ETL.** Results arrive from multiple workers and may be retried. Loads are
  deduplicated on (run, worker, series, metric), so re-running the pipeline is always safe.
- **DuckDB as the warehouse.** Zero-ops, columnar, real SQL. The schema and queries port
  directly to S3 + Athena/Glue at scale; `S3ResultStore` already implements the object-store
  side of that migration.
- **Median across workers per run.** Distributed runners have heterogeneous noise; the
  per-run median is robust to a single slow worker.

## Scaling path (local → AWS)

| Local (this repo) | Production equivalent |
|---|---|
| `LocalResultStore` (JSON files) | `S3ResultStore` (included) |
| DuckDB file | S3 + Athena, or Redshift |
| GitHub Actions matrix workers | Fleet of hardware-attached test runners (e.g. ECS tasks) |
| Alert payloads in `alerts/` | Slack webhook / SNS → ticket routing |
| `owners.yaml` | CODEOWNERS / PagerDuty service mapping |

## Tests

```bash
pytest -q
```

Covers detection math (flat series, step regressions, sustained regressions, noise floors,
insufficient history), changepoint localization accuracy, ETL validation, and load idempotency.
