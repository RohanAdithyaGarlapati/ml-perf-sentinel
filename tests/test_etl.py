import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.etl import flatten, load, validate
from pipeline.storage import LocalResultStore

DOC = {
    "schema_version": 1,
    "run_id": "run-0001",
    "git_sha": "abc123",
    "worker_id": "w0",
    "timestamp": "2026-06-10T12:00:00+00:00",
    "records": [
        {"workload": "mlp", "mode": "eager", "batch_size": 1, "backend": "torch",
         "metrics": {"latency_ms_p50": 2.5, "throughput_sps": 400.0}},
    ],
}


def test_validate_accepts_good_doc():
    assert validate(DOC) == []


def test_validate_rejects_missing_fields_and_bad_values():
    bad = {**DOC, "records": [{"workload": "mlp", "mode": "eager", "batch_size": 1,
                               "metrics": {"latency_ms_p50": -1}}]}
    errors = validate(bad)
    assert any("bad value" in e for e in errors)
    assert any("missing field" in e for e in validate({"records": []}))


def test_flatten_produces_long_format_rows():
    rows = flatten(DOC)
    assert len(rows) == 2  # one row per metric
    assert rows[0][0] == "run-0001" and rows[0][8] in {"latency_ms_p50", "throughput_sps"}


def test_load_is_idempotent(tmp_path):
    store = LocalResultStore(str(tmp_path / "raw"))
    store.write_result(DOC)
    db = str(tmp_path / "perf.duckdb")

    first = load(store, db)
    second = load(store, db)  # re-running must not duplicate rows

    assert first["rows_inserted"] == 2
    assert second["rows_inserted"] == 2


def test_rejected_files_are_skipped(tmp_path):
    store = LocalResultStore(str(tmp_path / "raw"))
    store.write_result(DOC)
    bad = {**DOC, "run_id": "run-0002", "records": [{"workload": "x"}]}
    store.write_result(bad)

    stats = load(store, str(tmp_path / "perf.duckdb"))
    assert stats["files"] == 1
    assert stats["files_rejected"] == 1
