"""Result storage backends.

The harness and ETL talk to a ResultStore interface rather than a filesystem
or cloud SDK directly. LocalResultStore is the default; S3ResultStore is a
drop-in replacement once AWS credentials and boto3 are available — no other
code changes required.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path


class ResultStore(ABC):
    """Interface for persisting and listing raw benchmark result documents."""

    @abstractmethod
    def write_result(self, result: dict) -> str: ...

    @abstractmethod
    def list_results(self) -> list[str]: ...

    @abstractmethod
    def read_result(self, key: str) -> dict: ...


class LocalResultStore(ResultStore):
    """Stores results as JSON files on the local filesystem."""

    def __init__(self, root: str = "results/raw") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _key(self, result: dict) -> str:
        return f"{result['run_id']}__{result['worker_id']}.json"

    def write_result(self, result: dict) -> str:
        path = self.root / self._key(result)
        path.write_text(json.dumps(result, indent=2))
        return str(path)

    def list_results(self) -> list[str]:
        return sorted(str(p) for p in self.root.glob("*.json"))

    def read_result(self, key: str) -> dict:
        return json.loads(Path(key).read_text())


class S3ResultStore(ResultStore):
    """S3-backed store. Requires `pip install boto3` and AWS credentials.

    Same key layout as LocalResultStore, so historical local results can be
    backfilled with a single `aws s3 sync`.
    """

    def __init__(self, bucket: str, prefix: str = "results/raw") -> None:
        import boto3  # imported lazily so the package is optional

        self.s3 = boto3.client("s3")
        self.bucket = bucket
        self.prefix = prefix.rstrip("/")

    def write_result(self, result: dict) -> str:
        key = f"{self.prefix}/{result['run_id']}__{result['worker_id']}.json"
        self.s3.put_object(Bucket=self.bucket, Key=key,
                           Body=json.dumps(result).encode("utf-8"),
                           ContentType="application/json")
        return f"s3://{self.bucket}/{key}"

    def list_results(self) -> list[str]:
        paginator = self.s3.get_paginator("list_objects_v2")
        keys: list[str] = []
        for page in paginator.paginate(Bucket=self.bucket, Prefix=self.prefix):
            keys.extend(obj["Key"] for obj in page.get("Contents", []))
        return sorted(keys)

    def read_result(self, key: str) -> dict:
        body = self.s3.get_object(Bucket=self.bucket, Key=key)["Body"].read()
        return json.loads(body)
