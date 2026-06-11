"""Alerting and owner routing.

Findings from the detector are turned into structured, Slack-webhook-shaped
alert payloads and routed to owners via fnmatch patterns in owners.yaml
(CODEOWNERS-style). Payloads are written to alerts/ — pointing them at a real
Slack webhook is a one-line change (POST the payload to the webhook URL).
"""
from __future__ import annotations

import fnmatch
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml


def load_routes(path: str = "owners.yaml") -> dict:
    return yaml.safe_load(Path(path).read_text())


def route_owner(workload: str, routes: dict) -> str:
    for rule in routes.get("routes", []):
        if fnmatch.fnmatch(workload, rule["pattern"]):
            return rule["owner"]
    return routes.get("default_owner", "@perf-infra-oncall")


def build_alert(finding: dict, changepoint: dict | None, owner: str) -> dict:
    cp_line = ""
    if changepoint and changepoint.get("run_id"):
        cp_line = (f"\n• Likely introduced in *{changepoint['run_id']}* "
                   f"(+{changepoint['shift_pct']}% mean shift)")
    return {
        "channel": "#perf-regressions",
        "owner": owner,
        "severity": "high" if finding.get("mw_confirms") else "medium",
        "text": (
            f":rotating_light: *Performance regression detected*\n"
            f"• Workload: *{finding['workload']}* ({finding['mode']}, "
            f"bs={finding['batch_size']})\n"
            f"• Metric: {finding['metric']} = {finding['latest_value']} "
            f"({finding['pct_change']:+.1f}% vs baseline, z={finding['z']:.1f})\n"
            f"• Mann-Whitney confirms: {finding.get('mw_confirms')}"
            f"{cp_line}\n"
            f"• Routed to: {owner}"
        ),
        "finding": finding,
        "changepoint": changepoint,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def emit_alerts(findings: list[dict], changepoints: dict[str, dict],
                routes_path: str = "owners.yaml", out_dir: str = "alerts") -> list[str]:
    routes = load_routes(routes_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for f in findings:
        key = f"{f['workload']}-{f['mode']}-bs{f['batch_size']}"
        alert = build_alert(f, changepoints.get(key), route_owner(f["workload"], routes))
        path = out / f"alert-{key}.json"
        path.write_text(json.dumps(alert, indent=2))
        written.append(str(path))
    return written
