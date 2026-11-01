"""Prometheus exporter: reads var/ric.db on every scrape and exposes /metrics.

Read-only (the database is opened with mode=ro): monitoring can never change the RIC's state.
Because both backends write the same KPI record, the same dashboard shows the simulator now
and the testbed from M07a, unchanged.

  uv run python -m monitoring.exporter --port 9108
  curl -s localhost:9108/metrics | grep ric_

Metrics (all gauges, current values; Prometheus keeps the history):
  ric_slice_*{slice}                     last tick: throughput, latency, load, PRB share and use, buffer
  ric_backend_clock_seconds              backend clock of the last tick
  ric_policy_in_force                    a1_policy id in force (0 = default objective)
  ric_policy_weight{slice}, ric_policy_min_thr_mbps{slice}, ric_policy_latency_bound_ms
  ric_served_model_info{version}         1, labelled with the served policy version
  ric_intents{state}                     number of intents per state
  ric_verifier_decisions{check}          intents per deciding verifier check ('ok' = accepted)
"""

import argparse
import json
import sqlite3
import time
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

from prometheus_client import start_http_server
from prometheus_client.core import REGISTRY, GaugeMetricFamily
from prometheus_client.registry import Collector

from store.db import db_path
from store.intents import STATES
from verifier.objective import DEFAULT_OBJECTIVE, SLICES

SLICE_FIELDS = {  # kpi column -> (metric name, help)
    "dl_thr_mbps": ("ric_slice_throughput_mbps", "Downlink throughput of the slice during the last tick"),
    "latency_ms": ("ric_slice_latency_ms", "Latency of the slice during the last tick"),
    "offered_load_mbps": ("ric_slice_offered_load_mbps", "Offered load of the slice during the last tick"),
    "prb_share_alloc": ("ric_slice_prb_share", "PRB share allocated to the slice by the RL runner"),
    "prb_used": ("ric_slice_prb_used", "Fraction of the slice's own PRBs that carried data"),
    "buffer_bytes": ("ric_slice_buffer_bytes", "Data queued for the slice at the end of the last tick"),
}
CHECKS = ("ok", "schema", "window", "latency_physics", "budget", "floor", "training_range", "conflict", "step")


class RicCollector(Collector):
    """Builds every metric from the database at scrape time (nothing cached in between)."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def collect(self) -> Iterator[GaugeMetricFamily]:
        if not self.path.exists():
            return  # no run yet: an empty scrape, not an error
        conn = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True, timeout=1.0)
        conn.row_factory = sqlite3.Row
        try:
            yield from self.slice_metrics(conn)
            yield from self.policy_metrics(conn)
            yield from self.intent_metrics(conn)
        except sqlite3.OperationalError:
            return  # tables not created yet, or the runner is clearing them: skip this scrape
        finally:
            conn.close()

    def slice_metrics(self, conn: sqlite3.Connection) -> Iterator[GaugeMetricFamily]:
        ts = conn.execute("SELECT MAX(ts) FROM kpi").fetchone()[0]
        if ts is None:
            return
        yield GaugeMetricFamily("ric_backend_clock_seconds", "Backend clock of the last tick", value=ts)
        rows = {r["slice"]: r for r in conn.execute("SELECT * FROM kpi WHERE ts = ?", (ts,))}
        for column, (name, help_text) in SLICE_FIELDS.items():
            family = GaugeMetricFamily(name, help_text, labels=["slice"])
            for s, row in rows.items():
                if row[column] is not None:
                    family.add_metric([s], row[column])
            yield family
        policy_id = next(iter(rows.values()))["policy_id"]
        yield GaugeMetricFamily("ric_policy_in_force", "a1_policy id in force during the last tick (0 = default)",
                                value=policy_id or 0)
        objective = DEFAULT_OBJECTIVE.model_dump()
        if policy_id is not None:
            row = conn.execute("SELECT objective FROM a1_policy WHERE policy_id = ?", (policy_id,)).fetchone()
            objective = json.loads(row["objective"])
        weights = GaugeMetricFamily("ric_policy_weight", "Weight of the slice in the objective in force",
                                    labels=["slice"])
        targets = GaugeMetricFamily("ric_policy_min_thr_mbps", "Minimum throughput of the slice in the objective"
                                    " in force", labels=["slice"])
        for s in SLICES:
            weights.add_metric([s], objective["weights"][s])
            targets.add_metric([s], objective["min_thr_mbps"][s])
        yield weights
        yield targets
        if objective["max_latency_ms"] is not None:
            yield GaugeMetricFamily("ric_policy_latency_bound_ms", "URLLC latency bound of the objective in force",
                                    value=objective["max_latency_ms"])

    def policy_metrics(self, conn: sqlite3.Connection) -> Iterator[GaugeMetricFamily]:
        row = conn.execute("SELECT model_version FROM backend_info WHERE id = 1").fetchone()
        if row is not None:
            info = GaugeMetricFamily("ric_served_model_info", "Policy served by the RL runner", labels=["version"])
            info.add_metric([row["model_version"]], 1)
            yield info

    def intent_metrics(self, conn: sqlite3.Connection) -> Iterator[GaugeMetricFamily]:
        states = Counter(r["state"] for r in conn.execute("SELECT state FROM intents"))
        family = GaugeMetricFamily("ric_intents", "Intents per state", labels=["state"])
        for state in STATES:
            family.add_metric([state], states.get(state, 0))
        yield family
        checks = Counter(json.loads(r["verifier"])["check"]
                         for r in conn.execute("SELECT verifier FROM intents WHERE verifier IS NOT NULL"))
        family = GaugeMetricFamily("ric_verifier_decisions", "Intents per deciding verifier check ('ok' = accepted)",
                                   labels=["check"])
        for check in CHECKS:
            family.add_metric([check], checks.get(check, 0))
        yield family


def main() -> None:
    parser = argparse.ArgumentParser(description="Prometheus exporter for var/ric.db")
    parser.add_argument("--port", type=int, default=9108)
    parser.add_argument("--host", default="0.0.0.0", help="0.0.0.0 so Prometheus in Docker can reach it")
    args = parser.parse_args()
    REGISTRY.register(RicCollector(db_path()))
    start_http_server(args.port, addr=args.host)
    print(f"exporter: http://localhost:{args.port}/metrics (reading {db_path()})")
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
