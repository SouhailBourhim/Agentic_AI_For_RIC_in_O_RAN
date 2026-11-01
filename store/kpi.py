"""KPI records (one per slice per tick) and the backend's cell info.

Written only by the RL runner, which gets them from its Backend; read by the agent (get_kpis,
the verifier's view of the clock and the cell) and by the Prometheus exporter. Both backends
(simulator, FlexRIC testbed) produce the same record, so nothing above this table needs to
know which one is running (docs/PLAN.md section 3).
"""

import json
import math
import sqlite3
import time
from dataclasses import asdict, dataclass

FIELDS = ("dl_thr_mbps", "prb_share_alloc", "prb_used", "latency_ms", "offered_load_mbps", "buffer_bytes")


@dataclass(frozen=True)
class KpiRecord:
    """What one slice did during one tick. Nullable fields are None when the backend cannot
    measure them (e.g. the offered load on the testbed is only the iperf3 target)."""

    ts: float  # backend clock at the end of the tick, s
    source: str  # 'sim' | 'flexric'
    cell_id: str
    slice: str
    dl_thr_mbps: float
    prb_share_alloc: float
    prb_used: float
    n_ue: int
    latency_ms: float | None = None
    offered_load_mbps: float | None = None
    buffer_bytes: float | None = None


@dataclass(frozen=True)
class CellInfo:
    """Fixed facts about the cell, published by the runner for the other processes."""

    cell_id: str
    source: str
    slices: tuple[str, ...]
    n_prb: int
    capacity_mbps: float
    urllc_base_latency_ms: float


def write_records(conn: sqlite3.Connection, records: list[KpiRecord], policy_id: int | None) -> None:
    """Append one tick's records, tagged with the a1_policy row that was in force."""
    rows = [(*asdict(r).values(), policy_id) for r in records]
    conn.executemany(
        "INSERT INTO kpi (ts, source, cell_id, slice, dl_thr_mbps, prb_share_alloc, prb_used, n_ue,"
        " latency_ms, offered_load_mbps, buffer_bytes, policy_id) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        rows,
    )


def latest_ts(conn: sqlite3.Connection) -> float | None:
    """The backend clock as last published by the runner (None before the first tick)."""
    return conn.execute("SELECT MAX(ts) FROM kpi").fetchone()[0]


def p95(values: list[float]) -> float:
    """95th percentile by the nearest-rank method: the smallest value with >= 95 % at or below it."""
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def get_kpis(conn: sqlite3.Connection, slice: str | None = None, window_s: float = 10.0) -> dict:
    """Mean, p95 and last value of each field per slice, over the last window_s of backend time.

    Returns {"ts": latest ts, "window_s": ..., "slices": {slice: {field: {mean, p95, last}}}}.
    Fields that are NULL over the whole window are left out.
    """
    end = latest_ts(conn)
    result: dict = {"ts": end, "window_s": window_s, "slices": {}}
    if end is None:
        return result
    query = "SELECT * FROM kpi WHERE ts > ? " + ("AND slice = ? " if slice else "") + "ORDER BY ts"
    params = (end - window_s, slice) if slice else (end - window_s,)
    by_slice: dict[str, list[sqlite3.Row]] = {}
    for row in conn.execute(query, params):
        by_slice.setdefault(row["slice"], []).append(row)
    for s, rows in by_slice.items():
        stats = {}
        for field in FIELDS:
            values = [r[field] for r in rows if r[field] is not None]
            if values:
                stats[field] = {"mean": sum(values) / len(values), "p95": p95(values), "last": values[-1]}
        result["slices"][s] = stats
    return result


def write_backend_info(conn: sqlite3.Connection, info: CellInfo, model_version: str) -> None:
    """Replace the single backend_info row (the runner calls this at startup)."""
    conn.execute(
        "INSERT OR REPLACE INTO backend_info VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?)",
        (info.cell_id, info.source, json.dumps(list(info.slices)), info.n_prb, info.capacity_mbps,
         info.urllc_base_latency_ms, model_version, time.time()),
    )


def read_backend_info(conn: sqlite3.Connection) -> tuple[CellInfo, str] | None:
    """(cell info, served model version), or None if no runner has started yet."""
    row = conn.execute("SELECT * FROM backend_info WHERE id = 1").fetchone()
    if row is None:
        return None
    info = CellInfo(row["cell_id"], row["source"], tuple(json.loads(row["slices"])), row["n_prb"],
                    row["capacity_mbps"], row["urllc_base_latency_ms"])
    return info, row["model_version"]
