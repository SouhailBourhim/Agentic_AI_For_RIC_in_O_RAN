"""The two small tables only the testbed path uses (M07b): kpm_latest and traffic_target.

On the simulator the RL runner gets its KPIs straight from the cell. On the testbed they arrive
from another process: the KPM xApp's reports, aggregated per slice by `xapps.kpm_monitor
--for-runner`, land in kpm_latest (one row per slice, replaced at every tick). FlexRICBackend
reads them, adds the PRB split it applied, and returns ordinary KPI records; the runner then
writes those to `kpi` with the policy in force, exactly as on the simulator. So `kpi` keeps a
single writer on both backends.

traffic_target holds the rate the traffic script is sending to each slice. KPM does not measure
the offered load, and the policy was trained with it as an input; the testbed uses this
"oracle" value and says so (docs/learn/M07b-slice-control.md).
"""

import sqlite3
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class SliceReport:
    """One slice's newest KPM values, before any knowledge of the split."""

    slice: str
    report_wall: float  # wall clock of the aggregator tick that produced it, s
    dl_thr_mbps: float
    prb_cell: float  # share of the WHOLE cell's PRBs the slice's UEs used, 0..1
    latency_ms: float | None
    n_ue: int


def write_latest(conn: sqlite3.Connection, reports: list[SliceReport]) -> None:
    conn.executemany(
        "INSERT OR REPLACE INTO kpm_latest VALUES (?, ?, ?, ?, ?, ?)",
        [(r.slice, r.report_wall, r.dl_thr_mbps, r.prb_cell, r.latency_ms, r.n_ue) for r in reports],
    )


def read_latest(conn: sqlite3.Connection) -> dict[str, SliceReport]:
    rows = conn.execute("SELECT * FROM kpm_latest").fetchall()
    return {r["slice"]: SliceReport(r["slice"], r["report_wall"], r["dl_thr_mbps"], r["prb_cell"],
                                    r["latency_ms"], r["n_ue"]) for r in rows}


def write_traffic_target(conn: sqlite3.Connection, rates_mbps: dict[str, float]) -> None:
    now = time.time()
    conn.executemany("INSERT OR REPLACE INTO traffic_target VALUES (?, ?, ?)",
                     [(s, float(rate), now) for s, rate in rates_mbps.items()])


def read_traffic_targets(conn: sqlite3.Connection) -> dict[str, float]:
    return {r["slice"]: r["offered_load_mbps"] for r in conn.execute("SELECT * FROM traffic_target")}
