"""KPM monitoring xApp, Python half: turns the C xApp's per-UE KPM reports into KPI records.

The C half (xapps/kpm/xapp_kpm_slices.c) runs in its own container, holds one KPM subscription
per slice and prints one JSON line per UE per report. This process reads those lines
(`docker logs -f xapp-kpm-slices`, or a saved file with --replay) and, once per second, writes
one KpiRecord per slice into the KPI store: the same record the simulator produces, so the
agent, the gateway and Grafana read testbed data with no change (docs/PLAN.md section 3).

Run from the repo root, with the testbed up and the xApp container running:

  uv run python -m xapps.kpm_monitor                   # live, into var/ric-testbed.db
  uv run python -m xapps.kpm_monitor --replay FILE     # offline, from saved JSON lines
  uv run python -m xapps.kpm_monitor --for-runner      # M07b: into kpm_latest, for FlexRICBackend

With --for-runner it does not write `kpi` itself: the RL runner does, from FlexRICBackend.read_kpis,
which knows the split it applied (store/testbed.py). The default mode is M07a's, unchanged.

Mapping from KPM to the record (decided in the spike, docs/learn/M06-spike.md section 6):
  dl_thr_mbps     = sum over the slice's UEs of DRB.UEThpDl (kb/s) / 1000
  prb_used        = (sum of RRU.PrbTotDl (% of the cell) / 100) / prb_share_alloc
  latency_ms      = max over the slice's UEs of DRB.RlcSduDelayDl (us) / 1000
  prb_share_alloc = 1.0 until M07b: the stock scheduler has no per-slice cap, so each slice may
                    use the whole cell
  offered_load_mbps, buffer_bytes: NULL (KPM does not measure them)
"""

import argparse
import json
import os
import subprocess
import threading
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from sim.cell import CellConfig
from store import db, kpi, testbed
from store.events import log_event
from store.kpi import CellInfo, KpiRecord
from verifier.objective import SLICES

TESTBED_CAPACITY_MBPS = 79.0  # measured for M08d (M08c used M07b's 69); eval/testbed_scenarios.py scales the scenarios to it
DEFAULT_TESTBED_DB = Path("var/ric-testbed.db")  # never the simulator's var/ric.db: the clocks differ
CONTAINER = "xapp-kpm-slices"
CELL_ID = "rfsim-gnb-e00-cell-12345678"  # gNB_ID 0xe00, nr_cellid 12345678 (testbed/rfsim-flexric/conf/gnb.conf)
TICK_S = 1.0  # one record per slice per second, the RL runner's tick
STALE_S = 2.0  # a slice's latest report older than this is not written (two missed reports)
PRB_SHARE_ALLOC = 1.0  # see the module docstring; M07b replaces it with the applied split
MODEL_VERSION = "none (monitoring only, M07a)"

# Measurement names in the xApp's JSON lines (3GPP TS 28.552 names, as the gNB reports them)
THP_DL, PRB_DL, DELAY_DL = "DRB.UEThpDl", "RRU.PrbTotDl", "DRB.RlcSduDelayDl"


@dataclass(frozen=True)
class UeReport:
    """One UE in one KPM report of one slice's subscription."""

    slice: str
    collect_start_us: int  # start of the granularity period (from the gNB's indication header)
    arrived_s: float  # when this process got the line (live) or the xApp received it (replay)
    ue: int | None  # AMF UE NGAP ID
    thp_dl_kbps: float | None
    prb_dl_pct: float | None
    rlc_delay_dl_us: float | None


def parse_line(line: str, arrived_s: float | None = None) -> UeReport | dict | None:
    """A UE report, an xApp event ({"event": ...}), or None for anything else.

    `docker logs` also carries FlexRIC's own log lines ("[xApp]: ..."), which are not JSON.
    """
    line = line.strip()
    if not line.startswith("{"):
        return None
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return None
    if "event" in obj:
        return obj
    if obj.get("slice") not in SLICES:
        return None
    return UeReport(
        slice=obj["slice"],
        collect_start_us=int(obj["collect_start_us"]),
        arrived_s=arrived_s if arrived_s is not None else obj["recv_us"] / 1e6,
        ue=obj.get("amf_ue_ngap_id"),
        thp_dl_kbps=obj.get(THP_DL),
        prb_dl_pct=obj.get(PRB_DL),
        rlc_delay_dl_us=obj.get(DELAY_DL),
    )


def implausible(r: UeReport, capacity_mbps: float) -> str | None:
    """Why a report cannot be a real measurement, or None.

    The gNB computes throughput and PRB % as differences of unsigned 32-bit counters, so a
    wrong "previous value" (the stock OAI v3.0.0 bug with several subscriptions, see
    docs/learn/M07a-kpm.md) shows up as a wrapped, huge number. Such a report is logged and
    not written: the KPI store must only hold values the RAN really measured.
    """
    if r.prb_dl_pct is not None and r.prb_dl_pct > 100:
        return f"PRB {r.prb_dl_pct:g} % of the cell"
    if r.thp_dl_kbps is not None and r.thp_dl_kbps / 1000 > 2 * capacity_mbps:
        return f"throughput {r.thp_dl_kbps / 1000:.0f} Mb/s, over twice the cell capacity"
    return None


def to_record(slice_name: str, ues: list[UeReport], ts: float, cell_id: str = CELL_ID) -> KpiRecord:
    """One slice's KPI record from the UE reports of one KPM report period."""
    thr = sum(u.thp_dl_kbps for u in ues if u.thp_dl_kbps is not None) / 1000
    prb_cell = sum(u.prb_dl_pct for u in ues if u.prb_dl_pct is not None) / 100
    delays = [u.rlc_delay_dl_us for u in ues if u.rlc_delay_dl_us is not None]
    return KpiRecord(
        ts=ts,
        source="flexric",
        cell_id=cell_id,
        slice=slice_name,
        dl_thr_mbps=thr,
        prb_share_alloc=PRB_SHARE_ALLOC,
        prb_used=prb_cell / PRB_SHARE_ALLOC,
        n_ue=len(ues),
        latency_ms=max(delays) / 1000 if delays else None,
    )


class SliceAggregator:
    """Collects UE reports and produces one batch of records per tick.

    The three subscriptions report once per second each, but at different moments. At each
    tick every slice gets the newest report period it has (sample and hold), and all records
    of a tick share one ts: the exporter and get_kpis read a tick as "the rows at one ts".
    """

    def __init__(self, capacity_mbps: float, events: bool = True) -> None:
        self.capacity_mbps = capacity_mbps
        self.events = events
        self.newest: dict[str, tuple[int, float, list[UeReport]]] = {}  # slice -> (period, arrived, UEs)
        self.periods_seen: dict[str, int] = {s: 0 for s in SLICES}
        self.last_period: dict[str, int] = {}  # slice -> collect_start_us of its latest report
        self.stale: set[str] = set()

    def log(self, event: str, ts: float | None = None, **data) -> None:
        if self.events:
            log_event("kpm_xapp", event, ts_backend=ts, data=data)

    def on_event(self, ev: dict) -> None:
        """A (re)subscription restarts the slice's count, so its first report is discarded again."""
        if ev.get("event") == "subscribed" and ev.get("slice") in SLICES:
            self.periods_seen[ev["slice"]] = 0
            self.last_period.pop(ev["slice"], None)
            self.newest.pop(ev["slice"], None)
        self.log(f"xapp_{ev.get('event')}", **{k: v for k, v in ev.items() if k != "event"})

    def add(self, r: UeReport) -> None:
        if r.collect_start_us != self.last_period.get(r.slice):  # a new report period of this slice
            self.last_period[r.slice] = r.collect_start_us
            self.periods_seen[r.slice] += 1
            if self.periods_seen[r.slice] == 1:
                self.log("report_discarded", slice=r.slice, reason="first report after subscription")
        if self.periods_seen[r.slice] == 1:
            # The first report after a subscription measures "since the counters started" (or
            # since another subscription read them), not one period. Checked before plausibility,
            # because that first value is often huge.
            return
        reason = implausible(r, self.capacity_mbps)
        if reason is not None:
            self.log("report_implausible", slice=r.slice, ue=r.ue, reason=reason)
            return
        current = self.newest.get(r.slice)
        if current is not None and current[0] == r.collect_start_us:
            current[2].append(r)  # another UE of the same report
        else:
            self.newest[r.slice] = (r.collect_start_us, r.arrived_s, [r])

    def tick(self, now: float) -> list[KpiRecord]:
        records = []
        for s in SLICES:
            found = self.newest.get(s)
            if found is None or now - found[1] > STALE_S:
                if s not in self.stale:
                    self.stale.add(s)
                    self.log("slice_stale", ts=now, slice=s,
                             last_report_age_s=None if found is None else round(now - found[1], 2))
                continue
            if s in self.stale:
                self.stale.discard(s)
                self.log("slice_fresh", ts=now, slice=s)
            records.append(to_record(s, found[2], now))
        return records


def cell_info() -> CellInfo:
    """What the agent and the verifier need to know about the testbed cell.

    n_prb is the gNB's 106 PRB. The capacity is what this cell carried with the slice scheduler, by KPM, at the
    loads the scenarios use (testbed/measure_capacity_m08d.py), not the simulator's 75: the verifier then judges
    what the cell can really carry, and the policy's inputs, fractions of the capacity, mean what they say. The
    URLLC base latency is the simulator's calibrated value, more than this cell's 0.6 ms (M08c, M08d's what-if).
    """
    cfg = CellConfig()
    return CellInfo(cell_id=CELL_ID, source="flexric", slices=tuple(SLICES), n_prb=cfg.n_prb,
                    capacity_mbps=TESTBED_CAPACITY_MBPS, urllc_base_latency_ms=cfg.base_latency_ms["urllc"])


def replay(lines: Iterable[str], capacity_mbps: float, events: bool = False) -> list[list[KpiRecord]]:
    """Feed saved lines in order, ticking every TICK_S of the xApp's receive clock.

    The same aggregation as the live loop, but deterministic: used by the tests and to rebuild
    the KPI rows of a recorded run.
    """
    agg = SliceAggregator(capacity_mbps, events=events)
    ticks: list[list[KpiRecord]] = []
    next_tick: float | None = None
    for line in lines:
        item = parse_line(line)
        if isinstance(item, dict):
            agg.on_event(item)
            continue
        if item is None:
            continue
        if next_tick is None:
            next_tick = item.arrived_s + TICK_S
        while item.arrived_s >= next_tick:  # ticks that happened before this line
            ticks.append(agg.tick(next_tick))
            next_tick += TICK_S
        agg.add(item)
    if next_tick is not None:
        ticks.append(agg.tick(next_tick))
    return [t for t in ticks if t]


def docker_lines(container: str) -> Iterator[str]:
    """The xApp's stdout, from now on (`--since 0s` skips what was printed before we started)."""
    proc = subprocess.Popen(["docker", "logs", "-f", "--since", "0s", container],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert proc.stdout is not None
    yield from proc.stdout


def write_kpi(conn, records: list[KpiRecord]) -> None:
    """M07a mode: the records go straight to `kpi` (nothing controls the split)."""
    kpi.write_records(conn, records, policy_id=None)


def write_for_runner(conn, records: list[KpiRecord]) -> None:
    """M07b mode: the newest values per slice, for FlexRICBackend. PRB_SHARE_ALLOC is 1.0, so the
    record's prb_used is the share of the whole cell."""
    testbed.write_latest(conn, [testbed.SliceReport(r.slice, r.ts, r.dl_thr_mbps, r.prb_used, r.latency_ms, r.n_ue)
                                for r in records])


def run_live(container: str, conn, raw_out: Path | None, write=write_kpi) -> None:
    """A reader thread feeds the aggregator; the main loop writes one tick per second."""
    agg = SliceAggregator(cell_info().capacity_mbps)
    lock = threading.Lock()
    raw = raw_out.open("a", encoding="utf-8") if raw_out else None

    def reader() -> None:
        for line in docker_lines(container):
            item = parse_line(line, arrived_s=time.time())
            if raw is not None and item is not None:
                raw.write(line if line.endswith("\n") else line + "\n")
                raw.flush()
            with lock:
                if isinstance(item, dict):
                    agg.on_event(item)
                elif item is not None:
                    agg.add(item)
        agg.log("xapp_log_closed", container=container)

    threading.Thread(target=reader, daemon=True).start()
    next_tick = time.time() + TICK_S
    while True:
        time.sleep(max(0.0, next_tick - time.time()))
        with lock:
            records = agg.tick(next_tick)
        if records:
            write(conn, records)
        next_tick += TICK_S


def main() -> None:
    parser = argparse.ArgumentParser(description="KPM reports -> KPI store (M07a)")
    parser.add_argument("--container", default=CONTAINER)
    parser.add_argument("--replay", type=Path, help="saved JSON lines instead of the live container")
    parser.add_argument("--raw-out", type=Path, help="live: also append the xApp's JSON lines to this file")
    parser.add_argument("--for-runner", action="store_true",
                        help="write kpm_latest for the RL runner (M07b) instead of kpi")
    args = parser.parse_args()
    if "RIC_DB" not in os.environ:
        os.environ["RIC_DB"] = str(DEFAULT_TESTBED_DB)
    conn = db.connect()
    if args.for_runner:  # the runner publishes backend_info and writes kpi
        print(f"kpm_latest in: {db.db_path()}")
        try:
            run_live(args.container, conn, args.raw_out, write=write_for_runner)
        except KeyboardInterrupt:
            pass
        return
    kpi.write_backend_info(conn, cell_info(), MODEL_VERSION)
    print(f"KPI store: {db.db_path()}")
    if args.replay:
        ticks = replay(args.replay.read_text().splitlines(), cell_info().capacity_mbps)
        for records in ticks:
            kpi.write_records(conn, records, policy_id=None)
        print(f"replayed {len(ticks)} ticks from {args.replay}")
        return
    try:
        run_live(args.container, conn, args.raw_out)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
