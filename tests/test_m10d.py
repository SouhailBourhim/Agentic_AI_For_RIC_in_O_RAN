"""M10d: the controller can run on the demand the gNB measures (KPM DRB.PdcpSduVolumeDL) instead of the iperf3
rate, without changing what scoring reads. No testbed here: the PDCP values are written by hand."""

import dataclasses
import json
import sqlite3
import time

import pytest

from backends.sim import SimBackend, scenario_traffic
from eval import m10d
from rl.policy import SlicePolicy
from rl.runner import DEMANDS, Runner, observation_from_records, seen_loads
from sim.cell import CellConfig
from store import db, testbed
from store.events import read_events
from store.kpi import KpiRecord
from test_flexric_backend import FAKE_XAPP, backend
from verifier.objective import DEFAULT_OBJECTIVE, SLICES
from xapps import kpm_monitor as km

ORACLE_GARBAGE = 1e6  # an offered load no cell has: if it reached the controller, its split would show it


def record(s: str, offered: float | None, measured: float | None, thr: float = 7.0) -> KpiRecord:
    return KpiRecord(ts=1.0, source="flexric", cell_id="c", slice=s, dl_thr_mbps=thr, prb_share_alloc=1 / 3,
                     prb_used=0.5, n_ue=1, latency_ms=1.0, offered_load_mbps=offered, measured_load_mbps=measured)


def test_each_demand_reads_its_own_field(ric_db) -> None:
    records = [record(s, offered=61.0, measured=42.0) for s in SLICES]
    assert seen_loads(records, "offered") == {s: 61.0 for s in SLICES}
    assert seen_loads(records, "measured") == {s: 42.0 for s in SLICES}
    assert seen_loads(records, "throughput") == {s: 7.0 for s in SLICES}
    assert seen_loads([record(s, None, None) for s in SLICES], "measured") == {s: 7.0 for s in SLICES}  # not reported
    assert seen_loads(records) == seen_loads(records, "offered")  # the default is what every earlier run used
    with pytest.raises(ValueError):
        seen_loads(records, "pdcp")
    with pytest.raises(ValueError):
        Runner(SimBackend(), SlicePolicy.__new__(SlicePolicy), db.connect(timeout_s=0.0), demand="pdcp")


def test_the_observation_ignores_the_oracle_in_measured_mode() -> None:
    cfg = CellConfig()
    truth = [record(s, offered=v, measured=None) for s, v in zip(SLICES, (40.0, 12.0, 20.0), strict=True)]
    measured = [record(s, offered=ORACLE_GARBAGE, measured=r.offered_load_mbps) for s, r in zip(SLICES, truth, strict=True)]
    assert (observation_from_records(measured, DEFAULT_OBJECTIVE, cfg, demand="measured")
            == pytest.approx(observation_from_records(truth, DEFAULT_OBJECTIVE, cfg, demand="offered")))


class MeasuredOnly(SimBackend):
    """The simulated cell, with its true arrivals moved to measured_load_mbps and garbage in offered_load_mbps."""

    def read_kpis(self) -> list[KpiRecord]:
        return [dataclasses.replace(r, measured_load_mbps=r.offered_load_mbps, offered_load_mbps=ORACLE_GARBAGE)
                for r in super().read_kpis()]


def test_measured_mode_never_reads_the_oracle_over_a_run(ric_db, tiny_policy_file) -> None:
    """Same cell, same policy: measured mode on MeasuredOnly splits exactly as offered mode on the plain cell,
    so neither the observation nor the load history saw ORACLE_GARBAGE."""
    cfg = CellConfig()

    def splits(backend, demand: str) -> list[tuple[float, ...]]:
        runner = Runner(backend, SlicePolicy.load(tiny_policy_file), db.connect(timeout_s=0.0), demand=demand)
        return [runner.tick().shares for _ in range(150)]  # past the demo's surge at 120 s

    plain = splits(SimBackend(cfg, scenario_traffic("demo", cfg), seed=3), "offered")
    assert splits(MeasuredOnly(cfg, scenario_traffic("demo", cfg), seed=3), "measured") == plain


def test_the_monitor_turns_pdcp_volume_into_mbps() -> None:
    line = json.dumps({"slice": "urllc", "collect_start_us": 1, "recv_us": 2_000_000, "amf_ue_ngap_id": 2,
                       km.THP_DL: 8000.0, km.PRB_DL: 9, km.DELAY_DL: 500.0, km.PDCP_DL: 12})
    report = km.parse_line(line)
    assert report.pdcp_dl_mb == 12
    assert km.to_record("urllc", [report], ts=2.0).measured_load_mbps == pytest.approx(12 / km.KPM_PERIOD_S)
    assert km.to_record("urllc", [dataclasses.replace(report, pdcp_dl_mb=None)], ts=2.0).measured_load_mbps is None


def test_a_wrapped_pdcp_counter_blanks_only_that_measure(ric_db) -> None:
    """A gNB without the M10d patch reports wrapped PDCP values: throughput, PRBs and delay must still go through."""
    agg = km.SliceAggregator(capacity_mbps=79.0)
    for period in (1, 2):  # the first report after a subscription is always discarded
        agg.add(km.UeReport("urllc", period, 1.0, 2, 8000.0, 9, 500.0, pdcp_dl_mb=4_294_967_000))
    (rec,) = [r for r in agg.tick(now=1.5) if r.slice == "urllc"]
    assert rec.dl_thr_mbps == pytest.approx(8.0) and rec.measured_load_mbps is None
    assert any(e["event"] == "pdcp_implausible" for e in read_events())


def test_flexric_backend_passes_the_measured_demand_and_keeps_the_truth(ric_db) -> None:
    b = backend(ric_db, FAKE_XAPP)
    now = time.time()
    testbed.write_latest(ric_db, [testbed.SliceReport(s, now, 5.0, 0.1, 1.0, 1, measured_load_mbps=m)
                                  for s, m in zip(SLICES, (30.0, 11.0, None), strict=True)])
    testbed.write_traffic_target(ric_db, {"embb": 31.0, "urllc": 10.0, "be": 5.0})
    by_slice = {r.slice: r for r in b.read_kpis()}
    assert [by_slice[s].measured_load_mbps for s in SLICES] == [30.0, 11.0, None]
    assert [by_slice[s].offered_load_mbps for s in SLICES] == [31.0, 10.0, 5.0]  # scoring's truth, unchanged
    b.close()


def test_an_older_database_gets_the_new_columns(tmp_path) -> None:
    path = tmp_path / "old.db"
    old = sqlite3.connect(path)
    old.executescript("""
        CREATE TABLE kpi (ts REAL NOT NULL, source TEXT NOT NULL, cell_id TEXT NOT NULL, slice TEXT NOT NULL,
            dl_thr_mbps REAL NOT NULL, prb_share_alloc REAL NOT NULL, prb_used REAL NOT NULL, n_ue INTEGER NOT NULL,
            latency_ms REAL, offered_load_mbps REAL, buffer_bytes REAL, policy_id INTEGER);
        CREATE TABLE kpm_latest (slice TEXT PRIMARY KEY, report_wall REAL NOT NULL, dl_thr_mbps REAL NOT NULL,
            prb_cell REAL NOT NULL, latency_ms REAL, n_ue INTEGER NOT NULL);
    """)
    old.close()
    conn = db.connect(path)
    for table in ("kpi", "kpm_latest"):
        assert "measured_load_mbps" in {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    testbed.write_latest(conn, [testbed.SliceReport("embb", 1.0, 5.0, 0.1, None, 1, 9.0)])
    assert testbed.read_latest(conn)["embb"].measured_load_mbps == 9.0


def test_demands_are_the_three_documented() -> None:
    assert DEMANDS == ("offered", "measured", "throughput")


# ---------------------------------------------------------------- eval/m10d.py: the rules, before any testbed run

PLAN_RUNS = [
    {"phase": "steady", "label": "steady", "rates_mbps": {"embb": 30.0, "urllc": 10.0, "be": 5.0}, "seconds": 60,
     "start_wall": 0.0, "end_wall": 60.0},
    {"phase": "steps", "label": "step_12", "rates_mbps": {"urllc": 12.0}, "seconds": 20, "start_wall": 100.0, "end_wall": 120.0},
    {"phase": "overload", "label": "overload", "rates_mbps": {"urllc": 30.0}, "seconds": 40, "start_wall": 200.0,
     "end_wall": 240.0},
]


def series(pdcp_of, thr_of) -> dict[str, list]:
    """One report per second per slice over 0-250 s; pdcp_of / thr_of give each slice's value at time t."""
    return {s: [(float(t), pdcp_of(s, t), thr_of(s, t)) for t in range(250)] for s in SLICES}


def demand(s: str, t: float) -> float:
    for run in PLAN_RUNS:
        if run["start_wall"] <= t < run["end_wall"]:
            return run["rates_mbps"].get(s, 0.0)
    return 0.0


def service(s: str, t: float) -> float:
    return min(demand(s, t), 3.0) if 200 <= t < 240 else demand(s, t)  # URLLC capped at ~3 Mb/s in the overload


def test_a_measure_of_the_demand_passes_the_calibration() -> None:
    result = m10d.calibration(PLAN_RUNS, series(demand, service))
    assert result["pass"], [c for c in result["checks"] if not c["ok"]]
    assert len(result["checks"]) == 3 + 1 + 2


def test_a_measure_of_the_service_fails_the_overload() -> None:
    result = m10d.calibration(PLAN_RUNS, series(service, service))  # what throughput would give
    failed = {c["check"] for c in result["checks"] if not c["ok"]}
    assert not result["pass"] and any(name.startswith("overload urllc at 30 Mb/s") for name in failed)


def test_a_slow_measure_fails_the_step() -> None:
    late = lambda s, t: demand(s, t - 3)  # noqa: E731  three reports late
    result = m10d.calibration(PLAN_RUNS, series(late, service))
    assert any(c["check"].startswith("step urllc") and not c["ok"] for c in result["checks"])


def test_the_tolerance_is_1_mbps_or_5_percent() -> None:
    assert m10d.tolerance(5.0) == 1.0 and m10d.tolerance(30.0) == pytest.approx(1.5)


def test_no_pdcp_values_fail_rather_than_pass() -> None:
    result = m10d.calibration(PLAN_RUNS, series(lambda s, t: None, service))
    assert not result["pass"]


def test_the_kpm_lines_become_report_series() -> None:
    lines = [json.dumps({"slice": "urllc", "collect_start_us": 5_000_000, "recv_us": 6_000_000, "amf_ue_ngap_id": 2,
                         km.THP_DL: 3000.0, km.PRB_DL: 5, km.DELAY_DL: 900.0, km.PDCP_DL: 30}), "[xApp]: not json"]
    assert m10d.reports_by_slice(lines) == {"urllc": [(5.0, 30.0, 3.0)]}


def test_the_matrix_is_18_runs_with_the_right_runner_arguments() -> None:
    assert len(m10d.CELLS) * 3 == 18
    assert m10d.extra_args("rl", "measured") == "--demand measured"
    assert m10d.extra_args("formula", "throughput") == "--baseline formula --demand throughput"
    assert m10d.run_of("formula", "offered", 2).run_id == "m10d_formula_offered/s1/proposed/repeat2"
    for repeat in (1, 2, 3):
        assert sorted(m10d.order(repeat)) == sorted(m10d.CELLS)
    assert m10d.order(1) != m10d.order(2)


def test_s1_is_refused_until_the_calibration_passes(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(m10d, "CALIB_OUT", tmp_path / "calibration.json")
    with pytest.raises(SystemExit):
        m10d.require_calibration_pass()
    (tmp_path / "calibration.json").write_text(json.dumps({"pass": False, "checks": []}))
    with pytest.raises(SystemExit):
        m10d.require_calibration_pass()
    (tmp_path / "calibration.json").write_text(json.dumps({"pass": True, "checks": []}))
    m10d.require_calibration_pass()


def test_h_d_is_the_gap_between_measured_and_offered_for_rl() -> None:
    rows = [{"executor": e, "demand": d, "repeat": r, "valid": True, "invalid_copy": False,
             "intent_met": {"offered": 0.96, "measured": 0.93, "throughput": 0.30}[d]}
            for e, d in m10d.CELLS for r in (1, 2, 3)]
    rows.append({**rows[0], "valid": False, "intent_met": 0.0})  # an invalid run is not counted
    summary = m10d.summarise(rows)
    assert summary["medians"]["rl/offered"] == {"median": 0.96, "n": 3}
    assert summary["h_d"]["rl_measured_minus_offered"] == pytest.approx(-0.03) and summary["h_d"]["holds"]


def test_revision_1_the_drain_after_the_sender_stops_is_not_scored() -> None:
    """The first calibration run: iperf3 returned 100 s after its 40 s, while the saturated slice drained its backlog
    and PDCP rightly read 0. Only the sending window counts."""
    draining = [{**run, "end_wall": run["end_wall"] + 100.0} if run["phase"] == "overload" else run for run in PLAN_RUNS]
    reports = {s: [(float(t), demand(s, t), service(s, t)) for t in range(400)] for s in SLICES}
    assert m10d.calibration(draining, reports)["pass"]


def test_calibration_data_from_before_revision_1_is_refused(tmp_path) -> None:
    run = {**PLAN_RUNS[0], "start_wall": m10d.REVISION_1_NOT_BEFORE - 1000}
    (tmp_path / "calib_runs.jsonl").write_text(json.dumps(run) + "\n")
    (tmp_path / "calib_kpm.jsonl").write_text("")
    with pytest.raises(SystemExit, match="before revision 1"):
        m10d.score_calibration(tmp_path)


def test_the_demand_series_uses_when_iperf3_really_sent(tmp_path) -> None:
    """A session that starts late (its handshake waited in the slice's queue) is 'sent' only from its start."""
    t0 = 1000.0
    (tmp_path / "run.json").write_text(json.dumps({"t0": t0}))
    events = [{"ts_wall": t0, "ts_backend": 0.0, "source": "scenario", "event": "segment_start", "intent_id": None,
               "data": {"segment": 0, "rates_mbps": {"embb": 40.0}, "seconds": 10}}]
    (tmp_path / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    (tmp_path / "seg0_embb.json").write_text(json.dumps({"start": {"timestamp": {"timesecs": t0 + 4}}}))
    database = tmp_path / "ric.db"
    conn = sqlite3.connect(database)
    conn.execute("CREATE TABLE kpi (ts REAL, slice TEXT, dl_thr_mbps REAL, offered_load_mbps REAL, measured_load_mbps REAL)")
    conn.executemany("INSERT INTO kpi VALUES (?, 'embb', 0, 40, 0)", [(t0 + t,) for t in (-1, 2, 5, 13, 15, 400)])
    conn.commit()
    out = tmp_path / "series.csv"
    assert m10d.export_demand_series(tmp_path, database, out) == 4  # -1 s and 400 s are outside S1's 360 s
    sent = {float(r.split(",")[0]): float(r.split(",")[-1]) for r in out.read_text().splitlines()[1:]}
    assert sent == {2.0: 0.0, 5.0: 40.0, 13.0: 40.0, 15.0: 0.0}  # sent over [4, 14), not the segment's [0, 10)
