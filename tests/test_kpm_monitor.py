"""M07a: KPM reports (the C xApp's JSON lines) -> one KPI record per slice per tick.

The testbed cannot run in CI, so these tests feed JSON lines in the xApp's format: synthetic
ones for each rule, and a short recording from the live testbed (tests/data/kpm_lines.jsonl).
"""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from store import kpi
from store.events import read_events
from xapps.kpm_monitor import (
    PRB_SHARE_ALLOC,
    SliceAggregator,
    UeReport,
    cell_info,
    implausible,
    parse_line,
    replay,
    to_record,
)

CAPACITY = 75.0
RECORDING = Path(__file__).parent / "data" / "kpm_lines.jsonl"


def line(slice_name: str, period: int, recv_s: float, thp_kbps: float, prb_pct: int, delay_us: float,
         ue: int = 1) -> str:
    """One xApp output line; period counts report periods (1 s each)."""
    return json.dumps({"slice": slice_name, "collect_start_us": period * 1_000_000, "recv_us": int(recv_s * 1e6),
                       "amf_ue_ngap_id": ue, "DRB.UEThpDl": thp_kbps, "RRU.PrbTotDl": prb_pct,
                       "DRB.RlcSduDelayDl": delay_us})


def report(slice_name: str = "embb", thp: float = 0.0, prb: float = 0.0, delay: float | None = 0.0,
           period: int = 1, arrived: float = 0.0, ue: int = 1) -> UeReport:
    return UeReport(slice_name, period, arrived, ue, thp, prb, delay)


def test_parse_line_tells_reports_events_and_flexric_logs_apart() -> None:
    assert parse_line("[xApp]: E42 SETUP-RESPONSE rx") is None
    assert parse_line('{"event":"subscribed","slice":"urllc","sst":2,"sd":"000001","ok":true}')["event"] == "subscribed"
    r = parse_line(line("urllc", 7, 12.5, 10_000.0, 14, 850.0))
    assert (r.slice, r.collect_start_us, r.arrived_s, r.thp_dl_kbps, r.prb_dl_pct, r.rlc_delay_dl_us) == (
        "urllc", 7_000_000, 12.5, 10_000.0, 14, 850.0)
    assert parse_line(line("mmtc", 1, 1.0, 1.0, 1, 1.0)) is None  # not one of our slices


def test_record_sums_throughput_and_prb_and_keeps_the_worst_latency() -> None:
    ues = [report(thp=20_000, prb=30, delay=4_000, ue=1), report(thp=10_000, prb=10, delay=9_000, ue=2)]
    r = to_record("embb", ues, ts=100.0)
    assert r.dl_thr_mbps == pytest.approx(30.0)  # kb/s -> Mb/s, summed over the slice's UEs
    assert r.prb_used == pytest.approx(0.40 / PRB_SHARE_ALLOC)  # % of the cell -> fraction of the slice's PRBs
    assert r.latency_ms == pytest.approx(9.0)  # us -> ms, worst UE
    assert (r.n_ue, r.source, r.prb_share_alloc, r.offered_load_mbps, r.buffer_bytes) == (2, "flexric", 1.0, None, None)


def test_wrapped_counters_are_not_written() -> None:
    """The stock-gNB bug shows up as unsigned wrap-around: such reports are refused."""
    assert implausible(report(prb=4_294_967_200), CAPACITY) is not None
    assert implausible(report(thp=3.4e8), CAPACITY) is not None
    assert implausible(report(thp=70_000, prb=95), CAPACITY) is None


def test_first_report_after_subscription_is_discarded(ric_db) -> None:
    agg = SliceAggregator(CAPACITY)
    agg.add(report(thp=5e5, period=1, arrived=1.0))  # "since the counters started": dropped
    assert agg.tick(1.5) == []
    agg.add(report(thp=30_000, period=2, arrived=2.0))
    [rec] = agg.tick(2.5)
    assert rec.dl_thr_mbps == pytest.approx(30.0)
    agg.on_event({"event": "subscribed", "slice": "embb", "ok": True})  # xApp restarted: count again
    agg.add(report(thp=1e5, period=3, arrived=3.0))
    assert agg.tick(3.5) == []
    assert "report_discarded" in [e["event"] for e in read_events()]


def test_one_tick_has_every_fresh_slice_at_one_ts_and_skips_stale_ones(ric_db) -> None:
    agg = SliceAggregator(CAPACITY)
    for s, arrived in (("embb", 0.2), ("urllc", 0.5), ("be", 0.9)):
        agg.add(report(s, period=1, arrived=arrived))  # discarded
        agg.add(report(s, thp=1000, period=2, arrived=arrived + 1))
    ticks = agg.tick(2.0)
    assert [r.slice for r in ticks] == ["embb", "urllc", "be"] and {r.ts for r in ticks} == {2.0}
    # Sample and hold: no new report for 1 s keeps the last value; past STALE_S the slice is left out.
    assert len(agg.tick(3.0)) == 3
    agg.add(report("urllc", thp=2000, period=3, arrived=3.4))
    assert [r.slice for r in agg.tick(4.0)] == ["urllc"]
    stale = [e["data"]["slice"] for e in read_events() if e["event"] == "slice_stale"]
    assert sorted(stale) == ["be", "embb"]


def test_several_ues_of_one_report_are_aggregated() -> None:
    agg = SliceAggregator(CAPACITY, events=False)
    agg.add(report(period=1, arrived=0.1))
    agg.add(report(thp=1000, prb=5, period=2, arrived=1.1, ue=1))
    agg.add(report(thp=3000, prb=7, period=2, arrived=1.1, ue=2))
    [rec] = agg.tick(1.5)
    assert (rec.n_ue, rec.dl_thr_mbps, rec.prb_used) == (2, pytest.approx(4.0), pytest.approx(0.12))


def test_replay_ticks_once_per_second_of_the_xapp_clock() -> None:
    lines = ["[xApp]: noise"]
    for k in range(1, 6):  # 5 report periods, the three slices at different phases
        lines += [line("embb", k, k + 0.1, 30_000, 40, 2_000), line("urllc", k, k + 0.4, 10_000, 14, 900),
                  line("be", k, k + 0.7, 5_000, 7, 1_500)]
    ticks = replay(lines, CAPACITY)
    assert all(len(t) == 3 for t in ticks[1:])
    last = {r.slice: r for r in ticks[-1]}
    assert last["embb"].dl_thr_mbps == pytest.approx(30.0) and last["urllc"].latency_ms == pytest.approx(0.9)


def test_testbed_records_are_read_by_get_kpis_and_the_agent_api(ric_db) -> None:
    """Meeting point #1: the same reader code serves testbed data, unchanged."""
    from agent.service import app

    kpi.write_backend_info(ric_db, cell_info(), "none (monitoring only, M07a)")
    lines = [line(s, k, k + off, thp, prb, d) for k in range(1, 13)
             for s, off, thp, prb, d in (("embb", 0.1, 30_000, 40, 2_000), ("urllc", 0.4, 10_000, 14, 900),
                                         ("be", 0.7, 5_000, 7, 1_500))]
    for records in replay(lines, CAPACITY):
        kpi.write_records(ric_db, records, policy_id=None)
    body = TestClient(app).get("/kpis", params={"window_s": 5}).json()
    assert set(body["slices"]) == {"embb", "urllc", "be"}
    assert body["slices"]["urllc"]["dl_thr_mbps"]["mean"] == pytest.approx(10.0)
    assert body["slices"]["embb"]["latency_ms"]["last"] == pytest.approx(2.0)
    status = TestClient(app).get("/status").json()
    assert status["backend"]["source"] == "flexric" and status["backend"]["n_prb"] == 106


@pytest.mark.skipif(not RECORDING.exists(), reason="recorded after the live M07a run")
def test_recording_from_the_testbed_gives_plausible_records() -> None:
    ticks = replay(RECORDING.read_text().splitlines(), CAPACITY)
    assert ticks, "no tick in the recording"
    for records in ticks:
        assert {r.slice for r in records} <= {"embb", "urllc", "be"}
        assert all(0 <= r.dl_thr_mbps <= 2 * CAPACITY and 0 <= r.prb_used <= 1.0 for r in records)
