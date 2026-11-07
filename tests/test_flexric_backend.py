"""M07b: FlexRICBackend without a testbed. A small Python program stands in for the slice-control
xApp (it reads the same lines and answers with the same JSON), and the KPM values are written to
kpm_latest by hand, as `xapps.kpm_monitor --for-runner` would.
"""

import sys
import time

import pytest

from backends.flexric import N_PRB, PRB_FULL_CELL, FlexRICBackend, shares_to_prbs
from rl.policy import SMOOTHING_ALPHA, SlicePolicy
from rl.runner import SERVED_SMOOTHING_ALPHA, Runner, load_policy
from store import testbed
from store.events import read_events
from xapps.kpm_monitor import cell_info

# Answers each line like xapp_slice_ctrl: {"event":"ack","prbs":[...]}; after "exit", stops.
FAKE_XAPP = """
import json, sys
print(json.dumps({"event": "ready"}), flush=True)
for line in sys.stdin:
    if line.strip() == "exit":
        break
    print(json.dumps({"event": "ack", "cmd": "add", "prbs": [int(x) for x in line.split()], "rtt_ms": 1.0}), flush=True)
"""
SLOW_XAPP = "import time; time.sleep(30)"  # never reads its stdin
DEAD_XAPP = "pass"


def wait_for(condition, timeout_s: float = 5.0) -> None:
    end = time.monotonic() + timeout_s
    while not condition():
        assert time.monotonic() < end, "timed out"
        time.sleep(0.02)


def backend(ric_db, program: str) -> FlexRICBackend:
    return FlexRICBackend(ric_db, cell_info(), xapp_cmd=[sys.executable, "-c", program])


@pytest.mark.parametrize("shares, expected", [
    ((1 / 3, 1 / 3, 1 / 3), [36, 35, 35]),  # 35.33 each: the one PRB left goes to the first slice
    ((0.6, 0.3, 0.1), [64, 32, 10]),  # 63.6 / 31.8 / 10.6
    ((0.05, 0.05, 0.9), [5, 5, 96]),  # 5.3 / 5.3 / 95.4: floors 5 5 95; the PRB left goes to BE (0.4)
])
def test_shares_to_prbs(shares, expected) -> None:
    prbs = shares_to_prbs(list(shares))
    assert sum(prbs) == N_PRB and min(prbs) >= 5
    assert prbs == expected


def test_shares_to_prbs_tops_up_to_the_minimum() -> None:
    assert shares_to_prbs([0.04, 0.48, 0.48]) == [5, 50, 51]  # 4, 51, 51 after rounding; one from the first largest


def test_split_goes_to_the_xapp_as_one_line_and_its_answer_is_logged(ric_db) -> None:
    b = backend(ric_db, FAKE_XAPP)
    b.apply_prb_split([0.6, 0.3, 0.1])
    b.apply_prb_split([0.6, 0.3, 0.1])  # unchanged: not sent again
    b.apply_prb_split([0.5, 0.4, 0.1])
    wait_for(lambda: sum(e["event"] == "xapp_ack" for e in read_events()) == 2)
    acks = [e["data"]["prbs"] for e in read_events() if e["event"] == "xapp_ack"]
    assert acks == [[64, 32, 10], [53, 42, 11]]
    b.close()


def test_a_busy_xapp_never_blocks_the_tick(ric_db) -> None:
    """The xApp does not read: the lines wait in the pipe; apply_prb_split returns at once."""
    b = backend(ric_db, SLOW_XAPP)
    start = time.monotonic()
    for i in range(50):
        b.apply_prb_split([0.3 + i / 1000, 0.3, 0.4 - i / 1000])
    assert time.monotonic() - start < 0.5
    b.xapp.kill()


def test_a_dead_xapp_is_logged_and_the_runner_carries_on(ric_db) -> None:
    b = backend(ric_db, DEAD_XAPP)
    wait_for(lambda: any(e["event"] == "xapp_exited" for e in read_events()))
    b.apply_prb_split([0.6, 0.3, 0.1])  # no exception
    b.apply_prb_split([0.2, 0.3, 0.5])
    assert b.last_prbs is None


def test_kpis_use_the_split_in_force_during_the_report(ric_db) -> None:
    b = backend(ric_db, FAKE_XAPP)
    before = time.time()
    # Before any command the gNB runs its stock scheduler: each slice may use the whole cell.
    testbed.write_latest(ric_db, [testbed.SliceReport("urllc", before, 12.0, 0.15, 3.5, 1)])
    testbed.write_traffic_target(ric_db, {"urllc": 20.0})
    urllc = {r.slice: r for r in b.read_kpis()}["urllc"]
    assert (urllc.prb_share_alloc, urllc.offered_load_mbps) == (1.0, 20.0)
    assert urllc.prb_used == pytest.approx(0.15 / PRB_FULL_CELL)

    b.apply_prb_split([0.5, 0.3, 0.2])  # URLLC: 32 PRBs
    time.sleep(1.1)  # the next report covers a full second under the new split
    testbed.write_latest(ric_db, [testbed.SliceReport("urllc", time.time(), 12.0, 0.15, 3.5, 1)])
    records = {r.slice: r for r in b.read_kpis()}
    assert records["urllc"].prb_share_alloc == pytest.approx(32 / N_PRB)
    assert records["urllc"].prb_used == pytest.approx(0.15 / PRB_FULL_CELL / (32 / N_PRB))
    assert records["embb"].n_ue == 0 and records["embb"].offered_load_mbps is None  # nothing reported yet
    b.close()


def test_stale_kpm_is_logged_once(ric_db) -> None:
    b = backend(ric_db, FAKE_XAPP)
    testbed.write_latest(ric_db, [testbed.SliceReport(s, time.time() - 60, 1.0, 0.1, None, 1)
                                  for s in ("embb", "urllc", "be")])
    b.read_kpis()
    b.read_kpis()
    assert [e["event"] for e in read_events() if e["event"].startswith("kpm_")] == ["kpm_stale"]
    b.close()


def test_the_unchanged_runner_ticks_on_the_testbed_backend(ric_db, tiny_policy_file, monkeypatch) -> None:
    """Same Runner class as on the simulator: it reads the KPIs, sends a split, writes kpi rows."""
    b = backend(ric_db, FAKE_XAPP)
    monkeypatch.setattr(b, "sleep", lambda s: None)  # no real second per tick in a test
    testbed.write_latest(ric_db, [testbed.SliceReport(s, time.time(), 10.0, 0.2, 1.0, 1)
                                  for s in ("embb", "urllc", "be")])
    runner = Runner(b, SlicePolicy.load(tiny_policy_file), ric_db)
    result = runner.tick()
    assert result.db_ok and sum(result.shares) == pytest.approx(1.0)
    rows = ric_db.execute("SELECT source, slice, policy_id FROM kpi").fetchall()
    assert sorted(tuple(r) for r in rows) == [("flexric", s, None) for s in ("be", "embb", "urllc")]
    wait_for(lambda: any(e["event"] == "xapp_ack" for e in read_events()))
    b.close()


def test_the_served_filter_is_stronger_than_the_one_training_used(tiny_policy_file) -> None:
    """Training and the champion's recorded evaluation used alpha 0.25; a loaded policy is served at 0.10."""
    assert load_policy(tiny_policy_file)[0].alpha == SERVED_SMOOTHING_ALPHA == 0.10
    assert SMOOTHING_ALPHA == 0.25
