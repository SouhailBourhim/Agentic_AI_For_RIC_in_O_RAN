"""M08c: the control evaluation on the testbed, the parts that need no testbed."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from agent.parser import parse_facts
from agent.schema import objective_from_decision
from eval import testbed_control as T
from eval.extraction import load_intents, score_one
from eval.metrics import load_run, measure_run
from eval.scenarios import SCENARIOS as SIM_SCENARIOS
from eval.testbed_scenarios import QUIET, SCENARIOS, SEGMENTS, STEADY, SURGE, TESTBED_CAPACITY_MBPS
from sim.cell import CellConfig
from sim.scenarios import s1_spike_traffic, s2_steady_traffic
from verifier.merge import ActiveIntent
from verifier.verify import VerifierState, verify
from xapps.kpm_monitor import cell_info

ROUNDING = 0.5 / TESTBED_CAPACITY_MBPS + 1e-9
ALL_INTENTS = [i for s in SCENARIOS.values() for i in s.intents]


def objective_for(intent, now: float):
    return objective_from_decision({}, intent.expected["min_thr_mbps"], intent.max_latency_ms, intent.expected["duration_s"], now)


def test_the_testbed_reports_the_capacity_it_measured() -> None:
    assert cell_info().capacity_mbps == TESTBED_CAPACITY_MBPS == 79.0


@pytest.mark.parametrize("intent", ALL_INTENTS, ids=lambda i: i.text[:25])
def test_each_goal_fits_alone_at_the_testbeds_capacity_and_is_read_as_stated(intent) -> None:
    decision = verify(objective_for(intent, intent.at_s), VerifierState(now_s=intent.at_s, capacity_mbps=TESTBED_CAPACITY_MBPS))
    assert decision.accepted, decision.reason
    assert score_one(intent.expected, parse_facts(intent.text))["correct"]


def test_s3_second_intent_conflicts_with_the_first() -> None:
    first, second = SCENARIOS["s3"].intents
    active = (ActiveIntent(intent_id="a", issued_at=60.0, objective=objective_for(first, 60.0)),)
    state = VerifierState(now_s=120.0, capacity_mbps=TESTBED_CAPACITY_MBPS, active=active)
    decision = verify(objective_for(second, 120.0), state)
    assert (decision.accepted, decision.check) == (False, "conflict")


def test_traffic_and_targets_keep_the_simulators_fractions() -> None:
    capacity = CellConfig().capacity_mbps
    spike = s1_spike_traffic(capacity)
    for slice_name, rate in QUIET.items():
        assert rate / TESTBED_CAPACITY_MBPS == pytest.approx(spike[slice_name].base_mbps / capacity, abs=ROUNDING)
    assert SURGE["urllc"] / TESTBED_CAPACITY_MBPS == pytest.approx(spike["urllc"].peak_mbps / capacity, abs=ROUNDING)
    steady = s2_steady_traffic(capacity)
    for slice_name, rate in STEADY.items():
        assert rate / TESTBED_CAPACITY_MBPS == pytest.approx(steady[slice_name].base_mbps / capacity, abs=ROUNDING)
    for name, scenario in SCENARIOS.items():
        for mine, theirs in zip(scenario.intents, SIM_SCENARIOS[name].intents, strict=True):
            assert mine.at_s == theirs.at_s and mine.expected["duration_s"] == theirs.expected["duration_s"]
            assert mine.max_latency_ms == theirs.max_latency_ms  # a latency bound isn't a share of anything
            for slice_name, target in mine.targets.items():
                assert target / TESTBED_CAPACITY_MBPS == pytest.approx(theirs.targets[slice_name] / capacity, abs=ROUNDING)


def test_segments_cover_each_scenarios_time() -> None:
    for name, scenario in SCENARIOS.items():
        assert sum(seconds for seconds, _ in SEGMENTS[name]) == scenario.seconds


def test_no_scenario_text_comes_from_the_held_out_split() -> None:
    assert not {r["text"] for r in load_intents("heldout")} & {i.text for i in ALL_INTENTS}


def test_the_order_rotates_and_covers_all_twelve_runs() -> None:
    orders = [T.order(r) for r in (1, 2, 3)]
    assert all(len(set(o)) == 12 for o in orders)
    assert orders[0] != orders[1] != orders[2] and set(orders[0]) == set(orders[1])
    assert orders[0][0] != orders[1][0] != orders[2][0]


def test_the_matrix_refuses_uncommitted_code() -> None:
    T.guard_final("abc1234")
    with pytest.raises(SystemExit, match="commit"):
        T.guard_final("abc1234+uncommitted")


def test_only_the_controllers_without_a_policy_use_the_baseline() -> None:
    assert [T.runner_args(c) for c in T.CONTROLLERS] == ["--baseline equal", "", "--baseline equal", ""]


def test_the_image_and_compose_pass_the_runners_arguments() -> None:
    assert "$RUNNER_ARGS" in Path("xapps/rl/Dockerfile").read_text()
    service = yaml.safe_load(Path("testbed/rfsim-flexric/docker-compose.yml").read_text())["services"]["rl-xapp"]
    assert "RUNNER_ARGS" in service["environment"]


# ---------------------------------------------------------------- validity: machinery, never a score


def test_a_clean_run_is_valid_whatever_it_scored() -> None:
    assert T.problems([{"event": "kpi"}], {"gnb": 0}, {"gnb": 0}, []) == []


@pytest.mark.parametrize("events, after, errors, expected", [
    ([{"event": "xapp_exited"}], {"gnb": 0}, [], "xApp exited"),
    ([{"event": "kpm_stale"}], {"gnb": 0}, [], "stale"),
    ([], {"gnb": 1}, [], "gnb restarted"),
    ([], {"gnb": 0}, ["seg0_embb.json: no iperf3 result"], "no iperf3"),
])
def test_each_invalid_case(events, after, errors, expected) -> None:
    assert expected in " ".join(T.problems(events, {"gnb": 0}, after, errors))


def test_iperf_failures_and_received_rate(tmp_path) -> None:
    good, bad, broken = tmp_path / "a.json", tmp_path / "b.json", tmp_path / "c.json"
    good.write_text(json.dumps({"end": {"sum": {"bits_per_second": 20e6, "lost_percent": 50.0, "seconds": 10.0}}}))
    bad.write_text(json.dumps({"error": "unable to connect"}))
    broken.write_text("")
    assert T.iperf_failure(good) is None
    assert "unable to connect" in T.iperf_failure(bad) and "no iperf3 result" in T.iperf_failure(broken)
    assert T.iperf_mbps([good]) == pytest.approx(10.0)


# ---------------------------------------------------------------- the metrics read a rebased run


def test_a_rebased_run_gives_the_metrics_their_clock(tmp_path) -> None:
    t0 = 1_800_000_000.0
    scenario = SCENARIOS["s2"]
    work = tmp_path
    rows = ["ts,slice,dl_thr_mbps,offered_load_mbps,latency_ms,prb_share_alloc,policy_id"]
    for t in np.arange(1.0, 361.0):
        met = t > 130  # the bound is met from 10 s after the submission at t=120
        rows.append(f"{t},urllc,16.5,16.5,{3.0 if met else 9.0},0.4,{1 if met else ''}")
        rows.append(f"{t},embb,20.0,34.5,100.0,0.4,{1 if met else ''}")
        rows.append(f"{t},be,14.0,14.0,50.0,0.2,{1 if met else ''}")
    (work / "kpi.csv").write_text("\n".join(rows) + "\n")
    events = T.rebase_events([{"event": "received", "ts_wall": t0 + 120, "ts_backend": t0 + 120.0, "intent_id": "i0001", "data": {"text": scenario.intents[0].text}},
                              {"event": "applied", "ts_wall": t0 + 124, "ts_backend": t0 + 124.0, "intent_id": "i0001", "data": {}},
                              {"event": "note", "ts_wall": t0, "ts_backend": None, "data": {}}], t0)
    (work / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    assert load_run(work)[0]["urllc"]["t"][0] == 1.0
    row = measure_run(work, scenario)["intents"][0]
    assert row["intent_met"] == pytest.approx(170 / 180)
    assert row["reaction_s"] == 10.0
    assert row["first_apply_s"] == 4.0


def test_reaction_is_never_negative_on_a_clock_that_is_not_whole_seconds() -> None:
    from eval.metrics import reaction_s

    t = np.arange(0.17, 12.0)  # the testbed's first report arrives 0.17 s after the scenario starts
    assert reaction_s(t, np.ones(len(t), bool), 0.0, 12.0) == 0.0


def test_s1s_reaction_clock_starts_when_the_spike_really_started() -> None:
    events = [{"event": "scenario_start", "data": {"t0": 1000.0}},
              {"event": "segment_start", "data": {"segment": 0, "wall": 1000.0}},
              {"event": "segment_start", "data": {"segment": 1, "wall": 1123.5}}]
    assert T.scenario_as_played("s1", events).intents[0].reaction_from_s == 123.5
    assert T.scenario_as_played("s2", events) is SCENARIOS["s2"]


# ---------------------------------------------------------------- M08d: the fixes


def test_s3s_rejection_rests_on_the_latency_margin_and_the_what_if_removes_it() -> None:
    first, second = SCENARIOS["s3"].intents
    active = (ActiveIntent(intent_id="a", issued_at=60.0, objective=objective_for(first, 60.0)),)
    told = VerifierState(now_s=120.0, capacity_mbps=TESTBED_CAPACITY_MBPS, active=active, urllc_base_latency_ms=T.WHAT_IF_BASE_LATENCY_MS)
    assert verify(objective_for(second, 120.0), told).accepted


def test_the_capacity_rule_reads_condition_c_and_keeps_69_when_close(monkeypatch) -> None:
    sys.path.insert(0, "testbed")
    import measure_capacity_m08d as M

    row = lambda condition, total: {"condition": condition, "kpm_total_mbps": total}  # noqa: E731
    assert M.capacity_from([row("A", 50.0), row("C", 80.2), row("C", 73.5), row("C", 78.9)]) == 79.0
    assert M.capacity_from([row("C", 71.9)]) == 69.0


def test_each_segment_starts_on_the_clock_on_its_own_port(monkeypatch, tmp_path) -> None:
    started, waited = [], []

    class Proc:
        def wait(self) -> None:
            waited.append(len(started))

    sleeps = []
    monkeypatch.setattr(T.time, "sleep", sleeps.append)
    monkeypatch.setattr(T.traffic, "iperf", lambda name, rate, seconds, out, segment=0: (started.append((name, segment)), out.write_text("{}"), Proc())[2])
    monkeypatch.setattr(T.traffic, "ping", lambda name, seconds, out: Proc())
    monkeypatch.setattr(T, "log_event", lambda *a, **k: None)
    errors: list[str] = []
    db_path = tmp_path / "ric.db"
    T.play_segments(T.TestbedRun("dev", "s1", "static", 1), tmp_path, 1.0, str(db_path), errors, T.time.time())
    assert [s for _, s in started] == [0, 0, 0, 1, 1, 1, 2, 2, 2]
    assert sleeps[1] == pytest.approx(120.0, abs=2.0) and sleeps[2] == pytest.approx(300.0, abs=2.0)
    assert waited[0] == len(started)  # nothing waited on before all the segments had started
    assert errors == []


def test_the_seconds_after_an_intent_are_counted_and_only_with_a_bound_and_a_tail() -> None:
    from eval.metrics import over_bound_after

    t = np.arange(1.0, 361.0)
    latency = np.where((t > 300) & (t <= 325), 2000.0, 0.5)
    kpis = {"urllc": {"t": t, "lat": latency}}
    s1, s2, s3 = SCENARIOS["s1"].intents[0], SCENARIOS["s2"].intents[0], SCENARIOS["s3"].intents[0]
    assert over_bound_after(kpis, s2) == 25
    assert over_bound_after(kpis, s1) is None  # the run ends with this intent
    assert over_bound_after(kpis, s3) is None  # no latency bound


def test_a_second_port_per_segment_keeps_slices_apart() -> None:
    ports = {spec["port"] + T.traffic.SEGMENT_PORT_STEP * segment for spec in T.traffic.SLICES.values() for segment in range(T.MAX_SEGMENTS)}
    assert len(ports) == len(T.traffic.SLICES) * T.MAX_SEGMENTS


def test_played_load_fills_only_missing_ticks():
    """M08d's correction for the offered-load gap: missing ticks take the rate iperf3 was playing, logged ticks stay as logged."""
    from eval.plot_testbed_m08d import with_played_load
    t = np.array([50.0, 100.0, 130.0])
    kpis = {s: {"t": t, "load": np.array([np.nan, np.nan, 99.0])} for s in ("embb", "urllc", "be")}
    filled = with_played_load(kpis, "s3")
    assert filled["be"]["load"].tolist() == [QUIET["be"], QUIET["be"], 99.0]
    assert filled["urllc"]["load"].tolist() == [QUIET["urllc"], QUIET["urllc"], 99.0]
