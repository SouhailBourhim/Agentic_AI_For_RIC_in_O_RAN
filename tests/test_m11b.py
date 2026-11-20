"""M11b and M11b′: the runtime monitor. A target in force missed long enough (the Trigger) while the formula would meet
it hands the segment to the formula, logged and shown; the next segment gives the policy back. The safe state wins.
M11b's trigger is N in a row (the served one, SERVED_TRIGGER); M11b′'s is k of the last W with the scorer's met rule.

Unit tests drive rl.monitor.FallbackMonitor with a stand-in formula (fixed split and need); the end-to-end tests
run the Runner on the simulated cell with a policy that starves URLLC.
"""

import ast
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from prometheus_client import CollectorRegistry, generate_latest

from agent.service import app
from backends.faults import FaultyBackend
from monitoring.exporter import RicCollector
from rl.baselines import FormulaEstPolicy, FormulaPolicy
from eval.generator import generate
from rl.monitor import M11B, M11B_PRIME, M11B_PRIME_STATED, SERVED_TRIGGER, TRIGGERS, FallbackMonitor, missed_slices
from rl.runner import RESUME_AFTER_TICKS, Runner
from sim.cell import CellConfig
from store import a1_policy, db, kpi
from store.events import read_events
from store.kpi import KpiRecord
from verifier.objective import DEFAULT_OBJECTIVE, SLICES, RLObjective, SliceValues

N = int(M11B.missed_s)  # M11b: N in a row, ticks of 1 s
W, K = int(M11B_PRIME.window_s), int(M11B_PRIME.missed_s)  # M11b′: K of the last W
STARVE_URLLC = [0.85, 0.05, 0.10]
OBS = np.zeros(34, dtype=np.float32)  # the stand-in formula doesn't read it
URLLC_10 = RLObjective(weights=SliceValues(embb=0.3, urllc=0.4, be=0.3), min_thr_mbps=SliceValues(urllc=10.0),
                       valid_from=0.0, valid_until=1000.0)


def records(thr: dict[str, float], load: dict[str, float], urllc_latency_ms: float | None = 5.0) -> list[KpiRecord]:
    return [KpiRecord(ts=0.0, source="sim", cell_id="c", slice=s, dl_thr_mbps=thr[s], prb_share_alloc=1 / 3,
                      prb_used=0.0, n_ue=1, latency_ms=urllc_latency_ms if s == "urllc" else 1.0,
                      offered_load_mbps=load[s]) for s in SLICES]


LOAD = {"embb": 30.0, "urllc": 12.0, "be": 20.0}
URLLC_STARVED = records({"embb": 30.0, "urllc": 4.0, "be": 20.0}, LOAD)
ALL_MET = records(LOAD, LOAD)


class StandInFormula:
    """The formula's split and need, fixed: the monitor's rule alone is under test."""

    def __init__(self, shares=(0.4, 0.3, 0.3), need=(0.2, 0.3, 0.1)) -> None:
        self.split, self.last_need = np.array(shares), np.array(need)

    def shares(self, _obs: np.ndarray) -> np.ndarray:
        return self.split


def monitor(formula=None, trigger=M11B) -> FallbackMonitor:
    m = FallbackMonitor(formula or StandInFormula(), n_prb=106, trigger=trigger)
    m.segment(1)
    return m


def steps(m: FallbackMonitor, recs: list[KpiRecord], n: int, rl=STARVE_URLLC, objective=URLLC_10) -> list:
    return [m.step(OBS, objective, recs, LOAD, np.array(rl)) for _ in range(n)]


# ---------------------------------------------------------------- the rule


def test_missed_uses_min_of_target_and_load_with_m10d_tolerance() -> None:
    assert missed_slices(URLLC_STARVED, URLLC_10, LOAD) == {"urllc": "throughput"}
    # 10 Mb/s target, 9.2 served: within max(1 Mb/s, 5 %) of it, not a miss
    assert missed_slices(records({**LOAD, "urllc": 9.2}, LOAD), URLLC_10, LOAD) == {}
    # an idle slice is never missed: it needs only its load
    idle = {**LOAD, "urllc": 0.5}
    assert missed_slices(records({**LOAD, "urllc": 0.5}, idle), URLLC_10, idle) == {}
    assert missed_slices(URLLC_STARVED, DEFAULT_OBJECTIVE, LOAD) == {}  # no targets: nothing to miss


def test_latency_over_the_bound_is_a_miss_unless_unknown() -> None:
    bounded = URLLC_10.model_copy(update={"min_thr_mbps": SliceValues(), "max_latency_ms": 20.0})
    assert missed_slices(records(LOAD, LOAD, urllc_latency_ms=35.0), bounded, LOAD) == {"urllc": "latency"}
    assert missed_slices(records(LOAD, LOAD, urllc_latency_ms=None), bounded, LOAD) == {}


def test_fires_after_exactly_n_qualifying_ticks() -> None:
    m = monitor()
    out = steps(m, URLLC_STARVED, N)
    assert all(x is None for x in out[:-1]) and out[-1] is not None
    assert m.fired is not None and m.fired.slice == "urllc" and m.fired.missed_s == M11B.missed_s
    assert m.fired.rl_share == 0.05 and m.fired.formula_share == 0.3


def test_m11b_is_n_in_a_row_with_m10ds_tolerance() -> None:
    """M11b's trigger as one case of the window: k = W = N, so one met tick in the window is enough to wait again."""
    assert M11B.window_s == M11B.missed_s == 10.0 and SERVED_TRIGGER is M11B
    assert (M11B.tolerance_mbps, M11B.tolerance_fraction) == (1.0, 0.05)
    assert set(TRIGGERS) == {"m11b", "m11b_prime", "m11b_prime_stated", "m11b_prime_k5", "m11b_prime_intent"}


def test_m11b_prime_fires_at_k_of_w_with_gaps() -> None:
    """Missed every other second: N in a row never fires, K of the last W does, at the K-th qualifying second."""
    m, row = monitor(trigger=M11B_PRIME), monitor()
    out, out_row = [], []
    for t in range(2 * K):
        recs = URLLC_STARVED if t % 2 == 0 else ALL_MET
        out.append(steps(m, recs, 1, objective=URLLC_10)[0])
        out_row.append(steps(row, recs, 1)[0])
    assert all(x is None for x in out_row)
    fired = [t for t, x in enumerate(out) if x is not None]
    assert fired and fired[0] == 2 * (K - 1)  # the K-th miss, on even seconds
    assert m.fired is None and m.active  # fired earlier; the formula keeps executing


def test_m11b_prime_does_not_fire_at_k_minus_one_in_the_window() -> None:
    m = monitor(trigger=M11B_PRIME)
    for t in range(3 * W):  # K - 1 misses in every W seconds: never K
        recs = URLLC_STARVED if t % W < K - 1 else ALL_MET
        assert steps(m, recs, 1)[0] is None


def test_m11b_prime_window_is_cleared_by_hold_and_segment() -> None:
    m = monitor(trigger=M11B_PRIME)
    steps(m, URLLC_STARVED, K - 1)
    m.hold()
    assert m.count == 0 and all(x is None for x in steps(m, URLLC_STARVED, K - 1))
    m.segment(2)
    assert m.count == 0 and steps(m, URLLC_STARVED, 1)[0] is None


def test_m11b_prime_uses_the_scorers_met_rule() -> None:
    """A 0.5 Mb/s shortfall: inside M11b's tolerance, a miss for M11b′ (and for intent_met)."""
    short = records({**LOAD, "urllc": 9.5}, LOAD)
    assert missed_slices(short, URLLC_10, LOAD, M11B) == {}
    assert missed_slices(short, URLLC_10, LOAD, M11B_PRIME) == {"urllc": "throughput"}
    assert missed_slices(records({**LOAD, "urllc": 10.0}, LOAD), URLLC_10, LOAD, M11B_PRIME) == {}


def test_a_stated_only_trigger_ignores_implied_targets() -> None:
    """r3: the stated target is eMBB's; URLLC's target is implied (with_fair_shares), so its miss is not watched."""
    both = URLLC_10.model_copy(update={"min_thr_mbps": SliceValues(embb=25.0, urllc=10.0)})
    assert missed_slices(URLLC_STARVED, both, LOAD, M11B_PRIME_STATED, ("embb",)) == {}
    assert missed_slices(URLLC_STARVED, both, LOAD, M11B_PRIME_STATED, ("urllc",)) == {"urllc": "throughput"}
    assert missed_slices(URLLC_STARVED, both, LOAD, M11B_PRIME_STATED, None) == {"urllc": "throughput"}  # unknown
    assert missed_slices(URLLC_STARVED, both, LOAD, M11B_PRIME, ("embb",)) == {"urllc": "throughput"}  # not stated_only
    bounded = both.model_copy(update={"max_latency_ms": 20.0})  # a bound is always stated: always watched
    assert missed_slices(records(LOAD, LOAD, urllc_latency_ms=35.0), bounded, LOAD, M11B_PRIME_STATED, ("embb",)) == {
        "urllc": "latency"}


def test_the_timetable_records_the_stated_slices(ric_db, tiny_policy_file) -> None:
    """The apply step stores which slices an intent stated a target for, before the implied ones are added."""
    from conftest import run_agent, start_cell

    runner = start_cell(ric_db, tiny_policy_file)
    runner.run(10)
    run_agent(ric_db, "garantir 10 Mb/s au slice URLLC pendant 2 minutes", "eval", 0.0)
    runner.run(10)
    run_agent(ric_db, "garantir 20 Mb/s au best effort pendant 1 min", "eval", 0.0)
    rows = a1_policy.current_schedule(ric_db)
    assert [r.stated_slices for r in rows] == [("urllc", "be"), ("urllc",)]
    assert rows[0].objective.min_thr_mbps.embb > 0  # eMBB's target is there, implied, and not stated
    assert a1_policy.in_force(ric_db, 25.0).stated_slices == ("urllc", "be")


def test_a_row_written_without_stated_slices_reads_as_unknown(ric_db) -> None:
    protect_urllc(ric_db)
    assert a1_policy.in_force(ric_db, 1.0).stated_slices is None


def test_generate_refuses_the_m11b_test_split() -> None:
    with pytest.raises(SystemExit, match="M11b"):
        generate("f1", 0, "m11b_test")


def test_a_met_tick_resets_the_count() -> None:
    m = monitor()
    steps(m, URLLC_STARVED, N - 1)
    steps(m, ALL_MET, 1)
    assert all(x is None for x in steps(m, URLLC_STARVED, N - 1))
    assert steps(m, URLLC_STARVED, 1)[0] is not None


def test_no_fallback_when_the_formula_cannot_meet_it_either() -> None:
    """Overload: the formula cuts URLLC below its own sized need, so switching wouldn't fix the miss."""
    m = monitor(StandInFormula(shares=(0.4, 0.3, 0.3), need=(0.2, 0.5, 0.1)))
    assert all(x is None for x in steps(m, URLLC_STARVED, 3 * N))


def test_no_fallback_when_the_policy_already_gives_the_slice_as_much() -> None:
    """The policy gives URLLC 0.3 and still misses: the channel, not the policy, is the problem."""
    m = monitor()
    assert all(x is None for x in steps(m, URLLC_STARVED, 3 * N, rl=[0.4, 0.3, 0.3]))
    # one PRB more is the least that counts as "more"
    assert all(x is None for x in steps(monitor(), URLLC_STARVED, 3 * N, rl=[0.4, 0.3 - 0.5 / 106, 0.3]))


def test_sticky_for_the_segment_then_back_to_the_policy() -> None:
    m = monitor()
    steps(m, URLLC_STARVED, N)
    assert all(x is not None for x in steps(m, ALL_MET, 5))  # met now, still the formula: no flapping
    assert m.segment(1) is False
    assert m.segment(2) is True  # a new segment: the policy decides again
    assert steps(m, URLLC_STARVED, 1)[0] is None and m.count == 1


def test_a_per_intent_handoff_lasts_across_segments_while_its_intent_is_in_force() -> None:
    from rl.monitor import M11B_PRIME_INTENT

    m = FallbackMonitor(StandInFormula(), n_prb=106, trigger=M11B_PRIME_INTENT)
    m.segment(1, ("i0001",))
    for _ in range(int(M11B_PRIME_INTENT.missed_s)):
        m.step(OBS, URLLC_10, URLLC_STARVED, LOAD, np.array(STARVE_URLLC), None, ("i0001",))
    assert m.active
    assert m.segment(2, ("i0001", "i0002")) is False and m.active  # i0002 joins: the formula keeps the segment
    assert m.segment(3, ("i0002",)) is False and m.active  # i0001 ended; i0002 was taken over with segment 2
    assert m.segment(None, ()) is True and not m.active  # nothing left: back to the policy
    per_segment = monitor()  # M11b's trigger: a new segment always gives the policy back
    steps(per_segment, URLLC_STARVED, N)
    assert per_segment.segment(2) is True


def test_hold_resets_the_count() -> None:
    m = monitor()
    steps(m, URLLC_STARVED, N - 1)
    m.hold()
    assert all(x is None for x in steps(m, URLLC_STARVED, N - 1))


def test_same_inputs_same_decisions() -> None:
    def play() -> list:
        m = FallbackMonitor(FormulaEstPolicy(5.0), n_prb=106)
        m.segment(1)
        rng = np.random.default_rng(3)
        out = []
        for _ in range(60):
            obs = rng.uniform(0, 1, 34).astype(np.float32)
            out.append(m.step(obs, URLLC_10, URLLC_STARVED, LOAD, np.array(STARVE_URLLC)))
        return [None if x is None else tuple(x) for x in out]

    assert play() == play()


def test_the_monitor_module_is_pure() -> None:
    """No store, database, network, agent or LLM import: it decides on what the runner hands it."""
    tree = ast.parse(Path("rl/monitor.py").read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module.split(".")[0])
    assert imported <= {"collections", "dataclasses", "numpy", "rl", "verifier"}
    rl_modules = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module.startswith("rl")}
    assert rl_modules == {"rl.baselines"}


def test_last_need_does_not_change_the_formulas_split() -> None:
    """The one line M11b added to FormulaPolicy only keeps a value: the split is the same as before."""
    rng = np.random.default_rng(0)
    for _ in range(50):
        obs = rng.uniform(0, 1.2, 34).astype(np.float32)
        a, b = FormulaPolicy(5.0), FormulaPolicy(5.0)
        split = a.shares(obs)
        del a.last_need
        assert np.array_equal(split, b.shares(obs))


# ---------------------------------------------------------------- the runner, on the simulated cell


class StarveUrllc:
    def shares(self, _obs: np.ndarray) -> np.ndarray:
        return np.array(STARVE_URLLC)


def protect_urllc(conn, intent_id: str = "i0001", start: float = 0.0, end: float = 100.0) -> None:
    """A segment guaranteeing URLLC its 7 Mb/s (the steady scenario offers 7.4) under a 50 ms bound."""
    objective = RLObjective(weights=SliceValues(embb=0.3, urllc=0.4, be=0.3), min_thr_mbps=SliceValues(urllc=7.0),
                            max_latency_ms=50.0, valid_from=start, valid_until=end)
    a1_policy.write_schedule(conn, [a1_policy.Segment(objective, (intent_id,), {"accepted": True}, "console",
                                                      start, end)], "test", start)


def served_runner(backend=None) -> Runner:
    backend = backend or FaultyBackend()
    cfg = CellConfig()
    return Runner(backend, StarveUrllc(), db.connect(timeout_s=0.0),
                  monitor=FallbackMonitor(FormulaEstPolicy(backend.info().urllc_base_latency_ms), cfg.n_prb))


def test_the_runner_hands_a_missed_intent_to_the_formula(ric_db) -> None:
    protect_urllc(ric_db)
    r = served_runner()
    results = [r.tick() for _ in range(N + 10)]
    fired_at = next(i for i, x in enumerate(results) if x.mode == "fallback")
    assert fired_at >= N - 1  # never before N qualifying seconds
    assert all(x.mode == "normal" and x.shares[1] == pytest.approx(0.05) for x in results[:fired_at])
    assert all(x.mode == "fallback" and x.shares[1] > 0.1 for x in results[fired_at:])
    event = next(e for e in read_events() if e["event"] == "executor_fallback")
    assert event["data"]["slice"] == "urllc" and event["data"]["policy_id"] == 1
    assert event["data"]["intent_ids"] == ["i0001"] and event["data"]["missed_s"] == SERVED_TRIGGER.missed_s
    assert kpi.read_runner_state(ric_db) == {"mode": "fallback", "reason": "target_missed:urllc",
                                             "since_backend": float(fired_at)}
    # the formula's split gets URLLC its load back
    assert r.backend.read_kpis()[1].dl_thr_mbps > 7.0 - 1.0


def test_the_next_segment_gives_the_policy_back(ric_db) -> None:
    protect_urllc(ric_db, end=30.0)
    r = served_runner()
    results = [r.tick() for _ in range(35)]
    assert results[29].mode == "fallback" and results[30].mode == "normal"  # the segment expired at 30 s
    assert results[30].shares[1] == pytest.approx(0.05)
    resume = next(e for e in read_events() if e["event"] == "executor_resume")
    assert resume["ts_backend"] == 30.0 and resume["data"]["policy_id"] is None
    assert kpi.read_runner_state(ric_db)["mode"] == "normal"


def test_the_filter_restarts_from_the_formulas_split(ric_db, tiny_policy_file) -> None:
    """Back to the policy, its filter starts from the split in force (the formula's last), as after M11a's safe state."""
    from rl.policy import SlicePolicy

    protect_urllc(ric_db, end=30.0)
    backend = FaultyBackend()
    policy = SlicePolicy.load(tiny_policy_file, 0.1)
    r = Runner(backend, policy, db.connect(timeout_s=0.0),
               monitor=FallbackMonitor(FormulaEstPolicy(backend.info().urllc_base_latency_ms), CellConfig().n_prb))
    seen = []
    policy.restart_from = lambda split: seen.append(np.array(split))  # noqa: E731  record the call only
    r.monitor.segment(1)
    r.monitor.active, r.executor = True, "formula"  # as if it had fired on segment 1
    results = [r.tick() for _ in range(31)]
    assert results[29].mode == "fallback" and results[30].mode == "normal"
    assert len(seen) == 1 and seen[0] == pytest.approx(np.array(results[29].shares))


def test_the_safe_state_wins_and_resets_the_count(ric_db) -> None:
    protect_urllc(ric_db)
    stale = set(range(N - 5, N - 3))  # two stale seconds before N: the count starts again after them
    r = served_runner(FaultyBackend(stale=stale))
    results = [r.tick() for _ in range(2 * N + 10)]
    safe_until = N - 4 + RESUME_AFTER_TICKS  # first tick back to the policy
    assert [x.mode for x in results[N - 5:safe_until]] == ["safe"] * (1 + RESUME_AFTER_TICKS)
    fired_at = next(i for i, x in enumerate(results) if x.mode == "fallback")
    # The first tick back reads the KPIs of the safe split's tick (URLLC met there): the count starts on the next one.
    assert fired_at == safe_until + N


def test_a_fallback_survives_the_safe_state(ric_db) -> None:
    protect_urllc(ric_db)
    r = served_runner(FaultyBackend(stale={N + 5}))
    results = [r.tick() for _ in range(N + 15)]
    after = [x.mode for x in results[N + 5:]]
    assert after[:RESUME_AFTER_TICKS] == ["safe"] * RESUME_AFTER_TICKS  # the stale second, then 2 of 3 clean ones
    assert set(after[RESUME_AFTER_TICKS:]) == {"fallback"}
    assert kpi.read_runner_state(ric_db)["mode"] == "fallback"


def test_llm_only_rows_and_the_default_objective_are_never_monitored(ric_db) -> None:
    r = served_runner()
    assert all(r.tick().mode == "normal" for _ in range(2 * N))  # default objective: no targets
    shares = (0.85, 0.05, 0.10)
    a1_policy.write_schedule(ric_db, [a1_policy.Segment(URLLC_10, ("i0002",), {"accepted": True}, "console",
                                                        2 * N, 1000.0, mode="direct_share", shares=shares)],
                             "test", 2 * N)
    assert all(r.tick().mode == "normal" for _ in range(2 * N))
    assert not [e for e in read_events() if e["event"] == "executor_fallback"]


def test_status_and_prometheus_show_the_fallback(ric_db) -> None:
    protect_urllc(ric_db)
    r = served_runner()
    r.run(N + 2)
    kpi.write_backend_info(ric_db, r.backend.info(), "starving test policy")
    status = TestClient(app).get("/status").json()["runner"]
    assert status["mode"] == "fallback" and status["reason"] == "target_missed:urllc"
    registry = CollectorRegistry()
    registry.register(RicCollector(db.db_path()))
    text = generate_latest(registry).decode()
    assert 'ric_runner_formula_fallback{reason="target_missed:urllc"} 1.0' in text
    assert 'ric_runner_safe_mode{reason="none"} 0.0' in text


def test_without_a_monitor_the_runner_is_unchanged(ric_db) -> None:
    protect_urllc(ric_db)
    r = Runner(FaultyBackend(), StarveUrllc(), db.connect(timeout_s=0.0))
    assert all(r.tick().shares[1] == pytest.approx(0.05) for _ in range(3 * N))
