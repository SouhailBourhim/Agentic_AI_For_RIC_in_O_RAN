"""M13b: the verifier sizes a bounded URLLC target from its traffic when the traffic is above the target.

Why (M13's testbed check, DECISIONS 2026-11-21): the offer "16 Mb/s under 5 ms" passed the verifier while URLLC
carried 31.5 Mb/s; holding 5 ms meant serving all of it, and the guarantee already in force fell from 0.96 to 0.22.
"""

import inspect

import numpy as np
import pytest

from agent.apply import LOAD_WINDOW_S, verifier_state
from eval import generator
from eval.harness import RunSpec
from eval.testbed_scenarios import EXTRA_SCENARIOS, PLAYABLE_SCENARIOS, SCENARIOS, SEGMENTS
from rl.baselines import PEAK_WINDOW_S
from store import kpi
from store.kpi import KpiRecord
from verifier.counter import with_offer
from verifier.merge import ActiveIntent
from verifier.objective import SLICES, RLObjective
from verifier.verify import VerifierState, required_shares, verify

WEIGHTS = {"embb": 0.4, "urllc": 0.3, "be": 0.3}


def s3(load: float | None) -> tuple[RLObjective, VerifierState]:
    """The testbed's S3 at t = 120 s: eMBB and BE guaranteed 23 Mb/s, URLLC asks 23 Mb/s under 5 ms."""
    first = RLObjective(weights=WEIGHTS, min_thr_mbps={"embb": 23, "be": 23}, valid_from=60, valid_until=360)
    state = VerifierState(now_s=120, capacity_mbps=79, urllc_base_latency_ms=2.0, urllc_load_mbps=load,
                          active=(ActiveIntent(intent_id="i0001", issued_at=60, objective=first),))
    new = RLObjective(weights=WEIGHTS, min_thr_mbps={"urllc": 23}, max_latency_ms=5, valid_from=120, valid_until=360)
    return new, state


def test_s3_traffic_above_every_offer_gives_no_offer() -> None:
    new, state = s3(31.5)
    decision = with_offer(verify(new, state), new, state)
    assert decision.check == "conflict" and decision.offer is None
    assert "sized from its traffic, 31.5 Mb/s" in decision.reason and "66%" in decision.reason


def test_s3b_traffic_under_the_offer_keeps_it() -> None:
    new, state = s3(15.0)
    decision = with_offer(verify(new, state), new, state)
    assert decision.check == "conflict" and decision.offer.min_thr_mbps.urllc == 16.0
    assert "sized from its traffic" not in decision.reason  # 15 < 23: the target decides


def test_m13s_offer_is_refused_once_the_traffic_is_known() -> None:
    offer = RLObjective(weights=WEIGHTS, min_thr_mbps={"urllc": 16}, max_latency_ms=5, valid_from=120, valid_until=360)
    assert verify(offer, s3(None)[1]).accepted  # M13: accepted, then starved intent 1 on the testbed
    assert not verify(offer, s3(31.5)[1]).accepted


def random_state(rng: np.random.Generator, load: float | None) -> tuple[RLObjective, VerifierState]:
    def objective(start: float) -> RLObjective:
        chosen = rng.choice(SLICES, size=int(rng.integers(1, 3)), replace=False)
        bound = float(rng.choice([5, 8, 10, 20])) if "urllc" in chosen and rng.random() < 0.7 else None
        return RLObjective(weights=WEIGHTS, min_thr_mbps={s: float(rng.integers(1, 32)) for s in chosen},
                           max_latency_ms=bound, valid_from=start, valid_until=start + 300)
    active = () if rng.random() < 0.3 else (ActiveIntent(intent_id="a", issued_at=0, objective=objective(0)),)
    state = VerifierState(now_s=50, capacity_mbps=float(rng.uniform(60, 90)),
                          urllc_base_latency_ms=float(rng.choice([0.592, 2.0])), active=active, urllc_load_mbps=load)
    return objective(50), state


def test_traffic_at_or_under_every_target_changes_nothing() -> None:
    """Many random states: with no load, or a load at or under every URLLC target, every decision is as before."""
    rng = np.random.default_rng(131)
    for _ in range(400):
        new, blind = random_state(rng, None)
        targets = [new.min_thr_mbps.urllc] + [i.objective.min_thr_mbps.urllc for i in blind.active]
        low = min([t for t in targets if t > 0], default=0.0)
        aware = blind.model_copy(update={"urllc_load_mbps": float(rng.uniform(0, low))})
        assert required_shares(new, aware) == required_shares(new, blind)
        assert with_offer(verify(new, aware), new, aware) == with_offer(verify(new, blind), new, blind)


def test_the_need_follows_the_traffic_only_under_a_bound() -> None:
    new, state = s3(46.0)
    assert required_shares(new, state)["urllc"] == pytest.approx(46.0 / 79 / (1 - 2.0 / 5))
    unbounded = new.model_copy(update={"max_latency_ms": None})
    assert required_shares(unbounded, state)["urllc"] == pytest.approx(23.0 / 79)  # a throughput target: min, not max
    no_target = RLObjective(weights=WEIGHTS, max_latency_ms=5, valid_from=120, valid_until=360)
    assert required_shares(no_target, state)["urllc"] == 0.0  # a bound alone keeps the old sizing (note, §5)


def test_an_active_bounded_urllc_above_its_target_protects_itself() -> None:
    guarantee = RLObjective(weights=WEIGHTS, min_thr_mbps={"urllc": 20}, max_latency_ms=5, valid_from=0, valid_until=300)
    active = (ActiveIntent(intent_id="u", issued_at=0, objective=guarantee),)
    request = RLObjective(weights=WEIGHTS, min_thr_mbps={"embb": 23}, valid_from=50, valid_until=300)
    blind = VerifierState(now_s=50, capacity_mbps=79, urllc_base_latency_ms=2.0, active=active)
    assert verify(request, blind).accepted
    aware = blind.model_copy(update={"urllc_load_mbps": 40.0})
    assert verify(request, aware).check == "conflict"


# ------------------------------------------------------------------ where the load comes from


def record(ts: float, slice: str, thr: float, offered: float | None, measured: float | None = None) -> KpiRecord:
    return KpiRecord(ts=ts, source="sim", cell_id="c", slice=slice, dl_thr_mbps=thr, prb_share_alloc=0.3,
                     prb_used=10, n_ue=1, offered_load_mbps=offered, measured_load_mbps=measured)


def test_peak_load_takes_the_window_and_the_controllers_order(ric_db) -> None:
    assert kpi.peak_load(ric_db, "urllc", 30) is None
    kpi.write_records(ric_db, [record(0, "urllc", 5, 50.0), record(0, "be", 1, 99.0)], None)  # outside the window
    kpi.write_records(ric_db, [record(20, "urllc", 5, None, 12.0)], None)  # no offered load: the measured one
    kpi.write_records(ric_db, [record(40, "urllc", 9, None)], None)  # neither: the throughput
    kpi.write_records(ric_db, [record(45, "urllc", 5, 8.0)], None)
    assert kpi.peak_load(ric_db, "urllc", 30) == 12.0
    assert kpi.peak_load(ric_db, "urllc", 50) == 50.0


def test_the_load_reaches_the_verifier_only_with_the_switch(ric_db, tiny_policy_file, monkeypatch) -> None:
    from conftest import start_cell
    start_cell(ric_db, tiny_policy_file).run(10)
    monkeypatch.delenv("RIC_BOUND_FROM_LOAD", raising=False)
    assert verifier_state(ric_db).urllc_load_mbps is None  # off by default until M13b's result
    monkeypatch.setenv("RIC_BOUND_FROM_LOAD", "1")
    load = verifier_state(ric_db).urllc_load_mbps
    assert load is not None and load == kpi.peak_load(ric_db, "urllc", LOAD_WINDOW_S)


def test_the_window_is_the_formulas() -> None:
    assert LOAD_WINDOW_S == PEAK_WINDOW_S


def test_earlier_runs_replay_load_blind() -> None:
    assert RunSpec("dev", "f1_00", "scripted", 1, offers=True).run_id == "dev/f1_00/scripted_offer/seed1"
    assert RunSpec("dev", "f7_00", "scripted", 1, offers=True, bound_from_load=True).run_id.endswith("_offer_load/seed1")
    from eval import testbed_control
    assert '"RIC_BOUND_FROM_LOAD": "0"' in inspect.getsource(testbed_control.play)


# ------------------------------------------------------------------ M13b's sets


def test_s3b_stays_out_of_earlier_studies() -> None:
    assert "s3b" not in SCENARIOS and "s3b" not in SEGMENTS and PLAYABLE_SCENARIOS["s3b"] is EXTRA_SCENARIOS["s3b"]
    assert EXTRA_SCENARIOS["s3b"].intents == SCENARIOS["s3"].intents


def test_f7_seeds_and_guard() -> None:
    assert generator.FAMILIES == ("f1", "f2", "f3", "f4", "f5")
    assert generator.seed_of("f6", 3, "dev") == 105_003 and generator.seed_of("f7", 3, "dev") == 106_003
    with pytest.raises(SystemExit):
        generator.generate("f7", 0, "m13b_test")
    assert len(generator.m13b_instances()) == 40


def test_f7_is_always_a_load_blind_conflict_with_an_offer() -> None:
    """At the told values, on dev and on the test split's seeds; the load-aware verifier offers only when the
    traffic is at or under what it would offer."""
    from agent.schema import objective_from_decision
    base = generator.calibration()["base_latency_ms_urllc"]
    offered = 0
    for split in ("dev", "m13b_test"):
        for i in range(generator.PER_FAMILY):
            inst = generator.draw("f7", i, split)
            first, second = inst.intents
            o1 = objective_from_decision({}, first.targets, None, first.expected["duration_s"], first.at_s)
            state = VerifierState(now_s=second.at_s, capacity_mbps=79, urllc_base_latency_ms=base,
                                  active=(ActiveIntent(intent_id="a", issued_at=first.at_s, objective=o1),))
            o2 = objective_from_decision({}, second.targets, second.max_latency_ms, second.expected["duration_s"],
                                         second.at_s)
            blind = with_offer(verify(o2, state), o2, state)
            assert blind.check == "conflict" and blind.offer is not None
            traffic = inst.steps["urllc"][-1][1]
            aware_state = state.model_copy(update={"urllc_load_mbps": traffic})
            aware = with_offer(verify(o2, aware_state), o2, aware_state)
            assert aware.check == "conflict"
            if aware.offer is not None:
                offered += 1
                assert traffic <= aware.offer.min_thr_mbps.urllc
    assert offered > 10  # G's subset is not empty
