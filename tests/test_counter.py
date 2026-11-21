"""M13: the verifier's offer after a rejection (verifier/counter.py), and where it goes.

The offer is information: the largest scaled-down version of a rejected proposal that verify() accepts in a cell
OFFER_MARGIN smaller and in the real one. These tests check its properties on fixed cases and on many random draws,
that the agent shows it only when told to, that the scripted operator applies it only through verify() and
apply_intent (invariant 3), and that M13's simulator set does what the pre-registration says.
"""

import json
import math

import numpy as np
import pytest

from agent.apply import decision_dict
from agent.llm import FakeLLM
from agent.intents import status
from conftest import run_agent, start_cell
from eval import generator
from eval.harness import RunSpec, apply_scripted
from eval.scenarios import ScenarioIntent, facts
from store import a1_policy
from store.events import read_events
from verifier.counter import OFFER_CHECKS, OFFER_MARGIN, offer_text, scaled, with_offer
from verifier.merge import ActiveIntent
from verifier.objective import SLICES, RLObjective
from verifier.verify import VerifierState, verify

WEIGHTS = {"embb": 0.4, "urllc": 0.3, "be": 0.3}


def s3_case() -> tuple[RLObjective, VerifierState]:
    """The testbed's S3 on 2026-11-21: eMBB and BE guaranteed 23 Mb/s, then URLLC asks 23 Mb/s under 5 ms."""
    first = RLObjective(weights=WEIGHTS, min_thr_mbps={"embb": 23, "be": 23}, valid_from=60, valid_until=360)
    state = VerifierState(now_s=120, capacity_mbps=79, urllc_base_latency_ms=2.0,
                          active=(ActiveIntent(intent_id="i0001", issued_at=60, objective=first),))
    new = RLObjective(weights=WEIGHTS, min_thr_mbps={"urllc": 23}, max_latency_ms=5, valid_from=120, valid_until=360)
    return new, state


def smaller(state: VerifierState) -> VerifierState:
    return state.model_copy(update={"capacity_mbps": state.capacity_mbps * (1 - OFFER_MARGIN)})


def test_the_margin_is_the_lowest_capacity_measured() -> None:
    assert 79 * (1 - OFFER_MARGIN) == pytest.approx(73.5)


def test_s3_gets_16_mbps_under_the_same_bound() -> None:
    new, state = s3_case()
    decision = with_offer(verify(new, state), new, state)
    assert decision.check == "conflict" and not decision.accepted
    assert decision.offer.min_thr_mbps.urllc == 16.0 and decision.offer.max_latency_ms == 5.0
    assert decision.offer.fraction == 0.73
    assert offer_text(decision.offer).startswith("what fits now: urllc 16 Mb/s with latency below 5 ms")
    candidate = new.model_copy(update={"min_thr_mbps": decision.offer.min_thr_mbps})
    assert verify(candidate, state).accepted and verify(candidate, smaller(state)).accepted
    bigger = RLObjective.model_validate({**new.model_dump(), "min_thr_mbps": {"urllc": 17.0}})
    assert not verify(bigger, smaller(state)).accepted


def test_an_accepted_decision_is_left_alone() -> None:
    new, state = s3_case()
    small = RLObjective.model_validate({**new.model_dump(), "min_thr_mbps": {"urllc": 10.0}})
    decision = verify(small, state)
    assert decision.accepted and with_offer(decision, small, state) is decision and decision.offer is None


@pytest.mark.parametrize("change, check", [
    ({"max_latency_ms": 1.0}, "latency_physics"),  # at or under the empty slice's latency: no target helps
    ({"max_latency_ms": 60.0}, "training_range"),  # a bound outside 5-50 ms: scaling targets doesn't fix it
    ({"valid_until": 125.0}, "window"),  # 5 s: too short
])
def test_no_offer_when_scaling_targets_cannot_help(change, check) -> None:
    new, state = s3_case()
    bad = RLObjective.model_validate({**new.model_dump(), **change})
    decision = with_offer(verify(bad, state), bad, state)
    assert decision.check == check and decision.offer is None


def test_no_offer_for_a_step_made_by_the_weights() -> None:
    new, state = s3_case()
    jumpy = RLObjective.model_validate({**new.model_dump(), "min_thr_mbps": {"urllc": 5.0},
                                        "weights": {"embb": 0.0, "urllc": 1.0, "be": 0.0}})
    decision = with_offer(verify(jumpy, state), jumpy, state)
    assert decision.check == "step" and "weight" in decision.reason and decision.offer is None


def random_case(rng: np.random.Generator) -> tuple[RLObjective, VerifierState]:
    """An active intent and a new proposal, both with random targets, a random bound and a random cell."""
    def objective(start: float, n: int) -> RLObjective:
        chosen = rng.choice(SLICES, size=n, replace=False)
        bound = float(rng.choice([5, 8, 10, 20, 50])) if "urllc" in chosen and rng.random() < 0.6 else None
        return RLObjective(weights=WEIGHTS, min_thr_mbps={s: float(rng.integers(1, 45)) for s in chosen},
                           max_latency_ms=bound, valid_from=start, valid_until=start + 300)
    active = () if rng.random() < 0.3 else (ActiveIntent(intent_id="a", issued_at=0, objective=objective(0, int(rng.integers(1, 3)))),)
    state = VerifierState(now_s=50, capacity_mbps=float(rng.uniform(60, 90)),
                          urllc_base_latency_ms=float(rng.choice([0.592, 2.0])), active=active)
    return objective(50, int(rng.integers(1, 3))), state


def test_offer_properties_on_many_random_cases() -> None:
    """Every offer passes both checks, scales each stated target by its fraction (rounded down, >= 1 Mb/s), keeps the
    bound, raises nothing, and no larger percentage would have passed. Rejections without an offer had no candidate."""
    rng = np.random.default_rng(13)
    seen = {"offer": 0, "none": 0}
    for _ in range(400):
        new, state = random_case(rng)
        decision = with_offer(verify(new, state), new, state)
        if decision.accepted:
            assert decision.offer is None
            continue
        offer = decision.offer
        if offer is None:
            seen["none"] += 1
            if decision.check in OFFER_CHECKS:  # a few candidates: none fits
                for p in (90, 50, 10):
                    c = scaled(new, p)
                    assert c is None or not (verify(c, smaller(state)).accepted and verify(c, state).accepted)
            continue
        seen["offer"] += 1
        assert decision.check in OFFER_CHECKS and offer.max_latency_ms == new.max_latency_ms
        percent = round(offer.fraction * 100)
        for s in SLICES:
            stated, offered = getattr(new.min_thr_mbps, s), getattr(offer.min_thr_mbps, s)
            assert offered == (math.floor(stated * percent / 100) if stated > 0 else 0.0)
            assert offered <= stated and (stated == 0 or offered >= 1)
        candidate = new.model_copy(update={"min_thr_mbps": offer.min_thr_mbps})
        assert verify(candidate, state).accepted and verify(candidate, smaller(state)).accepted
        for p in range(percent + 1, 100):
            c = scaled(new, p)
            assert c is None or c == candidate or not (verify(c, smaller(state)).accepted and verify(c, state).accepted)
    assert seen["offer"] > 50 and seen["none"] > 10  # both cases are exercised


def test_the_offer_travels_with_the_decision() -> None:
    new, state = s3_case()
    stored = decision_dict(with_offer(verify(new, state), new, state))
    assert stored["offer"]["min_thr_mbps"]["urllc"] == 16.0 and json.dumps(stored)
    assert "offer" not in decision_dict(verify(new, state))  # verify() alone never sets one


# ------------------------------------------------------------------ the agent


@pytest.fixture
def cell(ric_db, tiny_policy_file):
    runner = start_cell(ric_db, tiny_policy_file)
    runner.run(10)
    return runner


def parse(urllc: float) -> str:
    return json.dumps({"understood": True, "clarification": None, "priority": None,
                       "min_thr_mbps": {"embb": None, "urllc": urllc, "be": None}, "max_latency_ms": None,
                       "duration_s": 600.0})


def call(urllc: float):
    from agent.llm import tool_call
    return tool_call("set_rl_objective", importance={}, min_thr_mbps={"embb": None, "urllc": urllc, "be": None},
                     max_latency_ms=None, duration_s=600.0)


TEXT = "garantir 60 Mb/s au slice URLLC pendant 10 minutes"


def test_offers_are_off_by_default(monkeypatch) -> None:
    """The testbed S3 check (DECISIONS 2026-11-21) found the offer starving the guarantee in force: off until M13b."""
    from agent.tools import Settings
    from agent.worker import settings_from_env

    monkeypatch.delenv("RIC_OFFERS", raising=False)
    assert Settings().offers is False
    assert settings_from_env().offers is False


@pytest.mark.parametrize("offers", [False, True])
def test_eval_mode_shows_the_offer_to_the_model_only_when_on(ric_db, cell, offers) -> None:
    llm = FakeLLM([parse(60), call(60.0), call(60.0)])  # the same proposal again: the intent ends
    record = run_agent(ric_db, TEXT, "eval", llm=llm, runner=cell, observe_s=5.0, offers=offers)
    second_plan = llm.requests[2]["messages"][1]["content"]
    assert ("the largest that fits: set_rl_objective with" in second_plan) is offers
    assert ("what fits now" in record.reason) is offers
    assert record.verifier["offer"]["min_thr_mbps"]["urllc"] > 0  # always logged and stored, either way
    assert status(ric_db, record.intent_id, 0.0)["verifier"]["offer"] == record.verifier["offer"]  # get_status


def test_the_model_can_take_the_offer(ric_db, cell) -> None:
    """The offer, computed as propose_objective does on this cell, then proposed by the model: verified and applied."""
    from agent.apply import verifier_state
    from agent.schema import objective_from_decision
    state = verifier_state(ric_db)
    big = objective_from_decision({}, {"urllc": 60.0}, None, 600.0, state.now_s)
    urllc = with_offer(verify(big, state), big, state).offer.min_thr_mbps.urllc
    llm = FakeLLM([parse(60), call(60.0), call(urllc), json.dumps({"decision": "done", "reason": "ok"})])
    record = run_agent(ric_db, TEXT, "eval", llm=llm, runner=cell, observe_s=5.0, offers=True)
    assert record.state == "applied" and record.objective.min_thr_mbps.urllc == urllc


def test_demo_mode_still_never_relaxes_but_says_what_fits(ric_db, cell) -> None:
    llm = FakeLLM([parse(60), call(60.0), call(30.0)])
    record = run_agent(ric_db, TEXT, "demo", llm=llm, runner=cell, offers=True)
    assert record.state == "rejected" and "stated values" in record.reason and "what fits now: urllc" in record.reason
    assert a1_policy.current_schedule(ric_db) == []


# ------------------------------------------------------------------ the scripted operator (eval.harness)


def big_intent() -> ScenarioIntent:
    return ScenarioIntent(10.0, "60 Mb/s URLLC", facts(600.0, urllc=60.0), 10.0)


@pytest.mark.parametrize("take", [False, True])
def test_the_scripted_operator_takes_the_offer_through_the_verifier(ric_db, cell, take) -> None:
    apply_scripted(ric_db, big_intent(), None, 0.0, take_offer=take)
    events = read_events()
    taken = [e for e in events if e["event"] == "offer_taken"]
    schedule = a1_policy.current_schedule(ric_db)
    if not take:
        assert taken == [] and schedule == []
        return
    assert len(taken) == 1 and taken[0]["data"]["verifier"]["accepted"]  # verified again before apply_intent
    assert [e["event"] for e in events].index("applied") > [e["event"] for e in events].index("offer_taken")
    assert schedule[0].objective.min_thr_mbps.urllc == taken[0]["data"]["offer"]["min_thr_mbps"]["urllc"]


def test_run_ids_before_m13_are_unchanged() -> None:
    spec = RunSpec("dev", "f1_00", "scripted", 1, executor="v7", monitor="m11b_prime_meets")
    assert spec.run_id == "dev/f1_00/scripted_v7_mon-m11b_prime_meets/seed1"
    assert RunSpec("dev", "f1_00", "scripted", 1, offers=True).run_id.endswith("scripted_offer/seed1")
    assert RunSpec("dev", "f1_00", "scripted", 1, offers=False).run_id.endswith("scripted_nooffer/seed1")


# ------------------------------------------------------------------ M13's simulator set


def test_earlier_families_keep_their_seeds() -> None:
    assert generator.FAMILIES == ("f1", "f2", "f3", "f4", "f5")
    assert generator.seed_of("f5", 3, "dev") == 104_003 and generator.seed_of("f6", 3, "dev") == 105_003
    with pytest.raises(SystemExit):
        generator.generate("f6", 0, "m13_test")


def test_f6_second_intent_never_fits_beside_the_first() -> None:
    """At the told values (79 Mb/s, the calibrated URLLC base latency), on dev and on the test split's seeds."""
    from agent.schema import objective_from_decision
    base = generator.calibration()["base_latency_ms_urllc"]
    for split in ("dev", "m13_test"):
        for i in range(generator.PER_FAMILY):
            first, second = generator.draw("f6", i, split).intents
            o1 = objective_from_decision({}, first.targets, first.max_latency_ms, first.expected["duration_s"], first.at_s)
            assert verify(o1, VerifierState(now_s=first.at_s, capacity_mbps=79, urllc_base_latency_ms=base)).accepted
            state = VerifierState(now_s=second.at_s, capacity_mbps=79, urllc_base_latency_ms=base,
                                  active=(ActiveIntent(intent_id="a", issued_at=first.at_s, objective=o1),))
            o2 = objective_from_decision({}, second.targets, None, second.expected["duration_s"], second.at_s)
            decision = with_offer(verify(o2, state), o2, state)
            assert decision.check == "conflict" and decision.offer is not None


def test_m13_set_is_f4b_and_f6() -> None:
    insts = generator.m13_instances()
    assert [i.name for i in insts[:10]] == [f"f4_{k}" for k in range(30, 40)] and len(insts) == 50
    assert all(i.variant == "b" for i in insts[:10])
