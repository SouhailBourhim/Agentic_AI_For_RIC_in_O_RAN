"""verify(): table-driven accepted and rejected cases, and the edge cases (M03).

The cell is the simulator's: 75 Mb/s, empty URLLC slice at 2 ms. Now is t = 100 s.
"""

import pytest

from verifier.merge import ActiveIntent
from verifier.objective import RLObjective, SliceValues
from verifier.verify import DEFAULT_CONFIG, VerifierConfig, VerifierState, reverify_after_removal, verify

NOW = 100.0
CAPACITY = 75.0
STATE = VerifierState(now_s=NOW, capacity_mbps=CAPACITY)
# For cases far from the default objective that test another limit than the step.
NO_STEP_LIMIT = VerifierConfig(max_target_step_fraction=1.0)

# "Protect URLLC: 10 Mb/s within 10 ms for 10 minutes" — the base of most cases below.
PROTECT_URLLC = {
    "weights": {"embb": 0.1, "urllc": 0.8, "be": 0.1},
    "min_thr_mbps": {"urllc": 10.0},
    "max_latency_ms": 10.0,
    "valid_from": NOW,
    "valid_until": NOW + 600.0,
}


def proposal(**changes) -> dict:
    """PROTECT_URLLC with some fields replaced."""
    return {**PROTECT_URLLC, **changes}


def active(intent_id: str, issued_at: float, **changes) -> ActiveIntent:
    """An already accepted intent, built from PROTECT_URLLC with some fields replaced."""
    return ActiveIntent(intent_id=intent_id, issued_at=issued_at, objective=RLObjective(**proposal(**changes)))


BE_20 = active("be20", 50.0, weights={"embb": 0.3, "urllc": 0.3, "be": 0.4},
               min_thr_mbps={"be": 20.0}, max_latency_ms=None, valid_from=50.0)


@pytest.mark.parametrize(
    ("prop", "state", "config"),
    [
        pytest.param(proposal(), STATE, DEFAULT_CONFIG, id="protect-urllc"),
        pytest.param(proposal(weights={"embb": 0.4, "urllc": 0.3, "be": 0.3}, min_thr_mbps={},
                              max_latency_ms=None), STATE, DEFAULT_CONFIG,
                     id="default-like"),
        # 30 + 30 + 15 Mb/s = exactly the 75 Mb/s cell (no latency bound, so no URLLC reserve).
        pytest.param(proposal(min_thr_mbps={"embb": 30.0, "urllc": 30.0, "be": 15.0}, max_latency_ms=None),
                     STATE, NO_STEP_LIMIT, id="targets-exactly-on-budget"),
        # Default weights 0.4/0.3/0.3 -> 0.1/0.8/0.1: URLLC moves by exactly 0.5 (float gives 0.5000000000000001).
        pytest.param(proposal(), STATE, DEFAULT_CONFIG, id="weight-step-exactly-limit"),
        pytest.param(proposal(valid_until=NOW + 10.0), STATE, DEFAULT_CONFIG, id="window-exactly-min-duration"),
        pytest.param(proposal(weights={"embb": 0.3, "urllc": 0.3, "be": 0.4 + 5e-8}), STATE,
                     DEFAULT_CONFIG, id="weight-sum-within-tolerance"),
        pytest.param(proposal(min_thr_mbps={"urllc": "10"}), STATE, DEFAULT_CONFIG, id="number-as-string-coerced"),
        pytest.param(proposal(valid_from=0.0), STATE, DEFAULT_CONFIG, id="started-in-the-past-counts-from-now"),
        pytest.param(proposal(max_latency_ms=5.0), STATE, DEFAULT_CONFIG, id="bound-at-training-low-edge"),
        pytest.param(proposal(min_thr_mbps={"urllc": 30.0}, max_latency_ms=None), STATE,
                     NO_STEP_LIMIT, id="target-at-training-high-edge"),
        pytest.param(proposal(min_thr_mbps={"embb": 20.0}),
                     STATE.model_copy(update={"active": (BE_20,)}), DEFAULT_CONFIG, id="compatible-with-active"),
        # 22 + 22 Mb/s plus the old 30 Mb/s on BE would need 106 % of the cell, if it still counted.
        pytest.param(proposal(min_thr_mbps={"embb": 22.0, "urllc": 22.0}),
                     STATE.model_copy(update={"active": (active("old", 0.0, min_thr_mbps={"be": 30.0},
                                                                  valid_from=0.0, valid_until=NOW),)}),
                     DEFAULT_CONFIG, id="expired-active-ignored"),
        # The other intent ends exactly when the new one starts: touching windows do not overlap.
        pytest.param(proposal(min_thr_mbps={"embb": 22.0, "urllc": 22.0}, valid_from=NOW + 300.0,
                              valid_until=NOW + 600.0),
                     STATE.model_copy(update={"active": (active("until300", 50.0, min_thr_mbps={"be": 30.0},
                                                                  max_latency_ms=None, valid_from=50.0,
                                                                  valid_until=NOW + 300.0),)}),
                     DEFAULT_CONFIG, id="touching-windows-do-not-conflict"),
    ],
)
def test_accepted(prop: dict, state: VerifierState, config: VerifierConfig) -> None:
    decision = verify(prop, state, config, new_intent_id="i9")
    assert decision.accepted, decision.reason
    assert decision.check == "ok"
    # merged = what runs now; an intent that starts later is not in it yet.
    starts_now = float(prop["valid_from"]) <= state.now_s
    assert decision.merged is not None and ("i9" in decision.merged.intent_ids) == starts_now


@pytest.mark.parametrize(
    ("prop", "state", "check", "words"),
    [
        pytest.param(proposal(weights={"embb": 0.3, "urllc": 0.3, "be": 0.401}), STATE, "schema",
                     "sum to 1", id="weights-sum-1.001"),
        pytest.param(proposal(min_thr_mbps={"embb": float("inf")}), STATE, "schema", "finite",
                     id="inf-target"),
        pytest.param(proposal(min_thr_mbps={"urllc": float("nan")}), STATE, "schema", "finite",
                     id="nan-target"),
        pytest.param(proposal(priority="high"), STATE, "schema", "priority", id="unknown-field"),
        pytest.param(proposal(weights={"embb": 0.5, "urllc": 0.5, "mmtc": 0.0}), STATE, "schema",
                     "mmtc", id="unknown-slice"),
        pytest.param(proposal(valid_until=None), STATE, "window", "no end", id="open-ended"),
        pytest.param(proposal(valid_from=0.0, valid_until=NOW), STATE, "window", "ended",
                     id="window-already-over"),
        pytest.param(proposal(valid_until=NOW + 9.0), STATE, "window", "minimum", id="window-too-short"),
        pytest.param(proposal(valid_from=0.0, valid_until=NOW + 5.0), STATE, "window", "minimum",
                     id="little-left-of-old-window"),
        pytest.param(proposal(valid_until=NOW + 86_401.0), STATE, "window", "maximum", id="window-too-long"),
        pytest.param(proposal(valid_from=NOW + 3_601.0, valid_until=NOW + 4_000.0), STATE, "window",
                     "starts in", id="starts-too-late"),
        pytest.param(proposal(max_latency_ms=1.5), STATE, "latency_physics", "empty URLLC",
                     id="bound-below-base-latency"),
        pytest.param(proposal(max_latency_ms=2.0), STATE, "latency_physics", "empty URLLC",
                     id="bound-equal-base-latency"),
        pytest.param(proposal(min_thr_mbps={"embb": 30.0, "urllc": 30.0, "be": 16.0}, max_latency_ms=None),
                     STATE, "budget", "more than the cell has", id="budget-over-by-1-mbps"),
        # 30 Mb/s fits alone (40 % of the cell) but within 3 ms it needs 30 / (75 * (1 - 2/3)) = 120 %.
        pytest.param(proposal(min_thr_mbps={"urllc": 30.0}, max_latency_ms=3.0), STATE, "budget",
                     "urllc 120%", id="latency-bound-makes-target-infeasible"),
        # 30 Mb/s eMBB (40 %) + 26 Mb/s URLLC within 5 ms (26 / (75 * 0.6) = 58 %) leave 2 % for BE.
        pytest.param(proposal(min_thr_mbps={"embb": 30.0, "urllc": 26.0}, max_latency_ms=5.0), STATE,
                     "floor", "for be", id="floor-be-starved"),
        # 20 Mb/s within 2.78 ms needs 95 % of the cell: 5 % left for two slices.
        pytest.param(proposal(min_thr_mbps={"urllc": 20.0}, max_latency_ms=2.78), STATE,
                     "floor", "for embb and be", id="floor-two-slices-starved"),
        pytest.param(proposal(min_thr_mbps={"urllc": 30.8}, max_latency_ms=None), STATE, "training_range",
                     "30 Mb/s", id="target-above-training-range"),
        pytest.param(proposal(max_latency_ms=60.0), STATE, "training_range", "5-50 ms",
                     id="bound-above-training-range"),
        pytest.param(proposal(max_latency_ms=4.0), STATE, "training_range", "5-50 ms",
                     id="bound-below-training-range"),
        # Alone: 40 % + 37 %. With BE_20's 27 %: 103 %.
        pytest.param(proposal(min_thr_mbps={"embb": 30.0, "urllc": 22.0}),
                     STATE.model_copy(update={"active": (BE_20,)}), "conflict", "be20",
                     id="conflict-with-active"),
        # The other intent starts later, inside the new window: the conflict is found at its start.
        pytest.param(proposal(min_thr_mbps={"embb": 30.0, "urllc": 22.0}),
                     STATE.model_copy(update={"active": (active("future", 60.0, min_thr_mbps={"be": 20.0},
                                                                  max_latency_ms=None, valid_from=NOW + 200.0,
                                                                  valid_until=NOW + 400.0),)}),
                     "conflict", "t=300", id="conflict-with-future-intent"),
        pytest.param(proposal(weights={"embb": 0.09, "urllc": 0.81, "be": 0.1}), STATE, "step",
                     "urllc weight", id="weight-step-0.51"),
        # Default URLLC 0 -> 25 Mb/s: 25 > 22.5 Mb/s (30 % of 75).
        pytest.param(proposal(min_thr_mbps={"urllc": 25.0}, max_latency_ms=None), STATE, "step",
                     "urllc target", id="target-step-too-big"),
    ],
)
def test_rejected(prop: dict, state: VerifierState, check: str, words: str) -> None:
    decision = verify(prop, state)
    assert not decision.accepted
    assert decision.check == check, decision.reason
    assert words in decision.reason
    assert decision.merged is None


def test_accepted_merge_is_what_runs() -> None:
    """The accepted decision carries the merged objective: max targets, newest weights."""
    state = STATE.model_copy(update={"active": (BE_20,)})
    decision = verify(proposal(min_thr_mbps={"embb": 20.0}), state, new_intent_id="i2")
    merged = decision.merged
    assert merged is not None
    assert merged.intent_ids == ("be20", "i2")
    assert merged.objective.min_thr_mbps.embb == merged.objective.min_thr_mbps.be == 20.0
    assert merged.objective.min_thr_mbps.urllc == pytest.approx(21.0)  # 0.75 of the 46.7 % free, bound's headroom
    assert merged.objective.weights.as_list() == [0.1, 0.8, 0.1]  # the new intent's
    assert merged.objective.max_latency_ms == 10.0


def test_rlobjective_input_is_accepted_like_a_dict() -> None:
    assert verify(RLObjective(**proposal()), STATE) == verify(proposal(), STATE)


def test_first_failing_check_wins() -> None:
    """Infeasible budget AND out of training range: the budget check comes first."""
    decision = verify(proposal(min_thr_mbps={"embb": 40.0, "urllc": 40.0}, max_latency_ms=None), STATE)
    assert decision.check == "budget"


def test_removing_an_intent_keeps_the_merge_feasible() -> None:
    """After any accepted intent expires, the remaining merge is still accepted (it only loosens)."""
    first = verify(proposal(min_thr_mbps={"embb": 20.0}), STATE.model_copy(update={"active": (BE_20,)}),
                   new_intent_id="i2")
    assert first.accepted
    i2 = ActiveIntent(intent_id="i2", issued_at=NOW, objective=RLObjective(**proposal(min_thr_mbps={"embb": 20.0})))
    for remaining in [(BE_20,), (i2,), ()]:
        decision = reverify_after_removal(STATE.model_copy(update={"active": remaining}))
        assert decision.accepted, decision.reason


def test_newest_expiry_skips_step_check() -> None:
    """When the newest intent expires the weights jump back (0.1/0.8/0.1 -> 0.3/0.3/0.4 here,
    a 0.5 URLLC step); an expiry cannot be refused, so reverify accepts."""
    decision = reverify_after_removal(STATE.model_copy(update={"active": (BE_20,)}))
    assert decision.accepted
    assert decision.merged is not None
    assert decision.merged.objective.weights.as_list() == [0.3, 0.3, 0.4]


def test_reverify_rejects_if_cell_shrank() -> None:
    """The only way a re-check fails: the state changed, e.g. capacity recalibrated from 75 to 40 Mb/s."""
    state = VerifierState(now_s=NOW, capacity_mbps=40.0,
                          active=(BE_20, active("i2", NOW, min_thr_mbps={"embb": 20.0})))
    decision = reverify_after_removal(state)
    assert not decision.accepted and decision.check == "conflict"


def test_revision_step_is_measured_from_the_objective_in_force() -> None:
    """A revision raising URLLC from 0.8 to 0.9 is a small step from what runs (0.8) even though,
    without its parent, the default (0.3) would be the baseline (M04b, verify(..., replacing=))."""
    parent = active("i1", 50.0, weights={"embb": 0.1, "urllc": 0.8, "be": 0.1}, valid_from=50.0)
    state = STATE.model_copy(update={"active": (parent,)})
    revision = proposal(weights={"embb": 0.05, "urllc": 0.9, "be": 0.05})
    assert verify(revision, STATE).check == "step"  # as a fresh intent: 0.3 -> 0.9
    decision = verify(revision, state, new_intent_id="i2", replacing="i1")
    assert decision.accepted and decision.merged.intent_ids == ("i2",)  # the parent leaves the merge


# ---------------------------------------------------------------- implied targets (M08a, S3)


def run_objective(active_targets: dict, bound: float | None = None) -> tuple:
    from verifier.merge import ActiveIntent, merge
    from verifier.verify import DEFAULT_CONFIG, VerifierState, required_shares, with_fair_shares

    state = VerifierState(now_s=0.0, capacity_mbps=75.0)
    objective = RLObjective(weights=SliceValues(embb=0.4, urllc=0.3, be=0.3), min_thr_mbps=SliceValues(**active_targets),
                            max_latency_ms=bound, valid_until=100.0)
    merged = merge([ActiveIntent(intent_id="a", issued_at=0.0, objective=objective)], 0.0)
    runs = with_fair_shares(merged, state, DEFAULT_CONFIG)
    return runs, required_shares(runs, state)


def test_a_slice_with_no_target_gets_part_of_what_the_stated_targets_leave() -> None:
    runs, _ = run_objective({"embb": 22.0, "be": 22.0})
    assert runs.min_thr_mbps.embb == 22.0 and runs.min_thr_mbps.be == 22.0  # stated targets untouched
    assert runs.min_thr_mbps.urllc == pytest.approx(22.5)  # 0.75 x 31 Mb/s free, capped by the step limit


def test_implied_targets_share_the_free_prbs_and_never_overfill_the_cell() -> None:
    runs, need = run_objective({"urllc": 18.0}, bound=5.0)  # needs 40 % of the cell
    assert runs.min_thr_mbps.embb == pytest.approx(0.75 * 0.6 / 2 * 75.0)
    assert runs.min_thr_mbps.embb == runs.min_thr_mbps.be
    assert sum(need.values()) <= 1.0 + 1e-9


@pytest.mark.parametrize("bound", [5.0, 10.0, 50.0])
def test_an_implied_urllc_target_counts_the_headroom_of_its_bound(bound) -> None:
    runs, need = run_objective({"embb": 15.0, "be": 15.0}, bound=bound)
    assert 0.0 < runs.min_thr_mbps.urllc and sum(need.values()) <= 1.0 + 1e-9


def test_with_no_intent_active_nothing_is_implied() -> None:
    from verifier.merge import merge
    from verifier.verify import DEFAULT_CONFIG, VerifierState, with_fair_shares

    runs = with_fair_shares(merge([], 0.0), VerifierState(now_s=0.0, capacity_mbps=75.0), DEFAULT_CONFIG)
    assert runs.min_thr_mbps.as_list() == [0.0, 0.0, 0.0]


def test_an_implied_target_never_causes_a_conflict_and_a_stated_conflict_stays_one() -> None:
    from verifier.merge import ActiveIntent
    from verifier.verify import VerifierState

    first = RLObjective(weights=SliceValues(embb=0.4, urllc=0.3, be=0.3), min_thr_mbps=SliceValues(embb=22.0, be=22.0),
                        valid_until=300.0)
    state = VerifierState(now_s=10.0, capacity_mbps=75.0, active=(ActiveIntent(intent_id="a", issued_at=0.0, objective=first),))
    new = lambda urllc, bound=None: RLObjective(  # noqa: E731
        weights=SliceValues(embb=0.4, urllc=0.3, be=0.3), min_thr_mbps=SliceValues(urllc=urllc), max_latency_ms=bound,
        valid_from=10.0, valid_until=300.0)
    assert verify(new(20.0), state).accepted  # fits next to the stated targets, despite URLLC's implied 22.5
    rejected = verify(new(22.0, bound=5.0), state)
    assert (rejected.accepted, rejected.check) == (False, "conflict")


def test_the_decision_shows_what_runs_and_it_is_the_timetables(ric_db, tiny_policy_file) -> None:
    from conftest import run_agent, start_cell

    runner = start_cell(ric_db, tiny_policy_file)
    runner.run(2)
    record = run_agent(ric_db, "garantir 20 Mb/s au slice URLLC pendant 10 minutes", mode="eval", runner=runner)
    row = ric_db.execute("SELECT objective FROM a1_policy ORDER BY policy_id DESC LIMIT 1").fetchone()
    assert record.objective.min_thr_mbps.embb == 0.0  # the intent keeps what the operator stated
    assert '"embb": 0.0' not in row[0]  # what runs has the implied targets
