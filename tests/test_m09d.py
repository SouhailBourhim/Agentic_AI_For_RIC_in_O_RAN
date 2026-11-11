"""M09d: the scoring rules do what the pre-registration says, and the test set can be played and scored only once,
from a committed protocol. Nothing here draws a test result: the instances are generated, never played."""

import json

import numpy as np
import pytest

from eval import generator as G
from eval import m09d as D


def test_the_test_instances_are_fixed_and_apart_from_dev() -> None:
    a, b = G.test_instances(), G.test_instances()
    assert a == b and len(a) == len(G.FAMILIES) * G.PER_FAMILY
    assert {i.seed for i in a}.isdisjoint({i.seed for i in G.dev_instances()})
    assert all(i.split == "test" for i in a)
    with pytest.raises(SystemExit):  # the M09b paths still can't reach them
        G.generate("f1", 0, split="test")


def test_the_m09e_sample_is_fixed_two_per_family() -> None:
    sample = D.m09e_sample()
    assert sample == D.m09e_sample() and len(sample) == 8
    assert [f for f, _ in sample] == ["f1", "f1", "f2", "f2", "f3", "f3", "f5", "f5"]
    assert all(0 <= i < G.PER_FAMILY for _, i in sample)


def test_the_bootstrap_resamples_instances_not_intents() -> None:
    # Instance A has three intents at 1.0, instance B one intent at 0.0. A draw of two instances is {A, A} (mean 1),
    # {B, B} (mean 0) or {A, B} (mean 3/4). Resampling intents one by one would also give 1/4, 1/2, ...
    groups = [np.array([1.0, 1.0, 1.0]), np.array([0.0])]
    draws = D.bootstrap_means(groups, resamples=2000)
    assert set(np.round(draws, 6)) <= {0.0, 0.75, 1.0}
    assert D.bootstrap_means(groups, resamples=500).tolist() == D.bootstrap_means(groups, resamples=500).tolist()


def test_interval_levels() -> None:
    draws = np.arange(1, 1001, dtype=float)
    low95, high95 = D.interval(draws, 0.95)
    low99, high99 = D.interval(draws, 0.99)
    assert low99 < low95 < high95 < high99


@pytest.mark.parametrize("kind, low, high, expected", [
    ("superior", 0.001, 0.2, True), ("superior", 0.0, 0.2, False),
    ("noninferior", -0.049, 0.1, True), ("noninferior", -0.05, 0.1, False),
    ("faster", -9.0, -0.1, True), ("faster", -9.0, 0.0, False),
])
def test_the_pass_rules_at_their_boundary(kind, low, high, expected) -> None:
    assert D.verdict(low, high, kind) is expected


def test_the_hypotheses_are_the_registered_ones() -> None:
    table = {h.name: (h.family, h.other, h.kind, h.measure) for h in D.HYPOTHESES}
    assert table == {"P1": ("f1", "static", "superior", "intent_met"), "P2": ("f2", "rl_only", "superior", "intent_met"),
                     "P3a": ("f3", "llm_only_v2", "superior", "intent_met"), "P3b": ("f3", "llm_only_v2", "faster", "reaction"),
                     "P4": ("f3", "formula", "noninferior", "intent_met"), "P5": ("f5", "static", "noninferior", "intent_met")}
    assert D.MARGIN == 0.05 and D.LEVELS == (0.95, 0.99) and D.P6_PRECISION == 0.95


def test_feasible_if_the_oracle_or_the_equal_split_meets_it() -> None:
    assert D.feasible(0.5, 0.95) and D.feasible(0.95, 0.0)
    assert not D.feasible(0.89, 0.5)


def test_a_reaction_never_reached_counts_as_the_whole_window() -> None:
    inst = G.draw("f3", 0, "dev")
    intent = inst.intents[0]
    assert D.reaction_or_window({"reaction_s": "", "intent": 1}, inst) == intent.window[1] - intent.reaction_from_s
    assert D.reaction_or_window({"reaction_s": 7.0, "intent": 1}, inst) == 7.0


def _rows(family, ours, theirs, other="static"):
    rows = []
    for (inst, k), v in ours.items():
        rows.append({"family": family, "controller": "proposed", "instance": inst, "intent": k, "intent_met": v})
    for (inst, k), v in theirs.items():
        rows.append({"family": family, "controller": other, "instance": inst, "intent": k, "intent_met": v})
    return rows


def test_paired_groups_pair_intent_by_intent() -> None:
    rows = _rows("f1", {("a", 1): 1.0, ("a", 2): 0.5, ("b", 1): 0.2}, {("a", 1): 0.0, ("a", 2): 0.5, ("b", 1): 0.4})
    groups = D.paired_groups(rows, D.HYPOTHESES[0], {})
    assert [g.tolist() for g in groups] == [[1.0, 0.0], pytest.approx([-0.2])]


def _event(name, **data):
    return {"event": name, "data": data}


def test_p6c_catches_an_apply_without_an_accepted_verification() -> None:
    inst = G.draw("f1", 0, "dev")
    ok = [_event("reverified", accepted=True), _event("applied"),
          _event("verified_shares", accepted=True, shares=[0.3, 0.4, 0.3]), _event("shares_applied", shares=[0.3, 0.4, 0.3])]
    assert D.unverified_applies(ok, inst) == {"unverified": 0, "over_range": 0}
    twice = [_event("reverified", accepted=True), _event("applied"), _event("applied")]  # one verification, two applies
    assert D.unverified_applies(twice, inst)["unverified"] == 1
    rejected = [_event("reverified", accepted=False), _event("applied")]
    assert D.unverified_applies(rejected, inst)["unverified"] == 1
    other = [_event("verified_shares", accepted=True, shares=[0.3, 0.4, 0.3]), _event("shares_applied", shares=[0.9, 0.05, 0.05])]
    assert D.unverified_applies(other, inst)["unverified"] == 1


def test_p6c_catches_an_over_range_request_applied() -> None:
    inst = next(G.draw("f4", i, "dev") for i in range(30, 40))
    assert inst.variant == "b"
    (s, target), = inst.intents[0].targets.items()
    args = {"min_thr_mbps": {x: (target if x == s else None) for x in ("embb", "urllc", "be")}}
    events = [{"event": "agent_step", "data": {"node": "act", "tool": "set_rl_objective", "arguments": args}},
              _event("reverified", accepted=True), _event("applied")]
    assert D.unverified_applies(events, inst) == {"unverified": 0, "over_range": 1}


def test_play_and_score_refuse_an_uncommitted_tree(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(D, "commit", lambda: "abc1234+uncommitted")
    with pytest.raises(SystemExit, match="commit the code"):
        D.play_test(tmp_path / "spent.json")
    with pytest.raises(SystemExit, match="commit the code"):
        D.score(True, tmp_path / "spent.json")


def test_score_needs_final_and_refuses_a_spent_set(tmp_path) -> None:
    with pytest.raises(SystemExit, match="--final"):
        D.score(False)
    register = tmp_path / "spent.json"
    register.write_text(json.dumps({"sets": {D.SPENT_NAME: {"date": "2026-11-12", "commit": "x", "result_file": "r"}}}))
    for call in (lambda: D.play_test(register), lambda: D.play_repeats(register), lambda: D.score(True, register)):
        with pytest.raises(SystemExit, match="already scored"):
            call()
