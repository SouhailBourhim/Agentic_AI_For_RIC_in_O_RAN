"""M09g's diagnostics: the filtered formula (D1) is the served RL policy's filter around the formula, and its dev runs
go to their own folder, so M09b's runs are never touched."""

import numpy as np
import pytest

from eval import benchmark as B
from eval import generator as G
from rl.baselines import FormulaPolicy, SmoothedPolicy
from sim.cell import smooth_split
from sim.env import build_observation
from verifier.objective import SLICES, RLObjective, SliceValues


def _obs(urllc_load: float) -> np.ndarray:
    from sim.cell import CellConfig

    loads = {"embb": 40.0, "urllc": urllc_load, "be": 20.0}
    objective = RLObjective(weights=SliceValues(embb=0.4, urllc=0.3, be=0.3), min_thr_mbps=SliceValues(urllc=10.0),
                            max_latency_ms=10.0)
    return build_observation(loads, loads, {s: 1.0 for s in SLICES}, {s: 1 / 3 for s in SLICES},
                             {s: 1.0 for s in SLICES}, objective, CellConfig(capacity_mbps=79.0))


def test_the_smoothed_formula_is_the_formula_through_the_served_filter() -> None:
    plain, smoothed = FormulaPolicy(0.6), SmoothedPolicy(FormulaPolicy(0.6), 0.10)
    previous = None
    for load in (10.0, 30.0, 30.0, 5.0, 25.0):
        obs = _obs(load)
        previous = smooth_split(previous, plain.shares(obs), 0.10)
        got = smoothed.shares(obs)
        assert got == pytest.approx(previous.astype(np.float32))
        assert got.sum() == pytest.approx(1.0, abs=1e-6)
    smoothed.reset()  # forgets the filter only: the formula's own 30 s peak is the same in both
    assert smoothed.shares(_obs(10.0)) == pytest.approx(plain.shares(_obs(10.0)))  # first step after a reset: unfiltered


def test_alpha_one_is_the_plain_formula_and_alpha_zero_is_refused() -> None:
    plain, same = FormulaPolicy(0.6), SmoothedPolicy(FormulaPolicy(0.6), 1.0)
    for load in (10.0, 30.0, 5.0):
        assert same.shares(_obs(load)) == pytest.approx(plain.shares(_obs(load)))
    with pytest.raises(ValueError):
        SmoothedPolicy(FormulaPolicy(0.6), 0.0)


def test_the_filtered_formula_runs_in_its_own_folder() -> None:
    inst = G.dev_instances(["f3"], 1)[0]
    assert "/m09g_formula_a10/" in B.work_of(inst, "formula_a10").as_posix()
    assert "/m09b_dev/" in B.work_of(inst, "formula").as_posix()
    assert "/m09c_x/" in B.work_of(inst, "formula_a10", "m09c_x").as_posix()  # an explicit study still wins


def test_d1_pairs_intents_and_reads_the_drop_against_the_gap() -> None:
    from eval import diag_m09g as D

    rows = []
    for k in range(10):
        for c, met in (("proposed", 0.6), ("formula", 1.0), ("formula_a10", 0.8)):
            rows.append({"family": "f3", "instance": f"f3_{k:02d}", "intent": "0", "controller": c, "intent_met": str(met)})
    rows.append({"family": "f3", "instance": "f3_99", "intent": "0", "controller": "formula", "intent_met": "0.0"})  # unpaired
    f3 = next(r for r in D.d1(rows) if r["family"] == "f3")
    assert (f3["n"], f3["drop"], f3["share_of_gap"]) == (10, 0.2, 0.5)
    assert "HOLDS" in D.d1_verdict(D.d1(rows))  # 0.8 <= 0.863 and the drop is always 0.2 > 0


def test_d2_counts_the_seconds_just_short_of_the_target() -> None:
    from eval import diag_m09g as D

    found = {("f2", "proposed"): np.array([-0.5, -0.2, 0.0, 0.3, -3.0]), ("f2", "formula"): np.array([2.0, 3.0, 1.5, 2.5, 4.0])}
    rows = {r["controller"]: r for r in D.d2(found) if r["family"] == "f2"}
    assert (rows["proposed"]["just_short"], rows["proposed"]["short_more"], rows["proposed"]["met"]) == (0.4, 0.2, 0.4)
    assert rows["proposed"]["median_mbps"] == -0.2 and rows["formula"]["just_short"] == 0.0
    assert "SUPPORTS A2" in D.d2_verdict(D.d2(found))


def test_d3_tells_a_saturated_actor_from_a_healthy_one() -> None:
    from rl import diag_saturation as S

    rng = np.random.default_rng(0)
    stuck = S.describe(S.COLLAPSED.stem, np.tile([6.0, 5.0, -7.0], (200, 1)) + rng.normal(0, 0.1, (200, 3)))
    alive = S.describe(S.OBSERVER.stem, rng.normal(0, 1.0, (200, 3)))
    assert stuck["saturated_share"] == 1.0 and stuck["action_spread"] < 0.01 and stuck["mean_action"] == "1.00 1.00 0.00"
    assert alive["saturated_share"] < 0.10 and stuck["along_111_median"] == pytest.approx(4 / 3, abs=0.1)
    assert "SUPPORTS A3" in S.verdict([stuck, alive])
