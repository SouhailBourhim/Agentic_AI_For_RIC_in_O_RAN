"""M09a: the simulator's testbed profile, its calibration rules and the validation rule."""

import numpy as np
import pytest

from backends.sim import SimBackend
from eval import calibrate as C
from eval import validate_sim as V
from eval.sim_profiles import TOLD, TRUE_CAPACITY_MBPS
from eval.sim_profiles import testbed_cell as calibrated_cell  # a name starting with "test" would be collected
from sim.cell import CellConfig, step_slice, step_slices_batch
from sim.traffic import TrafficPattern

FLAT = CellConfig(delay_model="flat", max_backlog_mb=400.0, base_latency_ms={"embb": 0.6, "urllc": 0.6, "be": 0.6})


def test_the_flat_delay_stays_at_base_until_the_slice_saturates_then_grows_with_the_queue() -> None:
    calm = step_slice(0.5, 0.45 * FLAT.capacity_mbps, 0.0, 0.6, FLAT)  # 90 % busy: M/M/1 would give 6 ms
    assert calm.latency_ms == pytest.approx(0.6)
    full = step_slice(0.2, 30.0, 100.0, 0.6, FLAT)  # 15 Mb/s of capacity, 130 Mb waiting
    assert full.backlog_mb == pytest.approx(115.0) and full.latency_ms == pytest.approx(0.6 + 1000 * 115.0 / 15.0)
    assert step_slice(0.5, 0.45 * 75.0, 0.0, 2.0, CellConfig()).latency_ms == pytest.approx(2.0 / (1 - 0.9))  # M01 unchanged


def test_flat_batch_and_single_steps_agree() -> None:
    rng = np.random.default_rng(3)
    for _ in range(200):
        share, load, backlog = rng.uniform(0, 1), rng.uniform(0, 80), rng.uniform(0, 400)
        ref = step_slice(share, load, backlog, 0.6, FLAT)
        got = step_slices_batch(np.array([share]), np.array([load]), np.array([backlog]), np.array([0.6]), FLAT)
        assert got["latency_ms"][0] == pytest.approx(ref.latency_ms) and got["backlog_mb"][0] == pytest.approx(ref.backlog_mb)


def test_the_cell_tells_the_verifier_what_the_study_told_it_not_the_truth() -> None:
    cell = calibrated_cell("m08c", 9000)
    low, high = TRUE_CAPACITY_MBPS["m08c"]
    assert low <= cell.capacity_mbps <= high
    info = SimBackend(cell).info()
    assert (info.capacity_mbps, info.urllc_base_latency_ms) == TOLD["m08c"]
    assert calibrated_cell("m08d", 9000).capacity_mbps == calibrated_cell("m08d", 9000).capacity_mbps  # one value per seed
    assert calibrated_cell("m08d", 9000).capacity_mbps != calibrated_cell("m08d", 9001).capacity_mbps
    assert SimBackend(CellConfig()).info().capacity_mbps == 75.0  # the M01 cell is unchanged


def test_steps_play_segments_and_nothing_before_the_first() -> None:
    p = TrafficPattern("steps", steps=((10.0, 8.0), (120.0, 31.5), (360.0, 0.0)))
    assert [p.mean_load_at(t) for t in (0.0, 10.0, 119.9, 120.0, 359.0, 360.0)] == [0.0, 8.0, 8.0, 31.5, 31.5, 0.0]


def test_m08d_is_replayed_on_its_clock_with_no_gap() -> None:
    urllc = V.replay_traffic("m08d", "s1", "proposed")["urllc"]
    assert urllc.steps == ((0.0, 8.0), (120.0, 31.5), (300.0, 8.0), (360.0, 0.0))


def test_the_calibration_selects_seconds_by_load_never_by_score() -> None:
    k = {"t": np.array([10.0, 30.0, 40.0, 50.0]), "load": np.array([5.0, 5.0, 30.0, 20.0]), "thr": np.array([5.0, 5.0, 20.0, 20.0]),
         "share": np.full(4, 0.33), "lat": np.array([1.0, 0.6, 9000.0, 0.7])}
    assert C.unsaturated(k, 79.0).tolist() == [False, True, False, True]  # t <= 20 is the transient
    assert C.saturated(k).tolist() == [False, False, True, False]


def test_a_panel_fails_only_on_a_clear_reversal() -> None:
    testbed = {"static": 0.34, "rl_only": 0.21, "llm_only": 0.34, "proposed": 0.96}
    assert V.panel_mismatches(testbed, {"static": 0.30, "rl_only": 0.25, "llm_only": 0.37, "proposed": 0.90}) == []  # ties inside 0.05
    assert V.panel_mismatches(testbed, {"static": 0.34, "rl_only": 0.21, "llm_only": 0.34, "proposed": 0.20}) == [
        "static/proposed", "llm_only/proposed"]
