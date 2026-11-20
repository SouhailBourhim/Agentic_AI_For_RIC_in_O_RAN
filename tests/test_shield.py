"""M11d: the shielded RL executor. With a URLLC latency bound, URLLC never gets less than the formula's share and the
RL splits the rest in its own ratio; with no bound, the RL's split stands exactly."""

import numpy as np
import pytest

from backends.faults import FaultyBackend
from eval.generator import dev_instances
from eval.harness import RunSpec, make_policy, run_once
from rl.baselines import FormulaEstPolicy
from rl.runner import RESUME_AFTER_TICKS, Runner
from rl.shield import ShieldedPolicy, shield, urllc_bound_ms
from sim.cell import CellConfig
from sim.env import KPI_FEATURES_PER_SLICE, LATENCY_SCALE_MS
from store import db
from verifier.objective import SLICES

BOUND_INDEX = KPI_FEATURES_PER_SLICE * len(SLICES) + 2 * len(SLICES)


def observation(rng: np.random.Generator, bound_ms: float) -> np.ndarray:
    obs = rng.uniform(0.0, 1.0, 34).astype(np.float32)
    obs[BOUND_INDEX] = bound_ms / LATENCY_SCALE_MS
    return obs


class Fixed:
    """A stand-in executor: always the same split, counting its calls."""

    def __init__(self, split) -> None:
        self.split, self.calls, self.restarted = np.array(split, dtype=np.float64), 0, None

    def shares(self, _obs):
        self.calls += 1
        return self.split

    def restart_from(self, split) -> None:
        self.restarted = np.array(split)


def test_the_bound_is_read_where_the_formula_reads_it() -> None:
    rng = np.random.default_rng(0)
    assert urllc_bound_ms(observation(rng, 10.0)) == pytest.approx(10.0)
    assert urllc_bound_ms(observation(rng, 0.0)) == 0.0


def test_no_bound_returns_the_rls_split_exactly() -> None:
    rng = np.random.default_rng(1)
    for _ in range(200):
        rl = Fixed(rng.dirichlet(np.ones(3)))
        p = ShieldedPolicy(rl, Fixed([0.1, 0.8, 0.1]))
        out = p.shares(observation(rng, 0.0))
        assert out is rl.split and not p.last_shielded


def test_with_a_bound_urllc_is_raised_to_the_formulas_share_keeping_the_rls_ratio() -> None:
    out, changed = shield(np.array([0.6, 0.2, 0.2]), np.array([0.3, 0.5, 0.2]), 10.0)
    assert changed
    assert out == pytest.approx([0.375, 0.5, 0.125])  # eMBB:BE stays 3:1, URLLC 0.5
    assert out.sum() == pytest.approx(1.0)


def test_with_a_bound_a_larger_rl_share_is_left_alone() -> None:
    rl = np.array([0.2, 0.6, 0.2])
    out, changed = shield(rl, np.array([0.3, 0.5, 0.2]), 10.0)
    assert not changed and np.array_equal(out, rl)


def test_the_rest_is_split_equally_when_the_rl_gave_it_nothing() -> None:
    out, _ = shield(np.array([0.0, 0.0, 0.0]), np.array([0.3, 0.4, 0.3]), 10.0)
    assert out == pytest.approx([0.3, 0.4, 0.3])


def test_shares_are_valid_on_random_inputs() -> None:
    rng = np.random.default_rng(2)
    for _ in range(500):
        out, _ = shield(rng.dirichlet(np.ones(3)), rng.dirichlet(np.ones(3)), float(rng.choice([0.0, 5.0, 20.0])))
        assert out.min() >= 0.0 and out.sum() == pytest.approx(1.0)


def test_the_formula_is_called_every_tick_bound_or_not() -> None:
    formula = Fixed([0.3, 0.4, 0.3])
    p = ShieldedPolicy(Fixed([0.6, 0.2, 0.2]), formula)
    rng = np.random.default_rng(3)
    for bound in (0.0, 10.0, 0.0, 20.0):
        p.shares(observation(rng, bound))
    assert formula.calls == 4


def test_restart_from_reaches_the_rl() -> None:
    rl = Fixed([0.6, 0.2, 0.2])
    ShieldedPolicy(rl, Fixed([0.3, 0.4, 0.3])).restart_from(np.array([1 / 3, 1 / 3, 1 / 3]))
    assert rl.restarted == pytest.approx([1 / 3] * 3)


def test_the_runner_restarts_a_shielded_policy_after_the_safe_state(ric_db) -> None:
    rl = Fixed([0.6, 0.3, 0.1])
    p = ShieldedPolicy(rl, FormulaEstPolicy(5.0))
    r = Runner(FaultyBackend(stale={2}), p, db.connect(timeout_s=0.0))
    for _ in range(2 + 1 + RESUME_AFTER_TICKS):
        r.tick()
    assert rl.restarted == pytest.approx([1 / 3] * 3)  # the equal split it was holding


def test_the_harness_builds_and_plays_shielded_rl(ric_db, tiny_policy_file, tmp_path) -> None:
    inst = dev_instances(("f3",), 1)[0]
    spec = RunSpec("dev", inst.name, "scripted", inst.seed, executor="shielded_rl")
    cfg = CellConfig()
    assert isinstance(make_policy(spec, None, inst.scenario(), cfg, tiny_policy_file), ShieldedPolicy)
    work = run_once(spec, inst.scenario(), tmp_path / "run", tiny_policy_file, cfg=cfg, traffic=inst.traffic(0.0))
    assert (work / "run.json").exists()
