"""M04a: the per-slice minimum PRB share holds exactly, whatever the policy outputs.

One function (sim.cell.apply_min_share) and one constant (verifier.objective.MIN_SLICE_SHARE),
used by the training env, the RL runner, the oracle and the verifier's reserve rule.
"""

import numpy as np
import pytest

from conftest import start_cell
from rl.oracle import GreedyOracle
from sim.cell import apply_min_share, shares_from_action, smooth_split
from sim.env import SlicingEnv
from verifier.objective import DEFAULT_OBJECTIVE, MIN_SLICE_SHARE
from verifier.verify import DEFAULT_CONFIG

FLOOR = MIN_SLICE_SHARE
NASTY = [
    [np.nan, np.nan, np.nan], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [np.inf, 0.0, 0.0],
    [-np.inf, 1.0, 0.0], [-5.0, -1.0, -0.1], [1e300, 1e-300, 0.0], [np.nan, 1.0, 0.0], [0.04, 0.96, 0.0],
]


def assert_valid(split: np.ndarray) -> None:
    assert np.all(np.isfinite(split))
    assert np.all(split >= FLOOR - 1e-12), split
    assert split.sum() == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("raw", NASTY)
def test_floor_holds_for_nasty_outputs(raw: list[float]) -> None:
    assert_valid(apply_min_share(np.array(raw), FLOOR))


def test_floor_holds_for_random_outputs() -> None:
    raw = np.random.default_rng(0).normal(0.0, 2.0, size=(20_000, 3))  # negatives, any scale
    raw[::7, 1] = np.nan
    out = apply_min_share(raw, FLOOR)
    assert np.all(out >= FLOOR - 1e-12) and np.allclose(out.sum(axis=1), 1.0)


def test_split_already_above_the_floor_is_unchanged() -> None:
    splits = np.random.default_rng(1).dirichlet(np.ones(3), size=5000)
    ok = splits[(splits >= FLOOR).all(axis=1)]
    assert np.allclose(apply_min_share(ok, FLOOR), ok, rtol=0.0, atol=1e-15)  # only float rounding


def test_one_split_and_many_give_the_same_result() -> None:
    raw = np.array(NASTY)
    batch = apply_min_share(raw, FLOOR)
    for row, out in zip(raw, batch):
        assert np.allclose(apply_min_share(row, FLOOR), out)


def test_floor_only_takes_from_slices_above_it_in_proportion() -> None:
    """(0.7, 0.3, 0): BE needs 0.05, taken 0.65:0.25 from eMBB and URLLC (their excess over the floor)."""
    out = apply_min_share(np.array([0.7, 0.3, 0.0]), FLOOR)
    assert out == pytest.approx([0.7 - 0.05 * 0.65 / 0.9, 0.3 - 0.05 * 0.25 / 0.9, 0.05])


def test_invalid_floor_is_refused() -> None:
    with pytest.raises(ValueError):
        apply_min_share(np.array([1.0, 0.0, 0.0]), 0.4)  # 3 x 0.4 > 1


def test_one_constant_everywhere() -> None:
    assert DEFAULT_CONFIG.min_share_floor == MIN_SLICE_SHARE == 0.05


def test_env_applies_the_floor_after_the_filter_like_deployment() -> None:
    """Training env: filter (memory before the floor), then floor. Same sequence as SlicePolicy's
    filter followed by the runner's floor."""
    env = SlicingEnv(action_filter_alpha=0.25, min_share=FLOOR)
    env.reset(seed=0, options={"objective": DEFAULT_OBJECTIVE})
    with np.errstate(over="ignore"):  # 1e300 becomes inf in float32: deliberate, one more nasty input
        actions = [np.array(a, dtype=np.float32) for a in NASTY]
    previous = None  # what SlicePolicy keeps
    for action in actions:
        env.step(action)
        previous = smooth_split(previous, shares_from_action(action), 0.25)
        deployed = apply_min_share(previous, FLOOR)
        assert_valid(env.shares)
        assert env.shares == pytest.approx(deployed)


def test_env_without_floor_is_unchanged() -> None:
    env = SlicingEnv()
    env.reset(seed=0, options={"objective": DEFAULT_OBJECTIVE})
    env.step(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    assert env.shares.tolist() == [1.0, 0.0, 0.0]


class BrokenPolicy:
    """A 'policy' returning the nastiest outputs in turn."""

    def __init__(self) -> None:
        self.i = 0

    def shares(self, obs: np.ndarray) -> np.ndarray:
        self.i += 1
        return np.array(NASTY[self.i % len(NASTY)])


def test_runner_enforces_the_floor_whatever_the_policy_says(ric_db, tiny_policy_file) -> None:
    runner = start_cell(ric_db, tiny_policy_file)
    runner.policy = BrokenPolicy()  # type: ignore[assignment]
    for _ in range(3 * len(NASTY)):
        assert_valid(np.array(runner.tick().shares))
    low = ric_db.execute("SELECT MIN(prb_share_alloc) FROM kpi").fetchone()[0]
    assert low >= FLOOR - 1e-12


def test_oracle_with_floor_only_uses_floored_splits() -> None:
    env = SlicingEnv(min_share=FLOOR)
    env.reset(seed=0, options={"objective": DEFAULT_OBJECTIVE})
    oracle = GreedyOracle(DEFAULT_OBJECTIVE, min_share=FLOOR)
    for _ in range(10):
        env.step(oracle.act(env))
        assert_valid(env.shares)
