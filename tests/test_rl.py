"""Fast checks of the M02 code. No real training here: that is done by hand (rl/train.py)."""

from pathlib import Path

import numpy as np
import pytest

from rl.baselines import equal_split, proportional_to_load
from rl.evaluate import episode_metrics
from rl.policy import SlicePolicy
from rl.train import make_model
from sim.env import SlicingEnv
from sim.plot import run_episode
from sim.scenarios import urllc_spike


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A DDPG model trained for 300 steps, with updates starting at step 100.

    Far too short to learn anything; it proves the whole training path runs (replay buffer,
    actor and critic updates, saving).
    """
    model = make_model(SlicingEnv(episode_steps=50), seed=0)
    model.learning_starts = 100
    model.learn(total_timesteps=300)
    path = tmp_path_factory.mktemp("m02") / "tiny.zip"
    model.save(path)
    return path


def test_training_path_runs_and_saves(tiny_model: Path) -> None:
    assert tiny_model.exists() and tiny_model.stat().st_size > 10_000


def test_policy_shares_sum_to_one(tiny_model: Path) -> None:
    policy = SlicePolicy.load(tiny_model)
    obs, _ = SlicingEnv().reset(seed=3)
    action = policy.action(obs)
    assert action.shape == (3,) and np.all((action >= 0) & (action <= 1))
    assert policy.shares(obs).sum() == pytest.approx(1.0)


def test_policy_is_deterministic(tiny_model: Path) -> None:
    policy = SlicePolicy.load(tiny_model)
    obs, _ = SlicingEnv().reset(seed=4)
    assert np.array_equal(policy.action(obs), policy.action(obs))


def test_run_episode_is_reproducible() -> None:
    traffic, objective = urllc_spike()
    a = run_episode(proportional_to_load, traffic, objective, steps=20, seed=1)
    b = run_episode(proportional_to_load, traffic, objective, steps=20, seed=1)
    assert all(np.array_equal(a[k], b[k]) for k in a)


def test_proportional_to_load_follows_the_loads() -> None:
    obs = np.zeros(22, dtype=np.float32)
    obs[[0, 5, 10]] = [1.2, 0.4, 0.4]  # eMBB offers more than the whole cell (a spike)
    assert proportional_to_load(obs) == pytest.approx([0.6, 0.2, 0.2])


def test_proportional_to_load_with_no_load_is_equal_split() -> None:
    obs = np.zeros(22, dtype=np.float32)
    assert proportional_to_load(obs) == pytest.approx(equal_split(obs))


def test_controllers_see_the_same_episode() -> None:
    """Paired evaluation: same seed, same objective and traffic, whatever the controller."""
    env_a, env_b = SlicingEnv(), SlicingEnv()
    env_a.reset(seed=1000)
    env_b.reset(seed=1000)
    assert env_a.objective == env_b.objective
    m1 = episode_metrics(equal_split, seed=1000, steps=30)
    m2 = episode_metrics(equal_split, seed=1000, steps=30)
    assert m1 == m2 or all(
        (np.isnan(m1[k]) and np.isnan(m2[k])) or m1[k] == m2[k] for k in m1
    )


def test_td3_builds_and_learns() -> None:
    """TD3 uses the same hyperparameters; a 300-step run proves its training path works."""
    model = make_model(SlicingEnv(episode_steps=50), seed=0, algo="td3")
    model.learning_starts = 100
    model.learn(total_timesteps=300)
    assert type(model).__name__ == "TD3"


class FakeModel:
    """Stands in for an SB3 model: its 'actor' returns the first 3 observation values."""

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> tuple[np.ndarray, None]:
        return np.asarray(obs[:3], dtype=np.float32), None


def test_smoothing_filter_averages_with_the_previous_action() -> None:
    policy = SlicePolicy(FakeModel(), alpha=0.25)
    first = policy.action(np.array([1.0, 0.0, 0.0]))
    assert first == pytest.approx([1.0, 0.0, 0.0])  # nothing to average with yet
    second = policy.action(np.array([0.0, 1.0, 0.0]))
    assert second == pytest.approx([0.75, 0.25, 0.0])  # 0.25 * new + 0.75 * previous


def test_smoothing_filter_reset_forgets_the_past() -> None:
    policy = SlicePolicy(FakeModel(), alpha=0.25)
    policy.action(np.array([1.0, 0.0, 0.0]))
    policy.reset()
    assert policy.action(np.array([0.0, 0.0, 1.0])) == pytest.approx([0.0, 0.0, 1.0])


def test_alpha_one_turns_the_filter_off() -> None:
    policy = SlicePolicy(FakeModel(), alpha=1.0)
    policy.action(np.array([1.0, 0.0, 0.0]))
    assert policy.action(np.array([0.0, 1.0, 0.0])) == pytest.approx([0.0, 1.0, 0.0])


def test_smoothed_action_stays_in_range() -> None:
    policy = SlicePolicy(FakeModel(), alpha=0.25)
    rng = np.random.default_rng(0)
    for _ in range(50):
        a = policy.action(rng.uniform(0, 1, 3))
        assert np.all((a >= 0) & (a <= 1))


def test_invalid_alpha_is_rejected() -> None:
    with pytest.raises(ValueError):
        SlicePolicy(FakeModel(), alpha=0.0)


def test_selection_rule_needs_to_beat_equal_split_then_prefers_robust_goal_gap() -> None:
    from rl.select import select

    candidates = {
        "a": {"reward": 0.30, "mean_gap": 0.30, "share_gap_ge_0.10": 0.50},
        "b": {"reward": 0.26, "mean_gap": 0.20, "share_gap_ge_0.10": 0.90},
        "c": {"reward": 0.20, "mean_gap": 0.40, "share_gap_ge_0.10": 1.00},  # below equal split
    }
    assert select(candidates, equal_reward=0.21) == "b"


def test_selection_rule_breaks_ties_on_reward() -> None:
    from rl.select import select

    candidates = {
        "a": {"reward": 0.25, "mean_gap": 0.2, "share_gap_ge_0.10": 0.8},
        "b": {"reward": 0.27, "mean_gap": 0.2, "share_gap_ge_0.10": 0.8},
    }
    assert select(candidates, equal_reward=0.21) == "b"


def test_validation_seeds_are_disjoint_from_test_seeds() -> None:
    from rl.evaluate import EVAL_SEEDS
    from rl.select import VALIDATION_REWARD_SEEDS, VALIDATION_TRAFFIC_SEEDS

    assert not set(EVAL_SEEDS) & (set(VALIDATION_REWARD_SEEDS) | set(VALIDATION_TRAFFIC_SEEDS))
