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
