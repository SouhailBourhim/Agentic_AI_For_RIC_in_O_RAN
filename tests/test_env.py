"""The gymnasium environment in sim/env.py."""

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from sim.env import OBJECTIVE_FEATURES, OBS_SIZE, SlicingEnv
from sim.traffic import TrafficPattern
from verifier.objective import SLICES, RLObjective, SliceValues

STEADY = {s: TrafficPattern("steady", 10.0, 10.0, noise_std=0.0) for s in SLICES}


def test_passes_gymnasium_checker() -> None:
    check_env(SlicingEnv(), skip_render_check=True)


def test_spaces() -> None:
    env = SlicingEnv()
    assert env.observation_space.shape == (OBS_SIZE,) == (22,)
    assert env.action_space.shape == (3,)


@pytest.mark.parametrize("seed", range(5))
def test_applied_shares_sum_to_one(seed: int) -> None:
    env = SlicingEnv()
    env.reset(seed=seed)
    for action in [env.action_space.sample(), np.zeros(3, dtype=np.float32)]:
        _, _, _, _, info = env.step(action)
        assert sum(info["shares"].values()) == pytest.approx(1.0)


def test_observation_contains_the_objective() -> None:
    """Same traffic, different objective: only the objective part of the obs differs."""
    a = RLObjective(weights=SliceValues(embb=1.0))
    b = RLObjective(weights=SliceValues(urllc=1.0), min_thr_mbps=SliceValues(urllc=20.0), max_latency_ms=10.0)
    env = SlicingEnv()
    obs_a, _ = env.reset(seed=1, options={"objective": a, "traffic": STEADY})
    obs_b, _ = env.reset(seed=1, options={"objective": b, "traffic": STEADY})
    kpi = slice(0, OBS_SIZE - OBJECTIVE_FEATURES)
    assert np.array_equal(obs_a[kpi], obs_b[kpi])
    assert not np.array_equal(obs_a[-OBJECTIVE_FEATURES:], obs_b[-OBJECTIVE_FEATURES:])


def test_set_objective_changes_the_observation() -> None:
    env = SlicingEnv()
    env.reset(seed=2, options={"objective": RLObjective(weights=SliceValues(embb=1.0))})
    env.set_objective(RLObjective(weights=SliceValues(be=1.0)))
    obs, *_ = env.step(np.ones(3, dtype=np.float32))
    weights_in_obs = obs[-OBJECTIVE_FEATURES : -OBJECTIVE_FEATURES + 3]
    assert weights_in_obs == pytest.approx([0.0, 0.0, 1.0])


def test_meeting_a_min_target_earns_more_than_missing_it() -> None:
    """URLLC asks for 30 Mb/s with a 30 Mb/s target: 40 % of PRBs meets it, 5 % does not."""
    traffic = {
        "embb": TrafficPattern("steady", 60.0, 60.0, noise_std=0.0),
        "urllc": TrafficPattern("steady", 30.0, 30.0, noise_std=0.0),
        "be": TrafficPattern("steady", 30.0, 30.0, noise_std=0.0),
    }
    objective = RLObjective(weights=SliceValues(embb=0.6, urllc=0.2, be=0.2), min_thr_mbps=SliceValues(urllc=30.0))
    env = SlicingEnv()
    env.reset(seed=0, options={"objective": objective, "traffic": traffic})
    _, r_meet, *_ = env.step(np.array([0.4, 0.4, 0.2], dtype=np.float32))
    env.reset(seed=0, options={"objective": objective, "traffic": traffic})
    _, r_miss, *_ = env.step(np.array([0.75, 0.05, 0.2], dtype=np.float32))
    assert r_meet > r_miss


def test_same_seed_same_trajectory() -> None:
    def run(seed: int) -> list[float]:
        env = SlicingEnv(episode_steps=20)
        env.reset(seed=seed)
        rewards, done = [], False
        while not done:
            _, r, terminated, truncated, _ = env.step(np.array([0.5, 0.3, 0.2], dtype=np.float32))
            rewards.append(r)
            done = terminated or truncated
        return rewards

    assert run(3) == run(3)
    assert run(3) != run(4)


def test_episode_is_truncated_after_episode_steps() -> None:
    env = SlicingEnv(episode_steps=5)
    env.reset(seed=0)
    flags = [env.step(np.ones(3, dtype=np.float32))[3] for _ in range(5)]
    assert flags == [False, False, False, False, True]
