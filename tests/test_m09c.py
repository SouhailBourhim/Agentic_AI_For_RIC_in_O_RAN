"""M09c: the training setting of policy v8 matches how the policy is deployed, and v7 is untouched by it."""

from pathlib import Path

import numpy as np
import pytest

from backends.sim import DelayedBackend
from rl import m09
from rl.goal_test import M09_SEEDS, run_split
from rl.policy import SlicePolicy
from rl.relabel import RelabelReplayBuffer
from rl.runner import observation_from_records, seen_loads
from rl.train import make_model
from sim.cell import CellConfig
from sim.env import HISTORY_S, OBJECTIVE_SLICE, OBS_SIZE, OBS_SIZE_WITH_HISTORY, LoadHistory, SlicingEnv
from sim.reward import compute_reward_batch, overserving
from sim.traffic import TrafficPattern, sample_traffic
from verifier.objective import DEFAULT_OBJECTIVE, SLICES


def _play(env: SlicingEnv, steps: int, seed: int = 4) -> tuple[list[np.ndarray], list[dict]]:
    obs, info = env.reset(seed=seed)
    observations, infos = [obs], [info]
    for k in range(steps):
        obs, _, _, _, info = env.step(np.array([0.5, 0.2 + 0.01 * (k % 5), 0.3], dtype=np.float32))
        observations.append(obs)
        infos.append(info)
    return observations, infos


def test_the_new_options_off_change_nothing() -> None:
    """Off, the M09c options don't touch the random stream: M01-M09b episodes reproduce exactly."""
    a, _ = _play(SlicingEnv(episode_steps=40), 40)
    b, _ = _play(SlicingEnv(episode_steps=40, expiry_share=0.5, toggle_prob=0.0, history="off"), 40)
    assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))
    assert sample_traffic(np.random.default_rng(1), 50, 75.0) == sample_traffic(np.random.default_rng(1), 50, 75.0, toggle_prob=0.0)


def test_the_kpis_are_seen_late_and_the_objective_current() -> None:
    env = SlicingEnv(episode_steps=30, kpi_delay_range=(2, 2), objective_switch_prob=0.3, expiry_share=0.5)
    observations, infos = _play(env, 30)
    for t in range(3, 31):
        late = infos[t - 2]
        assert observations[t][0] == pytest.approx(late["load_mbps"]["embb"] / env.obs_cfg.capacity_mbps, rel=1e-5)
        assert observations[t][2] == pytest.approx(min(late["latency_ms"]["embb"] / 100.0, 10.0), rel=1e-5)


def test_an_expiry_returns_to_the_default_objective() -> None:
    env = SlicingEnv(episode_steps=200, objective_switch_prob=1.0, expiry_share=1.0)
    env.reset(seed=0)
    env.step(np.array([0.4, 0.3, 0.3], dtype=np.float32))
    assert env.objective == DEFAULT_OBJECTIVE


def test_the_policy_is_told_79_while_the_cell_carries_its_own_capacity() -> None:
    env = SlicingEnv(cfg=m09.cell("m09"), capacity_range=(73.5, 80.2), history="on")
    capacities = set()
    for seed in range(5):
        env.reset(seed=seed)
        capacities.add(round(env.cfg.capacity_mbps, 3))
        assert env.obs_cfg.capacity_mbps == 79.0 and 73.5 <= env.cfg.capacity_mbps <= 80.2
    assert len(capacities) == 5
    assert env.observation_space.shape == (OBS_SIZE_WITH_HISTORY,)


def test_the_history_is_the_peak_of_the_loads_the_policy_saw() -> None:
    env = SlicingEnv(episode_steps=60, kpi_delay_range=(1, 1), history="on")
    observations, infos = _play(env, 60)
    t = 50
    seen = [infos[k]["load_mbps"]["urllc"] for k in range(t - 1 - int(HISTORY_S) + 1, t)]  # one tick late, 30 s
    assert observations[t][OBS_SIZE + SLICES.index("urllc")] == pytest.approx(max(seen) / env.obs_cfg.capacity_mbps, rel=1e-5)


def test_the_runner_builds_the_same_history_from_its_records() -> None:
    """The deployed side: the runner feeds LoadHistory the loads it reads, and appends the same 3 features."""
    cfg = CellConfig(capacity_mbps=79.0)
    traffic = {s: TrafficPattern("spike", 10.0, 30.0, 5, 15, noise_std=0.0) for s in SLICES}
    backend = DelayedBackend(cfg, traffic, seed=0, delay=2)
    history, direct = LoadHistory(), LoadHistory()
    for _ in range(25):
        backend.apply_prb_split([1 / 3, 1 / 3, 1 / 3])
        backend.sleep(1.0)
        records = backend.read_kpis()
        history.update(seen_loads(records))
        direct.update({r.slice: r.offered_load_mbps for r in records})
        obs = observation_from_records(records, DEFAULT_OBJECTIVE, cfg, history)
    assert obs.shape == (OBS_SIZE_WITH_HISTORY,)
    assert np.array_equal(obs[OBS_SIZE:], direct.features(79.0))
    assert obs[OBS_SIZE] == pytest.approx(30.0 / 79.0)


def test_a_policy_trained_on_22_inputs_acts_the_same_on_25(tmp_path: Path) -> None:
    """v7 is served unchanged: the runner's extra history inputs are cut before its actor."""
    model = make_model(SlicingEnv(episode_steps=50), seed=0, algo="td3")
    model.learning_starts = 100
    model.learn(total_timesteps=200)
    policy = SlicePolicy(model, alpha=1.0)
    assert policy.n_inputs == OBS_SIZE
    rng = np.random.default_rng(0)
    for _ in range(5):
        obs = rng.uniform(0, 1, OBS_SIZE).astype(np.float32)
        longer = np.concatenate([obs, rng.uniform(0, 1, 3).astype(np.float32)])
        assert np.array_equal(policy.raw_action(obs), policy.raw_action(longer))


def test_reward_v3_charges_surplus_only_while_a_target_is_missed() -> None:
    w = np.array([[0.4, 0.3, 0.3]])
    nan = np.array([np.nan])
    loads = np.array([[40.0, 10.0, 30.0]])
    # BE asks 20 and gets 15 (short 5); URLLC was given 0.35 x 79 = 27.65 for a load of 10 (surplus 17.65).
    targets = np.array([[0.0, 0.0, 20.0]])
    thr = np.array([[30.0, 10.0, 15.0]])
    shares = np.array([[0.40, 0.35, 0.25]])
    args = (w, targets, nan, thr, loads, np.array([1.0]), 79.0)
    v1 = compute_reward_batch(*args, version="v1")
    v3 = compute_reward_batch(*args, version="v3", shares=shares, urllc_base_ms=0.6)
    assert v1 - v3 == pytest.approx(5.0 / 79.0)  # min(surplus, shortfall) = the 5 Mb/s BE misses
    met = np.array([[30.0, 10.0, 20.0]])  # nobody short: surplus is free
    assert overserving(targets, nan, met, loads, shares, 79.0, 0.6)[0] == 0.0
    with pytest.raises(ValueError):
        compute_reward_batch(*args, version="v3")


def test_relabeling_rewrites_the_objective_and_keeps_the_history() -> None:
    env = SlicingEnv(cfg=m09.cell("m09"), episode_steps=50, history="on", capacity_range=(73.5, 80.2))
    buf = RelabelReplayBuffer(100, env.observation_space, env.action_space, device="cpu", relabel_fraction=1.0,
                              capacity_mbps=79.0, reward_version="v3", urllc_base_ms=0.592)
    obs, _ = env.reset(seed=0)
    for _ in range(40):
        action = np.array([0.4, 0.3, 0.3], dtype=np.float32)
        nxt, reward, _, _, info = env.step(action)
        buf.add(obs[None], nxt[None], action[None], np.array([reward]), np.array([False]), [info])
        obs = nxt
    new, _, _ = buf.relabel(np.arange(40))
    stored = buf.observations[:40, 0]
    assert np.array_equal(new[:, OBS_SIZE:], stored[:, OBS_SIZE:])  # the history is untouched
    assert np.array_equal(new[:, :OBJECTIVE_SLICE.start], stored[:, :OBJECTIVE_SLICE.start])
    assert not np.array_equal(new[:, OBJECTIVE_SLICE], stored[:, OBJECTIVE_SLICE])


def test_every_recipe_builds_and_takes_a_step() -> None:
    for name, recipe in m09.RECIPES.items():
        env = SlicingEnv(episode_steps=m09.EPISODE_STEPS, action_filter_alpha=m09.FILTER_ALPHA, min_share=0.05,
                         reward_version=recipe["reward"], **m09.training_options(recipe["profile"], recipe["history"]))
        obs, _ = env.reset(seed=1)
        assert obs.shape == ((OBS_SIZE,) if recipe["history"] == "off" else (OBS_SIZE_WITH_HISTORY,)), name
        env.step(np.array([0.4, 0.3, 0.3], dtype=np.float32))


def test_the_m09_held_out_split_needs_final() -> None:
    assert "m09_test" in M09_SEEDS
    with pytest.raises(SystemExit):
        run_split("m09_test", [], final=False, min_share=0.05)
