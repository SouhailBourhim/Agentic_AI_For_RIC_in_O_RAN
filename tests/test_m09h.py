"""M09h (policy v9): each change is an option that is off by default, the same in training and when served, and the
v9 protocol's plumbing (fresh goal-test splits, T1a in the checkpoint rule, the median seed, the gate)."""

import json
from pathlib import Path

import numpy as np
import pytest
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

from backends.sim import DelayedBackend
from rl import m09
from rl.baselines import FormulaPolicy
from rl.goal_test import FINAL_SPLITS, case_set, run_split, setting_of
from rl.policy import SlicePolicy
from rl.relabel import RelabelReplayBuffer
from rl.residual import RESIDUAL_BOUND, ResidualEnv, ResidualPolicy, residual_shares
from rl.runner import Runner, observation_from_records, seen_loads
from rl.train import make_model
from sim.cell import CellConfig
from sim.env import OBS_SIZE, OBS_SIZE_WITH_APPLIED, OBS_SIZE_WITH_HISTORY, LoadHistory, SlicingEnv, objective_arrays
from sim.reward import compute_reward_batch, moved
from sim.traffic import TrafficPattern
from verifier.objective import DEFAULT_OBJECTIVE, SLICES


def _v9_env(recipe: str = "v9", **extra) -> SlicingEnv:
    r = m09.RECIPES[recipe]
    return SlicingEnv(episode_steps=m09.EPISODE_STEPS, action_filter_alpha=m09.FILTER_ALPHA, min_share=0.05,
                      reward_version=r["reward"], **m09.training_options(r["profile"], r["history"],
                                                                         m09.option(recipe, "applied"),
                                                                         m09.option(recipe, "switch_cost")), **extra)


# ---------------------------------------------------------------- off by default: v7 and v8 unchanged


def test_v8_recipes_keep_their_options_and_observation() -> None:
    for recipe in m09.V8_RECIPES:
        assert {k: m09.option(recipe, k) for k in m09.V9_DEFAULTS} == m09.V9_DEFAULTS
        assert m09.training_options(m09.RECIPES[recipe]["profile"], m09.RECIPES[recipe]["history"]).keys() == \
            m09.training_options(m09.RECIPES[recipe]["profile"], m09.RECIPES[recipe]["history"], False, 0.0).keys()
    env = _v9_env("v8_nooverserve")
    assert env.observation_space.shape == (OBS_SIZE_WITH_HISTORY,) and env.switch_cost == 0.0


def test_the_default_env_gives_the_same_episode_as_before_with_no_switch_cost() -> None:
    """Same seed, same actions: an env with switch_cost 0 and one built without the argument are identical."""
    a, b = SlicingEnv(episode_steps=40), SlicingEnv(episode_steps=40, switch_cost=0.0)
    oa, _ = a.reset(seed=3)
    ob, _ = b.reset(seed=3)
    rng = np.random.default_rng(0)
    for _ in range(40):
        act = rng.uniform(0, 1, 3).astype(np.float32)
        oa, ra, *_ = a.step(act)
        ob, rb, *_ = b.step(act)
        assert np.array_equal(oa, ob) and ra == rb


# ---------------------------------------------------------------- A1 switching cost


def test_the_switching_cost_charges_the_share_of_the_cell_moved() -> None:
    plain, charged = _v9_env("v9_noswitch"), _v9_env("v9")
    plain.reset(seed=4)
    charged.reset(seed=4)
    rng = np.random.default_rng(1)
    for _ in range(30):
        act = rng.uniform(0, 1, 3).astype(np.float32)
        _, r0, _, _, info = plain.step(act)
        _, r1, *_ = charged.step(act)
        step = moved(np.array([list(info["shares"].values())]), np.array([list(info["previous_shares"].values())]))[0]
        assert r0 - r1 == pytest.approx(m09.SWITCH_COST * step, abs=1e-12)
    assert moved(np.array([[0.5, 0.3, 0.2]]), np.array([[0.3, 0.3, 0.4]]))[0] == pytest.approx(0.2)


def test_a_relabeled_reward_keeps_the_switching_cost() -> None:
    """Relabeled with the step's own objective, the buffer gives back the env's reward, switching cost included."""
    env = _v9_env("v9")
    cell = env.base_cfg
    buf = RelabelReplayBuffer(100, env.observation_space, env.action_space, device="cpu", relabel_fraction=1.0,
                              capacity_mbps=cell.told_capacity_mbps, reward_version="v1",
                              urllc_base_ms=cell.told_urllc_base_latency_ms, switch_cost=m09.SWITCH_COST)
    obs, _ = env.reset(seed=2)
    rng = np.random.default_rng(2)
    for i in range(30):
        act = rng.uniform(0, 1, 3).astype(np.float32)
        w, t, b = objective_arrays(env.objective)  # the step's reward uses this one; a switch comes after it
        nxt, reward, _, _, info = env.step(act)
        buf.add(obs[None], nxt[None], act[None], np.array([reward]), np.array([False]), [info])
        k = buf.kpis[i:i + 1]
        again = compute_reward_batch(w, t, b, k[:, 0:3], k[:, 3:6], k[:, 6], cell.told_capacity_mbps, "v1",
                                     shares=k[:, 7:10], switch_cost=m09.SWITCH_COST, previous_shares=k[:, 10:13])
        assert again[0] == pytest.approx(reward, abs=1e-9)
        obs = nxt
    with pytest.raises(ValueError):
        compute_reward_batch(w, t, b, k[:, 0:3], k[:, 3:6], k[:, 6], 79.0, "v1", switch_cost=1.0)


# ---------------------------------------------------------------- A5 the applied split in the observation


def test_the_applied_split_follows_the_history_undelayed() -> None:
    env = _v9_env("v9")
    obs, _ = env.reset(seed=5)
    assert obs.shape == (OBS_SIZE_WITH_APPLIED,) and obs[OBS_SIZE_WITH_HISTORY:] == pytest.approx([1 / 3] * 3)
    for _ in range(10):
        obs, *_ = env.step(np.array([0.7, 0.2, 0.1], dtype=np.float32))
        assert np.array_equal(obs[OBS_SIZE_WITH_HISTORY:], env.shares.astype(np.float32))  # this step's, not 2 ago
    with pytest.raises(ValueError):
        SlicingEnv(applied_split=True)  # it goes after the history: history must be on


def test_the_runner_appends_the_split_it_applied() -> None:
    cfg = CellConfig(capacity_mbps=79.0)
    traffic = {s: TrafficPattern("steady", 20.0, 20.0, noise_std=0.0) for s in SLICES}
    backend = DelayedBackend(cfg, traffic, seed=0, delay=2)
    history = LoadHistory()
    applied = np.array([0.5, 0.3, 0.2])
    backend.apply_prb_split(list(applied))
    backend.sleep(1.0)
    records = backend.read_kpis()
    history.update(seen_loads(records))
    obs = observation_from_records(records, DEFAULT_OBJECTIVE, cfg, history, applied)
    assert obs.shape == (OBS_SIZE_WITH_APPLIED,) and obs[OBS_SIZE_WITH_HISTORY:] == pytest.approx(applied)
    assert np.array_equal(obs[:OBS_SIZE_WITH_HISTORY], observation_from_records(records, DEFAULT_OBJECTIVE, cfg, history))


def test_the_runner_remembers_what_it_applied(ric_db, tiny_policy_file) -> None:
    from tests.conftest import start_cell

    runner = start_cell(ric_db, tiny_policy_file)
    assert runner.applied == pytest.approx([1 / 3] * 3)
    result = runner.tick()
    assert runner.applied == pytest.approx(result.shares)


# ---------------------------------------------------------------- A4 noise, A6 gamma


def test_v9_noise_is_ou_with_a_ten_step_correlation_and_gamma_is_098() -> None:
    env = _v9_env("v9")
    model = make_model(env, seed=0, algo="td3", noise="ou", gamma=m09.V9_GAMMA)
    assert isinstance(model.action_noise, OrnsteinUhlenbeckActionNoise) and model.gamma == 0.98
    assert (model.action_noise._theta, model.action_noise._dt) == (m09.OU_THETA, m09.OU_DT)
    np.random.seed(0)
    x = np.array([model.action_noise()[0] for _ in range(20000)])
    assert np.corrcoef(x[:-1], x[1:])[0, 1] == pytest.approx(0.9, abs=0.02)
    assert x.std() == pytest.approx(m09.OU_SIGMA / np.sqrt(1 - 0.9 ** 2), rel=0.1)  # 0.23
    plain = make_model(SlicingEnv(episode_steps=50), seed=0, algo="td3")
    assert isinstance(plain.action_noise, NormalActionNoise) and plain.gamma == 0.95


# ---------------------------------------------------------------- A7 the residual


def test_the_residual_is_the_formula_at_mid_action_and_moves_each_slice_by_at_most_the_bound() -> None:
    base = np.array([0.5, 0.3, 0.2])
    assert residual_shares(base, np.full(3, 0.5)) == pytest.approx(base)
    up = residual_shares(base, np.array([1.0, 0.5, 0.5]))
    assert up.sum() == pytest.approx(1.0) and up[0] == pytest.approx((0.5 + RESIDUAL_BOUND) / (1 + RESIDUAL_BOUND))
    assert residual_shares(np.array([0.1, 0.1, 0.8]), np.array([0.0, 0.0, 0.5])) == pytest.approx([0, 0, 1])


class _Fixed:
    """A stand-in actor with a fixed output: enough to check that training and serving build the same split."""

    observation_space = None

    def predict(self, obs, deterministic=True):
        return np.array([0.8, 0.3, 0.5]), None


def test_the_residual_split_is_the_same_in_training_and_when_served() -> None:
    env = ResidualEnv(_v9_env("v9_residual"))
    policy = ResidualPolicy(_Fixed(), alpha=m09.FILTER_ALPHA)
    obs, _ = env.reset(seed=6)
    policy.reset()
    for _ in range(40):
        served = policy.shares(obs)  # filtered, before the floor: the env does the same with its filter
        obs, *_ = env.step(np.array([0.8, 0.3, 0.5], dtype=np.float32))
        assert env.unwrapped._filtered == pytest.approx(served, abs=1e-6)


def test_a_residual_model_file_loads_as_a_residual_policy(tmp_path: Path, tiny_policy_file: Path) -> None:
    copy = tmp_path / "td3_v9_residual_seed50_best.zip"
    copy.write_bytes(tiny_policy_file.read_bytes())
    assert isinstance(SlicePolicy.load(copy, 0.1), ResidualPolicy)
    assert type(SlicePolicy.load(tiny_policy_file, 0.1)) is SlicePolicy
    policy = SlicePolicy.load(copy, 1.0)
    obs = np.random.default_rng(0).uniform(0, 1, OBS_SIZE).astype(np.float32)
    base = FormulaPolicy(policy.base_ms).shares(obs)
    assert policy.shares(obs) == pytest.approx(residual_shares(base, policy.raw_action(obs)), abs=1e-6)


# ---------------------------------------------------------------- the protocol's plumbing


def test_the_m10_splits_are_fresh_and_in_the_v9_observation() -> None:
    assert "m10_test" in FINAL_SPLITS
    val = case_set("m10_validation")
    assert min(val.reward_seeds) == 11200 and len(val.reward_seeds) == 50
    assert setting_of("m10_test").env["applied_split"] and "applied_split" not in setting_of("m09_test").env
    with pytest.raises(SystemExit):
        run_split("m10_test", [], final=False, min_share=0.05)


def _history(path: Path, entries: list[dict]) -> None:
    path.with_name(f"{path.stem}_validation.json").write_text(json.dumps(entries))


def test_a_v9_checkpoint_must_pass_t1a_and_the_median_seed_is_picked(tmp_path: Path) -> None:
    from mlops.registry import best_validation
    from rl.sweep import median_seeds

    def entry(reward, t1a=True):
        return {"t2_pass": True, "t1a_pass": t1a, "t1a_required": True, "t1b_captured": 0.6, "reward": reward,
                "t1a_share_positive": 0.95 if t1a else 0.5, "step": 100}

    rewards = {50: 0.21, 51: 0.25, 52: 0.23, 53: None, 54: 0.22}  # 53 kept no checkpoint
    for seed, r in rewards.items():
        run = tmp_path / f"td3_v9_seed{seed}.zip"
        run.write_bytes(b"x")
        if r is not None:
            run.with_name(f"{run.stem}_best.zip").write_bytes(b"x")
            _history(run, [entry(r), entry(r + 0.5, t1a=False)])  # the higher reward failed T1a: not the checkpoint
    assert best_validation(tmp_path / "td3_v9_seed51_best.zip")["validation_reward"] == 0.25
    assert median_seeds(tmp_path)["v9"] == tmp_path / "td3_v9_seed54_best.zip"  # ranks None, .21, .22, .23, .25


def test_the_gate_accepts_m10_test_as_a_calibrated_held_out_split() -> None:
    from mlops.registry import CALIBRATED_TEST_SPLITS

    assert CALIBRATED_TEST_SPLITS == ("m09_test", "m10_test")


def test_every_v9_arm_builds_and_takes_a_step() -> None:
    for recipe in m09.V9_RECIPES:
        env = _v9_env(recipe)
        if m09.option(recipe, "residual"):
            env = ResidualEnv(env)
        obs, _ = env.reset(seed=1)
        size = OBS_SIZE_WITH_APPLIED if m09.option(recipe, "applied") else OBS_SIZE_WITH_HISTORY
        assert obs.shape == (size,), recipe
        env.step(np.array([0.4, 0.3, 0.3], dtype=np.float32))
    assert set(m09.V9_RECIPES) == {"v9", "v9_noswitch", "v9_noou", "v9_noapplied", "v9_gamma95", "v9_residual"}


def test_the_runner_serves_a_v9_policy_on_its_28_inputs(ric_db, tmp_path: Path) -> None:
    from tests.conftest import start_cell

    model = make_model(_v9_env("v9"), seed=0, algo="td3", noise="ou", gamma=0.98)
    path = tmp_path / "td3_v9_seed0.zip"
    model.save(path)
    runner = start_cell(ric_db, path)
    assert runner.policy.n_inputs == OBS_SIZE_WITH_APPLIED
    for _ in range(3):
        assert sum(runner.tick().shares) == pytest.approx(1.0)
    assert isinstance(runner, Runner)


def test_a_residual_recipe_refuses_a_model_name_without_the_mark(tmp_path: Path) -> None:
    from rl.train import train

    with pytest.raises(ValueError, match="_residual"):
        train(seed=0, timesteps=10, threads=1, algo="td3", min_share=0.05, m09_recipe="v9_residual",
              group=str(tmp_path), track=False)


def test_the_v9_checkpoint_runs_inside_training_and_records_t1a_and_the_spread(tmp_path: Path) -> None:
    """The callback as training calls it, on a residual arm: its first real call would otherwise come at 50k steps."""
    from rl.train import ValidationCheckpoint

    env = ResidualEnv(_v9_env("v9_residual"))
    model = make_model(env, seed=0, algo="td3", relabel=0.8, noise="ou", gamma=0.98, switch_cost=m09.SWITCH_COST)
    model.learning_starts = 100
    path = tmp_path / "td3_v9_residual_seed0.zip"
    callback = ValidationCheckpoint(every=200, path=path, start=200, min_share=0.05, m09_recipe="v9_residual")
    model.learn(total_timesteps=400, callback=callback)
    history = json.loads(path.with_name(f"{path.stem}_validation.json").read_text())
    assert [h["step"] for h in history] == [200, 400]
    assert all(h["t1a_required"] and h["action_spread"] >= 0 and isinstance(h["collapsed"], bool) for h in history)
    kept = path.with_name(f"{path.stem}_best.zip").exists()
    assert kept == any(h["t2_pass"] and h["t1a_pass"] for h in history)
