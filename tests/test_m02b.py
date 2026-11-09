"""M02b: reward v2, objective relabeling, filter in the loop, oracle and goal test v2."""

import json
from pathlib import Path

import numpy as np
import pytest

from rl.baselines import equal_split
from rl.goal_test import (FrozenObjective, GoalTest, case_set, goal_metrics, held_out_objectives,
                          held_out_scenarios, validation_objectives, validation_scenarios)
from rl.oracle import GreedyOracle, split_grid
from rl.policy import SMOOTHING_ALPHA, SlicePolicy
from rl.relabel import RelabelReplayBuffer, kpis_from_info
from rl.train import ValidationCheckpoint, make_model
from sim.cell import CellConfig, smooth_split, step_slice, step_slices_batch
from sim.env import (OBJECTIVE_FEATURES, SlicingEnv, objective_arrays, objective_to_vector,
                     objective_vectors, sample_objective, sample_objective_vectors)
from sim.reward import compute_reward, compute_reward_batch
from sim.scenarios import urllc_spike
from verifier.objective import SLICES, RLObjective, SliceValues

CFG = CellConfig()


# --- Reward v2 -------------------------------------------------------------------------------

def _urllc_latency_reward(latency_ms: float, version: str) -> float:
    from sim.cell import SliceStep
    steps = {s: SliceStep(10.0, 0.0, 0.0, 1.0, latency_ms if s == "urllc" else 10.0) for s in SLICES}
    objective = RLObjective(weights=SliceValues(urllc=1.0), max_latency_ms=10.0)
    return compute_reward(objective, steps, {s: 10.0 for s in SLICES}, CFG, version)  # type: ignore[arg-type]


def test_v2_latency_penalty_keeps_growing_where_v1_is_flat() -> None:
    r = {lat: _urllc_latency_reward(lat, "v2") for lat in (10.0, 20.0, 100.0, 1700.0)}
    assert r[10.0] > r[20.0] > r[100.0] > r[1700.0]
    assert r[10.0] - r[20.0] == pytest.approx(np.log(2.0))
    assert _urllc_latency_reward(100.0, "v1") == pytest.approx(_urllc_latency_reward(1700.0, "v1"))


def test_batch_reward_equals_scalar_reward_on_real_steps() -> None:
    env = SlicingEnv()
    obs, _ = env.reset(seed=5)
    for version in ("v1", "v2"):
        env = SlicingEnv(reward_version=version)  # type: ignore[arg-type]
        env.reset(seed=5)
        for _ in range(30):
            _, r, _, _, info = env.step(env.action_space.sample())
            k = np.array([kpis_from_info(info)])
            w, t, b = objective_arrays(env.objective)
            batch = compute_reward_batch(w, t, b, k[:, 0:3], k[:, 3:6], k[:, 6], CFG.capacity_mbps, version)  # type: ignore[arg-type]
            assert batch[0] == pytest.approx(r, abs=1e-12)


# --- Objective helpers and relabeling --------------------------------------------------------

def test_objective_vectors_match_the_scalar_version() -> None:
    rng = np.random.default_rng(1)
    for _ in range(20):
        objective = sample_objective(rng, CFG)
        assert np.array_equal(objective_vectors(*objective_arrays(objective), CFG)[0],
                              objective_to_vector(objective, CFG))


def test_vector_sampler_has_the_same_distribution_as_sample_objective() -> None:
    rng = np.random.default_rng(2)
    n = 20_000
    w, t, b = sample_objective_vectors(rng, n, CFG)
    objs = [sample_objective(rng, CFG) for _ in range(4_000)]
    ref_w = np.array([o.weights.as_list() for o in objs])
    ref_t = np.array([o.min_thr_mbps.as_list() for o in objs])
    ref_b = np.array([np.nan if o.max_latency_ms is None else o.max_latency_ms for o in objs])
    assert np.allclose(w.mean(axis=0), ref_w.mean(axis=0), atol=0.02)
    assert np.allclose((t > 0).mean(axis=0), (ref_t > 0).mean(axis=0), atol=0.03)
    assert np.allclose(t.mean(axis=0), ref_t.mean(axis=0), atol=0.6)
    assert np.isnan(b).mean() == pytest.approx(np.isnan(ref_b).mean(), abs=0.03)
    assert np.nanmean(b) == pytest.approx(np.nanmean(ref_b), abs=1.0)


def _filled_buffer(relabel_fraction: float) -> tuple[RelabelReplayBuffer, list]:
    env = SlicingEnv(episode_steps=50)
    buf = RelabelReplayBuffer(100, env.observation_space, env.action_space, device="cpu",
                              relabel_fraction=relabel_fraction, seed=0)
    obs, _ = env.reset(seed=3)
    steps = []
    for _ in range(40):
        action = env.action_space.sample()
        next_obs, r, term, trunc, info = env.step(action)
        buf.add(obs[None], next_obs[None], action[None], np.array([r]), np.array([term]), [info])
        steps.append(info)
        obs = next_obs
    return buf, steps


def test_relabeled_reward_is_what_the_env_would_give_under_that_objective() -> None:
    buf, steps = _filled_buffer(relabel_fraction=1.0)
    inds = np.arange(40)
    obs, next_obs, rewards = buf.relabel(inds)
    for i in inds:
        vector = obs[i, -OBJECTIVE_FEATURES:]
        assert np.array_equal(vector, next_obs[i, -OBJECTIVE_FEATURES:])
        # Rebuild the objective from the vector and score the stored step with the env's function.
        weights, targets_frac, bound = vector[:3], vector[3:6], vector[6]
        objective = RLObjective(weights=SliceValues(**dict(zip(SLICES, (weights / weights.sum()).tolist()))),
                                min_thr_mbps=SliceValues(**dict(zip(SLICES, (targets_frac * CFG.capacity_mbps).tolist()))),
                                max_latency_ms=None if bound == 0 else float(bound * 100.0))
        info = steps[i]
        from sim.cell import SliceStep
        st = {s: SliceStep(info["throughput_mbps"][s], 0.0, 0.0, 0.0, info["latency_ms"][s]) for s in SLICES}
        assert rewards[i] == pytest.approx(compute_reward(objective, st, info["load_mbps"], CFG), abs=1e-4)


def test_relabel_fraction_zero_returns_the_stored_transitions() -> None:
    buf, _ = _filled_buffer(relabel_fraction=0.0)
    inds = np.arange(40)
    obs, next_obs, rewards = buf.relabel(inds)
    assert np.array_equal(obs, buf.observations[inds, 0]) and np.array_equal(rewards, buf.rewards[inds, 0])


def test_kpis_do_not_change_the_state_part_of_observations() -> None:
    buf, _ = _filled_buffer(relabel_fraction=1.0)
    obs, _, _ = buf.relabel(np.arange(40))
    assert np.array_equal(obs[:, :-OBJECTIVE_FEATURES], buf.observations[:40, 0, :-OBJECTIVE_FEATURES])


def test_td3_trains_with_the_relabeling_buffer() -> None:
    model = make_model(SlicingEnv(episode_steps=50), seed=0, algo="td3", relabel=0.8)
    model.learning_starts = 100
    model.learn(total_timesteps=300)
    assert isinstance(model.replay_buffer, RelabelReplayBuffer)


# --- Filter in the loop ----------------------------------------------------------------------

class _Fake:
    def predict(self, obs: np.ndarray, deterministic: bool = True) -> tuple[np.ndarray, None]:
        return np.asarray(obs[:3], dtype=np.float32), None  # 'actor' returns the loads part


def test_filter_in_env_equals_filter_in_policy() -> None:
    """Training with the filter in the env = deploying with the filter in SlicePolicy."""
    def run(env_alpha: float | None, policy_alpha: float) -> np.ndarray:
        env = SlicingEnv(episode_steps=30, action_filter_alpha=env_alpha)
        obs, _ = env.reset(seed=8)
        policy = SlicePolicy(_Fake(), alpha=policy_alpha)
        shares = []
        for _ in range(30):
            obs, _, _, _, info = env.step(policy(obs))
            shares.append([info["shares"][s] for s in SLICES])
        return np.array(shares)

    assert np.allclose(run(SMOOTHING_ALPHA, 1.0), run(None, SMOOTHING_ALPHA), atol=1e-6)


def test_smooth_split_keeps_the_sum_at_one() -> None:
    out = smooth_split(np.array([0.2, 0.3, 0.5]), np.array([0.6, 0.1, 0.3]), 0.25)
    assert out.sum() == pytest.approx(1.0) and out == pytest.approx([0.3, 0.25, 0.45])


# --- Oracle ----------------------------------------------------------------------------------

def test_batch_cell_step_matches_step_slice() -> None:
    rng = np.random.default_rng(4)
    for _ in range(200):
        share, load, backlog = rng.uniform(0, 1), rng.uniform(0, 80), rng.uniform(0, 50)
        ref = step_slice(share, load, backlog, 2.0, CFG)
        got = step_slices_batch(np.array([share]), np.array([load]), np.array([backlog]), np.array([2.0]), CFG)
        assert got["throughput_mbps"][0] == pytest.approx(ref.throughput_mbps)
        assert got["backlog_mb"][0] == pytest.approx(ref.backlog_mb)
        assert got["latency_ms"][0] == pytest.approx(ref.latency_ms)


def test_split_grid_covers_the_simplex() -> None:
    grid = split_grid()
    assert grid.shape == (861, 3) and np.allclose(grid.sum(axis=1), 1.0)


@pytest.mark.parametrize("objective_name", ["protect URLLC", "maximise eMBB"])
def test_oracle_beats_every_fixed_split_on_its_objective(objective_name: str) -> None:
    from rl.evaluate import OBJECTIVES
    from rl.goal_test import rollout, score
    objective = OBJECTIVES[objective_name]
    traffic = urllc_spike()[0]
    oracle = GreedyOracle(objective)
    best = score(rollout(lambda env, _o: oracle.act(env), traffic, objective, oracle.reset), objective)
    for split in ([1 / 3] * 3, [0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.6, 0.1, 0.3]):
        fixed = score(rollout(lambda _e, _o: np.array(split), traffic, objective), objective)  # noqa: B023 (called within this iteration)
        assert best > fixed


# --- Goal test v2 ----------------------------------------------------------------------------

def test_validation_and_test_sets_are_disjoint() -> None:
    assert not set(validation_scenarios()) & set(held_out_scenarios())
    assert not set(validation_objectives()) & set(held_out_objectives())
    assert not set(case_set("validation").reward_seeds) & set(case_set("test").reward_seeds)


def test_blind_controllers_score_zero() -> None:
    test = GoalTest("validation")
    m = test.evaluate(equal_split, with_reward=False)
    assert m["material_cases"] >= 20
    assert m["t1a_share_positive"] == 0.0 and m["t1b_captured"] == 0.0


def test_frozen_objective_policy_is_blind(tmp_path: Path) -> None:
    model = make_model(SlicingEnv(episode_steps=50), seed=0, algo="td3")
    model.learning_starts = 100
    model.learn(total_timesteps=200)
    blind = FrozenObjective(SlicePolicy(model), held_out_objectives()["maximise eMBB"])
    m = GoalTest("test").evaluate(blind, with_reward=False)
    assert m["t1a_share_positive"] == 0.0 and m["t1b_captured"] == 0.0


def test_goal_metrics_clip_captured_fraction() -> None:
    oracle = {("x", "a", "b"): 0.1, ("x", "b", "a"): 0.1, ("x", "a", "c"): 0.0}
    policy = {("x", "a", "b"): 0.5, ("x", "b", "a"): -0.5, ("x", "a", "c"): 9.0}
    m = goal_metrics(policy, oracle)
    assert m["material_cases"] == 2  # the (a, c) case is not material
    assert m["t1a_share_positive"] == 0.5 and m["t1b_captured"] == 0.0  # (+1 clipped, -1 clipped)


def test_validation_checkpoint_saves_the_best_model(tmp_path: Path) -> None:
    model = make_model(SlicingEnv(episode_steps=50), seed=0, algo="td3")
    model.learning_starts = 100
    path = tmp_path / "tiny.zip"
    callback = ValidationCheckpoint(every=100, path=path, start=200)
    model.learn(total_timesteps=300, callback=callback)
    history = json.loads((tmp_path / "tiny_validation.json").read_text())
    assert [h["step"] for h in history] == [200, 300]


def test_oracle_knows_exactly_the_loads_the_env_draws_next() -> None:
    import copy
    env = SlicingEnv(episode_steps=20)
    env.reset(seed=6)
    for _ in range(20):
        rng = copy.deepcopy(env.np_random)
        predicted = [env.traffic[s].load_at(env.time_s, rng) for s in SLICES]
        _, _, _, _, info = env.step(env.action_space.sample())
        assert predicted == [info["load_mbps"][s] for s in SLICES]
